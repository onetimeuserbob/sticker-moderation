"""
Telethon-based userbot relay.

Why this exists: Telegram's Bot API hard-blocks bots from receiving
messages sent by other bots in groups (regardless of privacy mode or
admin status). To moderate applications posted by @sticker_bot we
therefore need a process logged in as a regular user account that
DOES see those messages, then hands them to our moderator bot for
review and reply.

This module owns:
  - a Telethon client logged in as the owner (session file persisted)
  - a per-chat debouncer that stitches @sticker_bot's media-group + text
    into a single application
  - photo download to local cache
  - delegation to ReviewBot.run_review() which uses the bot's identity
    to post the verdict back into the original chat.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from telethon import TelegramClient, events
from telethon.errors import RPCError
from telethon.tl.types import (
    MessageEntityMention,
    MessageEntityMentionName,
)

from assistant import ModeratorAssistant

if TYPE_CHECKING:
    from review_bot import ReviewBot, BotConfig


log = logging.getLogger("userbot")


_PACK_URL_RE = re.compile(r"https?://t\.me/addstickers/[A-Za-z0-9_]+", re.I)


def _find_pack_url_in_telethon_msg(msg) -> str:
    """Pull a t.me/addstickers/<slug> URL out of a Telethon Message,
    checking visible text, MessageEntityTextUrl entries, and reply
    markup buttons. Returns "" if none."""
    text = msg.message or ""
    m = _PACK_URL_RE.search(text)
    if m:
        return m.group(0)
    # Hidden URLs in entities (e.g. when @sticker_bot posts a button-style link).
    for ent in (msg.entities or []):
        url = getattr(ent, "url", None)
        if url and _PACK_URL_RE.search(url):
            return _PACK_URL_RE.search(url).group(0)
    # Inline keyboard buttons.
    rm = getattr(msg, "reply_markup", None)
    rows = getattr(rm, "rows", None) or []
    for row in rows:
        for btn in getattr(row, "buttons", []) or []:
            url = getattr(btn, "url", None)
            if url and _PACK_URL_RE.search(url):
                return _PACK_URL_RE.search(url).group(0)
    return ""


@dataclass
class _Buffer:
    messages: list = field(default_factory=list)
    seen_ids: set = field(default_factory=set)
    task: asyncio.Task | None = None


class UserbotRelay:
    """One Telethon client + a per-chat debouncer."""

    def __init__(self, review_bot: "ReviewBot", bot_cfg: "BotConfig"):
        self.review_bot = review_bot
        self.bot_cfg = bot_cfg
        self.client: TelegramClient | None = None
        self.buffers: dict[int, _Buffer] = {}
        self.debounce_s: float = 3.0
        self._me_id: int | None = None  # this user account's id
        self._me_username: str | None = None
        self._bot_id: int | None = None  # the moderator bot's id
        self.assistant = ModeratorAssistant(review_bot, self)

    async def start(self) -> None:
        session = str(self.bot_cfg.userbot_session_path)
        self.client = TelegramClient(
            session,
            self.bot_cfg.userbot_api_id,
            self.bot_cfg.userbot_api_hash,
        )
        await self.client.connect()
        if not await self.client.is_user_authorized():
            raise RuntimeError(
                f"Telethon session at {session} is not authorized. "
                "Run userbot_login.py locally to create one."
            )
        me = await self.client.get_me()
        self._me_id = me.id
        self._me_username = (me.username or "").lower() or None
        try:
            if self.review_bot.bot is not None:
                self._bot_id = (await self.review_bot.bot.get_me()).id
        except Exception:  # noqa: BLE001
            self._bot_id = None
        log.info(
            "userbot logged in as id=%s name=%r username=@%s; moderator bot id=%s",
            self._me_id, me.first_name, self._me_username or "(none)", self._bot_id,
        )
        self.client.add_event_handler(self._on_new_message, events.NewMessage())
        self.client.add_event_handler(self._on_chat_action, events.ChatAction())
        # Run in background; Telethon manages its own loop integration.
        # No await client.run_until_disconnected() — we want to return so
        # PTB's polling can take over the main loop. Telethon dispatches
        # events on the connected client without an explicit run loop.

    async def stop(self) -> None:
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # ---------- Outgoing primitives (used by ReviewBot.run_review) ----------
    # We expose send / delete / typing here so the moderation verdict can be
    # posted UNDER the Moderator account, not the bot. Keeps a single
    # identity in the chat.

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        markdown: bool = False,
    ):
        """Send via the Moderator user account. Returns the Telethon Message."""
        if self.client is None:
            raise RuntimeError("userbot client not started")
        return await self.client.send_message(
            chat_id,
            text,
            reply_to=reply_to,
            parse_mode="md" if markdown else None,
            link_preview=False,
        )

    async def send_message_safe(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
    ):
        """Markdown send with plain-text fallback (mirrors _safe_send for PTB).
        If Telethon's md parser rejects the text we strip Markdown and retry."""
        from review_bot import _strip_markdown
        try:
            sent = await self.send_message(
                chat_id, text, reply_to=reply_to, markdown=True
            )
            log.info(
                "userbot sent: chat=%s reply_to=%s msg_id=%s bytes=%d",
                chat_id, reply_to, getattr(sent, "id", None), len(text),
            )
            return sent
        except (RPCError, ValueError) as e:
            log.warning("telethon md send failed (%s); falling back to plain", e)
            sent = await self.send_message(
                chat_id, _strip_markdown(text), reply_to=reply_to, markdown=False
            )
            log.info(
                "userbot sent (plain fallback): chat=%s reply_to=%s msg_id=%s",
                chat_id, reply_to, getattr(sent, "id", None),
            )
            return sent

    async def delete_message(self, chat_id: int, message_id: int) -> None:
        if self.client is None:
            return
        try:
            await self.client.delete_messages(chat_id, [message_id])
        except Exception as e:  # noqa: BLE001
            log.warning("telethon delete_message failed: %s", e)

    async def send_typing(self, chat_id: int) -> None:
        if self.client is None:
            return
        try:
            async with self.client.action(chat_id, "typing"):
                pass
        except Exception:  # noqa: BLE001
            pass

    # ---------- chat membership tracking (auto-whitelist) ----------

    async def _on_chat_action(self, event) -> None:
        """When Moderator is added to a group, auto-whitelist it. When
        Moderator is removed, drop it from the allowlist."""
        if not (event.user_added or event.user_joined or event.user_left or event.user_kicked):
            return
        try:
            users = event.users if event.users is not None else []
        except Exception:  # noqa: BLE001
            users = []
        # Telethon ChatAction has user_id helpers; sometimes users[] empty.
        target_id = getattr(event, "user_id", None)
        if target_id is None and users:
            target_id = getattr(users[0], "id", None)
        if target_id != self._me_id:
            return  # not about Moderator

        chat_id = self._normalize_chat_id(event.chat_id)
        try:
            chat = await event.get_chat()
            title = getattr(chat, "title", "(untitled)")
            chat_type = chat.__class__.__name__
        except Exception:  # noqa: BLE001
            title, chat_type = "(unknown)", "?"

        if event.user_added or event.user_joined:
            self.review_bot.allowlist.add(chat_id, title=title, chat_type=chat_type)
            log.info("userbot auto-whitelisted chat %s (%s)", chat_id, title)
            try:
                await self.send_message_safe(
                    chat_id,
                    (
                        "👋 Hi — I'm Moderator. I'll review sticker-pack "
                        "applications posted here and answer questions about "
                        "the moderation rules. Reply to me or @-mention me "
                        "and I'll respond. Type something like _what are the "
                        "rules?_ to get started."
                    ),
                )
            except Exception:  # noqa: BLE001
                pass
        elif event.user_left or event.user_kicked:
            removed = self.review_bot.allowlist.remove(chat_id)
            if removed:
                log.info("userbot left chat %s; un-whitelisted", chat_id)

    # ---------- detection helpers ----------

    def _is_mentioned(self, msg) -> bool:
        """True if the message @-mentions Moderator (by username or by id)."""
        if not msg.entities:
            return False
        text = msg.message or ""
        for ent in msg.entities:
            if isinstance(ent, MessageEntityMentionName):
                if ent.user_id == self._me_id:
                    return True
            elif isinstance(ent, MessageEntityMention) and self._me_username:
                start, length = ent.offset, ent.length
                snippet = text[start:start + length].lstrip("@").lower()
                if snippet == self._me_username:
                    return True
        return False

    async def _is_reply_to_me(self, msg) -> bool:
        """True if msg.reply_to points to a message Moderator authored."""
        reply_to = getattr(msg, "reply_to", None)
        replied_to_id = getattr(reply_to, "reply_to_msg_id", None) if reply_to else None
        if not replied_to_id:
            return False
        # Cheap check: it's a verdict we posted (we know msg_id).
        if replied_to_id in self.review_bot.verdict_index:
            v = self.review_bot.verdict_index[replied_to_id]
            if v.get("transport") == "userbot":
                return True
        # Otherwise fetch the replied-to message and check the sender.
        try:
            replied = await msg.get_reply_message()
            return bool(replied and replied.sender_id == self._me_id)
        except Exception:  # noqa: BLE001
            return False

    async def _on_new_message(self, event) -> None:
        msg = event.message
        chat_id = self._normalize_chat_id(event.chat_id)

        sender = await event.get_sender()
        sender_id = getattr(sender, "id", None)
        if sender_id is None:
            return
        # Skip our own user account (don't review messages we typed).
        if sender_id == self._me_id:
            return
        # Skip the moderator bot itself (its own verdicts).
        if self._bot_id and sender_id == self._bot_id:
            return

        is_dm = bool(event.is_private)
        is_dm_from_owner = is_dm and sender_id == self.review_bot.bot_cfg.owner_user_id
        is_owner = sender_id == self.review_bot.bot_cfg.owner_user_id

        # Disagreement learning: did the owner reply to a verdict we
        # posted via the userbot? verdict_index is keyed by Telethon
        # message id for userbot-posted verdicts.
        reply_to = getattr(msg, "reply_to", None)
        replied_to_id = getattr(reply_to, "reply_to_msg_id", None) if reply_to else None
        if replied_to_id and sender_id == self.review_bot.bot_cfg.owner_user_id:
            if replied_to_id in self.review_bot.verdict_index:
                user_text = (msg.message or "").strip()
                if user_text:
                    log.info(
                        "userbot saw correction reply: chat=%s replied_to=%s text_len=%d",
                        chat_id, replied_to_id, len(user_text),
                    )
                    await self._handle_correction(
                        chat_id=chat_id,
                        anchor_msg_id=msg.id,
                        record=self.review_bot.verdict_index[replied_to_id],
                        user_text=user_text,
                    )
                    return
            else:
                # Owner replied to *something*, but we don't recognize it as
                # one of our verdicts. Most common cause: process restart
                # wiped verdict_index, or reply was to a different message.
                # Tell the owner so they don't think we silently swallowed
                # their feedback.
                log.warning(
                    "owner reply to unknown msg_id=%s in chat=%s "
                    "(verdict_index has %d entries) — feedback NOT applied",
                    replied_to_id, chat_id, len(self.review_bot.verdict_index),
                )
                try:
                    await self.send_message(
                        chat_id,
                        (
                            "⚠️ I see your reply but don't have a record of "
                            "the verdict you're replying to (probably because "
                            "I restarted after posting it). Please re-send the "
                            "application so I can re-review and then reply to "
                            "the new verdict — that one I'll remember."
                        ),
                        reply_to=msg.id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return

        text_blob = msg.message or ""
        pack_url = _find_pack_url_in_telethon_msg(msg)
        looks_like_app = bool(msg.photo or msg.grouped_id or pack_url)
        sender_is_bot = bool(getattr(sender, "bot", False))

        # ---- DM with Moderator ----
        if is_dm:
            # Only the owner gets to interact in DM (avoid spam from randoms).
            if not is_owner:
                return
            if looks_like_app:
                fwd = getattr(msg, "forward", None)
                fwd_info = ""
                if fwd is not None:
                    fwd_from_id = getattr(fwd, "sender_id", None) or getattr(fwd, "from_id", None)
                    fwd_info = f" fwd_from={fwd_from_id} fwd_msg_id={getattr(fwd, 'channel_post', None)}"
                log.info(
                    "userbot saw OWNER-DM app-part: chat=%s msg_id=%s photo=%s text_len=%d pack_url=%r%s",
                    chat_id, msg.id, bool(msg.photo),
                    len(text_blob), pack_url, fwd_info,
                )
                await self._buffer(chat_id, msg)
                return
            # Otherwise it's a question / command for the assistant.
            if text_blob.strip():
                log.info(
                    "userbot owner-DM assistant query: chat=%s text_len=%d",
                    chat_id, len(text_blob),
                )
                await self._handle_assistant(
                    chat_id, msg, sender_id, sender, text_blob, is_owner=True
                )
            return

        # ---- Group / supergroup ----
        if not self.review_bot.allowlist.contains(chat_id):
            return

        # 1) source-bot application: hand to review pipeline
        if sender_is_bot and (
            not self.bot_cfg.source_bot_ids
            or sender_id in self.bot_cfg.source_bot_ids
        ):
            if looks_like_app:
                log.info(
                    "userbot saw app-part: chat=%s sender=%s msg_id=%s photo=%s grouped=%s pack_url=%r",
                    chat_id, sender_id, msg.id, bool(msg.photo), msg.grouped_id, pack_url,
                )
                await self._buffer(chat_id, msg)
                return
            # bot posted something else → ignore
            return

        # 2) human message — only react if directly addressed to Moderator
        addressed = self._is_mentioned(msg) or await self._is_reply_to_me(msg)
        if not addressed:
            return

        # If a human pasted a pack link / photo while talking to Moderator,
        # treat it as an application too (handy for owner ad-hoc reviews).
        if looks_like_app:
            log.info(
                "userbot saw addressed app-part: chat=%s msg_id=%s pack_url=%r",
                chat_id, msg.id, pack_url,
            )
            await self._buffer(chat_id, msg)
            return

        # Otherwise: assistant Q&A.
        if text_blob.strip():
            log.info(
                "userbot assistant query in chat=%s sender=%s owner=%s text_len=%d",
                chat_id, sender_id, is_owner, len(text_blob),
            )
            await self._handle_assistant(
                chat_id, msg, sender_id, sender, text_blob, is_owner=is_owner
            )

    async def _handle_assistant(
        self,
        chat_id: int,
        msg,
        sender_id: int,
        sender,
        text: str,
        *,
        is_owner: bool,
    ) -> None:
        sender_name = (
            getattr(sender, "first_name", None)
            or getattr(sender, "username", None)
            or f"user_{sender_id}"
        )
        await self.send_typing(chat_id)
        try:
            reply = await self.assistant.respond(
                chat_id=chat_id,
                sender_id=sender_id,
                sender_name=sender_name,
                is_owner=is_owner,
                user_text=text,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("assistant.respond failed: %s", e)
            reply = f"Sorry, I errored ({e})."
        if not reply.strip():
            reply = "(I had nothing to say. Try rephrasing?)"
        await self.send_message_safe(chat_id, reply, reply_to=msg.id)

    @staticmethod
    def _normalize_chat_id(chat_id: int) -> int:
        """Telethon may give us a positive supergroup id; PTB / Bot API
        expect the -100… form. We match the bot's allowlist by trying
        both. Returns the value as-is; callers compare via Allowlist
        (which we extend below to accept either form)."""
        return chat_id

    async def _buffer(self, chat_id: int, msg) -> None:
        buf = self.buffers.get(chat_id)
        if buf is None:
            buf = _Buffer()
            self.buffers[chat_id] = buf
        if msg.id in buf.seen_ids:
            return
        buf.seen_ids.add(msg.id)
        buf.messages.append(msg)
        if buf.task and not buf.task.done():
            buf.task.cancel()
        buf.task = asyncio.create_task(self._fire(chat_id))

    async def _fire(self, chat_id: int) -> None:
        try:
            await asyncio.sleep(self.debounce_s)
        except asyncio.CancelledError:
            return
        buf = self.buffers.pop(chat_id, None)
        if not buf or not buf.messages:
            return
        msgs = sorted(buf.messages, key=lambda m: m.id)
        anchor = msgs[0]

        # Merge text from all parts; pull pack url from any of them.
        full_text = "\n".join((m.message or "") for m in msgs).strip()
        pack_url = ""
        for m in msgs:
            pack_url = _find_pack_url_in_telethon_msg(m)
            if pack_url:
                break
        if not pack_url:
            m = _PACK_URL_RE.search(full_text)
            if m:
                pack_url = m.group(0)

        # Download photos.
        photo_paths = await self._download_photos(msgs, chat_id)

        log.info(
            "userbot relay -> review: chat=%s anchor=%s msgs=%d photos=%d pack_url=%r",
            chat_id, anchor.id, len(msgs), len(photo_paths), pack_url,
        )

        try:
            await self.review_bot.run_review(
                chat_id=chat_id,
                anchor_message_id=anchor.id,
                photo_paths=photo_paths,
                full_text=full_text,
                pack_url=pack_url,
                source="userbot",
            )
        except Exception as e:  # noqa: BLE001
            log.exception("review failed for relayed application: %s", e)
            try:
                await self.send_message_safe(
                    chat_id,
                    f"❌ I crashed while reviewing this batch: `{e}`. "
                    "Please re-send and ping me if it keeps happening.",
                    reply_to=anchor.id,
                )
            except Exception:  # noqa: BLE001
                pass

    async def _handle_correction(
        self,
        *,
        chat_id: int,
        anchor_msg_id: int,
        record: dict,
        user_text: str,
    ) -> None:
        """Owner replied to one of our verdicts with a correction. Run the
        learning pipeline (Claude rule synthesis) and post the result."""
        await self.send_typing(chat_id)
        try:
            reply = await self.review_bot.process_correction(record, user_text)
        except Exception as e:  # noqa: BLE001
            log.exception("process_correction failed: %s", e)
            await self.send_message_safe(
                chat_id,
                "Tried to learn from your reply but Claude errored. Try again.",
                reply_to=anchor_msg_id,
            )
            return
        await self.send_message_safe(chat_id, reply, reply_to=anchor_msg_id)

    async def _download_photos(self, msgs, chat_id: int) -> list[Path]:
        out: list[Path] = []
        cache_dir = self.bot_cfg.tg_cache_dir / "_userbot"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for m in msgs:
            if not m.photo:
                continue
            dest = cache_dir / f"{chat_id}_{m.id}.jpg"
            try:
                if not dest.exists():
                    await m.download_media(file=str(dest))
                if dest.exists():
                    out.append(dest)
            except Exception as e:  # noqa: BLE001
                log.warning("userbot photo download failed for msg %s: %s", m.id, e)
        return out
