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
        self._bot_id: int | None = None  # the moderator bot's id

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
        try:
            self._bot_id = (await self.review_bot.bot.get_me()).id
        except Exception:  # noqa: BLE001
            self._bot_id = None
        log.info(
            "userbot logged in as id=%s name=%r; moderator bot id=%s",
            self._me_id, me.first_name, self._bot_id,
        )
        self.client.add_event_handler(self._on_new_message, events.NewMessage())
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
            return await self.send_message(
                chat_id, text, reply_to=reply_to, markdown=True
            )
        except (RPCError, ValueError) as e:
            log.warning("telethon md send failed (%s); falling back to plain", e)
            return await self.send_message(
                chat_id, _strip_markdown(text), reply_to=reply_to, markdown=False
            )

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

        is_dm_from_owner = (
            event.is_private
            and sender_id == self.review_bot.bot_cfg.owner_user_id
        )

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

        if is_dm_from_owner:
            # Owner is testing via DM with the Moderator account. Always
            # process; skip the chat-allowlist + is-bot gates below.
            text_blob = msg.message or ""
            pack_url = _find_pack_url_in_telethon_msg(msg)
            looks_like_app = bool(msg.photo or msg.grouped_id or pack_url)
            if not looks_like_app:
                return
            log.info(
                "userbot saw OWNER-DM app-part: chat=%s msg_id=%s photo=%s pack_url=%r",
                chat_id, msg.id, bool(msg.photo), pack_url,
            )
            await self._buffer(chat_id, msg)
            return

        # Group / supergroup path: only act in chats the owner has
        # whitelisted via the bot, and only on messages from configured
        # source bots (so we don't relay random human chit-chat).
        if not self.review_bot.allowlist.contains(chat_id):
            return
        is_bot = bool(getattr(sender, "bot", False))
        if not is_bot:
            return
        if self.bot_cfg.source_bot_ids and sender_id not in self.bot_cfg.source_bot_ids:
            return

        text_blob = msg.message or ""
        pack_url = _find_pack_url_in_telethon_msg(msg)
        looks_like_app = bool(msg.photo or msg.grouped_id or pack_url)
        if not looks_like_app:
            return

        log.info(
            "userbot saw app-part: chat=%s sender=%s msg_id=%s photo=%s grouped=%s pack_url=%r",
            chat_id, sender_id, msg.id, bool(msg.photo), msg.grouped_id, pack_url,
        )

        await self._buffer(chat_id, msg)

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
