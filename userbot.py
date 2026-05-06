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
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import (
    MessageEntityMention,
    MessageEntityMentionName,
    SendMessageTypingAction,
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
        # Liveness signals consumed by the health endpoint and heartbeat.
        self.started_at: float = time.time()
        self.last_event_at: float = time.time()
        self._heartbeat_task: asyncio.Task | None = None

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

        # Force Telethon to (a) populate the dialog cache for chats we're
        # already a member of and (b) consume any pending update difference
        # accumulated while we were offline. Without these, fresh deploys
        # have been observed to receive zero update events from chats that
        # existed before the redeploy — the receiver loop is running but
        # has nothing to dispatch because it never asked the server for
        # the catch-up state.
        try:
            n_dialogs = 0
            async for _ in self.client.iter_dialogs(limit=200):
                n_dialogs += 1
            log.info("userbot prefetched %d dialogs", n_dialogs)
        except Exception as e:  # noqa: BLE001
            log.warning("dialog prefetch failed: %s", e)
        try:
            await self.client.catch_up()
            log.info("userbot caught up on pending updates")
        except Exception as e:  # noqa: BLE001
            log.warning("catch_up failed: %s", e)

        # Active replay: walk recent history of every whitelisted chat
        # and process any source-bot applications we don't yet have a
        # verdict for. This is the safety net that makes "won't miss
        # an application" actually true: catch_up alone is at the mercy
        # of Telegram's update-difference window AND (in our experience)
        # doesn't reliably re-dispatch historical messages through
        # registered NewMessage handlers.
        try:
            await self._replay_missed_apps()
        except Exception as e:  # noqa: BLE001
            log.warning("startup replay failed: %s", e)

        # Heartbeat task: periodically log that we're alive. If these
        # stop appearing in logs, the dispatcher loop is dead and Fly's
        # health probe will catch it within the next minute.
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Telethon now dispatches events on the connected client; we
        # return and let the caller hold the asyncio loop open.

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # ---------- Liveness / health ----------

    def is_healthy(self) -> tuple[bool, str]:
        """Used by the HTTP health endpoint and the heartbeat log.

        Healthy iff:
          1. the Telethon client exists and is_connected() is True, AND
          2. we've received at least one event in the last EVENT_GAP_MAX
             seconds, OR we've been up for less than that gap (grace
             window for cold starts in idle chats).
        """
        EVENT_GAP_MAX = 30 * 60  # 30 min — generous for low-traffic chat
        if self.client is None:
            return False, "client not initialised"
        if not self.client.is_connected():
            return False, "telethon client disconnected"
        gap = time.time() - max(self.last_event_at, self.started_at)
        if gap > EVENT_GAP_MAX:
            return False, f"no events for {int(gap)}s (>{EVENT_GAP_MAX}s)"
        return True, "ok"

    async def _heartbeat_loop(self) -> None:
        """Log a one-line liveness signal every 5 minutes so the dispatcher
        going silent is visible in the log stream."""
        try:
            while True:
                await asyncio.sleep(300)
                ok, why = self.is_healthy()
                gap = int(time.time() - self.last_event_at)
                log.info(
                    "userbot heartbeat: connected=%s healthy=%s last_event=%ds_ago note=%s",
                    self.client.is_connected() if self.client else False,
                    ok, gap, why,
                )
        except asyncio.CancelledError:
            return

    # ---------- Startup replay of missed applications ----------

    async def _replay_missed_apps(self) -> None:
        """For each whitelisted chat, walk recent history and process
        any source-bot application we don't already have a verdict for.

        Bounded by REPLAY_AGE_MAX (24 h) and REPLAY_LIMIT (200 messages
        per chat) so we never replay ancient history. Idempotent: relies
        on verdict_index to skip applications already reviewed.

        Media-group anchoring: messages are fed through the same
        ``_buffer()`` debouncer as live traffic, so a multi-part
        application (logo + cover + text-with-link) coalesces into one
        review with the same anchor message id it would have had live —
        which means the verdict_index check is consistent across
        replays.
        """
        REPLAY_AGE_MAX = 24 * 3600
        REPLAY_LIMIT = 200
        if self.client is None:
            return
        cutoff_ts = time.time() - REPLAY_AGE_MAX

        # Set of (chat_id, anchor_msg_id) we've already verdict'd, used
        # to skip messages whose batch was already reviewed. We can only
        # match against the anchor (the first / lowest-id message in
        # the batch) — but iter_messages returns oldest-first when
        # reversed, and our buffer always anchors on min(id), so the
        # check is consistent.
        already = {
            (v.get("chat_id"), v.get("original_message_id"))
            for v in self.review_bot.verdict_index.values()
        }

        chats = self.review_bot.allowlist.all()
        log.info("replay: scanning %d whitelisted chat(s) for missed apps", len(chats))
        for entry in chats:
            chat_id = entry.get("chat_id")
            if chat_id is None:
                continue
            try:
                entity = await self.client.get_entity(chat_id)
            except Exception as e:  # noqa: BLE001
                log.warning("replay: get_entity(%s) failed: %s — skipping",
                            chat_id, e)
                continue

            candidates: list = []
            try:
                async for m in self.client.iter_messages(entity, limit=REPLAY_LIMIT):
                    # iter_messages yields newest-first; we exit when
                    # we cross the age cutoff.
                    if m.date and m.date.timestamp() < cutoff_ts:
                        break
                    s_id = m.sender_id
                    if (
                        not self.bot_cfg.source_bot_ids
                        or s_id not in self.bot_cfg.source_bot_ids
                    ):
                        continue
                    looks_like_app = bool(
                        m.photo or m.grouped_id
                        or _find_pack_url_in_telethon_msg(m)
                    )
                    if not looks_like_app:
                        continue
                    candidates.append(m)
            except Exception as e:  # noqa: BLE001
                log.warning("replay: iter_messages failed for chat %s: %s",
                            chat_id, e)
                continue

            if not candidates:
                log.info("replay: chat=%s — no app-shaped source-bot messages in window", chat_id)
                continue

            # Group by grouped_id (None grouped_id = standalone). For
            # each group, anchor = min(id). Skip groups whose anchor is
            # already in verdict_index.
            groups: dict[object, list] = {}
            for m in candidates:
                key = m.grouped_id if m.grouped_id else f"solo-{m.id}"
                groups.setdefault(key, []).append(m)

            replayed = 0
            for key, msgs in groups.items():
                anchor_id = min(m.id for m in msgs)
                if (chat_id, anchor_id) in already:
                    continue
                log.info(
                    "replay: re-injecting missed batch chat=%s anchor=%s parts=%d",
                    chat_id, anchor_id, len(msgs),
                )
                # Feed oldest-first into the debouncer; buffer will
                # combine them (3 s debounce) and _fire will run.
                for m in sorted(msgs, key=lambda x: x.id):
                    await self._buffer(chat_id, m)
                replayed += 1

            log.info(
                "replay: chat=%s — %d candidate batch(es), %d replayed, %d already done",
                chat_id, len(groups), replayed, len(groups) - replayed,
            )

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
        """Send as PLAIN TEXT — no Markdown parsing.

        Telethon's `md` parser is MarkdownV2-style (needs ``**bold**``,
        escaped ``-``/``.``/``!``, etc.). Most of our text — Claude's
        freeform reasoning, the rules summary, assistant replies — is
        written in legacy single-asterisk style and contains punctuation
        that V2 chokes on. Trying to parse it produced two failure modes
        for the user: literal ``*`` everywhere (parser silently dropped
        the message and we delivered the unparsed source) or stray ``\\_``
        in usernames like ``@sticker_bot``.

        The fix is the simplest one: send plain text. We strip any
        leftover Markdown escape characters defensively so we don't
        deliver ``\\_`` either.
        """
        from review_bot import _strip_markdown
        plain = _strip_markdown(text)
        sent = await self.send_message(
            chat_id, plain, reply_to=reply_to, markdown=False
        )
        log.info(
            "userbot sent: chat=%s reply_to=%s msg_id=%s bytes=%d",
            chat_id, reply_to, getattr(sent, "id", None), len(plain),
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
        """Show "Moderator is typing..." in the chat for ~5 seconds.

        We used to use ``async with self.client.action(chat, "typing"): pass``
        — but the context manager cancels the indicator on ``__aexit__``,
        so ``pass`` made it a no-op (typing fired and was cancelled within
        microseconds). Send the underlying request directly instead so
        the indicator actually lasts until Telegram's natural ~5s expiry,
        which is roughly how long Claude takes to respond anyway.
        """
        if self.client is None:
            return
        try:
            await self.client(SetTypingRequest(
                peer=chat_id,
                action=SendMessageTypingAction(),
            ))
        except Exception as e:  # noqa: BLE001
            log.debug("send_typing failed for chat %s: %s", chat_id, e)

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
                        "and I'll respond. Type something like \"what are "
                        "the rules?\" to get started."
                    ),
                )
            except Exception:  # noqa: BLE001
                pass
        elif event.user_left or event.user_kicked:
            removed = self.review_bot.allowlist.remove(chat_id)
            if removed:
                log.info("userbot left chat %s; un-whitelisted", chat_id)

    # ---------- detection helpers ----------

    def _mention_reason(self, msg) -> str | None:
        """Return a short tag for HOW the message addressed Moderator, or
        None if it didn't.

        Order of checks:
          1. Telegram's own ``msg.mentioned`` flag — server-side signal that
             the current user was mentioned (works for accounts WITHOUT a
             @username because the picker emits MessageEntityMentionName
             carrying our user_id).
          2. Explicit MessageEntityMentionName matching our user_id.
          3. MessageEntityMention text "@username" if we happen to have one.
        """
        if getattr(msg, "mentioned", False):
            return "tg-flag"
        if not msg.entities:
            return None
        text = msg.message or ""
        for ent in msg.entities:
            if isinstance(ent, MessageEntityMentionName) and ent.user_id == self._me_id:
                return "entity-mention-name"
            if isinstance(ent, MessageEntityMention) and self._me_username:
                start, length = ent.offset, ent.length
                snippet = text[start:start + length].lstrip("@").lower()
                if snippet == self._me_username:
                    return "entity-mention-username"
        return None

    async def _reply_to_me_reason(self, msg) -> str | None:
        """Return tag if ``msg`` is a reply to a message Moderator authored.

        Two paths: cheap lookup in our verdict_index (keyed by
        ``chat_id:msg_id``), or fetching the replied-to message and
        comparing sender_id. Some forum-style supergroups put
        ``reply_to_top_id`` (thread root) in addition to
        ``reply_to_msg_id`` — we always use the immediate parent.
        """
        reply_to = getattr(msg, "reply_to", None)
        replied_to_id = getattr(reply_to, "reply_to_msg_id", None) if reply_to else None
        if not replied_to_id:
            return None
        chat_id = self._normalize_chat_id(msg.chat_id)
        v = self.review_bot.verdict_index.get(
            self.review_bot._vk(chat_id, replied_to_id)
        )
        if v and v.get("transport") == "userbot":
            return "verdict-index"
        try:
            replied = await msg.get_reply_message()
        except Exception as e:  # noqa: BLE001
            log.debug("get_reply_message failed for msg %s: %s", msg.id, e)
            return None
        if replied and replied.sender_id == self._me_id:
            return "reply-fetch"
        return None

    async def _on_new_message(self, event) -> None:
        msg = event.message
        chat_id = self._normalize_chat_id(event.chat_id)
        # Liveness ping: any inbound event resets the watchdog.
        self.last_event_at = time.time()

        # Top-of-handler trace so EVERY inbound dispatch is visible in
        # logs. Cheap, single line per event. If we ever go silent again
        # this tells us instantly whether the dispatcher is alive.
        try:
            sender = await event.get_sender()
        except Exception as e:  # noqa: BLE001
            log.warning("get_sender failed: chat=%s msg_id=%s err=%s",
                        chat_id, getattr(msg, "id", None), e)
            sender = None
        sender_id = getattr(sender, "id", None) if sender is not None else None
        log.info(
            "userbot rx: chat=%s sender=%s msg_id=%s photo=%s grouped=%s text_len=%d",
            chat_id, sender_id, getattr(msg, "id", None),
            bool(getattr(msg, "photo", None)),
            getattr(msg, "grouped_id", None),
            len(getattr(msg, "message", "") or ""),
        )

        if sender_id is None:
            log.info("drop: sender_id is None (chat=%s msg_id=%s)",
                     chat_id, getattr(msg, "id", None))
            return
        if sender_id == self._me_id:
            return
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
            verdict_key = self.review_bot._vk(chat_id, replied_to_id)
            if verdict_key in self.review_bot.verdict_index:
                user_text = (msg.message or "").strip()
                if user_text:
                    log.info(
                        "userbot saw correction reply: chat=%s replied_to=%s text_len=%d",
                        chat_id, replied_to_id, len(user_text),
                    )
                    await self._handle_correction(
                        chat_id=chat_id,
                        anchor_msg_id=msg.id,
                        record=self.review_bot.verdict_index[verdict_key],
                        user_text=user_text,
                    )
                    return
            # else: replied_to_id isn't a known verdict. Don't treat it as a
            # missed correction — it's just a normal reply (e.g. asking a
            # question by replying to my intro / status / chat message). Fall
            # through to the assistant path below.

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

        # 1) source-bot application: hand to review pipeline.
        # IMPORTANT: this path requires NO @-mention or reply from
        # @sticker_bot — applications are detected purely by sender id
        # plus the application-shape heuristics (photo / grouped media /
        # t.me/addstickers/<slug> link).
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
            log.debug(
                "ignoring source-bot non-app msg: chat=%s sender=%s msg_id=%s text_len=%d",
                chat_id, sender_id, msg.id, len(text_blob),
            )
            return

        # 2) human (or third-party-bot) message: react ONLY if the
        # message directly addresses Moderator. Anything else in the chat
        # is none of our business — this is what stops random group
        # chatter from triggering the assistant.
        mention_tag = self._mention_reason(msg)
        reply_tag = await self._reply_to_me_reason(msg)
        addressed = bool(mention_tag or reply_tag)
        if not addressed:
            log.debug(
                "ignoring unaddressed msg: chat=%s sender=%s msg_id=%s text_len=%d "
                "(no mention, not a reply to me)",
                chat_id, sender_id, msg.id, len(text_blob),
            )
            return
        log.info(
            "userbot addressed: chat=%s sender=%s msg_id=%s via=%s text_len=%d",
            chat_id, sender_id, msg.id,
            "+".join(t for t in (mention_tag, reply_tag) if t),
            len(text_blob),
        )

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
                    f"❌ I crashed while reviewing this batch: {e}. "
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
