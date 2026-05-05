"""
Sticker Pad — live moderation bot.

Listens in chats where @sticker_bot posts new pack applications. Each
application is a 2-photo media group (logo + cover) with a caption that
includes a t.me/addstickers/<slug> link. For each application the bot
replies with a one-line ✅ APPROVE / ❌ REJECT suggestion and reason,
using the same Claude-Sonnet brain as the offline moderate_packs.py
pipeline, plus any operator amendments stored via /addrule.

Commands:
  /start             quick intro
  /help              same as /start
  /rules             show the current ruleset (summary + amendments)
  /addrule <text>    add a new rule (or send /addrule alone, then text)
  /delrule <id>      deactivate amendment by id
  /listrules         compact list of active amendments

Disagreement learning:
  Reply to any of my verdict messages with your correct decision (e.g.
  "approve, this is just satire" or "reject, NFT derivative"). I'll
  call Claude to decide whether your correction implies a rule change,
  and if so I'll add/amend a rule and confirm.

Run:
  python review_bot.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from telegram import Update, Message, InputMediaPhoto
from telegram.constants import ParseMode, ChatType
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Reuse the brain + ingest helpers from the offline pipeline.
from moderate_packs import (
    Config,
    CLAUDE_PROMPT,
    _claude_one_call,
    _normalize_to_part,
    claude_pack_review,
)
from tg_ingest import slug_from_url, ingest_pack
from rules_store import get_default_store, Amendment

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore


log = logging.getLogger("review_bot")


# ---------- env / config ----------

@dataclass
class BotConfig:
    review_token: str
    pipeline_token: str  # used by tg_ingest for getStickerSet/getFile
    source_bot_username: str
    owner_user_id: int
    seed_allowed_chat_ids: set[int]
    allowlist_path: Path
    media_group_debounce_s: float = 2.0
    tg_cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TG_CACHE_DIR", ".tg_cache"))
    )

    @classmethod
    def from_env(cls) -> "BotConfig":
        load_dotenv()
        review_token = os.getenv("REVIEW_BOT_TOKEN", "").strip()
        if not review_token:
            raise SystemExit("REVIEW_BOT_TOKEN not set in .env")
        pipeline_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or review_token
        owner_raw = (os.getenv("OWNER_USER_ID", "") or "").strip()
        if not owner_raw:
            raise SystemExit(
                "OWNER_USER_ID not set in .env — refusing to start without an owner. "
                "This bot must be owner-locked."
            )
        try:
            owner_id = int(owner_raw)
        except ValueError as e:
            raise SystemExit(f"OWNER_USER_ID must be an integer, got {owner_raw!r}") from e
        return cls(
            review_token=review_token,
            pipeline_token=pipeline_token,
            source_bot_username=os.getenv("SOURCE_BOT_USERNAME", "sticker_bot").lstrip("@"),
            owner_user_id=owner_id,
            seed_allowed_chat_ids={
                int(x) for x in (os.getenv("ALLOWED_CHAT_IDS", "") or "").split(",") if x.strip()
            },
            allowlist_path=Path(os.getenv("ALLOWLIST_PATH", ".allowed_chats.json")),
        )


# ---------- runtime allowlist ----------

class Allowlist:
    """Persisted set of chat ids the owner has added the bot to.

    The owner's DM chat (private chat with the bot) is implicitly always
    allowed and is not stored here.
    """

    def __init__(self, path: Path, seed: set[int] | None = None):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._chats: dict[int, dict] = {}  # chat_id -> {title, type, added_at}
        self._load()
        if seed:
            for cid in seed:
                if cid not in self._chats:
                    self._chats[cid] = {"title": "(env-seeded)", "type": "?", "added_at": time.time()}
            self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for entry in data.get("chats", []):
            try:
                self._chats[int(entry["chat_id"])] = {
                    "title": entry.get("title", ""),
                    "type": entry.get("type", ""),
                    "added_at": entry.get("added_at", time.time()),
                }
            except (KeyError, ValueError, TypeError):
                continue

    def _save(self) -> None:
        data = {
            "chats": [
                {"chat_id": cid, **meta} for cid, meta in self._chats.items()
            ],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def contains(self, chat_id: int) -> bool:
        with self._lock:
            return chat_id in self._chats

    def add(self, chat_id: int, title: str = "", chat_type: str = "") -> bool:
        with self._lock:
            if chat_id in self._chats:
                # Refresh metadata if we learned a better title.
                if title:
                    self._chats[chat_id]["title"] = title
                if chat_type:
                    self._chats[chat_id]["type"] = chat_type
                self._save()
                return False
            self._chats[chat_id] = {"title": title, "type": chat_type, "added_at": time.time()}
            self._save()
            return True

    def remove(self, chat_id: int) -> dict | None:
        with self._lock:
            entry = self._chats.pop(chat_id, None)
            if entry is not None:
                self._save()
            return entry

    def all(self) -> list[tuple[int, dict]]:
        with self._lock:
            return sorted(self._chats.items(), key=lambda kv: kv[1].get("added_at", 0))


# ---------- application detection ----------

PACK_URL_RE = re.compile(
    r"https?://t\.me/addstickers/([A-Za-z0-9_]+)",
    re.IGNORECASE,
)


def find_pack_url(text: str) -> str | None:
    if not text:
        return None
    m = PACK_URL_RE.search(text)
    return m.group(0) if m else None


def _find_pack_url_in_entities(msgs: list[Message]) -> str | None:
    """Look for a t.me/addstickers/<slug> URL hidden inside message
    entities (text_link, url) of any buffered message. @sticker_bot
    sometimes sends the link as a clickable entity rather than as
    visible text, in which case msg.text alone won't contain it.

    Telegram offers two relevant entity types:
      - "url"        — visible URL in the text (covered by the regex
                       on the visible text already, but checked again
                       to handle tricky encodings).
      - "text_link"  — visible label with a hidden URL on entity.url.
    We also walk reply_markup inline buttons for `url` fields.
    """
    for m in msgs:
        for ent in (m.entities or []) + (m.caption_entities or []):
            url = getattr(ent, "url", None)
            if not url:
                # For "url"-type entities the URL is the substring of
                # text/caption at [offset, offset+length).
                if getattr(ent, "type", "") == "url":
                    src = m.text or m.caption or ""
                    try:
                        url = src[ent.offset : ent.offset + ent.length]
                    except Exception:  # noqa: BLE001
                        url = None
            if url:
                hit = find_pack_url(url)
                if hit:
                    return hit
        markup = getattr(m, "reply_markup", None)
        if markup is not None:
            for row in getattr(markup, "inline_keyboard", []) or []:
                for btn in row or []:
                    btn_url = getattr(btn, "url", None)
                    if btn_url:
                        hit = find_pack_url(btn_url)
                        if hit:
                            return hit
        # link_preview_options.url — Telegram's auto-fetched preview.
        lpo = getattr(m, "link_preview_options", None)
        if lpo is not None:
            hit = find_pack_url(getattr(lpo, "url", "") or "")
            if hit:
                return hit
    return None


def message_text_or_caption(msg: Message) -> str:
    return (msg.text or msg.caption or "").strip()


def is_from_source_bot(msg: Message, source_username: str) -> bool:
    """True if this message either was sent by the source bot or was
    forwarded from it. Uses python-telegram-bot v21+ forward_origin API."""
    target = (source_username or "").lower().lstrip("@")
    if not target:
        return False

    sender = msg.from_user
    if sender and sender.username and sender.username.lower() == target:
        return True

    origin = getattr(msg, "forward_origin", None)
    if origin is None:
        return False

    # MessageOriginUser  -> .sender_user (User)
    user = getattr(origin, "sender_user", None)
    if user and user.username and user.username.lower() == target:
        return True

    # MessageOriginChat  -> .sender_chat (Chat)
    chat = getattr(origin, "sender_chat", None)
    if chat and getattr(chat, "username", "") and chat.username.lower() == target:
        return True

    # MessageOriginChannel  -> .chat (Chat)
    channel = getattr(origin, "chat", None)
    if channel and getattr(channel, "username", "") and channel.username.lower() == target:
        return True

    # MessageOriginHiddenUser  -> .sender_user_name (str)
    hidden_name = getattr(origin, "sender_user_name", None)
    if hidden_name and hidden_name.lower() == target:
        return True

    return False


# ---------- state ----------

@dataclass
class MediaGroupBuffer:
    messages: list[Message] = field(default_factory=list)
    task: asyncio.Task | None = None
    chat_id: int = 0


@dataclass
class ApplicationBuffer:
    """Per-chat buffer that groups all messages of a single application
    together, regardless of whether they share a media_group_id.

    @sticker_bot's applications come as ≥2 separate messages: a media
    group of 2 photos (logo + cover) and a separate text message with
    the description and pack link. Without buffering across messages we
    review the photos and the text as two independent applications."""
    messages: list[Message] = field(default_factory=list)
    seen_ids: set[int] = field(default_factory=set)
    task: asyncio.Task | None = None


@dataclass
class PendingRuleInput:
    """User typed /addrule with no args — waiting for the rule text in the next message."""
    user_id: int
    chat_id: int
    expires_at: float


# ---------- main bot ----------

class ReviewBot:
    def __init__(self, bot_cfg: BotConfig, model_cfg: Config):
        self.bot_cfg = bot_cfg
        self.model_cfg = model_cfg
        self.rules = get_default_store()

        if not model_cfg.anthropic_key:
            raise SystemExit("ANTHROPIC_API_KEY not set in .env")
        if Anthropic is None:
            raise SystemExit("anthropic package not installed (pip install -r requirements.txt)")
        self.claude = Anthropic(api_key=model_cfg.anthropic_key)

        self.media_groups: dict[str, MediaGroupBuffer] = {}
        self.app_buffers: dict[int, ApplicationBuffer] = {}  # by chat_id
        self.app_debounce_s: float = 3.0
        self.pending_rule_input: dict[int, PendingRuleInput] = {}  # by user_id
        self.started_at: float = time.time()
        self.reviews_served: int = 0  # incremented per posted verdict
        # Track verdict messages we posted so we can recognize replies as
        # disagreement signals: bot_message_id -> {original_msg_id, pack_url, verdict, reasoning}
        self.verdict_index: dict[int, dict] = {}

        # Persisted allowlist of group chats the owner has personally added
        # the bot to. The owner's DM chat is always allowed implicitly.
        self.allowlist = Allowlist(bot_cfg.allowlist_path, seed=bot_cfg.seed_allowed_chat_ids)
        # Set of non-owner DM chats we've already told "this bot is private",
        # so we don't keep re-spamming them.
        self._dm_warned: set[int] = set()

        bot_cfg.tg_cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- chat/owner gating ----------

    def is_owner(self, user_id: int | None) -> bool:
        return user_id is not None and user_id == self.bot_cfg.owner_user_id

    # Backward-compat alias used by older command paths.
    def is_admin(self, user_id: int) -> bool:
        return self.is_owner(user_id)

    def chat_allowed(self, msg: Message) -> bool:
        """A chat is allowed if (a) it's the owner's DM, or (b) it's a
        group/supergroup that's in the runtime allowlist.

        Bot's own commands invoked from non-owner DMs or non-allowlisted
        groups are silently dropped (a one-time DM is sent to non-owner
        DM users so they know the bot is private)."""
        chat = msg.chat
        if chat.type == ChatType.PRIVATE:
            user = msg.from_user
            return self.is_owner(user.id if user else None)
        # group / supergroup / channel
        return self.allowlist.contains(chat.id)

    # ---------- commands ----------

    async def _gate(self, update: Update, owner_only_in_dm: bool = True) -> Message | None:
        """Return msg if this update should be acted on, else None.

        Silently drops updates from non-owner users in DMs (so randos
        can't /rules-spam us) and from non-allowlisted groups."""
        msg = update.effective_message
        if not msg:
            return None
        if not self.chat_allowed(msg):
            if msg.chat.type == ChatType.PRIVATE and owner_only_in_dm:
                if msg.chat.id not in self._dm_warned:
                    self._dm_warned.add(msg.chat.id)
                    try:
                        await msg.reply_text(
                            "This bot is private and only responds to its owner."
                        )
                    except Exception:  # noqa: BLE001
                        pass
            return None
        return msg

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        if not msg:
            return
        text = (
            "👋 Hi, I'm the *Sticker Pad* moderation reviewer.\n\n"
            f"In group chats I read applications posted by @{_md_escape(self.bot_cfg.source_bot_username)} "
            "and reply with ✅ APPROVE or ❌ REJECT plus a one-line reason.\n\n"
            "*Test me here:* send a pack as 2 photos (logo + cover) with a caption containing "
            "the `t.me/addstickers/<slug>` link, or just paste the link alone — I'll review.\n\n"
            "Commands:\n"
            "• /rules — show the policy and any operator amendments\n"
            "• /addrule <text> — add a rule (omit text and I'll prompt you)\n"
            "• /delrule <id> — remove an amendment\n"
            "• /listrules — compact list of amendments only\n"
            "• /chats — list whitelisted group chats\n"
            "• /leavechat <chat_id> — un-whitelist and leave a group\n"
            "• /admin — runtime status, model, host, uptime\n\n"
            "If you disagree with a verdict, *reply to my message* with the correct decision "
            "(\"approve\" or \"reject\") and a short reason. I'll learn from it.\n\n"
            "_This bot is owner-locked: only you can DM me, and I only join chats *you* personally add me to._"
        )
        await msg.reply_text(text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    async def cmd_rules(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        if not msg:
            return
        text = self.rules.rules_summary(source_bot=self.bot_cfg.source_bot_username)
        # Telegram caps text at 4096 chars; chunk if needed.
        for chunk in _chunk(text, 3900):
            await msg.reply_text(chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    async def cmd_listrules(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        if not msg:
            return
        items = self.rules.active()
        if not items:
            await msg.reply_text("_No operator amendments yet._", parse_mode=ParseMode.MARKDOWN)
            return
        lines = ["*Active amendments*:"]
        for a in items:
            tag = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢", "NOTE": "🛈"}.get(a.category, "🛈")
            lines.append(f"`#{a.id}` {tag} *{a.category}* — {a.text}")
        await msg.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def cmd_addrule(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        user = update.effective_user
        if not msg or not user:
            return
        if not self.is_owner(user.id):
            await msg.reply_text("Only the owner can edit rules.")
            return
        raw = " ".join(ctx.args or []).strip()
        if not raw:
            self.pending_rule_input[user.id] = PendingRuleInput(
                user_id=user.id,
                chat_id=msg.chat_id,
                expires_at=time.time() + 300,
            )
            await msg.reply_text(
                "OK — send the rule text in your next message in this chat (within 5 min).\n"
                "I'll analyze it, fit it to the existing structure, and confirm what I added."
            )
            return
        await self._ingest_proposed_rule(msg, raw)

    async def cmd_delrule(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        user = update.effective_user
        if not msg or not user:
            return
        if not self.is_owner(user.id):
            await msg.reply_text("Only the owner can edit rules.")
            return
        if not ctx.args:
            await msg.reply_text("Usage: `/delrule <id>` (see /listrules)", parse_mode=ParseMode.MARKDOWN)
            return
        try:
            rule_id = int(ctx.args[0].lstrip("#"))
        except ValueError:
            await msg.reply_text("Rule id must be an integer.")
            return
        a = self.rules.deactivate(rule_id)
        if not a:
            await msg.reply_text(f"No active rule with id #{rule_id}.")
            return
        await msg.reply_text(
            f"🗑 Removed amendment `#{a.id}` ({a.category}): {a.text}",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def cmd_admin(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        user = update.effective_user
        if not msg or not user:
            return
        if not self.is_owner(user.id):
            return

        # Detect host: fly.io / Railway / Render / generic / local.
        host = "local"
        host_extra = ""
        if os.getenv("FLY_APP_NAME"):
            host = "fly.io"
            host_extra = (
                f" · app `{_md_escape(os.getenv('FLY_APP_NAME', ''))}`"
                f" · region `{_md_escape(os.getenv('FLY_REGION', '?'))}`"
                f" · machine `{_md_escape((os.getenv('FLY_MACHINE_ID', '') or '?')[:12])}`"
            )
        elif os.getenv("RAILWAY_ENVIRONMENT"):
            host = "railway"
        elif os.getenv("RENDER"):
            host = "render"
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            host = "kubernetes"

        uptime_s = time.time() - self.started_at
        started_iso = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(self.started_at))

        try:
            me = await ctx.bot.get_me()
            bot_handle = f"@{me.username}" if me.username else f"id={me.id}"
        except Exception:  # noqa: BLE001
            bot_handle = "(unknown)"

        py_ver = sys.version.split()[0]

        lines = [
            "*🛠 Admin status*",
            "",
            f"• *Bot:* {_md_escape(bot_handle)}",
            f"• *Host:* `{host}`{host_extra}",
            f"• *PID:* `{os.getpid()}` · *Python:* `{py_ver}`",
            f"• *Started:* {started_iso}",
            f"• *Uptime:* {_format_uptime(uptime_s)}",
            "",
            "*🤖 Models*",
            f"• *Review (vision + verdict):* `{_md_escape(self.model_cfg.claude_model)}`",
            f"• *Rule curator / learner:* `{_md_escape(self.model_cfg.claude_model)}` (same)",
            "",
            "*📊 Activity since boot*",
            f"• *Verdicts posted:* {self.reviews_served}",
            f"• *Verdict index size:* {len(self.verdict_index)}",
            f"• *Pending application buffers:* {len(self.app_buffers)}",
            "",
            "*📋 State*",
            f"• *Active rule amendments:* {len(self.rules.active())}",
            f"• *Whitelisted chats:* {len(self.allowlist.all())}",
            f"• *Owner id:* `{self.bot_cfg.owner_user_id}`",
            f"• *Source bot:* @{_md_escape(self.bot_cfg.source_bot_username)}",
            "",
            "*💾 Storage paths*",
            f"• *Rules store:* `{_md_escape(str(self.rules.path))}`",
            f"• *Allowlist:* `{_md_escape(str(self.allowlist.path))}`",
            f"• *TG cache:* `{_md_escape(str(self.bot_cfg.tg_cache_dir))}`",
        ]
        await _safe_reply(
            msg,
            "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )

    async def cmd_chats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        user = update.effective_user
        if not msg or not user:
            return
        if not self.is_owner(user.id):
            return
        rows = self.allowlist.all()
        if not rows:
            await msg.reply_text(
                "_No whitelisted group chats yet. Add me to a group to whitelist it._",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        lines = ["*Whitelisted chats:*"]
        for cid, meta in rows:
            title = _md_escape(meta.get("title") or "(untitled)")
            ctype = _md_escape(meta.get("type") or "?")
            lines.append(f"`{cid}` — {title} _({ctype})_")
        await msg.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    async def cmd_leavechat(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = await self._gate(update)
        user = update.effective_user
        if not msg or not user:
            return
        if not self.is_owner(user.id):
            return
        if not ctx.args:
            await msg.reply_text(
                "Usage: `/leavechat <chat_id>` (see /chats)", parse_mode=ParseMode.MARKDOWN
            )
            return
        try:
            target = int(ctx.args[0])
        except ValueError:
            await msg.reply_text("Chat id must be an integer.")
            return
        entry = self.allowlist.remove(target)
        if not entry:
            await msg.reply_text(f"Chat `{target}` was not whitelisted.", parse_mode=ParseMode.MARKDOWN)
            return
        try:
            await ctx.bot.leave_chat(chat_id=target)
            await msg.reply_text(
                f"👋 Left and un-whitelisted `{target}` ({_md_escape(entry.get('title', ''))}).",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as e:  # noqa: BLE001
            await msg.reply_text(
                f"Un-whitelisted `{target}` but failed to leave: {e}",
                parse_mode=ParseMode.MARKDOWN,
            )

    # ---------- bot-membership change handler ----------

    async def on_my_chat_member(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Fired when the bot's own membership changes in any chat.

        Owner-locked policy:
          • If the owner adds the bot to a group/supergroup → whitelist it,
            persist, DM the owner a confirmation.
          • If anyone else adds the bot anywhere → leave immediately and
            DM the owner a heads-up.
        """
        cmu = update.my_chat_member
        if not cmu:
            return
        new_status = cmu.new_chat_member.status if cmu.new_chat_member else None
        old_status = cmu.old_chat_member.status if cmu.old_chat_member else None
        chat = cmu.chat
        actor = cmu.from_user

        in_chat_now = new_status in ("member", "administrator", "creator")
        was_in_chat = old_status in ("member", "administrator", "creator")
        added = in_chat_now and not was_in_chat
        removed = (not in_chat_now) and was_in_chat

        # Don't react to private-chat membership "events" (DMs don't fire
        # this in any meaningful way for us).
        if chat.type == ChatType.PRIVATE:
            return

        if added:
            actor_id = actor.id if actor else None
            if self.is_owner(actor_id):
                added_now = self.allowlist.add(
                    chat_id=chat.id,
                    title=chat.title or chat.username or "",
                    chat_type=str(chat.type),
                )
                log.info(
                    "owner added bot to chat %s (%s) — %s",
                    chat.id,
                    chat.title or "?",
                    "whitelisted" if added_now else "already whitelisted",
                )
                await self._notify_owner(
                    ctx,
                    "✅ Whitelisted chat `{cid}` — *{title}* ({ctype}).\n"
                    "I'll review applications from @{src} here.\n\n"
                    "_Reminder: turn off privacy mode in @BotFather (`/setprivacy` → "
                    "Disable) or make me a chat admin so I can read group messages._".format(
                        cid=chat.id,
                        title=_md_escape(chat.title or "(untitled)"),
                        ctype=chat.type,
                        src=_md_escape(self.bot_cfg.source_bot_username),
                    ),
                )
            else:
                actor_label = (
                    f"@{actor.username}"
                    if actor and actor.username
                    else (f"id={actor.id}" if actor else "(unknown)")
                )
                log.warning(
                    "non-owner %s added bot to chat %s — leaving",
                    actor_label,
                    chat.id,
                )
                try:
                    await ctx.bot.leave_chat(chat_id=chat.id)
                except Exception as e:  # noqa: BLE001
                    log.warning("leave_chat failed for %s: %s", chat.id, e)
                await self._notify_owner(
                    ctx,
                    "⚠️ {who} tried to add me to chat `{cid}` (*{title}*). "
                    "I left immediately. Use /chats to verify.".format(
                        who=_md_escape(actor_label),
                        cid=chat.id,
                        title=_md_escape(chat.title or "(untitled)"),
                    ),
                )
            return

        if removed:
            # If we got kicked or left, drop from the allowlist quietly.
            entry = self.allowlist.remove(chat.id)
            if entry:
                log.info("removed from chat %s; un-whitelisted", chat.id)
                await self._notify_owner(
                    ctx,
                    "🚪 I was removed from chat `{cid}` ({title}). Un-whitelisted.".format(
                        cid=chat.id,
                        title=_md_escape(entry.get("title", "(untitled)")),
                    ),
                )

    async def _notify_owner(self, ctx: ContextTypes.DEFAULT_TYPE, text: str) -> None:
        """DM the owner. Silently no-ops if the owner hasn't /start-ed us yet."""
        try:
            await ctx.bot.send_message(
                chat_id=self.bot_cfg.owner_user_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("could not DM owner (%s): %s", self.bot_cfg.owner_user_id, e)

    # ---------- generic message router ----------

    async def on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        if not msg:
            return
        if not self.chat_allowed(msg):
            # Owner-locked: if a non-owner DM'd us, send a one-time polite
            # heads-up so they know not to expect a reply, then stay quiet.
            if msg.chat.type == ChatType.PRIVATE and not self.is_owner(
                msg.from_user.id if msg.from_user else None
            ):
                if msg.chat.id not in self._dm_warned:
                    self._dm_warned.add(msg.chat.id)
                    try:
                        await msg.reply_text(
                            "This bot is private and only responds to its owner. "
                            "Sorry for the noise."
                        )
                    except Exception:  # noqa: BLE001
                        pass
            return

        # 1) Pending /addrule input?
        user = update.effective_user
        if user and user.id in self.pending_rule_input:
            pending = self.pending_rule_input[user.id]
            if (
                pending.chat_id == msg.chat_id
                and time.time() < pending.expires_at
                and not (msg.text or "").startswith("/")
            ):
                self.pending_rule_input.pop(user.id, None)
                text = (msg.text or msg.caption or "").strip()
                if text:
                    await self._ingest_proposed_rule(msg, text)
                    return
            else:
                self.pending_rule_input.pop(user.id, None)

        # 2) Reply to one of our verdict messages? -> disagreement learning.
        if (
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.is_bot
            and msg.reply_to_message.message_id in self.verdict_index
        ):
            await self._handle_disagreement(msg)
            return

        # 3) Application detection. In groups: only act on @sticker_bot or
        #    when the bot is explicitly mentioned/replied to. In private
        #    chat: act on anything that looks like an application.
        is_private = msg.chat.type == ChatType.PRIVATE
        from_source = is_from_source_bot(msg, self.bot_cfg.source_bot_username)
        addressed_to_us = False
        if not is_private and not from_source:
            me = (await ctx.bot.get_me()).username or ""
            text_blob = message_text_or_caption(msg)
            if me and ("@" + me.lower()) in text_blob.lower():
                addressed_to_us = True
            elif msg.reply_to_message and msg.reply_to_message.from_user and \
                    msg.reply_to_message.from_user.username and \
                    msg.reply_to_message.from_user.username.lower() == me.lower():
                addressed_to_us = True

        if not (is_private or from_source or addressed_to_us):
            return

        # Only buffer messages that look like part of an application.
        # Anything else (random chit-chat in DM, unrelated text in group)
        # is silently ignored to avoid wasting Claude calls on it.
        looks_like_app_part = bool(
            msg.photo
            or msg.media_group_id
            or find_pack_url(message_text_or_caption(msg))
            or from_source
        )
        if not looks_like_app_part:
            return

        await self._buffer_application(msg, ctx)

    # ---------- per-chat application debouncer ----------

    async def _buffer_application(self, msg: Message, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Buffer all application-shaped messages from a single chat that
        arrive within `app_debounce_s` of each other, then review the
        whole batch as ONE application. This is what stitches the photo
        media group together with the separate text-with-pack-link
        message that @sticker_bot sends a moment later."""
        chat_id = msg.chat_id
        buf = self.app_buffers.get(chat_id)
        if buf is None:
            buf = ApplicationBuffer()
            self.app_buffers[chat_id] = buf
        if msg.message_id in buf.seen_ids:
            return  # already buffered (defensive against duplicate updates)
        buf.seen_ids.add(msg.message_id)
        buf.messages.append(msg)
        if buf.task and not buf.task.done():
            buf.task.cancel()
        buf.task = asyncio.create_task(self._fire_application(chat_id, ctx))

    async def _fire_application(self, chat_id: int, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await asyncio.sleep(self.app_debounce_s)
        except asyncio.CancelledError:
            return
        buf = self.app_buffers.pop(chat_id, None)
        if not buf or not buf.messages:
            return
        msgs = sorted(buf.messages, key=lambda m: m.message_id)
        await self._review_application(ctx, msgs)

    # ---------- core: run the review ----------

    async def _review_application(
        self,
        ctx: ContextTypes.DEFAULT_TYPE,
        msgs: list[Message],
    ) -> None:
        anchor = msgs[0]  # reply target
        chat_id = anchor.chat_id

        # Aggregate text and pack URL across all messages in the group.
        full_text = "\n".join(message_text_or_caption(m) for m in msgs).strip()
        # Some messages put the URL only in entities (text_link / url) rather
        # than the visible text. Sweep entities/caption_entities of every
        # buffered message so we don't miss a button-style link.
        pack_url = find_pack_url(full_text) or _find_pack_url_in_entities(msgs) or ""

        log.info(
            "review: msgs=%d photos=%d text_len=%d pack_url=%r",
            len(msgs),
            sum(1 for m in msgs if m.photo),
            len(full_text),
            pack_url,
        )
        if not pack_url and full_text:
            log.info("review: no pack URL parsed; text head=%r", full_text[:300])

        # Post a placeholder so the user knows we're working. We'll
        # delete it right before posting the real verdict. do_quote=True
        # makes Telegram show the visible "↳ replying to" indicator even
        # in private chats, so it's obvious which application this is for.
        checking_msg: Message | None = None
        try:
            checking_msg = await anchor.reply_text(
                "🔍 Checking application…",
                disable_notification=True,
                do_quote=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("could not send 'Checking…' placeholder: %s", e)
        try:
            await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:  # noqa: BLE001
            pass

        # Download the in-message photos to use as the marketing images.
        local_marketing = await self._download_message_photos(ctx, msgs)

        # If the message contains a pack URL, also pull individual sticker
        # thumbnails via the pipeline bot's token. This is what the offline
        # pipeline does and what the prompt is calibrated against.
        sticker_paths: list[Path] = []
        sticker_meta: dict = {}
        if pack_url:
            slug = slug_from_url(pack_url)
            if slug:
                try:
                    sticker_paths, sticker_meta = await asyncio.to_thread(
                        ingest_pack,
                        slug,
                        self.bot_cfg.pipeline_token,
                        self.bot_cfg.tg_cache_dir,
                        self.model_cfg.max_stickers_per_pack,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("ingest_pack failed for %s: %s", slug, e)
                    sticker_meta = {"error": str(e)}

        title = sticker_meta.get("title") or _first_line(full_text) or "(none)"
        description = full_text
        sticker_count = sticker_meta.get("count", 0)

        # Compose the prompt with operator amendments appended.
        amendments_block = self.rules.amendments_block()
        # We need to call Claude with the prompt but including the amendments.
        # _claude_one_call uses CLAUDE_PROMPT directly — patch it via a
        # lightweight wrapper that monkey-substitutes the prompt for this
        # call only.
        verdict = await asyncio.to_thread(
            self._claude_review_with_amendments,
            title,
            description,
            local_marketing,
            sticker_paths,
            amendments_block,
        )

        # Sticker-count hard rules (mirror the pipeline).
        if pack_url and sticker_meta.get("error"):
            verdict = {
                "risk_category": "RED",
                "risk_score": 100,
                "reasoning": f"Cannot read sticker pack via Telegram ({sticker_meta['error']}). Pack must be verifiable to publish.",
                "ip_concerns": [], "nsfw_concerns": [], "pii_concerns": [],
                "scam_concerns": [f"unverifiable: {sticker_meta['error']}"],
                "error": None,
            }
        elif pack_url and sticker_count and sticker_count < self.model_cfg.min_stickers:
            verdict = {
                "risk_category": "RED",
                "risk_score": 100,
                "reasoning": f"Pack contains only {sticker_count} sticker(s); minimum is {self.model_cfg.min_stickers}.",
                "ip_concerns": [], "nsfw_concerns": [], "pii_concerns": [],
                "scam_concerns": [f"min_stickers: {sticker_count} < {self.model_cfg.min_stickers}"],
                "error": None,
            }
        elif pack_url and sticker_count and sticker_count > self.model_cfg.max_stickers:
            verdict = {
                "risk_category": "RED",
                "risk_score": 100,
                "reasoning": f"Pack contains {sticker_count} sticker(s); maximum is {self.model_cfg.max_stickers}.",
                "ip_concerns": [], "nsfw_concerns": [], "pii_concerns": [],
                "scam_concerns": [f"max_stickers: {sticker_count} > {self.model_cfg.max_stickers}"],
                "error": None,
            }

        verdict_text = self._format_verdict(verdict, sticker_meta, pack_url)
        # Delete the "Checking…" placeholder before posting the verdict so
        # the chat has just one final message per application.
        if checking_msg is not None:
            try:
                await checking_msg.delete()
            except Exception as e:  # noqa: BLE001
                log.warning("could not delete 'Checking…' placeholder: %s", e)
        sent = await _safe_reply(
            anchor,
            verdict_text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
            do_quote=True,
        )

        self.reviews_served += 1
        # Index this verdict so a reply can be recognized as a correction.
        self.verdict_index[sent.message_id] = {
            "original_message_id": anchor.message_id,
            "pack_url": pack_url,
            "title": title,
            "description": description,
            "verdict": verdict,
            "sticker_meta": sticker_meta,
            "ts": time.time(),
        }
        # Trim the verdict index so it doesn't grow forever.
        if len(self.verdict_index) > 500:
            for old_id in sorted(self.verdict_index.keys())[:100]:
                self.verdict_index.pop(old_id, None)

    # ---------- Claude wrapper that injects amendments ----------

    def _claude_review_with_amendments(
        self,
        title: str,
        description: str,
        marketing_images: list[Path],
        sticker_paths: list[Path],
        amendments_block: str,
    ) -> dict:
        """Run Claude with the live prompt = base + active amendments.

        We monkey-patch moderate_packs.CLAUDE_PROMPT briefly because
        _claude_one_call references it by name; this keeps us in lockstep
        with the offline pipeline's exact behavior.
        """
        import moderate_packs as mp

        original = mp.CLAUDE_PROMPT
        try:
            mp.CLAUDE_PROMPT = original + (amendments_block or "")
            return mp.claude_pack_review(
                title=title,
                description=description,
                image_urls=[],
                local_images=marketing_images + sticker_paths,
                cfg=self.model_cfg,
                client=self.claude,
            )
        finally:
            mp.CLAUDE_PROMPT = original

    # ---------- photo download ----------

    async def _download_message_photos(
        self,
        ctx: ContextTypes.DEFAULT_TYPE,
        msgs: list[Message],
    ) -> list[Path]:
        out: list[Path] = []
        cache_dir = self.bot_cfg.tg_cache_dir / "_inmsg"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for m in msgs:
            if not m.photo:
                continue
            ph = m.photo[-1]  # highest-resolution
            try:
                tg_file = await ctx.bot.get_file(ph.file_id)
                dest = cache_dir / f"{m.chat_id}_{m.message_id}_{ph.file_unique_id}.jpg"
                if not dest.exists():
                    await tg_file.download_to_drive(custom_path=str(dest))
                out.append(dest)
            except Exception as e:  # noqa: BLE001
                log.warning("photo download failed: %s", e)
        return out

    # ---------- verdict formatting ----------

    @staticmethod
    def _format_verdict(verdict: dict, sticker_meta: dict, pack_url: str) -> str:
        cat = (verdict.get("risk_category") or "RED").upper()
        if cat == "GREEN":
            head = "✅ *APPROVE*"
        elif cat == "YELLOW":
            head = "🟡 *NEEDS REVIEW*"
        else:
            head = "❌ *REJECT*"

        score = verdict.get("risk_score")
        score_part = f" — risk {score}/100" if isinstance(score, int) else ""

        # Pack info line: title · sticker count · type · clickable slug.
        info_bits: list[str] = []
        if sticker_meta.get("title"):
            info_bits.append(f"*{_md_escape(sticker_meta['title'])}*")
        count = sticker_meta.get("count")
        if isinstance(count, int) and count > 0:
            info_bits.append(f"{count} sticker{'s' if count != 1 else ''}")
        if sticker_meta.get("is_video_pack"):
            info_bits.append("video")
        elif sticker_meta.get("is_animated_pack"):
            info_bits.append("animated")
        elif count:
            info_bits.append("static")
        # Show the slug as a clickable link to the pack so the human can
        # open it in one tap. Slug often contains underscores → escape.
        slug = slug_from_url(pack_url) if pack_url else ""
        if slug:
            info_bits.append(f"[{_md_escape(slug)}]({pack_url})")
        elif pack_url:
            info_bits.append(pack_url)
        info_line = "📦 " + " · ".join(info_bits) if info_bits else ""

        # Claude's reasoning + concerns are freeform text and routinely
        # contain *, _, `, [, ] which trip Telegram's Markdown parser
        # ("Can't parse entities" -> the message gets rejected entirely).
        # Escape them defensively before interpolating.
        reason = _md_escape((verdict.get("reasoning") or "").strip())
        if not reason:
            reason = "(no reason supplied)"

        concerns: list[str] = []
        for k in ("ip_concerns", "nsfw_concerns", "pii_concerns", "scam_concerns"):
            for c in verdict.get(k) or []:
                concerns.append(_md_escape(str(c)))
        concern_line = ""
        if concerns:
            concern_line = "\n_Flags:_ " + ", ".join(concerns[:6])

        # Errors that didn't escalate to a hard reject (e.g. a partial
        # Claude batch failure) — surface them so the human knows.
        err = verdict.get("error")
        err_line = f"\n_Notes:_ {_md_escape(str(err))}" if err else ""

        parts = [f"{head}{score_part}"]
        if info_line:
            parts.append(info_line)
        parts.append(reason)
        if concern_line:
            parts.append(concern_line.lstrip("\n"))
        if err_line:
            parts.append(err_line.lstrip("\n"))
        return "\n".join(parts)

    # ---------- /addrule ingestion via Claude ----------

    async def _ingest_proposed_rule(self, msg: Message, raw_text: str) -> None:
        """Send the raw text to Claude, get a clean rule + category, store it."""
        # Show typing while Claude analyzes.
        try:
            await msg.get_bot().send_chat_action(chat_id=msg.chat_id, action="typing")
        except Exception:  # noqa: BLE001
            pass

        result = await asyncio.to_thread(self._claude_propose_rule, raw_text)
        if result.get("decision") == "skip":
            await msg.reply_text(
                f"🤔 I didn't add a rule. {result.get('reason', '').strip()}",
            )
            return

        text = (result.get("text") or "").strip()
        category = (result.get("category") or "NOTE").upper()
        action = result.get("action", "add")

        if action == "amend" and result.get("amend_id"):
            existing_id = int(result["amend_id"])
            updated = self.rules.update(existing_id, text=text, category=category)
            if updated:
                await msg.reply_text(
                    f"♻️ Amended rule `#{updated.id}` ({updated.category}):\n{updated.text}",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

        amendment = self.rules.add(
            text=text,
            category=category,
            source="addrule",
            context=raw_text[:200],
        )
        await msg.reply_text(
            f"✅ Added rule `#{amendment.id}` ({amendment.category}):\n{amendment.text}\n\n"
            f"_{result.get('reason', '').strip()}_",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ---------- disagreement learning ----------

    async def _handle_disagreement(self, msg: Message) -> None:
        bot_msg = msg.reply_to_message
        record = self.verdict_index.get(bot_msg.message_id)
        if not record:
            return
        user = msg.from_user
        if user and not self.is_owner(user.id):
            await msg.reply_text(
                "Noted, but only the owner can change the ruleset.",
            )
            return

        user_text = (msg.text or msg.caption or "").strip()
        if not user_text:
            return

        try:
            await msg.get_bot().send_chat_action(chat_id=msg.chat_id, action="typing")
        except Exception:  # noqa: BLE001
            pass

        result = await asyncio.to_thread(self._claude_learn_from_correction, record, user_text)

        decision = result.get("decision", "noop")
        if decision == "noop":
            await msg.reply_text(
                "👌 Got it — recorded your view. "
                f"I won't change the ruleset because: _{result.get('reason', '').strip()}_",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        text = (result.get("text") or "").strip()
        category = (result.get("category") or "NOTE").upper()

        if decision == "amend" and result.get("amend_id"):
            existing_id = int(result["amend_id"])
            updated = self.rules.update(existing_id, text=text, category=category)
            if updated:
                await msg.reply_text(
                    f"♻️ Learned: amended rule `#{updated.id}` ({updated.category}).\n"
                    f"{updated.text}\n\n_{result.get('reason', '').strip()}_",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return
        if decision == "remove" and result.get("amend_id"):
            existing_id = int(result["amend_id"])
            removed = self.rules.deactivate(existing_id)
            if removed:
                await msg.reply_text(
                    f"🗑 Learned: removed rule `#{removed.id}` ({removed.category}): {removed.text}\n"
                    f"_{result.get('reason', '').strip()}_",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

        # default: add
        amendment = self.rules.add(
            text=text,
            category=category,
            source="learned",
            context=record.get("pack_url", "")[:200],
        )
        await msg.reply_text(
            f"✅ Learned: added rule `#{amendment.id}` ({amendment.category}).\n{amendment.text}\n\n"
            f"_{result.get('reason', '').strip()}_",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ---------- Claude helpers (rule synthesis / learning) ----------

    def _claude_propose_rule(self, raw_text: str) -> dict:
        """Ask Claude to turn freeform user text into a structured amendment.

        Returns: {decision: add|amend|skip, action: add|amend, text, category, reason, amend_id?}
        """
        existing = self.rules.active()
        existing_dump = "\n".join(
            f"  #{a.id} [{a.category}] {a.text}" for a in existing
        ) or "  (none)"
        prompt = f"""You are the rule-curator for a sticker-pack moderation bot.
The bot already has a long base policy. Operators add AMENDMENTS on top.

Existing active amendments:
{existing_dump}

A moderator just proposed a new rule (freeform):
\"\"\"{raw_text}\"\"\"

Your job:
1. Decide if this is a real, actionable moderation rule.
   - If it's vague, redundant, contradictory, or off-topic, return decision=skip.
2. If it's actionable, decide:
   - action=amend  if it clearly refines/replaces ONE existing amendment
                   (provide amend_id of that amendment).
   - action=add    otherwise.
3. Pick a category: RED (auto-reject), YELLOW (needs review), GREEN (override
   to allow), or NOTE (general guidance).
4. Rewrite the rule text in the same crisp style as the base policy:
   one or two sentences, concrete, with examples in parentheses if helpful.
   Do NOT include the category prefix — just the rule body.

Output ONLY a JSON object with these fields:
{{
  "decision": "add" | "amend" | "skip",
  "action": "add" | "amend",
  "amend_id": <int or null>,
  "category": "RED" | "YELLOW" | "GREEN" | "NOTE",
  "text": "<rewritten rule body, or empty if skip>",
  "reason": "<one sentence explaining your decision>"
}}"""
        return self._claude_json_call(prompt)

    def _claude_learn_from_correction(self, record: dict, user_text: str) -> dict:
        """Decide whether a human disagreement should change the ruleset.

        Returns: {decision: add|amend|remove|noop, text, category, reason, amend_id?}
        """
        existing = self.rules.active()
        existing_dump = "\n".join(
            f"  #{a.id} [{a.category}] {a.text}" for a in existing
        ) or "  (none)"
        verdict = record.get("verdict") or {}
        prompt = f"""You are the rule-curator for a sticker-pack moderation bot.
A human moderator disagreed with one of the bot's verdicts. Decide whether
this correction reflects a generalizable rule change or just a one-off
judgment call.

Pack title: {record.get('title') or '(none)'}
Pack URL: {record.get('pack_url') or '(none)'}
Pack description (first 500 chars): {(record.get('description') or '')[:500]}
Sticker count: {record.get('sticker_meta', {}).get('count', 'unknown')}

Bot's verdict:
  category: {verdict.get('risk_category')}
  risk_score: {verdict.get('risk_score')}
  reasoning: {verdict.get('reasoning')}
  ip_concerns: {verdict.get('ip_concerns')}
  nsfw_concerns: {verdict.get('nsfw_concerns')}
  pii_concerns: {verdict.get('pii_concerns')}
  scam_concerns: {verdict.get('scam_concerns')}

Human correction (raw): \"\"\"{user_text}\"\"\"

Existing active amendments:
{existing_dump}

Decide:
- decision=noop   : one-off disagreement, no rule change warranted.
                    (E.g. the human just said "nah, looks fine" without
                    articulating a generalizable principle.)
- decision=add    : add a NEW amendment. Use this when the correction
                    introduces a principle not yet captured.
- decision=amend  : refine ONE existing amendment (provide amend_id).
- decision=remove : the correction implies an existing amendment is
                    wrong — deactivate it (provide amend_id).

If you DO change rules, write the rule body in the crisp style of the
base policy: one or two sentences, concrete, with a parenthetical example
when useful. Pick a category: RED (auto-reject), YELLOW (review), GREEN
(override-to-allow), NOTE (general guidance).

Be conservative: if in doubt, return noop. Real-world moderation
disagreements often reflect taste, not policy.

Output ONLY a JSON object:
{{
  "decision": "noop" | "add" | "amend" | "remove",
  "amend_id": <int or null>,
  "category": "RED" | "YELLOW" | "GREEN" | "NOTE",
  "text": "<rule body, or empty if noop/remove>",
  "reason": "<one sentence>"
}}"""
        return self._claude_json_call(prompt)

    def _claude_json_call(self, prompt: str) -> dict:
        """Plain text-only Claude call expecting a JSON object back."""
        try:
            msg = self.claude.messages.create(
                model=self.model_cfg.claude_model,
                max_tokens=600,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                timeout=60.0,
            )
        except Exception as e:  # noqa: BLE001
            return {"decision": "noop", "reason": f"claude error: {e}"}
        raw = msg.content[0].text if msg.content else ""
        cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).replace("```", "")
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return {"decision": "noop", "reason": f"unparseable: {raw[:120]}"}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"decision": "noop", "reason": f"bad json: {raw[:120]}"}


# ---------- helpers ----------

def _start_health_server(port: int) -> None:
    """Tiny stdlib HTTP server in a daemon thread for deploy health checks.

    fly.io's "Deploy from GitHub" wizard adds a health check on the
    primary HTTP port even when fly.toml declares no http_service. The
    bot itself only speaks long-poll HTTPS *outbound*, so without this
    the deploy hangs at "timeout trying to get your app". Returns 200
    "ok" on any path; we don't expose any sensitive info.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def _send_ok(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"ok\n")

        def do_GET(self) -> None:  # noqa: N802
            self._send_ok()

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, fmt: str, *args) -> None:  # silence stderr noise
            return

    def _serve() -> None:
        try:
            srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
            log.info("health-check HTTP server listening on :%d", port)
            srv.serve_forever()
        except Exception as e:  # noqa: BLE001
            log.warning("health-check server died: %s", e)

    t = threading.Thread(target=_serve, name="health", daemon=True)
    t.start()


def _md_escape(s: str) -> str:
    """Escape characters that Telegram's legacy Markdown parser will eat.

    Underscores in usernames like @sticker_bot get turned into italic
    delimiters; lone `*` / `_` / `` ` `` / `[` in Claude's freeform
    reasoning trip "can't parse entities" and the whole message gets
    rejected. Order matters: backslash first so we don't double-escape
    our own escapes."""
    return (
        (s or "")
        .replace("\\", r"\\")
        .replace("_", r"\_")
        .replace("*", r"\*")
        .replace("`", r"\`")
        .replace("[", r"\[")
    )


def _strip_markdown(s: str) -> str:
    """Best-effort Markdown -> plain text for fallback delivery."""
    out = (s or "")
    out = re.sub(r"\\([_\\*`\[\]])", r"\1", out)  # un-escape
    out = re.sub(r"\*([^*\n]+)\*", r"\1", out)     # *bold*
    out = re.sub(r"_([^_\n]+)_", r"\1", out)       # _italic_
    out = re.sub(r"`([^`\n]+)`", r"\1", out)       # `code`
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", out)  # [t](u) -> t (u)
    return out


async def _safe_reply(
    msg: "Message",
    text: str,
    **kwargs,
) -> "Message | None":
    """Reply with Markdown; on parse failure, retry as plain text.

    Telegram returns BadRequest "Can't parse entities" if any
    Markdown delimiter is unbalanced — even one stray `*` in a long
    Claude reasoning string. The whole verdict is then dropped, which
    is much worse than losing the bold/italic. This wrapper guarantees
    delivery."""
    from telegram.error import BadRequest

    try:
        return await msg.reply_text(text, **kwargs)
    except BadRequest as e:
        if "parse entities" not in str(e).lower():
            raise
        log.warning("markdown parse failed, falling back to plain text: %s", e)
        kwargs.pop("parse_mode", None)
        return await msg.reply_text(_strip_markdown(text), **kwargs)


def _format_uptime(seconds: float) -> str:
    """Human-readable uptime: '3d 14h 22m', '2h 5m', '47s'."""
    s = int(max(seconds, 0))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, sec = divmod(r, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _first_line(s: str) -> str:
    for line in (s or "").splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return ""


def _chunk(s: str, n: int) -> list[str]:
    return [s[i : i + n] for i in range(0, len(s), n)] or [""]


# ---------- entrypoint ----------

def build_app(bot_cfg: BotConfig, model_cfg: Config) -> Application:
    bot = ReviewBot(bot_cfg, model_cfg)
    app = ApplicationBuilder().token(bot_cfg.review_token).build()

    app.add_handler(CommandHandler(["start", "help"], bot.cmd_start))
    app.add_handler(CommandHandler("rules", bot.cmd_rules))
    app.add_handler(CommandHandler("addrule", bot.cmd_addrule))
    app.add_handler(CommandHandler("delrule", bot.cmd_delrule))
    app.add_handler(CommandHandler("listrules", bot.cmd_listrules))
    app.add_handler(CommandHandler("chats", bot.cmd_chats))
    app.add_handler(CommandHandler("leavechat", bot.cmd_leavechat))
    app.add_handler(CommandHandler("admin", bot.cmd_admin))
    # Bot-membership changes (so we can auto-whitelist / auto-leave).
    app.add_handler(
        ChatMemberHandler(bot.on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER)
    )
    # Catch-all for non-command messages (text, captions, photos).
    app.add_handler(
        MessageHandler(
            (filters.TEXT | filters.CAPTION | filters.PHOTO) & ~filters.COMMAND,
            bot.on_message,
        )
    )
    return app


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )
    bot_cfg = BotConfig.from_env()
    model_cfg = Config.from_env()
    # Fly.io / Railway / Heroku expose $PORT and expect us to bind to it.
    # Locally there's no PORT and we skip the health server.
    port_env = os.getenv("PORT", "").strip()
    if port_env.isdigit():
        _start_health_server(int(port_env))
    app = build_app(bot_cfg, model_cfg)
    log.info(
        "review bot starting; source=@%s; owner=%s; allowlist_seed=%s; allowlist_path=%s",
        bot_cfg.source_bot_username,
        bot_cfg.owner_user_id,
        sorted(bot_cfg.seed_allowed_chat_ids) or "(none)",
        bot_cfg.allowlist_path,
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
