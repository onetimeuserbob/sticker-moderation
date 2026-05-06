"""
Moderator assistant — Claude-powered Q&A and rule-management interface.

Used by the userbot (Telethon) to handle messages addressed to Moderator
that aren't sticker-pack applications themselves: chat questions about
rules, requests to add / remove / change amendments, status queries,
explanations of why a verdict was given, etc.

The same Claude model the moderation pipeline uses also drives this so
its understanding of the policy is consistent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from moderate_packs import CLAUDE_PROMPT

if TYPE_CHECKING:
    from review_bot import ReviewBot


log = logging.getLogger("assistant")


TOOLS = [
    {
        "name": "list_amendments",
        "description": (
            "List all currently active operator amendments to the moderation "
            "policy. Use this whenever the user asks 'what are the rules', "
            "'show me the amendments', 'what custom rules exist', etc."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_amendment",
        "description": (
            "Add a new operator amendment on top of the base policy. Use this "
            "when the user proposes a new moderation rule (e.g. 'don't approve "
            "packs with X', 'always flag Y'). The text should be a clear, "
            "concrete one-or-two-sentence rule. Pick category from RED "
            "(auto-reject), YELLOW (needs human review), GREEN (allow override), "
            "or NOTE (general guidance). Owner-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The rule body, crisp policy language."},
                "category": {
                    "type": "string",
                    "enum": ["RED", "YELLOW", "GREEN", "NOTE"],
                    "description": "Severity / kind of the amendment.",
                },
            },
            "required": ["text", "category"],
        },
    },
    {
        "name": "deactivate_amendment",
        "description": (
            "Deactivate (remove) an existing amendment by its numeric id. "
            "Use this when the user says 'remove rule N', 'delete amendment N', "
            "'forget that rule', etc. Owner-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amendment_id": {"type": "integer", "description": "Numeric id of the amendment to deactivate."},
            },
            "required": ["amendment_id"],
        },
    },
    {
        "name": "get_status",
        "description": (
            "Return runtime status: which Claude model is in use, how long "
            "the bot has been alive, how many reviews served, storage paths. "
            "Use when the user asks for status / uptime / health / version."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "explain_recent_verdict",
        "description": (
            "Look up a recent verdict by its original message id (or the most "
            "recent one in this chat) and return the recorded reasoning, "
            "concerns, and pack metadata. Use when the user asks 'why did you "
            "reject that?', 'explain your last call', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "integer"},
                "message_id": {"type": "integer", "description": "Optional. If omitted, returns most recent in this chat."},
            },
            "required": ["chat_id"],
        },
    },
    {
        "name": "leave_chat",
        "description": "Leave a chat by its numeric id. Owner-only.",
        "input_schema": {
            "type": "object",
            "properties": {"chat_id": {"type": "integer"}},
            "required": ["chat_id"],
        },
    },
]


class ModeratorAssistant:
    def __init__(self, review_bot: "ReviewBot", userbot):
        self.review_bot = review_bot
        self.userbot = userbot  # UserbotRelay; for leave_chat etc.

    # ------------- system prompt -------------

    def _system_prompt(self) -> list[dict]:
        amends = self.review_bot.rules.active()
        if amends:
            amend_dump = "\n".join(
                f"  Amendment #{a.id} [{a.category}] ({a.source}) {a.text}"
                for a in amends
            )
        else:
            amend_dump = "  (none yet)"
        # Two-segment system: cacheable BASE policy + per-call dynamic part.
        return [
            {
                "type": "text",
                "text": (
                    "You are **Moderator**, a friendly assistant for the Sticker "
                    "Pad moderation team. You are the user-facing voice of an "
                    "automated review pipeline. You help the team manage the "
                    "ruleset and answer questions about how applications are "
                    "evaluated. Style: concise, plain English, no hype, no "
                    "apologising. Use Telegram-flavour Markdown sparingly "
                    "(only `*bold*`, `_italics_`, backticks for code/ids; no "
                    "tables, no headers). When the user proposes a rule "
                    "change, USE THE TOOLS — don't just say you'll add it; "
                    "actually add it. After a tool call, briefly confirm what "
                    "you did.\n\n"
                    "## BASE MODERATION POLICY (reference)\n\n"
                    + CLAUDE_PROMPT
                ),
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    "## ACTIVE OPERATOR AMENDMENTS (numbered separately from "
                    "the base policy above)\n\n" + amend_dump
                ),
            },
        ]

    # ------------- tool runner -------------

    async def _run_tool(
        self, name: str, args: dict, *, is_owner: bool, chat_id: int
    ) -> str:
        if name == "list_amendments":
            amends = self.review_bot.rules.active()
            if not amends:
                return "(no active amendments)"
            return "\n".join(
                f"#{a.id} [{a.category}] {a.text}" for a in amends
            )

        if name == "add_amendment":
            if not is_owner:
                return "ERROR: only the owner can change rules."
            text = (args.get("text") or "").strip()
            category = (args.get("category") or "NOTE").upper()
            if not text:
                return "ERROR: empty text."
            amend = self.review_bot.rules.add(
                text=text, category=category, source="assistant", context=""
            )
            log.info("assistant added amendment #%s [%s]", amend.id, amend.category)
            return f"Added amendment #{amend.id} [{amend.category}]: {amend.text}"

        if name == "deactivate_amendment":
            if not is_owner:
                return "ERROR: only the owner can change rules."
            try:
                aid = int(args.get("amendment_id"))
            except Exception:  # noqa: BLE001
                return "ERROR: amendment_id must be an integer."
            removed = self.review_bot.rules.deactivate(aid)
            if not removed:
                return f"No active amendment #{aid}."
            log.info("assistant deactivated amendment #%s", removed.id)
            return f"Deactivated amendment #{removed.id} ({removed.category}): {removed.text}"

        if name == "get_status":
            from review_bot import _format_uptime
            up = _format_uptime(__import__("time").time() - self.review_bot.started_at)
            return (
                f"model: {self.review_bot.model_cfg.claude_model}\n"
                f"host: fly.io ({__import__('socket').gethostname()})\n"
                f"uptime: {up}\n"
                f"reviews served: {self.review_bot.reviews_served}\n"
                f"active amendments: {len(self.review_bot.rules.active())}\n"
                f"verdict_index entries: {len(self.review_bot.verdict_index)}\n"
                f"whitelisted chats: {len(self.review_bot.allowlist.all())}\n"
            )

        if name == "explain_recent_verdict":
            target_chat = int(args.get("chat_id") or chat_id)
            mid = args.get("message_id")
            entries = [
                (k, v) for k, v in self.review_bot.verdict_index.items()
                if v.get("chat_id") == target_chat
            ]
            if not entries:
                return f"No verdicts on record for chat {target_chat}."
            if mid is not None:
                match = next(
                    ((k, v) for k, v in entries if v.get("original_message_id") == mid),
                    None,
                )
                if match is None:
                    return f"No verdict for message {mid} in chat {target_chat}."
                _, rec = match
            else:
                _, rec = max(entries, key=lambda kv: kv[1].get("ts", 0))
            v = rec.get("verdict") or {}
            sm = rec.get("sticker_meta") or {}
            return (
                f"pack: {rec.get('title', '?')} ({rec.get('pack_url', '?')})\n"
                f"sticker_count: {sm.get('count', '?')}\n"
                f"verdict: {v.get('risk_category', '?')} risk={v.get('risk_score', '?')}\n"
                f"reasoning: {v.get('reasoning', '')}\n"
                f"flags: " + ", ".join(
                    str(c) for k in ("ip_concerns", "nsfw_concerns", "pii_concerns", "scam_concerns")
                    for c in (v.get(k) or [])
                )
            )

        if name == "leave_chat":
            if not is_owner:
                return "ERROR: only the owner can make me leave chats."
            try:
                cid = int(args.get("chat_id"))
            except Exception:  # noqa: BLE001
                return "ERROR: chat_id must be an integer."
            try:
                if self.userbot and self.userbot.client:
                    await self.userbot.client.delete_dialog(cid)
                self.review_bot.allowlist.remove(cid)
                log.info("assistant left chat %s", cid)
                return f"Left chat {cid} and removed from allowlist."
            except Exception as e:  # noqa: BLE001
                return f"Failed to leave: {e}"

        return f"ERROR: unknown tool {name!r}"

    # ------------- main entrypoint -------------

    async def respond(
        self,
        *,
        chat_id: int,
        sender_id: int,
        sender_name: str,
        is_owner: bool,
        user_text: str,
    ) -> str:
        sys_blocks = self._system_prompt()
        msgs: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"From: {sender_name} (telegram id {sender_id}; "
                    f"owner={is_owner}; chat_id={chat_id})\n\n"
                    f"Message: {user_text}"
                ),
            }
        ]
        for _ in range(6):  # bounded tool-loop
            try:
                resp = await asyncio.to_thread(
                    self.review_bot.claude.messages.create,
                    model=self.review_bot.model_cfg.claude_model,
                    max_tokens=1500,
                    system=sys_blocks,
                    tools=TOOLS,
                    messages=msgs,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("assistant claude call failed: %s", e)
                return f"Sorry, I errored talking to the model ({e})."

            blocks = list(resp.content)
            tool_uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                # final answer
                text = "".join(getattr(b, "text", "") for b in blocks).strip()
                return text or "(no reply)"

            # Run each tool, append assistant + tool_result turn.
            msgs.append({"role": "assistant", "content": [b.model_dump() for b in blocks]})
            tool_results: list[dict] = []
            for b in tool_uses:
                try:
                    out = await self._run_tool(
                        b.name, b.input, is_owner=is_owner, chat_id=chat_id
                    )
                except Exception as e:  # noqa: BLE001
                    log.exception("tool %s failed: %s", b.name, e)
                    out = f"ERROR: tool failed: {e}"
                log.info(
                    "assistant tool: %s args=%s -> %d chars",
                    b.name, json.dumps(b.input)[:200], len(out),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": out,
                })
            msgs.append({"role": "user", "content": tool_results})

        return "(I went around in too many tool calls and gave up. Try rephrasing?)"
