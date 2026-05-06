"""
Rules store for the live review bot.

The base ruleset lives in moderate_packs.CLAUDE_PROMPT (long, well-tuned).
On top of that, operators can add AMENDMENTS via /addrule or have them
auto-proposed when a human disagrees with the bot's verdict. Amendments
are appended to the prompt under a clearly-labeled section so Claude
can apply them on top of the base policy.

Persistence: a single JSON file (.rules_store.json) with a list of
amendments. Each amendment has:
  id          stable int
  text        rule body, in the same voice as CLAUDE_PROMPT
  category    one of: RED, YELLOW, GREEN, NOTE
  source      "addrule" | "learned"
  created_at  unix ts
  context     optional: pack id / url that triggered a learned rule
  active      bool (soft-delete via /delrule)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import os as _os

DEFAULT_PATH = Path(_os.getenv("RULES_STORE_PATH", ".rules_store.json"))


@dataclass
class Amendment:
    id: int
    text: str
    category: str = "NOTE"  # RED | YELLOW | GREEN | NOTE
    source: str = "addrule"  # addrule | learned
    created_at: float = field(default_factory=time.time)
    context: str = ""
    active: bool = True


class RulesStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._amendments: list[Amendment] = []
        self._next_id = 1
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._amendments = [Amendment(**a) for a in data.get("amendments", [])]
        self._next_id = data.get("next_id", max((a.id for a in self._amendments), default=0) + 1)

    def _save(self) -> None:
        data = {
            "next_id": self._next_id,
            "amendments": [asdict(a) for a in self._amendments],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---------- mutations ----------

    def add(
        self,
        text: str,
        category: str = "NOTE",
        source: str = "addrule",
        context: str = "",
    ) -> Amendment:
        text = text.strip()
        if not text:
            raise ValueError("rule text is empty")
        category = category.upper()
        if category not in {"RED", "YELLOW", "GREEN", "NOTE"}:
            category = "NOTE"
        with self._lock:
            a = Amendment(
                id=self._next_id,
                text=text,
                category=category,
                source=source,
                context=context,
            )
            self._next_id += 1
            self._amendments.append(a)
            self._save()
        return a

    def update(self, rule_id: int, *, text: str | None = None, category: str | None = None) -> Amendment | None:
        with self._lock:
            for a in self._amendments:
                if a.id == rule_id and a.active:
                    if text is not None:
                        a.text = text.strip()
                    if category is not None:
                        category = category.upper()
                        if category in {"RED", "YELLOW", "GREEN", "NOTE"}:
                            a.category = category
                    self._save()
                    return a
        return None

    def deactivate(self, rule_id: int) -> Amendment | None:
        with self._lock:
            for a in self._amendments:
                if a.id == rule_id and a.active:
                    a.active = False
                    self._save()
                    return a
        return None

    # ---------- accessors ----------

    def active(self) -> list[Amendment]:
        with self._lock:
            return [a for a in self._amendments if a.active]

    def all(self) -> list[Amendment]:
        with self._lock:
            return list(self._amendments)

    def get(self, rule_id: int) -> Amendment | None:
        with self._lock:
            for a in self._amendments:
                if a.id == rule_id:
                    return a
        return None

    # ---------- prompt composition ----------

    def amendments_block(self) -> str:
        """Render active amendments as an extra prompt section.

        We append this to CLAUDE_PROMPT so Claude treats them as
        operator-supplied overrides on top of the base policy.
        """
        items = self.active()
        if not items:
            return ""
        lines = [
            "",
            "==================================================================",
            "OPERATOR AMENDMENTS (apply ON TOP of the base policy above)",
            "==================================================================",
            "These were added by human moderators to clarify or override the",
            "base rules. They take precedence over the defaults when in conflict.",
            "",
        ]
        # Group by category for readability.
        for cat in ("RED", "YELLOW", "GREEN", "NOTE"):
            cat_items = [a for a in items if a.category == cat]
            if not cat_items:
                continue
            lines.append(f"-- {cat} amendments --")
            for a in cat_items:
                lines.append(f"  [#{a.id}] {a.text}")
            lines.append("")
        return "\n".join(lines)

    # ---------- /rules summary for chat display ----------

    def rules_summary(self, source_bot: str = "sticker_bot") -> str:
        """Concise human summary of the moderation policy. Plain text — no
        Markdown — so it always renders identically across Telegram clients
        and Telethon's parser can't choke on it."""
        base = (
            "Sticker Pad — moderation policy (summary)\n"
            "\n"
            f"I review applications posted by @{source_bot} and suggest "
            "✅ APPROVE or ❌ REJECT with a one-line reason. Default is "
            "GREEN — I only carve out high-risk cases.\n"
            "\n"
            "🔴 RED (auto-reject)\n"
            "• Marketplace impersonation — pack logo/cover dominated by Sticker Pad branding, "
            "or a fake \"Tonkeeper / Binance Official\" pack.\n"
            "• Default Telegram or VK packs reused 1:1 (Great Minds, Animated Emojies, etc.).\n"
            "• NFT-collection mass derivatives without a specific token# (BAYC, Pudgy, CryptoPunks, Doodles, Azuki, etc.).\n"
            "• Active-conflict / war imagery (Russia–Ukraine, Israel–Palestine, etc. with combat/uniforms/Z-marks).\n"
            "• Telegram Gifts traced/imitated in their specific 3D-collectible style (Plush Pepe Gift, Durov's Cap, etc.).\n"
            "• Big-IP named characters as primary content (Disney, Marvel, Nintendo, Sanrio, Pokémon, popular anime).\n"
            "• Hard NSFW: explicit visible genitalia/sex acts, ANY sexualization of minors, real-photo gore.\n"
            "• PII / impersonation / doxing of private individuals; sexual deepfakes of public figures.\n"
            "• Pack core identity is a literal scam claim or fake giveaway/airdrop.\n"
            "• Pack inaccessible via Telegram, or sticker count out of range (min 4, max 30).\n"
            "\n"
            "🟡 YELLOW (human review)\n"
            "• Specific named-character riff that might be transformative.\n"
            "• Real public figure used semi-realistically and non-satirically.\n"
            "• 1–2 borderline stickers in an otherwise clean set.\n"
            "• NFT pack claiming a specific token# — needs ownership check.\n"
            "\n"
            "🟢 GREEN (auto-publish, the default)\n"
            "• Pepe / Wojak / Doge / Cheems / classic memes.\n"
            "• Crypto-native culture (gm, wagmi, hodl, TON / Notcoin themes, MAKE TON GREAT AGAIN).\n"
            "• Caricatures / parody of public figures (Trump, Musk, Durov) without hardcore humiliation.\n"
            "• \"In the style of\" anime / pixel-art / Roblox-look ORIGINAL characters.\n"
            "• Wordmark parodies (Abibas, Stonks, Lega).\n"
            "• Generic archetypes (frogs, cats, hippos), original mascots.\n"
            "• Suggestive cartoon / lingerie WITHOUT visible genitalia or minors.\n"
            "• Edgy / vulgar text (any language) that isn't hate-speech, doxing, or terror calls."
        )
        amendments = self.active()
        footer = (
            "\n\nIf you disagree with a verdict, reply to my message with the correct "
            "decision and a one-line reason — I'll consider amending the rules."
        )
        if amendments:
            lines = ["\n\nRules from replies:"]
            for a in amendments:
                lines.append(f"• {a.text}")
            return base + "\n".join(lines) + footer
        return base + "\n\nNo extra rules from replies yet." + footer


# ---------- module-level singleton convenience ----------

_default_store: RulesStore | None = None


def get_default_store(path: Path | None = None) -> RulesStore:
    global _default_store
    if _default_store is None:
        # Re-read env each call in case it changed since import time.
        target = path or Path(_os.getenv("RULES_STORE_PATH", ".rules_store.json"))
        _default_store = RulesStore(target)
    return _default_store
