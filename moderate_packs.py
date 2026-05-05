"""
Sticker Pad — automated pre-publish moderation pipeline.

Goal: triage incoming sticker packs into GREEN (auto-publish) /
YELLOW (human review) / RED (auto-reject) so we can drop the manual
queue from "review every pack" down to "review only the YELLOW slice".

Stack (all self-serve, no sales calls):
  OpenAI omni-moderation-latest  → free, fast pre-filter for NSFW /
                                   hate / violence / self-harm /
                                   sexual-minors / illicit
  Claude Sonnet 4.5 vision        → IP / brand / celebrity / character
                                   recognition, PII detection,
                                   stylized-content NSFW backstop,
                                   and final risk verdict

Usage:
  cp .env.example .env   # fill ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY)
  pip install -r requirements.txt
  python moderate_packs.py --input packs.csv --output moderated.csv --eval

CSV defaults match stickerpad-launches.csv:
  id, name, description, logo_url, cover_url, [decision]
Override with --col-* flags for any other schema.

If a `decision` column exists ("approved"/"rejected"), pass --eval to
print a confusion matrix vs the auto-pipeline output.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from tg_ingest import ingest_pack, slug_from_url
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------- config ----------

@dataclass
class Config:
    anthropic_key: str = ""
    claude_model: str = "claude-sonnet-4-5"
    openai_key: str = ""
    telegram_bot_token: str = ""

    max_images_per_pack: int = 12
    max_stickers_per_pack: int = 55
    claude_batch_size: int = 24
    min_stickers: int = 4
    max_stickers: int = 55
    pack_workers: int = 4

    claude_red: float = 70
    claude_yellow: float = 35
    combined_red: float = 70
    combined_yellow: float = 35

    openai_hard_threshold: float = 0.85
    openai_soft_threshold: float = 0.5

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            max_images_per_pack=int(os.getenv("MAX_IMAGES_PER_PACK", 12)),
            max_stickers_per_pack=int(os.getenv("MAX_STICKERS_PER_PACK", 55)),
            claude_batch_size=int(os.getenv("CLAUDE_BATCH_SIZE", 24)),
            min_stickers=int(os.getenv("MIN_STICKERS", 4)),
            max_stickers=int(os.getenv("MAX_STICKERS", 55)),
            pack_workers=int(os.getenv("PACK_WORKERS", 4)),
            claude_red=float(os.getenv("CLAUDE_RED_THRESHOLD", 70)),
            claude_yellow=float(os.getenv("CLAUDE_YELLOW_THRESHOLD", 35)),
            combined_red=float(os.getenv("COMBINED_RED", 70)),
            combined_yellow=float(os.getenv("COMBINED_YELLOW", 35)),
            openai_hard_threshold=float(os.getenv("OPENAI_HARD_THRESHOLD", 0.85)),
            openai_soft_threshold=float(os.getenv("OPENAI_SOFT_THRESHOLD", 0.5)),
        )


# ---------- OpenAI moderation pre-filter ----------

# Categories from omni-moderation-latest that should trigger a hard RED
# on their own.
OPENAI_HARD_CATEGORIES = {
    "sexual/minors",
    "self-harm/intent",
    "self-harm/instructions",
    "violence/graphic",
    "hate/threatening",
}

# Categories where we want soft attention (push to YELLOW, not RED).
OPENAI_SOFT_CATEGORIES = {
    "sexual",
    "violence",
    "harassment",
    "harassment/threatening",
    "hate",
    "self-harm",
    "illicit",
    "illicit/violent",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    reraise=True,
)
def openai_moderate(
    text: str,
    image_urls: list[str],
    cfg: Config,
    client: "OpenAI | None",
) -> dict:
    """One omni-moderation call covers text + multiple image URLs.

    Returns a normalized dict; never raises.
    """
    if client is None:
        return {
            "max_score": 0.0,
            "hard_flags": [],
            "soft_flags": [],
            "categories": {},
            "error": "no_client",
        }

    inputs: list[dict[str, Any]] = []
    if text.strip():
        inputs.append({"type": "text", "text": text[:4000]})
    for u in image_urls[: cfg.max_images_per_pack]:
        inputs.append({"type": "image_url", "image_url": {"url": u}})

    if not inputs:
        return {"max_score": 0.0, "hard_flags": [], "soft_flags": [], "categories": {}, "error": None}

    try:
        resp = client.moderations.create(
            model="omni-moderation-latest",
            input=inputs,
        )
    except Exception as e:  # noqa: BLE001
        return {"max_score": 0.0, "hard_flags": [], "soft_flags": [], "categories": {}, "error": str(e)}

    # Aggregate worst-case score per category across all input items.
    worst: dict[str, float] = {}
    for r in resp.results:
        for cat, score in r.category_scores.model_dump().items():
            if score is None:
                continue
            worst[cat] = max(worst.get(cat, 0.0), float(score))

    hard = [
        c for c, s in worst.items()
        if c in OPENAI_HARD_CATEGORIES and s >= cfg.openai_hard_threshold
    ] + [
        c for c, s in worst.items()
        if c not in OPENAI_HARD_CATEGORIES and s >= cfg.openai_hard_threshold
    ]
    soft = [
        c for c, s in worst.items()
        if c not in hard and c in OPENAI_SOFT_CATEGORIES and s >= cfg.openai_soft_threshold
    ]
    max_score = max(worst.values()) if worst else 0.0
    return {
        "max_score": max_score,
        "hard_flags": sorted(set(hard)),
        "soft_flags": sorted(set(soft)),
        "categories": {k: round(v, 3) for k, v in worst.items()},
        "error": None,
    }


# ---------- Claude review ----------

CLAUDE_PROMPT = """You are a pre-publish moderator for **Sticker Pad**,
a Telegram sticker marketplace that mints best-selling packs as NFTs.
Our ToS already disclaims responsibility — we moderate ONLY to avoid
catastrophic risk, NOT to enforce IP law on behalf of rights-holders.

We default to GREEN. The lists below are CARVE-OUTS from green, not a
checklist that must pass.

Pack metadata:
  Title: {title}
  Description: {description}

What we actually fear (and only these):
  1. Cease-and-desist letters from sue-happy IP holders
  2. "Crypto site sells [horrible thing]" press / PR damage
  3. Buyers defrauded by impersonation or fake endorsements
  4. Real-people legal exposure (defamation, doxing, NCII)

What we DO NOT fear:
  - Generic IP theory ("technically that frog has a copyright")
  - Style imitation (anime-style, Roblox-look, pixel-art, low-poly)
  - Cultural-commons memes
  - Parody, satire, transformative use of public figures

==================================================================
RED (auto-reject)
==================================================================

**Marketplace impersonation** — the marketplace is Sticker Pad /
Stickerdom. A pack whose own logo OR cover looks like Sticker Pad's
own branding causes buyer confusion at sale time. RED ONLY when the
logo OR cover image is DOMINATED (>40% of visible area) by:
  • **Sticker Pad / Stickerdom** — white smiley/moon-face icon
    (a circle with a slice cut out and two dot eyes + curved smile),
    typically on the platform's purple/blue gradient
  • **Direct impersonation** of a real wallet / exchange / project,
    framed to look like that brand's official sticker pack —
    e.g. fake "Tonkeeper Official", fake "Binance Stickers"

Themes celebrating Telegram, TON / Toncoin, Notcoin, Wallet /
Tonkeeper / MyTonWallet, or generic crypto exchanges (BTC/ETH/SOL
logos, Bitcoin charts, etc.) are **GREEN** — these are normal
crypto-culture themes for this marketplace, not impersonation.
Examples that are GREEN:
  • "MAKE TON GREAT AGAIN" caps, TON-rocket-to-moon imagery
  • Pack with Telegram paper-plane as the cover theme
  • Notcoin gold-coin pack themes
  • Crypto-trader memes with $TON / $BTC tickers and price charts
  • Telegram-themed mascots / Durov-themed packs (cartoon)

**Default Telegram bundled sticker packs** — RED ONLY when the pack
directly reuses art from one of Telegram's well-known DEFAULT packs
(the free packs Telegram ships with the app). Be conservative — only
flag when you're confident in the match. VK default packs are NOT
in scope. Known Telegram defaults to recognize (use these as anchors,
don't extrapolate):
  • **Great Minds** — drawn-portrait pack of Einstein, Newton,
    Tesla, Da Vinci, Marie Curie, etc., specific muted-palette
    illustrated-portrait style with Russian/English captions
  • **Cherry Pinky** — bright-pink/red anime-style schoolgirl-ish
    character, smiling expressions, kawaii
  • **Animated Emojies / Big Emoji** — TG's bundled emoji packs
  • Any TG-Premium pre-bundled pack reused 1:1

A pack with similar art *style* but original characters is GREEN —
only literal direct copies / asset reuse trip this. If you can't
clearly identify the pack as one of the above defaults, lean GREEN.
Examples:
  • Generic pink-haired anime girl pack → GREEN (NOT Cherry Pinky)
  • "Spotty: Degen Edition" with original spotty-dog characters
    → GREEN (not the actual Spotty default)
  • "Great Minds Classic" with the SAME Einstein/Newton portraits
    used in TG's pack → RED (direct reuse)

**NFT collection IP** — Top NFT collections (think OpenSea all-time
top ~50) with their own original IP. The risk is that sticker-pack
derivatives compete with or dilute the collection's own art rights.
Established collections to recognize:
  • **Bored Ape Yacht Club (BAYC) / MAYC** — colorful apes, that
    specific Yuga-Labs illustrated style
  • **Pudgy Penguins** — round cute cartoon penguins, simple
    line-work, blue/white/pink palette, often hats/accessories
  • **Sappy Seals** — pixelated cute cartoon seals, Pudgy-Labs IP
  • **CryptoPunks** — 8-bit pixel portrait avatars, 24×24
  • **Doodles** — pastel cartoon characters
  • **Azuki** — anime-style red-themed characters
  • **Moonbirds, Goblintown, Cool Cats, World of Women, Milady,
    DeGods, Lazy Lions, CloneX, VeeFriends, Mfers** — recognizable
    styles
  • Other top-50 OpenSea originals with proprietary IP

Verdict logic:
  • Pack title claims a SPECIFIC token number ("BAYC #8401",
    "Pudgy #1234", "CryptoPunk #5678") → **YELLOW** (human verifies
    ownership; many collections allow holders to make derivatives
    following the collection's policy)
  • Pack uses the collection's distinctive style across many stickers
    WITHOUT a specific token# claim ("Cool Blue Pengu", "Pudgy
    Friends", "Sappy Seals memes") → **RED**
  • A single small reference among original art → GREEN

**Low-confidence calibration — be conservative on IP claims.** Don't
flag based on superficial similarity. Generic categories of art look
similar to many collections:
  • A generic cute penguin ≠ Pudgy Penguin (the Pudgy line-work is
    very specific: thick black outlines, simple white belly, basic
    geometric shapes, 3-5 frame palette). Generic penguin → GREEN.
  • A generic cartoon ape ≠ BAYC (BAYC has specific Yuga textures,
    fur variants, and a "lazy bored" facial structure). "Apes Club"
    or generic monkey art → GREEN.
  • A generic anime-styled character ≠ Azuki, Hatsune Miku, etc.
  • A 3D plush frog ≠ Plush Pepe Telegram Gift unless it matches
    the TG Gift's specific render (see Telegram Gifts section).
If you cannot positively identify the specific collection's signature
visual markers, do NOT flag for that IP. Lean GREEN on uncertainty.

==================================================================
ACTIVE-CONFLICT POLITICS / WAR (RED)
==================================================================

Packs themed around active wars, armed conflicts, or politically
sensitive war-zone regions are RED — they create real PR / safety
risk for the marketplace regardless of which side they take. Includes
but not limited to:
  • Russia / Ukraine war imagery — military uniforms, soldiers, war
    zones, "Z" marks, Ukrainian trident in militaristic/national
    context, "support our troops" framing for either side, ribbons
    of St. George, Ukrainian flag combined with combat/military
    elements, etc.
  • Israel / Palestine / Hamas / Hezbollah / Iran / Houthi imagery
    in war/political context (flags + combat, religious-political
    symbols + conflict)
  • Other active conflicts (Syria, Yemen, Sudan, Myanmar, etc.)

Non-conflict political content remains GREEN:
  • Caricatures of public figures (Trump, Putin, Musk, Durov, Zelensky)
    in humorous/parody context WITHOUT war framing
  • "MAKE X GREAT AGAIN" slogan riffs
  • Generic political satire / shitposting without combat imagery
  • Just having "Russian", "Ukrainian", "Israeli" cultural elements
    (food, language, traditional dress, national pride) without
    militarized framing — these are YELLOW (human glance) at most

==================================================================
CARTOON CHARACTERS RESEMBLING REAL PEOPLE
==================================================================

Cartoon / illustrated characters that resemble real public figures
(Durov, Trump, Musk, etc.) are GREEN by default — even if highly
recognizable — as long as the pack is humorous, fan-art, or generic
caricature. Examples:
  • "I love Durov" pack with cartoon Durov in funny situations → GREEN
  • Cartoon Trump dancing, holding crypto bag → GREEN
  • Cartoon Musk-as-Doge → GREEN

ONLY flag when the pack crosses into:
  • **Hardcore humiliation** — sexual degradation, gore, suicide
    framing, slurs aimed at the figure → RED
  • **Pure impersonation** — pack literally pretends to BE the figure's
    official content (fake "Pavel Durov Official Stickers" framing,
    real-photo deepfake context) → RED

The bar for RED here is high — political satire, irreverent humor,
mild teasing, drinking/smoking-themed cartoons of public figures all
remain GREEN.

A small corner reference (e.g. "TON" text in 5% of the cover, or a
tiny Telegram icon as decoration) does NOT trip this.

**Telegram Gifts** — official Telegram-issued animated/static
collectibles. Copying/tracing these specific designs is RED, regardless
of how memey the underlying subject is. Known TG Gifts include (not
exhaustive):
  Plush Pepe, Durov's Cap, Light Sword, Lol Pop, Nail Bracelet,
  Magic Potion, Hex Pot, Vintage Cigar, Astral Shard, Crystal Ball,
  Heart Locket, Heroic Helmet, Diamond Ring, Eternal Rose, Berry Box,
  Toy Bear, Trapped Heart, Snake Box, Skull Flower, Genie Lamp,
  B-Day Candle, Bunny Muffin, Spy Agaric, Witch Hat, Hanging Star,
  Restless Jar, Jolly Chimp, Cookie Heart, Desk Calendar, Top Hat,
  Sakura Flower, Eternal Candle, Swiss Watch, Holiday Drink,
  Homemade Cake, Voodoo Doll, Mad Pumpkin, Hypno Lollipop,
  Easter Egg, Lush Bouquet, Big Year, Snow Mittens, Snow Globe,
  Kissed Frog, Star Notepad, Party Sparkler, Loot Bag, Lunar Snake,
  Santa Hat, Bonded Ring, Westside Sign, Heart Locket, Tama Gadget,
  Record Player, Joyful Bundle, Star Notepad, Toy Bear, Pet Snake,
  Ginger Cookie, Jingle Bells, Winter Wreath, Xmas Stocking, Evil Eye,
  Stellar Rocket, Light Show, Eternal Candle, Whip Cupcake, Neko Helmet
Tracing or close-imitation of any of these specific designs (the 3D
rendered, premium-collectible aesthetic) = RED. Generic art on the
same theme (a generic plushy frog, a generic crown, a generic sword)
that does NOT match the TG Gift's specific design = GREEN.

IP — ONLY the sue-happy stuff, ONLY when used as primary content:
  • Disney / Pixar / Marvel / Lucasfilm / Star Wars NAMED characters
    (Mickey, Stitch, Iron Man, Spider-Man, Baby Yoda, Frozen cast,
    etc.)
  • Nintendo / The Pokémon Company NAMED characters
    (Mario, Luigi, Pikachu, Zelda, Link)
  • Sanrio (Hello Kitty, Cinnamoroll, Kuromi, My Melody)
  • Hasbro NAMED characters (My Little Pony characters, Transformers)
  • DC / Warner Bros. NAMED characters (Batman, Superman, Looney Tunes)
  • Studio Ghibli NAMED characters (Totoro, No-Face, Kiki)
  • Currently-airing/popular licensed anime as primary content (One
    Piece, Demon Slayer, Jujutsu Kaisen, Naruto, Dragon Ball,
    Attack on Titan, Spy x Family) — note: ORIGINAL anime-style
    characters are GREEN, only specific named characters are RED
  • Nickelodeon / Viacom (SpongeBob, Squidward, Patrick — yes Squidward)
  • Sports league logos / pro-athlete photos used as branding
  • Brand logos used as the pack's identity
    (Coca-Cola, Nike swoosh, McDonald's golden arches, etc.)
  • Direct tracing of an identifiable copyrighted still (zero
    transformation)

Hard NSFW (ONLY these — animated / cartoon / stylized sexual content
that is suggestive but not explicit is GREEN, not RED):
  • Explicit sexual content with VISIBLE GENITALIA or SEX ACTS, in any
    style (photo, cartoon, 3D, anime — explicit-explicit only)
  • ANY sexualization involving minors → instant RED, all styles
  • Real-photo gore / violence / terrorism imagery (cartoon violence
    is GREEN, real-photo violence is RED)
  • Hate symbols (swastika, KKK imagery, 1488 numerology, etc.) used
    non-satirically as pack identity

Suggestive cartoon / lingerie / romance / anime-style ecchi without
visible genitalia or minors = GREEN, not RED. Stylized ≠ explicit.
"Commercial sexualization" of cartoon / 3D-rendered / AI-generated
ADULT women is also GREEN — pin-up art, lewd cartoon expressions,
ahegao faces, "hidden sexuality" with nudity-implied-but-not-shown
is all GREEN. Only RED if:
  (a) actual genitalia visible, (b) sex acts depicted,
  (c) anyone could plausibly be a minor, or (d) real-photo NSFW.

PII / real-people (FLOOR concern — a single bad sticker triggers RED
even when surrounded by clean meme content; few-bad-many-good does
NOT dilute this):
  • Recognizable real PRIVATE individuals as PRIMARY content without
    clear consent (especially children / minors — instant RED)
  • Real-looking phone numbers / addresses / plates / IDs / SSNs
  • Sexual or compromising deepfakes of real public figures
  • Pack IMPERSONATES a real project/brand
    (claims to be Tonkeeper, Telegram official, etc.)

NOT a PII concern (these are GREEN):
  • The author's own pet (dog, cat, etc.) photographed in their home
    or an everyday setting → personal pet photos are fine even with
    incidental background details (visible iPhone, food, furniture)
  • Adult crowds / unidentifiable extras in the background of a meme
  • Stylized cartoon characters that *resemble* a real person but
    are clearly drawn / fan-art (see "Cartoon characters resembling
    real people" section above)

Scams — RED ONLY when the pack's CORE IDENTITY is a literal scam
claim or impersonation. Satirical use of words like "scam", "rugpull",
"ponzi", "ngmi" in 1-2 stickers as commentary on crypto culture is
GREEN, not RED. Examples to calibrate:
  • Pack literally titled "SCAM" with self-warning stickers ("DO NOT
    TRUST", "FAKE AD") AS THE PRIMARY CONTENT → RED
  • Pack impersonating a real exchange / wallet / project's marketing
    → RED
  • Fake giveaway / fake airdrop framing visible across most stickers
    → RED
  • Pack titled "Hardly" / "Вряд-ли" with 1-2 satirical stickers
    saying "TON SCAM" as crypto-cynicism humor → GREEN (satire)
  • Pack with "WAGMI" / "NGMI" / pessimistic crypto memes → GREEN
  • Mood stickers (skeptical, pessimistic, grumpy reactions) that
    happen to include "scam" word → GREEN
A description that promises "utilities" or links to an external
channel is NOT a scam signal — that's normal crypto-pack behavior.

==================================================================
TEXT IN STICKERS — read all popular languages
==================================================================
Stickers commonly contain text in English, Russian/Cyrillic, Spanish,
Chinese, Arabic, Hindi, Turkish, German, French, Japanese, Korean,
Portuguese, Italian, Polish, Ukrainian, Persian, Vietnamese, etc.
Read and translate all of it.

Flag in `nsfw_concerns` ONLY truly extreme text — "dark red" trash
that would make a TechCrunch headline:
  • Hate-speech slurs targeting a protected group
  • Explicit calls to real-world violence against a person/group
  • Doxing (real name + personal info combined)
  • Solicitation of CSAM or sale of hard drugs / weapons
  • Calls for terrorism / genocide / ethnic cleansing
  • Detailed self-harm encouragement targeting individuals

Do NOT flag mild edgy / "light red" / yellow text — these are GREEN:
  • Profanity, vulgar slang, sexual innuendo (in any language)
  • Generic "fuck the system" / political shitposting
  • Crypto pump messages ("TO THE MOON", "$X 1000x")
  • Mocking or insulting public figures (without doxing)
  • Edgy / provocative meme captions
  • Drug references that are humorous (not selling)
  • Russian/Spanish/etc. equivalents of any of the above

==================================================================
YELLOW (human review)
==================================================================

  • Specific named-character riff that isn't an established meme
    (looks like Pikachu but might be transformative — let a human eyeball)
  • Real public figure used non-satirically in semi-realistic way
  • Trademarked phrase as a small element (not the pack's identity)
  • Suggestive (non-explicit) photographic content with real-looking people
  • 1–2 borderline stickers in an otherwise clean set
  • Borderline IP: the gray zone between "obvious meme" and "obvious Disney"
  • Drug/weapon imagery used aspirationally (not satire / not commentary)

==================================================================
GREEN (auto-publish) — DEFAULT
==================================================================

Public memes — cultural commons, ALWAYS GREEN:
  • Pepe family: classic Pepe, Sad Pepe, Apu Apustaja, Plush Pepe,
    Smug Pepe, etc. (Plush Pepe is also a TON NFT — in-context fine)
  • Doge, Cheems, Shibe
  • Wojak family: Chad, Virgin, NPC, Soyjak, Coomer, Yes Chad, Gigachad
  • Trollface, rage comics, classic 4chan/Reddit memes
  • Nyan Cat, Distracted Boyfriend, Drake, Galaxy Brain, Two Buttons,
    "This is fine", "Stonks"
  • Crypto-native: gm, wagmi, hodl, ngmi, "few", "this is the way",
    "probably nothing", LFG, "to the moon"
  • TON-community in-jokes (Tony, Durov-as-meme, plush-pepe variants)

Parody / satire / transformative — protected, GREEN:
  • "MAKE TON GREAT AGAIN" or any "Make X Great Again" riff (parody)
  • Brand-pun mashups that obviously aren't impersonation
    ("Stonks", "Ligma Inc.", crypto-flavored corporate logo riffs)
  • Caricatures of public figures with humorous intent
  • Political satire (no hate symbols)
  • Crypto-ifications of pop culture (Trump-as-bull, Musk-as-Doge)

"In the style of" — style isn't IP, GREEN:
  • Anime-style ORIGINAL characters (no specific named characters)
  • Pixel-art originals
  • Low-poly / Roblox-look / Minecraft-look ORIGINAL avatars
    (only specific named Roblox/Minecraft characters are RED)
  • Cartoon-style originals
  • 3D-render style without specific brand identity

Trademark wordmark parodies — GREEN:
  • Wordmark riffs ("Abibas" instead of Adidas, "Stonks" instead of
    Stocks, "Lega" instead of Lego, "Pradd" instead of Prada, etc.)
  • Even paired with style-evoking imagery (3 stripes, sport aesthetic)
Only RED/YELLOW for wordmark parody if the ACTUAL trademarked logo
glyph appears (real Adidas trefoil, real Nike swoosh, real Apple
bitten-apple) used as the pack's primary identity.

Generic archetypes — GREEN:
  • Animals (frogs, ducks, cats, hippos, bears) WITHOUT specific
    named characters. A cute baby hippo is GREEN even if it
    superficially resembles Moo Deng — Moo Deng is a real animal,
    not a copyrighted character.
  • Generic emoji-style (hearts, faces, gestures, food, weather)
  • Original mascots created by the pack author

Crypto / web3 — GREEN:
  • Tokens, wallets, blockchains as subject matter
    (without impersonating specific projects)
  • TON / community inside jokes

==================================================================
EDGE-CASE GUIDANCE
==================================================================
  • Pack containing the OFFICIAL Telegram Plush Pepe Gift design
    (specific 3D-rendered plushy with the characteristic glow/pose) → RED
  • "Plush Pepe TO THE MOON!" with generic plushy-frog illustrations
    that DO NOT match the TG Gift design → GREEN (Pepe + crypto bullposting)
  • "MAKE TON GREAT AGAIN" → GREEN (parody, transformative)
  • Squidward in romantic context → RED (Viacom IP, no parody intent)
  • Baby hippo art with no Moo-Deng reference in metadata → GREEN
  • Roblox-aesthetic ORIGINAL characters → GREEN
  • Anime-style original characters that vibe like a show but aren't
    specific named characters → GREEN
  • Pack literally titled "SCAM" with self-warnings → RED
    (intentional buyer confusion)
  • Pun-named pack ("TONy Stark") with NO Iron Man imagery in stickers
    → GREEN (wordplay alone isn't IP)
  • Pun-named pack ("TONy Stark") WITH Iron Man imagery → RED

==================================================================

When in doubt between two adjacent categories, prefer the more
permissive one (GREEN over YELLOW, YELLOW over RED). False rejections
have appeal cost; false GREEN-publishes only matter for the carve-out
list above.

Be cautious about hallucinating specific details from low-resolution
stickers. If you're unsure whether something is "an infant" vs "a
kitten", or "a dead person" vs "a sleeping person", lean toward the
benign reading and SAY SO in the reasoning ("appears to be X but
could be Y") rather than asserting confidently.

Return ONLY a JSON object with EXACTLY these fields, no prose, no
markdown fences:
{{
  "risk_score": <integer 0-100>,
  "risk_category": "GREEN" | "YELLOW" | "RED",
  "ip_concerns": [<recognizable carve-out-list IPs only, empty otherwise>],
  "nsfw_concerns": [<hard-NSFW concerns only, empty otherwise>],
  "pii_concerns": [<PII / impersonation concerns, empty otherwise>],
  "scam_concerns": [<scam framing concerns, empty otherwise>],
  "reasoning": "<one or two sentences>"
}}

Scoring guide (be calibrated):
  GREEN  (0-34):  passes the carve-outs. Memes, parody, generic
                  archetypes, "in the style of", original art.
  YELLOW (35-69): real ambiguity worth a human glance.
  RED    (70-100): clear sue-happy IP, clear hard-NSFW, clear scam,
                  clear impersonation, clear doxing."""


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} block out of Claude output, robust to fences."""
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_EXT_TO_MIME = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}

# Anthropic's many-image-request mode rejects any single image with a
# dimension > 2000px (applies to URL-source images too). Re-encode +
# downscale defensively for ALL images we send.
_MAX_IMAGE_DIM = 1280


def _normalize_to_part(im) -> dict | None:
    """PIL Image → Anthropic base64 image part, downscaled."""
    try:
        from PIL import Image as _Image  # noqa: F401
        import io
        if max(im.size) > _MAX_IMAGE_DIM:
            im.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        data = base64.standard_b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        }
    except Exception:  # noqa: BLE001
        return None


def _local_image_part(path: Path) -> dict | None:
    """Read a local image (any common format), downscale, return base64 part."""
    if path.suffix.lower() not in _EXT_TO_MIME:
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            return _normalize_to_part(im)
    except Exception:  # noqa: BLE001
        return None


def _url_image_part(url: str, timeout: int = 20) -> dict | None:
    """Download a URL image, downscale, return base64 part.

    Letting Anthropic fetch URLs directly would skip this download, but
    in many-image-request mode it enforces a 2000px max-dimension cap
    on URL images too, and our CDN serves some 2865px covers.
    """
    try:
        from PIL import Image
        import io as _io
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with Image.open(_io.BytesIO(r.content)) as im:
            im.load()
            return _normalize_to_part(im)
    except Exception:  # noqa: BLE001
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def _claude_one_call(
    title: str,
    description: str,
    image_urls: list[str],
    local_images: list[Path],
    cfg: Config,
    client: "Anthropic | None",
    batch_label: str = "",
) -> dict:
    """One Claude moderation call. Used directly OR by the batched wrapper."""
    if client is None:
        return _claude_default_red("anthropic SDK not configured")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": CLAUDE_PROMPT.format(
                title=title or "(none)",
                description=description or "(none)",
            ) + (f"\n\n(Reviewing batch: {batch_label})" if batch_label else ""),
        }
    ]
    for url in image_urls:
        part = _url_image_part(url)
        if part is not None:
            content.append(part)
    for p in local_images:
        part = _local_image_part(p)
        if part is not None:
            content.append(part)

    try:
        msg = client.messages.create(
            model=cfg.claude_model,
            max_tokens=900,
            messages=[{"role": "user", "content": content}],
            timeout=120.0,
        )
    except Exception as e:  # noqa: BLE001
        return _claude_default_red(f"claude error: {e}")

    raw = msg.content[0].text if msg.content else ""
    parsed = _extract_json(raw)
    if not parsed:
        return _claude_default_red(f"unparseable claude output: {raw[:200]}")

    return {
        "risk_score": int(parsed.get("risk_score", 90)),
        "risk_category": str(parsed.get("risk_category", "RED")).upper(),
        "ip_concerns": parsed.get("ip_concerns", []) or [],
        "nsfw_concerns": parsed.get("nsfw_concerns", []) or [],
        "pii_concerns": parsed.get("pii_concerns", []) or [],
        "scam_concerns": parsed.get("scam_concerns", []) or [],
        "reasoning": (parsed.get("reasoning", "") or "")[:400],
        "error": None,
    }


def _claude_default_red(reason: str) -> dict:
    return {
        "risk_score": 90,
        "risk_category": "RED",
        "ip_concerns": [],
        "nsfw_concerns": [],
        "pii_concerns": [],
        "scam_concerns": [],
        "reasoning": reason,
        "error": reason,
    }


_CAT_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def claude_pack_review(
    title: str,
    description: str,
    image_urls: list[str],
    local_images: list[Path],
    cfg: Config,
    client: "Anthropic | None",
) -> dict:
    """Review a pack, batching stickers to ensure ALL get inspected.

    Marketing images (logo+cover URLs) ride along with the first batch.
    Each subsequent batch is a fresh Claude call with later sticker
    thumbnails. Worst-case verdict wins; concerns are unioned.
    """
    batch = max(cfg.claude_batch_size, 4)
    if len(local_images) <= batch:
        return _claude_one_call(title, description, image_urls, local_images, cfg, client)

    chunks: list[tuple[list[str], list[Path], str]] = []
    total = len(local_images)
    for i in range(0, total, batch):
        slice_ = local_images[i : i + batch]
        urls = image_urls if i == 0 else []
        label = f"stickers {i}..{i + len(slice_) - 1} of {total}"
        chunks.append((urls, slice_, label))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as pool:
        futures = [
            pool.submit(_claude_one_call, title, description, urls, locs, cfg, client, label)
            for urls, locs, label in chunks
        ]
        for f in as_completed(futures):
            results.append(f.result())

    # Don't let a failed batch poison successful ones — aggregate over
    # the non-erroring batches if we have any.
    good = [r for r in results if not r.get("error")]
    voting = good if good else results

    def union(key: str) -> list[str]:
        seen: dict[str, None] = {}
        for r in voting:
            for c in r.get(key, []) or []:
                seen[c] = None
        return list(seen.keys())

    worst = max(voting, key=lambda r: _CAT_ORDER.get(r.get("risk_category", "RED"), 2))
    score = max(int(r.get("risk_score", 0)) for r in voting)
    reasonings = [r["reasoning"] for r in voting if r.get("risk_category") != "GREEN" and r.get("reasoning")]
    if not reasonings:
        reasonings = [voting[0].get("reasoning", "")]
    errors = "; ".join(r["error"] for r in results if r.get("error")) or None
    return {
        "risk_score": score,
        "risk_category": worst.get("risk_category", "RED"),
        "ip_concerns": union("ip_concerns"),
        "nsfw_concerns": union("nsfw_concerns"),
        "pii_concerns": union("pii_concerns"),
        "scam_concerns": union("scam_concerns"),
        "reasoning": " | ".join(reasonings)[:600],
        "error": errors,
    }


# ---------- decision ----------

@dataclass
class PackVerdict:
    pack_id: str
    decision: str
    combined_score: float
    openai_max: float
    claude_risk: int
    claude_category: str
    openai_hard_flags: list[str] = field(default_factory=list)
    openai_soft_flags: list[str] = field(default_factory=list)
    ip_concerns: list[str] = field(default_factory=list)
    nsfw_concerns: list[str] = field(default_factory=list)
    pii_concerns: list[str] = field(default_factory=list)
    scam_concerns: list[str] = field(default_factory=list)
    claude_reason: str = ""
    image_count: int = 0
    errors: list[str] = field(default_factory=list)


def combine(
    openai_max: float,
    openai_hard_flags: list[str],
    claude_score: int,
    claude_category: str,
    cfg: Config,
) -> tuple[str, float]:
    """Combine signals, with hard-overrides for hard flags / RED categories."""
    openai_norm = openai_max * 100  # already 0-1
    combined = 0.30 * openai_norm + 0.70 * claude_score

    if openai_hard_flags or claude_category == "RED" or combined >= cfg.combined_red:
        return "RED", combined
    if claude_category == "YELLOW" or combined >= cfg.combined_yellow:
        return "YELLOW", combined
    return "GREEN", combined


# ---------- pack pipeline ----------

def _collect_images(row: dict, cols: dict[str, str]) -> list[str]:
    urls: list[str] = []
    if cols.get("image_urls"):
        raw = row.get(cols["image_urls"])
        if isinstance(raw, str) and raw.strip():
            urls.extend(u.strip() for u in raw.split(",") if u.strip())
    for c in cols.get("image_url_cols", []) or []:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            urls.append(v.strip())
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def process_pack(
    row: dict,
    cfg: Config,
    claude_client: "Anthropic | None",
    openai_client: "OpenAI | None",
    cols: dict[str, str],
    tg_cache_dir: Path | None,
) -> PackVerdict:
    pack_id = str(row.get(cols["pack_id"], ""))
    title = str(row.get(cols["title"], "") or "")
    desc = str(row.get(cols["description"], "") or "")
    cdn_images = _collect_images(row, cols)[: cfg.max_images_per_pack]

    errors: list[str] = []
    tg_meta: dict = {}
    tg_paths: list[Path] = []

    # Phase 2: pull individual sticker thumbnails via Telegram Bot API.
    pack_url = ""
    if cols.get("pack_url"):
        pack_url = str(row.get(cols["pack_url"], "") or "")
    if cfg.telegram_bot_token and tg_cache_dir is not None and pack_url:
        slug = slug_from_url(pack_url)
        if slug:
            try:
                tg_paths, tg_meta = ingest_pack(
                    slug, cfg.telegram_bot_token, tg_cache_dir,
                    max_n=cfg.max_stickers_per_pack,
                )
                if tg_meta.get("error"):
                    errors.append(f"tg:{tg_meta['error']}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"tg:{e}")

    # Hard rejects we can decide before calling Claude:
    # 1) TG sticker set inaccessible (deleted, private, invalid slug)
    # 2) Pack has fewer stickers than min_stickers
    sticker_count = tg_meta.get("count", 0)
    if cfg.telegram_bot_token and tg_cache_dir is not None and pack_url:
        slug = slug_from_url(pack_url)
        if slug and tg_meta.get("error"):
            return PackVerdict(
                pack_id=pack_id,
                decision="RED",
                combined_score=100.0,
                openai_max=0.0,
                claude_risk=100,
                claude_category="RED",
                scam_concerns=[f"unverifiable: cannot read sticker pack ({tg_meta['error']})"],
                claude_reason=f"Auto-rejected: cannot access sticker pack via Telegram ({tg_meta['error']}). Pack must be verifiable to publish.",
                image_count=len(cdn_images) + len(tg_paths),
                errors=errors,
            )
    if cfg.telegram_bot_token and sticker_count and sticker_count < cfg.min_stickers:
        return PackVerdict(
            pack_id=pack_id,
            decision="RED",
            combined_score=100.0,
            openai_max=0.0,
            claude_risk=100,
            claude_category="RED",
            scam_concerns=[f"min_stickers: pack has only {sticker_count} stickers (<{cfg.min_stickers} required)"],
            claude_reason=f"Auto-rejected: pack contains {sticker_count} sticker(s); minimum is {cfg.min_stickers}.",
            image_count=len(cdn_images) + len(tg_paths),
            errors=errors,
        )
    if cfg.telegram_bot_token and sticker_count and sticker_count > cfg.max_stickers:
        return PackVerdict(
            pack_id=pack_id,
            decision="RED",
            combined_score=100.0,
            openai_max=0.0,
            claude_risk=100,
            claude_category="RED",
            scam_concerns=[f"max_stickers: pack has {sticker_count} stickers (>{cfg.max_stickers} allowed)"],
            claude_reason=f"Auto-rejected: pack contains {sticker_count} sticker(s); maximum allowed is {cfg.max_stickers}.",
            image_count=len(cdn_images) + len(tg_paths),
            errors=errors,
        )

    text_payload = (title + "\n" + desc + ("\n" + tg_meta.get("title", "") if tg_meta.get("title") else "")).strip()
    openai_res = openai_moderate(text_payload, cdn_images, cfg, openai_client)
    if openai_res.get("error") and openai_res["error"] != "no_client":
        errors.append(f"openai:{openai_res['error']}")

    claude_res = claude_pack_review(title, desc, cdn_images, tg_paths, cfg, claude_client)
    if claude_res.get("error"):
        errors.append(f"claude:{claude_res['error']}")

    decision, combined_score = combine(
        openai_max=openai_res["max_score"],
        openai_hard_flags=openai_res["hard_flags"],
        claude_score=claude_res["risk_score"],
        claude_category=claude_res["risk_category"],
        cfg=cfg,
    )

    return PackVerdict(
        pack_id=pack_id,
        decision=decision,
        combined_score=round(combined_score, 1),
        openai_max=round(openai_res["max_score"], 3),
        claude_risk=claude_res["risk_score"],
        claude_category=claude_res["risk_category"],
        openai_hard_flags=openai_res["hard_flags"],
        openai_soft_flags=openai_res["soft_flags"],
        ip_concerns=claude_res["ip_concerns"],
        nsfw_concerns=claude_res["nsfw_concerns"],
        pii_concerns=claude_res["pii_concerns"],
        scam_concerns=claude_res["scam_concerns"],
        claude_reason=claude_res["reasoning"],
        image_count=len(cdn_images) + len(tg_paths),
        errors=errors,
    )


# ---------- caching ----------

class JsonlCache:
    """Append-only per-pack cache so we can resume after crashes.

    Thread-safe: cache.put() is guarded by an internal lock so we can
    write from any thread without interleaving lines. We also de-dupe
    in put() so a re-submitted pack_id doesn't double up the file.
    """

    def __init__(self, path: Path):
        import threading
        self.path = path
        self.seen: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self.seen[str(obj["pack_id"])] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, pack_id: str) -> dict | None:
        return self.seen.get(str(pack_id))

    def put(self, verdict: PackVerdict) -> None:
        obj = asdict(verdict)
        pid = str(verdict.pack_id)
        with self._lock:
            self.seen[pid] = obj
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--output", default="moderated.csv", help="Output CSV path")
    p.add_argument("--cache", default=".pack_cache.jsonl", help="Resume cache (JSONL)")
    p.add_argument("--col-pack-id", default="id")
    p.add_argument("--col-title", default="name")
    p.add_argument("--col-description", default="description")
    p.add_argument(
        "--col-image-urls",
        default="",
        help="Single column of comma-separated URLs (e.g. image_urls). Empty = use --col-image-url-cols.",
    )
    p.add_argument(
        "--col-image-url-cols",
        default="logo_url,cover_url",
        help="Comma-list of single-URL columns to combine (default: logo_url,cover_url)",
    )
    p.add_argument("--col-pack-url", default="stickerpack_url",
                   help="Column with t.me/addstickers/<slug> URL (Phase 2). Empty disables Phase 2.")
    p.add_argument("--col-decision", default="decision", help="Ground-truth column for --eval")
    p.add_argument("--tg-cache", default=".tg_cache", help="Local cache for Telegram sticker thumbnails")
    p.add_argument("--no-tg", action="store_true", help="Disable Phase 2 (Telegram ingest) even if token is set")
    p.add_argument("--eval", action="store_true", help="Print confusion matrix vs ground-truth column")
    p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all)")
    p.add_argument("--no-cache", action="store_true", help="Ignore cache, redo everything")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.from_env()

    if not cfg.anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        return 2
    if Anthropic is None:
        print("ERROR: anthropic package not installed (pip install -r requirements.txt)", file=sys.stderr)
        return 2

    claude_client = Anthropic(api_key=cfg.anthropic_key)
    openai_client = None
    if cfg.openai_key and OpenAI is not None:
        openai_client = OpenAI(api_key=cfg.openai_key)
    elif not cfg.openai_key:
        print("note: OPENAI_API_KEY not set — running Claude-only (still works, slightly costlier)")

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    image_url_cols = [c.strip() for c in (args.col_image_url_cols or "").split(",") if c.strip()]
    cols = {
        "pack_id": args.col_pack_id,
        "title": args.col_title,
        "description": args.col_description,
        "image_urls": args.col_image_urls,
        "image_url_cols": image_url_cols,
        "pack_url": args.col_pack_url if args.col_pack_url and args.col_pack_url in df.columns else "",
    }
    required = [args.col_pack_id, args.col_title, args.col_description]
    if args.col_image_urls:
        required.append(args.col_image_urls)
    required.extend(image_url_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing CSV columns: {missing}. Found: {list(df.columns)}", file=sys.stderr)
        return 2

    tg_cache_dir: Path | None = None
    if cfg.telegram_bot_token and not args.no_tg and cols["pack_url"]:
        tg_cache_dir = Path(args.tg_cache)
        tg_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Phase 2 enabled: ingesting individual stickers via Telegram into {tg_cache_dir}/")
    else:
        if not cfg.telegram_bot_token:
            print("Phase 2 disabled: TELEGRAM_BOT_TOKEN not set (logo+cover only)")
        elif args.no_tg:
            print("Phase 2 disabled: --no-tg flag")
        else:
            print(f"Phase 2 disabled: column '{args.col_pack_url}' not in CSV")

    cache_path = Path(args.cache)
    if args.no_cache and cache_path.exists():
        cache_path.unlink()
    cache = JsonlCache(cache_path)

    rows = df.to_dict(orient="records")
    pending = [r for r in rows if not cache.get(r[cols["pack_id"]])]
    print(f"Total: {len(rows)} | Cached: {len(rows) - len(pending)} | To process: {len(pending)}")

    verdicts: list[PackVerdict] = []
    for r in rows:
        cached = cache.get(r[cols["pack_id"]])
        if cached:
            verdicts.append(PackVerdict(**cached))

    started = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=cfg.pack_workers) as pool:
            futures = {
                pool.submit(process_pack, r, cfg, claude_client, openai_client, cols, tg_cache_dir): r
                for r in pending
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="packs"):
                try:
                    v = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  ! pack failed: {e}", file=sys.stderr)
                    continue
                cache.put(v)
                verdicts.append(v)

    out_rows = []
    for v in verdicts:
        d = asdict(v)
        for k in ("openai_hard_flags", "openai_soft_flags"):
            d[k] = ",".join(d[k])
        for k in ("ip_concerns", "nsfw_concerns", "pii_concerns", "scam_concerns"):
            d[k] = "; ".join(d[k])
        d["errors"] = "; ".join(d["errors"])
        out_rows.append(d)

    out_df = pd.DataFrame(out_rows)
    merged = df.merge(out_df, left_on=cols["pack_id"], right_on="pack_id", how="left", suffixes=("", "_auto"))
    merged.to_csv(args.output, index=False)

    counts = out_df["decision"].value_counts().to_dict() if not out_df.empty else {}
    elapsed = time.time() - started
    print("\n=== Summary ===")
    for k in ("GREEN", "YELLOW", "RED"):
        c = counts.get(k, 0)
        pct = (c / max(len(out_df), 1)) * 100
        print(f"  {k:7s}: {c:5d}  ({pct:5.1f}%)")
    print(f"  total : {len(out_df)}")
    print(f"  errors: {sum(1 for v in verdicts if v.errors)}")
    if pending:
        print(f"  time  : {elapsed:.1f}s ({elapsed / len(pending):.2f}s/pack)")
    print(f"\nWrote {args.output}")

    if args.eval and args.col_decision in df.columns:
        print_eval(merged, args.col_decision)

    return 0


def print_eval(merged: pd.DataFrame, truth_col: str) -> None:
    """Compare auto decisions to historical human decisions."""
    df = merged.dropna(subset=[truth_col]).copy()
    df["truth"] = df[truth_col].astype(str).str.lower().map(
        {"approved": "publish", "rejected": "reject"}
    )
    df = df.dropna(subset=["truth"])

    auto_col = "decision_auto" if "decision_auto" in df.columns else "decision"
    auto = df[auto_col].astype(str).str.upper()
    truth = df["truth"]

    print("\n=== Eval vs ground truth ===")
    print(f"  rows with truth: {len(df)}")

    print("\n  Auto x Truth (counts):")
    print("                     publish   reject")
    for cat in ("GREEN", "YELLOW", "RED"):
        sub = df[auto == cat]
        p = (sub["truth"] == "publish").sum()
        r = (sub["truth"] == "reject").sum()
        total = len(sub)
        print(f"    {cat:7s} (n={total:3d})  {p:6d}    {r:6d}")

    def metrics(label: str, predict_reject: pd.Series) -> None:
        tp = ((predict_reject) & (truth == "reject")).sum()
        fp = ((predict_reject) & (truth == "publish")).sum()
        fn = ((~predict_reject) & (truth == "reject")).sum()
        tn = ((~predict_reject) & (truth == "publish")).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(
            f"  [{label}]"
            f"  TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  TN={tn:3d}"
            f"   precision={precision:.2%}  recall={recall:.2%}"
        )

    print("\n  As reject-classifier (RED only = reject):")
    metrics("strict   ", auto == "RED")
    print("  As reject-classifier (RED+YELLOW = reject):")
    metrics("inclusive", auto.isin(["RED", "YELLOW"]))

    auto_publish = (auto == "GREEN").sum()
    print(f"\n  Would auto-publish (GREEN): {auto_publish}/{len(df)} = "
          f"{auto_publish / max(len(df), 1):.1%}")
    green_correct = ((auto == "GREEN") & (truth == "publish")).sum()
    green_wrong = ((auto == "GREEN") & (truth == "reject")).sum()
    print(f"    of those:  {green_correct} truly publishable, "
          f"{green_wrong} would have been wrongly auto-published")
    if (auto == "GREEN").sum():
        print(f"    GREEN false-publish rate: "
              f"{green_wrong / (auto == 'GREEN').sum():.2%}")


if __name__ == "__main__":
    sys.exit(main())
