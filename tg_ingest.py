"""
Telegram sticker pack ingestion.

Given a t.me/addstickers/<slug> URL and a bot token, download
per-sticker static thumbnails to a local cache and return the paths.

We download `thumbnail.file_id` (Telegram pre-renders a static
PNG/WebP/JPG for every sticker — including TGS/animated and WebM/video
ones), so we sidestep lottie + ffmpeg decoding entirely.

Public API:
  slug_from_url(url) -> str | None
  ingest_pack(slug, bot_token, cache_dir, max_n=12) -> list[Path]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

TG_API = "https://api.telegram.org"


def slug_from_url(url: str) -> str | None:
    """Extract sticker-set slug from a t.me/addstickers/<slug> URL."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url.strip())
    except Exception:  # noqa: BLE001
        return None
    parts = [p for p in parsed.path.split("/") if p]
    # Expected: ['addstickers', '<slug>']
    if len(parts) >= 2 and parts[0].lower() == "addstickers":
        return parts[1]
    return None


@retry(
    retry=retry_if_exception_type((requests.RequestException,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _tg_get(token: str, method: str, **params) -> dict:
    r = requests.get(f"{TG_API}/bot{token}/{method}", params=params, timeout=30)
    # Telegram returns 200 with ok:false for app-level errors; also 400/401.
    # Don't blanket raise_for_status — we want to read the message.
    try:
        return r.json()
    except ValueError:
        r.raise_for_status()
        raise


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def ingest_pack(
    slug: str,
    bot_token: str,
    cache_dir: Path,
    max_n: int = 12,
) -> tuple[list[Path], dict]:
    """Download up to max_n sticker thumbnails for a pack.

    Returns (list_of_local_paths, metadata) where metadata contains
    the pack title and counts; list is empty on failure.
    """
    pack_dir = cache_dir / slug
    meta_path = pack_dir / ".meta.json"

    # Always fetch fresh metadata — getStickerSet is authoritative for count.
    # Disk cache is used ONLY to skip re-downloading thumbnails, never to
    # infer pack size (a partial cache from a previous interrupted run, or
    # from a concurrent process, would return a wrong count).
    info = _tg_get(bot_token, "getStickerSet", name=slug)
    if not info.get("ok"):
        return [], {"slug": slug, "error": f"getStickerSet: {info.get('description', info)}"}

    result = info["result"]
    all_stickers = result.get("stickers", [])
    total_count = len(all_stickers)
    stickers = all_stickers[:max_n]
    title = result.get("title", "")
    expected_to_download = len(stickers)

    # Disk cache short-circuit: only when we've already downloaded
    # at least as many files as we'd download now (i.e. cache is complete
    # for this max_n).
    if pack_dir.exists():
        existing = sorted(
            p for p in pack_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
        if len(existing) >= expected_to_download and expected_to_download > 0:
            meta = {
                "slug": slug,
                "title": title,
                "count": total_count,
                "downloaded": len(existing),
                "from_cache": True,
            }
            return existing[:max_n], meta

    pack_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i, st in enumerate(stickers):
        thumb = st.get("thumbnail") or st.get("thumb")
        if not thumb or "file_id" not in thumb:
            # Fallback: small static stickers may not have a separate
            # thumb; use the sticker file itself.
            file_id = st.get("file_id")
            if not file_id:
                continue
        else:
            file_id = thumb["file_id"]

        try:
            f = _tg_get(bot_token, "getFile", file_id=file_id)
        except Exception as e:  # noqa: BLE001
            log.warning("getFile failed for %s/%d: %s", slug, i, e)
            continue
        if not f.get("ok"):
            continue
        file_path = f["result"].get("file_path")
        if not file_path:
            continue

        # Telegram returns paths like "thumbnails/file_123.webp" — keep ext.
        ext = os.path.splitext(file_path)[1] or ".webp"
        dest = pack_dir / f"{i:02d}{ext}"
        url = f"{TG_API}/file/bot{bot_token}/{file_path}"
        try:
            _download(url, dest)
            paths.append(dest)
        except Exception as e:  # noqa: BLE001
            log.warning("download failed for %s/%d: %s", slug, i, e)
            continue

    meta = {
        "slug": slug,
        "title": title,
        "count": total_count,
        "downloaded": len(paths),
        "is_animated_pack": any(s.get("is_animated") for s in stickers),
        "is_video_pack": any(s.get("is_video") for s in stickers),
    }
    try:
        import json
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return paths, meta
