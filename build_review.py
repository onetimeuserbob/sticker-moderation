"""Render a CSV of moderation verdicts as a side-by-side HTML review page."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def slug_from_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    parts = [p for p in urlparse(u).path.split("/") if p]
    return parts[1] if len(parts) >= 2 and parts[0].lower() == "addstickers" else ""


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro", system-ui, sans-serif;
       background: #0e0e10; color: #eee; margin: 0; padding: 24px 24px 120px; }
h1 { font-weight: 600; margin: 0 0 16px; }
.summary { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; align-items: center; }
.tag { padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 13px; }
.tag-green, .tag-GREEN { background: #114a1f; color: #6fe896; }
.tag-yellow, .tag-YELLOW { background: #5c4310; color: #ffcc66; }
.tag-red, .tag-RED { background: #5c1010; color: #ff8080; }
.tag-approved { background: #0e3a4a; color: #7fd0ee; }
.tag-rejected { background: #4a2810; color: #ff9f6a; }
.pack { background: #18181b; border-radius: 12px; padding: 20px; margin-bottom: 16px;
        border-left: 6px solid #444; }
.pack-green { border-left-color: #2ea050; }
.pack-yellow { border-left-color: #d2a035; }
.pack-red { border-left-color: #d04040; }
.pack-disagree { box-shadow: 0 0 0 2px rgba(255,180,0,0.4); }
.pack[data-overridden="1"] { box-shadow: 0 0 0 2px #6cb0e0; }
.head { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.head h2 { margin: 0; font-size: 18px; }
.meta { color: #999; font-size: 13px; }
.meta a { color: #6cb0e0; text-decoration: none; }
.meta a:hover { text-decoration: underline; }
.imgs { display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-start; padding: 8px 0; }
.imgs img { max-height: 96px; max-width: 96px; background: #fff; border-radius: 6px;
            object-fit: contain; padding: 4px; }
.imgs .marketing { max-height: 140px; max-width: 200px; }
.imgs .label { font-size: 10px; color: #777; text-align: center; }
.imgs .item { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.reason { background: #222226; border-radius: 6px; padding: 10px 12px; font-size: 13px;
          line-height: 1.5; margin-top: 8px; }
.concern { display: inline-block; background: #2c1818; color: #f4a; border-radius: 4px;
           padding: 2px 8px; font-size: 11px; margin: 2px 4px 2px 0; }
.concern-ip { background: #2c1f0a; color: #ffba66; }
.concern-nsfw { background: #2c1818; color: #f47878; }
.concern-pii { background: #1f1c2c; color: #b0a0ff; }
.concern-scam { background: #2c2810; color: #fff66e; }
.desc { font-style: italic; color: #aaa; margin: 4px 0 8px; }

/* Feedback UI */
.fb { margin-top: 12px; padding-top: 12px; border-top: 1px dashed #333; }
.fb .row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.fb button { background: #2a2a2e; color: #ddd; border: 1px solid #444; border-radius: 6px;
             padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 600; }
.fb button:hover { background: #3a3a3e; }
.fb button.active.green { background: #114a1f; color: #6fe896; border-color: #2ea050; }
.fb button.active.yellow { background: #5c4310; color: #ffcc66; border-color: #d2a035; }
.fb button.active.red { background: #5c1010; color: #ff8080; border-color: #d04040; }
.fb button.active.unset { background: #333; color: #ddd; border-color: #555; }
.fb textarea { width: 100%; min-height: 50px; background: #0e0e10; color: #eee;
               border: 1px solid #333; border-radius: 6px; padding: 8px;
               font-family: inherit; font-size: 13px; box-sizing: border-box; }

/* Sticky footer with global notes + export */
.bar { position: fixed; bottom: 0; left: 0; right: 0; background: #18181bee;
       backdrop-filter: blur(8px); border-top: 1px solid #333;
       padding: 12px 24px; display: flex; gap: 12px; align-items: center; z-index: 99; }
.bar textarea { flex: 1; min-height: 36px; background: #0e0e10; color: #eee;
                border: 1px solid #444; border-radius: 6px; padding: 6px 10px;
                font-family: inherit; font-size: 13px; }
.bar button { background: #2a4a8a; color: #eee; border: none; border-radius: 6px;
              padding: 10px 16px; cursor: pointer; font-weight: 600; font-size: 13px; }
.bar button:hover { background: #3a5a9a; }
.bar button.secondary { background: #2a2a2e; }
.bar .count { font-size: 12px; color: #999; }
.toast { position: fixed; bottom: 90px; right: 24px; background: #2ea050; color: #fff;
         padding: 10px 16px; border-radius: 6px; font-size: 13px; opacity: 0;
         transition: opacity 0.2s; pointer-events: none; }
.toast.show { opacity: 1; }
"""


def render(df: pd.DataFrame, cache_dir: Path, seed_feedback: dict | None = None) -> str:
    counts = df["auto_decision"].value_counts().to_dict()
    rows_html = []
    for _, r in df.iterrows():
        auto = str(r["auto_decision"]).upper()
        human = str(r["human_decision"]).lower() if pd.notna(r.get("human_decision")) else ""
        agree = (auto == "RED" and human == "rejected") or (auto == "GREEN" and human == "approved")
        pack_class = {"GREEN": "pack-green", "YELLOW": "pack-yellow", "RED": "pack-red"}.get(auto, "")
        if not agree and human:
            pack_class += " pack-disagree"

        slug = slug_from_url(r.get("telegram_pack", ""))
        sticker_imgs = []
        if slug and (cache_dir / slug).exists():
            files = sorted(p for p in (cache_dir / slug).iterdir() if p.is_file() and not p.name.startswith("."))
            for p in files:
                rel = (cache_dir.name + "/" + slug + "/" + p.name)
                sticker_imgs.append(
                    f'<div class="item"><img src="{html.escape(rel)}" loading="lazy"/>'
                    f'<div class="label">{html.escape(p.stem)}</div></div>'
                )

        def mkconcerns(s, cls):
            if not isinstance(s, str) or not s.strip():
                return ""
            items = [c.strip() for c in s.split(";") if c.strip()]
            return "".join(f'<span class="concern {cls}">{html.escape(c)}</span>' for c in items)

        ip_html = mkconcerns(r.get("ip_concerns"), "concern-ip")
        nsfw_html = mkconcerns(r.get("nsfw_concerns"), "concern-nsfw")
        pii_html = mkconcerns(r.get("pii_concerns"), "concern-pii")
        scam_html = mkconcerns(r.get("scam_concerns"), "concern-scam")
        concerns_html = ip_html + nsfw_html + pii_html + scam_html

        logo = r.get("logo_url") or ""
        cover = r.get("cover_url") or ""
        marketing = ""
        if isinstance(logo, str) and logo:
            marketing += f'<div class="item"><img class="marketing" src="{html.escape(logo)}" loading="lazy"/><div class="label">logo</div></div>'
        if isinstance(cover, str) and cover:
            marketing += f'<div class="item"><img class="marketing" src="{html.escape(cover)}" loading="lazy"/><div class="label">cover</div></div>'

        human_tag = (
            f'<span class="tag tag-{human}">human: {human}</span>' if human else ""
        )
        auto_tag = f'<span class="tag tag-{auto.lower()}">auto: {auto} ({r.get("risk_score","?")})</span>'
        agree_marker = "✓" if agree else ("↯" if human else "")

        desc = html.escape(str(r.get("description") or ""))
        reason = html.escape(str(r.get("claude_reason") or ""))

        pack_key = html.escape(str(r["name"]))
        n_stickers = len(sticker_imgs)
        rows_html.append(f"""
<div class="pack {pack_class}" data-pack="{pack_key}" data-auto="{auto}">
  <div class="head">
    <h2>{pack_key}</h2>
    {auto_tag}
    {human_tag}
    <span class="meta">{agree_marker}</span>
    <span class="meta">·</span>
    <span class="meta">{n_stickers} sticker(s)</span>
    <span class="meta">·</span>
    <span class="meta"><a href="{html.escape(str(r.get('telegram_pack','')))}">telegram pack</a></span>
  </div>
  <div class="desc">{desc or "(no description)"}</div>
  <div class="imgs">{marketing}</div>
  <div class="imgs">{''.join(sticker_imgs)}</div>
  {f'<div>{concerns_html}</div>' if concerns_html else ''}
  <div class="reason">{reason}</div>
  <div class="fb">
    <div class="row">
      <span class="meta">override:</span>
      <button class="ovr" data-v="GREEN">GREEN</button>
      <button class="ovr" data-v="YELLOW">YELLOW</button>
      <button class="ovr" data-v="RED">RED</button>
      <button class="ovr" data-v="">clear</button>
      <span class="meta override-status"></span>
    </div>
    <textarea class="comment" placeholder="rubric tweak / feedback for this pack…"></textarea>
  </div>
</div>
""")

    summary = (
        f'<div class="summary">'
        f'<span class="tag tag-green">GREEN: {counts.get("GREEN",0)}</span>'
        f'<span class="tag tag-yellow">YELLOW: {counts.get("YELLOW",0)}</span>'
        f'<span class="tag tag-red">RED: {counts.get("RED",0)}</span>'
        f'<span class="tag" style="background:#222;color:#aaa">total: {len(df)}</span>'
        f'<span class="meta" style="margin-left:16px">click GREEN/YELLOW/RED on any pack to override; type comments freely. Press <b>Save feedback</b> at the bottom to download <code>feedback.json</code>.</span>'
        f'</div>'
    )
    seed_json = json.dumps(seed_feedback or {})
    js = """
const KEY = 'sticker_review_v1';
const SEED = __SEED__;
function load() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } }
function save(s) { localStorage.setItem(KEY, JSON.stringify(s)); }
function seedIfEmpty() {
  const cur = load();
  const empty = !cur || (!cur.general_notes && !(cur.packs && Object.keys(cur.packs).length));
  if (empty && SEED && (SEED.packs || SEED.general_notes)) save(SEED);
}
function reseed() {
  if (!confirm('Replace your current annotations with the embedded snapshot from feedback.json?')) return;
  save(SEED); renderAllFromState(); toast('reset to snapshot');
}
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}
function renderAllFromState() {
  const s = load();
  document.getElementById('global-notes').value = s.general_notes || '';
  document.querySelectorAll('.pack').forEach(p => {
    const key = p.dataset.pack;
    const e = (s.packs || {})[key] || {};
    p.dataset.overridden = e.override ? '1' : '0';
    p.querySelector('.comment').value = e.comment || '';
    p.querySelectorAll('.ovr').forEach(b => {
      b.classList.remove('active','green','yellow','red','unset');
      if ((b.dataset.v || '') === (e.override || '')) {
        b.classList.add('active');
        b.classList.add(e.override ? e.override.toLowerCase() : 'unset');
      }
    });
    const status = p.querySelector('.override-status');
    if (e.override) status.textContent = `→ ${e.override} (was ${p.dataset.auto})`;
    else status.textContent = '';
  });
  updateCount();
}
function updateCount() {
  const s = load();
  const n = Object.values(s.packs || {}).filter(p => p.override || (p.comment && p.comment.trim())).length;
  const gn = (s.general_notes || '').trim() ? 1 : 0;
  document.getElementById('count').textContent = `${n} pack feedback ${n===1?'item':'items'}` + (gn ? ' · global notes' : '');
}
function setOverride(name, value) {
  const s = load(); s.packs = s.packs || {};
  s.packs[name] = s.packs[name] || {};
  if (value) s.packs[name].override = value; else delete s.packs[name].override;
  save(s); renderAllFromState();
}
function setComment(name, value) {
  const s = load(); s.packs = s.packs || {};
  s.packs[name] = s.packs[name] || {};
  s.packs[name].comment = value;
  save(s); updateCount();
}
function setNotes(v) { const s = load(); s.general_notes = v; save(s); updateCount(); }
function downloadJSON() {
  const s = load();
  s.exported_at = new Date().toISOString();
  const blob = new Blob([JSON.stringify(s, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'feedback.json';
  a.click();
  toast('feedback.json downloaded');
}
function clearAll() {
  if (!confirm('Clear ALL feedback (overrides, comments, notes)?')) return;
  localStorage.removeItem(KEY);
  renderAllFromState();
  toast('cleared');
}
document.addEventListener('click', e => {
  const b = e.target.closest('.ovr');
  if (!b) return;
  const pack = b.closest('.pack');
  setOverride(pack.dataset.pack, b.dataset.v || null);
});
document.addEventListener('input', e => {
  if (e.target.matches('.comment')) {
    setComment(e.target.closest('.pack').dataset.pack, e.target.value);
  } else if (e.target.id === 'global-notes') {
    setNotes(e.target.value);
  }
});
document.addEventListener('DOMContentLoaded', () => { seedIfEmpty(); renderAllFromState(); });
""".replace("__SEED__", seed_json)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sticker moderation review</title>
<style>{CSS}</style></head>
<body>
<h1>Sticker moderation review</h1>
{summary}
{''.join(rows_html)}
<div class="bar">
  <textarea id="global-notes" placeholder="global rubric tweaks / general feedback for the agent…"></textarea>
  <span class="count" id="count">0 items</span>
  <button class="secondary" onclick="reseed()">reset to snapshot</button>
  <button class="secondary" onclick="clearAll()">clear all</button>
  <button onclick="downloadJSON()">save feedback</button>
</div>
<div class="toast" id="toast"></div>
<script>{js}</script>
</body></html>"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="sample20_review.csv")
    p.add_argument("--cache", default=".tg_cache")
    p.add_argument("--output", default="sample20_review.html")
    p.add_argument("--feedback", default="", help="Path to a feedback.json to embed as initial form state")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    seed = None
    if args.feedback and Path(args.feedback).exists():
        try:
            seed = json.loads(Path(args.feedback).read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"warning: feedback file is not valid JSON: {e}", file=sys.stderr)
    Path(args.output).write_text(render(df, Path(args.cache), seed), encoding="utf-8")
    print(f"wrote {args.output} ({len(df)} packs)" + (f", seeded with {args.feedback}" if seed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
