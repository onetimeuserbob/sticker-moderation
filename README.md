# Sticker Pad — moderation bot

Live moderation bot for [Sticker Pad](https://t.me/sticker_bot) sticker
pack applications. It watches a chat for new pack submissions, reviews
each one with Claude Sonnet 4.5 (using a long, well-tuned policy
prompt), and replies with an ✅ APPROVE / ❌ REJECT / 🟡 NEEDS REVIEW
suggestion plus a one-line reason.

The bot is **owner-locked**: only one Telegram user (`OWNER_USER_ID`) can
DM it, add it to chats, edit rules, or correct its decisions. Any chat
the owner adds it to is auto-whitelisted; anyone else trying to add it
causes the bot to leave immediately and DM the owner.

## Pieces

- `review_bot.py` — the live Telegram bot (long-polling, async, owner-locked).
- `rules_store.py` — JSON-backed amendment store; rules layer on top of the base policy.
- `moderate_packs.py` — offline batch pipeline (Claude + OpenAI omni-moderation).
  The bot reuses the same `CLAUDE_PROMPT` and review function so it stays in lockstep.
- `tg_ingest.py` — fetches per-sticker thumbnails from `t.me/addstickers/<slug>` URLs.

## Commands

| Command | Description |
|---|---|
| `/start`, `/help` | Quick intro |
| `/rules` | Show the policy summary + active operator amendments |
| `/listrules` | Compact list of just the amendments |
| `/addrule <text>` | Add a new amendment (Claude rewrites it into policy voice) |
| `/delrule <id>` | Deactivate an amendment |
| `/chats` | Whitelisted group chats |
| `/leavechat <chat_id>` | Un-whitelist + leave a group |

**Disagreement learning:** reply to any verdict message with the correct
decision (e.g. `approve, this is just satire`) and the bot calls Claude
to decide whether your correction implies a generalizable rule change —
add, amend, remove, or noop — and confirms what it did.

## Deploy on fly.io

1. Get the keys ready:
   - `REVIEW_BOT_TOKEN` — the bot itself (from @BotFather).
   - `TELEGRAM_BOT_TOKEN` — a second bot used to fetch sticker thumbnails (the offline pipeline's bot is fine; can be the same as `REVIEW_BOT_TOKEN`).
   - `ANTHROPIC_API_KEY` — Claude Sonnet 4.5 access.
   - `OWNER_USER_ID` — your numeric Telegram user id (the only allowed user).

2. Initial fly setup (run in this directory):

   ```bash
   fly launch --no-deploy --copy-config --name sticker-moderation-bot
   fly volumes create review_bot_data --size 1 --region ams   # 1 GB is plenty
   fly secrets set \
     ANTHROPIC_API_KEY=sk-ant-... \
     REVIEW_BOT_TOKEN=123:ABC... \
     TELEGRAM_BOT_TOKEN=456:DEF... \
     OWNER_USER_ID=2580397 \
     SOURCE_BOT_USERNAME=sticker_bot
   fly deploy
   ```

3. Two one-time Telegram setup steps:
   - DM the bot once with `/start` (so it can DM you back with whitelist confirmations).
   - In @BotFather: `/setprivacy` → pick the bot → **Disable** (so it can read group messages, not just commands).

4. Add the bot to the chat where `@sticker_bot` posts applications. You'll get a DM:
   `✅ Whitelisted chat <id> — <title>`. The bot will start replying to applications.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
python review_bot.py
```

## Persisted state

In production these live on the fly volume at `/data`. Locally they live
next to the source. Override with env vars `RULES_STORE_PATH`,
`ALLOWLIST_PATH`, `TG_CACHE_DIR`.

| File | What |
|---|---|
| `rules_store.json` | Operator amendments (added via `/addrule` or learned from corrections) |
| `allowed_chats.json` | Auto-managed group chat allowlist |
| `tg_cache/` | Per-pack sticker thumbnail cache (~1 MB per pack reviewed) |
