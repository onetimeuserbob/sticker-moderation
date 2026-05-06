"""
One-time interactive Telethon login.

Run this LOCALLY on your machine to create a userbot session file, then
upload that session file to the production host. The file embeds your
account's auth token — treat it like a password.

Usage:
    python userbot_login.py

It reads TELEGRAM_API_ID, TELEGRAM_API_HASH, USERBOT_PHONE, and
USERBOT_SESSION_PATH from .env (or the environment) and walks you
through the SMS / Telegram-app code flow.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient


async def main() -> int:
    load_dotenv()
    api_id_raw = (os.getenv("TELEGRAM_API_ID", "") or "").strip()
    api_hash = (os.getenv("TELEGRAM_API_HASH", "") or "").strip()
    phone = (os.getenv("USERBOT_PHONE", "") or "").strip()
    session_path = Path(
        os.getenv("USERBOT_SESSION_PATH", "userbot.session")
    )
    if not api_id_raw.isdigit() or not api_hash:
        print(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env "
            "(get them from https://my.telegram.org/auth → API development tools).",
            file=sys.stderr,
        )
        return 2
    if not phone:
        print("USERBOT_PHONE must be set in .env (e.g. +447300308971).", file=sys.stderr)
        return 2
    api_id = int(api_id_raw)

    print(f"Using session file: {session_path.resolve()}")
    print(f"Logging in as: {phone}")
    client = TelegramClient(str(session_path), api_id, api_hash)
    await client.connect()
    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already authorized as id={me.id} name={me.first_name!r}. Nothing to do.")
        await client.disconnect()
        return 0

    sent = await client.send_code_request(phone)
    print(f"Code sent via {sent.type.__class__.__name__}.")
    code = input("Enter the login code Telegram just sent you: ").strip()
    try:
        await client.sign_in(phone=phone, code=code)
    except Exception as e:  # noqa: BLE001
        # Could be 2FA. Try once with a password prompt.
        msg = str(e).lower()
        if "password" in msg or "two" in msg or "2fa" in msg:
            password = input("2FA password: ").strip()
            await client.sign_in(password=password)
        else:
            raise
    me = await client.get_me()
    print(f"Logged in as id={me.id} name={me.first_name!r}.")
    print(f"Session saved to: {session_path.resolve()}")
    print(
        "Next: upload this session file to your host's persistent volume, "
        "e.g.\n"
        "  fly ssh sftp shell -a sticker-moderation-bot\n"
        "  put " + str(session_path) + " /data/userbot.session"
    )
    await client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
