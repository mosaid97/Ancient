"""Bootstrap the single admin user.

Usage::

    uv run python scripts/setup_admin.py

The script:

1. Reads ``DATABASE_URL`` from the environment (or .env).
2. Refuses to run if a user already exists — single-tenant means
   exactly one row in ``users``. Pass ``--force`` to overwrite the
   existing admin's password (useful for password rotation; this
   *also* invalidates every encrypted API key because the KDF salt
   is replaced).
3. Prompts for username + password (twice for confirmation).
4. Generates an Argon2id password hash + a fresh 32-byte KDF salt.
5. Inserts the user row.

No HTTP endpoint exists for signup — that is intentional. Running this
script is the only way to create the first user.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import sys
from pathlib import Path

# Make `apps.backend...` importable from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from sqlalchemy import select  # noqa: E402

from apps.backend.auth.crypto import hash_password, new_kdf_salt  # noqa: E402
from apps.backend.auth.db import get_session_factory  # noqa: E402
from apps.backend.auth.models import User  # noqa: E402


MIN_PASSWORD_CHARS = 12


def _prompt_password() -> str:
    while True:
        pw = getpass.getpass("Password: ")
        if len(pw) < MIN_PASSWORD_CHARS:
            print(f"  ✗ Password must be at least {MIN_PASSWORD_CHARS} characters.")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if pw != confirm:
            print("  ✗ Passwords do not match. Try again.")
            continue
        return pw


async def _bootstrap(force: bool, username_arg: str | None) -> int:
    Session = get_session_factory()
    async with Session() as session:
        existing = (await session.execute(select(User))).scalars().all()

        if existing and not force:
            print(
                f"Refusing to bootstrap: {len(existing)} user(s) already exist.\n"
                "Pass --force to reset the existing admin's password "
                "(this WILL invalidate every encrypted API key)."
            )
            return 1

        if existing and force:
            print("\n⚠  --force will rotate the admin's password and KDF salt.")
            print("   Every API key in the vault becomes unrecoverable.")
            ans = input("Type 'rotate' to continue: ").strip()
            if ans != "rotate":
                print("Aborted.")
                return 1

        if username_arg:
            username = username_arg.strip()
        else:
            username = input("Username: ").strip()
        if not username:
            print("Username required.")
            return 1

        password = _prompt_password()
        salt = new_kdf_salt()
        pw_hash = hash_password(password)

        if existing:
            user = existing[0]
            user.username = username
            user.password_hash = pw_hash
            user.kdf_salt = salt
            # Drop any existing API keys — the new salt invalidates their
            # encryption key, so the rows would be unrecoverable anyway.
            user.api_keys.clear()
        else:
            user = User(
                username=username,
                password_hash=pw_hash,
                kdf_salt=salt,
                is_admin=True,
                is_active=True,
            )
            session.add(user)

        await session.commit()

    print(f"\n✓ Admin user {username!r} ready.")
    print(f"  Login at http://localhost:8000/ and add your API keys from the UI.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap the single admin user.")
    ap.add_argument("--force", action="store_true",
                    help="Rotate the existing admin's password (drops API keys).")
    ap.add_argument("--username", help="Skip the interactive username prompt.")
    args = ap.parse_args()

    rc = asyncio.run(_bootstrap(force=args.force, username_arg=args.username))
    sys.exit(rc)


if __name__ == "__main__":
    main()
