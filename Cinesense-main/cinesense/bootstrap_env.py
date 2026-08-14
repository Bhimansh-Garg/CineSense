#!/usr/bin/env python
"""Ensure a local .env exists with a fresh DJANGO_SECRET_KEY (never committed)."""
from __future__ import annotations

from pathlib import Path

try:
    from django.core.management.utils import get_random_secret_key
except ImportError:
    import secrets
    import string

    def get_random_secret_key() -> str:  # type: ignore[misc]
        chars = string.ascii_letters + string.digits + "!@#$%^&*(-_=+)"
        return "".join(secrets.choice(chars) for _ in range(50))


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
EXAMPLE_PATH = BASE_DIR / ".env.example"


def main() -> None:
    if ENV_PATH.exists():
        text = ENV_PATH.read_text(encoding="utf-8-sig")
        if "DJANGO_SECRET_KEY=" in text:
            for line in text.splitlines():
                if line.startswith("DJANGO_SECRET_KEY=") and line.split("=", 1)[1].strip():
                    print(f"{ENV_PATH} already has DJANGO_SECRET_KEY; leaving unchanged.")
                    return
        print(f"{ENV_PATH} exists but DJANGO_SECRET_KEY is empty; writing a new key.")
    elif EXAMPLE_PATH.exists():
        print(f"Creating {ENV_PATH} from {EXAMPLE_PATH.name}")
    else:
        print(f"Creating {ENV_PATH}")

    key = get_random_secret_key()
    ENV_PATH.write_text(
        "# Local secrets — do not commit this file.\n"
        f"DJANGO_SECRET_KEY={key}\n"
        # Local-only convenience; production default in settings is False when unset.
        "DJANGO_DEBUG=True\n"
        "DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1\n",
        encoding="utf-8",
    )
    print(f"Wrote a new DJANGO_SECRET_KEY to {ENV_PATH}")
    print("Restart the Django process so sessions signed with any old key are discarded.")


if __name__ == "__main__":
    main()
