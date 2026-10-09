"""Encryption for the API keys kept in the database.

The instance's own keys (``instance_settings``) and the per-account and
per-organisation ones (``credentials``) are stored encrypted with Fernet (AES-128
in CBC mode with an HMAC-SHA256 tag). The key is kept outside the database, so a
leaked dump or backup of the database does not leak the API keys in it.

Where the key comes from:

* ``IQUANA_SECRETS_KEY``: a Fernet key, or several separated by commas to rotate
  them. The first encrypts; all of them decrypt.
* otherwise the key file ``IQUANA_SECRETS_KEY_FILE`` (default
  ``<DATA_DIR>/secrets.key``), generated on first use. A fresh install then works
  with no setup, and the key still lives outside the database.

**Losing the key loses the stored API keys.** They can no longer be decrypted, read
back as unset, and have to be entered again. Back the key up, separately from the
database.

Stored values carry a ``fernet:`` prefix. A value without it is plaintext from
before encryption: it is still read, and `settings.encrypt_stored_secrets`
rewrites it when the backend starts.
"""
from __future__ import annotations

import os
from functools import lru_cache
from logging import getLogger

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from config import SECRETS_KEY_FILE

logger = getLogger(__name__)

PREFIX = "fernet:"


def _key_file() -> str:
    # Read at call time, so tests can point it somewhere else.
    return os.getenv("IQUANA_SECRETS_KEY_FILE") or SECRETS_KEY_FILE


def _read_or_create_key_file(path: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    key = Fernet.generate_key()
    try:
        # Exclusive create: two workers starting together must not each write a
        # different key, which would leave one of them unable to read the other's.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(path, "rb") as handle:
            return handle.read().strip()
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(key)
    logger.warning("Generated a new key for stored API keys at %s. Back it up separately "
                   "from the database: without it, stored API keys cannot be read.", path)
    return key


@lru_cache(maxsize=1)
def _fernet() -> MultiFernet:
    configured = (os.getenv("IQUANA_SECRETS_KEY") or "").strip()
    if configured:
        keys = [key.strip().encode() for key in configured.split(",") if key.strip()]
    else:
        keys = [_read_or_create_key_file(_key_file())]
    return MultiFernet([Fernet(key) for key in keys])


def reset_key_cache() -> None:
    """Forget the loaded key, e.g. after the environment changed (tests)."""
    _fernet.cache_clear()


def is_encrypted(stored: str | None) -> bool:
    return bool(stored) and stored.startswith(PREFIX)


def encrypt(plaintext: str) -> str:
    return PREFIX + _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(stored: str | None) -> str | None:
    """The plaintext of a stored secret, or None if it is unset or unreadable.

    A value without the prefix is returned as it is: plaintext from before stored
    secrets were encrypted. One that cannot be decrypted -- the key changed or was
    lost -- reads as unset, and is logged, rather than failing the request that
    needed it.
    """
    if stored is None:
        return None
    if not stored.startswith(PREFIX):
        return stored
    try:
        return _fernet().decrypt(stored[len(PREFIX):].encode("ascii")).decode("utf-8")
    except InvalidToken:
        logger.error("A stored API key cannot be decrypted with the configured key and is "
                     "treated as unset. Was IQUANA_SECRETS_KEY changed, or the key file lost?")
        return None


def hint(plaintext: str | None) -> str | None:
    """Enough of a secret to tell two apart, not enough to use one."""
    if not plaintext:
        return None
    return f"…{plaintext[-4:]}" if len(plaintext) > 4 else "…"
