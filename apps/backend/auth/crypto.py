"""Password hashing, key derivation, and AES-GCM helpers.

Choices
-------
- **Argon2id** for password hashing (via ``argon2-cffi``) AND for the
  password-derived encryption key. Argon2id is the OWASP-recommended
  memory-hard KDF and resists GPU / ASIC attacks better than PBKDF2.
- **AES-256-GCM** for the API-key vault: authenticated encryption with
  a 12-byte random nonce per encryption, ciphertext includes the tag.
- The **derived encryption key never touches disk**. It is rebuilt at
  login from (password, user.kdf_salt) and held in the session store
  in process memory only.

The Argon2 parameters here are conservative for a single-user research
tool. For multi-user / public deployment, raise ``ARGON2_MEMORY_KIB`` and
``ARGON2_TIME_COST`` and re-benchmark for ~250-500ms per hash on the
target server.
"""
from __future__ import annotations

import hashlib
import os
import secrets

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Argon2id params. ``hash_len`` and ``salt_len`` are the lengths embedded
# in the password-hash record; the KDF (`derive_key`) uses its own salt
# stored on the user row so it survives password-hash rehash cycles.
ARGON2_TIME_COST = 3
ARGON2_MEMORY_KIB = 64 * 1024  # 64 MiB
ARGON2_PARALLELISM = 4
ARGON2_HASH_LEN = 32
ARGON2_SALT_LEN = 16

# Length of the AES-256-GCM key in bytes.
AES_KEY_LEN = 32
AES_NONCE_LEN = 12

_HASHER = PasswordHasher(
    type=Type.ID,
    time_cost=ARGON2_TIME_COST,
    memory_cost=ARGON2_MEMORY_KIB,
    parallelism=ARGON2_PARALLELISM,
    hash_len=ARGON2_HASH_LEN,
    salt_len=ARGON2_SALT_LEN,
)


# ── Password hashing ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Return an Argon2id-encoded hash string (includes salt + params)."""
    return _HASHER.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time verify of ``password`` against an Argon2 hash."""
    try:
        _HASHER.verify(hashed, password)
        return True
    except VerifyMismatchError:
        return False


def needs_rehash(hashed: str) -> bool:
    """True when the hash's params are weaker than the current settings."""
    return _HASHER.check_needs_rehash(hashed)


# ── KDF for AES key (separate from the password hash) ────────────────────────

def new_kdf_salt() -> bytes:
    """Random 32-byte salt for the password-to-AES KDF."""
    return secrets.token_bytes(32)


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte AES key from (password, salt) via Argon2id.

    Uses the low-level ``argon2.low_level.hash_secret_raw`` (not the
    PasswordHasher) so we control the salt and get raw bytes back.
    """
    from argon2.low_level import Type as LowType
    from argon2.low_level import hash_secret_raw

    if len(salt) < 16:
        raise ValueError("kdf salt must be at least 16 bytes")
    return hash_secret_raw(
        secret=password.encode("utf-8"),
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_KIB,
        parallelism=ARGON2_PARALLELISM,
        hash_len=AES_KEY_LEN,
        type=LowType.ID,
    )


# ── AES-GCM encrypt / decrypt ────────────────────────────────────────────────

def encrypt_secret(key: bytes, plaintext: str) -> tuple[bytes, bytes]:
    """Return ``(nonce, ciphertext)``. The ciphertext bundles the auth tag."""
    if len(key) != AES_KEY_LEN:
        raise ValueError(f"key must be {AES_KEY_LEN} bytes, got {len(key)}")
    nonce = os.urandom(AES_NONCE_LEN)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), associated_data=None)
    return nonce, ct


def decrypt_secret(key: bytes, nonce: bytes, ciphertext: bytes) -> str:
    """Reverse of :func:`encrypt_secret`. Raises ``InvalidTag`` on tamper."""
    if len(key) != AES_KEY_LEN:
        raise ValueError(f"key must be {AES_KEY_LEN} bytes, got {len(key)}")
    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    return pt.decode("utf-8")


# ── Fingerprinting (one-way, NOT a password equivalent) ──────────────────────

def fingerprint(plaintext: str) -> str:
    """SHA-256 hex digest of the plaintext key. Used for dedupe only."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def last4(plaintext: str) -> str:
    """Return the last 4 visible chars (for UI display of a hidden key)."""
    s = plaintext.strip()
    return s[-4:] if len(s) >= 4 else s
