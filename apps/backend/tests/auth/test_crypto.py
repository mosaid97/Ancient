"""Unit tests for the auth crypto primitives.

Covers password hashing, key derivation, AES-GCM round-trip, and the
fingerprint/last4 helpers used by the API-key vault.
"""
from __future__ import annotations

import secrets

import pytest

from apps.backend.auth.crypto import (
    AES_KEY_LEN,
    decrypt_secret,
    derive_key,
    encrypt_secret,
    fingerprint,
    hash_password,
    last4,
    needs_rehash,
    new_kdf_salt,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_verify_roundtrip(self):
        pw = "correct horse battery staple"
        h = hash_password(pw)
        assert verify_password(pw, h) is True

    def test_wrong_password_rejected(self):
        h = hash_password("secret-one")
        assert verify_password("secret-two", h) is False

    def test_hash_is_salted(self):
        # Two hashes of the same password must differ — Argon2 picks a
        # random salt per hash.
        a = hash_password("same-pw")
        b = hash_password("same-pw")
        assert a != b

    def test_hash_does_not_match_current_params_after_param_bump(self):
        """needs_rehash should be False right after hashing with current params."""
        h = hash_password("anything")
        assert needs_rehash(h) is False


class TestKDF:
    def test_kdf_is_deterministic(self):
        salt = new_kdf_salt()
        k1 = derive_key("pw", salt)
        k2 = derive_key("pw", salt)
        assert k1 == k2
        assert len(k1) == AES_KEY_LEN

    def test_kdf_changes_with_password(self):
        salt = new_kdf_salt()
        assert derive_key("pw-a", salt) != derive_key("pw-b", salt)

    def test_kdf_changes_with_salt(self):
        assert derive_key("pw", new_kdf_salt()) != derive_key("pw", new_kdf_salt())

    def test_kdf_rejects_short_salt(self):
        with pytest.raises(ValueError):
            derive_key("pw", b"\x00" * 8)


class TestAesGcm:
    def test_encrypt_decrypt_roundtrip(self):
        key = secrets.token_bytes(AES_KEY_LEN)
        plaintext = "sk-very-secret-token-abcd"
        nonce, ct = encrypt_secret(key, plaintext)
        assert decrypt_secret(key, nonce, ct) == plaintext

    def test_wrong_key_fails(self):
        from cryptography.exceptions import InvalidTag

        key = secrets.token_bytes(AES_KEY_LEN)
        nonce, ct = encrypt_secret(key, "secret")
        bad_key = secrets.token_bytes(AES_KEY_LEN)
        with pytest.raises(InvalidTag):
            decrypt_secret(bad_key, nonce, ct)

    def test_tampered_ciphertext_fails(self):
        from cryptography.exceptions import InvalidTag

        key = secrets.token_bytes(AES_KEY_LEN)
        nonce, ct = encrypt_secret(key, "secret")
        tampered = bytes([ct[0] ^ 1]) + ct[1:]
        with pytest.raises(InvalidTag):
            decrypt_secret(key, nonce, tampered)

    def test_unique_nonce_per_encryption(self):
        # Same key + same plaintext must still produce distinct ciphertexts
        # because the nonce is random — a property AES-GCM needs to stay safe.
        key = secrets.token_bytes(AES_KEY_LEN)
        n1, ct1 = encrypt_secret(key, "secret")
        n2, ct2 = encrypt_secret(key, "secret")
        assert n1 != n2
        assert ct1 != ct2


class TestHelpers:
    def test_fingerprint_stable(self):
        assert fingerprint("sk-abc") == fingerprint("sk-abc")

    def test_fingerprint_distinct(self):
        assert fingerprint("sk-abc") != fingerprint("sk-abd")

    def test_last4(self):
        assert last4("sk-very-secret-WXYZ") == "WXYZ"
        assert last4("ab") == "ab"   # short keys returned verbatim
