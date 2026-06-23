"""Authentication, session management, and zero-knowledge API key vault.

This subpackage owns the single-user (for now) admin auth model: one
Postgres-backed user with an Argon2id-hashed password; password is also
used to derive an in-memory AES-GCM key that encrypts the user's external
API keys (e.g. Silra/OpenAI). The server never persists the derived key —
it lives in :mod:`apps.backend.auth.session` for the lifetime of one
session and is wiped on logout / server restart.

Layout
------
- ``db.py``       SQLAlchemy 2.0 async engine + ``get_session`` dependency.
- ``models.py``   ORM models: User, ApiKey, AuditLog.
- ``crypto.py``   Argon2id KDF + password hashing + AES-GCM helpers.
- ``session.py``  In-process session store mapping cookie -> derived key.
- ``router.py``   ``/api/auth/login | logout | me`` endpoints.
- ``keys.py``     ``/api/keys`` CRUD with re-prompt-for-password on mutate.
"""
