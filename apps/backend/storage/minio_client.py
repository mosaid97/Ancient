"""MinIO (S3-compatible) client + health probe.

Used from Phase 1 onward to store rasterized PDF page images, EPUB image
resources, and the 6-step preprocessing variants from Phase 2 (deskew /
dewarp / illumination / bleed / split / marginalia). Notebook 00 uses this
module only for a connectivity smoke test.

Environment variables (all loaded by the caller via ``python-dotenv`` per
AGENTS.md §2):

- ``MINIO_ENDPOINT`` — host:port (default ``localhost:9000``).
- ``MINIO_ROOT_USER`` — access key (default ``AncientChina``).
- ``MINIO_ROOT_PASSWORD`` — secret key (default ``AncientChina``).
- ``MINIO_SECURE`` — ``"true"`` to use HTTPS (default ``"false"`` for local dev).
- ``MINIO_BUCKET_PAGES`` — default page-image bucket (default ``ancient-pages``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "localhost:9000"
DEFAULT_ACCESS_KEY = "AncientChina"
DEFAULT_SECRET_KEY = "AncientChina"
DEFAULT_BUCKET_PAGES = "ancient-pages"


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_minio_client(
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    secure: bool | None = None,
) -> Minio:
    """Build a MinIO client from environment variables.

    Args:
        endpoint: ``host:port`` override (defaults to ``MINIO_ENDPOINT``).
        access_key: Override (defaults to ``MINIO_ROOT_USER``).
        secret_key: Override (defaults to ``MINIO_ROOT_PASSWORD``).
        secure: HTTPS toggle (defaults to ``MINIO_SECURE`` env, default False).

    Returns:
        A configured :class:`minio.Minio` client.
    """
    endpoint = endpoint or os.getenv("MINIO_ENDPOINT", DEFAULT_ENDPOINT)
    access_key = access_key or os.getenv("MINIO_ROOT_USER", DEFAULT_ACCESS_KEY)
    secret_key = secret_key or os.getenv("MINIO_ROOT_PASSWORD", DEFAULT_SECRET_KEY)
    if secure is None:
        secure = _env_bool("MINIO_SECURE", default=False)
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> bool:
    """Create ``bucket`` if it does not already exist (idempotent).

    Args:
        client: A MinIO client.
        bucket: Bucket name.

    Returns:
        ``True`` if the bucket was just created; ``False`` if it already existed.

    Raises:
        S3Error: For non-recoverable MinIO errors (auth, network, …).
    """
    if client.bucket_exists(bucket):
        return False
    client.make_bucket(bucket)
    logger.info("created MinIO bucket %s", bucket)
    return True


def ping(client: Minio | None = None) -> dict[str, Any]:
    """Health probe: list buckets and ensure the default page bucket exists.

    Returns:
        Dict::

            {
                "ok": bool,
                "endpoint": str,
                "secure": bool,
                "buckets": list[str],
                "default_bucket": str,
                "default_bucket_created": bool,
                "errors": list[str],
            }
    """
    errors: list[str] = []
    endpoint = os.getenv("MINIO_ENDPOINT", DEFAULT_ENDPOINT)
    secure = _env_bool("MINIO_SECURE", default=False)
    default_bucket = os.getenv("MINIO_BUCKET_PAGES", DEFAULT_BUCKET_PAGES)

    buckets: list[str] = []
    default_bucket_created = False

    try:
        client = client or get_minio_client()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"client init failed ({type(exc).__name__}): {exc}")
        return {
            "ok": False,
            "endpoint": endpoint,
            "secure": secure,
            "buckets": [],
            "default_bucket": default_bucket,
            "default_bucket_created": False,
            "errors": errors,
        }

    try:
        buckets = [b.name for b in client.list_buckets()]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"list_buckets failed ({type(exc).__name__}): {exc}")

    if not errors:
        try:
            default_bucket_created = ensure_bucket(client, default_bucket)
            if default_bucket not in buckets and not default_bucket_created:
                buckets.append(default_bucket)
            elif default_bucket_created:
                buckets.append(default_bucket)
        except S3Error as exc:
            errors.append(f"ensure_bucket({default_bucket}) failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"ensure_bucket({default_bucket}) failed "
                f"({type(exc).__name__}): {exc}"
            )

    return {
        "ok": not errors,
        "endpoint": endpoint,
        "secure": secure,
        "buckets": sorted(set(buckets)),
        "default_bucket": default_bucket,
        "default_bucket_created": default_bucket_created,
        "errors": errors,
    }
