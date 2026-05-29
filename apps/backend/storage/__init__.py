"""Object storage clients (MinIO / S3-compatible)."""

from apps.backend.storage.minio_client import (
    ensure_bucket,
    get_minio_client,
    ping,
)

__all__ = ["ensure_bucket", "get_minio_client", "ping"]
