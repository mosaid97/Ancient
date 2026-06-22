"""Media router — streams page images out of MinIO for the web UI.

Two variants are exposed because the project stores two image generations
per OCR page:

- ``original``     -> ``PAGE.imageUri`` (the rasterised source page).
- ``preprocessed`` -> ``PAGE.preprocessedImageUri`` (the deskewed/illumination-
  corrected variant whose coordinate space matches ``PAGE.layoutJson`` bboxes).

The HITL "Original Material" column uses ``original``; the interactive
hover overlay uses ``preprocessed`` so the layout boxes line up with the
displayed image.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from neo4j import Driver

from apps.backend.api.deps import get_driver

log = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_BUCKET = "ancient-pages"


def _uri_to_bucket_key(uri: str) -> tuple[str, str]:
    """Parse a stored MinIO URI into ``(bucket, key)``.

    The project stores bare ``<document_id>/page_N.png`` keys (bucket is
    implicit ``ancient-pages``); ``minio://<bucket>/<key>`` is also accepted
    for forward-compatibility.
    """
    if uri.startswith("minio://"):
        uri = uri.removeprefix("minio://")
        bucket, _, key = uri.partition("/")
        return bucket, key
    return _DEFAULT_BUCKET, uri


# Whitelist the only two PAGE properties that may be selected dynamically.
# All other dynamic Cypher property interpolation is forbidden — see
# AGENTS.md §4 (parametrized Cypher only).
_VARIANT_TO_PROP: dict[str, str] = {
    "original": "imageUri",
    "preprocessed": "preprocessedImageUri",
}

_CONTENT_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


def _guess_content_type(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    return _CONTENT_TYPES.get(ext, "image/png")


@router.get("/image/{page_id}")
async def page_image(
    page_id: str,
    variant: str = Query("original", description="original | preprocessed"),
    driver: Driver = Depends(get_driver),
) -> Response:
    """Stream a page image from MinIO.

    Args:
        page_id: The Neo4j ``PAGE.id``.
        variant: ``'original'`` (default) or ``'preprocessed'``.
    """
    prop = _VARIANT_TO_PROP.get(variant)
    if prop is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant {variant!r}; expected one of {sorted(_VARIANT_TO_PROP)}",
        )
    # `prop` is now guaranteed to be one of two hard-coded property names
    # from the whitelist above — safe to interpolate. Pure-property
    # references can't be parametrised in Cypher; the whitelist is what
    # keeps this injection-proof.
    with driver.session() as s:
        row = s.run(
            f"MATCH (p:PAGE {{id: $pid}}) RETURN p.{prop} AS uri, p.imageUri AS fallback",
            pid=page_id,
        ).single()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Page {page_id!r} not found")

    uri = row["uri"] or row["fallback"]
    if not uri:
        raise HTTPException(status_code=404, detail=f"No {variant} image for page {page_id!r}")

    bucket, key = _uri_to_bucket_key(uri)
    try:
        from apps.backend.storage import get_minio_client

        mc = get_minio_client()
        resp = mc.get_object(bucket, key)
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
    except Exception as exc:  # noqa: BLE001
        log.warning("MinIO fetch failed for %s/%s: %s", bucket, key, exc)
        raise HTTPException(status_code=502, detail=f"image fetch failed: {exc}")

    return Response(
        content=data,
        media_type=_guess_content_type(key),
        headers={"Cache-Control": "public, max-age=3600"},
    )
