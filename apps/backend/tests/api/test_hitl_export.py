"""Tests for the deterministic HITL export helpers (api/routers/hitl.py, Track E).

Covers the Word/PDF byte generators and the RFC 5987 Content-Disposition
encoding that fixes the CJK-filename ``latin-1`` crash.
"""
from __future__ import annotations

from urllib.parse import quote

from apps.backend.api.routers.hitl import _export_docx, _export_pdf


# ─────────────────────────────────────────────────────────────────────────────
# _export_docx
# ─────────────────────────────────────────────────────────────────────────────

def test_export_docx_is_valid_ooxml_zip():
    data = _export_docx("唐律疏議", 0, "名例律第一\n十惡尤切")
    # .docx is a ZIP container — magic bytes "PK\x03\x04".
    assert data[:4] == b"PK\x03\x04"
    assert len(data) > 1000


def test_export_docx_handles_empty_text():
    data = _export_docx("title", None, "")
    assert data[:2] == b"PK"


# ─────────────────────────────────────────────────────────────────────────────
# _export_pdf
# ─────────────────────────────────────────────────────────────────────────────

def test_export_pdf_is_valid_pdf_with_cjk():
    data = _export_pdf("唐律疏議", 3, "名例律第一\n十惡尤切不容首免")
    assert data[:5] == b"%PDF-"
    assert b"%%EOF" in data[-1024:]


def test_export_pdf_escapes_markup_chars():
    # Should not raise even with reportlab-significant chars.
    data = _export_pdf("a & b <tag>", 1, "x < y & z > w")
    assert data[:5] == b"%PDF-"


# ─────────────────────────────────────────────────────────────────────────────
# Content-Disposition encoding (mirrors the endpoint logic)
# ─────────────────────────────────────────────────────────────────────────────

def _build_disposition(page_id: str, ext: str) -> str:
    ascii_name = page_id.encode("ascii", "ignore").decode("ascii").strip("._") or "page"
    quoted = quote(f"{page_id}.{ext}", safe="")
    return f'attachment; filename="{ascii_name}.{ext}"; filename*=UTF-8\'\'{quoted}'


def test_disposition_is_latin1_encodable_for_cjk_page_id():
    page_id = "顾江龙_两晋南北朝__11e829b009::p00000"
    disposition = _build_disposition(page_id, "docx")
    # The whole header value must survive latin-1 encoding (starlette requirement).
    disposition.encode("latin-1")  # must not raise
    assert "filename*=UTF-8''" in disposition


def test_disposition_falls_back_to_page_when_no_ascii():
    page_id = "純中文標題"  # no ASCII chars at all
    disposition = _build_disposition(page_id, "pdf")
    disposition.encode("latin-1")
    assert 'filename="page.pdf"' in disposition
