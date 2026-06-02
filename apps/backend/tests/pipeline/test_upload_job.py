"""Tests for the UPLOAD_JOB state machine (pipeline/upload_job.py, Track E).

Neo4j is faked with a recording driver so we can assert the exact Cypher
params each lifecycle helper sends (mock-Neo4j convention, no live DB).
"""
from __future__ import annotations

from apps.backend.pipeline import upload_job


# ─────────────────────────────────────────────────────────────────────────────
# Recording fake driver
# ─────────────────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def consume(self):
        return None

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, recorder: list[tuple[str, dict]], rows: list[dict]) -> None:
        self._recorder = recorder
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):  # noqa: ANN003
        self._recorder.append((query, params))
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._rows = rows or []

    def session(self):
        return _FakeSession(self.calls, self._rows)


# ─────────────────────────────────────────────────────────────────────────────
# create_job
# ─────────────────────────────────────────────────────────────────────────────

def test_create_job_sends_all_fields():
    driver = _FakeDriver()
    job_id = upload_job.create_job(
        driver, job_id="job-1", filename="x.pdf", doc_type="book",
        ocr_required=True, ocr_model="qwen", language="chinese", tier="primary",
    )
    assert job_id == "job-1"
    assert len(driver.calls) == 1
    query, params = driver.calls[0]
    assert "MERGE (j:UPLOAD_JOB" in query
    assert params["id"] == "job-1"
    assert params["filename"] == "x.pdf"
    assert params["doc_type"] == "book"
    assert params["ocr_required"] is True
    assert params["ocr_model"] == "qwen"
    assert params["language"] == "chinese"
    assert params["tier"] == "primary"
    assert params["document_id"] is None


def test_create_job_with_document_id():
    driver = _FakeDriver()
    upload_job.create_job(
        driver, job_id="job-2", filename="y.png", doc_type="image",
        ocr_required=False, ocr_model="custom", language="japanese",
        tier="secondary", document_id="doc-abc",
    )
    _, params = driver.calls[0]
    assert params["document_id"] == "doc-abc"
    assert params["ocr_required"] is False


# ─────────────────────────────────────────────────────────────────────────────
# update_job — coalesce semantics (only non-None fields change)
# ─────────────────────────────────────────────────────────────────────────────

def test_update_job_passes_only_given_fields_as_none_elsewhere():
    driver = _FakeDriver()
    upload_job.update_job(driver, "job-1", status="running", stage="ocr", pct=42.0)
    query, params = driver.calls[0]
    assert "coalesce($status, j.status)" in query
    assert params["status"] == "running"
    assert params["stage"] == "ocr"
    assert params["pct"] == 42.0
    # untouched fields are None so Cypher coalesce keeps the prior value
    assert params["document_id"] is None
    assert params["error"] is None


def test_update_job_can_set_failed_with_error():
    driver = _FakeDriver()
    upload_job.update_job(driver, "job-9", status="failed", error="ingest blew up")
    _, params = driver.calls[0]
    assert params["status"] == "failed"
    assert params["error"] == "ingest blew up"


# ─────────────────────────────────────────────────────────────────────────────
# get_job / list_jobs
# ─────────────────────────────────────────────────────────────────────────────

def test_get_job_returns_dict_when_found():
    row = {"id": "job-1", "status": "awaiting_review", "stage": "layout"}
    driver = _FakeDriver(rows=[row])
    out = upload_job.get_job(driver, "job-1")
    assert out == row


def test_get_job_returns_none_when_missing():
    driver = _FakeDriver(rows=[])
    assert upload_job.get_job(driver, "nope") is None


def test_list_jobs_returns_all_rows_and_passes_limit():
    rows = [{"id": "a"}, {"id": "b"}]
    driver = _FakeDriver(rows=rows)
    out = upload_job.list_jobs(driver, limit=25)
    assert out == rows
    _, params = driver.calls[0]
    assert params["limit"] == 25
