"""Tests for the upload router (api/routers/upload.py, Track E).

Focus on the security-critical invariant: custom-OCR credentials are passed
to the spawned runner via ``env`` and NEVER as CLI args (argv leaks via
`ps`/`/proc/cmdline`). Also covers validation, size cap, back-pressure, and
spawn-failure → job 'failed'. Neo4j + subprocess are faked.
"""
from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.backend.api.deps import get_driver
from apps.backend.api.routers import upload as upload_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate staging writes and neutralise the job-node DB calls.
    monkeypatch.setattr(upload_mod, "_STAGING_ROOT", tmp_path / "staging")
    monkeypatch.setattr(upload_mod.upload_job, "create_job", lambda *a, **k: k.get("job_id"))
    monkeypatch.setattr(upload_mod.upload_job, "update_job", lambda *a, **k: None)
    monkeypatch.setattr(upload_mod.upload_job, "list_jobs", lambda *a, **k: [])

    app = FastAPI()
    app.include_router(upload_mod.router, prefix="/api/upload")
    app.dependency_overrides[get_driver] = lambda: object()
    return TestClient(app)


class _PopenRecorder:
    def __init__(self) -> None:
        self.cmd = None
        self.env = None

    def __call__(self, cmd, **kwargs):  # noqa: ANN001
        self.cmd = cmd
        self.env = kwargs.get("env")
        return object()  # stand-in Popen handle


def _post(client, **form):
    files = {"file": ("scan.png", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64), "image/png")}
    return client.post("/api/upload", files=files, data=form)


# ─────────────────────────────────────────────────────────────────────────────
# Security invariant: creds in env, not argv
# ─────────────────────────────────────────────────────────────────────────────

def test_custom_credentials_passed_via_env_not_argv(client, monkeypatch):
    rec = _PopenRecorder()
    monkeypatch.setattr(upload_mod.subprocess, "Popen", rec)

    resp = _post(
        client, ocr_model="custom", language="chinese", tier="primary",
        custom_base_url="https://vendor.example/v1",
        custom_api_key="sk-secret-123", custom_model="vendor/ocr-x",
    )
    assert resp.status_code == 200
    # The secret must NOT appear anywhere in the spawned argv.
    assert rec.cmd is not None
    assert all("sk-secret-123" not in str(part) for part in rec.cmd)
    assert all("https://vendor.example/v1" not in str(part) for part in rec.cmd)
    # It MUST be present in the subprocess environment instead.
    assert rec.env["CUSTOM_OCR_API_KEY"] == "sk-secret-123"
    assert rec.env["CUSTOM_OCR_BASE_URL"] == "https://vendor.example/v1"
    assert rec.env["CUSTOM_OCR_MODEL"] == "vendor/ocr-x"


def test_non_custom_upload_sets_no_custom_env(client, monkeypatch):
    rec = _PopenRecorder()
    monkeypatch.setattr(upload_mod.subprocess, "Popen", rec)
    resp = _post(client, ocr_model="qwen")
    assert resp.status_code == 200
    assert "CUSTOM_OCR_API_KEY" not in rec.env


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_ocr_model_rejected(client):
    resp = _post(client, ocr_model="bogus")
    assert resp.status_code == 400


def test_custom_without_credentials_rejected(client):
    resp = _post(client, ocr_model="custom")
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Guards: size cap + back-pressure
# ─────────────────────────────────────────────────────────────────────────────

def test_upload_exceeding_size_cap_is_413(client, monkeypatch):
    monkeypatch.setattr(upload_mod.subprocess, "Popen", _PopenRecorder())
    monkeypatch.setattr(upload_mod, "_MAX_UPLOAD_BYTES", 16)
    big = io.BytesIO(b"x" * 1024)
    resp = client.post("/api/upload", files={"file": ("big.bin", big, "application/octet-stream")},
                       data={"ocr_model": "qwen"})
    assert resp.status_code == 413


def test_back_pressure_returns_429_when_too_many_running(client, monkeypatch):
    monkeypatch.setattr(upload_mod.subprocess, "Popen", _PopenRecorder())
    monkeypatch.setattr(upload_mod, "_MAX_RUNNING_JOBS", 1)
    monkeypatch.setattr(
        upload_mod.upload_job, "list_jobs",
        lambda *a, **k: [{"status": "running"}, {"status": "running"}],
    )
    resp = _post(client, ocr_model="qwen")
    assert resp.status_code == 429


# ─────────────────────────────────────────────────────────────────────────────
# Spawn failure → job marked failed
# ─────────────────────────────────────────────────────────────────────────────

def test_spawn_failure_marks_job_failed(client, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(upload_mod.upload_job, "update_job",
                        lambda driver, job_id, **k: calls.append(k))

    def _boom(cmd, **kwargs):  # noqa: ANN001
        raise OSError("no fork")

    monkeypatch.setattr(upload_mod.subprocess, "Popen", _boom)
    resp = _post(client, ocr_model="qwen")
    assert resp.status_code == 500
    assert any(c.get("status") == "failed" for c in calls)
