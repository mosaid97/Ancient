"""
Watchdog for BGE reranker model download.

Shard 2 is being downloaded by an orphaned aria2c process (PID 33815).
This script:
  1. Waits for aria2c to finish shard 2
  2. Renames the .incomplete file to the final blob hash
  3. Downloads shard 3 via aria2c using the same modelscope.cn mirror
  4. Renames shard 3 .incomplete → final blob hash
  5. Verifies the model loads correctly
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BLOBS = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-gemma/blobs"
MODELSCOPE_BASE = "https://modelscope.cn/models/BAAI/bge-reranker-v2-gemma/resolve/master"

SHARD2_HASH = "d6aef5ed60f1410600a17b63de134b7ddef3a8f611e2ed1d62225ac4dacc13df"
SHARD3_HASH = "64a1c520e62df78ef25b4f066b746447dc4d58d0d3d37efa419c8ec8462e133b"
# Exact remote file sizes confirmed via Content-Range HEAD request to modelscope.cn.
# Shard 3 is only 128 MB — the existing 128 MB blob is already complete.
SHARD2_EXPECTED_BYTES = 4_978_830_584
SHARD3_EXPECTED_BYTES = 134_242_760
SHARD3_MIN_BYTES = SHARD3_EXPECTED_BYTES - 1  # accept anything >= full size

ARIA2C_PID = 33815
ARIA2C_FLAGS = [
    "--no-proxy",
    "--split=16",
    "--max-connection-per-server=16",
    "--min-split-size=50M",
    "--retry-wait=3",
    "--max-tries=10",
]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_for_aria2c(pid: int, incomplete: Path) -> None:
    print(f"Waiting for aria2c PID {pid} to finish shard 2 …", flush=True)
    last_size = 0
    stall_count = 0
    while pid_alive(pid):
        size = incomplete.stat().st_size if incomplete.exists() else 0
        mb = size / 1024**2
        speed = (size - last_size) / 1024**2
        print(f"  {mb:.0f} MB  (+{speed:.1f} MB/s)", flush=True)
        if size == last_size:
            stall_count += 1
            if stall_count > 10:
                print("  ⚠ download stalled >50 s — aria2c may have hung", flush=True)
        else:
            stall_count = 0
        last_size = size
        time.sleep(5)
    print("  aria2c exited.", flush=True)


def rename_incomplete(blob_hash: str) -> Path:
    incomplete = BLOBS / f"{blob_hash}.incomplete"
    aria2_ctrl = BLOBS / f"{blob_hash}.incomplete.aria2"
    target = BLOBS / blob_hash

    if target.exists():
        print(f"  {blob_hash[:16]}… already finalised ({target.stat().st_size/1024**2:.0f} MB)", flush=True)
        return target

    if not incomplete.exists():
        raise FileNotFoundError(f"Neither .incomplete nor final blob found for {blob_hash[:16]}…")

    size_mb = incomplete.stat().st_size / 1024**2
    print(f"  Renaming {blob_hash[:16]}….incomplete ({size_mb:.0f} MB) → final blob", flush=True)
    incomplete.rename(target)
    if aria2_ctrl.exists():
        aria2_ctrl.unlink()
    return target


def download_shard(filename: str, blob_hash: str, min_bytes: int = 0) -> None:
    target = BLOBS / blob_hash
    incomplete = BLOBS / f"{blob_hash}.incomplete"

    if target.exists():
        size = target.stat().st_size
        if size >= min_bytes:
            print(f"  {filename} already complete ({size/1024**2:.0f} MB) — skipping", flush=True)
            return
        # File exists but is smaller than expected — it's a partial from a previous
        # interrupted download that was never cleaned up.  Move it to .incomplete
        # so aria2c can resume from where it left off.
        print(f"  {filename} blob too small ({size/1024**2:.0f} MB < {min_bytes/1024**2:.0f} MB min) — treating as incomplete", flush=True)
        target.rename(incomplete)

    url = f"{MODELSCOPE_BASE}/{filename}"
    resume_flag = ["--continue=true"] if incomplete.exists() else []
    start_size = incomplete.stat().st_size if incomplete.exists() else 0
    print(f"  Downloading {filename} from modelscope.cn (resume from {start_size/1024**2:.0f} MB) …", flush=True)

    cmd = [
        "aria2c",
        *ARIA2C_FLAGS,
        *resume_flag,
        f"--out={blob_hash}.incomplete",
        f"--dir={BLOBS}",
        url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Stream progress
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line.strip():
            print(f"  aria2c: {line.rstrip()}", flush=True)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"aria2c failed with exit code {rc}")
    print(f"  {filename} download complete.", flush=True)


def verify_model() -> bool:
    print("\nVerifying model loads …", flush=True)
    code = (
        "from FlagEmbedding import FlagReranker; "
        "r = FlagReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True); "
        "score = r.compute_score([['query', 'passage']]); "
        "print('Score:', score); "
        "print('BGE reranker OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        print("  ✓", result.stdout.strip(), flush=True)
        return True
    else:
        print("  ✗ Model verification FAILED:", flush=True)
        print(result.stderr[-2000:], flush=True)
        return False


def main() -> None:
    print("=== BGE Reranker Download Watchdog ===", flush=True)

    # ── Shard 2 ──────────────────────────────────────────────────────────────
    shard2_target = BLOBS / SHARD2_HASH
    shard2_incomplete = BLOBS / f"{SHARD2_HASH}.incomplete"

    if shard2_target.exists():
        print(f"Shard 2 already complete ({shard2_target.stat().st_size/1024**2:.0f} MB)", flush=True)
    else:
        if pid_alive(ARIA2C_PID):
            wait_for_aria2c(ARIA2C_PID, shard2_incomplete)
        elif shard2_incomplete.exists():
            print(f"aria2c (PID {ARIA2C_PID}) already gone; incomplete blob at {shard2_incomplete.stat().st_size/1024**2:.0f} MB", flush=True)
            # Need to resume
            download_shard("model-00002-of-00003.safetensors", SHARD2_HASH)
        rename_incomplete(SHARD2_HASH)
        print(f"  Shard 2 final size: {(BLOBS / SHARD2_HASH).stat().st_size/1024**2:.0f} MB", flush=True)

    # ── Shard 3 ──────────────────────────────────────────────────────────────
    print("\nStarting shard 3 download …", flush=True)
    download_shard("model-00003-of-00003.safetensors", SHARD3_HASH, min_bytes=SHARD3_MIN_BYTES)
    rename_incomplete(SHARD3_HASH)
    print(f"  Shard 3 final size: {(BLOBS / SHARD3_HASH).stat().st_size/1024**2:.0f} MB", flush=True)

    # ── Final verification ────────────────────────────────────────────────────
    ok = verify_model()
    if ok:
        print("\n✓ BGE reranker model fully downloaded and verified!", flush=True)
    else:
        print("\n✗ Model verification failed — check logs above.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
