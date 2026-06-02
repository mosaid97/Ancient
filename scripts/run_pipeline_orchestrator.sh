#!/usr/bin/env bash
# Full pipeline orchestrator — runs steps A2→B2→B3→D2 in dependency order.
#
# Step 1 (now):  embed pending+failed chunks (no --recompute)
# Step 2 (now):  refresh corpus audit
# Step 3 (now):  citation linker B2 (corpus-wide)
# Step 4 (wait): wait for translation run2 (PID 61092) to finish
# Step 5 (then): embed --recompute all 40,239 chunks with textCanonical
# Step 6 (then): re-run community detection B3 (with frequency filter)
# Step 7 (then): eval harness D2 — first real NDCG@10 / faithfulness numbers
#
# Run with:
#   caffeinate -dimsu bash scripts/run_pipeline_orchestrator.sh 2>&1 | tee logs/orchestrator.log

set -euo pipefail
cd "$(dirname "$0")/.."

TRANSLATION_PID=61092
LOG_DIR=logs
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: Embed pending + failed ───────────────────────────────────────────
log "=== STEP 1: Embed pending+failed chunks ==="
uv run python scripts/run_embedding.py \
  --log-file "$LOG_DIR/embedding_run3.log"
log "Step 1 done."

# ── Step 2: Corpus audit ─────────────────────────────────────────────────────
log "=== STEP 2: Corpus audit ==="
uv run python scripts/audit_corpus_state.py 2>&1 | tee "$LOG_DIR/audit_post_embed.log"
log "Step 2 done."

# ── Step 3: Citation linker B2 ───────────────────────────────────────────────
log "=== STEP 3: Citation linker B2 (corpus-wide) ==="
uv run python scripts/run_citation_linker.py \
  --log-file "$LOG_DIR/citation_linker2.log"
log "Step 3 done."

# ── Step 4: Wait for translation run2 ────────────────────────────────────────
log "=== STEP 4: Waiting for translation (PID $TRANSLATION_PID) ==="
if ps -p "$TRANSLATION_PID" > /dev/null 2>&1; then
    log "Translation still running — polling every 5 min..."
    while ps -p "$TRANSLATION_PID" > /dev/null 2>&1; do
        sleep 300
    done
    log "Translation process finished."
else
    log "Translation process already done (PID $TRANSLATION_PID not found)."
fi

# ── Step 5: Full recompute embed ─────────────────────────────────────────────
log "=== STEP 5: Recompute embed (all 40,239 chunks with textCanonical) ==="
uv run python scripts/run_embedding.py \
  --recompute \
  --log-file "$LOG_DIR/embedding_recompute.log"
log "Step 5 done."

# ── Step 6: Community detection B3 ───────────────────────────────────────────
log "=== STEP 6: Community detection B3 (resolution=3.0, max_kw_freq=500) ==="
uv run python scripts/run_communities.py \
  --log-file "$LOG_DIR/community_rerun.log"
log "Step 6 done."

# ── Step 7: Eval harness D2 ──────────────────────────────────────────────────
log "=== STEP 7: Eval harness D2 ==="
uv run python -m eval.harness 2>&1 | tee "$LOG_DIR/eval_harness.log"
log "Step 7 done."

log "=== ALL STEPS COMPLETE ==="
