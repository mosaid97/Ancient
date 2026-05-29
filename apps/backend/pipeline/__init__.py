"""Celery / async pipeline tasks (ingest -> preprocess -> ocr -> ...).

Phase 1 lands the deterministic ingest piece (no Celery yet — orchestration
is sync inside the notebook). Phases 2+ wrap each step in a Celery task.
"""
