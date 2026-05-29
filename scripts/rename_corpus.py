"""Normalize ``raw/`` corpus filenames for consistent sorting + storage.

Policy (per user direction, 2026-05-18):

- **Primary** (``raw/Primary/``): keep only the book title. Strip Z-Library
  marketing tags, author parentheticals, edition annotations, 二十四史
  numbering prefixes, ``《》`` brackets, and "word版本"-style format hints.
  Preserve 上/下/补编 part markers (the corpus has ``唐大诏令集补编 上`` and
  ``唐大诏令集补编 下`` which are distinct volumes).
- **Secondary** (``raw/Secondary/``): keep ``<author>_<title>`` shape.
  Strip Z-Library-style trailing tags; otherwise leave the filename intact
  because most secondaries are already in that shape.
- **Subdirectory normalization**: rename
  ``raw/Secondary/comprehensive_出身:铨选`` to
  ``raw/Secondary/comprehensive_出身_铨选`` because the ASCII ``:`` is
  shell-fragile (macOS Finder treats ``:`` as a path separator and many
  shell tools mis-parse it).

The script is intentionally a dry-run by default. Pass ``--apply`` to
actually move files. A JSON audit log is always written to
``notebooks/_artifacts/corpus_rename/rename_map.json`` so the renames are
reversible.

Usage::

    uv run python scripts/rename_corpus.py            # dry-run, prints plan
    uv run python scripts/rename_corpus.py --apply    # execute renames

The script is **idempotent**: re-running on an already-clean corpus is a
no-op.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("rename_corpus")


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "raw"
PRIMARY = RAW / "Primary"
SECONDARY = RAW / "Secondary"
AUDIT_DIR = REPO_ROOT / "notebooks" / "_artifacts" / "corpus_rename"

# Extensions we will rename. Internal files inside .packed/ trees are
# untouched (the structure of those trees is part of the corpus).
TARGET_EXTS = {".pdf", ".epub", ".djvu"}

# Subdirectory renames. Only directories whose names are shell-fragile are
# normalized; Chinese typographic punctuation is left alone.
SUBDIR_RENAMES: dict[str, str] = {
    "comprehensive_出身:铨选": "comprehensive_出身_铨选",
}


@dataclass
class Rename:
    old_path: Path
    new_path: Path
    reason: str
    tier: str  # "primary" | "secondary" | "subdir"

    def as_record(self) -> dict[str, str]:
        return {
            "old_path": str(self.old_path.relative_to(REPO_ROOT)),
            "new_path": str(self.new_path.relative_to(REPO_ROOT)),
            "tier": self.tier,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Filename cleaning.
# ---------------------------------------------------------------------------


_PRIMARY_STRIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Z-Library / 1lib variants, both with and without spaces / parens.
    (re.compile(r"\s*\([^)]*z-?library[^)]*\)", re.IGNORECASE), ""),
    (re.compile(r"\s*\([^)]*1lib[^)]*\)", re.IGNORECASE), ""),
    (re.compile(r"\s*\([^)]*z-?lib(?:\.org|\.sk)?[^)]*\)", re.IGNORECASE), ""),
    # 二十四史 numbering prefix: 【二十四史】16：
    (re.compile(r"^【[^】]+】\s*\d*\s*[:：]\s*"), ""),
    # Edition annotation: 册府元龟（点校本 校订本）
    (re.compile(r"（[^）]*点校[^）]*）"), ""),
    (re.compile(r"（[^）]*校订[^）]*）"), ""),
    # Series annotation: 唐摭言 (历代笔记小说大观)
    (re.compile(r"\s*\(历代笔记小说大观\)"), ""),
    # "word版本" format hint: 唐會要word版本
    (re.compile(r"\s*word\s*版本", re.IGNORECASE), ""),
    # Trailing download-duplicate marker (1) (2) (3)
    (re.compile(r"\s*\(\d+\)\s*$"), ""),
    # 《...》 book-title brackets — they're decoration, not part of the title.
    (re.compile(r"^《([^》]+)》"), r"\1"),
    # Author parentheticals — anything after we've stripped the markers
    # above is presumed to be author/translator info.
    (re.compile(r"\s*\([^)]+\)"), ""),
    (re.compile(r"\s*（[^）]+）"), ""),
    # Final whitespace cleanup.
    (re.compile(r"\s{2,}"), " "),
]


_SECONDARY_STRIP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"_Z-Library\b", re.IGNORECASE), ""),
    (re.compile(r"\s*\(\d+\)\s*$"), ""),
    (re.compile(r"\s{2,}"), " "),
]


_PART_NORMALIZE: list[tuple[re.Pattern[str], str]] = [
    # Normalize part markers: （下册） → 下, （上册） → 上.
    (re.compile(r"\s*（上册）\s*"), " 上"),
    (re.compile(r"\s*（下册）\s*"), " 下"),
    (re.compile(r"\s*\(上册\)\s*"), " 上"),
    (re.compile(r"\s*\(下册\)\s*"), " 下"),
]


def clean_primary_stem(stem: str) -> str:
    """Strip everything but the book title from a Primary filename stem.

    Order matters — Z-Library tags are stripped first so the catch-all
    author-parenthetical rule doesn't accidentally swallow them with the
    wrong replacement.
    """
    s = stem
    # Normalize part markers BEFORE stripping parentheticals so we keep 上/下.
    for pat, repl in _PART_NORMALIZE:
        s = pat.sub(repl, s)
    for pat, repl in _PRIMARY_STRIP_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()


def clean_secondary_stem(stem: str) -> str:
    """Strip Z-Library and other inert tags from a Secondary filename stem.

    Keeps the conventional ``<author>_<title>`` shape (and any journal /
    book-context suffix that disambiguates a paper from a same-author
    counterpart).
    """
    s = stem
    for pat, repl in _SECONDARY_STRIP_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()


# ---------------------------------------------------------------------------
# Plan computation.
# ---------------------------------------------------------------------------


def _iter_target_files(root: Path) -> list[Path]:
    """List target files under ``root`` (top-level only by default).

    Recurses into subdirectories so a paper under ``specific_科举/`` is
    picked up, but ignores everything inside ``.packed/`` trees (those are
    internal markdown shards, not standalone documents).
    """
    if not root.exists():
        return []
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TARGET_EXTS:
            continue
        # Skip files inside .packed/ trees.
        if any(parent.name.endswith(".packed") for parent in path.parents):
            continue
        out.append(path)
    return sorted(out)


def build_plan() -> list[Rename]:
    """Compute (old → new) renames for files AND the one fragile subdir."""
    plan: list[Rename] = []

    # Stage 1: subdirectory normalization.
    for old_name, new_name in SUBDIR_RENAMES.items():
        old_dir = SECONDARY / old_name
        new_dir = SECONDARY / new_name
        if old_dir.exists():
            plan.append(
                Rename(
                    old_path=old_dir,
                    new_path=new_dir,
                    reason=f"shell-fragile char in '{old_name}'",
                    tier="subdir",
                )
            )

    # Stage 2: primary files — apply per-stem cleaner.
    for path in _iter_target_files(PRIMARY):
        new_stem = clean_primary_stem(path.stem)
        if not new_stem:
            logger.warning("primary cleaner produced empty stem for %s", path)
            continue
        new_path = path.with_name(new_stem + path.suffix)
        if new_path != path:
            plan.append(
                Rename(
                    old_path=path,
                    new_path=new_path,
                    reason="primary: keep only book title",
                    tier="primary",
                )
            )

    # Stage 3: secondary files — applied AFTER the subdir rename so the new
    # path is computed against the post-rename directory layout.
    for path in _iter_target_files(SECONDARY):
        # If this path is inside an about-to-be-renamed subdir, rewrite its
        # parent to the new directory name first.
        parts = list(path.parts)
        for i, seg in enumerate(parts):
            if seg in SUBDIR_RENAMES:
                parts[i] = SUBDIR_RENAMES[seg]
        rewritten_parent = Path(*parts).parent

        new_stem = clean_secondary_stem(path.stem)
        if not new_stem:
            logger.warning("secondary cleaner produced empty stem for %s", path)
            continue
        new_path = rewritten_parent / (new_stem + path.suffix)
        if new_path != path:
            plan.append(
                Rename(
                    old_path=path,
                    new_path=new_path,
                    reason="secondary: strip Z-Library tag / normalize",
                    tier="secondary",
                )
            )

    return plan


def detect_collisions(plan: list[Rename]) -> list[tuple[Rename, Rename]]:
    """Find pairs of renames whose new_path collides."""
    by_target: dict[Path, Rename] = {}
    collisions: list[tuple[Rename, Rename]] = []
    for r in plan:
        existing = by_target.get(r.new_path)
        if existing is not None:
            collisions.append((existing, r))
        else:
            by_target[r.new_path] = r
    return collisions


# ---------------------------------------------------------------------------
# Execution.
# ---------------------------------------------------------------------------


def apply_plan(plan: list[Rename]) -> tuple[list[Rename], list[tuple[Rename, str]]]:
    """Execute renames. Returns ``(applied, failed)``.

    Subdir renames go first (so subsequent file renames see the new layout).
    Then file renames in path order so nested dirs settle before children.
    """
    applied: list[Rename] = []
    failed: list[tuple[Rename, str]] = []

    subdir_renames = [r for r in plan if r.tier == "subdir"]
    file_renames = [r for r in plan if r.tier != "subdir"]

    for r in subdir_renames:
        try:
            if r.new_path.exists():
                failed.append((r, f"target already exists: {r.new_path}"))
                continue
            r.new_path.parent.mkdir(parents=True, exist_ok=True)
            r.old_path.rename(r.new_path)
            applied.append(r)
        except Exception as exc:  # noqa: BLE001
            failed.append((r, f"{type(exc).__name__}: {exc}"))

    for r in file_renames:
        try:
            # If the parent was renamed already, the old_path's parent
            # segment is stale — recompute the effective old path.
            effective_old = _rewrite_for_subdir_renames(r.old_path)
            # No-op case: the parent rename already moved this file into
            # its target slot, and the stem itself didn't change. Treat as
            # success so re-runs are clean.
            if effective_old == r.new_path and r.new_path.exists():
                applied.append(r)
                continue
            if r.new_path.exists() and r.new_path != r.old_path:
                failed.append((r, f"target already exists: {r.new_path}"))
                continue
            r.new_path.parent.mkdir(parents=True, exist_ok=True)
            if effective_old != r.old_path and effective_old.exists():
                effective_old.rename(r.new_path)
            else:
                r.old_path.rename(r.new_path)
            applied.append(r)
        except Exception as exc:  # noqa: BLE001
            failed.append((r, f"{type(exc).__name__}: {exc}"))

    return applied, failed


def _rewrite_for_subdir_renames(path: Path) -> Path:
    """Return ``path`` with any old subdir segment replaced by its new name."""
    parts = list(path.parts)
    changed = False
    for i, seg in enumerate(parts):
        if seg in SUBDIR_RENAMES:
            parts[i] = SUBDIR_RENAMES[seg]
            changed = True
    return Path(*parts) if changed else path


def write_audit(plan: list[Rename], *, applied: bool) -> Path:
    """Persist the plan + status to the audit JSON."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = AUDIT_DIR / f"rename_map_{ts}.json"
    payload = {
        "timestamp": ts,
        "applied": applied,
        "count": len(plan),
        "by_tier": {
            tier: sum(1 for r in plan if r.tier == tier)
            for tier in ("primary", "secondary", "subdir")
        },
        "renames": [r.as_record() for r in plan],
    }
    fname.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Also write a stable "latest" pointer for the notebook to read.
    latest = AUDIT_DIR / "rename_map.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fname


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute renames (default is dry-run).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-rename printout.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    plan = build_plan()
    collisions = detect_collisions(plan)

    by_tier = {tier: [r for r in plan if r.tier == tier] for tier in ("subdir", "primary", "secondary")}
    print(f"Planned renames: {len(plan)} total")
    for tier in ("subdir", "primary", "secondary"):
        print(f"  {tier}: {len(by_tier[tier])}")
    if collisions:
        print(f"\nCOLLISIONS ({len(collisions)}):")
        for a, b in collisions:
            print(f"  {a.old_path.name}  -> {a.new_path.name}")
            print(f"  {b.old_path.name}  -> {b.new_path.name}")
        print("\nRefusing to apply due to collisions.")
        return 2

    if not args.quiet:
        for tier in ("subdir", "primary", "secondary"):
            tier_plan = by_tier[tier]
            if not tier_plan:
                continue
            print(f"\n[{tier}] {len(tier_plan)} rename(s):")
            for r in tier_plan:
                old = r.old_path.relative_to(REPO_ROOT)
                new = r.new_path.relative_to(REPO_ROOT)
                print(f"  {old}")
                print(f"    -> {new}")

    if not args.apply:
        audit_path = write_audit(plan, applied=False)
        print(f"\nDry-run complete. Audit log: {audit_path.relative_to(REPO_ROOT)}")
        print("Re-run with --apply to execute.")
        return 0

    applied, failed = apply_plan(plan)
    audit_path = write_audit(applied + [r for r, _ in failed], applied=True)
    print(f"\nApplied: {len(applied)}, Failed: {len(failed)}")
    for r, reason in failed:
        print(f"  FAIL {r.old_path}: {reason}")
    print(f"Audit log: {audit_path.relative_to(REPO_ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
