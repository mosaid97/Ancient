# `data/seeds/` — Philological normalization seeds

These tables are the canonical inputs to `apps/backend/normalize/` per
plan §2.7 and AGENTS.md §11. They are **seeds** — minimum-viable curated
content that Phase 11 (HITL closed loop) grows over time.

| File | Purpose | Source (planned) | Source (current seed) |
|---|---|---|---|
| `variants_unihan.tsv` | 異體字 map | Unihan `kCompatibilityVariant` + `kSimplifiedVariant` + 教育部異體字字典 | hand-curated common pairs (~50) |
| `taboo_tang.yaml` | 唐朝 避諱 table | 史諱舉例 + 中國歷代避諱字表 | hand-curated Tang emperors (~16) |
| `taboo_song.yaml` | 宋朝 避諱 table | 史諱舉例 + 中國歷代避諱字表 | hand-curated Song stub (~5) |
| `taboo_ming.yaml` | 明朝 避諱 table | 史諱舉例 + 中國歷代避諱字表 | hand-curated Ming stub (~3) |
| `taboo_qing.yaml` | 清朝 避諱 table | 史諱舉例 + 中國歷代避諱字表 | hand-curated Qing stub (~5) |
| `loan_chars.tsv` | 通假字 map | 古代漢語通假字大字典 | hand-curated common pairs (~30) |
| `era_calendar.yaml` | 紀年 → CE | 年號 + 干支 + 帝王 lookup | full Tang + selected dynasties |

## Provenance + licensing

The seed content is hand-curated from public-domain sources (the *Records of
the Grand Historian*, *Old Book of Tang*, *New Book of Tang*, the Unihan
database, etc.). When this directory ships in `data/bench/normalization/`
(plan §10 side-deliverable), it is released under CC-BY-4.0 with attribution
to the Unicode Consortium for Unihan-derived rows.

## Format conventions

- TSV files: tab-separated, UTF-8, header row required, `#` lines are comments.
- YAML files: UTF-8, schema validated by the loader in `apps/backend/normalize/`.
- All Chinese text in canonical NFC form (verified by `nfc.normalize`).

## Adding new entries

1. Edit the appropriate file by hand for one-offs.
2. For batch additions from a HITL session, use `scripts/seed_*.py` (Phase 11).
3. Re-run `notebooks/00a_philological_seeds.ipynb` to regenerate the
   `_artifacts/00a_philological_seeds/seeds.json` checksum.
