---
name: 1-merge
description: Merge all source CSVs from data/raw/source_*.csv into unified dataset.parquet
allowed-tools: Bash, Read, Write, Glob
---

Merge all collected data sources into one unified dataset.

1. Write `scripts/1_merge.py`:
   - Glob `data/raw/source_*.csv` to find all source files
   - Read each CSV, validate schema (text, label, source, collected_at)
   - Concatenate all DataFrames
   - Deduplicate by text column
   - Save to `data/raw/dataset.parquet`
2. Create `config.yaml` documenting all sources used:
   - For each source file: name, type (hf/api/scrape), row count, URL/dataset name
3. Run: `uv run python scripts/1_merge.py`
4. Print: total rows, rows per source, duplicate count removed
