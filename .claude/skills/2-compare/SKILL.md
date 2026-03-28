---
name: 2-compare
description: Compare dataset before/after cleaning — row count, missing %, duplicates, outliers, class balance
allowed-tools: Bash, Read, Write
---

Compare raw vs cleaned dataset.

1. Write `scripts/2_compare.py`:
   - Load `data/raw/dataset.parquet` (before) and `data/cleaned/dataset.parquet` (after)
   - Compare metrics:
     - Row count: before vs after
     - Missing values per column: before vs after
     - Duplicate count: before vs after
     - Outlier count (IQR): before vs after
     - Class distribution: before vs after
   - Generate before/after comparison table
   - Include justification: why the chosen strategy is appropriate for this ML task
2. Run: `uv run python scripts/2_compare.py`
3. Save to `reports/quality_comparison.md`
4. Print the comparison table
