---
name: 2-fix
description: Apply cleaning strategy to dataset — accepts strategy as argument (conservative or aggressive)
allowed-tools: Bash, Read, Write
argument-hint: [strategy]
---

Apply cleaning strategy to `data/raw/dataset.parquet`. Strategy is passed as argument: `$ARGUMENTS`.

## Conservative strategy (median + IQR):
- Missing values: fill with median (numeric) / mode (categorical)
- Duplicates: drop exact duplicates
- Outliers: clip text length to IQR bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)

## Aggressive strategy (drop + z-score):
- Missing values: drop rows with any missing value
- Duplicates: drop exact duplicates
- Outliers: remove rows where text length z-score > 3

## Steps
1. Write `scripts/2_fix.py` implementing the selected strategy
2. Read `data/raw/dataset.parquet`, apply transformations
3. Save to `data/cleaned/dataset.parquet`
4. Run: `uv run python scripts/2_fix.py`
5. Print: rows before, rows after, what was changed
