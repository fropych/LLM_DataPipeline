---
name: 2-detect-issues
description: Detect data quality issues — missing values, duplicates, outliers (IQR + z-score), class imbalance
allowed-tools: Bash, Read, Write
---

Analyze `data/raw/dataset.parquet` for quality issues.

1. Write `scripts/2_detect_issues.py`:
   - **Missing values**: count per column, percentage, heatmap → `reports/quality_plots/missing_heatmap.png`
   - **Duplicates**: exact duplicate row count
   - **Outliers**: text length outliers via IQR method AND z-score, box plot → `reports/quality_plots/outliers_boxplot.png`
   - **Class imbalance**: class distribution, imbalance ratio, bar chart → `reports/quality_plots/class_imbalance.png`
2. Install deps and run: `uv add matplotlib seaborn scipy && uv run python scripts/2_detect_issues.py`
3. Generate `reports/quality_report.md`:
   - Summary table of all issues found
   - Severity assessment per issue
   - Recommended cleaning actions
4. Print clear summary so user can choose a cleaning strategy
