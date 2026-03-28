---
name: 3-check-quality
description: Check annotation quality — Cohen's kappa, label distribution, confidence stats, flag low-confidence examples
allowed-tools: Bash, Read, Write
---

Check quality of auto-labeled data in `data/labeled/dataset.parquet`.

1. Write `scripts/3_check_quality.py`:
   - Compute Cohen's kappa between `predicted_label` and original `label` (if exists)
   - Label distribution statistics (counts and percentages)
   - Confidence score distribution: mean, std, histogram → `reports/quality_plots/confidence_hist.png`
   - Flag examples with confidence < 0.7 → save flagged rows to `review_queue.csv`
   - Summary: total flagged, percentage of dataset
2. Install deps and run: `uv add scikit-learn matplotlib && uv run python scripts/3_check_quality.py`
3. Save metrics to `reports/annotation_report.md`
4. Print: kappa score, mean confidence, number of flagged examples
