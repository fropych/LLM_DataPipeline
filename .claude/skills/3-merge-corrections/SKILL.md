---
name: 3-merge-corrections
description: Merge human-reviewed corrections from review_queue_corrected.csv back into labeled dataset
allowed-tools: Bash, Read, Write
---

Merge HITL corrections into the labeled dataset.

1. Write `scripts/3_merge_corrections.py`:
   - Read `data/labeled/dataset.parquet` (auto-labeled data)
   - Check if `review_queue_corrected.csv` exists:
     - **If yes**: read corrections, replace `predicted_label` for corrected rows, log changes
     - **If no**: log warning "No corrections file found, using auto-labels as-is"
   - Save to `data/labeled/dataset_reviewed.parquet`
2. Run: `uv run python scripts/3_merge_corrections.py`
3. Print: X of Y flagged examples were corrected by human
