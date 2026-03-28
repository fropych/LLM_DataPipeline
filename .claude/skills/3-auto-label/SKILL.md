---
name: 3-auto-label
description: Auto-label dataset with zero-shot classification, add predicted_label and confidence columns
allowed-tools: Bash, Read, Write
---

Auto-label `data/cleaned/dataset.parquet` using zero-shot classification.

1. Write `scripts/3_auto_label.py`:
   - Load cleaned dataset
   - Use `transformers` zero-shot classification pipeline (e.g. `facebook/bart-large-mnli`)
   - For each text, predict label from the user's target classes
   - Add columns: `predicted_label`, `confidence`
   - Save to `data/labeled/dataset.parquet`
2. Install deps and run: `uv add transformers torch && uv run python scripts/3_auto_label.py`
3. Print: label distribution, mean confidence, min/max confidence
