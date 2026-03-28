---
name: 1-load-dataset
description: Load dataset from HuggingFace or Kaggle, standardize schema to (text, label, source, collected_at)
allowed-tools: Bash, Read, Write
---

Load a dataset from HuggingFace or Kaggle based on the user's task description.

1. Write `scripts/1_load_dataset.py`:
   - Use `datasets` library to load the specified dataset
   - Map original columns to standard schema: `text`, `label`
   - Add `source` column (e.g. `"huggingface:{dataset_name}"`)
   - Add `collected_at` column with current ISO timestamp
   - Save to `data/raw/source_hf.csv`
2. Install deps and run: `uv add datasets && uv run python scripts/1_load_dataset.py`
3. Print: number of rows loaded, unique labels found
