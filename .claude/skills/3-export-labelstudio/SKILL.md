---
name: 3-export-labelstudio
description: Export labeled data to LabelStudio import JSON format
allowed-tools: Bash, Read, Write
---

Export `data/labeled/dataset.parquet` to LabelStudio format.

1. Write `scripts/3_export_labelstudio.py`:
   - Read labeled dataset
   - Convert each row to LabelStudio task format:
     ```json
     {"data": {"text": "..."}, "predictions": [{"result": [{"from_name": "label", "to_name": "text", "type": "choices", "value": {"choices": ["..."]}}], "score": 0.85}]}
     ```
   - Save to `labelstudio_import.json`
2. Run: `uv run python scripts/3_export_labelstudio.py`
3. Print: number of tasks exported, file size
