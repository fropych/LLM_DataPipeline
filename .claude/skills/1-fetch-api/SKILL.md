---
name: 1-fetch-api
description: Fetch data from a REST API endpoint, parse JSON, standardize to (text, label, source, collected_at)
allowed-tools: Bash, Read, Write
---

Fetch data from a REST API based on the user's task description.

1. Write `scripts/1_fetch_api.py`:
   - Send GET/POST request to the specified API endpoint
   - Parse JSON response, extract text and label fields
   - Handle pagination if the API supports it
   - Map to standard schema: `text`, `label`, `source`, `collected_at`
   - Save to `data/raw/source_api.csv`
2. Install deps and run: `uv add requests && uv run python scripts/1_fetch_api.py`
3. Print: number of rows fetched, API endpoint used
