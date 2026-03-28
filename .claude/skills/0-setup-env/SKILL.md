---
name: 0-setup-env
description: Initialize ML project environment with uv and create directory structure
allowed-tools: Bash, Read, Write, Glob
---

Initialize the project environment.

1. If `pyproject.toml` does not exist, run `uv init`
2. Create directories:
   - `data/raw/`
   - `data/cleaned/`
   - `data/labeled/`
   - `models/`
   - `reports/eda_plots/`
   - `reports/quality_plots/`
   - `notebooks/`
   - `scripts/`
3. Add base dependencies: `uv add pandas pyarrow`
4. Verify: `uv run python -c "import pandas; print('Setup OK')"`
