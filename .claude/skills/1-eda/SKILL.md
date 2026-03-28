---
name: 1-eda
description: Exploratory data analysis on merged dataset — class distribution, text lengths, top-20 words
allowed-tools: Bash, Read, Write
---

Run EDA on `data/raw/dataset.parquet`.

1. Write `scripts/1_eda.py`:
   - Class distribution — bar chart → `reports/eda_plots/class_distribution.png`
   - Text length distribution — histogram → `reports/eda_plots/text_lengths.png`
   - Top-20 most frequent words (after stopword removal) — bar chart → `reports/eda_plots/top_words.png`
   - Source distribution — pie chart → `reports/eda_plots/source_distribution.png`
   - Basic stats: total rows, columns, dtypes, missing values summary
2. Install deps and run: `uv add matplotlib seaborn nltk && uv run python scripts/1_eda.py`
3. Generate `notebooks/eda.ipynb` with the same analysis, inline plots, and markdown explanations
4. Print: summary stats table
