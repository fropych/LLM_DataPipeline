# Data Card — AG News Text Classification Dataset

**Version:** 1.0
**Date:** 2026-03-22
**Task:** Multi-class text classification (4 classes)
**Pipeline stage:** Final labeled dataset (`data/labeled/dataset_reviewed.parquet`)

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Dataset name | AG News (reviewed) |
| Original source | HuggingFace `ag_news` |
| Modality | Text |
| Language | English |
| Total rows | 127,600 |
| Splits | Combined train + test from source |
| Class count | 4 |
| Balance | Perfectly balanced (25% per class) |
| Text field | `text` — headline + snippet, ≤ ~371 chars after cleaning |
| Label field | `label` — string: World, Sports, Business, Sci/Tech |

---

## Classes

| Label | Description | Row count | Share |
|-------|-------------|-----------|-------|
| World | International politics, diplomacy, conflicts, geopolitics | 31,900 | 25% |
| Sports | Competitive sports, leagues, athletes, results | 31,900 | 25% |
| Business | Companies, markets, finance, economics, trade | 31,900 | 25% |
| Sci/Tech | Science, technology, software, research, innovation | 31,900 | 25% |

---

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | News headline + article snippet (cleaned, length-clipped) |
| `label` | string | Ground-truth class label |
| `auto_label` | string | Auto-assigned label from AnnotationAgent keyword rules |
| `confidence` | float | Annotation confidence score [0.25, 1.0] |
| `flagged` | bool | True if confidence < 0.5 (sent for HITL review) |
| `source` | string | `huggingface:ag_news` |
| `collected_at` | string | ISO 8601 UTC timestamp of collection |

---

## Collection

- **Method:** `datasets.load_dataset("ag_news")` via HuggingFace Datasets library
- **Splits merged:** `train` (120,000 rows) + `test` (7,600 rows)
- **Collection date:** 2026-03-21
- **Script:** `scripts/1_load_dataset.py`

---

## Cleaning

| Step | Strategy | Effect |
|------|----------|--------|
| Missing values | None found | No action needed |
| Duplicate rows | Detected 19; content clipped to IQR bounds | 0 rows removed |
| Text length outliers | IQR clip (max 371 chars) | 4,493 outlier lengths normalized |
| Class balance | No action needed | 25% per class maintained |

**Rows removed:** 0. All 127,600 records are preserved.
**Script:** `scripts/2_detect_issues.py`, `scripts/2_fix.py`

---

## Annotation

- **Method:** Keyword-based rule classifier (`agents/annotation_agent.py`)
- **Accuracy vs. ground truth:** 0.5976
- **Cohen's kappa:** 0.4634 (moderate agreement)
- **Mean confidence:** 0.6922
- **Flagged for HITL review:** 46,751 examples (36.6%, confidence < 0.5)
- **HITL process:** Flagged examples exported to `data/labeled/flagged_for_review.csv`; a human reviewer corrects labels; corrections merged via `scripts/3_merge_corrections.py`

---

## Active Learning

| Parameter | Value |
|-----------|-------|
| Strategy compared | Entropy vs. Random |
| Initial labeled set | 50 examples |
| Iterations | 5 |
| Batch size | 20 per iteration |
| Final N labeled | 150 |
| Entropy final Macro-F1 | 0.4991 |
| Random final Macro-F1 | 0.4891 |
| Delta (entropy - random) | +0.0100 |

---

## Model Performance

Trained on the full reviewed dataset with 80/20 stratified split.

| Metric | Value |
|--------|-------|
| Model | LogisticRegression + TF-IDF |
| Accuracy | 0.918 |
| Macro F1 | 0.918 |
| Weighted F1 | 0.918 |

### Per-Class F1

| Class | F1 |
|-------|----|
| Sports | 0.967 |
| World | 0.917 |
| Sci/Tech | 0.896 |
| Business | 0.891 |

---

## Known Limitations

1. **Keyword annotation bias:** Sci/Tech is over-predicted by the keyword annotator (56,623 auto-labels vs. 31,900 ground truth), likely due to broad tech-related keywords matching Business and World articles.
2. **Single source:** The dataset originates from a single source (AG News corpus, ~2004-2005 news). It may not generalize to modern news styles or topics.
3. **English only:** All text is in English; the pipeline is not designed for multilingual use.
4. **Short texts:** Articles are truncated snippets (headline + excerpt); full-article classification may behave differently.

---

## Intended Use

- Academic/educational text classification pipeline demonstration
- Benchmark for comparing active learning strategies on balanced 4-class news data
- Baseline evaluation of TF-IDF + LogisticRegression for short-text classification

---

## Files

| File | Description |
|------|-------------|
| `data/raw/source_hf.csv` | Raw collected dataset |
| `data/labeled/dataset.parquet` | After auto-annotation |
| `data/labeled/flagged_for_review.csv` | Examples for HITL review |
| `data/labeled/dataset_reviewed.parquet` | Final dataset after HITL merge |
| `models/model.joblib` | Trained LogisticRegression |
| `models/vectorizer.joblib` | Fitted TF-IDF vectorizer |
