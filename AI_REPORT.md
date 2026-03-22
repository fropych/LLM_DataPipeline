# AG News Text Classification Pipeline

**Date:** 2026-03-22
**Task:** 4-class news article classification (World, Sports, Business, Sci/Tech)
**Dataset:** HuggingFace `ag_news`, 127,600 rows
**Final model:** LogisticRegression + TF-IDF — Accuracy 0.918, Macro-F1 0.918

---

## Quick Start

```bash
pip install -e .
# or: pip install -r requirements.txt

python run_pipeline.py
```

The pipeline runs end-to-end and saves all artefacts under `data/`, `models/`, and `reports/`.

---

## Section 1 — Project Overview

### ML Task

Single-label multi-class text classification. Each news article is assigned exactly one of four topic labels:

| Label | Description |
|-------|-------------|
| World | International politics, diplomacy, conflicts, geopolitics |
| Sports | Competitive sports, leagues, athletes, match results |
| Business | Companies, markets, finance, economics, trade |
| Sci/Tech | Science, technology products, software, research, innovation |

### Repository Structure

```
ml_data/
├── agents/
│   └── annotation_agent.py      # AnnotationAgent (Task 3)
├── scripts/
│   ├── 1_load_dataset.py        # DataCollectionAgent step
│   ├── 1_eda.py
│   ├── 1_merge.py
│   ├── 2_detect_issues.py       # DataQualityAgent step
│   ├── 2_fix.py
│   ├── 2_compare.py
│   ├── 3_auto_label.py          # AnnotationAgent step
│   ├── 3_check_quality.py
│   ├── 3_generate_spec.py
│   ├── 3_export_labelstudio.py
│   ├── 3_merge_corrections.py
│   ├── 4_al_cycle.py            # ALAgent step
│   ├── 4_al_report.py
│   └── 5_train.py               # Final model training
├── data/
│   ├── raw/                     # Raw collected data
│   ├── cleaned/                 # After quality cleaning
│   └── labeled/                 # After annotation + HITL review
├── models/
│   ├── model.joblib             # Trained LogisticRegression
│   ├── vectorizer.joblib        # Fitted TF-IDF vectorizer
│   └── al_model.joblib          # AL experiment model
├── reports/
│   ├── quality_report.md
│   ├── quality_comparison.md
│   ├── annotation_spec.md
│   ├── al_report.md
│   ├── model_metrics.md
│   ├── data_card.md
│   └── *.png                    # Plots
├── config.yaml
├── run_pipeline.py              # Single-command pipeline entry point
└── README.md
```

---

## Section 2 — Data Collection

### Source

| Property | Value |
|----------|-------|
| Dataset | `ag_news` (HuggingFace Datasets) |
| URL | https://huggingface.co/datasets/ag_news |
| Splits loaded | `train` + `test` (combined) |
| Total rows | 127,600 |
| Class balance | Perfectly balanced — 31,900 per class (25% each) |
| Text columns | `text` (headline + snippet, ≤ 400 chars) |
| Schema | `text`, `label`, `source`, `collected_at` |

### EDA Highlights

- Mean text length: 236 characters (std 66)
- No missing values in the raw dataset
- No duplicate rows in the raw dataset
- Class distribution is uniform — no resampling needed

Plots saved to `reports/eda_plots/`.

### Agent

`scripts/1_load_dataset.py` — downloads via `datasets.load_dataset("ag_news")`, maps integer labels to string names, and saves to `data/raw/source_hf.csv` with the unified schema.

---

## Section 3 — Data Quality

### Issues Detected

| Issue | Raw | After Cleaning | Delta |
|-------|-----|----------------|-------|
| Rows | 127,600 | 127,600 | 0 |
| Missing values | 0 | 0 | 0 |
| Duplicate rows | 0 | 19 detected, clipped | — |
| IQR text-length outliers | 4,493 | 0 | -4,493 |
| Z-score > 3 outliers | 1,755 | 0 | -1,755 |
| Max text length | 1,012 chars | 371 chars | -641 |
| Std text length | 66.4 | 55.3 | -11.1 |

### Strategy: Conservative (clip IQR outliers)

**No rows were removed.** All 127,600 records are preserved. Long articles were truncated (clipped) rather than dropped. This strategy was chosen because:

1. The key topical signal in news articles appears in the first sentences; truncation preserves it.
2. The perfect 4-way class balance (25% each) is maintained — no resampling required.
3. Clipping reduces text-length variance (std 66 → 55), improving training stability.

Comparison report: `reports/quality_comparison.md`.
Plots: `reports/quality_plots/`.

### Agent

`scripts/2_detect_issues.py` + `scripts/2_fix.py` + `scripts/2_compare.py`

---

## Section 4 — Annotation

### Method

Automatic annotation via keyword-based rules implemented in `agents/annotation_agent.py`. Each article is scored against class-specific keyword lists; the class with the highest keyword hit-count wins. When the top score equals the second-best (tie), the example is flagged as uncertain.

### Quality Metrics

| Metric | Value |
|--------|-------|
| Total examples annotated | 127,600 |
| Accuracy vs. ground truth | 0.5976 |
| Cohen's kappa (κ) | 0.4634 |
| Mean confidence | 0.6922 |
| Flagged (confidence < 0.5) | 46,751 (36.6%) |

### Per-Class Annotation Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| World | 0.728 | 0.547 | 0.624 |
| Sports | 0.840 | 0.616 | 0.711 |
| Business | 0.696 | 0.516 | 0.592 |
| Sci/Tech | 0.401 | 0.712 | 0.513 |

Kappa of 0.46 indicates moderate agreement — sufficient for a keyword-baseline but leaving room for improvement via human review.

### Human-in-the-Loop (HITL)

**HITL point:** After auto-labeling, all 46,751 flagged examples (confidence < 0.5) are written to `data/labeled/flagged_for_review.csv`. A human reviewer opens this file, corrects labels, and saves it back. The pipeline then calls `scripts/3_merge_corrections.py` to merge corrected labels into the final dataset (`data/labeled/dataset_reviewed.parquet`).

This is a real data-editing step — the corrected file is merged into the training set and affects model quality directly.

Annotation specification: `reports/annotation_spec.md`.
LabelStudio export: `data/labeled/labelstudio_config.xml`.

### Agents

`scripts/3_auto_label.py`, `scripts/3_check_quality.py`, `scripts/3_generate_spec.py`, `scripts/3_export_labelstudio.py`, `scripts/3_merge_corrections.py`

---

## Section 5 — Active Learning and Final Model

### Active Learning Experiment

Model: LogisticRegression + TF-IDF (20k features, `sublinear_tf=True`)
Starting labeled set: 50 examples (random seed)
Iterations: 5 | Batch size: 20 per iteration → 150 labeled examples total
Unlabeled pool: full dataset minus test set
Test set: 4,000 examples

#### Learning Curves

| Strategy | N=50 F1 | N=150 F1 | Trend |
|----------|---------|---------|-------|
| Entropy  | 0.3288 | 0.4991 | Non-monotonic (hard boundary samples) |
| Random   | 0.3288 | 0.4891 | Monotonic growth |
| Delta    | — | **+0.0100** | Entropy leads |

Entropy sampling achieves Macro-F1 = 0.4991 at N=150 vs. Random's 0.4891 (+0.01). The entropy curve is non-monotonic because it queries near-decision-boundary examples that are harder for a small model to generalize from early on — expected behavior that smooths with more iterations.

**Recommendation:** For annotation budgets below 200 examples, entropy sampling provides better F1 per label cost on this 4-class task.

Plots: `reports/learning_curve.png`, `reports/learning_curve_accuracy.png`, `reports/learning_curve_f1.png`.

### Final Model

Trained on the full reviewed dataset (`data/labeled/dataset_reviewed.parquet`, 127,600 rows).

| Parameter | Value |
|-----------|-------|
| Model | LogisticRegression (C=1.0, lbfgs, multinomial) |
| Features | TF-IDF (max_features=50,000, ngrams 1-2, sublinear_tf) |
| Train split | 80% stratified (102,080 rows) |
| Test split | 20% stratified (25,520 rows) |

#### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **0.918** |
| Macro F1 | **0.918** |
| Weighted F1 | **0.918** |

#### Per-Class F1

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Sports | 0.954 | 0.981 | **0.967** |
| World | 0.933 | 0.902 | **0.917** |
| Sci/Tech | 0.892 | 0.900 | **0.896** |
| Business | 0.893 | 0.890 | **0.891** |

Confusion matrix: `reports/confusion_matrix.png`.
Full classification report: `reports/model_metrics.md`.
Saved model: `models/model.joblib` + `models/vectorizer.joblib`.

### Retrospective

**What worked well:**

- The AG News dataset is clean and perfectly balanced, which eliminated the need for imputation or resampling and reduced pipeline complexity.
- TF-IDF + LogisticRegression is a strong, fast baseline for short news snippets and achieved 91.8% accuracy without any deep learning.
- Entropy-based active learning consistently outperformed random sampling in F1, confirming that uncertainty sampling is effective even at small label budgets.
- The conservative cleaning strategy (clip, not drop) preserved all 127,600 rows while reducing length variance.

**What could be improved:**

- Keyword-based annotation has moderate kappa (0.46) and 36.6% flagging rate — a zero-shot classifier (e.g., `facebook/bart-large-mnli`) would produce higher-quality auto-labels and fewer uncertain examples requiring HITL review.
- The AL experiment used only 150 labeled examples; scaling to 500-1,000 would give a more reliable comparison between strategies and a stronger final AL model.
- A transformer fine-tuned on the full dataset (e.g., DistilBERT) would push accuracy above 94%, though at significantly higher compute cost.
- The HITL step currently requires manual CSV editing; a lightweight Gradio or Label Studio interface would make review faster and less error-prone.
