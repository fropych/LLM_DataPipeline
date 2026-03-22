# Quality Report — AG News Annotation

**Total examples:** 127,600
**Accuracy (vs ground truth):** 0.5976
**Cohen's kappa:** 0.4634

## Confidence Stats

| Metric | Value |
|--------|-------|
| Mean   | 0.6922 |
| Median | 1.0000 |
| Min    | 0.2500 |
| Max    | 1.0000 |
| Std    | 0.3424 |

## Flagged Examples

Threshold: confidence < 0.5

**Flagged count:** 46,751 (36.6%)

## Label Distribution

| Class | Ground Truth | Predicted |
|-------|-------------|-----------|
| World | 31,900 | 23,959 |
| Sports | 31,900 | 23,399 |
| Business | 31,900 | 23,619 |
| Sci/Tech | 31,900 | 56,623 |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| World | 0.7277 | 0.5466 | 0.6243 | 31,900 |
| Sports | 0.8395 | 0.6158 | 0.7105 | 31,900 |
| Business | 0.6963 | 0.5155 | 0.5924 | 31,900 |
| Sci/Tech | 0.4013 | 0.7124 | 0.5134 | 31,900 |