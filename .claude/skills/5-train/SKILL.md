---
name: 5-train
description: Train final classification model on labeled data, evaluate accuracy + F1
allowed-tools: Bash, Read, Write
---

Train the final model on `data/labeled/dataset_reviewed.parquet`.

1. Write `scripts/5_train.py`:
   - Feature extraction: TF-IDF vectorizer on text column
   - Train/test split: 80/20 stratified by label
   - Train LogisticRegression (or SVM if appropriate)
   - Evaluate: accuracy, F1 (macro + weighted), classification report
   - Confusion matrix → `reports/confusion_matrix.png`
   - Save model to `models/model.joblib`
   - Save vectorizer to `models/vectorizer.joblib`
2. Run: `uv run python scripts/5_train.py`
3. Save metrics to `reports/model_metrics.md`
4. Print: accuracy, F1, classification report table
