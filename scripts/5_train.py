"""Train final classification model on labeled data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib
from pathlib import Path

DATA = Path("data/labeled/dataset_reviewed.parquet")
MODELS = Path("models")
REPORTS = Path("reports")

df = pd.read_parquet(DATA)
print(f"Dataset: {len(df)} rows")

# Use ground-truth label for training
X_text = df["text"]
y = df["label"]
labels = sorted(y.unique())
print(f"Classes: {labels}")

# Train/test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print(f"Features: {X_train.shape[1]}")

# Train
model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred, digits=4)

print(f"\n=== Results ===")
print(f"Accuracy:    {acc:.4f}")
print(f"F1 (macro):  {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"\n{report}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title(f"Confusion Matrix (Accuracy={acc:.4f})")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
fig.tight_layout()
fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"→ Saved {REPORTS / 'confusion_matrix.png'}")

# Save model + vectorizer
joblib.dump(model, MODELS / "model.joblib")
joblib.dump(vectorizer, MODELS / "vectorizer.joblib")
print(f"→ Saved model to {MODELS / 'model.joblib'}")
print(f"→ Saved vectorizer to {MODELS / 'vectorizer.joblib'}")

# Save metrics report
with open(REPORTS / "model_metrics.md", "w") as f:
    f.write("# Model Metrics\n\n")
    f.write(f"**Model**: LogisticRegression (C=1.0, multinomial, lbfgs)\n\n")
    f.write(f"**Features**: TF-IDF (max_features=50000, ngrams=1-2, sublinear_tf)\n\n")
    f.write(f"**Train/Test Split**: 80/20 stratified (train={len(X_train_text)}, test={len(X_test_text)})\n\n")
    f.write("## Overall Metrics\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| Accuracy | {acc:.4f} |\n")
    f.write(f"| F1 (macro) | {f1_macro:.4f} |\n")
    f.write(f"| F1 (weighted) | {f1_weighted:.4f} |\n\n")
    f.write("## Classification Report\n\n")
    f.write("```\n")
    f.write(report)
    f.write("```\n\n")
    f.write("![Confusion Matrix](confusion_matrix.png)\n")

print(f"→ Saved {REPORTS / 'model_metrics.md'}")
