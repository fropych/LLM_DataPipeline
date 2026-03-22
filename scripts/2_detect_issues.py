"""Detect data quality issues in dataset.parquet."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Paths
DATA = Path("data/raw/dataset.parquet")
PLOTS = Path("reports/quality_plots")
PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)
print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}\n")

issues = []

# --- 1. Missing values ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("=== Missing Values ===")
for col in df.columns:
    print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
    if missing[col] > 0:
        issues.append(("Missing values", col, missing[col], missing_pct[col], "HIGH"))

# Missing heatmap
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df.isnull().T, cbar=True, yticklabels=df.columns, cmap="YlOrRd", ax=ax)
ax.set_title("Missing Values Heatmap")
ax.set_xlabel("Row index")
fig.tight_layout()
fig.savefig(PLOTS / "missing_heatmap.png", dpi=150)
plt.close(fig)
print(f"  → Saved {PLOTS / 'missing_heatmap.png'}\n")

# --- 2. Duplicates ---
n_dupes = df.duplicated().sum()
print(f"=== Duplicates ===\n  Exact duplicate rows: {n_dupes}")
if n_dupes > 0:
    issues.append(("Duplicates", "all columns", n_dupes, round(n_dupes / len(df) * 100, 2), "MEDIUM"))

# Also check text-only duplicates
text_dupes = df["text"].duplicated().sum()
print(f"  Text-only duplicates: {text_dupes}")
if text_dupes > 0:
    issues.append(("Text duplicates", "text", text_dupes, round(text_dupes / len(df) * 100, 2), "MEDIUM"))
print()

# --- 3. Outliers (text length) ---
df["text_len"] = df["text"].str.len()
Q1 = df["text_len"].quantile(0.25)
Q3 = df["text_len"].quantile(0.75)
IQR = Q3 - Q1
lower_iqr = Q1 - 1.5 * IQR
upper_iqr = Q3 + 1.5 * IQR
iqr_outliers = ((df["text_len"] < lower_iqr) | (df["text_len"] > upper_iqr)).sum()

z_scores = np.abs(stats.zscore(df["text_len"]))
z_outliers = (z_scores > 3).sum()

print("=== Outliers (text length) ===")
print(f"  Mean: {df['text_len'].mean():.1f}, Std: {df['text_len'].std():.1f}")
print(f"  IQR range: [{lower_iqr:.0f}, {upper_iqr:.0f}]")
print(f"  IQR outliers: {iqr_outliers} ({iqr_outliers / len(df) * 100:.2f}%)")
print(f"  Z-score > 3 outliers: {z_outliers} ({z_outliers / len(df) * 100:.2f}%)")
if iqr_outliers > 0:
    issues.append(("IQR outliers", "text_len", iqr_outliers, round(iqr_outliers / len(df) * 100, 2), "LOW"))
if z_outliers > 0:
    issues.append(("Z-score outliers", "text_len", z_outliers, round(z_outliers / len(df) * 100, 2), "LOW"))

# Boxplot
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x="label", y="text_len", data=df, ax=ax, palette="Set2")
ax.set_title("Text Length Distribution by Class (Outlier Detection)")
ax.set_ylabel("Text length (chars)")
ax.set_xlabel("Label")
fig.tight_layout()
fig.savefig(PLOTS / "outliers_boxplot.png", dpi=150)
plt.close(fig)
print(f"  → Saved {PLOTS / 'outliers_boxplot.png'}\n")

# --- 4. Class imbalance ---
class_counts = df["label"].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
print("=== Class Imbalance ===")
for label, count in class_counts.items():
    print(f"  {label}: {count} ({count / len(df) * 100:.1f}%)")
print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}")
if imbalance_ratio > 1.5:
    issues.append(("Class imbalance", "label", f"ratio={imbalance_ratio:.2f}", "", "MEDIUM"))

# Bar chart
fig, ax = plt.subplots(figsize=(8, 4))
class_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(class_counts)))
ax.set_title("Class Distribution")
ax.set_ylabel("Count")
ax.set_xlabel("Label")
ax.bar_label(ax.containers[0])
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(PLOTS / "class_imbalance.png", dpi=150)
plt.close(fig)
print(f"  → Saved {PLOTS / 'class_imbalance.png'}\n")

# --- Generate quality report ---
report = Path("reports/quality_report.md")
with open(report, "w") as f:
    f.write("# Data Quality Report\n\n")
    f.write(f"**Dataset**: `{DATA}` — {len(df)} rows, {len(df.columns)} columns\n\n")

    f.write("## Issues Summary\n\n")
    f.write("| Issue | Column | Count | % of data | Severity |\n")
    f.write("|-------|--------|-------|-----------|----------|\n")
    if not issues:
        f.write("| No issues found | — | — | — | — |\n")
    for issue_type, col, count, pct, severity in issues:
        f.write(f"| {issue_type} | {col} | {count} | {pct}% | {severity} |\n")

    f.write("\n## Details\n\n")
    f.write(f"### Missing Values\n")
    f.write(f"- Total missing: {missing.sum()} across all columns\n")
    f.write(f"- ![Missing Heatmap](quality_plots/missing_heatmap.png)\n\n")

    f.write(f"### Duplicates\n")
    f.write(f"- Exact duplicates: {n_dupes}\n")
    f.write(f"- Text-only duplicates: {text_dupes}\n\n")

    f.write(f"### Outliers (text length)\n")
    f.write(f"- IQR method: {iqr_outliers} outliers outside [{lower_iqr:.0f}, {upper_iqr:.0f}]\n")
    f.write(f"- Z-score > 3: {z_outliers} outliers\n")
    f.write(f"- ![Outliers Boxplot](quality_plots/outliers_boxplot.png)\n\n")

    f.write(f"### Class Imbalance\n")
    f.write(f"- Imbalance ratio: {imbalance_ratio:.2f}\n")
    for label, count in class_counts.items():
        f.write(f"- {label}: {count} ({count / len(df) * 100:.1f}%)\n")
    f.write(f"- ![Class Distribution](quality_plots/class_imbalance.png)\n\n")

    f.write("## Recommended Actions\n\n")
    if n_dupes > 0 or text_dupes > 0:
        f.write("- **Drop duplicates**: Remove exact and/or text-only duplicate rows\n")
    if iqr_outliers > 0:
        f.write("- **Handle outliers**: Clip or remove text length outliers (IQR or z-score method)\n")
    if missing.sum() > 0:
        f.write("- **Fill missing**: Median fill for numeric, mode fill for categorical, or drop rows\n")
    if imbalance_ratio > 1.5:
        f.write("- **Balance classes**: Undersample majority or oversample minority\n")
    if not issues:
        f.write("- Dataset is clean — no significant issues detected\n")

print(f"→ Report saved to {report}")
print("\n=== SUMMARY ===")
print(f"Issues found: {len(issues)}")
for issue_type, col, count, pct, severity in issues:
    print(f"  [{severity}] {issue_type} in '{col}': {count} ({pct}%)")
if not issues:
    print("  No significant issues found!")

df.drop(columns=["text_len"], inplace=True)
