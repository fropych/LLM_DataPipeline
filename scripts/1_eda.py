"""EDA script for AG News dataset (data/raw/dataset.parquet)."""

import os
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.parquet"
PLOTS_DIR = BASE_DIR / "reports" / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NLTK stopwords
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)

# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------
print("=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"Total rows      : {len(df):,}")
print(f"Columns         : {df.columns.tolist()}")
print()
print("Dtypes:")
print(df.dtypes.to_string())
print()
print("Missing values:")
print(df.isnull().sum().to_string())
print()
print("Class distribution:")
label_counts = df["label"].value_counts().sort_index()
print(label_counts.to_string())
print()

# Compute text lengths
df["text_len"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

print("Text length (chars) statistics:")
print(df["text_len"].describe().to_string())
print()
print("Word count statistics:")
print(df["word_count"].describe().to_string())

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

def save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Class distribution bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(label_counts.index, label_counts.values, color=PALETTE)
for bar, val in zip(bars, label_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 150,
        f"{val:,}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_ylim(0, label_counts.max() * 1.12)
sns.despine(ax=ax)
save(fig, "class_distribution.png")

# ---------------------------------------------------------------------------
# 2. Text length histogram (chars)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, xlabel, title in [
    (axes[0], "text_len", "Characters", "Text Length Distribution (chars)"),
    (axes[1], "word_count", "Words", "Text Length Distribution (words)"),
]:
    ax.hist(df[col], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(df[col].median(), color="#DD8452", linestyle="--", linewidth=1.5,
               label=f"Median = {df[col].median():.0f}")
    ax.axvline(df[col].mean(), color="#C44E52", linestyle="-.", linewidth=1.5,
               label=f"Mean = {df[col].mean():.0f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

fig.tight_layout()
save(fig, "text_lengths.png")

# ---------------------------------------------------------------------------
# 3. Top-20 most frequent words (stopwords removed)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


word_freq: Counter = Counter()
for text in df["text"]:
    word_freq.update(tokenize(text))

top20 = word_freq.most_common(20)
words, counts = zip(*top20)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(words[::-1], counts[::-1], color="#4C72B0")
ax.set_title("Top-20 Most Frequent Words (stopwords removed)", fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency")
for bar, val in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8)
sns.despine(ax=ax)
fig.tight_layout()
save(fig, "top_words.png")

# ---------------------------------------------------------------------------
# 4. Source distribution pie chart
# ---------------------------------------------------------------------------
source_counts = df["source"].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedge_props = {"edgecolor": "white", "linewidth": 1.5}
ax.pie(
    source_counts.values,
    labels=source_counts.index,
    autopct="%1.1f%%",
    colors=PALETTE[: len(source_counts)],
    wedgeprops=wedge_props,
    startangle=140,
)
ax.set_title("Source Distribution", fontsize=13, fontweight="bold")
save(fig, "source_distribution.png")

# ---------------------------------------------------------------------------
# Summary stats table
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY STATS TABLE")
print("=" * 60)
summary = pd.DataFrame({
    "Metric": [
        "Total rows",
        "Columns",
        "Classes",
        "Rows per class",
        "Unique sources",
        "Missing values",
        "Avg text length (chars)",
        "Median text length (chars)",
        "Min text length (chars)",
        "Max text length (chars)",
        "Avg word count",
        "Median word count",
        "Min word count",
        "Max word count",
        "Vocabulary size (no stopwords)",
    ],
    "Value": [
        f"{len(df):,}",
        str(df.shape[1]),
        str(df['label'].nunique()),
        f"{len(df) // df['label'].nunique():,}",
        str(df['source'].nunique()),
        str(df.isnull().sum().sum()),
        f"{df['text_len'].mean():.1f}",
        f"{df['text_len'].median():.0f}",
        f"{df['text_len'].min()}",
        f"{df['text_len'].max():,}",
        f"{df['word_count'].mean():.1f}",
        f"{df['word_count'].median():.0f}",
        f"{df['word_count'].min()}",
        f"{df['word_count'].max()}",
        f"{len(word_freq):,}",
    ],
})
print(summary.to_string(index=False))
print()
print("Top-20 words:", ", ".join(words))
print()
print("All plots saved to:", PLOTS_DIR)
