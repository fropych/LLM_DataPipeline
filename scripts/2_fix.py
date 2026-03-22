"""Apply CONSERVATIVE cleaning: median fill, drop duplicates, clip IQR."""
import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/raw/dataset.parquet")
OUTPUT = Path("data/cleaned/dataset.parquet")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)
rows_before = len(df)
print(f"Input: {rows_before} rows")
changes = []

# 1. Missing values: median fill (numeric) / mode fill (categorical)
for col in df.columns:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        if df[col].dtype in ("float64", "int64"):
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            changes.append(f"Filled {n_missing} missing in '{col}' with median ({fill_val})")
        else:
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            changes.append(f"Filled {n_missing} missing in '{col}' with mode ('{fill_val}')")

if not any("Filled" in c for c in changes):
    changes.append("No missing values found")

# 2. Drop exact duplicates
n_dupes = df.duplicated().sum()
if n_dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    changes.append(f"Dropped {n_dupes} exact duplicate rows")
else:
    changes.append("No duplicate rows found")

# 3. Clip text length to IQR bounds
df["text_len"] = df["text"].str.len()
Q1 = df["text_len"].quantile(0.25)
Q3 = df["text_len"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

n_short = (df["text_len"] < lower).sum()
n_long = (df["text_len"] > upper).sum()

# Clip by truncating long texts and padding info for short ones
# For text data, "clipping" means truncating to upper bound chars
if n_long > 0:
    mask_long = df["text_len"] > upper
    df.loc[mask_long, "text"] = df.loc[mask_long, "text"].str[:int(upper)]
    changes.append(f"Clipped {n_long} long texts to {int(upper)} chars (IQR upper bound)")

if n_short > 0:
    changes.append(f"Note: {n_short} short texts below {int(lower)} chars (kept as-is, lower bound is below min)")

df = df.drop(columns=["text_len"])

rows_after = len(df)
df.to_parquet(OUTPUT, index=False)

print(f"\nOutput: {rows_after} rows → {OUTPUT}")
print(f"Rows removed: {rows_before - rows_after}")
print(f"\nChanges applied:")
for c in changes:
    print(f"  - {c}")
