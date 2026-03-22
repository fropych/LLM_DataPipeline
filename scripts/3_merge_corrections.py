"""Merge HITL corrections into labeled dataset."""
import pandas as pd
from pathlib import Path

LABELED = Path("data/labeled/dataset.parquet")
CORRECTIONS = Path("data/labeled/review_queue_corrected.csv")
OUTPUT = Path("data/labeled/dataset_reviewed.parquet")

df = pd.read_parquet(LABELED)
print(f"Labeled dataset: {len(df)} rows")

if CORRECTIONS.exists():
    corr = pd.read_csv(CORRECTIONS)
    print(f"Corrections file: {len(corr)} rows")

    # Match on text column and apply corrected_label
    if "corrected_label" in corr.columns:
        corr_map = dict(zip(corr["text"], corr["corrected_label"]))
        mask = df["text"].isin(corr_map)
        n_matched = mask.sum()
        n_changed = 0
        for idx in df[mask].index:
            old = df.at[idx, "predicted_label"]
            new = corr_map[df.at[idx, "text"]]
            if old != new:
                df.at[idx, "predicted_label"] = new
                n_changed += 1
        print(f"\nMatched {n_matched} rows from corrections")
        print(f"Changed {n_changed} labels (corrected by human)")
        print(f"{n_matched - n_changed} labels confirmed (no change needed)")
    else:
        print("WARNING: No 'corrected_label' column in corrections file")
else:
    print("WARNING: No corrections file found, using auto-labels as-is")

df.to_parquet(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT} ({len(df)} rows)")
