"""Load ag_news dataset from HuggingFace and save to data/raw/source_hf.csv."""

import os
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset

# Label mapping for ag_news
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

DATASET_NAME = "ag_news"
OUTPUT_PATH = "/Users/yromko/prj/itmo/ml_data/data/raw/source_hf.csv"


def main():
    print(f"Loading dataset '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)

    # Combine train and test splits
    all_splits = []
    for split_name, split_data in dataset.items():
        df = split_data.to_pandas()
        df["split"] = split_name
        all_splits.append(df)

    df = pd.concat(all_splits, ignore_index=True)
    print(f"Raw dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Standardize schema
    collected_at = datetime.now(timezone.utc).isoformat()

    result = pd.DataFrame({
        "text": df["text"],
        "label": df["label"].map(LABEL_MAP),
        "source": f"huggingface:{DATASET_NAME}",
        "collected_at": collected_at,
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"\nRows loaded: {len(result)}")
    print(f"Unique labels found: {sorted(result['label'].unique().tolist())}")
    print(f"Label distribution:\n{result['label'].value_counts().to_string()}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
