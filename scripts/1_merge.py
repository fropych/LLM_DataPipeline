"""Merge all source_*.csv files from data/raw/ into data/raw/dataset.parquet."""

import glob
import os
import sys

import pandas as pd
import yaml

RAW_DIR = "/Users/yromko/prj/itmo/ml_data/data/raw"
OUTPUT_PATH = os.path.join(RAW_DIR, "dataset.parquet")
CONFIG_PATH = "/Users/yromko/prj/itmo/ml_data/config.yaml"
REQUIRED_COLUMNS = {"text", "label", "source", "collected_at"}


def validate_schema(df: pd.DataFrame, path: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: {path} is missing columns: {missing}")
        sys.exit(1)


def main():
    pattern = os.path.join(RAW_DIR, "source_*.csv")
    source_files = sorted(glob.glob(pattern))

    if not source_files:
        print(f"No source files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(source_files)} source file(s): {[os.path.basename(f) for f in source_files]}")

    frames = []
    source_meta = []

    for path in source_files:
        df = pd.read_csv(path)
        validate_schema(df, path)
        row_count_raw = len(df)

        # Infer source type and dataset name from the 'source' column
        source_value = df["source"].iloc[0] if len(df) > 0 else "unknown"
        if source_value.startswith("huggingface:"):
            src_type = "hf"
            dataset_name = source_value.split("huggingface:", 1)[1]
            url = f"https://huggingface.co/datasets/{dataset_name}"
        elif source_value.startswith("api:"):
            src_type = "api"
            dataset_name = source_value.split("api:", 1)[1]
            url = dataset_name
        elif source_value.startswith("scrape:"):
            src_type = "scrape"
            dataset_name = source_value.split("scrape:", 1)[1]
            url = dataset_name
        else:
            src_type = "unknown"
            dataset_name = source_value
            url = ""

        source_meta.append({
            "file": os.path.basename(path),
            "type": src_type,
            "dataset": dataset_name,
            "url": url,
            "rows_raw": row_count_raw,
        })

        frames.append(df)
        print(f"  {os.path.basename(path)}: {row_count_raw} rows, source={source_value!r}")

    combined = pd.concat(frames, ignore_index=True)
    total_before = len(combined)

    # Deduplicate by text
    combined_deduped = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    duplicates_removed = total_before - len(combined_deduped)

    combined_deduped.to_parquet(OUTPUT_PATH, index=False)

    # Write config.yaml
    config = {
        "dataset": {
            "output": OUTPUT_PATH,
            "total_rows": len(combined_deduped),
            "duplicates_removed": duplicates_removed,
        },
        "sources": source_meta,
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nTotal rows before dedup : {total_before}")
    print(f"Duplicates removed      : {duplicates_removed}")
    print(f"Total rows after dedup  : {len(combined_deduped)}")
    print(f"\nRows per source:")
    for meta in source_meta:
        print(f"  {meta['file']}: {meta['rows_raw']}")
    print(f"\nSaved parquet : {OUTPUT_PATH}")
    print(f"Saved config  : {CONFIG_PATH}")


if __name__ == "__main__":
    main()
