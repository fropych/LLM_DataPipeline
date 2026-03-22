"""
Skill: export_to_labelstudio
Exports the labeled dataset to Label Studio JSON import format.

Two exports are produced:
  1. data/labeled/labelstudio_import.json      — all 127,600 rows
  2. data/labeled/labelstudio_flagged.json     — only flagged (low-confidence) rows

Label Studio pre-annotation format used:
  Each task has:
    - data.text          : the raw text
    - data.meta          : confidence, source, etc.
    - predictions        : one prediction object with a classification result
"""

import json
import pathlib
import uuid
import pandas as pd

CONFIDENCE_THRESHOLD = 0.5

# Label Studio classification tag names (must match the labeling config)
FROM_NAME = "label"
TO_NAME = "text"
LABEL_TYPE = "choices"

# To keep the full export manageable for Label Studio, we cap at a sample
# when exporting all rows; the flagged export is always complete.
FULL_EXPORT_SAMPLE = 5000  # set to None to export all 127,600 rows


def _task(row: pd.Series) -> dict:
    """Convert a DataFrame row to a Label Studio task dict."""
    task_id = str(uuid.uuid4())
    return {
        "id": task_id,
        "data": {
            "text": row["text"],
            "meta": {
                "confidence": round(float(row["confidence"]), 4),
                "source": str(row.get("source", "")),
                "collected_at": str(row.get("collected_at", "")),
            },
        },
        "predictions": [
            {
                "model_version": "keyword-rules-v1",
                "score": round(float(row["confidence"]), 4),
                "result": [
                    {
                        "id": str(uuid.uuid4()),
                        "type": LABEL_TYPE,
                        "value": {
                            "choices": [row["predicted_label"]],
                        },
                        "from_name": FROM_NAME,
                        "to_name": TO_NAME,
                    }
                ],
            }
        ],
    }


def export_to_labelstudio(
    df: pd.DataFrame,
    output_path: pathlib.Path,
    sample: int | None = None,
) -> list[dict]:
    """Convert df to Label Studio tasks and write JSON. Returns list of tasks."""
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=42)

    tasks = [_task(row) for _, row in df.iterrows()]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    return tasks


def main() -> None:
    base = pathlib.Path(__file__).parent.parent
    src = base / "data" / "labeled" / "dataset.parquet"
    out_full = base / "data" / "labeled" / "labelstudio_import.json"
    out_flagged = base / "data" / "labeled" / "labelstudio_flagged.json"

    # Also write the Label Studio labeling config for reference
    config_path = base / "data" / "labeled" / "labelstudio_config.xml"

    print(f"Loading {src} ...")
    df = pd.read_parquet(src)
    print(f"  Rows: {len(df):,}")

    # ---- Export: sampled full set ----
    print(f"\nExporting full dataset (sample={FULL_EXPORT_SAMPLE}) ...")
    tasks_full = export_to_labelstudio(df, out_full, sample=FULL_EXPORT_SAMPLE)
    size_kb = out_full.stat().st_size / 1024
    print(f"  Tasks written: {len(tasks_full):,}")
    print(f"  File size:     {size_kb:,.1f} KB")
    print(f"  Path:          {out_full}")

    # ---- Export: flagged only (complete, for HITL review) ----
    flagged_df = df[df["confidence"] < CONFIDENCE_THRESHOLD]
    print(f"\nExporting flagged examples (conf < {CONFIDENCE_THRESHOLD}) ...")
    tasks_flagged = export_to_labelstudio(flagged_df, out_flagged, sample=None)
    size_kb2 = out_flagged.stat().st_size / 1024
    print(f"  Tasks written: {len(tasks_flagged):,}")
    print(f"  File size:     {size_kb2:,.1f} KB")
    print(f"  Path:          {out_flagged}")

    # ---- Write Label Studio XML config ----
    ls_config = """\
<View>
  <Text name="text" value="$text" granularity="word"/>
  <Header value="Confidence: $meta.confidence"/>
  <Choices name="label" toName="text" choice="single" showInline="true">
    <Choice value="World"/>
    <Choice value="Sports"/>
    <Choice value="Business"/>
    <Choice value="Sci/Tech"/>
  </Choices>
</View>
"""
    config_path.write_text(ls_config, encoding="utf-8")
    print(f"\nLabel Studio labeling config saved to {config_path}")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("LABEL STUDIO EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Full export (sample):  {len(tasks_full):>7,} tasks")
    print(f"  Flagged export:        {len(tasks_flagged):>7,} tasks")
    print()
    print("  Flagged label distribution:")
    flagged_dist = flagged_df["predicted_label"].value_counts().sort_index()
    for cls, cnt in flagged_dist.items():
        pct = cnt / len(flagged_df) * 100
        print(f"    {cls:<12} {cnt:>7,}  ({pct:5.1f}%)")
    print()
    print("  Instructions to import into Label Studio:")
    print("    1. Create a new project with task type: Text Classification")
    print(f"    2. Paste the XML config from: {config_path}")
    print(f"    3. Import: {out_flagged}  (for HITL review)")
    print(f"       or:    {out_full}   (for full review sample)")


if __name__ == "__main__":
    main()
