"""
Skill: check_quality
Computes annotation quality metrics for the labeled AG News dataset.

Metrics:
  - Label distribution (predicted vs. ground-truth)
  - Cohen's kappa  (predicted vs. ground-truth labels)
  - Confidence statistics
  - Flagged examples (confidence < CONFIDENCE_THRESHOLD)

Flagged examples are saved to data/labeled/flagged_for_review.parquet
and data/labeled/flagged_for_review.csv for human review (HITL).
"""

import pathlib
import json
import pandas as pd
import numpy as np
from collections import Counter

CONFIDENCE_THRESHOLD = 0.5  # flag examples below this value
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


# ---------------------------------------------------------------------------
# Cohen's kappa (multi-class)
# ---------------------------------------------------------------------------
def cohens_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute multi-class Cohen's kappa from two label series."""
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    n = len(y_true)
    if n == 0:
        return 0.0

    # Observed agreement
    p_o = (y_true == y_pred).sum() / n

    # Expected agreement
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    p_e = sum(
        (true_counts.get(lbl, 0) / n) * (pred_counts.get(lbl, 0) / n)
        for lbl in labels
    )

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Per-class precision / recall / F1 (lightweight, no sklearn dependency)
# ---------------------------------------------------------------------------
def per_class_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "class": cls,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int((y_true == cls).sum()),
            }
        )
    return pd.DataFrame(rows)


def check_quality(df: pd.DataFrame) -> dict:
    """Run all quality checks and return a metrics dict."""
    y_true = df["label"]
    y_pred = df["predicted_label"]
    conf = df["confidence"]

    # ---- Accuracy ----
    accuracy = (y_true == y_pred).mean()

    # ---- Cohen's kappa ----
    kappa = cohens_kappa(y_true, y_pred)

    # ---- Label distributions ----
    gt_dist = y_true.value_counts().to_dict()
    pred_dist = y_pred.value_counts().to_dict()

    # ---- Confidence stats ----
    conf_stats = {
        "mean": float(conf.mean()),
        "median": float(conf.median()),
        "min": float(conf.min()),
        "max": float(conf.max()),
        "std": float(conf.std()),
    }

    # ---- Flagged (low-confidence) ----
    flagged_mask = conf < CONFIDENCE_THRESHOLD
    flagged_count = int(flagged_mask.sum())
    flagged_pct = flagged_count / len(df) * 100

    # ---- Per-class metrics ----
    pcm = per_class_metrics(y_true, y_pred)

    metrics = {
        "n_total": len(df),
        "accuracy": round(float(accuracy), 4),
        "kappa": round(float(kappa), 4),
        "confidence": conf_stats,
        "flagged_count": flagged_count,
        "flagged_pct": round(flagged_pct, 2),
        "ground_truth_dist": {k: int(v) for k, v in gt_dist.items()},
        "predicted_dist": {k: int(v) for k, v in pred_dist.items()},
        "per_class_metrics": pcm.to_dict(orient="records"),
    }
    return metrics, df[flagged_mask]


def main() -> None:
    base = pathlib.Path(__file__).parent.parent
    src = base / "data" / "labeled" / "dataset.parquet"
    flagged_parquet = base / "data" / "labeled" / "flagged_for_review.parquet"
    flagged_csv = base / "data" / "labeled" / "flagged_for_review.csv"
    report_path = base / "reports" / "quality_report.json"
    report_md = base / "reports" / "quality_report.md"

    print(f"Loading {src} ...")
    df = pd.read_parquet(src)
    print(f"  Rows: {len(df):,}")

    print("Computing quality metrics ...")
    metrics, flagged_df = check_quality(df)

    # Save flagged examples
    flagged_df.to_parquet(flagged_parquet, index=False)
    flagged_df.to_csv(flagged_csv, index=False)
    print(f"Saved {len(flagged_df):,} flagged examples to:")
    print(f"  {flagged_parquet}")
    print(f"  {flagged_csv}")

    # Save JSON report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved quality report to {report_path}")

    # Save Markdown report
    pcm_df = pd.DataFrame(metrics["per_class_metrics"])
    md_lines = [
        "# Quality Report — AG News Annotation",
        "",
        f"**Total examples:** {metrics['n_total']:,}",
        f"**Accuracy (vs ground truth):** {metrics['accuracy']:.4f}",
        f"**Cohen's kappa:** {metrics['kappa']:.4f}",
        "",
        "## Confidence Stats",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean   | {metrics['confidence']['mean']:.4f} |",
        f"| Median | {metrics['confidence']['median']:.4f} |",
        f"| Min    | {metrics['confidence']['min']:.4f} |",
        f"| Max    | {metrics['confidence']['max']:.4f} |",
        f"| Std    | {metrics['confidence']['std']:.4f} |",
        "",
        "## Flagged Examples",
        "",
        f"Threshold: confidence < {CONFIDENCE_THRESHOLD}",
        "",
        f"**Flagged count:** {metrics['flagged_count']:,} "
        f"({metrics['flagged_pct']:.1f}%)",
        "",
        "## Label Distribution",
        "",
        "| Class | Ground Truth | Predicted |",
        "|-------|-------------|-----------|",
    ]
    for cls in CLASSES:
        gt = metrics["ground_truth_dist"].get(cls, 0)
        pr = metrics["predicted_dist"].get(cls, 0)
        md_lines.append(f"| {cls} | {gt:,} | {pr:,} |")

    md_lines += [
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for row in metrics["per_class_metrics"]:
        md_lines.append(
            f"| {row['class']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['support']:,} |"
        )

    report_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved Markdown report to {report_md}")

    # ---- Print summary ----
    print()
    print("=" * 60)
    print("QUALITY METRICS SUMMARY")
    print("=" * 60)
    print(f"  Total examples:     {metrics['n_total']:>10,}")
    print(f"  Accuracy:           {metrics['accuracy']:>10.4f}")
    print(f"  Cohen's kappa:      {metrics['kappa']:>10.4f}")
    print()
    print("  Confidence:")
    for k, v in metrics["confidence"].items():
        print(f"    {k:<8} {v:.4f}")
    print()
    print(f"  Flagged (conf < {CONFIDENCE_THRESHOLD}): "
          f"{metrics['flagged_count']:,}  ({metrics['flagged_pct']:.1f}%)")
    print()
    print("  Label distribution (Ground Truth | Predicted):")
    for cls in CLASSES:
        gt = metrics["ground_truth_dist"].get(cls, 0)
        pr = metrics["predicted_dist"].get(cls, 0)
        print(f"    {cls:<12}  GT={gt:>7,}  Pred={pr:>7,}")
    print()
    print("  Per-class metrics:")
    print(f"    {'Class':<12} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}")
    for row in metrics["per_class_metrics"]:
        print(
            f"    {row['class']:<12} {row['precision']:>7.4f} "
            f"{row['recall']:>7.4f} {row['f1']:>7.4f} {row['support']:>9,}"
        )


if __name__ == "__main__":
    main()
