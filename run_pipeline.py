"""
run_pipeline.py — End-to-end AG News text classification pipeline.

Steps:
  1. Data Collection  — load ag_news from HuggingFace
  2. EDA              — exploratory analysis and plots
  3. Data Quality     — detect issues, apply conservative cleaning, compare
  4. Annotation       — auto-label with keyword rules, generate spec,
                        export to LabelStudio, check quality
  5. HITL pause       — human reviews flagged_for_review.csv and saves
                        flagged_for_review_corrected.csv (or skips)
  6. Merge corrections— merge human corrections into reviewed dataset
  7. Active Learning  — entropy vs random sampling experiment + report
  8. Train model      — LogisticRegression + TF-IDF on reviewed dataset

Usage:
    python run_pipeline.py              # full pipeline with HITL pause
    python run_pipeline.py --skip-hitl  # skip HITL pause (use auto-labels as-is)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SCRIPTS = ROOT / "scripts"
DATA_LABELED = ROOT / "data" / "labeled"
REVIEW_FILE = DATA_LABELED / "flagged_for_review.csv"
REVIEW_CORRECTED = DATA_LABELED / "flagged_for_review_corrected.csv"


def run(script: str, label: str) -> None:
    """Run a script under the project root and exit on failure."""
    path = SCRIPTS / script
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  Script: {path.relative_to(ROOT)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s")


def hitl_pause() -> None:
    """
    Human-in-the-loop checkpoint.

    The AnnotationAgent flags examples with confidence < 0.5 and writes them
    to data/labeled/flagged_for_review.csv.

    Instructions for the reviewer:
      1. Open data/labeled/flagged_for_review.csv
      2. Inspect the 'auto_label' column; correct any wrong labels in that column
      3. Save the corrected file as data/labeled/flagged_for_review_corrected.csv
      4. Press Enter here to continue

    If no corrected file is found, the pipeline continues with the original
    auto-labels for the flagged examples.
    """
    print(f"\n{'='*60}")
    print("  STEP: HUMAN-IN-THE-LOOP REVIEW")
    print(f"{'='*60}")

    if not REVIEW_FILE.exists():
        print("  No flagged_for_review.csv found — skipping HITL.")
        return

    row_count = sum(1 for _ in open(REVIEW_FILE)) - 1  # subtract header
    print(f"\n  {row_count} examples flagged for review.")
    print(f"\n  File to review  : {REVIEW_FILE}")
    print(f"  Save corrections: {REVIEW_CORRECTED}")
    print("\n  Instructions:")
    print("    1. Open flagged_for_review.csv")
    print("    2. Correct wrong labels in the 'auto_label' column")
    print("    3. Save as flagged_for_review_corrected.csv in the same directory")
    print("    4. Press Enter to continue (or Ctrl-C to abort)")

    try:
        input("\n  [HITL] Press Enter when done (or just Enter to skip)... ")
    except KeyboardInterrupt:
        print("\n  Aborted by user.")
        sys.exit(1)

    if REVIEW_CORRECTED.exists():
        print(f"  Corrected file found: {REVIEW_CORRECTED}")
    else:
        print(
            "  No corrected file found — pipeline will use original auto-labels "
            "for flagged examples."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AG News ML pipeline.")
    parser.add_argument(
        "--skip-hitl",
        action="store_true",
        help="Skip the human-in-the-loop review pause.",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        metavar="N",
        help="Start from step N (1-8). Useful for resuming after a failure.",
    )
    args = parser.parse_args()

    steps = [
        # (step_number, script_filename, description)
        (1, "1_load_dataset.py", "Data Collection — load ag_news from HuggingFace"),
        (2, "1_eda.py",           "EDA — exploratory analysis and plots"),
        (3, "2_detect_issues.py", "Data Quality — detect issues"),
        (4, "2_fix.py",           "Data Quality — apply conservative cleaning"),
        (5, "2_compare.py",       "Data Quality — before/after comparison report"),
        (6, "3_auto_label.py",    "Annotation — keyword-based auto-labeling"),
        (7, "3_generate_spec.py", "Annotation — generate annotation spec"),
        (8, "3_export_labelstudio.py", "Annotation — export to LabelStudio format"),
        (9, "3_check_quality.py", "Annotation — quality metrics (kappa, confidence)"),
    ]

    print("\n" + "="*60)
    print("  AG NEWS CLASSIFICATION PIPELINE")
    print("  Starting from step:", args.from_step)
    print("  HITL pause:", "disabled" if args.skip_hitl else "enabled")
    print("="*60)

    for step_num, script, label in steps:
        if step_num < args.from_step:
            print(f"  [skip] Step {step_num}: {label}")
            continue
        run(script, f"Step {step_num}: {label}")

    # HITL pause between annotation and merge
    if not args.skip_hitl and args.from_step <= 10:
        hitl_pause()
    else:
        print("\n  [skip] HITL pause (--skip-hitl flag set)")

    # Merge corrections (step 10)
    if args.from_step <= 10:
        run("3_merge_corrections.py", "Step 10: Annotation — merge HITL corrections")

    # Active learning (step 11-12)
    if args.from_step <= 11:
        run("4_al_cycle.py", "Step 11: Active Learning — entropy vs random cycle")
    if args.from_step <= 12:
        run("4_al_report.py", "Step 12: Active Learning — generate report and plots")

    # Final model (step 13)
    if args.from_step <= 13:
        run("5_train.py", "Step 13: Train final model — LogisticRegression + TF-IDF")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print("\nOutput artefacts:")
    print(f"  Labeled dataset  : {ROOT / 'data' / 'labeled' / 'dataset_reviewed.parquet'}")
    print(f"  Trained model    : {ROOT / 'models' / 'model.joblib'}")
    print(f"  TF-IDF vectorizer: {ROOT / 'models' / 'vectorizer.joblib'}")
    print(f"  Model metrics    : {ROOT / 'reports' / 'model_metrics.md'}")
    print(f"  Quality report   : {ROOT / 'reports' / 'quality_report.md'}")
    print(f"  AL report        : {ROOT / 'reports' / 'al_report.md'}")
    print(f"  Data card        : {ROOT / 'reports' / 'data_card.md'}")
    print(f"  Confusion matrix : {ROOT / 'reports' / 'confusion_matrix.png'}")
    print(f"  Learning curves  : {ROOT / 'reports' / 'learning_curve.png'}")
    print()


if __name__ == "__main__":
    main()
