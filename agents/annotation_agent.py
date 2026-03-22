"""
AnnotationAgent — Assignment 3
Wraps all annotation skills into a single agent class.

Skills:
  auto_label(df)              -> DataFrame with predicted_label, confidence
  generate_spec(df, task)     -> AnnotationSpec (writes annotation_spec.md)
  check_quality(df_labeled)   -> dict with kappa, label_dist, confidence_mean, flagged
  export_to_labelstudio(df)   -> writes labelstudio_import.json, returns list[dict]

Usage:
    from agents.annotation_agent import AnnotationAgent
    agent = AnnotationAgent(modality='text')
    df_labeled = agent.auto_label(df)
    spec = agent.generate_spec(df_labeled, task='ag_news_classification')
    metrics = agent.check_quality(df_labeled)
    agent.export_to_labelstudio(df_labeled)
"""

from __future__ import annotations

import pathlib
import sys

# Add project root to path so scripts/ modules are importable
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import pandas as pd

# Lazy imports of skill modules
import importlib


class AnnotationAgent:
    """Single-agent facade over the three annotation skills."""

    def __init__(
        self,
        modality: str = "text",
        confidence_threshold: float = 0.5,
        output_dir: pathlib.Path | str | None = None,
    ) -> None:
        if modality != "text":
            raise ValueError(f"Unsupported modality: {modality!r}. Only 'text' is supported.")
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self.output_dir = pathlib.Path(output_dir) if output_dir else _ROOT

    # ------------------------------------------------------------------
    # Skill 1: auto_label
    # ------------------------------------------------------------------
    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply keyword-rule labeling. Returns df with predicted_label and confidence."""
        mod = importlib.import_module("3_auto_label")
        return mod.auto_label(df)

    # ------------------------------------------------------------------
    # Skill 2: generate_spec
    # ------------------------------------------------------------------
    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "text_classification",
    ) -> str:
        """Generate Markdown annotation spec. Returns spec text."""
        mod = importlib.import_module("3_generate_spec")
        out = self.output_dir / "reports" / "annotation_spec.md"
        return mod.generate_spec(df, out)

    # ------------------------------------------------------------------
    # Skill 3: check_quality
    # ------------------------------------------------------------------
    def check_quality(self, df_labeled: pd.DataFrame) -> dict:
        """
        Returns dict:
          {
            'kappa': float,
            'accuracy': float,
            'label_dist': {class: count},
            'confidence_mean': float,
            'flagged_count': int,
            'flagged_pct': float,
            'per_class_metrics': list[dict],
          }
        Also saves flagged examples and quality reports.
        """
        mod = importlib.import_module("3_check_quality")
        metrics, flagged_df = mod.check_quality(df_labeled)

        # Save flagged
        flagged_parquet = self.output_dir / "data" / "labeled" / "flagged_for_review.parquet"
        flagged_csv = self.output_dir / "data" / "labeled" / "flagged_for_review.csv"
        flagged_parquet.parent.mkdir(parents=True, exist_ok=True)
        flagged_df.to_parquet(flagged_parquet, index=False)
        flagged_df.to_csv(flagged_csv, index=False)

        # Flatten for public API
        return {
            "kappa": metrics["kappa"],
            "accuracy": metrics["accuracy"],
            "label_dist": metrics["predicted_dist"],
            "confidence_mean": metrics["confidence"]["mean"],
            "confidence_median": metrics["confidence"]["median"],
            "flagged_count": metrics["flagged_count"],
            "flagged_pct": metrics["flagged_pct"],
            "per_class_metrics": metrics["per_class_metrics"],
        }

    # ------------------------------------------------------------------
    # Skill 4: export_to_labelstudio
    # ------------------------------------------------------------------
    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        sample: int | None = 5000,
    ) -> list[dict]:
        """
        Exports to Label Studio JSON.
        Returns list of task dicts.
        Writes:
          data/labeled/labelstudio_import.json  (sampled)
          data/labeled/labelstudio_flagged.json (all flagged)
        """
        mod = importlib.import_module("3_export_labelstudio")
        out = self.output_dir / "data" / "labeled" / "labelstudio_import.json"
        tasks = mod.export_to_labelstudio(df, out, sample=sample)

        flagged_df = df[df["confidence"] < self.confidence_threshold]
        out_flagged = self.output_dir / "data" / "labeled" / "labelstudio_flagged.json"
        mod.export_to_labelstudio(flagged_df, out_flagged, sample=None)
        return tasks

    # ------------------------------------------------------------------
    # Convenience: run full pipeline
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, task: str = "text_classification") -> dict:
        """Run all four skills end-to-end. Returns quality metrics dict."""
        print("[AnnotationAgent] Step 1/4: auto_label ...")
        df_labeled = self.auto_label(df)

        print("[AnnotationAgent] Step 2/4: generate_spec ...")
        self.generate_spec(df_labeled, task=task)

        print("[AnnotationAgent] Step 3/4: check_quality ...")
        metrics = self.check_quality(df_labeled)

        print("[AnnotationAgent] Step 4/4: export_to_labelstudio ...")
        self.export_to_labelstudio(df_labeled)

        print("[AnnotationAgent] Done.")
        return metrics
