"""
Skill: auto_label
Method: keyword rules
Dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)

Applies keyword-based heuristic matching to each text.
Confidence = (max_score - second_max_score) / (max_score + epsilon)
or 1.0 when only one class has matches.
"""

import re
import pathlib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
KEYWORD_RULES: dict[str, list[str]] = {
    "World": [
        "war", "government", "president", "minister", "country", "nation",
        "military", "troops", "UN", "peace", "diplomatic", "refugee",
        "terrorism", "attack", "bomb", "election", "vote", "democracy",
        "humanitarian", "sanctions", "treaty", "conflict", "ceasefire",
        "protest", "embassy",
    ],
    "Sports": [
        "game", "team", "player", "coach", "score", "win", "loss",
        "championship", "league", "tournament", "season", "match", "goal",
        "race", "olympic", "medal", "athlete", "football", "basketball",
        "baseball", "soccer", "tennis", "hockey", "boxing", "swimming",
    ],
    "Business": [
        "market", "stock", "company", "profit", "revenue", "bank", "trade",
        "economy", "price", "investor", "CEO", "shares", "earnings", "growth",
        "acquisition", "merger", "billion", "million", "fiscal", "quarterly",
        "dividend", "IPO", "Wall Street", "interest rate", "inflation",
    ],
    "Sci/Tech": [
        "software", "computer", "internet", "technology", "digital",
        "research", "science", "data", "network", "chip", "processor",
        "Microsoft", "Google", "Apple", "device", "mobile", "wireless",
        "security", "virus", "patent", "innovation", "algorithm", "satellite",
        "genome", "robot",
    ],
}

CLASSES = list(KEYWORD_RULES.keys())

# Pre-compile patterns (case-insensitive, whole-word for single-word terms)
_PATTERNS: dict[str, list[re.Pattern]] = {}
for _cls, _kws in KEYWORD_RULES.items():
    _pats = []
    for kw in _kws:
        if " " in kw:
            # multi-word phrase: literal match
            _pats.append(re.compile(re.escape(kw), re.IGNORECASE))
        else:
            # single word: word-boundary match
            _pats.append(re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
    _PATTERNS[_cls] = _pats


def _score_text(text: str) -> dict[str, float]:
    """Return raw keyword hit counts per class."""
    scores: dict[str, float] = {}
    for cls, pats in _PATTERNS.items():
        count = sum(1 for p in pats if p.search(text))
        scores[cls] = float(count)
    return scores


def _predict(text: str) -> tuple[str, float]:
    """Return (predicted_label, confidence) for a single text."""
    scores = _score_text(text)
    total = sum(scores.values())

    if total == 0:
        # No keyword match — fall back to 'Sci/Tech' with low confidence
        return "Sci/Tech", 0.25

    # Sort descending
    sorted_cls = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_cls, best_score = sorted_cls[0]
    second_score = sorted_cls[1][1] if len(sorted_cls) > 1 else 0.0

    # Confidence: margin normalised to [0, 1]
    epsilon = 1e-6
    confidence = (best_score - second_score) / (best_score + epsilon)
    # Clamp to (0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # If tied (confidence == 0) use proportion of top class
    if confidence == 0.0:
        confidence = best_score / total

    return best_cls, confidence


def auto_label(df: pd.DataFrame) -> pd.DataFrame:
    """Apply keyword rules and return df with predicted_label and confidence."""
    results = df["text"].map(_predict)
    df = df.copy()
    df["predicted_label"] = results.map(lambda x: x[0])
    df["confidence"] = results.map(lambda x: x[1])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    base = pathlib.Path(__file__).parent.parent
    src = base / "data" / "cleaned" / "dataset.parquet"
    dst_dir = base / "data" / "labeled"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "dataset.parquet"

    print(f"Loading {src} ...")
    df = pd.read_parquet(src)
    print(f"  Rows: {len(df):,}")

    print("Applying keyword rules ...")
    df_labeled = auto_label(df)

    # Save
    df_labeled.to_parquet(dst, index=False)
    print(f"Saved labeled dataset to {dst}\n")

    # --- Report ---
    print("=" * 55)
    print("LABEL DISTRIBUTION (predicted)")
    print("=" * 55)
    dist = df_labeled["predicted_label"].value_counts().sort_index()
    for cls, cnt in dist.items():
        pct = cnt / len(df_labeled) * 100
        print(f"  {cls:<12} {cnt:>7,}  ({pct:5.1f}%)")

    print()
    print("=" * 55)
    print("CONFIDENCE STATS")
    print("=" * 55)
    conf = df_labeled["confidence"]
    print(f"  Mean:   {conf.mean():.4f}")
    print(f"  Median: {conf.median():.4f}")
    print(f"  Min:    {conf.min():.4f}")
    print(f"  Max:    {conf.max():.4f}")
    print(f"  Std:    {conf.std():.4f}")

    low_conf_threshold = 0.5
    flagged = (conf < low_conf_threshold).sum()
    print(f"\n  Flagged (conf < {low_conf_threshold}): {flagged:,}  "
          f"({flagged / len(df_labeled) * 100:.1f}%)")


if __name__ == "__main__":
    main()
