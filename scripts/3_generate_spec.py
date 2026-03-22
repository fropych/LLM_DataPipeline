"""
Skill: generate_spec
Generates annotation_spec.md for AG News 4-class text classification.
Pulls real examples from the labeled dataset for each class.
"""

import pathlib
import pandas as pd
from datetime import datetime

EXAMPLES_PER_CLASS = 5  # >= 3 as required


def generate_spec(df: pd.DataFrame, output_path: pathlib.Path) -> str:
    """Generate Markdown annotation specification and write to output_path."""

    class_definitions = {
        "World": (
            "News about international politics, diplomacy, armed conflicts, "
            "governments, heads of state, elections, humanitarian crises, "
            "international organisations (UN, NATO, EU), treaties, sanctions, "
            "and cross-border events that are primarily geopolitical in nature."
        ),
        "Sports": (
            "News about any competitive sport, including professional leagues, "
            "international tournaments, Olympics, individual athletes, team "
            "results, transfers, coaching changes, match reports, and sports "
            "business directly tied to athletic competition."
        ),
        "Business": (
            "News about companies, financial markets, macroeconomics, trade, "
            "corporate strategy (mergers, acquisitions, IPOs), earnings and "
            "revenue, stock prices, banking, commodities markets, and economic "
            "policy with a primary focus on commerce and finance."
        ),
        "Sci/Tech": (
            "News about scientific research, technology products and companies, "
            "software, hardware, the internet, telecommunications, "
            "biotechnology, space exploration, medical breakthroughs, patents, "
            "and innovation — where the primary angle is technical or scientific."
        ),
    }

    edge_cases = [
        (
            "Tech company earnings",
            "If the story is primarily about a tech company's financial results "
            "(stock price, revenue, profit), label as **Business**. "
            "If it focuses on a new product or technology breakthrough, label as **Sci/Tech**.",
        ),
        (
            "Sports business / transfers",
            "Transfer fees, stadium financing, or team ownership deals are "
            "**Business** only when the financial angle dominates. "
            "Player transfers discussed in a sporting context are **Sports**.",
        ),
        (
            "Election violence / war protests",
            "Stories that mix political unrest with military action: "
            "if troops or weapons are central, use **World**; "
            "if the story is about a domestic protest or vote, still **World**.",
        ),
        (
            "Medical research vs. pharma business",
            "A clinical trial or genome study is **Sci/Tech**. "
            "A drug company's patent dispute or quarterly earnings is **Business**.",
        ),
        (
            "Oil prices",
            "Oil price movements driven by OPEC decisions or geopolitics → **World** or **Business**. "
            "When the primary angle is market/investor impact, use **Business**.",
        ),
        (
            "Olympic host city politics",
            "Use **World** if the story is about a bid decision by a government "
            "or political controversy; use **Sports** if the story is about "
            "athletes, results, or competition logistics.",
        ),
    ]

    # Pull real examples from the dataset
    examples: dict[str, list[str]] = {}
    for cls in class_definitions:
        subset = df[df["predicted_label"] == cls]["text"]
        # prefer high-confidence examples
        if "confidence" in df.columns:
            hi = df[(df["predicted_label"] == cls) & (df["confidence"] > 0.8)]["text"]
            if len(hi) >= EXAMPLES_PER_CLASS:
                subset = hi
        sampled = subset.sample(min(EXAMPLES_PER_CLASS, len(subset)), random_state=42)
        examples[cls] = sampled.tolist()

    # Build Markdown
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Annotation Specification — AG News Text Classification",
        "",
        f"**Generated:** {now}",
        "**Dataset:** AG News  |  **Rows:** {:,}".format(len(df)),
        "**Task type:** Single-label multi-class text classification",
        "**Classes:** World, Sports, Business, Sci/Tech",
        "",
        "---",
        "",
        "## 1. Task Description",
        "",
        "Given a short news headline + snippet (≤ 400 characters), assign **exactly one** label "
        "that best describes the primary topic of the article.",
        "",
        "Rules:",
        "- Choose the label that reflects the **dominant theme**, not a minor mention.",
        "- If genuinely ambiguous after reading the full snippet, mark the example as **uncertain** "
        "  (flag for human review).",
        "- Do **not** use knowledge outside the provided text — base the decision solely on what is written.",
        "",
        "---",
        "",
        "## 2. Class Definitions",
        "",
    ]

    for cls, definition in class_definitions.items():
        lines += [
            f"### {cls}",
            "",
            definition,
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Examples per Class",
        "",
    ]

    for cls, exs in examples.items():
        lines += [f"### {cls}", ""]
        for i, ex in enumerate(exs, 1):
            # Truncate for readability
            snippet = ex[:300].replace("\n", " ").strip()
            if len(ex) > 300:
                snippet += "..."
            lines += [f"**Example {i}:**", "", f"> {snippet}", ""]

    lines += [
        "---",
        "",
        "## 4. Edge Cases & Decision Rules",
        "",
    ]

    for title, rule in edge_cases:
        lines += [
            f"### {title}",
            "",
            rule,
            "",
        ]

    lines += [
        "---",
        "",
        "## 5. Labeling Workflow",
        "",
        "1. Read the full text snippet.",
        "2. Identify the **primary** topic (ignore secondary mentions).",
        "3. Apply the class definition. If still unclear, consult Section 4 edge cases.",
        "4. If confidence < 70 %, add the example to the **uncertain** pile for a second annotator.",
        "5. Record label in Label Studio; do **not** skip any example.",
        "",
        "---",
        "",
        "## 6. Quality Targets",
        "",
        "| Metric | Target |",
        "|--------|--------|",
        "| Inter-annotator agreement (Cohen's κ) | ≥ 0.70 |",
        "| Per-class coverage | ≥ 20 % each |",
        "| Uncertain / flagged rate | ≤ 15 % |",
        "",
    ]

    spec_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(spec_text, encoding="utf-8")
    return spec_text


def main() -> None:
    base = pathlib.Path(__file__).parent.parent
    src = base / "data" / "labeled" / "dataset.parquet"
    out = base / "reports" / "annotation_spec.md"

    print(f"Loading labeled dataset from {src} ...")
    df = pd.read_parquet(src)

    print("Generating annotation spec ...")
    spec = generate_spec(df, out)

    print(f"Saved annotation spec to {out}")
    print(f"  Lines: {len(spec.splitlines())}")
    print(f"  Characters: {len(spec):,}")


if __name__ == "__main__":
    main()
