"""
Active Learning Report Generator
Reads AL histories, produces plots, markdown report, and Jupyter notebook.
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = "/Users/yromko/prj/itmo/ml_data"
DATA_DIR    = f"{BASE}/data"
REPORTS_DIR = f"{BASE}/reports"
NB_DIR      = f"{BASE}/notebooks"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(NB_DIR, exist_ok=True)

# ── Load histories ────────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/al_history_entropy.json") as f:
    entropy_hist = json.load(f)
with open(f"{DATA_DIR}/al_history_random.json") as f:
    random_hist  = json.load(f)

def extract(hist, key):
    return [h[key] for h in hist]

e_n   = extract(entropy_hist, "n_labeled")
e_acc = extract(entropy_hist, "accuracy")
e_f1  = extract(entropy_hist, "f1")

r_n   = extract(random_hist,  "n_labeled")
r_acc = extract(random_hist,  "accuracy")
r_f1  = extract(random_hist,  "f1")

# ── Plotting helpers ──────────────────────────────────────────────────────────
STYLE = {"linewidth": 2.2, "marker": "o", "markersize": 6}
COLORS = {"entropy": "#E15759", "random": "#4E79A7"}

def _finalize(ax, ylabel, title, path):
    ax.set_xlabel("Number of Labeled Examples", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(alpha=0.35, linestyle="--")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ── Plot 1 – Accuracy ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(e_n, e_acc, label="Entropy",  color=COLORS["entropy"], **STYLE)
ax.plot(r_n, r_acc, label="Random",   color=COLORS["random"],  **STYLE)
_finalize(ax, "Accuracy", "Active Learning – Accuracy vs Labeled Set Size",
          f"{REPORTS_DIR}/learning_curve_accuracy.png")

# ── Plot 2 – F1 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(e_n, e_f1, label="Entropy",  color=COLORS["entropy"], **STYLE)
ax.plot(r_n, r_f1, label="Random",   color=COLORS["random"],  **STYLE)
_finalize(ax, "Macro F1", "Active Learning – Macro-F1 vs Labeled Set Size",
          f"{REPORTS_DIR}/learning_curve_f1.png")

# ── Plot 3 – Combined (2 subplots) ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(e_n, e_acc, label="Entropy", color=COLORS["entropy"], **STYLE)
axes[0].plot(r_n, r_acc, label="Random",  color=COLORS["random"],  **STYLE)
axes[0].set_title("Accuracy", fontweight="bold")
axes[0].set_xlabel("N labeled"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(alpha=0.3, linestyle="--")

axes[1].plot(e_n, e_f1, label="Entropy", color=COLORS["entropy"], **STYLE)
axes[1].plot(r_n, r_f1, label="Random",  color=COLORS["random"],  **STYLE)
axes[1].set_title("Macro F1", fontweight="bold")
axes[1].set_xlabel("N labeled"); axes[1].set_ylabel("Macro F1")
axes[1].legend(); axes[1].grid(alpha=0.3, linestyle="--")

fig.suptitle("Active Learning: Entropy vs Random — AG News (4 classes)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {REPORTS_DIR}/learning_curve.png")

# ── Analysis: examples saved ──────────────────────────────────────────────────
e_final_f1  = e_f1[-1]
e_final_n   = e_n[-1]
r_final_f1  = r_f1[-1]
r_final_n   = r_n[-1]

# Crossover: first n in random that reaches/exceeds entropy's final F1
crossover_n = None
for n, f in zip(r_n, r_f1):
    if f >= e_final_f1:
        crossover_n = n
        break

if crossover_n is not None:
    examples_saved = crossover_n - e_final_n
    saved_msg = (
        f"Entropy reaches F1={e_final_f1:.4f} with {e_final_n} labels; "
        f"Random needs {crossover_n} labels for the same quality.\n"
        f"Examples saved by entropy: {examples_saved}"
    )
else:
    examples_saved = None
    saved_msg = (
        f"Random never reached entropy's final F1={e_final_f1:.4f} "
        f"within {r_final_n} labeled examples.\n"
        f"At {r_final_n} labels, random F1={r_final_f1:.4f} vs entropy F1={e_final_f1:.4f} "
        f"(entropy leads by {e_final_f1 - r_final_f1:+.4f})."
    )

print("\n" + "=" * 60)
print("EXAMPLES SAVED ANALYSIS")
print("=" * 60)
print(saved_msg)
print("=" * 60)

# ── Markdown report ───────────────────────────────────────────────────────────
report_md = textwrap.dedent(f"""\
# Active Learning Experiment Report
**Dataset:** AG News (4 classes: World, Sports, Business, Sci/Tech)
**Date:** 2026-03-22
**Model:** Logistic Regression + TF-IDF (max_features=20 000, sublinear_tf=True)
**Initial labeled set:** 50 (random)
**Iterations:** 5 | **Batch size:** 20 per iteration

---

## Setup

| Parameter | Value |
|-----------|-------|
| Initial N labeled | 50 |
| Iterations | 5 |
| Batch per iteration | 20 |
| Final N labeled | 150 |
| Test set size | 4 000 |
| Feature extraction | TF-IDF (20k features) |
| Classifier | Logistic Regression (C=1.0, max_iter=1000) |

---

## Results

### Entropy Strategy
| Iter | N Labeled | Accuracy | Macro F1 |
|------|-----------|----------|----------|
""")

for h in entropy_hist:
    report_md += f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"

report_md += "\n### Random Strategy\n"
report_md += "| Iter | N Labeled | Accuracy | Macro F1 |\n"
report_md += "|------|-----------|----------|----------|\n"
for h in random_hist:
    report_md += f"| {h['iteration']} | {h['n_labeled']} | {h['accuracy']:.4f} | {h['f1']:.4f} |\n"

report_md += textwrap.dedent(f"""
---

## Final Comparison (N=150 labeled)

| Strategy | Accuracy | Macro F1 |
|----------|----------|----------|
| Entropy  | {e_acc[-1]:.4f}   | {e_f1[-1]:.4f}   |
| Random   | {r_acc[-1]:.4f}   | {r_f1[-1]:.4f}   |
| Delta    | {e_acc[-1]-r_acc[-1]:+.4f}  | {e_f1[-1]-r_f1[-1]:+.4f}  |

---

## Examples Saved Analysis

{saved_msg}

---

## Conclusions

1. **Entropy sampling converges faster in F1** — by the final iteration (N=150),
   entropy achieves Macro-F1={e_final_f1:.4f} vs Random's {r_final_f1:.4f}, a delta of
   {e_final_f1 - r_final_f1:+.4f}.

2. **Non-monotonic entropy curve** — entropy selection can temporarily reduce
   accuracy when it queries hard, near-boundary samples that are difficult for
   a small model to generalize from.  This is expected behavior and smooths out
   with more iterations.

3. **Both strategies start identically** — iteration 0 is identical because the
   initial 50 labels are the same random seed.

4. **Recommendation** — for annotation budgets below 200 examples, entropy
   sampling provides better F1 quality per label cost on this 4-class text
   classification task.

## Plots

- `reports/learning_curve_accuracy.png` — Accuracy vs N labeled
- `reports/learning_curve_f1.png`       — Macro-F1 vs N labeled
- `reports/learning_curve.png`          — Combined side-by-side view
""")

report_path = f"{REPORTS_DIR}/al_report.md"
with open(report_path, "w") as f:
    f.write(report_md)
print(f"Saved: {report_path}")

# ── Jupyter Notebook ──────────────────────────────────────────────────────────
nb_cells_source = [
    # Cell 0 – title markdown
    ("markdown", """\
# Active Learning Experiment — AG News
**Entropy vs Random sampling strategies**
Dataset: AG News (4 classes)  |  Model: TF-IDF + Logistic Regression
"""),
    # Cell 1 – imports
    ("code", """\
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline
"""),
    # Cell 2 – load histories
    ("code", f"""\
BASE = "/Users/yromko/prj/itmo/ml_data"

with open(f"{{BASE}}/data/al_history_entropy.json") as f:
    entropy_hist = json.load(f)
with open(f"{{BASE}}/data/al_history_random.json") as f:
    random_hist  = json.load(f)

def extract(hist, key):
    return [h[key] for h in hist]

e_n, e_acc, e_f1 = extract(entropy_hist,"n_labeled"), extract(entropy_hist,"accuracy"), extract(entropy_hist,"f1")
r_n, r_acc, r_f1 = extract(random_hist, "n_labeled"), extract(random_hist, "accuracy"), extract(random_hist, "f1")

pd.DataFrame(entropy_hist).rename(columns=lambda c: "entropy_"+c if c!="iteration" and c!="n_labeled" else c)
"""),
    # Cell 3 – accuracy plot
    ("code", """\
COLORS = {"entropy": "#E15759", "random": "#4E79A7"}
STYLE  = dict(linewidth=2.2, marker="o", markersize=6)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(e_n, e_acc, label="Entropy", color=COLORS["entropy"], **STYLE)
ax.plot(r_n, r_acc, label="Random",  color=COLORS["random"],  **STYLE)
ax.set_xlabel("Number of Labeled Examples", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Active Learning – Accuracy vs Labeled Set Size", fontweight="bold")
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(alpha=0.35, linestyle="--"); ax.legend(fontsize=11)
plt.tight_layout(); plt.show()
"""),
    # Cell 4 – F1 plot
    ("code", """\
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(e_n, e_f1, label="Entropy", color=COLORS["entropy"], **STYLE)
ax.plot(r_n, r_f1, label="Random",  color=COLORS["random"],  **STYLE)
ax.set_xlabel("Number of Labeled Examples", fontsize=12)
ax.set_ylabel("Macro F1", fontsize=12)
ax.set_title("Active Learning – Macro-F1 vs Labeled Set Size", fontweight="bold")
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.grid(alpha=0.35, linestyle="--"); ax.legend(fontsize=11)
plt.tight_layout(); plt.show()
"""),
    # Cell 5 – combined plot
    ("code", """\
fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, metric, e_vals, r_vals, ylabel in [
    (axes[0], "Accuracy", e_acc, r_acc, "Accuracy"),
    (axes[1], "Macro F1", e_f1,  r_f1,  "Macro F1"),
]:
    ax.plot(e_n, e_vals, label="Entropy", color=COLORS["entropy"], **STYLE)
    ax.plot(r_n, r_vals, label="Random",  color=COLORS["random"],  **STYLE)
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("N labeled"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.3, linestyle="--")
fig.suptitle("Active Learning: Entropy vs Random — AG News (4 classes)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout(); plt.show()
"""),
    # Cell 6 – analysis table
    ("code", f"""\
summary = {{
    "Strategy": ["Entropy", "Random", "Delta"],
    "Accuracy": [e_acc[-1], r_acc[-1], round(e_acc[-1]-r_acc[-1], 4)],
    "Macro F1": [e_f1[-1],  r_f1[-1],  round(e_f1[-1]-r_f1[-1],  4)],
    "N Labeled":[e_n[-1],   r_n[-1],   0],
}}
pd.DataFrame(summary)
"""),
    # Cell 7 – examples saved
    ("code", f"""\
e_final_f1 = e_f1[-1]
crossover_n = next((n for n, f in zip(r_n, r_f1) if f >= e_final_f1), None)
if crossover_n:
    print(f"Entropy reaches F1={{e_final_f1:.4f}} with {{e_n[-1]}} labels.")
    print(f"Random needs {{crossover_n}} labels for the same quality.")
    print(f"Examples saved: {{crossover_n - e_n[-1]}}")
else:
    print(f"Random never reached entropy's final F1={{e_final_f1:.4f}} within {{r_n[-1]}} labels.")
    print(f"Entropy leads Random by {{e_final_f1 - r_f1[-1]:+.4f}} F1 at N=150.")
"""),
    # Cell 8 – conclusions markdown
    ("markdown", """\
## Conclusions

1. **Entropy sampling converges faster in F1** — entropy achieves higher Macro-F1
   than random at the same labeling budget.
2. **Non-monotonic entropy curve** — hard near-boundary samples can temporarily
   hurt a small model; this resolves with more iterations.
3. **Recommendation** — for budgets < 200 examples, entropy sampling is
   preferred for 4-class text classification.
"""),
]

import nbformat

nb = nbformat.v4.new_notebook()
for cell_type, src in nb_cells_source:
    if cell_type == "markdown":
        nb.cells.append(nbformat.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbformat.v4.new_code_cell(src))

nb_path = f"{NB_DIR}/al_experiment.ipynb"
with open(nb_path, "w") as f:
    nbformat.write(nb, f)
print(f"Saved: {nb_path}")

# ── Final summary printout ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"ENTROPY  final: acc={e_acc[-1]:.4f}  f1={e_f1[-1]:.4f}  (N={e_n[-1]})")
print(f"RANDOM   final: acc={r_acc[-1]:.4f}  f1={r_f1[-1]:.4f}  (N={r_n[-1]})")
print(saved_msg)
print("=" * 60)
