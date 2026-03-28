---
name: 4-al-report
description: Generate AL report — learning curves (entropy vs random), examples saved analysis
allowed-tools: Bash, Read, Write
---

Generate Active Learning report from experiment results.

1. Write `scripts/4_al_report.py`:
   - Read `data/al_history_entropy.json` and `data/al_history_random.json`
   - Plot: accuracy vs n_labeled (entropy + random on ONE graph) → `reports/learning_curve_accuracy.png`
   - Plot: F1 vs n_labeled (entropy + random on ONE graph) → `reports/learning_curve_f1.png`
   - Calculate: at what n_labeled does entropy reach random's final F1?
   - Calculate: how many examples saved by entropy vs random at same quality
   - Save combined plot to `reports/learning_curve.png`
2. Install deps and run: `uv add matplotlib && uv run python scripts/4_al_report.py`
3. Generate `reports/al_report.md` with analysis and conclusions
4. Generate `notebooks/al_experiment.ipynb` with full experiment and visualizations
5. Print: examples saved summary
