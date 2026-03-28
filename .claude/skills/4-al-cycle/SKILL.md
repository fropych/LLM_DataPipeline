---
name: 4-al-cycle
description: Run Active Learning cycle — fit + query + evaluate, 5 iterations, entropy and random strategies
allowed-tools: Bash, Read, Write
---

Run the Active Learning experiment on `data/labeled/dataset_reviewed.parquet`.

1. Write `scripts/4_al_cycle.py`:
   - Feature extraction: TF-IDF on text column
   - Split: initial labeled set (N=50 random), pool, test set (20% held out)
   - **ENTROPY strategy** (5 iterations, batch=20):
     a. `fit`: train LogisticRegression on labeled set
     b. `query`: predict on pool, select 20 highest-entropy samples
     c. `evaluate`: accuracy + F1 on test set
     d. Move selected from pool to labeled set
     e. Record: `{iteration, n_labeled, accuracy, f1}`
   - **RANDOM strategy** (same setup, random selection) as baseline
   - Save histories to `data/al_history_entropy.json` and `data/al_history_random.json`
   - Save final entropy model to `models/al_model.joblib`
2. Install deps and run: `uv add scikit-learn joblib && uv run python scripts/4_al_cycle.py`
3. Print: final accuracy/F1 for both strategies
