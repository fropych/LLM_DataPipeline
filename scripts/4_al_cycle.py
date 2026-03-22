"""
Active Learning Cycle: ENTROPY vs RANDOM strategies
Dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/labeled/dataset_reviewed.parquet"
OUT_DIR     = "data"
MODEL_DIR   = "models"
N_INITIAL   = 50
N_ITER      = 5
BATCH_SIZE  = 20
TEST_FRAC   = 0.20
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
df = df[["text", "label"]].dropna().reset_index(drop=True)

# Use a manageable subset so TF-IDF fits in memory quickly
# Keep at most 20 000 rows (balanced) for pool + test; full dataset is 127 600
frames = []
for lbl, grp in df.groupby("label"):
    frames.append(grp.sample(5000, random_state=RANDOM_SEED))
df_sub = pd.concat(frames).reset_index(drop=True)
print(f"Working subset: {len(df_sub)} rows  |  classes: {df_sub['label'].unique().tolist()}")

# ── Train / test split ───────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df_sub, test_size=TEST_FRAC, stratify=df_sub["label"], random_state=RANDOM_SEED
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"Train pool: {len(train_df)}  |  Test: {len(test_df)}")

# ── TF-IDF (fit on train pool texts) ────────────────────────────────────────
print("Fitting TF-IDF…")
vectorizer = TfidfVectorizer(max_features=20_000, sublinear_tf=True)
X_all_train = vectorizer.fit_transform(train_df["text"])
X_test      = vectorizer.transform(test_df["text"])
y_test      = test_df["label"].values

# ── Helpers ──────────────────────────────────────────────────────────────────

def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (base-2) row-wise for a 2-D probability matrix."""
    eps = 1e-10
    return -np.sum(probs * np.log2(probs + eps), axis=1)


def run_al(strategy: str, seed: int = RANDOM_SEED) -> list[dict]:
    """Run one AL loop; strategy in {'entropy', 'random'}."""
    rng = np.random.default_rng(seed)

    # Initial labeled indices (random from pool)
    all_idx   = np.arange(len(train_df))
    labeled   = list(rng.choice(all_idx, size=N_INITIAL, replace=False))
    pool      = [i for i in all_idx if i not in set(labeled)]

    history = []

    for iteration in range(N_ITER + 1):           # +1 to record iter-0 baseline
        # -- Train
        X_lab = X_all_train[labeled]
        y_lab = train_df["label"].iloc[labeled].values

        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(X_lab, y_lab)

        # -- Evaluate
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro")
        history.append(
            {
                "iteration": iteration,
                "n_labeled": len(labeled),
                "accuracy": round(float(acc), 4),
                "f1": round(float(f1), 4),
            }
        )
        print(
            f"  [{strategy.upper():7s}] iter={iteration:2d}  "
            f"n_labeled={len(labeled):4d}  acc={acc:.4f}  f1={f1:.4f}"
        )

        if iteration == N_ITER or not pool:
            break

        # -- Query
        if strategy == "entropy":
            X_pool  = X_all_train[pool]
            probs   = clf.predict_proba(X_pool)
            scores  = entropy(probs)
            top_idx = np.argsort(scores)[::-1][:BATCH_SIZE]
            selected = [pool[i] for i in top_idx]
        else:  # random
            selected = list(rng.choice(pool, size=min(BATCH_SIZE, len(pool)), replace=False))

        labeled.extend(selected)
        pool = [i for i in pool if i not in set(selected)]

    return history


# ── Run both strategies ──────────────────────────────────────────────────────
print("\n=== ENTROPY strategy ===")
entropy_history = run_al("entropy")

print("\n=== RANDOM strategy ===")
random_history  = run_al("random")

# ── Save histories ───────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/al_history_entropy.json", "w") as f:
    json.dump(entropy_history, f, indent=2)

with open(f"{OUT_DIR}/al_history_random.json", "w") as f:
    json.dump(random_history, f, indent=2)

print(f"\nHistories saved to {OUT_DIR}/al_history_entropy.json  and  al_history_random.json")

# ── Save final entropy model ─────────────────────────────────────────────────
# Re-train on full final labeled set for the entropy run (last state)
rng_final    = np.random.default_rng(RANDOM_SEED)
all_idx_f    = np.arange(len(train_df))
labeled_f    = list(rng_final.choice(all_idx_f, size=N_INITIAL, replace=False))
pool_f       = [i for i in all_idx_f if i not in set(labeled_f)]

for _ in range(N_ITER):
    if not pool_f:
        break
    X_lab = X_all_train[labeled_f]
    y_lab = train_df["label"].iloc[labeled_f].values
    clf_f = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf_f.fit(X_lab, y_lab)
    probs   = clf_f.predict_proba(X_all_train[pool_f])
    scores  = entropy(probs)
    top_idx = np.argsort(scores)[::-1][:BATCH_SIZE]
    selected = [pool_f[i] for i in top_idx]
    labeled_f.extend(selected)
    pool_f = [i for i in pool_f if i not in set(selected)]

clf_final = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
clf_final.fit(X_all_train[labeled_f], train_df["label"].iloc[labeled_f].values)

model_path = f"{MODEL_DIR}/al_model.joblib"
joblib.dump({"model": clf_final, "vectorizer": vectorizer}, model_path)
print(f"Final entropy model saved to {model_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
e_final = entropy_history[-1]
r_final = random_history[-1]

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"ENTROPY  — n_labeled={e_final['n_labeled']}  acc={e_final['accuracy']:.4f}  f1={e_final['f1']:.4f}")
print(f"RANDOM   — n_labeled={r_final['n_labeled']}  acc={r_final['accuracy']:.4f}  f1={r_final['f1']:.4f}")

# Examples saved: at equal n_labeled, how many more examples would random need
# to match entropy's final f1?  We compute the crossover.
entropy_f1_final = e_final["f1"]
random_f1_vals   = [h["f1"] for h in random_history]
random_n_vals    = [h["n_labeled"] for h in random_history]

crossover = None
for n, f in zip(random_n_vals, random_f1_vals):
    if f >= entropy_f1_final:
        crossover = n
        break

if crossover is not None:
    saved = crossover - e_final["n_labeled"]
    print(f"\nExamples saved by ENTROPY over RANDOM: {saved}")
    print(f"  (Random needs {crossover} labels to reach entropy's final F1={entropy_f1_final:.4f})")
else:
    print("\nRandom never reached entropy's final F1 within the experiment window.")
    saved = None

print("=" * 60)
