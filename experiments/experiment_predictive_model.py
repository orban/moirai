#!/usr/bin/env python3
"""Predictive model experiment on swe_smith (1000 IID single-run tasks).

Tests whether moirai's behavioral features add predictive value beyond
task metadata for predicting task success/failure.

Models:
  A — mutation_type only (one-hot)
  B — behavioral features only
  C — mutation_type + behavioral features
  D — mutation_type + behavioral features + gapped motif membership
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

# Make moirai importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moirai.compress import step_enriched_name
from moirai.normalize import normalize_run
from moirai.schema import Run

# sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# 1. Load runs
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Users/ryo/dev/moirai/examples/swe_smith")


def load_runs() -> list[Run]:
    runs: list[Run] = []
    for f in sorted(DATA_DIR.iterdir()):
        if not f.suffix == ".json":
            continue
        with open(f) as fh:
            raw = json.load(fh)
        run, _warnings = normalize_run(raw)
        if run.result.success is not None:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

def extract_mutation_type(task_id: str) -> str:
    """Extract the mutation type from a swe_smith task_id.

    task_id pattern: repo__name.commit.func_pm_<mutation>__hash
    Some tasks have pr_<number> instead of func_pm_*.
    """
    parts = task_id.split(".")
    if len(parts) < 3:
        return "unknown"
    mutation_part = parts[2]
    # Strip trailing hash after last __
    if "__" in mutation_part:
        mutation = mutation_part.rsplit("__", 1)[0]
    else:
        mutation = mutation_part
    return mutation


def extract_repo(task_id: str) -> str:
    """First dotted segment = repo."""
    return task_id.split(".")[0]


def enriched_names(run: Run) -> list[str]:
    """Get the filtered enriched step name sequence for a run."""
    return [n for s in run.steps if (n := step_enriched_name(s)) is not None]


def has_test_fail_loop(names: list[str], threshold: int = 3) -> bool:
    """True if there are `threshold`+ consecutive test(fail) without an edit/write."""
    consecutive_fails = 0
    for name in names:
        if name == "test(fail)":
            consecutive_fails += 1
            if consecutive_fails >= threshold:
                return True
        elif name in ("edit", "write", "edit(source)", "edit(test_file)",
                      "edit(config)", "edit(other)", "write(source)",
                      "write(test_file)", "write(config)", "write(other)"):
            consecutive_fails = 0
        # other steps don't reset the counter
    return False


def extract_behavioral_features(run: Run) -> dict[str, float]:
    """Extract behavioral features from a single run."""
    names = enriched_names(run)
    total = len(names) if names else 1  # avoid div-by-zero

    edit_write_steps = sum(1 for n in names if n.startswith(("edit", "write")))
    read_search_steps = sum(1 for n in names if n.startswith(("read", "search")))
    test_fail = sum(1 for n in names if n == "test(fail)")
    test_pass = sum(1 for n in names if n == "test(pass)")

    return {
        "step_count": len(names),
        "edit_density": edit_write_steps / total,
        "exploration_ratio": read_search_steps / total,
        "test_fail_count": test_fail,
        "test_pass_count": test_pass,
        "has_test_fail_loop": float(has_test_fail_loop(names)),
    }


# ---------------------------------------------------------------------------
# 3. Gapped motif membership features
# ---------------------------------------------------------------------------

def discover_motifs(runs: list[Run], top_k: int = 20, q_threshold: float = 0.10,
                    verbose: bool = True) -> list:
    """Discover gapped motifs from a set of runs (training data only)."""
    from moirai.analyze.motifs import find_gapped_motifs

    motifs, n_tested = find_gapped_motifs(runs, max_length=3, min_count=5, q_threshold=q_threshold)
    motifs = motifs[:top_k]

    if verbose:
        print(f"  gapped motifs: {len(motifs)} significant (of {n_tested} tested, q<{q_threshold})")
        for m in motifs[:5]:
            print(f"    {m.display}: lift={m.lift:.2f}, q={m.q_value:.4f}, n={m.total_runs}")

    return motifs


def encode_motif_membership(runs: list[Run], motifs: list) -> np.ndarray:
    """Encode runs as binary motif membership given a pre-discovered motif list."""
    if not motifs:
        return np.zeros((len(runs), 0))

    matrix = np.zeros((len(runs), len(motifs)), dtype=np.float64)
    for i, run in enumerate(runs):
        names = enriched_names(run)
        for j, motif in enumerate(motifs):
            if _contains_subsequence(names, motif.anchors):
                matrix[i, j] = 1.0

    return matrix


def _contains_subsequence(sequence: list[str], anchors: tuple[str, ...]) -> bool:
    """Check if anchors appear in order (with gaps) in sequence."""
    it = iter(sequence)
    return all(a in it for a in anchors)


# ---------------------------------------------------------------------------
# 4. Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name: str, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> tuple[float, float]:
    """5-fold stratified CV, returns (mean_auc, std_auc). Uses GradientBoosting."""
    if X.shape[1] == 0:
        print(f"  {name}: no features, skipping")
        return 0.5, 0.0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})  [folds: {', '.join(f'{a:.3f}' for a in aucs)}]")
    return mean_auc, std_auc


def evaluate_model_with_motifs(
    name: str,
    X_base: np.ndarray,
    y: np.ndarray,
    runs: list[Run],
    top_k: int = 20,
    q_threshold: float = 0.10,
    n_folds: int = 5,
) -> tuple[float, float]:
    """CV with motif discovery inside each fold to prevent leakage.

    For each fold, discovers motifs on training runs only, then encodes
    both train and test runs against those training-derived motifs.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    motif_counts = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_base, y)):
        train_runs = [runs[i] for i in train_idx]
        motifs = discover_motifs(train_runs, top_k=top_k, q_threshold=q_threshold, verbose=False)
        motif_counts.append(len(motifs))

        X_motifs_train = encode_motif_membership(train_runs, motifs)
        X_motifs_test = encode_motif_membership([runs[i] for i in test_idx], motifs)

        X_train = np.hstack([X_base[train_idx], X_motifs_train])
        X_test = np.hstack([X_base[test_idx], X_motifs_test])
        y_train, y_test = y[train_idx], y[test_idx]

        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        aucs.append(auc)

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    avg_motifs = float(np.mean(motif_counts))
    print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})  [folds: {', '.join(f'{a:.3f}' for a in aucs)}]")
    print(f"    (avg {avg_motifs:.1f} motifs discovered per fold, q<{q_threshold})")
    return mean_auc, std_auc


def feature_importances(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    """Train on full data and report feature importances."""
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n  Feature importances (Model C, full data):")
    for rank, idx in enumerate(sorted_idx[:15]):
        print(f"    {rank+1:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Predictive model experiment: swe_smith (1000 IID tasks)")
    print("=" * 70)

    # Load
    print("\n[1] Loading runs...")
    runs = load_runs()
    y = np.array([1 if r.result.success else 0 for r in runs])
    print(f"  {len(runs)} runs loaded, {y.sum()} pass ({y.mean()*100:.1f}%), {len(runs)-y.sum()} fail")

    # Extract mutation types
    print("\n[2] Extracting features...")
    mutation_types = [extract_mutation_type(r.task_id) for r in runs]
    repos = [extract_repo(r.task_id) for r in runs]

    # One-hot encode mutation type
    mt_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_mutation = mt_encoder.fit_transform(np.array(mutation_types).reshape(-1, 1))
    mutation_feature_names = [f"mt:{c}" for c in mt_encoder.categories_[0]]
    print(f"  mutation types: {len(mt_encoder.categories_[0])} unique -> {X_mutation.shape[1]} features")

    # Behavioral features
    behavioral_rows = [extract_behavioral_features(r) for r in runs]
    behavioral_feature_names = list(behavioral_rows[0].keys())
    X_behavioral = np.array([[row[k] for k in behavioral_feature_names] for row in behavioral_rows])
    print(f"  behavioral features: {X_behavioral.shape[1]}")

    # Print behavioral feature stats
    for i, name in enumerate(behavioral_feature_names):
        col = X_behavioral[:, i]
        print(f"    {name:25s}  mean={col.mean():.3f}  std={col.std():.3f}  min={col.min():.1f}  max={col.max():.1f}")

    # Build combined feature matrices (NO motifs here — motifs go inside CV)
    X_A = X_mutation
    X_B = X_behavioral
    X_C = np.hstack([X_mutation, X_behavioral])

    names_C = mutation_feature_names + behavioral_feature_names

    # Evaluate Models A-C (fixed features, standard CV)
    print("\n[3] 5-fold stratified cross-validation (GradientBoosting)...")
    print("-" * 60)
    auc_a, std_a = evaluate_model("Model A (mutation_type only)", X_A, y)
    auc_b, std_b = evaluate_model("Model B (behavioral only)", X_B, y)
    auc_c, std_c = evaluate_model("Model C (mutation + behavioral)", X_C, y)

    # Model D: motifs discovered INSIDE each CV fold (no leakage)
    print()
    print("  Model D: motif discovery inside CV folds (leak-free)...")
    auc_d, std_d = evaluate_model_with_motifs(
        "Model D (mutation + behavioral + motifs, CV-internal)",
        X_C, y, runs, top_k=20, q_threshold=0.10,
    )

    # Also run with the old (leaked) approach for comparison
    print()
    print("  Model D-leaked: motifs discovered on full data (for comparison)...")
    all_motifs = discover_motifs(runs, top_k=20, q_threshold=0.10)
    X_motifs_leaked = encode_motif_membership(runs, all_motifs)
    X_D_leaked = np.hstack([X_C, X_motifs_leaked]) if X_motifs_leaked.shape[1] > 0 else X_C
    auc_d_leaked, std_d_leaked = evaluate_model(
        "Model D-leaked (motifs on full data, BIASED)", X_D_leaked, y,
    )
    print("-" * 60)

    # Deltas
    print("\n[4] AUC lift over baseline (Model A):")
    print(f"  B vs A (behavioral only vs metadata):       {auc_b - auc_a:+.4f}")
    print(f"  C vs A (combined vs metadata):              {auc_c - auc_a:+.4f}")
    print(f"  D vs A (combined+motifs vs metadata):       {auc_d - auc_a:+.4f}")
    print(f"  D vs C (motifs marginal, leak-free):        {auc_d - auc_c:+.4f}")
    print(f"  D-leaked vs C (motifs marginal, BIASED):    {auc_d_leaked - auc_c:+.4f}")
    print(f"  Leakage inflation (D-leaked - D):           {auc_d_leaked - auc_d:+.4f}")

    # Feature importances for Model C
    print("\n[5] Feature importances...")
    feature_importances(X_C, y, names_C)

    # Quick sanity: logistic regression comparison
    print("\n[6] Logistic regression comparison (sanity check)...")
    print("-" * 60)

    def eval_lr(name: str, X: np.ndarray) -> float:
        if X.shape[1] == 0:
            print(f"  {name}: no features, skipping")
            return 0.5
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            y_prob = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], y_prob))
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")
        return mean_auc

    lr_a = eval_lr("LR Model A (mutation only)", X_A)
    lr_b = eval_lr("LR Model B (behavioral only)", X_B)
    lr_c = eval_lr("LR Model C (combined)", X_C)
    print("-" * 60)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline success rate: {y.mean()*100:.1f}%")
    print(f"  N = {len(runs)} tasks, {len(mt_encoder.categories_[0])} mutation types, {len(set(repos))} repos")
    print()
    print(f"  {'Model':<50s} {'GBM AUC':>10s}  {'LR AUC':>10s}")
    print(f"  {'-'*50} {'-'*10}  {'-'*10}")
    print(f"  {'A: mutation_type only':<50s} {auc_a:>10.4f}  {lr_a:>10.4f}")
    print(f"  {'B: behavioral features only':<50s} {auc_b:>10.4f}  {lr_b:>10.4f}")
    print(f"  {'C: mutation_type + behavioral':<50s} {auc_c:>10.4f}  {lr_c:>10.4f}")
    print(f"  {'D: C + motifs (CV-internal, honest)':<50s} {auc_d:>10.4f}  {'—':>10s}")
    print(f"  {'D-leaked: C + motifs (full-data, BIASED)':<50s} {auc_d_leaked:>10.4f}  {'—':>10s}")
    print()

    if auc_c > auc_a + 0.01:
        print("  -> Behavioral features add meaningful signal beyond task metadata.")
    elif auc_c > auc_a:
        print("  -> Behavioral features add marginal signal beyond task metadata.")
    else:
        print("  -> Behavioral features do NOT add signal beyond task metadata.")

    if auc_d > auc_c + 0.01:
        print("  -> Gapped motifs add meaningful signal (leak-free).")
    elif auc_d > auc_c:
        print("  -> Gapped motifs add marginal signal (leak-free).")
    else:
        print("  -> Gapped motifs do NOT add signal once leakage is fixed.")

    if auc_d_leaked > auc_d + 0.005:
        print(f"  !! Leakage inflated motif AUC by {auc_d_leaked - auc_d:+.4f} — previous claims were overstated.")


if __name__ == "__main__":
    main()
