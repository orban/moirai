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
import os
import sys
import warnings
from collections import Counter
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

def build_motif_features(runs: list[Run], top_k: int = 20) -> np.ndarray:
    """Run gapped motif discovery and return a binary membership matrix.

    Returns (n_runs, top_k) array where entry [i, j] = 1 if run i
    contains motif j's anchor subsequence.
    """
    from moirai.analyze.motifs import find_gapped_motifs

    motifs, n_tested = find_gapped_motifs(runs, max_length=3, min_count=5, q_threshold=0.20)
    motifs = motifs[:top_k]

    print(f"  gapped motifs: {len(motifs)} significant (of {n_tested} tested, q<0.20)")
    for m in motifs[:5]:
        print(f"    {m.display}: lift={m.lift:.2f}, q={m.q_value:.4f}, n={m.total_runs}")

    if not motifs:
        return np.zeros((len(runs), 0))

    # For each run, check membership in each motif
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

def evaluate_model(name: str, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> float:
    """5-fold stratified CV, returns mean AUC. Uses GradientBoosting."""
    if X.shape[1] == 0:
        print(f"  {name}: no features, skipping")
        return 0.5

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

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})  [folds: {', '.join(f'{a:.3f}' for a in aucs)}]")
    return mean_auc


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

    # Gapped motif features
    print("\n[3] Computing gapped motif features...")
    X_motifs = build_motif_features(runs, top_k=20)
    n_motif_features = X_motifs.shape[1]
    motif_feature_names = [f"motif_{j}" for j in range(n_motif_features)]
    print(f"  motif membership features: {n_motif_features}")

    # Build combined feature matrices
    X_A = X_mutation
    X_B = X_behavioral
    X_C = np.hstack([X_mutation, X_behavioral])
    X_D = np.hstack([X_mutation, X_behavioral, X_motifs]) if n_motif_features > 0 else X_C

    names_C = mutation_feature_names + behavioral_feature_names
    names_D = names_C + motif_feature_names

    # Evaluate
    print("\n[4] 5-fold stratified cross-validation (GradientBoosting)...")
    print("-" * 60)
    auc_a = evaluate_model("Model A (mutation_type only)", X_A, y)
    auc_b = evaluate_model("Model B (behavioral only)", X_B, y)
    auc_c = evaluate_model("Model C (mutation + behavioral)", X_C, y)
    auc_d = evaluate_model("Model D (mutation + behavioral + motifs)", X_D, y)
    print("-" * 60)

    # Deltas
    print("\n[5] AUC lift over baseline (Model A):")
    print(f"  B vs A (behavioral only vs metadata):       {auc_b - auc_a:+.4f}")
    print(f"  C vs A (combined vs metadata):              {auc_c - auc_a:+.4f}")
    print(f"  D vs A (combined+motifs vs metadata):       {auc_d - auc_a:+.4f}")
    print(f"  D vs C (motifs marginal contribution):      {auc_d - auc_c:+.4f}")

    # Feature importances for Model C
    print("\n[6] Feature importances...")
    feature_importances(X_C, y, names_C)

    # Also show feature importances for Model D if motifs exist
    if n_motif_features > 0:
        feature_importances(X_D, y, names_D)

    # Quick sanity: logistic regression comparison
    print("\n[7] Logistic regression comparison (sanity check)...")
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
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"  {name}: AUC = {mean_auc:.4f} (+/- {std_auc:.4f})")
        return mean_auc

    lr_a = eval_lr("LR Model A (mutation only)", X_A)
    lr_b = eval_lr("LR Model B (behavioral only)", X_B)
    lr_c = eval_lr("LR Model C (combined)", X_C)
    lr_d = eval_lr("LR Model D (combined + motifs)", X_D)
    print("-" * 60)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline success rate: {y.mean()*100:.1f}%")
    print(f"  N = {len(runs)} tasks, {len(mt_encoder.categories_[0])} mutation types, {len(set(repos))} repos")
    print()
    print(f"  {'Model':<45s} {'GBM AUC':>10s}  {'LR AUC':>10s}")
    print(f"  {'-'*45} {'-'*10}  {'-'*10}")
    print(f"  {'A: mutation_type only':<45s} {auc_a:>10.4f}  {lr_a:>10.4f}")
    print(f"  {'B: behavioral features only':<45s} {auc_b:>10.4f}  {lr_b:>10.4f}")
    print(f"  {'C: mutation_type + behavioral':<45s} {auc_c:>10.4f}  {lr_c:>10.4f}")
    print(f"  {'D: C + gapped motif membership':<45s} {auc_d:>10.4f}  {lr_d:>10.4f}")
    print()

    if auc_c > auc_a + 0.01:
        print("  -> Behavioral features add meaningful signal beyond task metadata.")
    elif auc_c > auc_a:
        print("  -> Behavioral features add marginal signal beyond task metadata.")
    else:
        print("  -> Behavioral features do NOT add signal beyond task metadata.")

    if auc_d > auc_c + 0.01:
        print("  -> Gapped motifs add meaningful signal beyond behavioral features.")
    elif auc_d > auc_c:
        print("  -> Gapped motifs add marginal signal beyond behavioral features.")
    else:
        print("  -> Gapped motifs do NOT add signal beyond behavioral features.")


if __name__ == "__main__":
    main()
