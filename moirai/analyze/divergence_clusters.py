"""Cluster divergence points by semantic content.

Takes the raw divergence points from pairwise alignment and groups them
by *what kind of divergence* they represent — not the step type labels,
but the actual content of the agent's reasoning and actions.

This is the bridge between "here's where trajectories split" (structural)
and "here's what your agent keeps getting wrong" (actionable diagnosis).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from moirai.analyze.dpo import PreferencePair


@dataclass
class DivergenceCluster:
    """A group of similar divergence points across tasks."""
    cluster_id: int
    size: int
    label: str                          # auto-generated summary
    mean_position: float                # average divergence position (0-1)
    task_ids: list[str]                 # tasks in this cluster
    preferred_labels: dict[str, int]    # step type counts for pass side
    dispreferred_labels: dict[str, int] # step type counts for fail side
    representative_pairs: list[PreferencePair]  # exemplars
    action_summary: str                 # what the fail run typically does wrong


def format_divergence_text(pair: PreferencePair) -> str:
    """Format a divergence point as text for embedding.

    Captures what matters: what context led here, what the pass run did,
    what the fail run did, and how they differ.
    """
    pref_out = pair.preferred_step.get("output", {})
    dis_out = pair.dispreferred_step.get("output", {})

    pref_reasoning = pref_out.get("reasoning", "")[:300]
    pref_action = pref_out.get("action", "")[:200]
    pref_result = str(pref_out.get("result", ""))[:200]

    dis_reasoning = dis_out.get("reasoning", "")[:300]
    dis_action = dis_out.get("action", "")[:200]
    dis_result = str(dis_out.get("result", ""))[:200]

    context = " > ".join(pair.context_sequence[-5:]) if pair.context_sequence else "start"

    parts = [f"Context: {context}"]
    parts.append(f"Pass action: {pair.preferred_label}")
    if pref_reasoning:
        parts.append(f"Pass reasoning: {pref_reasoning}")
    if pref_action:
        parts.append(f"Pass command: {pref_action}")

    parts.append(f"Fail action: {pair.dispreferred_label}")
    if dis_reasoning:
        parts.append(f"Fail reasoning: {dis_reasoning}")
    if dis_action:
        parts.append(f"Fail command: {dis_action}")

    return "\n".join(parts)


def cluster_divergences(
    pairs: list[PreferencePair],
    n_clusters: int | None = None,
    min_cluster_size: int = 5,
    method: str = "tfidf",
    max_pairs: int = 5000,
) -> list[DivergenceCluster]:
    """Cluster divergence points by content similarity.

    Methods:
        - "tfidf": TF-IDF on divergence text + KMeans (fast, no GPU)
        - "embedding": Sentence-transformer embeddings + KMeans (better, needs model)

    Returns clusters sorted by size descending.
    """
    if not pairs:
        return []

    # Cap for performance
    if len(pairs) > max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_pairs)

    # Format each divergence as text
    texts = [format_divergence_text(p) for p in pairs]

    # Compute embeddings
    if method == "embedding":
        embeddings = _embed_sentences(texts)
    else:
        embeddings = _tfidf_vectors(texts)

    # Determine number of clusters
    if n_clusters is None:
        # Heuristic: sqrt(n) capped at 20
        n_clusters = min(20, max(3, int(len(pairs) ** 0.5)))

    # Cluster
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Build cluster objects
    from collections import Counter, defaultdict

    cluster_pairs: dict[int, list[tuple[int, PreferencePair]]] = defaultdict(list)
    for i, (label, pair) in enumerate(zip(labels, pairs)):
        cluster_pairs[label].append((i, pair))

    clusters = []
    for cid in sorted(cluster_pairs, key=lambda c: -len(cluster_pairs[c])):
        members = cluster_pairs[cid]
        if len(members) < min_cluster_size:
            continue

        member_pairs = [p for _, p in members]
        positions = [p.divergence_position_norm for p in member_pairs]
        pref_labels = Counter(p.preferred_label for p in member_pairs)
        dis_labels = Counter(p.dispreferred_label for p in member_pairs)
        task_ids = list(set(p.task_id for p in member_pairs))

        # Pick representative: closest to centroid
        member_indices = [i for i, _ in members]
        member_vecs = embeddings[member_indices]
        centroid = member_vecs.mean(axis=0)
        dists = np.linalg.norm(member_vecs - centroid, axis=1)
        rep_indices = np.argsort(dists)[:3]
        representatives = [member_pairs[i] for i in rep_indices]

        # Auto-generate label and summary
        label, summary = _auto_label(member_pairs, pref_labels, dis_labels)

        clusters.append(DivergenceCluster(
            cluster_id=len(clusters),
            size=len(members),
            label=label,
            mean_position=float(np.mean(positions)),
            task_ids=task_ids,
            preferred_labels=dict(pref_labels),
            dispreferred_labels=dict(dis_labels),
            representative_pairs=representatives,
            action_summary=summary,
        ))

    return clusters


def _tfidf_vectors(texts: list[str]) -> np.ndarray:
    """TF-IDF vectorization — fast, no external model needed."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray()


def _embed_sentences(texts: list[str]) -> np.ndarray:
    """Sentence-transformer embeddings — better quality, needs model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


def _auto_label(
    pairs: list[PreferencePair],
    pref_labels: dict[str, int],
    dis_labels: dict[str, int],
) -> tuple[str, str]:
    """Generate a human-readable label and summary for a cluster."""
    top_pref = max(pref_labels, key=pref_labels.get) if pref_labels else "?"
    top_dis = max(dis_labels, key=dis_labels.get) if dis_labels else "?"
    n = len(pairs)

    # Check for specific patterns
    dis_actions = []
    pref_actions = []
    for p in pairs[:20]:
        da = p.dispreferred_step.get("output", {}).get("action", "").lower()
        pa = p.preferred_step.get("output", {}).get("action", "").lower()
        dis_actions.append(da)
        pref_actions.append(pa)

    # Detect noop orientation
    noop_count = sum(1 for a in dis_actions if any(
        p in a for p in ["&& pwd", "pwd)", "--version", "which "]
    ))
    if noop_count > len(dis_actions) * 0.4:
        return (
            "Wasted step: no-op orientation",
            f"Fail runs waste a step on pwd/version checks ({noop_count}/{len(dis_actions[:20])} cases). "
            f"Pass runs immediately {top_pref}."
        )

    # Detect different file targeting
    import re
    path_mismatch = 0
    for pa, da in zip(pref_actions, dis_actions):
        pp = re.findall(r"path=([^\s,)]+)", pa)
        dp = re.findall(r"path=([^\s,)]+)", da)
        if pp and dp and pp[0] != dp[0]:
            path_mismatch += 1
    if path_mismatch > len(dis_actions) * 0.4:
        return (
            "Wrong target: different file selected",
            f"Pass and fail runs target different files ({path_mismatch}/{len(dis_actions[:20])} cases). "
            f"Fail runs typically {top_dis} the wrong location."
        )

    # Detect reasoning vs acting
    if top_pref == "reason":
        return (
            "Skipped reasoning: acted without thinking",
            f"Pass runs reason about the problem first. "
            f"Fail runs jump directly to {top_dis}."
        )

    # Detect test/run before understanding
    if top_dis in ("bash(python)", "test") and top_pref in ("read", "read(source)", "read(other)"):
        return (
            "Premature execution: tested before reading",
            f"Fail runs execute code ({top_dis}) before understanding the problem. "
            f"Pass runs {top_pref} first."
        )

    # Generic label from dominant step types
    mean_pos = np.mean([p.divergence_position_norm for p in pairs])
    phase = "early" if mean_pos < 0.2 else "mid" if mean_pos < 0.6 else "late"
    return (
        f"{phase.title()} divergence: {top_pref} vs {top_dis}",
        f"At ~{mean_pos:.0%} through the trajectory, pass runs {top_pref} "
        f"while fail runs {top_dis} ({n} instances across {len(set(p.task_id for p in pairs))} tasks)."
    )
