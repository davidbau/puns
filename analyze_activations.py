#!/usr/bin/env python3
"""
Pure numpy analysis functions for contrastive activation data.

Provides PCA, contrastive projection, separation metrics, and per-pair
analysis — all without sklearn or plotting dependencies.

API overview:

    Data loading:
        load_activations(meta_file) → (meta, layer_data, layer_indices)
        get_pair_indices(meta) → (pair_ids, is_funny, is_straight)

    Contrastive pair analysis:
        pair_differences(X, meta) → (n_pairs, hidden_dim)
        contrastive_direction(X, meta) → (hidden_dim,)

    Projection:
        pca_projection(X, n_components=2) → (X_pca, components, var_ratios)
        contrastive_projection(X, meta, n_components=2) → (X_proj, components, var_ratios)

    Separation metrics:
        fisher_separation(X, meta) → float
        cohens_d(X, meta, direction=None) → float
        pair_distances(X, meta) → (n_pairs,)

    Multi-layer:
        analyze_all_layers(layer_data, meta) → dict


    Detailed predictions:
        load_detailed_predictions(pred_file) → dict
        pun_boost_per_pair(detailed_preds) → {pair_id: float}
"""

import json
from pathlib import Path

import numpy as np


# ── Data loading ─────────────────────────────────────────────────────────────

def load_activations(meta_file):
    """
    Load metadata and per-layer activation files.

    Handles both merged files (layer{NN}.npy) and partial batch files
    (layer{NN}_batch{BB}.npy) from an in-progress collection.  When
    batch files exist but no merged file, they are concatenated on load.
    Metadata samples are trimmed to match the actual number of rows.

    Parameters:
        meta_file: Path to the *_meta.json file.

    Returns:
        meta: dict with metadata (model info, samples, etc.)
        layer_data: dict {layer_idx: np.array of shape (n_prompts, hidden_dim)}
        layer_indices: sorted list of available layer indices
    """
    meta_file = Path(meta_file)
    with open(meta_file) as f:
        meta = json.load(f)

    naming = meta["naming"]
    raw_dir = meta_file.parent
    prefix = naming.replace("_layer{NN}.npy", "")

    layer_data = {}
    for layer_idx in range(meta["n_layers_total"]):
        filename = naming.replace("{NN}", f"{layer_idx:02d}")
        path = raw_dir / filename

        if path.exists():
            layer_data[layer_idx] = np.load(path)
        else:
            # Try loading batch files (tolerates gaps from partial re-collection)
            parts = []
            for b in range(100):  # generous upper bound
                bf = raw_dir / f"{prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
                if bf.exists():
                    parts.append(np.load(bf))
            if parts:
                layer_data[layer_idx] = np.concatenate(parts, axis=0)

    layer_indices = sorted(layer_data.keys())

    # Trim metadata samples to match actual row count (for partial data)
    if layer_indices:
        n_rows = layer_data[layer_indices[0]].shape[0]
        if n_rows < len(meta["samples"]):
            meta = dict(meta)  # don't mutate the original
            meta["samples"] = meta["samples"][:n_rows]
            meta["n_prompts"] = n_rows

    return meta, layer_data, layer_indices


def get_pair_indices(meta):
    """
    Extract pair structure from metadata.

    Returns:
        pair_ids: int array (n_prompts,) — pair ID for each sample
        is_funny: bool array (n_prompts,) — True for funny-context samples
        is_straight: bool array (n_prompts,) — True for straight-context samples
    """
    samples = meta["samples"]
    pair_ids = np.array([s["pair_id"] for s in samples])
    is_funny = np.array([s["type"] == "funny" for s in samples])
    is_straight = ~is_funny
    return pair_ids, is_funny, is_straight


# ── Contrastive pair analysis ────────────────────────────────────────────────

def pair_differences(X, meta):
    """
    Compute per-pair difference vectors (funny − straight).

    For each pair_id, finds the funny and straight sample and returns
    their difference.  Pairs are returned sorted by pair_id.

    Parameters:
        X: (n_prompts, hidden_dim) activation matrix
        meta: metadata dict with "samples" key

    Returns:
        diffs: (n_pairs, hidden_dim) array of per-pair difference vectors
    """
    samples = meta["samples"]
    # Group sample indices by pair_id
    pairs = {}
    for i, s in enumerate(samples):
        pid = s["pair_id"]
        pairs.setdefault(pid, {})[s["type"]] = i

    n_pairs = len(pairs)
    hidden_dim = X.shape[1]
    diffs = np.zeros((n_pairs, hidden_dim), dtype=X.dtype)

    for k, pid in enumerate(sorted(pairs.keys())):
        p = pairs[pid]
        diffs[k] = X[p["funny"]] - X[p["straight"]]

    return diffs


def contrastive_direction(X, meta):
    """
    Mean pair-difference direction, unit-normalized.

    This is the average of (funny − straight) across all pairs,
    normalized to unit length.  It represents the dominant direction
    along which funny and straight contexts differ.

    Parameters:
        X: (n_prompts, hidden_dim) activation matrix
        meta: metadata dict

    Returns:
        direction: (hidden_dim,) unit vector
    """
    diffs = pair_differences(X, meta)
    mean_diff = diffs.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_diff)
    if norm > 0:
        return mean_diff / norm
    return mean_diff


# ── Projection ───────────────────────────────────────────────────────────────

def pca_projection(X, n_components=2):
    """
    PCA via truncated SVD (no sklearn).

    Centers the data, computes the top principal components via SVD,
    and projects the data onto them.

    Parameters:
        X: (n_samples, n_features) data matrix
        n_components: number of components to keep

    Returns:
        X_pca: (n_samples, n_components) projected data
        components: (n_components, n_features) principal component directions
        var_ratios: (n_components,) fraction of variance explained per component
    """
    X = np.asarray(X, dtype=np.float32)
    X_centered = X - X.mean(axis=0)
    # Economy SVD: U is (n, k), S is (k,), Vt is (k, d) where k = min(n, d)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components]  # (n_components, n_features)
    X_pca = X_centered @ components.T  # (n_samples, n_components)

    # Variance explained: proportional to squared singular values
    total_var = np.sum(S ** 2)
    var_ratios = (S[:n_components] ** 2) / total_var

    return X_pca, components, var_ratios


def contrastive_projection(X, meta, n_components=2):
    """
    Contrastive projection: contrastive direction as axis 0, residual PCA for axis 1+.

    Axis 0 is the projection onto the mean pair-difference direction
    (the "contrastive axis").  Axes 1+ are PCA of the residual space
    orthogonal to that direction.

    Parameters:
        X: (n_samples, hidden_dim) activation matrix
        meta: metadata dict
        n_components: total number of output dimensions (≥ 1)

    Returns:
        X_proj: (n_samples, n_components) projected data
            Column 0 = projection onto contrastive direction
            Columns 1+ = residual PCA components
        components: (n_components, hidden_dim) projection directions
            Row 0 = contrastive direction (unit vector)
            Rows 1+ = residual PCA directions
        var_ratios: (n_components,) fraction of total variance per axis
    """
    d = contrastive_direction(X, meta)  # (hidden_dim,)

    X = np.asarray(X, dtype=np.float32)
    X_centered = X - X.mean(axis=0)

    # Axis 0: projection onto contrastive direction
    proj_contrast = X_centered @ d  # (n_samples,)

    total_var = np.sum(X_centered ** 2)

    result_components = [d]
    result_proj = [proj_contrast]
    result_var = [np.sum(proj_contrast ** 2) / total_var]

    # Residual PCA for remaining axes
    if n_components > 1:
        # Remove contrastive component from the data
        X_resid = X_centered - np.outer(proj_contrast, d)

        # SVD of residual
        U_r, S_r, Vt_r = np.linalg.svd(X_resid, full_matrices=False)
        n_resid = n_components - 1

        resid_components = Vt_r[:n_resid]  # (n_resid, hidden_dim)
        resid_proj = X_resid @ resid_components.T  # (n_samples, n_resid)
        resid_var = (S_r[:n_resid] ** 2) / total_var

        for i in range(n_resid):
            result_components.append(resid_components[i])
            result_proj.append(resid_proj[:, i])
            result_var.append(resid_var[i])

    X_proj = np.column_stack(result_proj)
    components = np.stack(result_components)
    var_ratios = np.array(result_var)

    return X_proj, components, var_ratios


# ── Separation metrics ───────────────────────────────────────────────────────

def fisher_separation(X, meta):
    """
    Fisher-like separation score: between-group distance / within-group spread.

    Computes the ratio of the Euclidean distance between group means to
    the average within-group distance to centroids.

    Parameters:
        X: (n_prompts, hidden_dim) activation matrix
        meta: metadata dict

    Returns:
        score: float — higher means better separation
    """
    _, is_funny, is_straight = get_pair_indices(meta)

    X = np.asarray(X, dtype=np.float32)
    mean_s = X[is_straight].mean(axis=0)
    mean_f = X[is_funny].mean(axis=0)

    between = np.linalg.norm(mean_f - mean_s)

    within_s = np.mean(np.linalg.norm(X[is_straight] - mean_s, axis=1))
    within_f = np.mean(np.linalg.norm(X[is_funny] - mean_f, axis=1))
    within = (within_s + within_f) / 2

    return between / within if within > 0 else 0.0


def cohens_d(X, meta, direction=None):
    """
    Cohen's d effect size along a given direction.

    Projects activations onto `direction` (defaults to the contrastive
    direction) and computes the standardized mean difference.

    Parameters:
        X: (n_prompts, hidden_dim) activation matrix
        meta: metadata dict
        direction: (hidden_dim,) unit vector, or None for contrastive direction

    Returns:
        d: float — Cohen's d (positive means funny > straight along direction)
    """
    if direction is None:
        direction = contrastive_direction(X, meta)

    _, is_funny, is_straight = get_pair_indices(meta)

    X = np.asarray(X, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    projections = X @ direction
    proj_s = projections[is_straight]
    proj_f = projections[is_funny]

    gap = proj_f.mean() - proj_s.mean()
    pooled_std = np.sqrt((proj_s.var() + proj_f.var()) / 2)

    return gap / pooled_std if pooled_std > 0 else 0.0


def pair_distances(X, meta):
    """
    Per-pair Euclidean distances between funny and straight activations.

    Parameters:
        X: (n_prompts, hidden_dim) activation matrix
        meta: metadata dict

    Returns:
        distances: (n_pairs,) Euclidean distance for each pair
    """
    diffs = pair_differences(X, meta).astype(np.float32)
    return np.linalg.norm(diffs, axis=1)


# ── Multi-layer analysis ─────────────────────────────────────────────────────

def analyze_all_layers(layer_data, meta):
    """
    Compute separation metrics across all available layers.

    Parameters:
        layer_data: dict {layer_idx: np.array (n_prompts, hidden_dim)}
        meta: metadata dict

    Returns:
        dict with keys:
            layer_indices: sorted list of layer indices
            fisher: (n_layers,) Fisher separation scores
            cohens_d: (n_layers,) Cohen's d values
            mean_pair_dist: (n_layers,) mean per-pair Euclidean distance
            pair_dist: (n_layers, n_pairs) per-pair distances
            peak_fisher_layer: int — layer with highest Fisher score
            peak_cohens_d_layer: int — layer with highest Cohen's d
    """
    indices = sorted(layer_data.keys())
    n_layers = len(indices)

    fisher_arr = np.zeros(n_layers)
    cohens_arr = np.zeros(n_layers)
    pair_dist_list = []

    for i, layer_idx in enumerate(indices):
        X = layer_data[layer_idx]
        fisher_arr[i] = fisher_separation(X, meta)
        cohens_arr[i] = cohens_d(X, meta)
        pair_dist_list.append(pair_distances(X, meta))

    pair_dist_arr = np.stack(pair_dist_list)  # (n_layers, n_pairs)
    mean_pair_dist = pair_dist_arr.mean(axis=1)

    return {
        "layer_indices": indices,
        "fisher": fisher_arr,
        "cohens_d": cohens_arr,
        "mean_pair_dist": mean_pair_dist,
        "pair_dist": pair_dist_arr,
        "peak_fisher_layer": indices[int(np.argmax(fisher_arr))],
        "peak_cohens_d_layer": indices[int(np.argmax(cohens_arr))],
    }


# ── Detailed predictions ────────────────────────────────────────────────────

def load_detailed_predictions(pred_file):
    """
    Load detailed predictions JSON (top-k tokens + target word log-probs).

    Parameters:
        pred_file: Path to *_detailed_preds.json

    Returns:
        dict with keys: model, position, top_k, n_prompts, results
    """
    with open(pred_file) as f:
        return json.load(f)


def pun_boost_per_pair(detailed_preds):
    """
    Compute pun-word probability boost ratio per pair.

    For each pair, computes P(pun words | funny context) / P(pun words | straight context).

    Parameters:
        detailed_preds: dict from load_detailed_predictions()

    Returns:
        dict {pair_id: float} — ratio of pun word probabilities (funny / straight).
            Values > 1 mean the model assigns higher pun probability in funny context.
            inf if straight probability is ~0 but funny is positive.
            1.0 if both are ~0.
    """
    by_pair = {}
    for r in detailed_preds["results"]:
        pid = r["pair_id"]
        pun_prob = sum(v["prob"] for v in r["pun_word_probs"].values())
        by_pair.setdefault(pid, {})[r["type"]] = pun_prob

    ratios = {}
    for pid, probs in by_pair.items():
        pf = probs.get("funny", 0)
        ps = probs.get("straight", 0)
        if ps > 1e-8:
            ratios[pid] = pf / ps
        elif pf > 1e-8:
            ratios[pid] = float("inf")
        else:
            ratios[pid] = 1.0
    return ratios
