#!/usr/bin/env python3
"""
Visualize contrastive activations using PCA and mean-difference analysis.

Reads per-layer activation files and metadata, applies PCA at each layer,
and produces plots showing how well straight-primed and funny-primed
prompts separate in activation space.

Also computes the mean-difference direction (funny - straight) and
projects activations onto it for a 1D separation view.

Output:
    results/figures/{position}_separation_curve.png
    results/figures/{position}_pca_by_layer.png
    results/figures/{position}_pca_peak_layer.png
    results/figures/{position}_mean_diff_projection.png

Usage:
    python3 visualize_activations.py --position pred_c
    python3 visualize_activations.py --position pred_b
    python3 visualize_activations.py --position pred_c --layers 20 40 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
RAW_DIR = BASE / "results" / "raw_activations"
FIGURES_DIR = BASE / "results" / "figures"


def load_activations(meta_file):
    """
    Load metadata and per-layer activation files.

    Returns:
        meta: dict with metadata
        layer_data: dict {layer_idx: np.array of shape (n_prompts, hidden_dim)}
        layer_indices: sorted list of available layer indices
    """
    with open(meta_file) as f:
        meta = json.load(f)

    naming = meta["naming"]
    raw_dir = meta_file.parent

    # Discover which layer files exist
    layer_data = {}
    for layer_idx in range(meta["n_layers_total"]):
        filename = naming.replace("{NN}", f"{layer_idx:02d}")
        path = raw_dir / filename
        if path.exists():
            layer_data[layer_idx] = np.load(path)

    layer_indices = sorted(layer_data.keys())
    return meta, layer_data, layer_indices


def compute_separation(X_straight, X_funny):
    """
    Compute a separation score between two groups of points.

    Uses the ratio of between-group distance to within-group spread
    (a simplified Fisher discriminant criterion).
    """
    mean_s = X_straight.mean(axis=0)
    mean_f = X_funny.mean(axis=0)

    between = np.linalg.norm(mean_f - mean_s)

    within_s = np.mean(np.linalg.norm(X_straight - mean_s, axis=1))
    within_f = np.mean(np.linalg.norm(X_funny - mean_f, axis=1))
    within = (within_s + within_f) / 2

    return between / within if within > 0 else 0


def compute_mean_diff_direction(X_straight, X_funny):
    """
    Compute the unit vector in the direction of (mean_funny - mean_straight).
    """
    diff = X_funny.mean(axis=0) - X_straight.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm > 0:
        return diff / norm
    return diff


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    parser = argparse.ArgumentParser(
        description="Visualize contrastive activations")
    parser.add_argument("--position", required=True, choices=["pred_c", "pred_b"],
                        help="Token position to visualize")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to include (default: all available)")
    parser.add_argument("--model-short", default="llama31_70b_instruct",
                        help="Model short name (default: llama31_70b_instruct)")
    args = parser.parse_args()

    # Load metadata and activations
    meta_file = RAW_DIR / f"{args.model_short}_{args.position}_meta.json"
    if not meta_file.exists():
        print(f"Metadata not found: {meta_file}")
        sys.exit(1)

    print(f"Loading metadata: {meta_file.name}")
    meta, layer_data, layer_indices = load_activations(meta_file)

    if not layer_indices:
        print("No layer files found.")
        sys.exit(1)

    # Filter to requested layers
    if args.layers:
        layer_indices = [l for l in layer_indices if l in args.layers]

    n_layers = len(layer_indices)
    n_prompts = meta["n_prompts"]
    samples = meta["samples"]

    print(f"Loaded {n_layers} layers, {n_prompts} prompts")

    # Split by condition
    is_funny = np.array([s["type"] == "funny" for s in samples])
    is_straight = ~is_funny
    pair_ids = np.array([s["pair_id"] for s in samples])

    print(f"Straight: {is_straight.sum()}, Funny: {is_funny.sum()}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pos = args.position

    # ── 1. Separation score across layers ────────────────────────────────────
    print("\n--- Computing separation scores ---")
    separations = []
    for layer_idx in layer_indices:
        X = layer_data[layer_idx]
        sep = compute_separation(X[is_straight], X[is_funny])
        separations.append(sep)

    peak_idx = np.argmax(separations)
    peak_layer = layer_indices[peak_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layer_indices, separations, color="#E85D75", linewidth=2)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Separation score\n(between / within)", fontsize=11)
    ax.set_title(f"Straight vs. Funny Context Separation by Layer\n"
                 f"(position={pos}, {n_prompts} contrastive prompts)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(f"Peak: layer {peak_layer}\n(score={separations[peak_idx]:.3f})",
                xy=(peak_layer, separations[peak_idx]),
                xytext=(peak_layer + 3, separations[peak_idx]),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="#333"))

    fig.tight_layout()
    out_path = FIGURES_DIR / f"{pos}_separation_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    print(f"  Peak: layer {peak_layer} (score={separations[peak_idx]:.3f})")

    # ── 2. PCA scatter plots at selected layers ──────────────────────────────
    quartile_layers = [
        layer_indices[n_layers // 4],
        layer_indices[n_layers // 2],
        layer_indices[3 * n_layers // 4],
        peak_layer,
    ]
    plot_layers = list(dict.fromkeys(quartile_layers))

    print(f"\n--- PCA scatter plots for layers: {plot_layers} ---")

    n_plots = len(plot_layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, layer in zip(axes, plot_layers):
        X = layer_data[layer]

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        ax.scatter(X_pca[is_straight, 0], X_pca[is_straight, 1],
                   c="#4A90D9", alpha=0.7, s=40, label="Straight ctx",
                   edgecolors="white", linewidths=0.5)
        ax.scatter(X_pca[is_funny, 0], X_pca[is_funny, 1],
                   c="#E85D75", alpha=0.7, s=40, label="Funny ctx",
                   edgecolors="white", linewidths=0.5)

        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)", fontsize=9)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"PCA of Activations: Straight vs. Funny Context (position={pos})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = FIGURES_DIR / f"{pos}_pca_by_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── 3. Detailed PCA at peak layer ────────────────────────────────────────
    X_peak = layer_data[peak_layer]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_peak)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_pca[is_straight, 0], X_pca[is_straight, 1],
               c="#4A90D9", alpha=0.7, s=50, label="Straight ctx",
               edgecolors="white", linewidths=0.5, zorder=2)
    ax.scatter(X_pca[is_funny, 0], X_pca[is_funny, 1],
               c="#E85D75", alpha=0.7, s=50, label="Funny ctx",
               edgecolors="white", linewidths=0.5, zorder=2)

    # Draw lines connecting matched pairs
    for pid in range(max(pair_ids) + 1):
        mask = pair_ids == pid
        if mask.sum() == 2:
            pts = X_pca[mask]
            ax.plot(pts[:, 0], pts[:, 1], color="#999", alpha=0.3,
                    linewidth=0.8, zorder=1)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance explained)", fontsize=11)
    ax.set_title(f"PCA at Layer {peak_layer} (Peak Separation, position={pos})\n"
                 f"Lines connect matched straight/funny pairs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"{pos}_pca_peak_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── 4. Mean-difference direction projection ──────────────────────────────
    print(f"\n--- Mean-difference direction analysis ---")

    # Compute mean-diff direction at each layer and project
    diff_separations = []
    for layer_idx in layer_indices:
        X = layer_data[layer_idx]
        direction = compute_mean_diff_direction(X[is_straight], X[is_funny])
        projections = X @ direction
        proj_s = projections[is_straight]
        proj_f = projections[is_funny]

        # 1D separation: difference in means / pooled std
        gap = proj_f.mean() - proj_s.mean()
        pooled_std = np.sqrt((proj_s.var() + proj_f.var()) / 2)
        d = gap / pooled_std if pooled_std > 0 else 0
        diff_separations.append(d)

    diff_peak_idx = np.argmax(diff_separations)
    diff_peak_layer = layer_indices[diff_peak_idx]

    # Plot d' across layers
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layer_indices, diff_separations, color="#2EAD6B", linewidth=2)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Cohen's d\n(mean-diff direction)", fontsize=11)
    ax.set_title(f"Separation Along Mean-Difference Direction by Layer\n"
                 f"(position={pos}, {n_prompts} contrastive prompts)",
                 fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(f"Peak: layer {diff_peak_layer}\n(d={diff_separations[diff_peak_idx]:.2f})",
                xy=(diff_peak_layer, diff_separations[diff_peak_idx]),
                xytext=(diff_peak_layer + 3, diff_separations[diff_peak_idx]),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="#333"))

    fig.tight_layout()
    out_path = FIGURES_DIR / f"{pos}_mean_diff_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Histogram of projections at peak mean-diff layer
    X_diff_peak = layer_data[diff_peak_layer]
    direction = compute_mean_diff_direction(X_diff_peak[is_straight], X_diff_peak[is_funny])
    projections = X_diff_peak @ direction

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(projections[is_straight], bins=15, alpha=0.6, color="#4A90D9",
            label="Straight ctx", edgecolor="white")
    ax.hist(projections[is_funny], bins=15, alpha=0.6, color="#E85D75",
            label="Funny ctx", edgecolor="white")
    ax.set_xlabel("Projection onto mean-difference direction", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Projections at Layer {diff_peak_layer} (position={pos})\n"
                 f"Cohen's d = {diff_separations[diff_peak_idx]:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"{pos}_mean_diff_projection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {meta['model']}")
    print(f"  Position: {pos}")
    print(f"  Prompts: {n_prompts} ({is_straight.sum()} straight, {is_funny.sum()} funny)")
    print(f"  Layers analyzed: {n_layers}")
    print(f"  Fisher separation peak: layer {peak_layer} "
          f"(score={separations[peak_idx]:.3f})")
    print(f"  Mean-diff peak: layer {diff_peak_layer} "
          f"(d={diff_separations[diff_peak_idx]:.2f})")
    print(f"  PCA variance at Fisher peak: "
          f"PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")


if __name__ == "__main__":
    main()
