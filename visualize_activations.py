#!/usr/bin/env python3
"""
Visualize contrastive activations using the analyze_activations API.

Reads per-layer activation files and metadata, computes separation metrics
and projections, and produces a suite of diagnostic plots.

Output figures (in results/figures/):
    {pos}_separation_curves.png     — Fisher + Cohen's d across layers
    {pos}_pca_by_layer.png          — PCA scatter at quartile + peak layers
    {pos}_pca_peak_layer.png        — PCA scatter at peak layer with pair lines
    {pos}_contrastive_scatter.png   — Contrastive projection at peak layer
    {pos}_pair_diff_histogram.png   — Per-pair distance histogram at peak layer
    {pos}_mean_diff_projection.png  — 1D projection histogram at peak Cohen's d layer

Usage:
    python3 visualize_activations.py --position pred_c
    python3 visualize_activations.py --position pred_b
    python3 visualize_activations.py --position pred_c --layers 20 40 60
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from analyze_activations import (
    load_activations,
    get_pair_indices,
    pair_differences,
    contrastive_direction,
    pca_projection,
    contrastive_projection,
    fisher_separation,
    cohens_d,
    pair_distances,
    analyze_all_layers,
    load_detailed_predictions,
    pun_boost_per_pair,
)

BASE = Path(__file__).parent
RAW_DIR = BASE / "results" / "raw_activations"
FIGURES_DIR = BASE / "results" / "figures"

# ── Plot styling ─────────────────────────────────────────────────────────────

COLOR_STRAIGHT = "#4A90D9"
COLOR_FUNNY = "#E85D75"
COLOR_FISHER = "#E85D75"
COLOR_COHENS = "#2EAD6B"
COLOR_PAIR_LINE = "#999"


def _clean_axes(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _pun_boost_markers(meta, pred_file, threshold=2.0):
    """
    Build a boolean array marking samples whose pair has a pun-word
    probability boost >= threshold (funny context vs straight context).

    Parameters:
        meta: metadata dict with "samples" key
        pred_file: path to *_detailed_preds.json
        threshold: minimum P(pun|funny)/P(pun|straight) ratio for a star

    Returns:
        has_boost: bool array (n_prompts,) — True if pair has boost >= threshold,
                   or None if pred_file doesn't exist
    """
    if not Path(pred_file).exists():
        return None

    detailed = load_detailed_predictions(pred_file)
    ratios = pun_boost_per_pair(detailed)

    samples = meta["samples"]
    has_boost = np.array(
        [ratios.get(s["pair_id"], 1.0) >= threshold for s in samples],
        dtype=bool,
    )
    return has_boost


def _pair_labels(meta, tests_file=None):
    """
    Build short labels for each pair: "subject...punchline".

    Uses joke_c_sentence for the subject noun and contrast_completion
    from the tests file for the pun word.  Falls back to sentence-only
    labels if the tests file isn't available.

    Parameters:
        meta: metadata dict
        tests_file: path to contextual_cloze_tests_100.json (optional;
                    defaults to datasets/contextual_cloze_tests_100.json)

    Returns dict {pair_id: "subject...punchline"}
    """
    import json

    # Load pun words from tests file if available
    pun_words = {}
    if tests_file is None:
        tests_file = BASE / "datasets" / "contextual_cloze_tests_100.json"
    if Path(tests_file).exists():
        with open(tests_file) as f:
            tests = json.load(f)
        for t in tests:
            pid = t["pair_id"]
            if pid not in pun_words and t.get("contrast_completion"):
                pun_words[pid] = t["contrast_completion"][0]

    samples = meta["samples"]
    labels = {}
    for s in samples:
        pid = s["pair_id"]
        if pid in labels:
            continue
        sentence = s.get("joke_c_sentence", "")
        # Extract subject: first noun phrase after "The"
        words = sentence.replace("The ", "").replace("the ", "").split()
        subject = words[0] if words else "?"
        if subject.endswith("'s"):
            subject = subject[:-2]
        pun = pun_words.get(pid, "?")
        labels[pid] = f"{subject}...{pun}"
    return labels


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_separation_curves(layer_results, pos, n_prompts, out_dir):
    """Fisher separation and Cohen's d across layers (dual y-axis)."""
    import matplotlib.pyplot as plt

    indices = layer_results["layer_indices"]
    fisher = layer_results["fisher"]
    cd = layer_results["cohens_d"]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(indices, fisher, color=COLOR_FISHER, linewidth=2, label="Fisher separation")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Fisher separation\n(between / within)", fontsize=11, color=COLOR_FISHER)
    ax1.tick_params(axis="y", labelcolor=COLOR_FISHER)
    _clean_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(indices, cd, color=COLOR_COHENS, linewidth=2, linestyle="--", label="Cohen's d")
    ax2.set_ylabel("Cohen's d\n(contrastive direction)", fontsize=11, color=COLOR_COHENS)
    ax2.tick_params(axis="y", labelcolor=COLOR_COHENS)
    ax2.spines["top"].set_visible(False)

    # Peak annotations
    peak_f = layer_results["peak_fisher_layer"]
    peak_f_idx = indices.index(peak_f)
    ax1.annotate(f"Fisher peak: L{peak_f} ({fisher[peak_f_idx]:.2f})",
                 xy=(peak_f, fisher[peak_f_idx]),
                 xytext=(peak_f + 3, fisher[peak_f_idx]),
                 fontsize=8, ha="left", color=COLOR_FISHER,
                 arrowprops=dict(arrowstyle="->", color=COLOR_FISHER))

    peak_d = layer_results["peak_cohens_d_layer"]
    peak_d_idx = indices.index(peak_d)
    ax2.annotate(f"Cohen's d peak: L{peak_d} ({cd[peak_d_idx]:.2f})",
                 xy=(peak_d, cd[peak_d_idx]),
                 xytext=(peak_d + 3, cd[peak_d_idx]),
                 fontsize=8, ha="left", color=COLOR_COHENS,
                 arrowprops=dict(arrowstyle="->", color=COLOR_COHENS))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.suptitle(f"Straight vs. Funny Separation by Layer\n"
                 f"(position={pos}, {n_prompts} contrastive prompts)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"{pos}_separation_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pca_by_layer(layer_data, layer_list, meta, pos, out_dir):
    """PCA scatter plots at selected layers."""
    import matplotlib.pyplot as plt

    _, is_funny, is_straight = get_pair_indices(meta)

    n_plots = len(layer_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, layer in zip(axes, layer_list):
        X = layer_data[layer]
        X_pca, _, var_ratios = pca_projection(X, n_components=2)

        ax.scatter(X_pca[is_straight, 0], X_pca[is_straight, 1],
                   c=COLOR_STRAIGHT, alpha=0.7, s=40, label="Straight ctx",
                   edgecolors="white", linewidths=0.5)
        ax.scatter(X_pca[is_funny, 0], X_pca[is_funny, 1],
                   c=COLOR_FUNNY, alpha=0.7, s=40, label="Funny ctx",
                   edgecolors="white", linewidths=0.5)

        ax.set_xlabel(f"PC1 ({var_ratios[0]:.1%} var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_ratios[1]:.1%} var)", fontsize=9)
        ax.set_title(f"Layer {layer}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        _clean_axes(ax)

    fig.suptitle(f"PCA of Activations: Straight vs. Funny Context (position={pos})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / f"{pos}_pca_by_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _scatter_by_context_and_boost(ax, X_2d, is_funny, has_boost):
    """
    Scatter points with color = context and marker = pun-probability boost.

    Color:  blue = straight context, pink = funny context
    Marker: circle = no significant boost, star = 2x+ pun probability boost
    """
    groups = [
        (~is_funny & ~has_boost, COLOR_STRAIGHT, "o", "Straight ctx"),
        (~is_funny & has_boost,  COLOR_STRAIGHT, "*", "Straight ctx, 2x+ pun boost"),
        (is_funny & ~has_boost,  COLOR_FUNNY,    "o", "Funny ctx"),
        (is_funny & has_boost,   COLOR_FUNNY,    "*", "Funny ctx, 2x+ pun boost"),
    ]
    for mask, color, marker, label in groups:
        if mask.sum() == 0:
            continue
        size = 120 if marker == "*" else 50
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, marker=marker, alpha=0.7, s=size, label=label,
                   edgecolors="white", linewidths=0.5, zorder=2)


def plot_pca_peak(layer_data, peak_layer, meta, pos, out_dir, has_boost=None):
    """Detailed PCA at peak layer with lines connecting matched pairs."""
    import matplotlib.pyplot as plt

    pair_ids, is_funny, is_straight = get_pair_indices(meta)
    labels = _pair_labels(meta)
    X = layer_data[peak_layer]
    X_pca, _, var_ratios = pca_projection(X, n_components=2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Pair lines with labels
    for pid in sorted(set(pair_ids)):
        mask = pair_ids == pid
        if mask.sum() == 2:
            pts = X_pca[mask]
            ax.plot(pts[:, 0], pts[:, 1], color=COLOR_PAIR_LINE, alpha=0.3,
                    linewidth=0.8, zorder=1)
            mid = pts.mean(axis=0)
            ax.annotate(labels.get(pid, ""), xy=mid, fontsize=5, color="#666",
                        ha="center", va="center", zorder=3)

    if has_boost is not None:
        _scatter_by_context_and_boost(ax, X_pca, is_funny, has_boost)
    else:
        ax.scatter(X_pca[is_straight, 0], X_pca[is_straight, 1],
                   c=COLOR_STRAIGHT, alpha=0.7, s=50, label="Straight ctx",
                   edgecolors="white", linewidths=0.5, zorder=2)
        ax.scatter(X_pca[is_funny, 0], X_pca[is_funny, 1],
                   c=COLOR_FUNNY, alpha=0.7, s=50, label="Funny ctx",
                   edgecolors="white", linewidths=0.5, zorder=2)

    ax.set_xlabel(f"PC1 ({var_ratios[0]:.1%} variance explained)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_ratios[1]:.1%} variance explained)", fontsize=11)
    star_label = "★ = 2x+ pun probability boost" if has_boost is not None else ""
    ax.set_title(f"PCA at Layer {peak_layer} (Peak Separation, position={pos})\n"
                 f"Lines connect matched pairs — {star_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    _clean_axes(ax)

    fig.tight_layout()
    out_path = out_dir / f"{pos}_pca_peak_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_contrastive_scatter(layer_data, peak_layer, meta, pos, out_dir,
                             has_boost=None):
    """
    Contrastive projection scatter: contrastive direction on x-axis,
    residual PC1 on y-axis.  Lines connect matched pairs.
    """
    import matplotlib.pyplot as plt

    pair_ids, is_funny, is_straight = get_pair_indices(meta)
    labels = _pair_labels(meta)
    X = layer_data[peak_layer]
    X_proj, _, var_ratios = contrastive_projection(X, meta, n_components=2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Pair lines with labels
    for pid in sorted(set(pair_ids)):
        mask = pair_ids == pid
        if mask.sum() == 2:
            pts = X_proj[mask]
            ax.plot(pts[:, 0], pts[:, 1], color=COLOR_PAIR_LINE, alpha=0.3,
                    linewidth=0.8, zorder=1)
            mid = pts.mean(axis=0)
            ax.annotate(labels.get(pid, ""), xy=mid, fontsize=5, color="#666",
                        ha="center", va="center", zorder=3)

    if has_boost is not None:
        _scatter_by_context_and_boost(ax, X_proj, is_funny, has_boost)
    else:
        ax.scatter(X_proj[is_straight, 0], X_proj[is_straight, 1],
                   c=COLOR_STRAIGHT, alpha=0.7, s=50, label="Straight ctx",
                   edgecolors="white", linewidths=0.5, zorder=2)
        ax.scatter(X_proj[is_funny, 0], X_proj[is_funny, 1],
                   c=COLOR_FUNNY, alpha=0.7, s=50, label="Funny ctx",
                   edgecolors="white", linewidths=0.5, zorder=2)

    ax.set_xlabel(f"Contrastive direction ({var_ratios[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"Residual PC1 ({var_ratios[1]:.1%} var)", fontsize=11)
    star_label = "★ = 2x+ pun probability boost" if has_boost is not None else ""
    ax.set_title(f"Contrastive Projection at Layer {peak_layer} (position={pos})\n"
                 f"X-axis = mean(funny−straight) direction — {star_label}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    _clean_axes(ax)

    fig.tight_layout()
    out_path = out_dir / f"{pos}_contrastive_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pair_diff_histogram(layer_data, peak_layer, meta, pos, out_dir):
    """Histogram of per-pair Euclidean distances at peak layer."""
    import matplotlib.pyplot as plt

    X = layer_data[peak_layer]
    dists = pair_distances(X, meta)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dists, bins=20, color=COLOR_FUNNY, alpha=0.7, edgecolor="white")
    ax.axvline(dists.mean(), color="#333", linestyle="--", linewidth=1.5,
               label=f"Mean = {dists.mean():.1f}")
    ax.set_xlabel("Per-pair Euclidean distance (funny − straight)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Pair Difference Magnitudes at Layer {peak_layer} (position={pos})\n"
                 f"{len(dists)} contrastive pairs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _clean_axes(ax)

    fig.tight_layout()
    out_path = out_dir / f"{pos}_pair_diff_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mean_diff_projection(layer_data, peak_layer, peak_d_value, meta, pos, out_dir):
    """1D histogram of projections onto contrastive direction at peak Cohen's d layer."""
    import matplotlib.pyplot as plt

    _, is_funny, is_straight = get_pair_indices(meta)
    X = layer_data[peak_layer]
    direction = contrastive_direction(X, meta)
    projections = X @ direction

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(projections[is_straight], bins=15, alpha=0.6, color=COLOR_STRAIGHT,
            label="Straight ctx", edgecolor="white")
    ax.hist(projections[is_funny], bins=15, alpha=0.6, color=COLOR_FUNNY,
            label="Funny ctx", edgecolor="white")
    ax.set_xlabel("Projection onto contrastive direction", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Contrastive Direction Projections at Layer {peak_layer} (position={pos})\n"
                 f"Cohen's d = {peak_d_value:.2f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _clean_axes(ax)

    fig.tight_layout()
    out_path = out_dir / f"{pos}_mean_diff_projection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Visualize contrastive activations")
    parser.add_argument("--position", required=True, choices=["pred_c", "pred_b"],
                        help="Token position to visualize")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to include (default: all available)")
    parser.add_argument("--model-short", default="llama31_70b_instruct",
                        help="Model short name (default: llama31_70b_instruct)")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    meta_file = RAW_DIR / f"{args.model_short}_{args.position}_meta.json"
    if not meta_file.exists():
        print(f"Metadata not found: {meta_file}")
        sys.exit(1)

    print(f"Loading metadata: {meta_file.name}")
    meta, layer_data, layer_indices = load_activations(meta_file)

    if not layer_indices:
        print("No layer files found.")
        sys.exit(1)

    if args.layers:
        layer_indices = [l for l in layer_indices if l in args.layers]
        layer_data = {l: layer_data[l] for l in layer_indices}

    n_layers = len(layer_indices)
    n_prompts = meta["n_prompts"]
    _, is_funny, is_straight = get_pair_indices(meta)
    pos = args.position

    print(f"Loaded {n_layers} layers, {n_prompts} prompts "
          f"({is_straight.sum()} straight, {is_funny.sum()} funny)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load detailed predictions (for pun boost markers) ────────────────────
    pred_file = RAW_DIR / f"{args.model_short}_{pos}_detailed_preds.json"
    has_boost = _pun_boost_markers(meta, pred_file, threshold=2.0)
    if has_boost is not None:
        n_boost = has_boost.sum()
        print(f"  Pun boost markers (2x+ threshold): "
              f"{n_boost // 2} pairs ({n_boost} samples) starred")
    else:
        print("  No detailed predictions found — stars disabled")

    # ── Compute metrics across all layers ────────────────────────────────────
    print("\n--- Computing separation metrics across layers ---")
    layer_results = analyze_all_layers(layer_data, meta)

    peak_fisher = layer_results["peak_fisher_layer"]
    peak_d = layer_results["peak_cohens_d_layer"]
    peak_d_idx = layer_indices.index(peak_d)
    peak_d_value = layer_results["cohens_d"][peak_d_idx]

    print(f"  Fisher peak: layer {peak_fisher} "
          f"(score={layer_results['fisher'][layer_indices.index(peak_fisher)]:.3f})")
    print(f"  Cohen's d peak: layer {peak_d} (d={peak_d_value:.2f})")

    # ── Generate all plots ───────────────────────────────────────────────────

    # 1. Separation curves
    out = plot_separation_curves(layer_results, pos, n_prompts, FIGURES_DIR)
    print(f"  Saved {out}")

    # 2. PCA scatter at quartile layers + peak
    quartile_layers = [
        layer_indices[n_layers // 4],
        layer_indices[n_layers // 2],
        layer_indices[3 * n_layers // 4],
        peak_fisher,
    ]
    plot_layers = list(dict.fromkeys(quartile_layers))  # deduplicate, preserve order

    out = plot_pca_by_layer(layer_data, plot_layers, meta, pos, FIGURES_DIR)
    print(f"  Saved {out}")

    # 3. PCA at peak layer with pair lines
    out = plot_pca_peak(layer_data, peak_fisher, meta, pos, FIGURES_DIR,
                        has_boost=has_boost)
    print(f"  Saved {out}")

    # 4. Contrastive projection scatter at peak Cohen's d layer
    out = plot_contrastive_scatter(layer_data, peak_d, meta, pos, FIGURES_DIR,
                                   has_boost=has_boost)
    print(f"  Saved {out}")

    # 5. Per-pair distance histogram at peak layer
    out = plot_pair_diff_histogram(layer_data, peak_d, meta, pos, FIGURES_DIR)
    print(f"  Saved {out}")

    # 6. Mean-difference projection histogram
    out = plot_mean_diff_projection(layer_data, peak_d, peak_d_value, meta, pos, FIGURES_DIR)
    print(f"  Saved {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {meta['model']}")
    print(f"  Position: {pos}")
    print(f"  Prompts: {n_prompts} ({is_straight.sum()} straight, {is_funny.sum()} funny)")
    print(f"  Layers analyzed: {n_layers}")
    print(f"  Fisher separation peak: layer {peak_fisher} "
          f"(score={layer_results['fisher'][layer_indices.index(peak_fisher)]:.3f})")
    print(f"  Cohen's d peak: layer {peak_d} (d={peak_d_value:.2f})")

    # PCA variance at peak
    X_peak = layer_data[peak_fisher]
    _, _, var_ratios = pca_projection(X_peak, n_components=2)
    print(f"  PCA variance at Fisher peak: "
          f"PC1={var_ratios[0]:.1%}, PC2={var_ratios[1]:.1%}")

    # Contrastive projection variance at peak
    _, _, c_var = contrastive_projection(layer_data[peak_d], meta, n_components=2)
    print(f"  Contrastive projection variance at Cohen's d peak: "
          f"contrast={c_var[0]:.1%}, resid_PC1={c_var[1]:.1%}")


if __name__ == "__main__":
    main()
