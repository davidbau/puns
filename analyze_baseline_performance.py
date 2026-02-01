#!/usr/bin/env python3
"""
Analyze baseline pun completion rates across models and joke tiers.

Reads puns_205.json and summarizes how often each model produces a pun
completion (from 20 diverse samples per joke), broken down by tier.
This measures raw pun-awareness in isolation — before any contrastive
context manipulation.

Output:
  results/figures/baseline_funny_rates.png  — bar chart by model
  results/figures/baseline_by_tier.png      — grouped bars by model and tier

Usage:
    python3 analyze_baseline_performance.py
"""

import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
PUNS_FILE = BASE / "datasets" / "puns_205.json"
FIGURES_DIR = BASE / "results" / "figures"

MODEL_ORDER = ["3b", "8b", "70b", "3.3-70b", "405b"]
MODEL_LABELS = {
    "3b": "Llama 3.2\n3B",
    "8b": "Llama 3.1\n8B",
    "70b": "Llama 3.1\n70B",
    "3.3-70b": "Llama 3.3\n70B",
    "405b": "Llama 3.1\n405B",
}
MODEL_LABELS_INLINE = {
    "3b": "Llama-3.2-3B",
    "8b": "Llama-3.1-8B",
    "70b": "Llama-3.1-70B",
    "3.3-70b": "Llama-3.3-70B",
    "405b": "Llama-3.1-405B",
}

TIER_ORDER = ["straight_dominated", "leaning", "balanced", "funny_dominated"]
TIER_LABELS = {
    "straight_dominated": "Straight-\ndominated",
    "leaning": "Leaning",
    "balanced": "Balanced",
    "funny_dominated": "Funny-\ndominated",
}


def main():
    with open(PUNS_FILE) as f:
        jokes = json.load(f)

    # ── Compute per-model averages ────────────────────────────────────────
    print(f"Baseline Pun Completion Rates (205 jokes, 20 samples each)")
    print(f"{'='*70}")

    overall = {}
    for m in MODEL_ORDER:
        rates = [j.get("cloze_model_funny_rate", {}).get(m, 0) for j in jokes]
        avg = sum(rates) / len(rates)
        overall[m] = avg

    print(f"\n  {'Model':<16} {'Avg Funny Rate':>14}  {'(across 205 jokes)':>20}")
    print(f"  {'─'*16} {'─'*14}  {'─'*20}")
    for m in MODEL_ORDER:
        print(f"  {MODEL_LABELS_INLINE[m]:<16} {overall[m]:>13.1%}")

    # ── Per-tier breakdown ────────────────────────────────────────────────
    by_tier = defaultdict(lambda: defaultdict(list))
    tier_counts = defaultdict(int)
    for j in jokes:
        tier = j.get("cloze_tier", "unknown")
        tier_counts[tier] += 1
        for m in MODEL_ORDER:
            rate = j.get("cloze_model_funny_rate", {}).get(m, 0)
            by_tier[tier][m].append(rate)

    print(f"\n  {'Tier':<22} {'N':>4}", end="")
    for m in MODEL_ORDER:
        print(f"  {m:>8}", end="")
    print()
    print(f"  {'─'*22} {'─'*4}", end="")
    for _ in MODEL_ORDER:
        print(f"  {'─'*8}", end="")
    print()

    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        n = tier_counts[tier]
        print(f"  {tier:<22} {n:>4}", end="")
        for m in MODEL_ORDER:
            rates = by_tier[tier][m]
            avg = sum(rates) / len(rates) if rates else 0
            print(f"  {avg:>7.1%}", end="")
        print()

    # ── Generate plots ────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Overall funny rates by model
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    rates = [overall[m] * 100 for m in MODEL_ORDER]
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

    bars = ax.bar(x, rates, color="#E85D75", edgecolor="white", linewidth=0.5,
                  width=0.6)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{r:.0f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Average funny completion rate (%)", fontsize=11)
    ax.set_title("Baseline Pun Completion Rate by Model Size\n"
                 "(20 diverse samples per joke, 205 jokes)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(rates) + 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_funny_rates.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {FIGURES_DIR / 'baseline_funny_rates.png'}")

    # Plot 2: Funny rates by model and tier
    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(MODEL_ORDER)
    n_tiers = len(TIER_ORDER)
    width = 0.15
    x = np.arange(n_tiers)

    colors = ["#8BC1F7", "#4A90D9", "#E85D75", "#C9190B", "#7D1007"]
    for i, m in enumerate(MODEL_ORDER):
        tier_rates = []
        for tier in TIER_ORDER:
            rates_list = by_tier[tier][m]
            avg = sum(rates_list) / len(rates_list) if rates_list else 0
            tier_rates.append(avg * 100)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, tier_rates, width, label=MODEL_LABELS_INLINE[m],
                      color=colors[i], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Funny completion rate (%)", fontsize=11)
    ax.set_title("Pun Completion Rate by Joke Tier and Model Size",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER], fontsize=9)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "baseline_by_tier.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'baseline_by_tier.png'}")

    # ── Print markdown table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MARKDOWN TABLE (for EXPERIMENTS.md)")
    print(f"{'='*70}\n")
    print("| Model | Avg Funny Rate |")
    print("|-------|:-:|")
    for m in MODEL_ORDER:
        print(f"| {MODEL_LABELS_INLINE[m]} | {overall[m]:.0%} |")


if __name__ == "__main__":
    main()
