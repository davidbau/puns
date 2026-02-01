#!/usr/bin/env python3
"""
Analyze contrastive cloze benchmark results.

Reads raw benchmark results and the puns_205 dataset, classifies each model
response as straight/funny/other using the joke's word lists, then computes
per-model context-effect metrics and generates summary tables and plots.

Output:
  results/cloze_analysis.json          — structured analysis data
  results/figures/context_effect.png   — grouped bar chart of funny rates
  results/figures/funny_shift.png      — bar chart of funny-rate deltas

Usage:
    python3 analyze_cloze_results.py
"""

import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent
RESULTS_RAW = BASE / "results" / "cloze_benchmark_raw.json"
PUNS_FILE = BASE / "datasets" / "puns_205.json"
TESTS_FILE = BASE / "datasets" / "contextual_cloze_tests_100.json"
ANALYSIS_FILE = BASE / "results" / "cloze_analysis.json"
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


def classify_word(word, straight_list, funny_list):
    """Classify a word as straight/funny/other using the joke's word lists."""
    if not word:
        return "other"
    w = word.lower()
    for s in straight_list:
        if w == s.lower() or w == s.lower().split()[0]:
            return "straight"
    for f in funny_list:
        if w == f.lower() or w == f.lower().split()[0]:
            return "funny"
    return "other"


def load_and_classify():
    """Load raw results, tests, and jokes; classify every response."""
    with open(RESULTS_RAW) as f:
        raw = json.load(f)
    with open(PUNS_FILE) as f:
        jokes = json.load(f)
    with open(TESTS_FILE) as f:
        tests = json.load(f)

    joke_by_index = {j["index"]: j for j in jokes}
    test_lookup = {(t["pair_id"], t["type"]): t for t in tests}

    # Classify each response
    classified = {}  # model -> list of {pair_id, context, word, classification, ...}

    for model_nick in MODEL_ORDER:
        if model_nick not in raw:
            continue
        entries = []
        responses = raw[model_nick]["responses"]
        for key, resp in responses.items():
            test = test_lookup.get((resp["pair_id"], resp["type"]))
            if not test:
                continue
            joke = joke_by_index.get(test["joke_c_index"])
            if not joke:
                continue

            cls = classify_word(resp["extracted_word"],
                                joke["straight"], joke["punny"])
            entries.append({
                "pair_id": resp["pair_id"],
                "context": resp["type"],  # "straight" or "funny" priming
                "word": resp["extracted_word"],
                "classification": cls,
                "joke_index": test["joke_c_index"],
            })
        classified[model_nick] = entries

    return classified, tests


def compute_summaries(classified):
    """Compute per-model, per-context summary statistics."""
    summaries = {}
    for model_nick in MODEL_ORDER:
        if model_nick not in classified:
            continue
        entries = classified[model_nick]
        model_summary = {}
        for ctx in ["straight", "funny"]:
            subset = [e for e in entries if e["context"] == ctx]
            n = len(subset)
            n_straight = sum(1 for e in subset if e["classification"] == "straight")
            n_funny = sum(1 for e in subset if e["classification"] == "funny")
            n_other = n - n_straight - n_funny
            model_summary[ctx] = {
                "n": n,
                "straight": n_straight,
                "funny": n_funny,
                "other": n_other,
                "pct_straight": n_straight / n if n else 0,
                "pct_funny": n_funny / n if n else 0,
                "pct_other": n_other / n if n else 0,
            }
        summaries[model_nick] = model_summary
    return summaries


def print_tables(summaries):
    """Print summary tables to stdout."""
    print(f"\n{'='*90}")
    print("  RESPONSE CLASSIFICATION BY MODEL AND CONTEXT")
    print(f"{'='*90}")
    print(f"  {'Model':<14} {'Context':<10} {'Straight':>9} {'Funny':>9} {'Other':>9} {'N':>5}")
    print(f"  {'─'*14} {'─'*10} {'─'*9} {'─'*9} {'─'*9} {'─'*5}")

    for nick in MODEL_ORDER:
        if nick not in summaries:
            continue
        for ctx in ["straight", "funny"]:
            s = summaries[nick][ctx]
            label = MODEL_LABELS_INLINE[nick] if ctx == "straight" else ""
            print(f"  {label:<14} {ctx:<10} {s['pct_straight']:>8.1%} "
                  f"{s['pct_funny']:>9.1%} {s['pct_other']:>9.1%} {s['n']:>5}")
        print(f"  {'─'*14} {'─'*10} {'─'*9} {'─'*9} {'─'*9} {'─'*5}")

    print(f"\n{'='*90}")
    print("  CONTEXT EFFECT: FUNNY-RATE SHIFT")
    print(f"{'='*90}")
    print(f"  {'Model':<16} {'P(fun|str_ctx)':>14} {'P(fun|fun_ctx)':>14} "
          f"{'Delta':>8} {'P(str|str_ctx)':>14} {'P(str|fun_ctx)':>14} {'Delta':>8}")
    print(f"  {'─'*16} {'─'*14} {'─'*14} {'─'*8} {'─'*14} {'─'*14} {'─'*8}")

    for nick in MODEL_ORDER:
        if nick not in summaries:
            continue
        ms = summaries[nick]
        pf_str = ms["straight"]["pct_funny"]
        pf_fun = ms["funny"]["pct_funny"]
        ps_str = ms["straight"]["pct_straight"]
        ps_fun = ms["funny"]["pct_straight"]

        print(f"  {MODEL_LABELS_INLINE[nick]:<16} {pf_str:>13.1%} {pf_fun:>14.1%} "
              f"{pf_fun - pf_str:>+7.1%} {ps_str:>13.1%} {ps_fun:>14.1%} "
              f"{ps_fun - ps_str:>+7.1%}")


def make_plots(summaries):
    """Generate matplotlib figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models = [n for n in MODEL_ORDER if n in summaries]
    labels = [MODEL_LABELS[n] for n in models]

    # ── Figure 1: Grouped bar chart — funny vs straight rates by context ─────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    x = np.arange(len(models))
    width = 0.32

    for ax, ctx, title in [(axes[0], "straight", "Straight-Primed Context"),
                            (axes[1], "funny", "Funny-Primed Context")]:
        pct_straight = [summaries[n][ctx]["pct_straight"] * 100 for n in models]
        pct_funny = [summaries[n][ctx]["pct_funny"] * 100 for n in models]
        pct_other = [summaries[n][ctx]["pct_other"] * 100 for n in models]

        bars_s = ax.bar(x - width, pct_straight, width, label="Straight",
                        color="#4A90D9", edgecolor="white", linewidth=0.5)
        bars_f = ax.bar(x, pct_funny, width, label="Funny",
                        color="#E85D75", edgecolor="white", linewidth=0.5)
        bars_o = ax.bar(x + width, pct_other, width, label="Other",
                        color="#AAAAAA", edgecolor="white", linewidth=0.5)

        # Value labels
        for bars in [bars_s, bars_f, bars_o]:
            for bar in bars:
                h = bar.get_height()
                if h >= 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                            f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("% of responses" if ctx == "straight" else "")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Model Response Classification by Priming Context",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "context_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'context_effect.png'}")

    # ── Figure 2: Funny-rate shift bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    pf_str = [summaries[n]["straight"]["pct_funny"] * 100 for n in models]
    pf_fun = [summaries[n]["funny"]["pct_funny"] * 100 for n in models]
    deltas = [f - s for f, s in zip(pf_fun, pf_str)]

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width / 2, pf_str, width, label="Straight context",
           color="#4A90D9", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, pf_fun, width, label="Funny context",
           color="#E85D75", edgecolor="white", linewidth=0.5)

    # Delta annotations
    for i in range(len(models)):
        y_top = max(pf_str[i], pf_fun[i])
        d = deltas[i]
        label = f"+{d:.0f}pp" if d >= 0 else f"{d:.0f}pp"
        ax.annotate(label, xy=(x[i], y_top + 2),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="#333333")

    ax.set_ylabel("Funny completion rate (%)", fontsize=11)
    ax.set_title("Context Effect on Pun Completion Rate",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(pf_fun) + 15)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "funny_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'funny_shift.png'}")


def save_analysis(summaries, classified):
    """Save structured analysis to JSON."""
    analysis = {
        "model_summaries": {},
        "per_response": {},
    }

    for nick in MODEL_ORDER:
        if nick not in summaries:
            continue
        ms = summaries[nick]
        pf_str = ms["straight"]["pct_funny"]
        pf_fun = ms["funny"]["pct_funny"]
        ps_str = ms["straight"]["pct_straight"]
        ps_fun = ms["funny"]["pct_straight"]

        analysis["model_summaries"][nick] = {
            "straight_context": ms["straight"],
            "funny_context": ms["funny"],
            "funny_rate_shift": round(pf_fun - pf_str, 4),
            "straight_rate_shift": round(ps_fun - ps_str, 4),
        }

    for nick in MODEL_ORDER:
        if nick not in classified:
            continue
        analysis["per_response"][nick] = classified[nick]

    with open(ANALYSIS_FILE, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved {ANALYSIS_FILE.relative_to(BASE)}")


def main():
    classified, tests = load_and_classify()
    summaries = compute_summaries(classified)
    print_tables(summaries)
    print()
    make_plots(summaries)
    save_analysis(summaries, classified)

    # Print markdown table for EXPERIMENTS.md
    print(f"\n{'='*90}")
    print("  MARKDOWN TABLE (for EXPERIMENTS.md)")
    print(f"{'='*90}")
    print()
    print("| Model | P(funny \\| straight ctx) | P(funny \\| funny ctx) | Delta | P(straight \\| straight ctx) | P(straight \\| funny ctx) |")
    print("|-------|------------------------:|---------------------:|------:|---------------------------:|------------------------:|")
    for nick in MODEL_ORDER:
        if nick not in summaries:
            continue
        ms = summaries[nick]
        pf_s = ms["straight"]["pct_funny"]
        pf_f = ms["funny"]["pct_funny"]
        ps_s = ms["straight"]["pct_straight"]
        ps_f = ms["funny"]["pct_straight"]
        delta = (pf_f - pf_s) * 100
        delta_str = f"+{delta:.0f}pp" if delta >= 0 else f"{delta:.0f}pp"
        print(f"| {MODEL_LABELS_INLINE[nick]} | {pf_s:.0%} | {pf_f:.0%} | "
              f"{delta_str} | {ps_s:.0%} | {ps_f:.0%} |")


if __name__ == "__main__":
    main()
