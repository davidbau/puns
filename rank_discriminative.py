#!/usr/bin/env python3
"""
Rank jokes by how discriminative they are — i.e., how much model predictions
are split between straight and funny completions.

Uses the classified_completions.json data (100 samples per joke across 5 models)
to compute the straight/funny/other sample split, then ranks by a discriminativeness
score based on how balanced the straight vs funny competition is.

Discriminativeness = min(p_straight, p_funny) / max(p_straight, p_funny)
  - 1.0 = perfectly balanced (50/50 straight-funny split)
  - 0.0 = completely one-sided (all straight or all funny)
  - Only computed when both straight and funny counts > 0
"""

import json
from pathlib import Path

BASE = Path(__file__).parent


def main():
    with open(BASE / "classified_completions.json") as f:
        completions = json.load(f)
    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)

    joke_by_index = {j["index"]: j for j in jokes}

    rows = []
    for key, data in completions.items():
        idx = data["index"]
        sentence = data["sentence"]
        word_counts = data["aggregated"]["word_counts"]
        classifications = data["classifications"]
        total = data["aggregated"]["total_samples"]

        # Tally samples by classification
        n_straight = 0
        n_funny = 0
        n_other = 0
        straight_words = {}
        funny_words = {}

        for word, count in word_counts.items():
            cls = classifications.get(word, "other")
            if cls == "straight":
                n_straight += count
                straight_words[word] = count
            elif cls == "funny":
                n_funny += count
                funny_words[word] = count
            else:
                n_other += count

        p_straight = n_straight / total if total else 0
        p_funny = n_funny / total if total else 0
        p_other = n_other / total if total else 0

        # Discriminativeness: how balanced is the straight/funny competition?
        if n_straight > 0 and n_funny > 0:
            # Use min/max ratio — 1.0 = perfect balance
            balance = min(p_straight, p_funny) / max(p_straight, p_funny)
        elif n_funny > 0:
            balance = 0.0  # all funny, no competition
        else:
            balance = 0.0  # all straight or other, no competition

        # Also compute per-model breakdown
        model_funny = {}
        for model_name, mdata in data["models"].items():
            samples = mdata.get("samples", [])
            mf = sum(1 for s in samples if classifications.get(s, "other") == "funny")
            model_funny[model_name] = mf / len(samples) if samples else 0

        rows.append({
            "index": idx,
            "sentence": sentence,
            "n_straight": n_straight,
            "n_funny": n_funny,
            "n_other": n_other,
            "total": total,
            "p_straight": p_straight,
            "p_funny": p_funny,
            "p_other": p_other,
            "balance": balance,
            "straight_words": straight_words,
            "funny_words": funny_words,
            "model_funny": model_funny,
        })

    # Sort by discriminativeness (balance), then by p_funny to break ties
    rows.sort(key=lambda r: (r["balance"], r["p_funny"]), reverse=True)

    # ── Print results ──
    print(f"{'='*100}")
    print("  JOKE DISCRIMINATIVENESS RANKING")
    print(f"  (balance = min(p_str,p_fun)/max(p_str,p_fun); 1.0 = perfectly split)")
    print(f"{'='*100}")

    print(f"\n  {'Idx':>3} {'Bal':>5} │ {'Str%':>5} {'Fun%':>5} {'Oth%':>5} │ "
          f"{'3b':>4} {'8b':>4} {'70b':>4} {'3.3':>4} {'405':>4} │ Joke")
    print(f"  {'─'*3} {'─'*5} ┼ {'─'*5} {'─'*5} {'─'*5} ┼ "
          f"{'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4} ┼ {'─'*45}")

    for r in rows:
        short = r["sentence"][:50] + "..." if len(r["sentence"]) > 50 else r["sentence"]
        mf = r["model_funny"]
        print(f"  {r['index']:>3} {r['balance']:>5.2f} │ "
              f"{r['p_straight']:>4.0%} {r['p_funny']:>5.0%} {r['p_other']:>5.0%} │ "
              f"{mf.get('3b',0):>3.0%} {mf.get('8b',0):>4.0%} "
              f"{mf.get('70b',0):>4.0%} {mf.get('3.3-70b',0):>4.0%} "
              f"{mf.get('405b',0):>4.0%} │ {short}")

    # ── Tier summary ──
    high = [r for r in rows if r["balance"] >= 0.3 and r["p_funny"] >= 0.05]
    medium = [r for r in rows if 0.05 < r["balance"] < 0.3 and r["p_funny"] >= 0.05]
    low_funny_only = [r for r in rows if r["balance"] == 0 and r["p_funny"] > 0.5]
    low_straight_only = [r for r in rows if r["p_funny"] < 0.05]

    print(f"\n{'='*100}")
    print(f"  TIER SUMMARY")
    print(f"{'='*100}")
    print(f"  High discriminativeness (balance >= 0.3, p_funny >= 5%):  {len(high)} jokes")
    print(f"  Medium (0.05 < balance < 0.3, p_funny >= 5%):            {len(medium)} jokes")
    print(f"  Low — funny-dominated (p_funny > 50%, balance = 0):      {len(low_funny_only)} jokes")
    print(f"  Low — straight-dominated (p_funny < 5%):                 {len(low_straight_only)} jokes")

    # ── Top discriminative jokes detail ──
    print(f"\n{'='*100}")
    print(f"  TOP 30 MOST DISCRIMINATIVE JOKES (detailed)")
    print(f"{'='*100}")

    for r in rows[:30]:
        print(f"\n  [{r['index']}] {r['sentence']}")
        top_str = sorted(r["straight_words"].items(), key=lambda x: -x[1])[:5]
        top_fun = sorted(r["funny_words"].items(), key=lambda x: -x[1])[:5]
        str_str = ", ".join(f"{w}({c})" for w, c in top_str)
        fun_str = ", ".join(f"{w}({c})" for w, c in top_fun)
        print(f"    Straight {r['p_straight']:.0%}: {str_str}")
        print(f"    Funny    {r['p_funny']:.0%}: {fun_str}")
        print(f"    Balance: {r['balance']:.2f}  |  "
              f"Per-model funny%: 3b={r['model_funny'].get('3b',0):.0%} "
              f"8b={r['model_funny'].get('8b',0):.0%} "
              f"70b={r['model_funny'].get('70b',0):.0%} "
              f"3.3-70b={r['model_funny'].get('3.3-70b',0):.0%} "
              f"405b={r['model_funny'].get('405b',0):.0%}")

    # ── Save ranked data ──
    output = []
    for r in rows:
        output.append({
            "index": r["index"],
            "sentence": r["sentence"],
            "balance": round(r["balance"], 4),
            "p_straight": round(r["p_straight"], 4),
            "p_funny": round(r["p_funny"], 4),
            "p_other": round(r["p_other"], 4),
            "n_straight": r["n_straight"],
            "n_funny": r["n_funny"],
            "n_other": r["n_other"],
            "straight_words": r["straight_words"],
            "funny_words": r["funny_words"],
            "model_funny_rate": {k: round(v, 4) for k, v in r["model_funny"].items()},
        })

    with open(BASE / "discriminative_ranking.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved ranking to discriminative_ranking.json")


if __name__ == "__main__":
    main()
