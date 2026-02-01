#!/usr/bin/env python3
"""
Analyze the contextual priming effect on pun completions.

For each model response in the cloze benchmark, classify the word as
straight/funny/other using the expanded puns_205.json lists (absolute
classification). Then compute how the funny/straight split shifts between
straight and funny priming contexts.

Key metrics:
  - P(funny | straight_context) vs P(funny | funny_context) per model
  - P(straight | straight_context) vs P(straight | funny_context) per model
  - Per-question breakdown of the funny-rate shift
"""

import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent


def load_data():
    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)
    with open(BASE / "cloze_benchmark_results.json") as f:
        results = json.load(f)
    with open(BASE / "contextual_cloze_tests_100.json") as f:
        tests = json.load(f)
    return jokes, results, tests


def classify_word(word, straight_list, funny_list):
    """Classify a word as 'straight', 'funny', or 'other' using absolute lists."""
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


def main():
    jokes, results, tests = load_data()

    # Build joke lookup by index
    joke_by_index = {j["index"]: j for j in jokes}

    # Build test lookup by (pair_id, type)
    test_lookup = {}
    for t in tests:
        test_lookup[(t["pair_id"], t["type"])] = t

    # ── Per-model analysis ──
    print(f"{'='*80}")
    print("  CONTEXTUAL PRIMING EFFECT: Funny/Straight Split by Context")
    print(f"{'='*80}")

    # Store per-model, per-question data for later
    all_classifications = {}  # model -> list of {pair_id, context, classification, word}

    for model_nick, mdata in results.items():
        classifications = []
        for d in mdata["details"]:
            pair_id = d["pair_id"]
            context_type = d["type"]  # "straight" or "funny" = the priming context
            word = d["extracted_word"]

            # Look up the target joke
            test = test_lookup.get((pair_id, context_type))
            if not test:
                continue
            joke_idx = test["joke_c_index"]
            joke = joke_by_index.get(joke_idx)
            if not joke:
                continue

            cls = classify_word(word, joke["straight"], joke["punny"])
            classifications.append({
                "pair_id": pair_id,
                "context": context_type,
                "classification": cls,
                "word": word,
                "joke_index": joke_idx,
                "joke_sentence": joke["sentence"],
            })

        all_classifications[model_nick] = classifications

    # ── Summary table: per model ──
    print(f"\n  {'Model':<12} │ {'Context':<10} │ {'Straight':>8} {'Funny':>8} {'Other':>8} │ {'N':>3}")
    print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*3}")

    model_summaries = {}
    for model_nick in results:
        cls_list = all_classifications[model_nick]
        for ctx in ["straight", "funny"]:
            subset = [c for c in cls_list if c["context"] == ctx]
            n = len(subset)
            n_straight = sum(1 for c in subset if c["classification"] == "straight")
            n_funny = sum(1 for c in subset if c["classification"] == "funny")
            n_other = sum(1 for c in subset if c["classification"] == "other")

            pct_s = n_straight / n if n else 0
            pct_f = n_funny / n if n else 0
            pct_o = n_other / n if n else 0

            print(f"  {model_nick:<12} │ {ctx:<10} │ {pct_s:>7.1%} {pct_f:>8.1%} {pct_o:>8.1%} │ {n:>3}")

            if model_nick not in model_summaries:
                model_summaries[model_nick] = {}
            model_summaries[model_nick][ctx] = {
                "n": n, "straight": n_straight, "funny": n_funny, "other": n_other,
                "pct_straight": pct_s, "pct_funny": pct_f, "pct_other": pct_o,
            }
        print(f"  {'─'*12}─┼─{'─'*10}─┼─{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*3}")

    # ── Funny-rate shift per model ──
    print(f"\n  FUNNY-RATE SHIFT (increase in P(funny) from straight→funny context)")
    print(f"  {'Model':<12} │ P(funny|str) │ P(funny|fun) │ {'Δ funny':>8} │ P(str|str) │ P(str|fun) │ {'Δ str':>8}")
    print(f"  {'─'*12}─┼──{'─'*11}─┼──{'─'*11}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}")

    for model_nick in results:
        ms = model_summaries[model_nick]
        pf_str = ms["straight"]["pct_funny"]
        pf_fun = ms["funny"]["pct_funny"]
        delta_f = pf_fun - pf_str

        ps_str = ms["straight"]["pct_straight"]
        ps_fun = ms["funny"]["pct_straight"]
        delta_s = ps_fun - ps_str

        print(f"  {model_nick:<12} │ {pf_str:>11.1%} │ {pf_fun:>11.1%} │ {delta_f:>+7.1%} │ "
              f"{ps_str:>10.1%} │ {ps_fun:>10.1%} │ {delta_s:>+7.1%}")

    # ── Per-question (pair_id) analysis ──
    print(f"\n{'='*80}")
    print("  PER-QUESTION FUNNY-RATE SHIFT (averaged across all models)")
    print(f"{'='*80}")

    # Gather per-question stats across all models
    pair_ids = sorted(set(c["pair_id"] for cls_list in all_classifications.values()
                         for c in cls_list))

    question_stats = {}
    for pid in pair_ids:
        stats = {"straight": {"funny": 0, "straight": 0, "other": 0, "n": 0},
                 "funny": {"funny": 0, "straight": 0, "other": 0, "n": 0}}
        for model_nick in results:
            for c in all_classifications[model_nick]:
                if c["pair_id"] == pid:
                    ctx = c["context"]
                    stats[ctx][c["classification"]] += 1
                    stats[ctx]["n"] += 1
        question_stats[pid] = stats

    # Compute funny-rate shift per question
    shifts = []
    for pid in pair_ids:
        s = question_stats[pid]
        n_str = s["straight"]["n"]
        n_fun = s["funny"]["n"]
        pf_str = s["straight"]["funny"] / n_str if n_str else 0
        pf_fun = s["funny"]["funny"] / n_fun if n_fun else 0
        ps_str = s["straight"]["straight"] / n_str if n_str else 0
        ps_fun = s["funny"]["straight"] / n_fun if n_fun else 0
        delta = pf_fun - pf_str

        # Get the joke sentence for display
        sentence = ""
        for cls_list in all_classifications.values():
            for c in cls_list:
                if c["pair_id"] == pid:
                    sentence = c["joke_sentence"]
                    break
            if sentence:
                break

        shifts.append((pid, delta, pf_str, pf_fun, ps_str, ps_fun, sentence))

    # Sort by shift magnitude (descending)
    shifts.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'PID':>3} │ {'P(fun|str)':>10} {'P(fun|fun)':>10} {'Δ funny':>8} │ "
          f"{'P(str|str)':>10} {'P(str|fun)':>10} {'Δ str':>8} │ Joke")
    print(f"  {'─'*3}─┼─{'─'*10}─{'─'*10}─{'─'*8}─┼─{'─'*10}─{'─'*10}─{'─'*8}─┼─{'─'*40}")

    for pid, delta, pf_str, pf_fun, ps_str, ps_fun, sentence in shifts:
        short = sentence[:50] + "..." if len(sentence) > 50 else sentence
        print(f"  {pid:>3} │ {pf_str:>9.0%} {pf_fun:>10.0%} {delta:>+7.0%} │ "
              f"{ps_str:>9.0%} {ps_fun:>10.0%} {ps_fun-ps_str:>+7.0%} │ {short}")

    # ── Per-model × per-question detail ──
    print(f"\n{'='*80}")
    print("  PER-MODEL × PER-QUESTION DETAIL (top 15 questions by funny shift)")
    print(f"{'='*80}")

    top_questions = [s[0] for s in shifts[:15]]
    for pid in top_questions:
        sentence = ""
        for s in shifts:
            if s[0] == pid:
                sentence = s[6]
                break
        print(f"\n  Pair {pid}: {sentence[:70]}")
        print(f"  {'Model':<12} │ Str ctx → word (cls)          │ Fun ctx → word (cls)")
        print(f"  {'─'*12}─┼─{'─'*30}─┼─{'─'*30}")

        for model_nick in results:
            str_entries = [c for c in all_classifications[model_nick]
                          if c["pair_id"] == pid and c["context"] == "straight"]
            fun_entries = [c for c in all_classifications[model_nick]
                          if c["pair_id"] == pid and c["context"] == "funny"]

            str_word = str_entries[0]["word"] if str_entries else "?"
            str_cls = str_entries[0]["classification"] if str_entries else "?"
            fun_word = fun_entries[0]["word"] if fun_entries else "?"
            fun_cls = fun_entries[0]["classification"] if fun_entries else "?"

            marker_s = "✓" if str_cls == "straight" else ("✗" if str_cls == "funny" else "·")
            marker_f = "✓" if fun_cls == "funny" else ("✗" if fun_cls == "straight" else "·")

            print(f"  {model_nick:<12} │ {str_word:>15} ({str_cls:<8}) {marker_s} │ "
                  f"{fun_word:>15} ({fun_cls:<8}) {marker_f}")

    # ── Save analysis results ──
    analysis = {
        "model_summaries": model_summaries,
        "question_shifts": [
            {
                "pair_id": pid,
                "funny_shift": delta,
                "p_funny_straight_ctx": pf_str,
                "p_funny_funny_ctx": pf_fun,
                "p_straight_straight_ctx": ps_str,
                "p_straight_funny_ctx": ps_fun,
                "joke_sentence": sentence,
            }
            for pid, delta, pf_str, pf_fun, ps_str, ps_fun, sentence in shifts
        ],
    }
    with open(BASE / "context_effect_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n\nSaved analysis to context_effect_analysis.json")


if __name__ == "__main__":
    main()
