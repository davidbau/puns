#!/usr/bin/env python3
"""
Build 100 contrastive cloze test prompts from puns_205.json using tier-aware
triple selection.

Algorithm:
  1. Partition 205 jokes by cloze_tier
  2. Sort leaning jokes by cloze_balance DESC (most balanced first)
  3. Split leaning into top-8 (→ C pool) and next-12 (→ AB pool)
  4. C_pool  = balanced(42) + leaning_top8(8)   = 50 target jokes
     AB_pool = straight_dominated(88) + leaning_next12(12) = 100 context jokes
  5. Shuffle both pools deterministically (seed=42)
  6. Form 50 triples: A=AB[2i], B=AB[2i+1], C=C_pool[i]
  7. Emit 2 tests per triple (straight-primed and funny-primed)

Output: 100 test entries in contextual_cloze_tests_100.json

Usage:
    python3 build_cloze_tests.py                          # Generate tests
    python3 build_cloze_tests.py --dry-run                # Preview pool composition
    python3 build_cloze_tests.py --seed 42 --output out.json
"""

import json
import random
import argparse
from pathlib import Path

BASE = Path(__file__).parent


def fill_sentence(sentence, completion):
    """Replace ___ in sentence with the completion."""
    return sentence.replace("___", completion)


def truncate_before_blank(sentence):
    """Truncate sentence just before ___, removing trailing whitespace."""
    idx = sentence.index("___")
    return sentence[:idx].rstrip()


def build_pools(jokes):
    """Partition jokes into C_pool (targets) and AB_pool (context) by tier."""
    balanced = [j for j in jokes if j.get("cloze_tier") == "balanced"]
    leaning = [j for j in jokes if j.get("cloze_tier") == "leaning"]
    straight_dominated = [j for j in jokes if j.get("cloze_tier") == "straight_dominated"]
    funny_dominated = [j for j in jokes if j.get("cloze_tier") == "funny_dominated"]

    # Sort leaning by balance DESC (most balanced first → best targets)
    leaning.sort(key=lambda j: j.get("cloze_balance", 0), reverse=True)

    leaning_for_c = leaning[:8]
    leaning_for_ab = leaning[8:20]

    c_pool = balanced + leaning_for_c
    ab_pool = straight_dominated + leaning_for_ab

    return c_pool, ab_pool, {
        "balanced": len(balanced),
        "leaning": len(leaning),
        "straight_dominated": len(straight_dominated),
        "funny_dominated": len(funny_dominated),
        "leaning_for_c": len(leaning_for_c),
        "leaning_for_ab": len(leaning_for_ab),
        "c_pool": len(c_pool),
        "ab_pool": len(ab_pool),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build tier-aware contrastive cloze tests from puns_205.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output", type=str,
                        default="contextual_cloze_tests_100.json",
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pool composition and sample triples; don't write")
    args = parser.parse_args()

    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)

    c_pool, ab_pool, stats = build_pools(jokes)

    # ── Print pool composition ───────────────────────────────────────────
    print(f"Pool composition (seed={args.seed}):")
    print(f"  Tier counts: {stats['balanced']} balanced, {stats['leaning']} leaning, "
          f"{stats['straight_dominated']} straight_dominated, {stats['funny_dominated']} funny_dominated")
    print(f"  Leaning split: top {stats['leaning_for_c']} → C pool, "
          f"next {stats['leaning_for_ab']} → AB pool")
    print(f"  C pool  (targets):  {stats['c_pool']} jokes "
          f"({stats['balanced']} balanced + {stats['leaning_for_c']} leaning)")
    print(f"  AB pool (context):  {stats['ab_pool']} jokes "
          f"({stats['straight_dominated']} straight_dominated + {stats['leaning_for_ab']} leaning)")

    assert len(c_pool) == 50, f"Expected 50 C-pool jokes, got {len(c_pool)}"
    assert len(ab_pool) == 100, f"Expected 100 AB-pool jokes, got {len(ab_pool)}"

    # Verify no overlap
    c_indices = {j["index"] for j in c_pool}
    ab_indices = {j["index"] for j in ab_pool}
    overlap = c_indices & ab_indices
    assert not overlap, f"Overlap between C and AB pools: {overlap}"

    # ── Shuffle deterministically ────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(c_pool)
    random.shuffle(ab_pool)

    if args.dry_run:
        print(f"\n  Sample triples (first 5):")
        for i in range(min(5, 50)):
            a, b, c = ab_pool[2 * i], ab_pool[2 * i + 1], c_pool[i]
            print(f"    Triple {i}: A=idx{a['index']}({a.get('cloze_tier','?')}) "
                  f"B=idx{b['index']}({b.get('cloze_tier','?')}) "
                  f"C=idx{c['index']}({c.get('cloze_tier','?')}, "
                  f"bal={c.get('cloze_balance',0):.2f})")
            print(f"      C: {c['sentence'][:70]}...")
            print(f"      C straight[0]: {c['straight'][0]}  |  C punny[0]: {c['punny'][0]}")
        print(f"\n  (dry-run: no output file written)")
        return

    # ── Form 50 triples → 100 tests ─────────────────────────────────────
    tests = []
    for i in range(50):
        a = ab_pool[2 * i]
        b = ab_pool[2 * i + 1]
        c = c_pool[i]

        a_straight = a["straight"][0]
        b_straight = b["straight"][0]
        a_funny = a["punny"][0]
        b_funny = b["punny"][0]

        sa = fill_sentence(a["sentence"], a_straight)
        sb = fill_sentence(b["sentence"], b_straight)
        fa = fill_sentence(a["sentence"], a_funny)
        fb = fill_sentence(b["sentence"], b_funny)

        c_truncated = truncate_before_blank(c["sentence"])

        straight_prompt = f"{sa} {sb} {c_truncated}"
        funny_prompt = f"{fa} {fb} {c_truncated}"

        base_fields = {
            "joke_a_index": a["index"],
            "joke_b_index": b["index"],
            "joke_c_index": c["index"],
            "joke_c_sentence": c["sentence"],
            "joke_a_tier": a.get("cloze_tier", ""),
            "joke_b_tier": b.get("cloze_tier", ""),
            "joke_c_tier": c.get("cloze_tier", ""),
            "joke_c_balance": c.get("cloze_balance", 0),
        }

        tests.append({
            "pair_id": i,
            "type": "straight",
            "prompt": straight_prompt,
            "expected_completion": c["straight"],
            "contrast_completion": c["punny"],
            **base_fields,
        })

        tests.append({
            "pair_id": i,
            "type": "funny",
            "prompt": funny_prompt,
            "expected_completion": c["punny"],
            "contrast_completion": c["straight"],
            **base_fields,
        })

    # ── Write output ─────────────────────────────────────────────────────
    output_path = BASE / args.output
    with open(output_path, "w") as f:
        json.dump(tests, f, indent=2)

    print(f"\nWrote {len(tests)} test prompts to {args.output}")
    print(f"  50 pairs x 2 (straight + funny) = {len(tests)} entries")

    # ── Show examples ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("EXAMPLE PAIR 0:")
    print(f"{'='*70}")
    s, f_ = tests[0], tests[1]
    print(f"\n  STRAIGHT prompt:")
    print(f"    {s['prompt'][:120]}...")
    print(f"    Expected: {s['expected_completion'][:5]}")
    print(f"    A_tier={s['joke_a_tier']}  B_tier={s['joke_b_tier']}  "
          f"C_tier={s['joke_c_tier']}  C_bal={s['joke_c_balance']:.2f}")
    print(f"\n  FUNNY prompt:")
    print(f"    {f_['prompt'][:120]}...")
    print(f"    Expected: {f_['expected_completion'][:5]}")

    print(f"\n{'='*70}")
    print("EXAMPLE PAIR 1:")
    print(f"{'='*70}")
    s, f_ = tests[2], tests[3]
    print(f"\n  STRAIGHT prompt:")
    print(f"    {s['prompt'][:120]}...")
    print(f"    Expected: {s['expected_completion'][:5]}")
    print(f"\n  FUNNY prompt:")
    print(f"    {f_['prompt'][:120]}...")
    print(f"    Expected: {f_['expected_completion'][:5]}")


if __name__ == "__main__":
    main()
