#!/usr/bin/env python3
"""
Build 150 short context cloze test prompts from puns_205.json.

Format: Each prompt is "A_completed. C_truncated" where:
  - A is a context joke completed with either straight or funny word
  - C is a target joke truncated before the blank

This is a simpler 2-sentence format compared to the 3-sentence format in
contextual_cloze_tests_100.json.

Algorithm:
  1. Partition 205 jokes by cloze_tier
  2. C_pool (targets): 75 from 88 straight_dominated jokes
  3. A_pool (context): 75 from remaining straight_dominated + balanced + leaning
     (exclude funny_dominated - they always trigger pun completions)
  4. Form 75 pairs: each pair (A, C) produces 2 prompts:
     - straight: "S(A). C" where S(A) = A completed with straight word
     - funny: "F(A). C" where F(A) = A completed with funny/pun word
  5. Output: 150 test entries (75 pairs × 2 types)

Usage:
    python3 build_short_contrast_150.py                    # Generate tests
    python3 build_short_contrast_150.py --dry-run          # Preview pools
    python3 build_short_contrast_150.py --seed 42 --output out.json
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


def build_pools(jokes, seed=42):
    """Partition jokes into C_pool (targets) and A_pool (context) by tier.

    Strategy:
    - C targets: straight_dominated jokes (models default to straight)
    - A context: balanced + leaning + remaining straight_dominated
    - Exclude funny_dominated (always puns, no contrast)
    """
    balanced = [j for j in jokes if j.get("cloze_tier") == "balanced"]
    leaning = [j for j in jokes if j.get("cloze_tier") == "leaning"]
    straight_dominated = [j for j in jokes if j.get("cloze_tier") == "straight_dominated"]
    funny_dominated = [j for j in jokes if j.get("cloze_tier") == "funny_dominated"]

    rng = random.Random(seed)

    # Shuffle straight_dominated: first 75 for C, remainder available for A
    sd_shuffled = list(straight_dominated)
    rng.shuffle(sd_shuffled)

    c_pool = sd_shuffled[:75]
    sd_remainder = sd_shuffled[75:]  # 13 leftover

    # A pool: balanced + leaning + remaining straight_dominated
    # Need 75, have 42 + 51 + 13 = 106 available
    a_candidates = balanced + leaning + sd_remainder
    rng.shuffle(a_candidates)
    a_pool = a_candidates[:75]

    return c_pool, a_pool, {
        "balanced": len(balanced),
        "leaning": len(leaning),
        "straight_dominated": len(straight_dominated),
        "funny_dominated": len(funny_dominated),
        "c_pool": len(c_pool),
        "sd_remainder": len(sd_remainder),
        "a_candidates": len(a_candidates),
        "a_pool": len(a_pool),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build short 2-sentence contrastive cloze tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output", type=str,
                        default="short_context_cloze_150.json",
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pool composition; don't write")
    args = parser.parse_args()

    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)

    c_pool, a_pool, stats = build_pools(jokes, seed=args.seed)

    print("Pool composition:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\nSample pairs (first 5):")
        for i in range(min(5, len(c_pool))):
            a, c = a_pool[i], c_pool[i]
            print(f"\n  Pair {i}:")
            print(f"    A ({a['cloze_tier']}): {a['sentence']}")
            print(f"      straight: {a['straight'][0] if a['straight'] else '?'}")
            print(f"      funny: {a['punny'][0] if a['punny'] else '?'}")
            print(f"    C ({c['cloze_tier']}): {c['sentence']}")
        return

    # Build test entries
    tests = []
    for pair_id, (a_joke, c_joke) in enumerate(zip(a_pool, c_pool)):
        # Get completions
        a_straight = a_joke["straight"][0] if a_joke["straight"] else ""
        a_funny = a_joke["punny"][0] if a_joke["punny"] else ""

        # Build the two prompt variants
        a_straight_filled = fill_sentence(a_joke["sentence"], a_straight)
        a_funny_filled = fill_sentence(a_joke["sentence"], a_funny)
        c_truncated = truncate_before_blank(c_joke["sentence"])

        # Expected completions for C
        c_straight = c_joke["straight"]
        c_funny = c_joke["punny"]

        # Straight-primed prompt: "S(A). C"
        tests.append({
            "pair_id": pair_id,
            "type": "straight",
            "prompt": f"{a_straight_filled} {c_truncated}",
            "expected_completion": c_straight,
            "contrast_completion": c_funny,
            "joke_a_index": a_joke["index"],
            "joke_c_index": c_joke["index"],
            "joke_a_sentence": a_joke["sentence"],
            "joke_c_sentence": c_joke["sentence"],
            "joke_a_tier": a_joke["cloze_tier"],
            "joke_c_tier": c_joke["cloze_tier"],
            "joke_c_balance": c_joke.get("cloze_balance", 0),
            "a_completion": a_straight,
        })

        # Funny-primed prompt: "F(A). C"
        tests.append({
            "pair_id": pair_id,
            "type": "funny",
            "prompt": f"{a_funny_filled} {c_truncated}",
            "expected_completion": c_funny,
            "contrast_completion": c_straight,
            "joke_a_index": a_joke["index"],
            "joke_c_index": c_joke["index"],
            "joke_a_sentence": a_joke["sentence"],
            "joke_c_sentence": c_joke["sentence"],
            "joke_a_tier": a_joke["cloze_tier"],
            "joke_c_tier": c_joke["cloze_tier"],
            "joke_c_balance": c_joke.get("cloze_balance", 0),
            "a_completion": a_funny,
        })

    # Write output
    output_path = BASE / args.output
    with open(output_path, "w") as f:
        json.dump(tests, f, indent=2)

    print(f"\nGenerated {len(tests)} test entries ({len(tests)//2} pairs)")
    print(f"Output: {output_path}")

    # Show a few examples
    print("\nExample prompts:")
    for i in range(0, min(6, len(tests)), 2):
        s, fu = tests[i], tests[i+1]
        print(f"\n  Pair {s['pair_id']}:")
        print(f"    straight: \"{s['prompt']}\"")
        print(f"    funny:    \"{fu['prompt']}\"")
        print(f"    C expects: {s['expected_completion'][:3]}... vs {s['contrast_completion'][:1]}...")


if __name__ == "__main__":
    main()
