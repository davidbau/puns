#!/usr/bin/env python3
"""
Build 150 single-sentence funny/serious prompts from puns_205.json.

Format: Each prompt is a single completed sentence "C_filled."
  - S(C): sentence completed with straight/serious word
  - F(C): sentence completed with funny/pun word

This dataset tests whether the model's representation at the sentence end
differs based on whether the completion was funny or serious.

Algorithm:
  1. Select 75 jokes from the 88 straight_dominated jokes
  2. Each joke produces 2 prompts:
     - serious: "The tailor won his court case because he had an excellent lawyer."
     - funny:   "The tailor won his court case because he had an excellent suit."
  3. Output: 150 prompts (75 jokes × 2 types)

Token position for activation collection: last token (the period).

Usage:
    python3 build_funny_serious_150.py                    # Generate tests
    python3 build_funny_serious_150.py --dry-run          # Preview
    python3 build_funny_serious_150.py --seed 42 --output out.json
"""

import json
import random
import argparse
from pathlib import Path

BASE = Path(__file__).parent


def fill_sentence(sentence, completion):
    """Replace ___ in sentence with the completion and add period."""
    filled = sentence.replace("___", completion)
    # Ensure sentence ends with period
    filled = filled.rstrip()
    if not filled.endswith('.'):
        filled += '.'
    return filled


def main():
    parser = argparse.ArgumentParser(
        description="Build single-sentence funny/serious test prompts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output", type=str,
                        default="funny_serious_150.json",
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample prompts; don't write")
    args = parser.parse_args()

    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)

    # Select straight_dominated jokes
    straight_dominated = [j for j in jokes if j.get("cloze_tier") == "straight_dominated"]
    print(f"Found {len(straight_dominated)} straight_dominated jokes")

    # Shuffle and select 75
    rng = random.Random(args.seed)
    sd_shuffled = list(straight_dominated)
    rng.shuffle(sd_shuffled)
    selected = sd_shuffled[:75]

    print(f"Selected {len(selected)} jokes for dataset")

    if args.dry_run:
        print("\nSample prompts (first 5):")
        for i, joke in enumerate(selected[:5]):
            straight_word = joke["straight"][0] if joke["straight"] else "?"
            funny_word = joke["punny"][0] if joke["punny"] else "?"
            print(f"\n  Joke {i} (index={joke['index']}):")
            print(f"    serious: \"{fill_sentence(joke['sentence'], straight_word)}\"")
            print(f"    funny:   \"{fill_sentence(joke['sentence'], funny_word)}\"")
        return

    # Build test entries
    tests = []
    for pair_id, joke in enumerate(selected):
        straight_word = joke["straight"][0] if joke["straight"] else ""
        funny_word = joke["punny"][0] if joke["punny"] else ""

        # Serious prompt
        tests.append({
            "pair_id": pair_id,
            "type": "straight",
            "prompt": fill_sentence(joke["sentence"], straight_word),
            "completion_word": straight_word,
            "contrast_word": funny_word,
            "joke_index": joke["index"],
            "joke_sentence": joke["sentence"],
            "straight_words": joke["straight"],
            "punny_words": joke["punny"],
        })

        # Funny prompt
        tests.append({
            "pair_id": pair_id,
            "type": "funny",
            "prompt": fill_sentence(joke["sentence"], funny_word),
            "completion_word": funny_word,
            "contrast_word": straight_word,
            "joke_index": joke["index"],
            "joke_sentence": joke["sentence"],
            "straight_words": joke["straight"],
            "punny_words": joke["punny"],
        })

    # Write output
    output_path = BASE / args.output
    with open(output_path, "w") as f:
        json.dump(tests, f, indent=2)

    print(f"\nGenerated {len(tests)} test entries ({len(tests)//2} jokes)")
    print(f"Output: {output_path}")

    # Show examples
    print("\nExample prompts:")
    for i in range(0, min(6, len(tests)), 2):
        s, fu = tests[i], tests[i+1]
        print(f"\n  Pair {s['pair_id']}:")
        print(f"    serious: \"{s['prompt']}\"")
        print(f"    funny:   \"{fu['prompt']}\"")


if __name__ == "__main__":
    main()
