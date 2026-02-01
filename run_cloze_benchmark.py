#!/usr/bin/env python3
"""
Benchmark language models on contextual cloze pun tests using Together.ai.

Tests whether models can distinguish straight vs. funny completions
when primed with 2 context sentences filled in straight or funny.

Models tested are chosen to overlap with NDIF hot models for later
interpretability analysis via nnsight.

Usage:
    python3 run_cloze_benchmark.py                    # Run all models
    python3 run_cloze_benchmark.py --models 8b 70b    # Run subset by nickname
    python3 run_cloze_benchmark.py --dry-run           # Show config, don't call API
    python3 run_cloze_benchmark.py --limit 10          # Only run first 10 tests
"""

import json
import os
import sys
import time
import argparse
import re
from pathlib import Path

# Load environment variables from .env.local
BASE = Path(__file__).parent
env_path = BASE / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
if not TOGETHER_API_KEY:
    print("ERROR: TOGETHER_API_KEY not found in environment or .env.local")
    sys.exit(1)

MAX_RETRIES = 3
RETRY_DELAY = 5

# ── Model registry ──────────────────────────────────────────────────
# Each entry: (nickname, together_id, ndif_id_or_note, api_mode)
# api_mode: "chat" for chat/completions, "completion" for /v1/completions
MODELS = [
    ("3b",    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
              "meta-llama/Llama-3.2-3B (NDIF cold)", "chat"),
    ("8b",    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
              "meta-llama/Llama-3.1-8B (NDIF hot)", "chat"),
    ("70b",   "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
              "meta-llama/Llama-3.1-70B (NDIF hot)", "chat"),
    ("3.3-70b", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
              "meta-llama/Llama-3.3-70B-Instruct (NDIF hot)", "chat"),
    ("405b",  "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
              "meta-llama/Llama-3.1-405B-Instruct (NDIF hot)", "chat"),
]

SYSTEM_PROMPT = (
    "You are completing a text. Given the text below, output only the "
    "single next word that best continues it. Output ONLY that one word, "
    "nothing else — no punctuation, no explanation."
)


def call_together_chat(model_id, prompt, max_retries=MAX_RETRIES):
    """Call Together.ai chat completions and return the raw response text."""
    import requests as req_lib

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 10,
        "temperature": 0,
        "top_p": 1,
        "stop": ["\n", ".", ",", "!"],
    }

    for attempt in range(max_retries):
        try:
            resp = req_lib.post(
                "https://api.together.xyz/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"    FAILED after {max_retries} attempts: {e}")
                return None


def extract_first_word(text):
    """Extract the first word from model output, lowercased, stripped of punctuation."""
    if not text:
        return ""
    # Take first word, strip surrounding punctuation
    word = text.split()[0] if text.split() else ""
    word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
    return word.lower()


def matches_any(word, expected_list):
    """Check if word matches any item in expected list (case-insensitive)."""
    word = word.lower()
    for exp in expected_list:
        if word == exp.lower():
            return True
        # Also check if the expected is multi-word and word matches the first word
        first_exp = exp.lower().split()[0]
        if word == first_exp:
            return True
    return False


def run_benchmark(models, tests, dry_run=False):
    """Run all models on all tests. Returns results dict."""
    results = {}

    for nickname, together_id, ndif_note, api_mode in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {nickname} ({together_id})")
        print(f"  NDIF:  {ndif_note}")
        print(f"{'='*60}")

        if dry_run:
            print("  [dry-run] Skipping API calls")
            continue

        model_results = []
        correct = 0
        contrast_match = 0
        neither = 0

        for i, test in enumerate(tests):
            response = call_together_chat(together_id, test["prompt"])
            word = extract_first_word(response)

            hit_expected = matches_any(word, test["expected_completion"])
            hit_contrast = matches_any(word, test["contrast_completion"])

            if hit_expected:
                correct += 1
                outcome = "correct"
            elif hit_contrast:
                contrast_match += 1
                outcome = "contrast"
            else:
                neither += 1
                outcome = "other"

            model_results.append({
                "pair_id": test["pair_id"],
                "type": test["type"],
                "raw_response": response,
                "extracted_word": word,
                "outcome": outcome,
                "expected": test["expected_completion"],
                "contrast": test["contrast_completion"],
            })

            # Progress every 10
            if (i + 1) % 10 == 0 or i == len(tests) - 1:
                print(f"  [{i+1}/{len(tests)}] "
                      f"correct={correct} contrast={contrast_match} other={neither}")

        # Compute breakdowns
        straight_tests = [r for r in model_results if r["type"] == "straight"]
        funny_tests = [r for r in model_results if r["type"] == "funny"]

        def accuracy(subset):
            if not subset:
                return 0.0
            return sum(1 for r in subset if r["outcome"] == "correct") / len(subset)

        def contrast_rate(subset):
            if not subset:
                return 0.0
            return sum(1 for r in subset if r["outcome"] == "contrast") / len(subset)

        summary = {
            "model": together_id,
            "nickname": nickname,
            "ndif": ndif_note,
            "total": len(model_results),
            "correct": correct,
            "contrast_match": contrast_match,
            "other": neither,
            "accuracy": accuracy(model_results),
            "straight_accuracy": accuracy(straight_tests),
            "funny_accuracy": accuracy(funny_tests),
            "straight_contrast_rate": contrast_rate(straight_tests),
            "funny_contrast_rate": contrast_rate(funny_tests),
        }

        print(f"\n  SUMMARY for {nickname}:")
        print(f"    Overall accuracy:  {summary['accuracy']:.1%} "
              f"({correct}/{len(model_results)})")
        print(f"    Straight accuracy: {summary['straight_accuracy']:.1%}")
        print(f"    Funny accuracy:    {summary['funny_accuracy']:.1%}")
        print(f"    Contrast matches:  {contrast_match} "
              f"(straight: {summary['straight_contrast_rate']:.1%}, "
              f"funny: {summary['funny_contrast_rate']:.1%})")
        print(f"    Other/neither:     {neither}")

        results[nickname] = {
            "summary": summary,
            "details": model_results,
        }

    return results


def print_final_table(results):
    """Print a summary comparison table."""
    print(f"\n{'='*80}")
    print("  FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<12} {'Overall':>8} {'Straight':>9} {'Funny':>8} "
          f"{'Contrast':>9} {'Other':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")

    for nick, data in results.items():
        s = data["summary"]
        print(f"  {nick:<12} {s['accuracy']:>7.1%} {s['straight_accuracy']:>8.1%} "
              f"{s['funny_accuracy']:>7.1%} "
              f"{s['contrast_match']:>5}/{s['total']:<3} "
              f"{s['other']:>4}/{s['total']:<3}")

    print()
    print("  Accuracy = model's first word matches expected completion list")
    print("  Contrast = model gave the opposite condition's word (funny when straight expected, etc.)")
    print("  Other    = model gave a word not in either list")


def main():
    parser = argparse.ArgumentParser(description="Cloze pun benchmark via Together.ai")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Model nicknames to test: {[m[0] for m in MODELS]}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration without calling APIs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N tests")
    parser.add_argument("--output", type=str, default="cloze_benchmark_results.json",
                        help="Output filename (default: cloze_benchmark_results.json)")
    args = parser.parse_args()

    # Load tests
    test_file = BASE / "contextual_cloze_tests_100.json"
    with open(test_file) as f:
        tests = json.load(f)
    print(f"Loaded {len(tests)} cloze tests from {test_file.name}")

    if args.limit:
        tests = tests[:args.limit]
        print(f"  (limited to first {args.limit})")

    # Filter models
    if args.models:
        selected = [m for m in MODELS if m[0] in args.models]
        if not selected:
            print(f"ERROR: No matching models. Available: {[m[0] for m in MODELS]}")
            sys.exit(1)
    else:
        selected = MODELS

    print(f"\nModels to test ({len(selected)}):")
    for nick, tid, ndif, mode in selected:
        print(f"  {nick:>10}: {tid}")
        print(f"             → {ndif}")

    # Run benchmark
    results = run_benchmark(selected, tests, dry_run=args.dry_run)

    if not args.dry_run and results:
        # Print comparison table
        print_final_table(results)

        # Save results
        out_path = BASE / args.output
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {out_path.name}")


if __name__ == "__main__":
    main()
