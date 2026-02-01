#!/usr/bin/env python3
"""
Run contrastive cloze benchmark: test Llama models on pun completion tasks.

For each of 100 test prompts (50 pairs x 2 contexts), sends the prompt to
each model via Together.ai and records the first-word completion. Results
are saved as a checkpoint file that supports incremental backfill.

Usage:
    python3 run_cloze_benchmark.py                     # Run all models
    python3 run_cloze_benchmark.py --models 8b 70b     # Run subset
    python3 run_cloze_benchmark.py --backfill           # Retry failures
    python3 run_cloze_benchmark.py --dry-run            # Show config only
"""

import json
import os
import sys
import time
import re
import argparse
from pathlib import Path

BASE = Path(__file__).parent

# ── Load environment ─────────────────────────────────────────────────────────
env_path = BASE / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# ── Paths ────────────────────────────────────────────────────────────────────
TESTS_FILE = BASE / "datasets" / "contextual_cloze_tests_100.json"
RESULTS_FILE = BASE / "results" / "cloze_benchmark_raw.json"

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = [
    ("3b",     "meta-llama/Llama-3.2-3B-Instruct-Turbo"),
    ("8b",     "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("70b",    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("3.3-70b","meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("405b",   "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
]

SYSTEM_PROMPT = (
    "You are completing a text. Given the text below, output only the "
    "single next word that best continues it. Output ONLY that one word, "
    "nothing else — no punctuation, no explanation."
)

MAX_RETRIES = 3
RETRY_DELAY = 5

import requests as req_lib


def call_model(model_id, prompt):
    """Call Together.ai and return raw response text, or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = req_lib.post(
                "https://api.together.xyz/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 10,
                    "temperature": 0,
                    "top_p": 1,
                    "stop": ["\n", ".", ",", "!"],
                },
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"      FAILED: {e}")
                return None


def extract_first_word(text):
    """Extract first word, lowercase, stripped of punctuation."""
    if not text:
        return ""
    word = text.split()[0] if text.split() else ""
    word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
    return word.lower()


def load_results():
    """Load existing checkpoint, or return empty structure."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    """Save results checkpoint."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def run_model(nickname, model_id, tests, results, backfill=False):
    """Run one model on all tests, with checkpoint support."""
    if nickname not in results:
        results[nickname] = {
            "model_id": model_id,
            "responses": {},
        }

    model_data = results[nickname]
    responses = model_data["responses"]

    # Determine which tests need running
    to_run = []
    for i, test in enumerate(tests):
        key = str(i)
        if key in responses and responses[key].get("raw_response") is not None:
            if not backfill:
                continue  # Already have a good result
        elif key in responses and responses[key].get("raw_response") is None:
            pass  # Previous failure — always retry
        else:
            if backfill:
                continue  # In backfill mode, only retry failures
        to_run.append((i, key, test))

    if not to_run:
        n_ok = sum(1 for r in responses.values() if r.get("raw_response") is not None)
        n_fail = sum(1 for r in responses.values() if r.get("raw_response") is None)
        print(f"  {nickname}: nothing to do ({n_ok} ok, {n_fail} failed)")
        return

    print(f"  {nickname}: running {len(to_run)} tests" +
          (" (backfill)" if backfill else ""))

    done = 0
    for i, key, test in to_run:
        raw = call_model(model_id, test["prompt"])
        word = extract_first_word(raw)

        responses[key] = {
            "pair_id": test["pair_id"],
            "type": test["type"],
            "raw_response": raw,
            "extracted_word": word,
        }
        done += 1

        if done % 20 == 0 or done == len(to_run):
            save_results(results)
            n_ok = sum(1 for r in responses.values() if r.get("raw_response") is not None)
            print(f"    [{done}/{len(to_run)}] checkpoint ({n_ok}/{len(tests)} complete)")


def print_summary(results, tests):
    """Print quick summary of results completeness."""
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Complete':>8} {'Failed':>8} {'Missing':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

    for nickname, _ in MODELS:
        if nickname not in results:
            print(f"  {nickname:<12} {'—':>8} {'—':>8} {len(tests):>8}")
            continue
        responses = results[nickname]["responses"]
        n_ok = sum(1 for r in responses.values() if r.get("raw_response") is not None)
        n_fail = sum(1 for r in responses.values() if r.get("raw_response") is None)
        n_missing = len(tests) - len(responses)
        print(f"  {nickname:<12} {n_ok:>8} {n_fail:>8} {n_missing:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Run contrastive cloze benchmark via Together.ai")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Model nicknames: {[m[0] for m in MODELS]}")
    parser.add_argument("--backfill", action="store_true",
                        help="Retry only previously failed tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config without calling APIs")
    args = parser.parse_args()

    if not TOGETHER_API_KEY:
        print("ERROR: TOGETHER_API_KEY not found in environment or .env.local")
        sys.exit(1)

    # Load tests
    with open(TESTS_FILE) as f:
        tests = json.load(f)
    print(f"Loaded {len(tests)} tests from {TESTS_FILE.relative_to(BASE)}")

    # Select models
    if args.models:
        selected = [(n, m) for n, m in MODELS if n in args.models]
        if not selected:
            print(f"ERROR: No matching models. Available: {[m[0] for m in MODELS]}")
            sys.exit(1)
    else:
        selected = MODELS

    print(f"Models: {[n for n, _ in selected]}")

    if args.dry_run:
        print("  (dry-run: no API calls)")
        results = load_results()
        print_summary(results, tests)
        return

    # Load checkpoint
    results = load_results()

    # Run each model
    for nickname, model_id in selected:
        run_model(nickname, model_id, tests, results, backfill=args.backfill)

    save_results(results)
    print_summary(results, tests)
    print(f"\nResults saved to {RESULTS_FILE.relative_to(BASE)}")


if __name__ == "__main__":
    main()
