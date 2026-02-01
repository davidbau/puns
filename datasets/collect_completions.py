#!/usr/bin/env python3
"""
Collect diverse cloze completions for all 205 puns from multiple models.

For each joke, sends the truncated sentence (before ___) to 5 models
via Together.ai with n=20 samples, collecting candidate completions
and their frequencies. Then classifies each candidate as straight,
funny, or other using Claude, and updates puns_205.json.

Usage:
    python3 collect_completions.py                  # Full run
    python3 collect_completions.py --limit 5        # Test with 5 jokes
    python3 collect_completions.py --phase collect   # Only collect (skip classify)
    python3 collect_completions.py --phase classify  # Only classify (skip collect)
    python3 collect_completions.py --phase update    # Only update puns_205.json
"""

import json
import os
import sys
import time
import re
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(__file__).parent

# Load env
env_path = BASE.parent / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODELS = [
    ("3b", "meta-llama/Llama-3.2-3B-Instruct-Turbo"),
    ("8b", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("70b", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("3.3-70b", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("405b", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
]

SYSTEM_PROMPT = (
    "You are completing a text. Output only the single next word "
    "that best continues the text. Output ONLY that one word, "
    "nothing else — no punctuation, no explanation."
)

SAMPLES_FILE = BASE / "raw_completions.json"
CLASSIFIED_FILE = BASE / "classified_completions.json"

import requests as req_lib


def truncate_before_blank(sentence):
    idx = sentence.index("___")
    return sentence[:idx].rstrip()


def extract_word(text):
    """Extract first word, lowercase, stripped of punctuation."""
    if not text:
        return ""
    text = text.strip()
    word = text.split()[0] if text.split() else ""
    word = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
    return word.lower()


MAX_RETRIES = 3
RETRY_DELAY = 5


def _call_together(payload, timeout=60):
    """Make a Together API call with retries on 503/timeout."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = req_lib.post(
                "https://api.together.xyz/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {TOGETHER_KEY}"},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except (req_lib.exceptions.HTTPError, req_lib.exceptions.Timeout,
                req_lib.exceptions.ConnectionError) as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if attempt < MAX_RETRIES - 1 and (status in (503, 429) or status is None):
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            raise
    return None


def collect_one(model_id, prompt, n=20, temperature=0.8):
    """Collect n completions from one model for one prompt."""
    try:
        data = _call_together({
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 5,
            "temperature": temperature,
            "n": n,
            "stop": ["\n"],
        })
        words = []
        for choice in data["choices"]:
            w = extract_word(choice["message"]["content"])
            if w:
                words.append(w)
        return words
    except Exception as e:
        print(f"    ERROR ({model_id}): {e}")
        return []


def collect_greedy(model_id, prompt):
    """Collect greedy (temp=0) completion."""
    try:
        data = _call_together({
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 5,
            "temperature": 0,
            "stop": ["\n"],
        }, timeout=30)
        w = extract_word(data["choices"][0]["message"]["content"])
        return w
    except Exception as e:
        print(f"    ERROR greedy ({model_id}): {e}")
        return ""


def phase_collect(jokes, limit=None):
    """Phase 1: Collect completions from all models for all jokes."""
    if limit:
        jokes = jokes[:limit]

    total = len(jokes) * len(MODELS)
    print(f"Collecting completions: {len(jokes)} jokes × {len(MODELS)} models = {total} calls")
    print(f"  Plus {total} greedy calls = {total*2} total API calls")

    results = {}

    for ji, joke in enumerate(jokes):
        idx = joke["index"]
        prompt = truncate_before_blank(joke["sentence"])
        joke_results = {
            "index": idx,
            "sentence": joke["sentence"],
            "prompt": prompt,
            "existing_straight": joke["straight"],
            "existing_punny": joke["punny"],
            "models": {},
        }

        for nick, model_id in MODELS:
            # Diverse samples
            samples = collect_one(model_id, prompt, n=20, temperature=0.8)
            # Greedy
            greedy = collect_greedy(model_id, prompt)

            joke_results["models"][nick] = {
                "greedy": greedy,
                "samples": samples,
            }

        # Aggregate across all models
        all_words = Counter()
        greedy_words = Counter()
        for nick in joke_results["models"]:
            mdata = joke_results["models"][nick]
            greedy_words[mdata["greedy"]] += 1
            for w in mdata["samples"]:
                all_words[w] += 1

        joke_results["aggregated"] = {
            "word_counts": dict(all_words.most_common()),
            "greedy_counts": dict(greedy_words.most_common()),
            "total_samples": sum(all_words.values()),
        }

        results[str(idx)] = joke_results

        if (ji + 1) % 10 == 0 or ji == len(jokes) - 1:
            n_unique = len(all_words)
            top3 = ", ".join(f"{w}({c})" for w, c in all_words.most_common(3))
            print(f"  [{ji+1}/{len(jokes)}] joke {idx}: {n_unique} unique words. Top: {top3}")

            # Checkpoint save
            with open(SAMPLES_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved raw completions for {len(results)} jokes to {SAMPLES_FILE.name}")
    return results


def phase_backfill(jokes):
    """Re-collect for any jokes/models that had errors (empty samples)."""
    with open(SAMPLES_FILE) as f:
        results = json.load(f)

    jokes_by_index = {j["index"]: j for j in jokes}
    expected_sample_count = 15  # flag if fewer than this (out of 20)
    gaps = []

    for idx_str, joke_data in results.items():
        for nick, model_id in MODELS:
            mdata = joke_data["models"].get(nick, {})
            n_samples = len(mdata.get("samples", []))
            has_greedy = bool(mdata.get("greedy", ""))
            if n_samples < expected_sample_count or not has_greedy:
                gaps.append((int(idx_str), nick, model_id, n_samples, has_greedy))

    if not gaps:
        print("No gaps found — all jokes have sufficient data.")
        return results

    print(f"Found {len(gaps)} gaps to backfill:")
    for idx, nick, mid, n, g in gaps[:10]:
        print(f"  joke {idx} / {nick}: {n} samples, greedy={'yes' if g else 'NO'}")
    if len(gaps) > 10:
        print(f"  ... and {len(gaps)-10} more")

    filled = 0
    for idx, nick, model_id, n_samples, has_greedy in gaps:
        joke_data = results[str(idx)]
        prompt = joke_data["prompt"]

        if n_samples < expected_sample_count:
            new_samples = collect_one(model_id, prompt, n=20, temperature=0.8)
            if new_samples:
                joke_data["models"][nick]["samples"] = new_samples
                filled += 1

        if not has_greedy:
            new_greedy = collect_greedy(model_id, prompt)
            if new_greedy:
                joke_data["models"][nick]["greedy"] = new_greedy
                filled += 1

        # Re-aggregate
        all_words = Counter()
        greedy_words = Counter()
        for mn in joke_data["models"]:
            md = joke_data["models"][mn]
            if md.get("greedy"):
                greedy_words[md["greedy"]] += 1
            for w in md.get("samples", []):
                all_words[w] += 1
        joke_data["aggregated"] = {
            "word_counts": dict(all_words.most_common()),
            "greedy_counts": dict(greedy_words.most_common()),
            "total_samples": sum(all_words.values()),
        }

    with open(SAMPLES_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Backfilled {filled} gaps. Saved to {SAMPLES_FILE.name}")
    return results


def phase_classify(results):
    """Phase 2: Classify all candidate words using Claude."""
    # Build batches for classification
    # For each joke, gather all unique candidate words that aren't already classified
    to_classify = []
    for idx_str, joke_data in results.items():
        existing_straight = set(w.lower() for w in joke_data["existing_straight"])
        existing_punny = set(w.lower() for w in joke_data["existing_punny"])
        candidates = list(joke_data["aggregated"]["word_counts"].keys())

        unknown = []
        for w in candidates:
            if w in existing_straight or w in existing_punny:
                continue
            unknown.append(w)

        if unknown:
            to_classify.append({
                "index": joke_data["index"],
                "sentence": joke_data["sentence"],
                "existing_straight": joke_data["existing_straight"],
                "existing_punny": joke_data["existing_punny"],
                "candidates": unknown,
            })

    print(f"Need to classify candidates for {len(to_classify)} jokes")
    total_candidates = sum(len(j["candidates"]) for j in to_classify)
    print(f"  Total candidates to classify: {total_candidates}")

    # Batch classify using Claude (batches of 10 jokes)
    batch_size = 10
    all_classifications = {}

    for batch_start in range(0, len(to_classify), batch_size):
        batch = to_classify[batch_start:batch_start + batch_size]

        prompt_parts = []
        for item in batch:
            prompt_parts.append(
                f"Joke index {item['index']}:\n"
                f"  Sentence: {item['sentence']}\n"
                f"  Known straight completions: {item['existing_straight']}\n"
                f"  Known punny completions: {item['existing_punny']}\n"
                f"  Candidates to classify: {item['candidates']}\n"
            )

        classify_prompt = (
            "For each joke below, classify each candidate word as:\n"
            "- 'straight': a sensible literal/non-punny completion for the blank (___)\n"
            "- 'funny': exploits a double meaning, wordplay, or pun in the context\n"
            "- 'other': doesn't fit the blank well, is a filler/function word, or is nonsensical\n\n"
            "A word is 'funny' if it works BOTH literally in the sentence AND has a second "
            "meaning that connects to the subject. A word is 'straight' if it's a reasonable "
            "literal completion without wordplay.\n\n"
            + "\n".join(prompt_parts) +
            "\nRespond with JSON only. Format: {\"<index>\": {\"<word>\": \"straight|funny|other\", ...}, ...}\n"
            "No markdown fences, just the raw JSON."
        )

        for attempt in range(3):
            try:
                resp = req_lib.post(
                    "https://api.anthropic.com/v1/messages",
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": classify_prompt}],
                    },
                    headers={
                        "x-api-key": ANTHROPIC_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                text = resp.json()["content"][0]["text"].strip()
                # Clean up potential markdown fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    text = text.rsplit("```", 1)[0]
                classifications = json.loads(text)
                all_classifications.update(classifications)
                break
            except Exception as e:
                print(f"    Classify batch attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)

        batch_end = min(batch_start + batch_size, len(to_classify))
        print(f"  Classified batch {batch_start//batch_size + 1} "
              f"(jokes {batch_start+1}-{batch_end}/{len(to_classify)})")

    # Merge classifications back into results
    for idx_str, joke_data in results.items():
        idx = str(joke_data["index"])
        existing_straight = set(w.lower() for w in joke_data["existing_straight"])
        existing_punny = set(w.lower() for w in joke_data["existing_punny"])
        word_counts = joke_data["aggregated"]["word_counts"]

        classified = {}
        for w in word_counts:
            if w in existing_straight:
                classified[w] = "straight"
            elif w in existing_punny:
                classified[w] = "funny"
            elif idx in all_classifications and w in all_classifications[idx]:
                classified[w] = all_classifications[idx][w]
            else:
                classified[w] = "other"

        joke_data["classifications"] = classified

    # Save
    with open(CLASSIFIED_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classified completions to {CLASSIFIED_FILE.name}")

    # Summary stats
    total_s = total_f = total_o = 0
    for joke_data in results.values():
        for w, cls in joke_data.get("classifications", {}).items():
            if cls == "straight": total_s += 1
            elif cls == "funny": total_f += 1
            else: total_o += 1
    print(f"  Total classified: {total_s} straight, {total_f} funny, {total_o} other")

    return results


def phase_update(results):
    """Phase 3: Update puns_205.json with expanded completion lists."""
    with open(BASE / "puns_205.json") as f:
        puns = json.load(f)

    # Build lookup by index
    puns_by_index = {p["index"]: p for p in puns}

    for idx_str, joke_data in results.items():
        idx = joke_data["index"]
        if idx not in puns_by_index:
            continue

        pun = puns_by_index[idx]
        classifications = joke_data.get("classifications", {})
        word_counts = joke_data["aggregated"]["word_counts"]

        # Build ordered lists: straight and funny words by frequency
        straight_words = []
        funny_words = []
        for w, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            cls = classifications.get(w, "other")
            if cls == "straight":
                straight_words.append(w)
            elif cls == "funny":
                funny_words.append(w)

        # Also include original words that may not have appeared in samples
        for w in pun["straight"]:
            if w.lower() not in [x.lower() for x in straight_words]:
                straight_words.append(w.lower())
        for w in pun["punny"]:
            if w.lower() not in [x.lower() for x in funny_words]:
                funny_words.append(w.lower())

        pun["straight"] = straight_words
        pun["punny"] = funny_words

    # Save
    with open(BASE / "puns_205.json", "w") as f:
        json.dump(puns, f, indent=2)
    print(f"Updated puns_205.json with expanded completion lists")

    # Stats
    total_s = sum(len(p["straight"]) for p in puns)
    total_f = sum(len(p["punny"]) for p in puns)
    print(f"  Avg straight per joke: {total_s/len(puns):.1f}")
    print(f"  Avg funny per joke:    {total_f/len(puns):.1f}")

    return puns


def verify_against_cloze(puns):
    """Check if expanded lists capture all contrastive test labels."""
    # Load the contrastive test classifications (from classify_other_completions.py)
    with open(BASE / "contextual_cloze_tests_100.json") as f:
        tests = json.load(f)

    puns_by_index = {p["index"]: p for p in puns}

    missing_straight = []
    missing_funny = []

    for test in tests:
        c_idx = test["joke_c_index"]
        if c_idx not in puns_by_index:
            continue
        pun = puns_by_index[c_idx]

        pun_straight_lower = [w.lower() for w in pun["straight"]]
        pun_funny_lower = [w.lower() for w in pun["punny"]]

        # Check expected completions from the cloze test
        for exp in test["expected_completion"]:
            exp_lower = exp.lower()
            first_word = exp_lower.split()[0]
            if test["type"] == "straight":
                if exp_lower not in pun_straight_lower and first_word not in pun_straight_lower:
                    missing_straight.append((c_idx, exp, pun["sentence"]))
            else:
                if exp_lower not in pun_funny_lower and first_word not in pun_funny_lower:
                    missing_funny.append((c_idx, exp, pun["sentence"]))

        # Also check contrast completions
        for exp in test["contrast_completion"]:
            exp_lower = exp.lower()
            first_word = exp_lower.split()[0]
            if test["type"] == "straight":
                # contrast = funny words
                if exp_lower not in pun_funny_lower and first_word not in pun_funny_lower:
                    missing_funny.append((c_idx, exp, pun["sentence"]))
            else:
                # contrast = straight words
                if exp_lower not in pun_straight_lower and first_word not in pun_straight_lower:
                    missing_straight.append((c_idx, exp, pun["sentence"]))

    # Deduplicate
    missing_straight = list(set(missing_straight))
    missing_funny = list(set(missing_funny))

    print(f"\n{'='*70}")
    print("VERIFICATION: Do expanded lists capture all contrastive test labels?")
    print(f"{'='*70}")

    if not missing_straight and not missing_funny:
        print("  YES — all contrastive test labels are covered!")
    else:
        if missing_straight:
            print(f"\n  Missing from straight lists ({len(missing_straight)}):")
            for idx, word, sent in sorted(missing_straight):
                print(f"    joke {idx}: '{word}' — {sent[:60]}...")
        if missing_funny:
            print(f"\n  Missing from funny lists ({len(missing_funny)}):")
            for idx, word, sent in sorted(missing_funny):
                print(f"    joke {idx}: '{word}' — {sent[:60]}...")

    return len(missing_straight) == 0 and len(missing_funny) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--phase",
                        choices=["collect", "backfill", "classify", "update", "all"],
                        default="all")
    args = parser.parse_args()

    with open(BASE / "puns_205.json") as f:
        jokes = json.load(f)
    print(f"Loaded {len(jokes)} jokes from puns_205.json")

    if args.phase == "backfill":
        results = phase_backfill(jokes)
        return

    if args.phase in ("collect", "all"):
        results = phase_collect(jokes, limit=args.limit)
    else:
        with open(SAMPLES_FILE) as f:
            results = json.load(f)
        print(f"Loaded existing raw completions ({len(results)} jokes)")

    if args.phase in ("classify", "all"):
        results = phase_classify(results)
    elif args.phase == "update":
        with open(CLASSIFIED_FILE) as f:
            results = json.load(f)
        print(f"Loaded existing classifications ({len(results)} jokes)")

    if args.phase in ("update", "all"):
        puns = phase_update(results)
        verify_against_cloze(puns)


if __name__ == "__main__":
    main()
