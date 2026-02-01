#!/usr/bin/env python3
"""
Annotate puns_205.json with model completion data, classifications, and tier rankings.

Consolidates the full annotation pipeline into four phases:

  Phase 1: collect   — 205 jokes x 5 models x (20 samples + greedy) via Together.ai
  Phase 2: classify  — Claude classifies unique candidates as straight/funny/other
  Phase 3: rank      — compute balance, p_straight, p_funny, assign tiers
  Phase 4: update    — merge expanded lists + tier annotations into puns_205.json

Usage:
    python3 annotate_completions.py                       # Full pipeline
    python3 annotate_completions.py --phase collect       # Only collect
    python3 annotate_completions.py --phase classify      # Only classify
    python3 annotate_completions.py --phase rank          # Only rank
    python3 annotate_completions.py --phase update        # Only update puns_205.json
    python3 annotate_completions.py --limit 5             # Test with 5 jokes
"""

import json
import os
import re
import time
import argparse
from pathlib import Path
from collections import Counter

import requests as req_lib

BASE = Path(__file__).parent

# ── Load environment ─────────────────────────────────────────────────────────
env_path = BASE / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Constants ────────────────────────────────────────────────────────────────

MODELS = [
    ("3b",     "meta-llama/Llama-3.2-3B-Instruct-Turbo"),
    ("8b",     "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("70b",    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    ("3.3-70b","meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("405b",   "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
]

SYSTEM_PROMPT = (
    "You are completing a text. Output only the single next word "
    "that best continues the text. Output ONLY that one word, "
    "nothing else — no punctuation, no explanation."
)

SAMPLES_FILE    = BASE / "raw_completions.json"
CLASSIFIED_FILE = BASE / "classified_completions.json"
PUNS_FILE       = BASE / "puns_205.json"

MAX_RETRIES = 3
RETRY_DELAY = 5

# ── Tier thresholds ──────────────────────────────────────────────────────────

def assign_tier(balance, p_funny):
    """Assign a cloze tier based on balance and p_funny."""
    if balance >= 0.3 and p_funny >= 0.05:
        return "balanced"
    if balance > 0.05 and p_funny >= 0.05:
        return "leaning"
    if p_funny > 0.5 and balance <= 0.05:
        return "funny_dominated"
    return "straight_dominated"


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Together.ai API ──────────────────────────────────────────────────────────

def _call_together(payload, timeout=60):
    """Make a Together API call with retries on 503/429/timeout."""
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


def collect_samples(model_id, prompt, n=20, temperature=0.8):
    """Collect n diverse completions from one model."""
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
        return [w for c in data["choices"]
                if (w := extract_word(c["message"]["content"]))]
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
        return extract_word(data["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"    ERROR greedy ({model_id}): {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Collect completions from all models
# ══════════════════════════════════════════════════════════════════════════════

def phase_collect(jokes, limit=None):
    """Collect 20 diverse samples + 1 greedy per model per joke."""
    if limit:
        jokes = jokes[:limit]

    n_calls = len(jokes) * len(MODELS) * 2
    print(f"Phase 1: COLLECT")
    print(f"  {len(jokes)} jokes x {len(MODELS)} models x (20 samples + greedy) = {n_calls} API calls")

    results = {}

    for ji, joke in enumerate(jokes):
        idx = joke["index"]
        prompt = truncate_before_blank(joke["sentence"])
        entry = {
            "index": idx,
            "sentence": joke["sentence"],
            "prompt": prompt,
            "existing_straight": joke["straight"],
            "existing_punny": joke["punny"],
            "models": {},
        }

        for nick, model_id in MODELS:
            entry["models"][nick] = {
                "greedy": collect_greedy(model_id, prompt),
                "samples": collect_samples(model_id, prompt, n=20, temperature=0.8),
            }

        # Aggregate across all models
        all_words = Counter()
        greedy_words = Counter()
        for mdata in entry["models"].values():
            if mdata["greedy"]:
                greedy_words[mdata["greedy"]] += 1
            for w in mdata["samples"]:
                all_words[w] += 1

        entry["aggregated"] = {
            "word_counts": dict(all_words.most_common()),
            "greedy_counts": dict(greedy_words.most_common()),
            "total_samples": sum(all_words.values()),
        }

        results[str(idx)] = entry

        if (ji + 1) % 10 == 0 or ji == len(jokes) - 1:
            top3 = ", ".join(f"{w}({c})" for w, c in all_words.most_common(3))
            print(f"  [{ji+1}/{len(jokes)}] joke {idx}: {len(all_words)} unique — {top3}")
            with open(SAMPLES_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"  Saved raw completions for {len(results)} jokes → {SAMPLES_FILE.name}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Classify candidates using Claude
# ══════════════════════════════════════════════════════════════════════════════

def phase_classify(results):
    """Classify each candidate word as straight/funny/other via Claude."""
    print(f"\nPhase 2: CLASSIFY")

    # Identify unknown candidates
    to_classify = []
    for joke_data in results.values():
        known_straight = set(w.lower() for w in joke_data["existing_straight"])
        known_punny = set(w.lower() for w in joke_data["existing_punny"])
        unknown = [w for w in joke_data["aggregated"]["word_counts"]
                   if w not in known_straight and w not in known_punny]
        if unknown:
            to_classify.append({
                "index": joke_data["index"],
                "sentence": joke_data["sentence"],
                "existing_straight": joke_data["existing_straight"],
                "existing_punny": joke_data["existing_punny"],
                "candidates": unknown,
            })

    total_cands = sum(len(j["candidates"]) for j in to_classify)
    print(f"  {len(to_classify)} jokes with unknown candidates ({total_cands} words total)")

    # Batch classify (10 jokes per Claude call)
    batch_size = 10
    all_classifications = {}

    for batch_start in range(0, len(to_classify), batch_size):
        batch = to_classify[batch_start:batch_start + batch_size]

        prompt_parts = []
        for item in batch:
            prompt_parts.append(
                f"Joke index {item['index']}:\n"
                f"  Sentence: {item['sentence']}\n"
                f"  Known straight: {item['existing_straight']}\n"
                f"  Known punny: {item['existing_punny']}\n"
                f"  Candidates: {item['candidates']}\n"
            )

        classify_prompt = (
            "For each joke below, classify each candidate word as:\n"
            "- 'straight': a sensible literal/non-punny completion for the blank (___)\n"
            "- 'funny': exploits a double meaning, wordplay, or pun in the context\n"
            "- 'other': doesn't fit the blank well, is a filler/function word, or is nonsensical\n\n"
            "A word is 'funny' if it works BOTH literally in the sentence AND has a second "
            "meaning that connects to the subject. A word is 'straight' if it's a reasonable "
            "literal completion without wordplay.\n\n"
            + "\n".join(prompt_parts)
            + "\nRespond with JSON only. Format: "
            "{\"<index>\": {\"<word>\": \"straight|funny|other\", ...}, ...}\n"
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
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    text = text.rsplit("```", 1)[0]
                all_classifications.update(json.loads(text))
                break
            except Exception as e:
                print(f"    Batch attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)

        batch_end = min(batch_start + batch_size, len(to_classify))
        print(f"  Classified batch {batch_start // batch_size + 1} "
              f"(jokes {batch_start + 1}-{batch_end}/{len(to_classify)})")

    # Merge classifications into results
    for joke_data in results.values():
        idx = str(joke_data["index"])
        known_straight = set(w.lower() for w in joke_data["existing_straight"])
        known_punny = set(w.lower() for w in joke_data["existing_punny"])

        classified = {}
        for w in joke_data["aggregated"]["word_counts"]:
            if w in known_straight:
                classified[w] = "straight"
            elif w in known_punny:
                classified[w] = "funny"
            elif idx in all_classifications and w in all_classifications[idx]:
                classified[w] = all_classifications[idx][w]
            else:
                classified[w] = "other"
        joke_data["classifications"] = classified

    with open(CLASSIFIED_FILE, "w") as f:
        json.dump(results, f, indent=2)

    total_s = sum(1 for jd in results.values()
                  for c in jd.get("classifications", {}).values() if c == "straight")
    total_f = sum(1 for jd in results.values()
                  for c in jd.get("classifications", {}).values() if c == "funny")
    total_o = sum(1 for jd in results.values()
                  for c in jd.get("classifications", {}).values() if c == "other")
    print(f"  Saved → {CLASSIFIED_FILE.name}")
    print(f"  Totals: {total_s} straight, {total_f} funny, {total_o} other")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Rank — compute balance, p_straight, p_funny, assign tiers
# ══════════════════════════════════════════════════════════════════════════════

def phase_rank(results):
    """Compute discriminativeness metrics and assign tiers."""
    print(f"\nPhase 3: RANK")

    rankings = {}
    for joke_data in results.values():
        idx = joke_data["index"]
        word_counts = joke_data["aggregated"]["word_counts"]
        classifications = joke_data["classifications"]
        total = joke_data["aggregated"]["total_samples"]

        n_straight = n_funny = n_other = 0
        for word, count in word_counts.items():
            cls = classifications.get(word, "other")
            if cls == "straight":
                n_straight += count
            elif cls == "funny":
                n_funny += count
            else:
                n_other += count

        p_straight = n_straight / total if total else 0
        p_funny = n_funny / total if total else 0
        p_other = n_other / total if total else 0

        if n_straight > 0 and n_funny > 0:
            balance = min(p_straight, p_funny) / max(p_straight, p_funny)
        else:
            balance = 0.0

        # Per-model funny rate
        model_funny_rate = {}
        for model_name, mdata in joke_data["models"].items():
            samples = mdata.get("samples", [])
            mf = sum(1 for s in samples if classifications.get(s, "other") == "funny")
            model_funny_rate[model_name] = round(mf / len(samples), 4) if samples else 0

        tier = assign_tier(balance, p_funny)

        rankings[idx] = {
            "balance": round(balance, 4),
            "p_straight": round(p_straight, 4),
            "p_funny": round(p_funny, 4),
            "p_other": round(p_other, 4),
            "tier": tier,
            "model_funny_rate": model_funny_rate,
        }

    # Print tier summary
    from collections import Counter as C
    tier_counts = C(r["tier"] for r in rankings.values())
    print(f"  Tier distribution:")
    for tier in ["balanced", "leaning", "funny_dominated", "straight_dominated"]:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    return rankings


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Update puns_205.json with expanded lists + tier annotations
# ══════════════════════════════════════════════════════════════════════════════

def phase_update(results, rankings):
    """Merge expanded completion lists and tier annotations into puns_205.json."""
    print(f"\nPhase 4: UPDATE")

    with open(PUNS_FILE) as f:
        puns = json.load(f)

    puns_by_index = {p["index"]: p for p in puns}

    for joke_data in results.values():
        idx = joke_data["index"]
        if idx not in puns_by_index:
            continue

        pun = puns_by_index[idx]
        classifications = joke_data.get("classifications", {})
        word_counts = joke_data["aggregated"]["word_counts"]

        # Build ordered lists by frequency
        straight_words = []
        funny_words = []
        for w, _count in sorted(word_counts.items(), key=lambda x: -x[1]):
            cls = classifications.get(w, "other")
            if cls == "straight":
                straight_words.append(w)
            elif cls == "funny":
                funny_words.append(w)

        # Append original words not seen in samples
        for w in pun["straight"]:
            if w.lower() not in [x.lower() for x in straight_words]:
                straight_words.append(w.lower())
        for w in pun["punny"]:
            if w.lower() not in [x.lower() for x in funny_words]:
                funny_words.append(w.lower())

        pun["straight"] = straight_words
        pun["punny"] = funny_words

    # Merge tier annotations
    for pun in puns:
        idx = pun["index"]
        if idx in rankings:
            r = rankings[idx]
            pun["cloze_balance"] = r["balance"]
            pun["cloze_p_straight"] = r["p_straight"]
            pun["cloze_p_funny"] = r["p_funny"]
            pun["cloze_p_other"] = r["p_other"]
            pun["cloze_tier"] = r["tier"]
            pun["cloze_model_funny_rate"] = r["model_funny_rate"]

    with open(PUNS_FILE, "w") as f:
        json.dump(puns, f, indent=2)

    total_s = sum(len(p["straight"]) for p in puns)
    total_f = sum(len(p["punny"]) for p in puns)
    print(f"  Updated {PUNS_FILE.name}")
    print(f"  Avg straight/joke: {total_s / len(puns):.1f}")
    print(f"  Avg funny/joke:    {total_f / len(puns):.1f}")

    return puns


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Annotate puns_205.json with model completions, classifications, and tiers.")
    parser.add_argument("--phase",
                        choices=["collect", "classify", "rank", "update", "all"],
                        default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N jokes (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (unused currently, reserved)")
    args = parser.parse_args()

    with open(PUNS_FILE) as f:
        jokes = json.load(f)
    print(f"Loaded {len(jokes)} jokes from {PUNS_FILE.name}")

    # ── Phase 1: Collect ─────────────────────────────────────────────────
    if args.phase in ("collect", "all"):
        results = phase_collect(jokes, limit=args.limit)
    else:
        with open(SAMPLES_FILE) as f:
            results = json.load(f)
        print(f"Loaded raw completions ({len(results)} jokes) from {SAMPLES_FILE.name}")

    # ── Phase 2: Classify ────────────────────────────────────────────────
    if args.phase in ("classify", "all"):
        results = phase_classify(results)
    elif args.phase in ("rank", "update"):
        with open(CLASSIFIED_FILE) as f:
            results = json.load(f)
        print(f"Loaded classifications ({len(results)} jokes) from {CLASSIFIED_FILE.name}")

    # ── Phase 3: Rank ────────────────────────────────────────────────────
    if args.phase in ("rank", "update", "all"):
        rankings = phase_rank(results)
    else:
        rankings = None

    # ── Phase 4: Update ──────────────────────────────────────────────────
    if args.phase in ("update", "all"):
        phase_update(results, rankings)


if __name__ == "__main__":
    main()
