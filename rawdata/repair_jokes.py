#!/usr/bin/env python3
"""
Repair pun jokes that scored below threshold using Claude Sonnet.

Reads annotated_samples.json, classifies each joke as "already funny"
(avg of gemini + claudecode raw ratings >= 7.0) or "needs repair",
sends below-threshold jokes to Claude for creative repair, and writes
all 502 jokes to repaired_jokes.json.

Usage:
    python3 repair_jokes.py                   # Full run
    python3 repair_jokes.py --dry-run         # Show split without API calls
    python3 repair_jokes.py --limit 5         # Repair only first 5 jokes
    python3 repair_jokes.py --batch-size 3    # Smaller batches
    python3 repair_jokes.py --threshold 6.0   # Different threshold
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

# Load environment variables from .env.local
env_path = Path(__file__).parent / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

DEFAULT_BATCH_SIZE = 5
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
DEFAULT_THRESHOLD = 7.0

REPAIR_SYSTEM_PROMPT = """You are an expert comedy writer specializing in pun jokes. Your job is to REPAIR weak pun jokes so they actually land.

## How pun jokes work
Each joke is a fill-in-the-blank sentence. The "punny" answer is a word with a double meaning:
- Meaning A fits the literal/professional context of the setup
- Meaning B fits as a natural sentence completion

A great pun makes BOTH meanings work simultaneously and naturally.

## Example of a good repair

ORIGINAL: "The king was only 12 inches tall, making him a terrible ___" (punny: ruler)
PROBLEM: A 12-inch ruler is actually standard/good size, so "terrible" contradicts the measurement meaning.
REPAIRED: "The king was only 12 inches tall, making him a pretty good ___" (punny: ruler)
WHY: Now both meanings work — he's a "pretty good" ruler (12 inches = a standard ruler), and being only 12 inches tall makes "pretty good" ironic for a king.

## Common failure types to fix
1. **Contradictory setup**: The setup says one thing but the pun implies the opposite (like "terrible" for a 12-inch ruler)
2. **No real double meaning**: The "punny" word only works in one sense given the setup
3. **Forced phrasing**: The sentence is awkwardly constructed to shoehorn in the pun
4. **Pun in setup not blank**: The wordplay happens in the sentence body, not in the blank
5. **Too obvious/predictable**: The setup telegraphs the pun with no surprise

## Your repair guidelines
- Keep the same core pun word when possible
- Change the setup/framing to make BOTH meanings work simultaneously
- The sentence should read naturally — someone should be able to say it in conversation
- The straight answers should also change if the new sentence needs different non-punny completions
- Be creative! Sometimes a small tweak fixes everything; sometimes you need to reimagine the setup entirely

## Response format
Respond with ONLY a JSON array of objects. Each object must have:
- "id": the joke's id number
- "repaired_sentence": the improved fill-in-the-blank sentence
- "repaired_straight": array of 2-3 non-punny completions for the new sentence
- "repaired_punny": array of punny completions (usually keep original pun word)
- "repair_explanation": 1-2 sentences explaining what was wrong and how you fixed it

No markdown, no code fences, just the raw JSON array."""


def make_repair_prompt(jokes):
    """Create the user prompt for a batch of jokes needing repair."""
    batch = []
    for j in jokes:
        entry = {
            "id": j["id"],
            "sentence": j["sentence"],
            "straight": j["straight"],
            "punny": j["punny"],
        }
        # Include all raters' feedback to inform the repair
        for rater in ["gemini", "openai", "anthropic", "claudecode"]:
            exp_key = f"{rater}_explanation"
            rat_key = f"{rater}_rating"
            if exp_key in j:
                entry[f"{rater}_explanation"] = j[exp_key]
            if rat_key in j:
                entry[f"{rater}_rating"] = j[rat_key]
        batch.append(entry)
    return f"Repair these weak pun jokes. Use the raters' explanations to understand what's wrong with each one:\n\n{json.dumps(batch, indent=2)}"


def parse_json_response(text):
    """Extract JSON array from response text, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def repair_with_anthropic(jokes, batch_num, total_batches):
    """Send a batch of jokes to Claude Sonnet for repair."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = make_repair_prompt(jokes)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=REPAIR_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = response.content[0].text
            results = parse_json_response(text)
            print(f"  Batch {batch_num}/{total_batches}: "
                  f"repaired {len(results)} jokes", flush=True)
            return results
        except Exception as e:
            print(f"  Batch {batch_num} attempt {attempt+1} failed: {e}",
                  flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def classify_jokes(jokes, threshold):
    """Split jokes into already-funny and needs-repair lists."""
    funny = []
    needs_repair = []
    for j in jokes:
        gr = j.get("gemini_rating", 0)
        cr = j.get("claudecode_rating", 0)
        avg = (gr + cr) / 2.0
        if avg >= threshold:
            funny.append(j)
        else:
            needs_repair.append(j)
    return funny, needs_repair


def load_checkpoint(output_path):
    """Load existing output file for resume support."""
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        # Build set of IDs that already have repairs
        repaired_ids = set()
        for j in data:
            if "repaired_sentence" in j:
                repaired_ids.add(j["id"])
        return data, repaired_ids
    return None, set()


def save_checkpoint(output_path, all_jokes):
    """Save current state for resume support."""
    with open(output_path, "w") as f:
        json.dump(all_jokes, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Repair weak pun jokes using Claude Sonnet")
    parser.add_argument("--input", default="annotated_samples.json",
                        help="Input JSON file (default: annotated_samples.json)")
    parser.add_argument("--output", default="repaired_jokes.json",
                        help="Output JSON file (default: repaired_jokes.json)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Jokes per API call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Avg rating threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show classification split without making API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only repair first N jokes (for testing)")
    args = parser.parse_args()

    # Load jokes
    input_path = Path(__file__).parent / args.input
    with open(input_path) as f:
        jokes = json.load(f)
    print(f"Loaded {len(jokes)} jokes from {input_path.name}", flush=True)

    # Classify
    funny, needs_repair = classify_jokes(jokes, args.threshold)
    print(f"Already funny (avg >= {args.threshold}): {len(funny)}", flush=True)
    print(f"Needs repair  (avg <  {args.threshold}): {len(needs_repair)}", flush=True)

    if args.dry_run:
        print("\n--dry-run: no API calls made.", flush=True)
        # Show some stats about the repair candidates
        if needs_repair:
            avgs = []
            for j in needs_repair:
                gr = j.get("gemini_rating", 0)
                cr = j.get("claudecode_rating", 0)
                avgs.append((gr + cr) / 2.0)
            avgs.sort()
            print(f"  Repair candidates avg rating range: "
                  f"{avgs[0]:.1f} - {avgs[-1]:.1f}", flush=True)
            print(f"  Median avg rating: {avgs[len(avgs)//2]:.1f}", flush=True)
            print(f"  Batches needed: "
                  f"{(len(needs_repair) + args.batch_size - 1) // args.batch_size}",
                  flush=True)
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set!", flush=True)
        sys.exit(1)

    # Apply limit
    repair_list = needs_repair
    if args.limit is not None:
        repair_list = needs_repair[:args.limit]
        print(f"Limiting to first {len(repair_list)} repair candidates", flush=True)

    # Check for checkpoint / resume
    output_path = Path(__file__).parent / args.output
    checkpoint_data, already_repaired_ids = load_checkpoint(output_path)

    # Build the output list: start with all jokes (original data)
    # We'll use a dict keyed by id for easy updating
    output_by_id = {}
    if checkpoint_data:
        for j in checkpoint_data:
            output_by_id[j["id"]] = j
        print(f"Resumed from checkpoint: {len(already_repaired_ids)} already repaired",
              flush=True)
    else:
        for j in jokes:
            output_by_id[j["id"]] = dict(j)

    # Filter out already-repaired jokes from the repair list
    repair_list = [j for j in repair_list if j["id"] not in already_repaired_ids]

    if not repair_list:
        print("All jokes already repaired! Nothing to do.", flush=True)
        # Still save final output in case checkpoint had partial data
        ordered = [output_by_id[j["id"]] for j in jokes if j["id"] in output_by_id]
        save_checkpoint(output_path, ordered)
        print(f"Wrote {len(ordered)} jokes to {output_path.name}", flush=True)
        return

    # Batch and repair
    batches = [repair_list[i:i+args.batch_size]
               for i in range(0, len(repair_list), args.batch_size)]
    total_batches = len(batches)

    print(f"\n{'='*60}", flush=True)
    print(f"  Repairing {len(repair_list)} jokes in {total_batches} batches "
          f"(batch size: {args.batch_size})", flush=True)
    print(f"{'='*60}", flush=True)

    repaired_count = 0
    failed_count = 0

    for i, batch in enumerate(batches, 1):
        results = repair_with_anthropic(batch, i, total_batches)
        if results:
            # Index results by id
            results_by_id = {r["id"]: r for r in results}
            for joke in batch:
                jid = joke["id"]
                if jid in results_by_id:
                    r = results_by_id[jid]
                    # Normalize field names — model sometimes drops "repaired_" prefix
                    field_map = {
                        "repaired_sentence": ["repaired_sentence", "sentence"],
                        "repaired_straight": ["repaired_straight", "straight"],
                        "repaired_punny": ["repaired_punny", "punny"],
                        "repair_explanation": ["repair_explanation", "explanation"],
                    }
                    normalized = {}
                    for canonical, aliases in field_map.items():
                        for alias in aliases:
                            if alias in r:
                                normalized[canonical] = r[alias]
                                break
                    missing = [k for k in field_map if k not in normalized]
                    if missing:
                        print(f"  WARNING: Joke {jid} missing fields {missing}, skipping",
                              flush=True)
                        failed_count += 1
                        continue
                    for k, v in normalized.items():
                        output_by_id[jid][k] = v
                    repaired_count += 1
                else:
                    print(f"  WARNING: No repair returned for joke {jid}", flush=True)
                    failed_count += 1
        else:
            print(f"  WARNING: Batch {i} failed after all retries!", flush=True)
            failed_count += len(batch)

        # Save checkpoint after each batch
        ordered = [output_by_id[j["id"]] for j in jokes if j["id"] in output_by_id]
        save_checkpoint(output_path, ordered)

        # Rate limiting pause between batches
        if i < total_batches:
            time.sleep(1)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total jokes:     {len(jokes)}", flush=True)
    print(f"  Already funny:   {len(funny)}", flush=True)
    print(f"  Repaired:        {repaired_count}", flush=True)
    if failed_count:
        print(f"  Failed:          {failed_count}", flush=True)
    print(f"  Output file:     {output_path.name}", flush=True)


if __name__ == "__main__":
    main()
