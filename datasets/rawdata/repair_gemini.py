#!/usr/bin/env python3
"""Re-run Gemini analysis for jokes that failed in the initial run."""

import json
import os
import time
from pathlib import Path

# Load environment variables from .env.local
env_path = Path(__file__).parent / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from google import genai

SYSTEM_PROMPT = """You are a comedy critic and linguistics expert analyzing pun jokes.

Each joke is a fill-in-the-blank sentence with:
- "straight" answers: reasonable non-punny completions
- "punny" answers: words/phrases with a double meaning that create the humor

For each joke, provide:
1. "explanation": A 1-2 sentence explanation of what makes the pun work (the double meaning). If the joke doesn't land well, critique what's off about it.
2. "rating": An integer 0-10 rating of how funny/clever the joke is.

Rating guidelines:
- 8-10: Exceptional — surprising pun, tight double meaning, natural setup
- 6-7: Good — solid pun with clear double meaning, decent setup
- 4-5: Average — pun works but is predictable or setup is forced
- 2-3: Weak — double meaning is a stretch, or setup doesn't flow
- 0-1: Bad — pun barely works or joke doesn't make sense

Be a tough but fair critic. Most jokes should be 4-7. Only truly brilliant ones get 8+.

IMPORTANT: Respond with ONLY a valid JSON array. Do not use any special characters,
curly quotes, or em-dashes in your text. Use only straight quotes and regular hyphens.
No markdown, no code fences, just the raw JSON array."""

MAX_RETRIES = 5
RETRY_DELAY = 5
BATCH_SIZE = 10  # smaller batches to reduce JSON parse failures


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def analyze_batch(client, jokes, batch_num, total):
    batch_text = json.dumps(jokes, indent=2)
    prompt = f"Analyze these pun jokes and rate each one:\n\n{batch_text}"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.3,
                },
            )
            results = parse_json_response(response.text)
            print(f"  [Gemini] Batch {batch_num}/{total}: analyzed {len(results)} jokes",
                  flush=True)
            return results
        except Exception as e:
            print(f"  [Gemini] Batch {batch_num} attempt {attempt+1} failed: {e}",
                  flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def main():
    # Load annotated data
    ann_path = Path(__file__).parent / "annotated_samples.json"
    with open(ann_path) as f:
        annotated = json.load(f)

    # Find failed jokes
    failed = [j for j in annotated if j.get("gemini_rating", -1) == -1]
    print(f"Found {len(failed)} jokes with failed Gemini ratings", flush=True)

    if not failed:
        print("Nothing to repair!", flush=True)
        return

    # Prepare jokes for re-analysis (strip provider annotations)
    jokes_to_analyze = []
    for j in failed:
        jokes_to_analyze.append({
            "id": j["id"],
            "sentence": j["sentence"],
            "straight": j["straight"],
            "punny": j["punny"],
        })

    # Run in smaller batches
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    batches = [jokes_to_analyze[i:i+BATCH_SIZE]
               for i in range(0, len(jokes_to_analyze), BATCH_SIZE)]
    total = len(batches)

    results = {}
    for i, batch in enumerate(batches, 1):
        batch_results = analyze_batch(client, batch, i, total)
        if batch_results:
            for r in batch_results:
                results[r["id"]] = {
                    "explanation": r["explanation"],
                    "rating": r["rating"],
                }
        else:
            print(f"  WARNING: Batch {i} still failed!", flush=True)
        if i < total:
            time.sleep(1)

    # Merge back into annotated data
    repaired = 0
    still_failed = 0
    for j in annotated:
        if j["id"] in results:
            j["gemini_explanation"] = results[j["id"]]["explanation"]
            j["gemini_rating"] = results[j["id"]]["rating"]
            repaired += 1
        elif j.get("gemini_rating", -1) == -1:
            still_failed += 1

    # Save
    with open(ann_path, "w") as f:
        json.dump(annotated, f, indent=2)

    print(f"\nRepaired {repaired} jokes, {still_failed} still failed", flush=True)
    print(f"Updated {ann_path.name}", flush=True)


if __name__ == "__main__":
    main()
