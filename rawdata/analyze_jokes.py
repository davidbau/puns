#!/usr/bin/env python3
"""
Analyze pun jokes using Gemini, OpenAI, and Anthropic APIs.

Reads raw_samples.json, sends batches to each model for analysis,
and produces annotated output files with explanations and ratings.

Usage:
    python3 analyze_jokes.py                     # Run all available providers
    python3 analyze_jokes.py --gemini-only       # Run only Gemini
    python3 analyze_jokes.py --openai-only       # Run only OpenAI
    python3 analyze_jokes.py --anthropic-only    # Run only Anthropic
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

BATCH_SIZE = 25  # jokes per API call
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

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

Respond with ONLY a JSON array of objects, each with "id", "explanation", and "rating" fields.
No markdown, no code fences, just the raw JSON array."""


def make_batch_prompt(jokes):
    """Create the user prompt for a batch of jokes."""
    # Only send core fields — strip any existing ratings/explanations
    # so the model forms its own independent judgment.
    core_fields = {"id", "sentence", "straight", "punny"}
    stripped = [{k: v for k, v in j.items() if k in core_fields} for j in jokes]
    batch_text = json.dumps(stripped, indent=2)
    return f"Analyze these pun jokes and rate each one:\n\n{batch_text}"


def parse_json_response(text):
    """Extract JSON array from response text, handling markdown fences."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix common issues: smart quotes, em-dashes, unescaped control chars
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
        text = text.replace("\u2014", "-").replace("\u2013", "-")  # em/en dashes
        # Fix unescaped newlines/tabs inside JSON strings
        import re
        text = re.sub(r'(?<=": ")(.*?)(?="[,\s*}])', lambda m: m.group(0).replace('\n', '\\n').replace('\t', '\\t'), text, flags=re.DOTALL)
        return json.loads(text)


def analyze_with_gemini(jokes, batch_num, total_batches):
    """Analyze a batch of jokes using Gemini API."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = make_batch_prompt(jokes)

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
            print(f"  [Gemini] Batch {batch_num}/{total_batches}: "
                  f"analyzed {len(results)} jokes", flush=True)
            return results
        except Exception as e:
            print(f"  [Gemini] Batch {batch_num} attempt {attempt+1} failed: {e}",
                  flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def analyze_with_openai(jokes, batch_num, total_batches):
    """Analyze a batch of jokes using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = make_batch_prompt(jokes)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = response.choices[0].message.content
            results = parse_json_response(text)
            print(f"  [OpenAI] Batch {batch_num}/{total_batches}: "
                  f"analyzed {len(results)} jokes", flush=True)
            return results
        except Exception as e:
            print(f"  [OpenAI] Batch {batch_num} attempt {attempt+1} failed: {e}",
                  flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def analyze_with_anthropic(jokes, batch_num, total_batches):
    """Analyze a batch of jokes using Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = make_batch_prompt(jokes)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            text = response.content[0].text
            results = parse_json_response(text)
            print(f"  [Anthropic] Batch {batch_num}/{total_batches}: "
                  f"analyzed {len(results)} jokes", flush=True)
            return results
        except Exception as e:
            print(f"  [Anthropic] Batch {batch_num} attempt {attempt+1} failed: {e}",
                  flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def run_analysis(jokes, analyze_fn, provider_name):
    """Run analysis on all jokes in batches using the given function."""
    all_results = {}
    batches = [jokes[i:i+BATCH_SIZE] for i in range(0, len(jokes), BATCH_SIZE)]
    total = len(batches)

    print(f"\n{'='*60}", flush=True)
    print(f"  {provider_name}: Analyzing {len(jokes)} jokes in {total} batches",
          flush=True)
    print(f"{'='*60}", flush=True)

    for i, batch in enumerate(batches, 1):
        results = analyze_fn(batch, i, total)
        if results:
            for r in results:
                all_results[r["id"]] = {
                    "explanation": r["explanation"],
                    "rating": r["rating"],
                }
        else:
            print(f"  WARNING: Batch {i} failed after all retries!", flush=True)
            for joke in batch:
                all_results[joke["id"]] = {
                    "explanation": "Analysis failed",
                    "rating": -1,
                }
        # Rate limiting pause between batches
        if i < total:
            time.sleep(1)

    return all_results


def merge_results(jokes, results_by_provider):
    """Merge original jokes with analysis results from each provider."""
    annotated = []
    for joke in jokes:
        entry = dict(joke)  # copy original fields
        for provider, results in results_by_provider.items():
            if joke["id"] in results:
                r = results[joke["id"]]
                entry[f"{provider}_explanation"] = r["explanation"]
                entry[f"{provider}_rating"] = r["rating"]
        annotated.append(entry)
    return annotated


def print_summary(annotated, providers):
    """Print rating distribution summary."""
    from collections import Counter

    print(f"\n{'='*60}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for provider in providers:
        key = f"{provider}_rating"
        ratings = [j[key] for j in annotated if key in j and j[key] >= 0]
        if not ratings:
            continue
        avg = sum(ratings) / len(ratings)
        dist = Counter(ratings)
        print(f"\n  {provider} ({len(ratings)} jokes rated, avg: {avg:.2f}):",
              flush=True)
        for r in sorted(dist.keys()):
            bar = "#" * dist[r]
            print(f"    {r:>2}: {dist[r]:>3}  {bar}", flush=True)

    # Cross-provider correlation if multiple providers
    if len(providers) >= 2:
        print(f"\n  Cross-provider comparison:", flush=True)
        p1, p2 = providers[0], providers[1]
        k1, k2 = f"{p1}_rating", f"{p2}_rating"
        both = [(j[k1], j[k2]) for j in annotated
                if k1 in j and k2 in j and j[k1] >= 0 and j[k2] >= 0]
        if both:
            diffs = [abs(a - b) for a, b in both]
            avg_diff = sum(diffs) / len(diffs)
            agree = sum(1 for a, b in both if a == b)
            print(f"    {p1} vs {p2}: avg difference {avg_diff:.2f}, "
                  f"exact agreement {agree}/{len(both)} "
                  f"({100*agree/len(both):.0f}%)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze pun jokes with LLM APIs")
    parser.add_argument("--gemini-only", action="store_true",
                        help="Only use Gemini")
    parser.add_argument("--openai-only", action="store_true",
                        help="Only use OpenAI")
    parser.add_argument("--anthropic-only", action="store_true",
                        help="Only use Anthropic")
    parser.add_argument("--input", default="raw_samples.json",
                        help="Input JSON file (default: raw_samples.json)")
    parser.add_argument("--output", default="annotated_samples.json",
                        help="Output JSON file (default: annotated_samples.json)")
    args = parser.parse_args()

    only_flags = [args.gemini_only, args.openai_only, args.anthropic_only]
    use_only = any(only_flags)

    # Load jokes
    input_path = Path(__file__).parent / args.input
    with open(input_path) as f:
        jokes = json.load(f)
    print(f"Loaded {len(jokes)} jokes from {input_path.name}", flush=True)

    # Determine which providers to run
    providers = {}
    if not use_only or args.gemini_only:
        if os.environ.get("GEMINI_API_KEY"):
            providers["gemini"] = analyze_with_gemini
        else:
            print("WARNING: GEMINI_API_KEY not set, skipping Gemini", flush=True)
    if not use_only or args.openai_only:
        if os.environ.get("OPENAI_API_KEY"):
            providers["openai"] = analyze_with_openai
        else:
            print("WARNING: OPENAI_API_KEY not set, skipping OpenAI", flush=True)
    if not use_only or args.anthropic_only:
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers["anthropic"] = analyze_with_anthropic
        else:
            print("WARNING: ANTHROPIC_API_KEY not set, skipping Anthropic", flush=True)

    if not providers:
        print("ERROR: No API keys available!", flush=True)
        sys.exit(1)

    print(f"Running with providers: {', '.join(providers.keys())}", flush=True)

    # Run analyses
    results_by_provider = {}
    for name, fn in providers.items():
        results_by_provider[name] = run_analysis(jokes, fn, name.upper())

    # Merge and save
    annotated = merge_results(jokes, results_by_provider)

    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"\nWrote {len(annotated)} annotated jokes to {output_path.name}", flush=True)

    # Print summary
    print_summary(annotated, list(providers.keys()))


if __name__ == "__main__":
    main()
