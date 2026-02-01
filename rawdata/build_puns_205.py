#!/usr/bin/env python3
"""
Build puns_205.json by merging the 205 curated jokes with fresh ratings
from Gemini, Anthropic, and claudecode.

Output format per joke:
  - id, sentence, straight, punny
  - explanation (from claudecode)
  - gemini_rating, anthropic_rating, claudecode_rating
  - consensus_score (gemini + claudecode)
  - sorted by consensus_score desc, then anthropic_rating desc, then sentence length desc
"""

import json
from pathlib import Path

BASE = Path(__file__).parent
SCRATCHPAD = Path("/private/tmp/claude/-Users-davidbau-git-puns/73e3fb6d-9317-45d9-8719-faa2fbd54731/scratchpad")

def main():
    # Load the 205 base jokes
    with open(BASE / "top205_full.json") as f:
        jokes = json.load(f)
    print(f"Loaded {len(jokes)} base jokes")

    # Load Gemini ratings
    with open(BASE / "annotated_205_gemini.json") as f:
        gemini_data = json.load(f)
    gemini_by_id = {}
    for j in gemini_data:
        gemini_by_id[j["id"]] = {
            "gemini_explanation": j.get("gemini_explanation", ""),
            "gemini_rating": j.get("gemini_rating", -1),
        }
    print(f"Loaded Gemini ratings: {len(gemini_by_id)}")

    # Load Anthropic ratings
    with open(BASE / "annotated_205_anthropic.json") as f:
        anthropic_data = json.load(f)
    anthropic_by_id = {}
    for j in anthropic_data:
        anthropic_by_id[j["id"]] = {
            "anthropic_explanation": j.get("anthropic_explanation", ""),
            "anthropic_rating": j.get("anthropic_rating", -1),
        }
    print(f"Loaded Anthropic ratings: {len(anthropic_by_id)}")

    # Load claudecode ratings (5 batches)
    cc_by_id = {}
    for i in range(1, 6):
        with open(SCRATCHPAD / f"cc_batch{i}.json") as f:
            batch = json.load(f)
        for j in batch:
            cc_by_id[j["id"]] = {
                "claudecode_explanation": j.get("explanation", ""),
                "claudecode_rating": j.get("rating", -1),
            }
    print(f"Loaded claudecode ratings: {len(cc_by_id)}")

    # Merge
    result = []
    missing = {"gemini": 0, "anthropic": 0, "claudecode": 0}
    for joke in jokes:
        jid = joke["id"]
        entry = {
            "id": jid,
            "sentence": joke["sentence"],
            "straight": joke["straight"],
            "punny": joke["punny"],
        }

        # Add ratings
        if jid in gemini_by_id:
            entry.update(gemini_by_id[jid])
        else:
            missing["gemini"] += 1
            entry["gemini_explanation"] = ""
            entry["gemini_rating"] = -1

        if jid in anthropic_by_id:
            entry.update(anthropic_by_id[jid])
        else:
            missing["anthropic"] += 1
            entry["anthropic_explanation"] = ""
            entry["anthropic_rating"] = -1

        if jid in cc_by_id:
            entry.update(cc_by_id[jid])
        else:
            missing["claudecode"] += 1
            entry["claudecode_explanation"] = ""
            entry["claudecode_rating"] = -1

        # Compute consensus score (gemini + claudecode)
        g = entry["gemini_rating"] if entry["gemini_rating"] >= 0 else 0
        cc = entry["claudecode_rating"] if entry["claudecode_rating"] >= 0 else 0
        a = entry["anthropic_rating"] if entry["anthropic_rating"] >= 0 else 0
        entry["consensus_score"] = g + cc

        # Use claudecode explanation as the primary explanation
        entry["explanation"] = entry["claudecode_explanation"]

        result.append(entry)

    if any(v > 0 for v in missing.values()):
        print(f"  Missing ratings: {missing}")

    # Sort: consensus_score desc, anthropic desc, sentence length desc
    result.sort(key=lambda j: (
        j["consensus_score"],
        j.get("anthropic_rating", 0),
        len(j["sentence"]),
    ), reverse=True)

    # Clean output: select final field order
    output = []
    for j in result:
        output.append({
            "id": j["id"],
            "sentence": j["sentence"],
            "straight": j["straight"],
            "punny": j["punny"],
            "explanation": j["explanation"],
            "gemini_rating": j["gemini_rating"],
            "anthropic_rating": j["anthropic_rating"],
            "claudecode_rating": j["claudecode_rating"],
            "consensus_score": j["consensus_score"],
        })

    with open(BASE / "puns_205.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {len(output)} jokes to puns_205.json")

    # Summary stats
    g_ratings = [j["gemini_rating"] for j in output if j["gemini_rating"] >= 0]
    a_ratings = [j["anthropic_rating"] for j in output if j["anthropic_rating"] >= 0]
    cc_ratings = [j["claudecode_rating"] for j in output if j["claudecode_rating"] >= 0]
    consensus = [j["consensus_score"] for j in output]

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Gemini:    avg {sum(g_ratings)/len(g_ratings):.2f}  (n={len(g_ratings)})")
    print(f"  Anthropic: avg {sum(a_ratings)/len(a_ratings):.2f}  (n={len(a_ratings)})")
    print(f"  Claudecode: avg {sum(cc_ratings)/len(cc_ratings):.2f}  (n={len(cc_ratings)})")
    print(f"  Consensus: avg {sum(consensus)/len(consensus):.2f}  (range {min(consensus)}-{max(consensus)})")

    # Show top 10 and bottom 10
    print(f"\n  TOP 10:")
    for j in output[:10]:
        print(f"    [{j['consensus_score']}] (g={j['gemini_rating']}, a={j['anthropic_rating']}, cc={j['claudecode_rating']}) "
              f"{j['sentence'][:70]}... => {j['punny']}")

    print(f"\n  BOTTOM 10:")
    for j in output[-10:]:
        print(f"    [{j['consensus_score']}] (g={j['gemini_rating']}, a={j['anthropic_rating']}, cc={j['claudecode_rating']}) "
              f"{j['sentence'][:70]}... => {j['punny']}")


if __name__ == "__main__":
    main()
