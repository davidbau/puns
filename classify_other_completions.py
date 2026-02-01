#!/usr/bin/env python3
"""
Classify "other" completions from the cloze benchmark as straight, funny, or truly other.
Then update contextual_cloze_tests_100.json with expanded completion lists and re-score.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent

# ── Manual classification of observed "other" words per pair ──
# Based on reviewing all model outputs against the joke context.
# Format: pair_id -> {"also_straight": [...], "also_funny": [...]}
#
# Criteria:
#   also_straight: a sensible literal/non-punny completion for the blank
#   also_funny:    exploits a double meaning, wordplay, or pun (even if
#                  a different pun than the intended one)

CLASSIFICATIONS = {
    0: {  # electrician quit politics, party lost ___
        "also_straight": ["election"],  # not great but literal
        "also_funny": [],
    },
    1: {  # traffic light embarrassed, had to ___
        "also_straight": ["fail", "admit"],
        "also_funny": [],
    },
    2: {  # coin maker went crazy, stopped making ___
        "also_straight": ["coins", "money"],
        "also_funny": ["change"],  # change = coins AND transformation
    },
    3: {  # chicken percussion contest, incredible ___
        "also_straight": ["drumming", "talents", "skills", "performance"],
        "also_funny": [],
    },
    4: {  # hockey player banker, handling ___
        "also_straight": [],
        "also_funny": [],
    },
    5: {  # lightning bolt getting ___
        "also_straight": ["electrocuted"],
        "also_funny": ["fired", "struck"],  # fired/struck = both termination AND electricity
    },
    6: {  # wizard music school, great ___
        "also_straight": ["ear", "imagination", "teacher", "sense", "melody", "gift"],
        "also_funny": ["spell"],  # spell = magic AND musical quality
    },
    7: {  # shovel's contribution, truly ___
        "also_straight": ["remarkable", "underappreciated", "amazing"],
        "also_funny": ["digging"],  # digging = shovels AND appreciated (slang)
    },
    8: {  # commuter's experiences, hard to ___
        "also_straight": ["translate", "quantify", "relate", "believe"],
        "also_funny": ["track"],  # track = commuter rails AND follow
    },
    9: {  # union workers bowling, got a ___
        "also_straight": ["discount", "trophy", "win"],
        "also_funny": [],
    },
    10: {  # coffee got ___
        "also_straight": ["spilled"],
        "also_funny": [],
    },
    11: {  # king 12 inches, pretty good ___
        "also_straight": ["candidate"],
        "also_funny": [],
    },
    12: {  # bicycle fell, was ___
        "also_straight": [],
        "also_funny": ["tired", "two-tired"],  # tired/two-tired = pun variants of "two tired"
    },
    13: {  # boat mechanic, everything ___
        "also_straight": ["together", "operational"],
        "also_funny": [],
    },
    14: {  # knife salesman, sharp ___
        "also_straight": [],
        "also_funny": ["point"],  # point = sharp object AND argument point
    },
    15: {  # vulture one ___ (carrion)
        "also_straight": ["animal"],
        "also_funny": ["carry", "carryon", "carry-on"],  # carry-on = sounds like carrion
    },
    16: {  # knife sharpener, work was ___
        "also_straight": ["scarce", "unbearable", "unfulfilling"],
        "also_funny": ["cutting", "cutthroat"],  # knives + harsh
    },
    17: {  # fireworks been ___
        "also_straight": ["prohibited", "increasing", "growing"],
        "also_funny": [],
    },
    18: {  # tennis stringer, making a ___
        "also_straight": [],
        "also_funny": [],
    },
    19: {  # electrician mall, emotional ___
        "also_straight": ["outing", "break"],
        "also_funny": ["charge"],  # charge = electricity AND emotional intensity
    },
    20: {  # mushroom salary, always a ___
        "also_straight": ["shortage"],
        "also_funny": ["fun-gi", "fungus"],  # fungi/fun-gi = mushroom + fun guy
    },
    21: {  # crab never shared, was ___
        "also_straight": [],
        "also_funny": ["selfish"],  # selfish sounds like shellfish
    },
    22: {  # boxer party, bring the ___
        "also_straight": ["catering"],
        "also_funny": ["punchline"],  # punch + line
    },
    23: {  # lizard keeper IT, worked with ___
        "also_straight": ["reptiles", "computers", "snakes"],
        "also_funny": ["scale"],  # scale = reptile AND tech scaling
    },
    24: {  # elephant packed light, his ___
        "also_straight": ["backpack"],
        "also_funny": [],
    },
    25: {  # lonely CEO, some ___
        "also_straight": ["companionship", "humanity"],
        "also_funny": [],
    },
    26: {  # dyslexic zombie, named ___
        "also_straight": [],
        "also_funny": ["braaaaad"],  # zombie pronunciation of Brad/Brain
    },
    27: {  # Lancelot armor, many ___
        "also_straight": ["pranks", "all-nighters"],
        "also_funny": ["knights"],  # knights = knight shifts
    },
    28: {  # student ate homework, piece of ___
        "also_straight": ["work"],
        "also_funny": [],
    },
    29: {  # mushroom parties, was a ___
        "also_straight": [],
        "also_funny": ["fun-gi", "fun", "funny"],  # variants of fungi/fun-guy
    },
    30: {  # fish smart, in a ___
        "also_straight": ["tank"],
        "also_funny": [],
    },
    31: {  # homeowner leak, hit the ___
        "also_straight": [],
        "also_funny": ["roof"],  # hit the roof = angry AND it's about a leak
    },
    33: {  # train conductor, excellent ___
        "also_straight": ["communication"],
        "also_funny": ["record"],  # record = partial of "track record" (trains have tracks)
    },
    34: {  # butcher poker, raised the ___
        "also_straight": [],
        "also_funny": ["stakes", "stake", "steak"],  # steaks/stakes pun variants
    },
    36: {  # walrus notarizing, perfect ___
        "also_straight": ["penmanship"],
        "also_funny": [],
    },
    37: {  # data scientist, so ___
        "also_straight": ["introverted", "nerdy", "absorbed"],
        "also_funny": ["analytical"],  # analytical = data analysis AND personality trait
    },
    38: {  # pirate treasure, ___ pains
        "also_straight": ["backache"],  # backache is a valid straight completion
        "also_funny": ["arrhh"],  # pirate sound
    },
    39: {  # invisible man, couldn't ___
        "also_straight": ["afford"],
        "also_funny": [],
    },
    42: {  # possessive geologist, was ___
        "also_straight": ["his"],  # possessive = his/hers
        "also_funny": ["hers"],
    },
    43: {  # chemistry marriage counselor, perfect ___
        "also_straight": ["match"],
        "also_funny": ["formula", "bond"],  # chemistry puns
    },
    44: {  # pharmacist argue, has a ___
        "also_straight": [],
        "also_funny": ["prescription", "solution"],  # pharmacy/chemistry puns
    },
    45: {  # scuba instructor, feel ___
        "also_straight": ["uncomfortable", "nervous", "overwhelmed"],
        "also_funny": ["pressured", "pressurized"],  # pressure = scuba AND emotional
    },
    46: {  # mattress salesman, lying on ___
        "also_straight": ["mattress"],
        "also_funny": [],
    },
    47: {  # indecisive chef, still had ___
        "also_straight": ["trouble"],
        "also_funny": ["cold"],  # cold feet = indecision AND cold food
    },
    48: {  # dry cleaner, so many ___
        "also_straight": ["customers", "garments", "shirts", "stains"],
        "also_funny": [],
    },
    49: {  # leopard hide-and-seek, always ___
        "also_straight": ["visible", "seen", "found"],
        "also_funny": [],
    },
}


def main():
    # Load current tests
    with open(BASE / "contextual_cloze_tests_100.json") as f:
        tests = json.load(f)

    # Expand completion lists
    updates = 0
    for test in tests:
        pid = test["pair_id"]
        if pid not in CLASSIFICATIONS:
            continue
        cls = CLASSIFICATIONS[pid]

        if test["type"] == "straight":
            new_straight = cls["also_straight"]
            new_funny = cls["also_funny"]
        else:
            new_straight = cls["also_straight"]
            new_funny = cls["also_funny"]

        # Add to expected/contrast based on test type
        if test["type"] == "straight":
            for w in new_straight:
                if w.lower() not in [x.lower() for x in test["expected_completion"]]:
                    test["expected_completion"].append(w)
                    updates += 1
            for w in new_funny:
                if w.lower() not in [x.lower() for x in test["contrast_completion"]]:
                    test["contrast_completion"].append(w)
                    updates += 1
        else:  # funny
            for w in new_funny:
                if w.lower() not in [x.lower() for x in test["expected_completion"]]:
                    test["expected_completion"].append(w)
                    updates += 1
            for w in new_straight:
                if w.lower() not in [x.lower() for x in test["contrast_completion"]]:
                    test["contrast_completion"].append(w)
                    updates += 1

    # Save updated tests
    with open(BASE / "contextual_cloze_tests_100.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(f"Updated {updates} completion entries across tests")

    # Re-score the benchmark results
    with open(BASE / "cloze_benchmark_results.json") as f:
        results = json.load(f)

    # Build test lookup
    test_lookup = {}
    for t in tests:
        test_lookup[(t["pair_id"], t["type"])] = t

    print(f"\n{'='*80}")
    print("  RE-SCORED RESULTS (with expanded completion lists)")
    print(f"{'='*80}")
    print(f"  {'Model':<12} {'Overall':>8} {'Straight':>9} {'Funny':>8} "
          f"{'Contrast':>9} {'Other':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")

    for model_nick, mdata in results.items():
        correct = 0
        contrast = 0
        other = 0
        s_correct = s_contrast = s_other = s_total = 0
        f_correct = f_contrast = f_other = f_total = 0

        for d in mdata["details"]:
            word = d["extracted_word"]
            t = test_lookup[(d["pair_id"], d["type"])]

            hit_expected = any(
                word == exp.lower() or word == exp.lower().split()[0]
                for exp in t["expected_completion"]
            ) if word else False
            hit_contrast = any(
                word == exp.lower() or word == exp.lower().split()[0]
                for exp in t["contrast_completion"]
            ) if word else False

            if hit_expected:
                correct += 1
                outcome = "correct"
            elif hit_contrast:
                contrast += 1
                outcome = "contrast"
            else:
                other += 1
                outcome = "other"

            # Update detail record
            d["outcome_rescored"] = outcome

            if d["type"] == "straight":
                s_total += 1
                if outcome == "correct": s_correct += 1
                elif outcome == "contrast": s_contrast += 1
                else: s_other += 1
            else:
                f_total += 1
                if outcome == "correct": f_correct += 1
                elif outcome == "contrast": f_contrast += 1
                else: f_other += 1

        total = len(mdata["details"])
        s_acc = s_correct / s_total if s_total else 0
        f_acc = f_correct / f_total if f_total else 0
        acc = correct / total if total else 0

        mdata["summary_rescored"] = {
            "accuracy": acc,
            "straight_accuracy": s_acc,
            "funny_accuracy": f_acc,
            "correct": correct,
            "contrast": contrast,
            "other": other,
        }

        print(f"  {model_nick:<12} {acc:>7.1%} {s_acc:>8.1%} "
              f"{f_acc:>7.1%} "
              f"{contrast:>5}/{total:<3} "
              f"{other:>4}/{total:<3}")

    # Also show original for comparison
    print(f"\n  ORIGINAL SCORES (before expanding lists):")
    print(f"  {'Model':<12} {'Overall':>8} {'Straight':>9} {'Funny':>8} "
          f"{'Contrast':>9} {'Other':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")
    for model_nick, mdata in results.items():
        s = mdata["summary"]
        print(f"  {model_nick:<12} {s['accuracy']:>7.1%} {s['straight_accuracy']:>8.1%} "
              f"{s['funny_accuracy']:>7.1%} "
              f"{s['contrast_match']:>5}/{s['total']:<3} "
              f"{s['other']:>4}/{s['total']:<3}")

    # Save re-scored results
    with open(BASE / "cloze_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved re-scored results to cloze_benchmark_results.json")

    # Print remaining "truly other" words for inspection
    print(f"\n{'='*80}")
    print("  REMAINING 'OTHER' WORDS (not reclassified)")
    print(f"{'='*80}")
    from collections import Counter
    remaining_straight = Counter()
    remaining_funny = Counter()
    for model_nick, mdata in results.items():
        for d in mdata["details"]:
            if d.get("outcome_rescored", d["outcome"]) == "other" and d["extracted_word"]:
                if d["type"] == "straight":
                    remaining_straight[d["extracted_word"]] += 1
                else:
                    remaining_funny[d["extracted_word"]] += 1

    print("\n  Straight 'truly other':")
    for w, c in remaining_straight.most_common(20):
        print(f"    {w:20s} {c}")
    print("\n  Funny 'truly other':")
    for w, c in remaining_funny.most_common(20):
        print(f"    {w:20s} {c}")


if __name__ == "__main__":
    main()
