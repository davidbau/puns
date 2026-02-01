#!/usr/bin/env python3
"""
Apply hand-edits from top_300_jokes.json to unique_jokes.json and all_jokes.json.

Edits are encoded as "note" or "notes" fields:
- Removal: "Not funny", "Not funny enough", "Duplicative", "Duplicate of ...", etc.
- Sentence edit: a replacement sentence (usually contains "___")
"""

import json
import re
from pathlib import Path

BASE = Path(__file__).parent

# Removal patterns (case-insensitive)
REMOVAL_PATTERNS = [
    r"^not\s+(really\s+)?funn",     # "Not funny", "Not really funny", "Not funnty"
    r"^not\s+very\s+funn",          # "Not very funny"
    r"^duplicat",                     # "Duplicative", "Duplicate of ..."
    r"^last word is not funny",      # specific critique on id 122
    r"^not funny",                    # "not funny" lowercase
]

def is_removal(text):
    """Check if annotation text indicates the joke should be removed."""
    text = text.strip().rstrip(".")
    for pat in REMOVAL_PATTERNS:
        if re.match(pat, text, re.IGNORECASE):
            return True
    return False

def load_edited_json(path):
    """Load the edited JSON, handling quirks like duplicate keys and typos."""
    text = path.read_text()
    # Fix the "9straight" typo (id 182) -> "straight"
    text = text.replace('"9straight"', '"straight"')
    # Fix trailing commas before ] or }
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)
    return json.loads(text)

def classify_edits(edited_jokes):
    """Classify each joke as 'remove', 'edit', or 'keep'."""
    removals = set()
    edits = {}  # id -> {sentence, straight, punny}

    for joke in edited_jokes:
        jid = joke["id"]
        note = joke.get("note", joke.get("notes", ""))

        if not note:
            # No annotation — keep as-is but still update straight/punny
            # in case user edited those directly
            edits[jid] = {
                "sentence": joke["sentence"],
                "straight": joke.get("straight", []),
                "punny": joke.get("punny", []),
            }
            continue

        if is_removal(note):
            removals.add(jid)
        else:
            # It's a sentence edit — use the note as the new sentence
            edits[jid] = {
                "sentence": note,
                "straight": joke.get("straight", []),
                "punny": joke.get("punny", []),
            }

    return removals, edits

def apply_to_jokes(jokes, removals, edits):
    """Apply removals and edits to a joke list."""
    result = []
    removed_count = 0
    edited_count = 0

    for joke in jokes:
        jid = joke["id"]
        if jid in removals:
            removed_count += 1
            continue

        if jid in edits:
            edit = edits[jid]
            joke = dict(joke)  # copy
            joke["sentence"] = edit["sentence"]
            joke["straight"] = edit["straight"]
            joke["punny"] = edit["punny"]
            # Remove any note/notes fields that were added during editing
            joke.pop("note", None)
            joke.pop("notes", None)
            edited_count += 1

        result.append(joke)

    return result, removed_count, edited_count

def main():
    # Load the hand-edited file
    edited_path = BASE / "top_300_jokes.json"
    edited_jokes = load_edited_json(edited_path)
    print(f"Loaded {len(edited_jokes)} jokes from edited file")

    # Classify
    removals, edits = classify_edits(edited_jokes)
    kept = len(edited_jokes) - len(removals)
    edit_count = sum(1 for j in edited_jokes
                     if j["id"] not in removals
                     and (j.get("note") or j.get("notes", "")))
    print(f"\n  Removals: {len(removals)} jokes marked not funny/duplicative")
    print(f"  Edits: {edit_count} jokes with sentence changes")
    print(f"  Unchanged: {kept - edit_count} jokes kept as-is")
    print(f"  Total kept from top 300: {kept}")

    # Apply to unique_jokes.json
    unique_path = BASE / "unique_jokes.json"
    unique_jokes = json.load(open(unique_path))
    print(f"\nLoaded {len(unique_jokes)} jokes from unique_jokes.json")

    updated_unique, u_removed, u_edited = apply_to_jokes(unique_jokes, removals, edits)
    print(f"  Removed: {u_removed}")
    print(f"  Edited: {u_edited}")
    print(f"  Result: {len(updated_unique)} jokes")

    with open(unique_path, "w") as f:
        json.dump(updated_unique, f, indent=2)
    print(f"  Wrote {len(updated_unique)} jokes to unique_jokes.json")

    # Apply to all_jokes.json
    all_path = BASE / "all_jokes.json"
    all_jokes = json.load(open(all_path))
    print(f"\nLoaded {len(all_jokes)} jokes from all_jokes.json")

    updated_all, a_removed, a_edited = apply_to_jokes(all_jokes, removals, edits)
    print(f"  Removed: {a_removed}")
    print(f"  Edited: {a_edited}")
    print(f"  Result: {len(updated_all)} jokes")

    with open(all_path, "w") as f:
        json.dump(updated_all, f, indent=2)
    print(f"  Wrote {len(updated_all)} jokes to all_jokes.json")

    # Show removal details
    print(f"\n{'='*60}")
    print(f"  REMOVED JOKE IDS ({len(removals)}):")
    print(f"{'='*60}")
    for jid in sorted(removals):
        print(f"  {jid}")

    # Show edit details
    print(f"\n{'='*60}")
    print(f"  EDITED JOKES ({edit_count}):")
    print(f"{'='*60}")
    for joke in edited_jokes:
        jid = joke["id"]
        if jid in removals:
            continue
        note = joke.get("note", joke.get("notes", ""))
        if note:
            print(f"  {jid}: {note}")

if __name__ == "__main__":
    main()
