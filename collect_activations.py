#!/usr/bin/env python3
"""
Collect hidden-state activations from Llama-3.1-70B-Instruct via NDIF.

For each of the 100 contrastive cloze prompts, collects residual stream
activations at specified token positions across all layers, saving one
.npy file per layer.

Token positions:
    pred_c  — last token of the prompt (predicting C's completion)
    pred_b  — last token before B's completion word (predicting B's answer)

Naming convention:
    results/raw_activations/{model}_{position}_layer{NN}.npy   (100, 8192)
    results/raw_activations/{model}_{position}_meta.json        metadata

Usage:
    python3 collect_activations.py --position pred_c
    python3 collect_activations.py --position pred_b
    python3 collect_activations.py --position pred_c --layers 30 40 50 60
    python3 collect_activations.py --position pred_c --dry-run

Requires:
    nnsight, torch, numpy (see setup_env.sh)
    NDIF_API_KEY and HF_TOKEN in .env.local
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent

# ── Load environment ──────────────────────────────────────────────────────────
env_path = BASE / ".env.local"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

os.environ["NNSIGHT_API_KEY"] = os.getenv("NDIF_API_KEY", "")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from nnsight import LanguageModel

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_FILE = BASE / "datasets" / "contextual_cloze_tests_100.json"
PUNS_FILE = BASE / "datasets" / "puns_205.json"
OUTPUT_DIR = BASE / "results" / "raw_activations"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
MODEL_SHORT = "llama31_70b_instruct"

SYSTEM_PROMPT = (
    "You are completing a text. Given the text below, output only the "
    "single next word that best continues it. Output ONLY that one word, "
    "nothing else — no punctuation, no explanation."
)


# ── Prompt formatting ─────────────────────────────────────────────────────────

def format_chat_prompt(tokenizer, user_text):
    """Apply the instruct chat template to match the behavioral experiment."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def fill_sentence(sentence, completion):
    """Replace ___ in sentence with the completion."""
    return sentence.replace("___", completion)


def truncate_before_blank(sentence):
    """Truncate sentence just before ___, removing trailing whitespace."""
    idx = sentence.index("___")
    return sentence[:idx].rstrip()


# ── Token position finding ────────────────────────────────────────────────────

def find_pred_c_position(tokenizer, formatted_prompt, user_text):
    """
    Last token of the user's text — where model predicts C's completion.

    The chat template appends an assistant header after the user text,
    so we find the last content token by comparing the full formatted
    prompt with a version that has slightly shorter user text.
    """
    # Build a version with one less word to find where user content ends
    shorter = format_chat_prompt(tokenizer, user_text.rsplit(None, 1)[0])
    full_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    short_tokens = tokenizer.encode(shorter, add_special_tokens=False)

    # Find divergence point
    shared = 0
    for a, b in zip(full_tokens, short_tokens):
        if a != b:
            break
        shared += 1

    # The last token of the full user text is somewhere after the shared prefix.
    # We want the last token before the eot/assistant header.
    # Since the full prompt has more content tokens, count forward from divergence.
    # Actually, just count how many extra tokens the full prompt has before
    # the template suffix kicks in.
    #
    # Simpler: use the full prompt tokens and find the last non-special content token.
    # The generation prompt ends with the assistant header.  Count backwards to find
    # the last user content token.

    # Find where the post-user-content section starts by looking at the suffix
    suffix_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "X"}],
        tokenize=False, add_generation_prompt=True
    )
    suffix_after_content = suffix_text.split("X", 1)[1]
    suffix_tokens = tokenizer.encode(suffix_after_content, add_special_tokens=False)
    n_suffix = len(suffix_tokens)

    return len(full_tokens) - n_suffix - 1


def find_pred_b_position(tokenizer, formatted_prompt, formatted_prefix):
    """
    Last shared token between full prompt and prefix (truncated before B's word).

    We tokenize both and find where they first diverge.
    """
    full_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(formatted_prefix, add_special_tokens=False)

    shared = 0
    for a, b in zip(full_tokens, prefix_tokens):
        if a != b:
            break
        shared += 1

    return shared - 1


def compute_positions(tests, jokes_by_index, tokenizer, position_type):
    """
    Compute token positions for all 100 prompts.

    Returns list of (formatted_prompt, token_position) pairs.
    """
    results = []

    for test in tests:
        formatted = format_chat_prompt(tokenizer, test["prompt"])

        if position_type == "pred_c":
            pos = find_pred_c_position(tokenizer, formatted, test["prompt"])
            results.append((formatted, pos))

        elif position_type == "pred_b":
            # Reconstruct the prefix up to B's blank
            joke_a = jokes_by_index[test["joke_a_index"]]
            joke_b = jokes_by_index[test["joke_b_index"]]

            # Which completion was used?  straight or funny
            if test["type"] == "straight":
                a_word = joke_a["straight"][0]
            else:
                a_word = joke_a["punny"][0]

            a_filled = fill_sentence(joke_a["sentence"], a_word)
            b_truncated = truncate_before_blank(joke_b["sentence"])
            prefix_text = f"{a_filled} {b_truncated}"

            formatted_prefix = format_chat_prompt(tokenizer, prefix_text)
            pos = find_pred_b_position(tokenizer, formatted, formatted_prefix)
            results.append((formatted, pos))

        else:
            raise ValueError(f"Unknown position type: {position_type}")

    return results


# ── Activation collection ─────────────────────────────────────────────────────

def collect_all_layers_batch(model, layers_module, prompts_and_positions,
                             layer_indices, batch_size=10, remote=True):
    """
    Collect activations at ALL layers for a batch of prompts in a single
    NDIF call per batch.  Much more efficient than one call per layer.

    Returns dict: {layer_idx: np.array of shape (n_prompts, hidden_dim)}
    """
    n = len(prompts_and_positions)
    n_layers = len(layer_indices)

    # Accumulate per-layer results across batches
    layer_results = {l: [] for l in layer_indices}

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        # Precompute index map: flat_idx -> (prompt_idx_in_batch, layer_idx)
        index_map = []
        for p_idx in range(batch_n):
            for layer_idx in layer_indices:
                index_map.append((p_idx, layer_idx))

        with model.trace(remote=remote) as tracer:
            proxy_list = []
            for prompt, pos in batch:
                with tracer.invoke(prompt):
                    for layer_idx in layer_indices:
                        hidden = layers_module[layer_idx].output[0]
                        emb = hidden[pos, :]
                        proxy_list.append(emb)
            saved = proxy_list.save()

        # Unpack saved proxies using precomputed index map
        for i, (p_idx, layer_idx) in enumerate(index_map):
            layer_results[layer_idx].append(saved[i].float().cpu().numpy())

        print(f"    batch {batch_start//batch_size + 1}: "
              f"prompts {batch_start}-{batch_end-1} "
              f"({batch_n} x {n_layers} layers = {batch_n * n_layers} tensors)")

    # Stack each layer's results
    return {l: np.stack(vecs) for l, vecs in layer_results.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect activations from Llama-3.1-70B via NDIF")
    parser.add_argument("--position", required=True, choices=["pred_c", "pred_b"],
                        help="Token position to collect: pred_c or pred_b")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layer indices (default: all)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Prompts per NDIF batch (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config and positions without calling NDIF")
    args = parser.parse_args()

    # Load data
    with open(TESTS_FILE) as f:
        tests = json.load(f)
    with open(PUNS_FILE) as f:
        jokes = json.load(f)
    jokes_by_index = {j["index"]: j for j in jokes}

    print(f"Loaded {len(tests)} cloze tests, {len(jokes)} jokes")

    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}")
    model = LanguageModel(MODEL_NAME, device_map="auto")

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Select layers
    if args.layers:
        layer_indices = sorted(args.layers)
    else:
        layer_indices = list(range(n_layers))
    print(f"  Collecting {len(layer_indices)} layers")

    # Compute token positions
    print(f"\n--- Computing {args.position} positions ---")
    prompts_and_positions = compute_positions(
        tests, jokes_by_index, model.tokenizer, args.position
    )

    positions = [pos for _, pos in prompts_and_positions]
    print(f"  Position range: min={min(positions)}, max={max(positions)}, "
          f"mean={sum(positions)/len(positions):.0f}")

    # Show examples
    for i in [0, 1]:
        prompt, pos = prompts_and_positions[i]
        tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
        tok_at = model.tokenizer.decode([tokens[pos]])
        print(f"  [{tests[i]['type']:>8}] pair={tests[i]['pair_id']} "
              f"pos={pos} token='{tok_at}'")

    if args.dry_run:
        print("\n  (dry-run: no NDIF calls)")
        return

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layers_module = model.model.layers

    # Save metadata once
    metadata = []
    for i, test in enumerate(tests):
        metadata.append({
            "index": i,
            "pair_id": test["pair_id"],
            "type": test["type"],
            "prompt": test["prompt"],
            "joke_c_sentence": test["joke_c_sentence"],
            "token_position": positions[i],
        })

    meta_filename = f"{MODEL_SHORT}_{args.position}_meta.json"
    meta_out = {
        "model": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "position": args.position,
        "n_prompts": len(metadata),
        "n_layers_total": n_layers,
        "hidden_dim": hidden_dim,
        "system_prompt": SYSTEM_PROMPT,
        "file_shape": [len(metadata), hidden_dim],
        "file_axes": ["prompt", "hidden_dim"],
        "naming": f"{MODEL_SHORT}_{args.position}_layer{{NN}}.npy",
        "samples": metadata,
    }
    with open(OUTPUT_DIR / meta_filename, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nSaved metadata: {meta_filename}")

    # Check which layers still need collecting
    needed_layers = []
    for layer_idx in layer_indices:
        filename = f"{MODEL_SHORT}_{args.position}_layer{layer_idx:02d}.npy"
        if not (OUTPUT_DIR / filename).exists():
            needed_layers.append(layer_idx)

    if not needed_layers:
        print(f"\nAll {len(layer_indices)} layers already collected. Nothing to do.")
        return

    skipped = len(layer_indices) - len(needed_layers)
    if skipped:
        print(f"\n  Skipping {skipped} layers (already exist)")

    # Collect all needed layers in batched NDIF calls
    # Each call collects all layers for a batch of prompts
    # ~25 MB per batch of 10 prompts x 80 layers
    print(f"\n--- Collecting {args.position}: {len(needed_layers)} layers x "
          f"{len(prompts_and_positions)} prompts ---")
    print(f"  Estimated download: "
          f"{len(needed_layers) * len(prompts_and_positions) * 8192 * 4 / 1e6:.0f} MB "
          f"in {(len(prompts_and_positions) + args.batch_size - 1) // args.batch_size} "
          f"NDIF calls")

    layer_data = collect_all_layers_batch(
        model, layers_module, prompts_and_positions, needed_layers,
        batch_size=args.batch_size, remote=True
    )

    # Save one file per layer
    for layer_idx in needed_layers:
        filename = f"{MODEL_SHORT}_{args.position}_layer{layer_idx:02d}.npy"
        X = layer_data[layer_idx]
        np.save(OUTPUT_DIR / filename, X)

    print(f"\n  Saved {len(needed_layers)} layer files to {OUTPUT_DIR}/")
    print(f"  File shape: {X.shape}  ({X.nbytes / 1e6:.1f} MB each)")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
