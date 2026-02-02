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
                             layer_indices, batch_size=10, remote=True,
                             save_dir=None, file_prefix="",
                             batch_offset=0):
    """
    Collect activations at ALL layers and top-1 predictions for batches of
    prompts via NDIF.  Activations and predictions come from the same
    forward pass.

    Server-side optimization: for each prompt, all layer vectors are
    stacked into one compact (n_layers, hidden_dim) tensor before
    .save(), so only ~1.3 MB per prompt (float16) is transferred instead
    of full layer outputs.  Predictions (argmax token id) ride along for
    free.

    Accounts for:
    - Left-padding: nnsight left-pads shorter prompts when batching,
      so token positions are shifted by (max_batch_len - prompt_len).
    - Output structure: auto-detects tuple (GPT-2) vs bare tensor (Llama).
    - Incremental saving: per-layer .npy files and per-batch prediction
      files are written after each batch.

    Parameters:
        save_dir: Path — if provided, saves incrementally to this directory
        file_prefix: str — filename prefix, e.g. "llama31_70b_instruct_pred_c"
        batch_offset: int — starting batch number for filenames (for resume)

    Returns:
        layer_data: dict {layer_idx: np.array of shape (n_prompts, hidden_dim)}
        pred_token_ids: list of int — predicted token id at each position
    """
    n = len(prompts_and_positions)
    n_layers = len(layer_indices)
    n_batches = (n + batch_size - 1) // batch_size

    # Accumulate per-layer results across batches
    layer_results = {l: [] for l in layer_indices}
    all_pred_ids = []

    # Probe layer output structure: some architectures (GPT-2) return a
    # tuple (hidden_states, ...) while others return a bare tensor.
    with model.trace(remote=remote) as tracer:
        with tracer.invoke("test"):
            _probe = layers_module[layer_indices[0]].output.save()
    output_is_tuple = isinstance(_probe, tuple)
    print(f"    Layer output type: {'tuple' if output_is_tuple else 'tensor'}",
          flush=True)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        # Compute per-prompt token lengths and left-padding offsets.
        # nnsight left-pads all prompts in a trace to the longest one,
        # so position indices must be shifted by (max_len - this_len).
        token_lengths = [
            len(model.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, pos in batch
        ]
        max_len = max(token_lengths)
        pad_offsets = [max_len - tl for tl in token_lengths]

        # ── Server-side computation ──────────────────────────────────
        # For each prompt: stack all layer activations into one compact
        # (n_layers, hidden_dim) tensor, and grab the top-1 prediction.
        # All proxy operations (indexing, stacking, argmax) execute on
        # the server; only the compact results are transferred.
        with model.trace(remote=remote) as tracer:
            prompt_results = []
            for (prompt, pos), pad_offset in zip(batch, pad_offsets):
                adjusted_pos = pos + pad_offset
                with tracer.invoke(prompt):
                    # Stack all layer activations: (n_layers, hidden_dim)
                    # .cpu() is needed because the 70B model is sharded
                    # across GPUs — layers on different devices can't be
                    # stacked directly.  This runs on the server (free).
                    layer_vecs = []
                    for layer_idx in layer_indices:
                        out = layers_module[layer_idx].output
                        hidden = out[0] if output_is_tuple else out
                        layer_vecs.append(hidden[0, adjusted_pos, :].cpu())
                    stacked = torch.stack(layer_vecs)

                    # Top-1 prediction from the same forward pass
                    logits = model.lm_head.output[0, adjusted_pos, :]
                    pred_id = logits.argmax(-1)

                    prompt_results.append({
                        "activations": stacked,    # (n_layers, hidden_dim)
                        "pred_token_id": pred_id,  # scalar
                    })
            saved_batch = prompt_results.save()

        # ── Client-side unpacking ────────────────────────────────────
        # Keep native float16 precision (convert bfloat16 → float16
        # for numpy compatibility).
        batch_vectors = {l: [] for l in layer_indices}
        batch_preds = []
        for i in range(batch_n):
            acts = saved_batch[i]["activations"]
            if hasattr(acts, 'dtype') and acts.dtype == torch.bfloat16:
                acts = acts.half()
            acts = acts.cpu().numpy()  # (n_layers, hidden_dim) in float16

            pred = int(saved_batch[i]["pred_token_id"].item())
            all_pred_ids.append(pred)
            batch_preds.append(pred)

            for j, layer_idx in enumerate(layer_indices):
                vec = acts[j]
                layer_results[layer_idx].append(vec)
                batch_vectors[layer_idx].append(vec)

        # ── Incremental save ─────────────────────────────────────────
        if save_dir is not None:
            batch_num = batch_offset + batch_start // batch_size
            for layer_idx in layer_indices:
                filename = f"{file_prefix}_layer{layer_idx:02d}_batch{batch_num:02d}.npy"
                np.save(save_dir / filename, np.stack(batch_vectors[layer_idx]))
            # Predictions for this batch
            pred_filename = f"{file_prefix}_preds_batch{batch_num:02d}.npy"
            np.save(save_dir / pred_filename, np.array(batch_preds, dtype=np.int64))

        batch_display = batch_offset + batch_start // batch_size + 1
        total_batches = n_batches + batch_offset
        pad_range = f"pad=[{min(pad_offsets)}-{max(pad_offsets)}]"
        print(f"    batch {batch_display}/{total_batches}: "
              f"prompts {batch_start}-{batch_end-1} "
              f"({batch_n} prompts, {n_layers} layers) "
              f"{pad_range}", flush=True)

    # Stack each layer's results
    layer_data = {l: np.stack(vecs) for l, vecs in layer_results.items()}
    return layer_data, all_pred_ids


def collect_predictions(model, prompts_and_positions, batch_size=10, remote=True):
    """
    Lightweight prediction pass: collect only the top-1 predicted token
    at each target position.  No layer activations — much smaller download.

    Returns:
        pred_token_ids: list of int — argmax token id at each position
    """
    n = len(prompts_and_positions)
    n_batches = (n + batch_size - 1) // batch_size
    all_pred_ids = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        token_lengths = [
            len(model.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, pos in batch
        ]
        max_len = max(token_lengths)
        pad_offsets = [max_len - tl for tl in token_lengths]

        with model.trace(remote=remote) as tracer:
            pred_proxies = []
            for (prompt, pos), pad_offset in zip(batch, pad_offsets):
                adjusted_pos = pos + pad_offset
                with tracer.invoke(prompt):
                    logits = model.lm_head.output[0, adjusted_pos, :]
                    pred_proxies.append(logits.argmax(-1))
            saved_preds = pred_proxies.save()

        for i in range(batch_n):
            all_pred_ids.append(int(saved_preds[i].item()))

        batch_num = batch_start // batch_size + 1
        print(f"    pred batch {batch_num}/{n_batches}: "
              f"prompts {batch_start}-{batch_end-1}", flush=True)

    return all_pred_ids


def collect_detailed_predictions(model, prompts_and_positions,
                                 target_token_ids_per_prompt,
                                 batch_size=10, remote=True, top_k=20):
    """
    Collect top-k predictions and log-probabilities for specific target
    tokens at each position.  Lightweight — no layer activations.

    Server-side: computes log_softmax, extracts top-k and indexed target
    token log-probs.  Transfer per prompt is ~200 bytes.

    Parameters:
        prompts_and_positions: list of (prompt_str, token_position)
        target_token_ids_per_prompt: list of lists of int
            For each prompt, the specific token IDs to collect log-probs for.
        top_k: number of top tokens to return per prompt

    Returns:
        list of dicts per prompt:
            topk_ids: list of int (top_k token IDs)
            topk_logprobs: list of float (top_k log-probabilities)
            target_logprobs: list of float (one per target token ID)
    """
    n = len(prompts_and_positions)
    n_batches = (n + batch_size - 1) // batch_size
    all_results = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        token_lengths = [
            len(model.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, pos in batch
        ]
        max_len = max(token_lengths)
        pad_offsets = [max_len - tl for tl in token_lengths]

        batch_targets = target_token_ids_per_prompt[batch_start:batch_end]

        with model.trace(remote=remote) as tracer:
            prompt_results = []
            for idx, ((prompt, pos), pad_offset) in enumerate(
                zip(batch, pad_offsets)
            ):
                adjusted_pos = pos + pad_offset
                tids = batch_targets[idx]

                with tracer.invoke(prompt):
                    logits = model.lm_head.output[0, adjusted_pos, :].cpu()
                    log_probs = torch.log_softmax(logits.float(), dim=-1)

                    topk = log_probs.topk(top_k)

                    # Index specific target tokens one at a time
                    target_lps = []
                    for tid in tids:
                        target_lps.append(log_probs[tid])
                    target_tensor = torch.stack(target_lps) if target_lps \
                        else topk.values[:0]  # empty with right dtype

                    prompt_results.append({
                        "topk_vals": topk.values,
                        "topk_ids": topk.indices,
                        "target_lps": target_tensor,
                    })
            saved = prompt_results.save()

        for idx in range(batch_n):
            r = saved[idx]
            all_results.append({
                "topk_ids": r["topk_ids"].numpy().tolist(),
                "topk_logprobs": r["topk_vals"].numpy().tolist(),
                "target_logprobs": r["target_lps"].numpy().tolist(),
            })

        batch_num = batch_start // batch_size + 1
        print(f"    pred batch {batch_num}/{n_batches}: "
              f"prompts {batch_start}-{batch_end-1}", flush=True)

    return all_results


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

    print(f"Loaded {len(tests)} cloze tests, {len(jokes)} jokes", flush=True)

    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}", flush=True)
    model = LanguageModel(MODEL_NAME, device_map="auto")

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}", flush=True)

    # Select layers
    if args.layers:
        layer_indices = sorted(args.layers)
    else:
        layer_indices = list(range(n_layers))
    print(f"  Collecting {len(layer_indices)} layers", flush=True)

    # Compute token positions
    print(f"\n--- Computing {args.position} positions ---", flush=True)
    prompts_and_positions = compute_positions(
        tests, jokes_by_index, model.tokenizer, args.position
    )

    positions = [pos for _, pos in prompts_and_positions]
    print(f"  Position range: min={min(positions)}, max={max(positions)}, "
          f"mean={sum(positions)/len(positions):.0f}", flush=True)

    # Show examples
    for i in [0, 1]:
        prompt, pos = prompts_and_positions[i]
        tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
        tok_at = model.tokenizer.decode([tokens[pos]])
        print(f"  [{tests[i]['type']:>8}] pair={tests[i]['pair_id']} "
              f"pos={pos} token='{tok_at}'", flush=True)

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
    print(f"\nSaved metadata: {meta_filename}", flush=True)

    # Check which layers already have a merged file with all prompts
    file_prefix = f"{MODEL_SHORT}_{args.position}"
    n_prompts = len(prompts_and_positions)
    n_batches = (n_prompts + args.batch_size - 1) // args.batch_size

    needed_layers = []
    for layer_idx in layer_indices:
        merged = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy"
        if merged.exists():
            existing = np.load(merged)
            if existing.shape[0] >= n_prompts:
                continue  # already complete
        needed_layers.append(layer_idx)

    # ── Collect activations (skip if already complete) ───────────────────
    if needed_layers:
        skipped = len(layer_indices) - len(needed_layers)
        if skipped:
            print(f"\n  Skipping {skipped} layers (already complete)", flush=True)

        # Check for partial batch files from a previous interrupted run
        existing_batches = 0
        sample_layer = needed_layers[0]
        for b in range(n_batches):
            bf = OUTPUT_DIR / f"{file_prefix}_layer{sample_layer:02d}_batch{b:02d}.npy"
            if bf.exists():
                existing_batches = b + 1
            else:
                break
        remaining = prompts_and_positions
        if existing_batches > 0:
            skip_prompts = existing_batches * args.batch_size
            print(f"\n  Found {existing_batches} batch files, "
                  f"resuming from prompt {skip_prompts}", flush=True)
            remaining = prompts_and_positions[skip_prompts:]

        print(f"\n--- Collecting {args.position}: {len(needed_layers)} layers x "
              f"{len(remaining)} prompts ---", flush=True)
        print(f"  Estimated download: "
              f"{len(needed_layers) * len(remaining) * hidden_dim * 2 / 1e6:.0f} MB "
              f"(float16) in "
              f"{(len(remaining) + args.batch_size - 1) // args.batch_size} "
              f"NDIF calls", flush=True)

        collect_all_layers_batch(
            model, layers_module, remaining, needed_layers,
            batch_size=args.batch_size, remote=True,
            save_dir=OUTPUT_DIR, file_prefix=file_prefix,
            batch_offset=existing_batches,
        )

        # Merge batch files into single per-layer files
        print(f"\n  Merging batch files...", flush=True)
        for layer_idx in needed_layers:
            parts = []
            for b in range(n_batches):
                bf = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
                if bf.exists():
                    parts.append(np.load(bf))
            if parts:
                merged = np.concatenate(parts, axis=0)
                np.save(OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy", merged)
            for b in range(n_batches):
                bf = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
                if bf.exists():
                    bf.unlink()

        sample = np.load(OUTPUT_DIR / f"{file_prefix}_layer{needed_layers[0]:02d}.npy")
        print(f"  Saved {len(needed_layers)} layer files to {OUTPUT_DIR}/")
        print(f"  File shape: {sample.shape}  dtype={sample.dtype} "
              f"({sample.nbytes / 1e6:.1f} MB each)")

        # Merge prediction batch files
        pred_parts = []
        for b in range(n_batches):
            pf = OUTPUT_DIR / f"{file_prefix}_preds_batch{b:02d}.npy"
            if pf.exists():
                pred_parts.append(np.load(pf))
        if pred_parts:
            pred_token_ids = np.concatenate(pred_parts).tolist()
            for b in range(n_batches):
                pf = OUTPUT_DIR / f"{file_prefix}_preds_batch{b:02d}.npy"
                if pf.exists():
                    pf.unlink()

            # Classify and embed in metadata
            pun_words = {}
            for test in tests:
                if test["type"] == "funny":
                    pun_words[test["pair_id"]] = [
                        w.lower() for w in test["expected_completion"]
                    ]
            n_funny_pred = 0
            for i, tok_id in enumerate(pred_token_ids):
                word = model.tokenizer.decode([tok_id]).strip().lower()
                test = tests[i]
                is_pun = word in pun_words.get(test["pair_id"], [])
                if is_pun:
                    n_funny_pred += 1
                meta_out["samples"][i]["predicted_token_id"] = tok_id
                meta_out["samples"][i]["predicted_word"] = word
                meta_out["samples"][i]["predicted_funny"] = is_pun

            print(f"  {n_funny_pred}/{len(pred_token_ids)} predicted the pun word",
                  flush=True)
            with open(OUTPUT_DIR / meta_filename, "w") as f:
                json.dump(meta_out, f, indent=2)
    else:
        print(f"\nAll {len(layer_indices)} layers already collected.")

    # ── Detailed predictions: top-k tokens and target word probs ──────────
    # Always collect: lightweight pass (~200 bytes/prompt transferred).
    print(f"\n--- Collecting detailed predictions (top-20 + target word "
          f"log-probs) ---", flush=True)

    # Build target word lists per pair: pun words and straight words
    pun_words_by_pair = {}
    straight_words_by_pair = {}
    for test in tests:
        pid = test["pair_id"]
        words = [w.lower() for w in test["expected_completion"]]
        if test["type"] == "funny":
            pun_words_by_pair[pid] = words
        else:
            straight_words_by_pair[pid] = words

    # Tokenize target words (space-prefixed, first token)
    tokenizer = model.tokenizer
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def word_to_token(word):
        toks = tokenizer.encode(" " + word, add_special_tokens=False)
        return toks[0] if toks else None

    # Build per-prompt target token ID lists and word mappings
    target_token_ids_per_prompt = []
    target_word_maps = []  # parallel list: describes what each token ID is
    for i, test in enumerate(tests):
        pid = test["pair_id"]
        tids = []
        word_map = []

        for w in pun_words_by_pair.get(pid, []):
            tid = word_to_token(w)
            if tid is not None:
                tids.append(tid)
                word_map.append(("pun", w, tid))

        for w in straight_words_by_pair.get(pid, []):
            tid = word_to_token(w)
            if tid is not None:
                tids.append(tid)
                word_map.append(("straight", w, tid))

        tids.append(eot_id)
        word_map.append(("eot", "<|eot_id|>", eot_id))

        target_token_ids_per_prompt.append(tids)
        target_word_maps.append(word_map)

    raw_results = collect_detailed_predictions(
        model, prompts_and_positions, target_token_ids_per_prompt,
        batch_size=args.batch_size, remote=True, top_k=20,
    )

    # Assemble into a structured JSON
    detailed = []
    for i, (raw, wmap) in enumerate(zip(raw_results, target_word_maps)):
        test = tests[i]
        # Decode top-k tokens
        top_tokens = []
        for tid, lp in zip(raw["topk_ids"], raw["topk_logprobs"]):
            word = tokenizer.decode([tid]).strip()
            top_tokens.append({
                "token_id": tid, "word": word,
                "logprob": round(lp, 4),
                "prob": round(float(np.exp(lp)), 6),
            })

        # Map target logprobs back to word labels
        pun_probs = {}
        straight_probs = {}
        eot_prob = None
        for j, (category, word, tid) in enumerate(wmap):
            lp = raw["target_logprobs"][j]
            entry = {
                "token_id": tid, "logprob": round(lp, 4),
                "prob": round(float(np.exp(lp)), 6),
            }
            if category == "pun":
                pun_probs[word] = entry
            elif category == "straight":
                straight_probs[word] = entry
            else:
                eot_prob = entry

        detailed.append({
            "index": i,
            "pair_id": test["pair_id"],
            "type": test["type"],
            "top_tokens": top_tokens,
            "pun_word_probs": pun_probs,
            "straight_word_probs": straight_probs,
            "eot": eot_prob,
        })

    pred_file = OUTPUT_DIR / f"{file_prefix}_detailed_preds.json"
    with open(pred_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "position": args.position,
            "top_k": 20,
            "n_prompts": len(detailed),
            "results": detailed,
        }, f, indent=2)
    print(f"  Saved detailed predictions: {pred_file.name}", flush=True)

    # Quick summary
    total_pun_prob_funny = []
    total_pun_prob_straight = []
    for d in detailed:
        pun_total = sum(v["prob"] for v in d["pun_word_probs"].values())
        if d["type"] == "funny":
            total_pun_prob_funny.append(pun_total)
        else:
            total_pun_prob_straight.append(pun_total)
    if total_pun_prob_funny and total_pun_prob_straight:
        print(f"  Mean P(pun words): funny ctx={np.mean(total_pun_prob_funny):.4f}, "
              f"straight ctx={np.mean(total_pun_prob_straight):.4f}", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
