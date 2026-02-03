#!/usr/bin/env python3
"""
Collect hidden-state activations and/or predictions from Llama-3.1-70B via NDIF.

For each contrastive cloze prompt in the specified dataset, collects residual
stream activations and/or detailed predictions (top-20 + target word log-probs)
in a single forward pass per batch.

Token positions:
    pred_c  — last token of the prompt (predicting C's completion)
    pred_b  — last token before B's completion word (requires 3-sentence format)

Naming convention:
    results/raw_activations/{model}_{dataset}_{position}_layer{NN}.npy
    results/raw_activations/{model}_{dataset}_{position}_meta.json
    results/raw_activations/{model}_{dataset}_{position}_detailed_preds.json

Usage:
    python3 collect_activations.py --position pred_c
    python3 collect_activations.py --position pred_c --dataset short_context_cloze_150.json
    python3 collect_activations.py --position pred_c --layers 30 40 50 60
    python3 collect_activations.py --position pred_c --skip-detailed-preds   # activations only
    python3 collect_activations.py --position pred_c --skip-activations      # predictions only
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
DEFAULT_TESTS_FILE = "contextual_cloze_tests_100.json"
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

def collect_batch(model, layers_module, prompts_and_positions,
                  layer_indices=None, batch_size=10, remote=True,
                  save_dir=None, file_prefix="", batch_offset=0,
                  target_token_ids_per_prompt=None, top_k=20):
    """
    Unified collection pass: activations and/or detailed predictions from
    a single forward pass per batch via NDIF.

    Collects:
    - Layer activations (if layer_indices is provided and non-empty)
    - Detailed predictions: top-k tokens + target token log-probs
      (if target_token_ids_per_prompt is provided)

    Server-side optimization: all proxy operations (indexing, stacking,
    log_softmax, topk) execute on the server; only compact results are
    transferred.

    Parameters:
        layer_indices: list of int — layers to collect (None/[] = skip activations)
        target_token_ids_per_prompt: list of lists — target tokens per prompt
            (None = skip detailed predictions, just collect top-1)
        top_k: int — number of top tokens to collect (default 20)
        save_dir: Path — if provided, saves incrementally
        file_prefix: str — filename prefix
        batch_offset: int — starting batch number for filenames (for resume)

    Returns:
        layer_data: dict {layer_idx: np.array} or {} if no activations
        detailed_preds: list of dicts with topk_ids, topk_logprobs,
            target_logprobs — or None if target_token_ids not provided
    """
    n = len(prompts_and_positions)
    n_batches = (n + batch_size - 1) // batch_size

    collect_activations = layer_indices and len(layer_indices) > 0
    collect_detailed = target_token_ids_per_prompt is not None
    n_layers = len(layer_indices) if collect_activations else 0

    # Accumulate results across batches
    layer_results = {l: [] for l in layer_indices} if collect_activations else {}
    all_detailed = [] if collect_detailed else None

    # Layer outputs are tuples for most models (hidden_states, ...), access [0]
    # We assume tuple format - this works for Llama and similar architectures
    if collect_activations:
        print(f"    Layer output: assuming tuple format (hidden_states, ...)", flush=True)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = prompts_and_positions[batch_start:batch_end]
        batch_n = len(batch)

        # Compute per-prompt token lengths and left-padding offsets
        token_lengths = [
            len(model.tokenizer.encode(prompt, add_special_tokens=False))
            for prompt, pos in batch
        ]
        max_len = max(token_lengths)
        pad_offsets = [max_len - tl for tl in token_lengths]

        # Get target tokens for this batch if collecting detailed preds
        batch_targets = None
        if collect_detailed:
            batch_targets = target_token_ids_per_prompt[batch_start:batch_end]

        # ── Server-side computation ──────────────────────────────────
        with model.trace(remote=remote) as tracer:
            prompt_results = []
            for idx, ((prompt, pos), pad_offset) in enumerate(
                zip(batch, pad_offsets)
            ):
                adjusted_pos = pos + pad_offset
                with tracer.invoke(prompt):
                    result = {}

                    # Stack all layer activations: (n_layers, hidden_dim)
                    if collect_activations:
                        layer_vecs = []
                        for layer_idx in layer_indices:
                            out = layers_module[layer_idx].output
                            # Layer output is tuple (hidden_states, ...), access [0]
                            hidden = out[0]
                            layer_vecs.append(hidden[0, adjusted_pos, :].cpu())
                        result["activations"] = torch.stack(layer_vecs)

                    # Predictions from the same forward pass
                    logits = model.lm_head.output[0, adjusted_pos, :].cpu()

                    if collect_detailed:
                        # Full log-softmax for top-k and target probs
                        log_probs = torch.log_softmax(logits.float(), dim=-1)
                        topk = log_probs.topk(top_k)
                        result["topk_vals"] = topk.values
                        result["topk_ids"] = topk.indices

                        # Target token log-probs
                        tids = batch_targets[idx]
                        target_lps = []
                        for tid in tids:
                            target_lps.append(log_probs[tid])
                        result["target_lps"] = torch.stack(target_lps) if target_lps \
                            else topk.values[:0]
                    else:
                        # Just top-1 prediction
                        result["pred_token_id"] = logits.argmax(-1)

                    prompt_results.append(result)
            saved_batch = prompt_results.save()

        # ── Client-side unpacking ────────────────────────────────────
        batch_vectors = {l: [] for l in layer_indices} if collect_activations else {}

        for i in range(batch_n):
            if collect_activations:
                acts = saved_batch[i]["activations"]
                if hasattr(acts, 'dtype') and acts.dtype == torch.bfloat16:
                    acts = acts.half()
                acts = acts.cpu().numpy()

                for j, layer_idx in enumerate(layer_indices):
                    layer_results[layer_idx].append(acts[j])
                    batch_vectors[layer_idx].append(acts[j])

            if collect_detailed:
                all_detailed.append({
                    "topk_ids": saved_batch[i]["topk_ids"].numpy().tolist(),
                    "topk_logprobs": saved_batch[i]["topk_vals"].numpy().tolist(),
                    "target_logprobs": saved_batch[i]["target_lps"].numpy().tolist(),
                })

        # ── Incremental save ─────────────────────────────────────────
        if save_dir is not None:
            batch_num = batch_offset + batch_start // batch_size

            if collect_activations:
                for layer_idx in layer_indices:
                    filename = f"{file_prefix}_layer{layer_idx:02d}_batch{batch_num:02d}.npy"
                    np.save(save_dir / filename, np.stack(batch_vectors[layer_idx]))

            if collect_detailed:
                # Save detailed predictions for this batch as JSON
                pred_filename = f"{file_prefix}_preds_batch{batch_num:02d}.json"
                batch_detailed = all_detailed[-batch_n:]
                with open(save_dir / pred_filename, "w") as f:
                    json.dump(batch_detailed, f)

        batch_display = batch_offset + batch_start // batch_size + 1
        total_batches = n_batches + batch_offset
        pad_range = f"pad=[{min(pad_offsets)}-{max(pad_offsets)}]"
        parts = []
        if collect_activations:
            parts.append(f"{n_layers} layers")
        if collect_detailed:
            parts.append(f"top-{top_k} preds")
        print(f"    batch {batch_display}/{total_batches}: "
              f"prompts {batch_start}-{batch_end-1} "
              f"({batch_n} prompts, {', '.join(parts)}) "
              f"{pad_range}", flush=True)

    # Stack each layer's results
    layer_data = {l: np.stack(vecs) for l, vecs in layer_results.items()} \
        if collect_activations else {}
    return layer_data, all_detailed


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
    parser.add_argument("--dataset", type=str, default=DEFAULT_TESTS_FILE,
                        help=f"Dataset JSON file in datasets/ (default: {DEFAULT_TESTS_FILE})")
    parser.add_argument("--skip-activations", action="store_true",
                        help="Skip layer activation collection")
    parser.add_argument("--skip-detailed-preds", action="store_true",
                        help="Skip detailed predictions collection (top-20, log-probs)")
    args = parser.parse_args()

    if args.skip_activations and args.skip_detailed_preds:
        print("Error: Cannot skip both activations and predictions")
        sys.exit(1)

    # Load data
    tests_file = BASE / "datasets" / args.dataset
    if not tests_file.exists():
        print(f"Error: Dataset file not found: {tests_file}")
        sys.exit(1)

    # Derive a short name for output files from the dataset name
    dataset_short = args.dataset.replace(".json", "").replace("_", "")

    with open(tests_file) as f:
        tests = json.load(f)
    with open(PUNS_FILE) as f:
        jokes = json.load(f)
    jokes_by_index = {j["index"]: j for j in jokes}

    # Check if pred_b is valid for this dataset (requires joke_b_index)
    if args.position == "pred_b" and tests and "joke_b_index" not in tests[0]:
        print(f"Error: --position pred_b requires dataset with joke_b_index field")
        print(f"  Dataset {args.dataset} appears to use 2-sentence format (no joke_b)")
        sys.exit(1)

    print(f"Loaded {len(tests)} cloze tests from {args.dataset}", flush=True)
    print(f"  (also loaded {len(jokes)} jokes from puns_205.json)", flush=True)

    # Initialize model
    print(f"\nInitializing model: {MODEL_NAME}", flush=True)
    model = LanguageModel(MODEL_NAME, device_map="auto")

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}", flush=True)

    # Select layers (if collecting activations)
    if args.skip_activations:
        layer_indices = []
        print(f"  Skipping activation collection", flush=True)
    elif args.layers:
        layer_indices = sorted(args.layers)
        print(f"  Collecting {len(layer_indices)} layers", flush=True)
    else:
        layer_indices = list(range(n_layers))
        print(f"  Collecting all {len(layer_indices)} layers", flush=True)

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
    # Handle different field naming: joke_c_sentence vs joke_sentence
    sentence_key = "joke_c_sentence" if "joke_c_sentence" in tests[0] else "joke_sentence"
    metadata = []
    for i, test in enumerate(tests):
        metadata.append({
            "index": i,
            "pair_id": test["pair_id"],
            "type": test["type"],
            "prompt": test["prompt"],
            "joke_c_sentence": test.get(sentence_key, test["prompt"]),
            "token_position": positions[i],
        })

    # Build file prefix - include dataset name if not default
    if args.dataset == DEFAULT_TESTS_FILE:
        file_prefix = f"{MODEL_SHORT}_{args.position}"
    else:
        file_prefix = f"{MODEL_SHORT}_{dataset_short}_{args.position}"

    meta_filename = f"{file_prefix}_meta.json"
    meta_out = {
        "model": MODEL_NAME,
        "model_short": MODEL_SHORT,
        "dataset": args.dataset,
        "position": args.position,
        "n_prompts": len(metadata),
        "n_layers_total": n_layers,
        "hidden_dim": hidden_dim,
        "system_prompt": SYSTEM_PROMPT,
        "file_shape": [len(metadata), hidden_dim],
        "file_axes": ["prompt", "hidden_dim"],
        "naming": f"{file_prefix}_layer{{NN}}.npy",
        "samples": metadata,
    }
    with open(OUTPUT_DIR / meta_filename, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"\nSaved metadata: {meta_filename}", flush=True)

    # ── Determine what to collect ─────────────────────────────────────────
    n_prompts = len(prompts_and_positions)
    n_batches = (n_prompts + args.batch_size - 1) // args.batch_size

    collect_activations = not args.skip_activations
    collect_detailed = not args.skip_detailed_preds

    # Check which layers still need collection (if collecting activations)
    needed_layers = []
    if collect_activations:
        for layer_idx in layer_indices:
            merged = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy"
            if merged.exists():
                existing = np.load(merged)
                if existing.shape[0] >= n_prompts:
                    continue
            needed_layers.append(layer_idx)

        if not needed_layers:
            print(f"\nAll {len(layer_indices)} layers already collected.")
            collect_activations = False

    # Check if detailed predictions already collected
    pred_file = OUTPUT_DIR / f"{file_prefix}_detailed_preds.json"
    if collect_detailed and pred_file.exists():
        with open(pred_file) as f:
            existing_preds = json.load(f)
        if existing_preds.get("n_prompts", 0) >= n_prompts:
            print(f"\nDetailed predictions already collected.")
            collect_detailed = False

    if not collect_activations and not collect_detailed:
        print(f"\nNothing to collect. Done.", flush=True)
        return

    # ── Build target token IDs for detailed predictions ─────────────────
    target_token_ids_per_prompt = None
    target_word_maps = None
    tokenizer = model.tokenizer

    if collect_detailed:
        pun_words_by_pair = {}
        straight_words_by_pair = {}
        for test in tests:
            pid = test["pair_id"]
            # Handle different field naming conventions
            if "expected_completion" in test:
                # Original format: expected_completion contains the relevant words
                words = [w.lower() for w in test["expected_completion"]]
                if test["type"] == "funny":
                    pun_words_by_pair[pid] = words
                else:
                    straight_words_by_pair[pid] = words
            else:
                # funny_serious format: punny_words and straight_words are explicit
                if "punny_words" in test:
                    pun_words_by_pair[pid] = [w.lower() for w in test["punny_words"]]
                if "straight_words" in test:
                    straight_words_by_pair[pid] = [w.lower() for w in test["straight_words"]]

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

        def word_to_token(word):
            toks = tokenizer.encode(" " + word, add_special_tokens=False)
            return toks[0] if toks else None

        target_token_ids_per_prompt = []
        target_word_maps = []
        for test in tests:
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

    # ── Check for partial batch files (resume support) ──────────────────
    existing_batches = 0
    remaining = prompts_and_positions
    remaining_targets = target_token_ids_per_prompt

    if collect_activations and needed_layers:
        sample_layer = needed_layers[0]
        for b in range(n_batches):
            bf = OUTPUT_DIR / f"{file_prefix}_layer{sample_layer:02d}_batch{b:02d}.npy"
            if bf.exists():
                existing_batches = b + 1
            else:
                break

        if existing_batches > 0:
            skip_prompts = existing_batches * args.batch_size
            print(f"\n  Found {existing_batches} batch files, "
                  f"resuming from prompt {skip_prompts}", flush=True)
            remaining = prompts_and_positions[skip_prompts:]
            if remaining_targets:
                remaining_targets = target_token_ids_per_prompt[skip_prompts:]

    # ── Show collection plan ────────────────────────────────────────────
    parts = []
    if collect_activations and needed_layers:
        parts.append(f"{len(needed_layers)} layers")
    if collect_detailed:
        parts.append("top-20 + target log-probs")

    print(f"\n--- Collecting {args.position}: {', '.join(parts)} x "
          f"{len(remaining)} prompts ---", flush=True)

    if collect_activations and needed_layers:
        print(f"  Estimated activation download: "
              f"{len(needed_layers) * len(remaining) * hidden_dim * 2 / 1e6:.0f} MB",
              flush=True)

    # ── Run unified collection ──────────────────────────────────────────
    layer_data, detailed_raw = collect_batch(
        model, layers_module, remaining,
        layer_indices=needed_layers if collect_activations else None,
        batch_size=args.batch_size, remote=True,
        save_dir=OUTPUT_DIR, file_prefix=file_prefix,
        batch_offset=existing_batches,
        target_token_ids_per_prompt=remaining_targets,
        top_k=20,
    )

    # ── Post-process activations ────────────────────────────────────────
    if collect_activations and needed_layers:
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

    # ── Post-process detailed predictions ───────────────────────────────
    if collect_detailed and detailed_raw:
        # Merge batch files if resuming
        all_detailed_raw = []
        for b in range(n_batches):
            pf = OUTPUT_DIR / f"{file_prefix}_preds_batch{b:02d}.json"
            if pf.exists():
                with open(pf) as f:
                    all_detailed_raw.extend(json.load(f))
                pf.unlink()

        # Assemble into structured JSON
        detailed = []
        for i, (raw, wmap) in enumerate(zip(all_detailed_raw, target_word_maps)):
            test = tests[i]
            top_tokens = []
            for tid, lp in zip(raw["topk_ids"], raw["topk_logprobs"]):
                word = tokenizer.decode([tid]).strip()
                top_tokens.append({
                    "token_id": tid, "word": word,
                    "logprob": round(lp, 4),
                    "prob": round(float(np.exp(lp)), 6),
                })

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

            # Get top-1 prediction info for metadata
            top1_id = raw["topk_ids"][0]
            top1_word = tokenizer.decode([top1_id]).strip().lower()
            pun_words = list(pun_probs.keys())
            is_pun_pred = top1_word in pun_words

            meta_out["samples"][i]["predicted_token_id"] = top1_id
            meta_out["samples"][i]["predicted_word"] = top1_word
            meta_out["samples"][i]["predicted_funny"] = is_pun_pred

            detailed.append({
                "index": i,
                "pair_id": test["pair_id"],
                "type": test["type"],
                "top_tokens": top_tokens,
                "pun_word_probs": pun_probs,
                "straight_word_probs": straight_probs,
                "eot": eot_prob,
            })

        # Save detailed predictions
        with open(pred_file, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "position": args.position,
                "top_k": 20,
                "n_prompts": len(detailed),
                "results": detailed,
            }, f, indent=2)
        print(f"  Saved detailed predictions: {pred_file.name}", flush=True)

        # Update metadata with prediction info
        with open(OUTPUT_DIR / meta_filename, "w") as f:
            json.dump(meta_out, f, indent=2)

        # Summary stats
        total_pun_prob_funny = []
        total_pun_prob_straight = []
        n_pun_pred = 0
        for d in detailed:
            pun_total = sum(v["prob"] for v in d["pun_word_probs"].values())
            if d["type"] == "funny":
                total_pun_prob_funny.append(pun_total)
            else:
                total_pun_prob_straight.append(pun_total)
            # Check if top-1 is a pun word
            top1 = d["top_tokens"][0]["word"].lower()
            if top1 in d["pun_word_probs"]:
                n_pun_pred += 1

        print(f"  {n_pun_pred}/{len(detailed)} predicted the pun word", flush=True)
        if total_pun_prob_funny and total_pun_prob_straight:
            print(f"  Mean P(pun words): funny ctx={np.mean(total_pun_prob_funny):.4f}, "
                  f"straight ctx={np.mean(total_pun_prob_straight):.4f}", flush=True)

    print(f"\nDone.", flush=True)


# ── Notebook-friendly collection ─────────────────────────────────────────────

def ensure_activations(dataset_file, position="pred_c", layers=None,
                       collect_predictions=True, display_progress=True):
    """
    Ensure activations (and optionally predictions) are collected for a dataset.

    If data already exists, returns immediately. Otherwise, runs collection
    with progress display suitable for Jupyter notebooks.

    Parameters:
        dataset_file: str or Path — dataset JSON file (relative to datasets/)
        position: str — token position ("pred_c" or "pred_b")
        layers: list of int — specific layers to collect (None = all layers)
        collect_predictions: bool — also collect detailed predictions
        display_progress: bool — show progress during collection

    Returns:
        dict with keys:
            meta_file: Path to metadata JSON
            pred_file: Path to predictions JSON (or None)
            already_existed: bool — True if data was already present
            n_prompts: int — number of prompts
            n_layers: int — number of layers collected

    Example:
        from collect_activations import ensure_activations
        result = ensure_activations("funny_serious_150.json")
        print(f"Meta file: {result['meta_file']}")
    """
    from IPython.display import clear_output, display
    import time

    dataset_path = Path(dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = BASE / "datasets" / dataset_file

    # Determine output filenames
    dataset_stem = dataset_path.stem.lower().replace("-", "").replace("_", "")
    file_prefix = f"{MODEL_SHORT}_{dataset_stem}_{position}"
    meta_file = OUTPUT_DIR / f"{file_prefix}_meta.json"
    pred_file = OUTPUT_DIR / f"{file_prefix}_detailed_preds.json"

    # Check if data already exists
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        n_prompts = meta.get("n_prompts", 0)
        n_layers = meta.get("n_layers_total", 0)

        # Check if all layer files exist
        naming = meta.get("naming", "")
        all_present = True
        for layer_idx in range(n_layers):
            layer_file = OUTPUT_DIR / naming.replace("{NN}", f"{layer_idx:02d}")
            if not layer_file.exists():
                all_present = False
                break

        if all_present:
            if display_progress:
                print(f"Data already exists: {meta_file.name}")
                print(f"  {n_prompts} prompts, {n_layers} layers")
            return {
                "meta_file": meta_file,
                "pred_file": pred_file if pred_file.exists() else None,
                "already_existed": True,
                "n_prompts": n_prompts,
                "n_layers": n_layers,
            }

    # Need to collect data
    if display_progress:
        print(f"Collecting activations for {dataset_path.name}...")
        print(f"  This requires NDIF (remote GPU inference)")

    # Load dataset
    with open(dataset_path) as f:
        tests = json.load(f)
    n_prompts = len(tests)

    # Initialize model
    if display_progress:
        print(f"  Initializing model: {MODEL_NAME}")

    model = LanguageModel(MODEL_NAME, device_map="auto", dispatch=True)
    tokenizer = model.tokenizer
    n_layers_total = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    if layers is None:
        layer_indices = list(range(n_layers_total))
    else:
        layer_indices = sorted(layers)

    if display_progress:
        print(f"  Layers: {len(layer_indices)}, Hidden dim: {hidden_dim}")

    # Determine layer module path
    layers_module = model.model.layers

    # Build prompts and positions
    prompts_and_positions = []
    target_token_ids_per_prompt = [] if collect_predictions else None

    for test in tests:
        prompt = test.get("prompt", "")
        if not prompt:
            # Build prompt from components
            if "joke_a_sentence" in test:
                # 3-sentence format
                A = fill_sentence(test["joke_a_sentence"],
                    test["punny_words"][0] if test["type"] == "funny" else test["straight_words"][0])
                B = fill_sentence(test["joke_b_sentence"],
                    test["punny_words"][1] if test["type"] == "funny" and len(test.get("punny_words", [])) > 1
                    else test["straight_words"][1] if len(test.get("straight_words", [])) > 1 else "")
                C = truncate_before_blank(test["joke_c_sentence"])
                prompt = f"{A} {B} {C}"
            else:
                prompt = test.get("prompt", "")

        formatted = format_chat_prompt(tokenizer, prompt)
        pos = find_pred_c_position(tokenizer, formatted, prompt)
        prompts_and_positions.append((formatted, pos))

        if collect_predictions:
            # Get target token IDs for this prompt
            target_words = []
            pun_words = test.get("punny_words", test.get("expected_completion", []))
            straight_words = test.get("straight_words", test.get("contrast_completion", []))
            if isinstance(pun_words, str):
                pun_words = [pun_words]
            if isinstance(straight_words, str):
                straight_words = [straight_words]

            target_ids = []
            for w in pun_words + straight_words:
                tid = tokenizer.encode(" " + w, add_special_tokens=False)
                if tid:
                    target_ids.append(tid[0])
            target_token_ids_per_prompt.append(target_ids)

    # Prepare metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples_meta = []
    for i, test in enumerate(tests):
        samples_meta.append({
            "index": i,
            "pair_id": test["pair_id"],
            "type": test["type"],
            "joke_c_sentence": test.get("joke_c_sentence", test.get("prompt", "")),
        })

    meta_out = {
        "model": MODEL_NAME,
        "dataset": dataset_path.name,
        "position": position,
        "n_prompts": n_prompts,
        "n_layers_total": n_layers_total,
        "hidden_dim": hidden_dim,
        "naming": f"{file_prefix}_layer{{NN}}.npy",
        "samples": samples_meta,
    }

    with open(meta_file, "w") as f:
        json.dump(meta_out, f, indent=2)

    if display_progress:
        print(f"  Collecting {len(layer_indices)} layers × {n_prompts} prompts...")

    # Run collection with progress
    batch_size = 10
    n_batches = (n_prompts + batch_size - 1) // batch_size

    layer_data, detailed_raw = collect_batch(
        model, layers_module, prompts_and_positions,
        layer_indices=layer_indices,
        batch_size=batch_size,
        remote=True,
        save_dir=OUTPUT_DIR,
        file_prefix=file_prefix,
        target_token_ids_per_prompt=target_token_ids_per_prompt,
        top_k=20,
    )

    # Merge batch files into single layer files
    if display_progress:
        print("  Merging batch files...")

    for layer_idx in layer_indices:
        parts = []
        for b in range(n_batches):
            bf = OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}_batch{b:02d}.npy"
            if bf.exists():
                parts.append(np.load(bf))
                bf.unlink()  # Remove batch file
        if parts:
            merged = np.concatenate(parts, axis=0)
            np.save(OUTPUT_DIR / f"{file_prefix}_layer{layer_idx:02d}.npy", merged)

    # Process detailed predictions if collected
    if collect_predictions and detailed_raw:
        if display_progress:
            print("  Processing predictions...")

        # Merge prediction batch files
        all_preds = []
        for b in range(n_batches):
            pf = OUTPUT_DIR / f"{file_prefix}_preds_batch{b:02d}.json"
            if pf.exists():
                with open(pf) as f:
                    all_preds.extend(json.load(f))
                pf.unlink()

        # Build detailed predictions JSON
        detailed = []
        for i, (test, raw) in enumerate(zip(tests, all_preds)):
            pun_words = test.get("punny_words", test.get("expected_completion", []))
            straight_words = test.get("straight_words", test.get("contrast_completion", []))
            if isinstance(pun_words, str):
                pun_words = [pun_words]
            if isinstance(straight_words, str):
                straight_words = [straight_words]

            top_tokens = []
            for tid, lp in zip(raw["topk_ids"], raw["topk_logprobs"]):
                word = tokenizer.decode([tid]).strip()
                top_tokens.append({
                    "token_id": tid, "word": word,
                    "logprob": round(lp, 4),
                    "prob": round(float(np.exp(lp)), 6),
                })

            # Match target logprobs to words
            pun_probs = {}
            straight_probs = {}
            all_target_words = pun_words + straight_words
            for j, word in enumerate(all_target_words):
                if j < len(raw["target_logprobs"]):
                    lp = raw["target_logprobs"][j]
                    tid = tokenizer.encode(" " + word, add_special_tokens=False)
                    entry = {
                        "token_id": tid[0] if tid else 0,
                        "logprob": round(lp, 4),
                        "prob": round(float(np.exp(lp)), 6),
                    }
                    if j < len(pun_words):
                        pun_probs[word] = entry
                    else:
                        straight_probs[word] = entry

            detailed.append({
                "index": i,
                "pair_id": test["pair_id"],
                "type": test["type"],
                "top_tokens": top_tokens,
                "pun_word_probs": pun_probs,
                "straight_word_probs": straight_probs,
            })

        with open(pred_file, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "position": position,
                "top_k": 20,
                "n_prompts": len(detailed),
                "results": detailed,
            }, f, indent=2)

    if display_progress:
        print(f"  Done! Saved to {OUTPUT_DIR.name}/")

    return {
        "meta_file": meta_file,
        "pred_file": pred_file if pred_file.exists() else None,
        "already_existed": False,
        "n_prompts": n_prompts,
        "n_layers": len(layer_indices),
    }


if __name__ == "__main__":
    main()
