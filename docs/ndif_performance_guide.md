# Performance-Oriented Programming with NDIF and nnsight

A practical guide to writing efficient, robust programs that collect
neural network activation data from remote models via NDIF.  Written
for both human researchers and LLM agents who need to get up to speed
quickly.

---

## 1. The Central Constraint: `.save()` Transfers Data Over the Internet

When you write nnsight code that runs on NDIF, your Python trace code is
serialized, sent to a remote GPU cluster, executed there, and then the
results are sent back to you.  The key method that bridges this gap is
**`.save()`**.

Think of `.save()` as analogous to PyTorch's `.cpu()` — except instead
of moving data from GPU memory to CPU memory (nanoseconds, ~100 GB/s),
it moves data from a remote server to your local machine over the
internet (seconds, ~10–100 MB/s).  This makes `.save()` the single
most performance-critical call in almost any nnsight program.

```python
# Inside a trace, .save() marks a value for transfer back to the client.
# Nothing is transferred until the trace exits.
with model.trace(remote=True) as tracer:
    with tracer.invoke("The cat sat on the"):
        hidden = model.model.layers[40].output
        result = hidden[0, -1, :].save()   # <-- this gets shipped back to you

# After the trace, `result` holds an actual tensor on your local machine.
print(result.shape)  # torch.Size([8192])
```

**Rule of thumb:** Every byte you `.save()` must travel over the
internet.  Treat bandwidth as a scarce resource.

---

## 2. Minimize Bandwidth: Reduce Data on the Server Before Saving

The most common performance mistake is saving too much data.  Since
nnsight lets you run arbitrary tensor operations inside a trace, you
should do as much slicing, indexing, aggregating, and sparsifying
**on the server** as possible, and only `.save()` the compact result.

### Bad: Saving full layer outputs and slicing locally

```python
# BAD — transfers the FULL output tensor for every layer
with model.trace(remote=True) as tracer:
    with tracer.invoke(prompt):
        saved_outputs = []
        for i in range(80):
            full_output = model.model.layers[i].output.save()  # (1, seq_len, 8192)
            saved_outputs.append(full_output)

# Then slicing locally — but you already paid for the full transfer!
vectors = [out[0, target_pos, :] for out in saved_outputs]
```

This transfers **80 x seq_len x 8192 x 2 bytes** — potentially over a
gigabyte per prompt.

### Good: Slice on the server, stack into one tensor, save once

```python
# GOOD — slice and stack on the server, transfer only what you need
with model.trace(remote=True) as tracer:
    with tracer.invoke(prompt):
        embs = []
        for i in range(80):
            out = model.model.layers[i].output  # server-side, not transferred
            embs.append(out[0, target_pos, :])  # (8192,) small slice
        result = torch.stack(embs).save()  # (80, 8192) — one compact tensor

# result is (80, 8192) = 1.3 MB in float16, vs >1 GB for full outputs
```

The key insight: **remote operations are cheap** — they execute on the
server.  Only `.save()` triggers a data transfer.  So do your indexing,
stacking, reshaping, and even arithmetic on proxies before saving.

### Saving non-tensor data

nnsight adds `.save()` to all Python objects within a trace context,
not just tensors.  You can save dictionaries, lists, and scalar values:

```python
with model.trace(remote=True) as tracer:
    with tracer.invoke(prompt):
        logits = model.lm_head.output[0, target_pos, :]
        top_val, top_idx = logits.max(-1)

        # Save a structured result — one compact transfer
        info = {
            "predicted_token": top_idx,
            "confidence": top_val,
            "top5_tokens": logits.topk(5).indices,
        }
        saved_info = info.save()
```

This lets you do substantial analysis on the server and only send back
the specific results you need.

### Use float16 when you can

Model activations are typically computed in float16 or bfloat16.
Converting to float32 before saving doubles your bandwidth for no
benefit if you only need float16 precision:

```python
# Unnecessary — doubles transfer size
vec = hidden[0, pos, :].float().save()   # float32: 32 KB per vector

# Better — keep native precision
vec = hidden[0, pos, :].save()           # float16: 16 KB per vector
```

When saving to .npy files later, `numpy` handles float16 natively.
Only convert to float32 if your downstream analysis specifically
requires it.

---

## 3. Batching: Right-Aligned Padding and Throughput

### How batching works in nnsight

When you issue multiple `tracer.invoke()` calls within a single
`model.trace()` context, nnsight batches them into one remote
execution.  Because different prompts have different lengths, nnsight
**left-pads** shorter prompts by default so that all sequences are
right-aligned to the length of the longest prompt in the batch.

```
Prompt A (12 tokens):  [pad] [pad] [pad] [tok] [tok] [tok] ... [tok]
Prompt B (15 tokens):  [tok] [tok] [tok] [tok] [tok] [tok] ... [tok]
Prompt C (10 tokens):  [pad] [pad] [pad] [pad] [pad] [tok] ... [tok]
                                                              ^
                                    sequences are RIGHT-ALIGNED here
```

That means **token position indices must be adjusted** to account for
padding.  If you want position `p` in the original prompt, you need
position `p + (max_len - this_prompt_len)` in the padded batch:

```python
token_lengths = [
    len(model.tokenizer.encode(prompt))
    for prompt, pos in batch
]
max_len = max(token_lengths)

with model.trace(remote=True) as tracer:
    for (prompt, pos), length in zip(batch, token_lengths):
        pad_offset = max_len - length
        adjusted_pos = pos + pad_offset
        with tracer.invoke(prompt):
            emb = model.model.layers[40].output[0, adjusted_pos, :]
            ...
```

If you forget the padding adjustment, you'll silently read from the
wrong position (often a padding token), producing garbage data that
looks superficially reasonable.

### Choosing batch sizes

Batching involves a three-way trade-off:

| Concern | Small batches | Large batches |
|---------|--------------|---------------|
| Latency overhead | High (many round-trips) | Low (fewer round-trips) |
| Download size | Small, fast to transfer | Large, slow to transfer |
| Incremental analysis | Easy (partial data quickly) | Harder (wait for big download) |
| Memory on server | Low | High (may OOM) |

**Recommended sweet spot:** Aim for batches that produce **a few dozen
megabytes** of saved data.  For example, collecting 80 layers of 8192-dim
activations at one position per prompt:

- 10 prompts x 80 layers x 8192 x 2 bytes (float16) = **13 MB** per batch
- This is small enough to transfer in seconds and easy to save incrementally

If you're collecting fewer layers or smaller models, you can increase
the batch size.  If you're collecting more data per prompt (e.g.,
multiple positions, full sequence activations), decrease it.

---

## 4. Architectural Design: Collection vs. Analysis

A robust activation study has two distinct programs:

### Program 1: Data Collection (nnsight + NDIF)

This program runs nnsight traces against NDIF to collect data.
It may run for hours and make hundreds of requests, so it must be
**robust and resumable**.

**Key design principles:**

#### Save incrementally, in small files

Save each batch's results immediately after receiving them.  Don't
accumulate everything in memory and save at the end — if the program
crashes after batch 97 of 100, you'll lose everything.

```python
# Save one file per batch (or per-layer per-batch for multi-layer data)
for batch_num, batch in enumerate(batches):
    results = run_nnsight_trace(model, batch)

    for layer_idx, data in results.items():
        filename = f"{prefix}_layer{layer_idx:02d}_batch{batch_num:02d}.npy"
        np.save(output_dir / filename, data)

    print(f"Batch {batch_num + 1}/{n_batches} saved", flush=True)
```

Use simple, self-describing file formats: `.npy` for single arrays,
`.npz` for collections of arrays.  These are fast to read/write and
universally supported.

#### Make it resumable

Before collecting a batch, check if its output files already exist.
This lets you restart after a crash, timeout, or interruption without
re-downloading data you already have:

```python
# Check which batches are already done
existing_batches = set()
for b in range(n_batches):
    path = output_dir / f"{prefix}_layer{first_layer:02d}_batch{b:02d}.npy"
    if path.exists():
        existing_batches.add(b)

# Only collect missing batches
for batch_num in range(n_batches):
    if batch_num in existing_batches:
        print(f"Batch {batch_num} already exists, skipping")
        continue
    # ... collect and save this batch
```

Note: check for existence of **individual batch files** rather than
counting sequentially from zero.  Sequential counting breaks if a
middle batch failed — it would skip everything after the gap.

#### Handle transient failures gracefully

NDIF requests can fail due to timeouts, network issues, or server
load.  Don't let one failed batch abort the entire collection:

```python
for batch_num in range(n_batches):
    if batch_num in existing_batches:
        continue
    try:
        results = run_nnsight_trace(model, batches[batch_num])
        save_batch(results, batch_num)
    except Exception as e:
        print(f"Batch {batch_num} failed: {e} — will retry on next run")
        continue
```

On the next run, the resume logic picks up any failed batches
automatically.

#### Save metadata alongside data

Write a JSON metadata file that describes the collection: model name,
prompt details, what each sample represents, file naming conventions,
and any per-sample annotations.  This makes the data self-documenting
and consumable by the analysis pipeline:

```python
metadata = {
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "n_prompts": 100,
    "n_layers": 80,
    "hidden_dim": 8192,
    "naming": "activations_layer{NN}_batch{BB}.npy",
    "samples": [
        {"index": 0, "prompt": "...", "condition": "control", ...},
        ...
    ],
}
with open(output_dir / "meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### Program 2: Analysis and Visualization (pure numpy/matplotlib)

This program loads the saved .npy files and metadata, computes
statistics, and produces plots.  It should have **no dependency on
nnsight or NDIF** — it works entirely with local data.

**Key design principles:**

#### Handle partial data gracefully

During a long collection, you'll want to inspect intermediate results.
The analysis pipeline should work with whatever data is available:

```python
def load_activations(meta_file):
    """Load all available batch files, even if collection is incomplete."""
    # ... load merged files if they exist ...
    # ... otherwise concatenate available batch files ...
    # ... trim metadata to match actual row count ...
    if n_rows < len(meta["samples"]):
        meta["samples"] = meta["samples"][:n_rows]
    return meta, layer_data
```

This means you can run the same analysis code on:
- A tiny test sample (2 prompts) for debugging
- Partial data (30 of 100 prompts) for early insight
- The full dataset when collection completes

#### Separate analysis from visualization

Keep pure-numpy analysis functions (projections, metrics, statistics)
separate from plotting code.  This makes the analysis importable for
notebooks, scripts, or further programmatic use:

```
analyze.py    — pure numpy, no plotting dependencies
visualize.py  — imports from analyze.py, uses matplotlib
```

It is also a common pattern to use an ipython notebook for the
visualization code, so that results can be viewed interatively.
The notebook could also import and call methods from the
analysis or data collection scripts to allow the user to
understand and drive the whole pipeine from the notebook.

---

## Summary: The Five Rules

1. **`.save()` = internet transfer.**  Treat it like shipping a
   package — minimize weight and frequency.

2. **Compute on the server, save the result.**  Use proxy operations
   (indexing, stacking, aggregation, arithmetic) to reduce data before
   calling `.save()`.

3. **Mind the padding.**  nnsight left-pads batched prompts.  Always
   adjust token positions by `(max_len - prompt_len)`.

4. **Target a few dozen MB per batch.**  This balances throughput
   (fewer round-trips) against practical download speed and incremental
   analyzability.

5. **Design for resilience.**  Save incrementally, resume from
   failures, and build analysis that works on partial data.
