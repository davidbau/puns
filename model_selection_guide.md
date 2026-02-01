# Selecting Models for the Pun Cloze Benchmark

This guide walks through how we chose which language models to benchmark
on our contextual cloze pun tests, and the technical/strategic reasoning
behind each step.

## Goal

We want to test whether language models can distinguish between "straight"
(literal) and "funny" (punny) completions when primed with context.
Crucially, we also want to later examine **model internals** using
[NDIF](https://ndif.us/) (National Deep Inference Fabric) and the
[nnsight](https://nnsight.net/) library, so we need models that are
available on both a fast inference API (for benchmarking) and NDIF
(for interpretability).

## Step 1: Check what's hot on NDIF

NDIF keeps certain models loaded ("hot") on GPU, ready for immediate
inference and intervention. Cold models can be loaded on demand but
take time to spin up.

```bash
curl -s https://api.ndif.us/status | python3 -c "
import json, sys
data = json.load(sys.stdin)
for key, model in data['deployments'].items():
    if model.get('deployment_level') == 'HOT' and model.get('application_state') == 'RUNNING':
        print(f\"{model['repo_id']:45s}  params={model.get('n_params',0)/1e9:.1f}B\")
"
```

As of this writing, the NDIF hot models are:

| Model | Parameters | Type |
|-------|-----------|------|
| openai-community/gpt2 | 0.1B | Base |
| EleutherAI/gpt-j-6b | 6.1B | Base |
| meta-llama/Llama-2-7b-hf | 6.7B | Base |
| google/codegemma-7b-it | 8.5B | Instruct |
| meta-llama/Llama-3.1-8B | 8.0B | Base |
| google/gemma-2-9b-it | 9.2B | Instruct |
| meta-llama/Llama-3.1-70B | 70.6B | Base |
| meta-llama/Llama-3.1-70B-Instruct | 70.6B | Instruct |
| meta-llama/Llama-3.3-70B-Instruct | 70.6B | Instruct |
| meta-llama/Llama-3.1-405B-Instruct | 405.9B | Instruct |

This gives us a good range: tiny (GPT-2, 0.1B) through massive (405B),
covering both base models (which do raw text completion) and
instruction-tuned models (which follow chat-style prompts).

## Step 2: Find a fast inference provider

For benchmarking 100 cloze tests across multiple models, we need a
hosted API with fast turnaround. We use [Together.ai](https://www.together.ai)
because it offers an OpenAI-compatible API with a wide selection of
open-source models at low cost.

```bash
# List all available models on Together.ai
curl -s -H "Authorization: Bearer $TOGETHER_API_KEY" \
  https://api.together.xyz/v1/models | python3 -c "
import json, sys
for m in json.load(sys.stdin):
    print(f\"{m['id']:55s}  type={m['type']}\")
"
```

Together.ai classifies models as:
- **chat**: instruction-tuned, accessed via `/v1/chat/completions`
- **language**: base models, accessed via `/v1/completions`
- **image**, **audio**, etc.: not relevant here

## Step 3: Match NDIF models to Together.ai

This is where it gets interesting. Not every NDIF model has an exact
match on Together.ai.

| NDIF Hot Model | Together.ai Model | Match Quality |
|---|---|---|
| meta-llama/Llama-3.1-8B | Meta-Llama-3.1-8B-Instruct-Turbo | Instruct variant only |
| meta-llama/Llama-3.1-70B | Meta-Llama-3.1-70B-Instruct-Turbo | Instruct variant only |
| meta-llama/Llama-3.1-70B-Instruct | Meta-Llama-3.1-70B-Instruct-Turbo | Close (Turbo = optimized) |
| meta-llama/Llama-3.3-70B-Instruct | Llama-3.3-70B-Instruct-Turbo | Close (Turbo = optimized) |
| meta-llama/Llama-3.1-405B-Instruct | Meta-Llama-3.1-405B-Instruct-Turbo | Close (Turbo = optimized) |
| openai-community/gpt2 | *not available* | -- |
| EleutherAI/gpt-j-6b | *not available* | -- |
| meta-llama/Llama-2-7b-hf | *not available* | -- |
| google/gemma-2-9b-it | *not available* | -- |
| google/codegemma-7b-it | *not available* | -- |

**Key observations:**

1. **Together.ai is mostly instruct models.** They only offer one base
   ("language") model: `meta-llama/Llama-3-70b-hf`. The smaller base
   models (GPT-2, GPT-J, Llama-2-7B) are not available.

2. **"Turbo" variants** are Together's optimized serving of the same
   weights. For behavioral benchmarking they should produce identical
   or near-identical outputs. For interpretability on NDIF, we'll use
   the original non-Turbo weights.

3. **Base vs. Instruct gap.** NDIF has `Llama-3.1-8B` (base) and
   `Llama-3.1-70B` (base) hot, but Together only has the instruct
   versions. Base models complete text directly; instruct models
   follow instructions. For our cloze test, we adapt the approach:
   instruct models get a system prompt saying "output only the next word."

## Step 4: Choose the benchmark model set

Given the constraints, we selected these models for the Together.ai
benchmark, spanning 1B to 405B parameters:

| Nickname | Together.ai Model | NDIF Counterpart |
|----------|-------------------|------------------|
| 1b | Llama-3.2-1B-Instruct | Llama-3.2-1B (NDIF cold) |
| 3b | Llama-3.2-3B-Instruct-Turbo | Llama-3.2-3B (NDIF cold) |
| 8b | Meta-Llama-3.1-8B-Instruct-Turbo | Llama-3.1-8B (NDIF hot) |
| 70b | Meta-Llama-3.1-70B-Instruct-Turbo | Llama-3.1-70B (NDIF hot) |
| 3.3-70b | Llama-3.3-70B-Instruct-Turbo | Llama-3.3-70B-Instruct (NDIF hot) |
| 405b | Meta-Llama-3.1-405B-Instruct-Turbo | Llama-3.1-405B-Instruct (NDIF hot) |

We included the 1B and 3B models (NDIF cold, loadable on demand) to see
whether there's a minimum scale threshold for pun comprehension.

## Step 5: Design the evaluation

For instruct models, each cloze prompt is sent as a chat message with a
system prompt:

> "You are completing a text. Given the text below, output only the
> single next word that best continues it. Output ONLY that one word,
> nothing else -- no punctuation, no explanation."

We use `temperature=0` for deterministic output and `max_tokens=10`
with stop sequences for newlines and punctuation to get a clean single word.

The model's first word is extracted and compared against:
- **expected_completion**: the words matching the context condition
  (straight words after straight priming, funny after funny)
- **contrast_completion**: the words from the opposite condition

This gives us three outcomes per test:
- **Correct**: model produced a word from the expected list
- **Contrast**: model produced a word from the *wrong* condition's list
- **Other**: model produced an unrelated word

## Limitations and next steps

1. **Instruct bias.** Using instruct models with a "complete this text"
   prompt is not the same as raw text completion with a base model.
   The instruction-following layer may override subtle contextual priming.
   NDIF testing with base models (GPT-2, GPT-J, Llama-3.1-8B base)
   will provide a cleaner signal.

2. **Single-token evaluation.** We only check the first generated word.
   Some expected completions are multi-word ("stop traffic"). We handle
   this by matching the first word of multi-word expectations.

3. **Logprob analysis.** A richer evaluation would examine the probability
   distribution over tokens, not just the argmax. Together.ai supports
   `logprobs` in the API, and NDIF/nnsight gives full access to the
   model's internal probability distribution. This is a natural next step.

4. **Small base models on NDIF.** The most interesting interpretability
   targets (GPT-2, GPT-J, Llama-3.1-8B base) aren't available on
   Together.ai. These need to be tested directly on NDIF using nnsight,
   where we can also examine attention patterns and residual stream
   activations to understand *how* models process pun contexts.
