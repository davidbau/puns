# Creating the Puns 205 Dataset

## Overview

`puns_205.json` is a curated dataset of 205 fill-in-the-blank pun sentences designed
for **contextual cloze testing** — measuring whether language models can be primed toward
straight or punny completions depending on the surrounding context.

Each entry contains a sentence with a blank (`___`) that admits both a literal ("straight")
completion and a pun ("punny") completion. The dataset includes model-derived completion
frequencies, classifications, and discriminativeness tiers that characterize how ambiguous
each joke is to language models.

The final benchmark artifact is `contextual_cloze_tests_100.json`: 100 contrastive test
prompts (50 pairs) where each pair tests the same target joke under straight-primed and
funny-primed context.

## Stage 1: Raw Collection & Rating

**Scripts:** `rawdata/analyze_jokes.py`, `rawdata/repair_jokes.py`, `rawdata/apply_edits.py`

Starting from a pool of ~1,000 candidate cloze pun sentences (`rawdata/raw_samples.json`),
three LLM providers rated each joke on quality (0-10):

- **Gemini** — rated in batches of 25 via `analyze_jokes.py`
- **Anthropic (Claude)** — rated in the same batch pipeline
- **Claude Code** — additional rating pass for consensus scoring

Low-scoring jokes (avg < 7.0) were sent to Claude Sonnet for repair via `repair_jokes.py`,
which attempted to improve the pun while preserving the fill-in-the-blank structure.
Hand edits were applied via `apply_edits.py` to remove duplicates and fix sentence wording.

## Stage 2: Building puns_205.json

**Script:** `rawdata/build_puns_205.py`

The top 205 jokes were selected by merging ratings from all three providers:

1. Merge `top205_full.json` with `annotated_205_gemini.json` and `annotated_205_anthropic.json`
2. Compute `consensus_score = gemini_rating + claudecode_rating`
3. Sort by consensus_score DESC, then anthropic_rating DESC, then sentence length DESC
4. Assign sequential `index` (0-204) and preserve original `id`

Output fields at this stage: `index`, `id`, `sentence`, `straight`, `punny`, `explanation`,
`gemini_rating`, `anthropic_rating`, `claudecode_rating`, `consensus_score`.

## Stage 3: Completion Collection

**Script:** `collect_completions.py` (phase: collect) / `annotate_completions.py` (phase: collect)

For each of the 205 jokes, the sentence is truncated before `___` and sent to 5 Llama
models via Together.ai:

| Nickname | Model ID |
|----------|----------|
| 3b | meta-llama/Llama-3.2-3B-Instruct-Turbo |
| 8b | meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo |
| 70b | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |
| 3.3-70b | meta-llama/Llama-3.3-70B-Instruct-Turbo |
| 405b | meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo |

Per model per joke:
- **20 diverse samples** at temperature=0.8, n=20
- **1 greedy sample** at temperature=0

Total: 205 jokes x 5 models x (20+1) = 21,525 completions.
After extracting the first word and lowercasing, results are aggregated into frequency
counts across all 100 diverse samples per joke.

**Checkpoint:** `raw_completions.json`

## Stage 4: Classification

**Script:** `collect_completions.py` (phase: classify) / `annotate_completions.py` (phase: classify)

For each joke, candidate words not already in the known `straight` or `punny` lists are
classified by Claude Sonnet into three categories:

- **straight** — a sensible literal completion without wordplay
- **funny** — exploits a double meaning or pun in context
- **other** — doesn't fit well, is a filler word, or is nonsensical

Classification is done in batches of 10 jokes per API call. Known words are
pre-classified: existing straight words → "straight", existing punny words → "funny".

**Checkpoint:** `classified_completions.json`

## Stage 5: Discriminative Ranking

**Script:** `rank_discriminative.py` / `annotate_completions.py` (phase: rank)

For each joke, the 100 model samples are tallied by classification to compute:

- `p_straight` = fraction of samples classified as straight
- `p_funny` = fraction classified as funny
- `p_other` = fraction classified as other
- `balance = min(p_straight, p_funny) / max(p_straight, p_funny)`
  - 1.0 = perfectly split between straight and funny
  - 0.0 = completely one-sided

Each joke is assigned a **tier** based on these thresholds:

| Tier | Condition | Count |
|------|-----------|-------|
| `balanced` | balance >= 0.3 and p_funny >= 5% | 42 |
| `leaning` | 0.05 < balance < 0.3 and p_funny >= 5% | 51 |
| `funny_dominated` | p_funny > 50% and balance <= 0.05 | 24 |
| `straight_dominated` | p_funny < 5% (everything else) | 88 |

Per-model funny rates are also computed, showing which models are more likely to produce
pun completions.

## Stage 6: Contrastive Test Construction

**Script:** `build_cloze_tests.py`

The test builder uses **tier-aware triple selection** to ensure target jokes (C) are
genuinely ambiguous while context jokes (A, B) are one-sided:

```
1. Partition 205 jokes by cloze_tier
2. Sort leaning jokes by cloze_balance DESC
3. Split leaning: top 8 → C pool, next 12 → AB pool
4. C_pool  = balanced(42) + leaning_for_C(8)   = 50 target jokes
   AB_pool = straight_dominated(88) + leaning_for_AB(12) = 100 context jokes
5. Shuffle both pools (seed=42)
6. Form 50 triples: A = AB[2i], B = AB[2i+1], C = C_pool[i]
7. For each triple, emit 2 tests (straight-primed and funny-primed)
```

**Prefill logic:**
- Straight prefill: `joke["straight"][0]` (highest-frequency straight word)
- Funny prefill: `joke["punny"][0]` (highest-frequency pun word)
- Context: `fill(A, word) + " " + fill(B, word) + " " + truncate(C)`

The 24 funny-dominated jokes and 31 remaining leaning jokes are excluded from tests —
funny-dominated jokes don't work as straight context, and the remaining leaning jokes
are neither balanced enough for targets nor one-sided enough for context.

**Output:** `contextual_cloze_tests_100.json` (100 entries = 50 pairs x 2)

## Dataset Schema

### puns_205.json

Each of the 205 entries contains:

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Sequential position (0-204) |
| `id` | int | Original joke ID from raw collection |
| `sentence` | string | Fill-in-the-blank sentence with `___` |
| `straight` | string[] | Non-pun completions, ordered by model frequency |
| `punny` | string[] | Pun completions, ordered by model frequency |
| `explanation` | string | Why the pun works (from initial rating) |
| `gemini_rating` | int | Gemini quality score (0-10) |
| `anthropic_rating` | int | Anthropic quality score (0-10) |
| `claudecode_rating` | int | Claude Code quality score (0-10) |
| `consensus_score` | int | gemini_rating + claudecode_rating |
| `cloze_balance` | float | Discriminativeness: min(p_str,p_fun)/max(p_str,p_fun) |
| `cloze_p_straight` | float | Fraction of model samples classified straight |
| `cloze_p_funny` | float | Fraction classified funny |
| `cloze_p_other` | float | Fraction classified other |
| `cloze_tier` | string | One of: balanced, leaning, funny_dominated, straight_dominated |
| `cloze_model_funny_rate` | object | Per-model funny completion rates {3b, 8b, 70b, 3.3-70b, 405b} |

### contextual_cloze_tests_100.json

Each of the 100 test entries contains:

| Field | Type | Description |
|-------|------|-------------|
| `pair_id` | int | Which of 50 pairs (0-49) |
| `type` | string | "straight" or "funny" — the priming context type |
| `prompt` | string | Filled A + filled B + truncated C |
| `expected_completion` | string[] | Completions matching the priming direction |
| `contrast_completion` | string[] | Completions for the opposite direction |
| `joke_a_index` | int | Index of context joke A in puns_205.json |
| `joke_b_index` | int | Index of context joke B in puns_205.json |
| `joke_c_index` | int | Index of target joke C in puns_205.json |
| `joke_c_sentence` | string | Full sentence of joke C (with `___`) |
| `joke_a_tier` | string | Cloze tier of joke A |
| `joke_b_tier` | string | Cloze tier of joke B |
| `joke_c_tier` | string | Cloze tier of joke C |
| `joke_c_balance` | float | Discriminativeness balance of joke C |

## Tier Distribution

```
balanced:           42 jokes  (balance >= 0.3, p_funny >= 5%)
leaning:            51 jokes  (0.05 < balance < 0.3, p_funny >= 5%)
straight_dominated: 88 jokes  (p_funny < 5%)
funny_dominated:    24 jokes  (p_funny > 50%, balance <= 0.05)
                   ───
Total:             205 jokes
```

In the contrastive tests:
- **C pool (targets):** 42 balanced + 8 leaning = 50 jokes
- **AB pool (context):** 88 straight_dominated + 12 leaning = 100 jokes
- **Excluded:** 24 funny_dominated + 31 remaining leaning = 55 jokes
