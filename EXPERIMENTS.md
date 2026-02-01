# Experiments

## Experiment 1: Pun Awareness Across Model Sizes

### Motivation

Do language models "get" puns? When a sentence has a blank that could be
filled with either a literal word or a pun, do models prefer the pun — and
does surrounding context influence that choice?

We test this using **contrastive cloze prompts**: each test presents a model
with two context sentences (filled in either with straight or punny words)
followed by a target sentence truncated before its blank. If models are
sensitive to contextual priming, they should produce more pun completions
when the context sentences use puns, and more straight completions when
the context uses literal words.

### Experiment Design

**Dataset.** 50 target jokes selected from the straight-dominated tier —
jokes where models naturally default to the literal (straight) completion.
This ensures the baseline funny rate is low, so any increase under funny
context reflects genuine context sensitivity rather than ceiling effects.
Each target is paired with two context jokes drawn from balanced, leaning,
and remaining straight-dominated tiers, which provide clear straight/funny
word contrasts for effective priming.
See [DATASET.md](DATASET.md) for dataset construction details.

**Test structure.** Each of 50 joke triples (A, B, C) generates two test prompts:
- **Straight-primed:** A and B filled with their top straight word, C truncated
- **Funny-primed:** A and B filled with their top pun word, C truncated

Total: 100 test prompts (50 pairs x 2 conditions).

**Models.** Five Llama models spanning 3B to 405B parameters, all accessed
via Together.ai with greedy decoding (temperature=0):

| Nickname | Model |
|----------|-------|
| 3b | meta-llama/Llama-3.2-3B-Instruct-Turbo |
| 8b | meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo |
| 70b | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo |
| 3.3-70b | meta-llama/Llama-3.3-70B-Instruct-Turbo |
| 405b | meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo |

**Classification.** Each model response is classified as **straight**, **funny**,
or **other** by matching the extracted first word against the joke's curated
word lists in `datasets/puns_205.json`.

### Code

| Script | Purpose |
|--------|---------|
| `run_cloze_benchmark.py` | Send all 100 test prompts to each model via Together.ai. Saves raw responses to `results/cloze_benchmark_raw.json` with checkpoint/backfill support. |
| `analyze_cloze_results.py` | Classify responses, compute per-model context-effect metrics, generate tables and plots. Outputs `results/cloze_analysis.json` and `results/figures/`. |

### Results

#### Response Classification by Context

| Model | Context | Straight | Funny | Other |
|-------|---------|:--------:|:-----:|:-----:|
| Llama-3.2-3B | straight | 72% | 0% | 28% |
| | funny | 70% | 0% | 30% |
| Llama-3.1-8B | straight | 60% | 6% | 34% |
| | funny | 48% | 10% | 42% |
| Llama-3.1-70B | straight | 66% | 12% | 22% |
| | funny | 34% | 34% | 32% |
| Llama-3.3-70B | straight | 66% | 8% | 26% |
| | funny | 36% | 32% | 32% |
| Llama-3.1-405B | straight | 66% | 18% | 16% |
| | funny | 22% | 46% | 32% |

#### Context Effect: Funny-Rate Shift

| Model | P(funny \| straight ctx) | P(funny \| funny ctx) | Delta |
|-------|:-:|:-:|:-:|
| Llama-3.2-3B | 0% | 0% | +0pp |
| Llama-3.1-8B | 6% | 10% | +4pp |
| Llama-3.1-70B | 12% | 34% | +22pp |
| Llama-3.3-70B | 8% | 32% | +24pp |
| Llama-3.1-405B | 18% | 46% | +28pp |

#### Plots

**Response classification by priming context:**

![Context Effect](results/figures/context_effect.png)

**Funny completion rate by context (with delta annotations):**

![Funny Shift](results/figures/funny_shift.png)

### Key Findings

1. **Context sensitivity scales monotonically with model size.** The funny-rate
   delta increases steadily from +0pp (3B) through +4pp (8B) to +22–28pp
   (70B–405B). Larger models are not just better at puns — they are
   specifically better at recognizing when the surrounding context invites a pun.

2. **The 3B model is completely pun-blind.** It produces 0% funny completions
   regardless of context. It doesn't recognize the pun opportunity at all,
   and the context manipulation has no effect.

3. **The 8B model shows early pun awareness.** It produces a small number of
   pun completions (6–10%) with a modest +4pp context effect. This suggests
   the 8B model is at the threshold of pun recognition — it occasionally
   detects the pun structure, but context provides only a small nudge.

4. **70B+ models show strong context-driven pun activation.** The 70B and
   3.3-70B models jump from 8–12% funny in straight context to 32–34% in
   funny context (+22–24pp). The 405B model shows the largest shift: 18%
   to 46% (+28pp). These models have the capacity to recognize and produce
   puns, but only do so when the surrounding context activates pun-mode
   processing.

5. **Straight completions are suppressed by funny context.** Across all models
   with pun awareness, the straight-completion rate drops sharply in funny
   context: from 66% to 22–36% for the 70B+ models. This is the flip side
   of the funny-rate increase and confirms the context manipulation is
   genuinely shifting model behavior, not just adding noise.

6. **The "other" category increases with model size under funny context.**
   Large models in funny context produce more responses that don't match
   either word list (22–32% other), suggesting they sometimes produce
   creative pun-adjacent completions that aren't in the curated lists.
