# Development Setup

## Prerequisites

- Python 3.10+ (tested with 3.12)
- API keys for Together.ai, NDIF, and Anthropic (for dataset construction)

## Quick Start

```bash
# 1. Create and activate the virtual environment
bash setup_env.sh
source venv/bin/activate

# 2. Copy .env.local.example to .env.local and fill in your API keys
cp .env.local.example .env.local
# Edit .env.local with your keys

# 3. Run experiments
python3 analyze_baseline_performance.py    # Baseline pun rates
python3 analyze_cloze_results.py           # Contrastive experiment analysis
```

## Environment

All scripts load API keys from `.env.local` in the project root:

```
TOGETHER_API_KEY=...     # Together.ai (Llama model inference)
NDIF_API_KEY=...         # NDIF (remote activation collection via nnsight)
HF_TOKEN=...             # HuggingFace (model tokenizer access)
ANTHROPIC_API_KEY=...    # Anthropic (dataset construction, classification)
```

## Dependencies

Installed by `setup_env.sh`:

| Package | Purpose |
|---------|---------|
| `nnsight==0.5.11` | Neural network inspection, NDIF remote execution |
| `torch` | Tensor operations, model architecture |
| `numpy` | Numerical computation, activation storage |
| `matplotlib` | Plot generation |
| `scikit-learn` | PCA, dimensionality reduction |
| `requests` | Together.ai API calls |
| `python-dotenv` | Environment variable loading |

## Scripts

### Dataset Construction

These scripts were used during the initial dataset creation phase and
don't need to be re-run:

| Script | Purpose |
|--------|---------|
| `datasets/rawdata/analyze_jokes.py` | Multi-provider joke rating |
| `datasets/rawdata/repair_jokes.py` | Weak joke repair via Claude |
| `datasets/rawdata/apply_edits.py` | Hand-edit application |
| `datasets/rawdata/build_puns_205.py` | Final dataset assembly |

### Experiment Scripts

| Script | Purpose |
|--------|---------|
| `datasets/annotate_completions.py` | Collect & classify model completions |
| `datasets/build_cloze_tests.py` | Build contrastive test prompts |
| `run_cloze_benchmark.py` | Send test prompts to models via Together.ai |
| `analyze_cloze_results.py` | Classify responses, generate tables + plots |
| `analyze_baseline_performance.py` | Baseline pun-rate analysis across models |

### Interpretability Scripts

| Script | Purpose |
|--------|---------|
| `collect_activations.py` | Collect hidden-state activations via NDIF |
| `visualize_activations.py` | PCA visualization of activation differences |

## Project Data

| File | Description |
|------|-------------|
| `datasets/puns_205.json` | 205 curated pun sentences with word lists |
| `datasets/contextual_cloze_tests_100.json` | 100 contrastive test prompts |
| `results/cloze_benchmark_raw.json` | Raw model responses |
| `results/cloze_analysis.json` | Classified results |
| `results/activations/` | Collected activation data |
| `results/figures/` | Generated plots |
