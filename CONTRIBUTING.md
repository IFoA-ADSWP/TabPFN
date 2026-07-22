# Contributing

This is a research project comparing TabPFN against traditional actuarial models for insurance tasks. All contributions welcome.

## Setup

```bash
# 1. Clone
git clone https://github.com/IFoA-ADSWP/TabPFN
cd TabPFN

# 2. Create environment (conda or venv)
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Repository Layout

```
src/              # Shared Python modules (data loading, metrics, baseline models)
notebooks/
  adswp_project/              # Domain-specific TabPFN applications
  baseline_experiments/       # Head-to-head model comparisons
data/raw/                     # Datasets (eudirectlapse.csv, freMTPL2freq.csv, ...)
outputs/
  current/                    # Latest figures + tables
  archive/                    # Historical experiment outputs (gitignored)
scripts/                      # One-off experiment scripts
docs/
  reports/                    # Analysis reports
  analyses/                   # Analysis summaries
```

## Running Notebooks

Notebooks are numbered in execution order within each directory:

```bash
# ADSWP project notebooks
jupyter notebook notebooks/adswp_project/01_TabPFN_classifier_eudirectlapse.ipynb

# Baseline experiments
jupyter notebook notebooks/baseline_experiments/01_claims_classification_baseline.ipynb
```

The main replication notebook is `notebooks/adswp_project/REPLICATION_There_Is_Life_in_the_Old_GLM_Yet.ipynb`.

## Adding Experiments

1. Add a numbered notebook in the appropriate `notebooks/` subdirectory
2. Import shared utilities from `src/` rather than duplicating code
3. Save outputs to `outputs/current/` with versioned archive in `outputs/archive/`
4. Add a brief summary in `docs/analyses/` or `docs/reports/` if significant

## Code Style

Formatting is handled by ruff. Run before committing:

```bash
pip install ruff
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Requests

- Use the PR template
- Keep changes focused (one experiment or fix per PR)
- Update `CHANGELOG.md` for user-facing changes
- Pre-existing CI checks run automatically
