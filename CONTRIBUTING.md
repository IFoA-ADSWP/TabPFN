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
src/                    # Shared Python modules (data loading, metrics, baseline models)
notebooks/
  adswp_project/        # Domain-specific TabPFN applications (real insurance data)
  baseline_experiments/ # Head-to-head model comparisons (benchmarks)
data/raw/               # Datasets (eudirectlapse.csv, freMTPL2freq.csv, ...)
  processed/            # Intermediate benchmark results (CSV)
outputs/
  current/              # Latest figures, tables, logs (git-tracked)
  replication/          # Paper replication snapshots (git-tracked)
  archive/              # Historical experiment outputs (gitignored)
scripts/                # One-off experiment scripts (not notebooks)
docs/
  reports/              # Analysis reports (must register in REPORT_REGISTRY.md)
  analyses/             # Methodology notes (lighter than reports)
  papers/               # Paper content and LaTeX style files
  status/               # Project status and security record
tests/                  # Smoke tests
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

## Contribution Map — Where Things Go

| If you want to... | Put it in... | Convention |
|---|---|---|
| Run a new experiment on real insurance data | `notebooks/adswp_project/NN_descriptive_name.ipynb` | Two-digit number, snake_case name |
| Run a head-to-head model comparison | `notebooks/baseline_experiments/NN_descriptive_name.ipynb` | Same |
| Write a reusable utility | `src/` as a Python module | Importable function/class, notebook-inline code does not belong here |
| Run a one-off analysis (not a notebook) | `scripts/descriptive_name.py` | snake_case, `if __name__ == "__main__":` guard |
| Write an analysis report | `docs/reports/DESCRIPTIVE_NAME.md` | ALL_CAPS snake_case, register in `REPORT_REGISTRY.md` |
| Write a lightweight methodology note | `docs/analyses/descriptive_name.md` | snake_case, no registry entry needed |
| Update the paper | `docs/papers/` | Markdown or `.sty` as appropriate |
| Save output figures | `outputs/current/figures/` | Descriptive PNG name |
| Save output tables | `outputs/current/tables/` | Descriptive CSV name |
| Save run logs | `outputs/current/logs/` | Descriptive markdown name |
| Archive old outputs | `outputs/archive/` | Gitignored — not committed |
| Add a test | `tests/test_descriptive_name.py` | One test file per `src/` module |

## Standards Checklist

Every contribution should satisfy these before review:

- [ ] **Naming** — notebooks are two-digit numbered + snake_case; reports are ALL_CAPS snake_case; scripts and modules are snake_case
- [ ] **No code duplication** — import from `src/` rather than copying utility code into notebooks
- [ ] **Idempotent notebooks** — runs top-to-bottom without pre-existing state; outputs are regenerated, not committed (unless a deliberate reproducibility snapshot)
- [ ] **Reports registered** — any new file in `docs/reports/` must have a row in `docs/reports/REPORT_REGISTRY.md` with topic_key, audience, source_workbooks, and evidence_files
- [ ] **Changelog updated** — user-facing changes get an entry in `CHANGELOG.md` under `[Unreleased]`
- [ ] **Pre-commit green** — see below
- [ ] **Smoke test passes** — `pytest tests/` after any change to `src/`
- [ ] **PR focused** — one experiment or fix per pull request

## Pre-commit Setup

Install once after cloning:

```bash
pip install pre-commit
pre-commit install
```

This runs ruff (format + lint), commitizen (commit message convention), and other checks automatically on every commit. Run against all files manually when first setting up:

```bash
pre-commit run --all-files
```

To run ruff directly:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Report Registration

If you add a file to `docs/reports/`, open `docs/reports/REPORT_REGISTRY.md` and add a row with:

- `report_path` — relative path to your file
- `topic_key` — short kebab-case identifier (check for duplicates first)
- `audience` — `technical` or `non-technical`
- `status` — `active`, `draft`, or `archived`
- `source_workbooks` — semicolon-separated paths to source notebooks
- `evidence_files` — semicolon-separated paths to tables/figures/logs
- `last_updated` — ISO date

## Pull Requests

- Use the PR template
- Keep changes focused (one experiment or fix per PR)
- Update `CHANGELOG.md` for user-facing changes
- Pre-existing CI checks run automatically
