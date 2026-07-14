# Maintenance Backlog — Production Readiness

> **Why this file exists:** GitHub Issues are currently disabled on this repository. Until they are re-enabled, this document serves as the canonical backlog of work required to make the repository production-ready. Each issue below is a self-contained unit of work with problem statement, evidence, proposed fix, acceptance criteria, and effort estimate.
>
> **When Issues are re-enabled:** each issue can be opened by running the `gh issue create` commands in the [Import section](#importing-to-github-issues-when-enabled).

---

## Backlog Summary

| # | Title | Tier | Effort | Status |
|---|-------|------|--------|--------|
| 1 | Add LICENSE file | T1 | 5 min | ⬜ Not started |
| 2 | Untrack `.venv/` and `.venv312/` | T1 | 5 min | ⬜ Not started |
| 3 | Add `pyproject.toml` and pinned `requirements.txt` | T1 | 2 h | ⬜ Not started |
| 4 | Rewrite broken CI workflow (`.github/workflows/pull_request.yml`) | T1 | 1 h | ⬜ Not started |
| 5 | Add `tests/` directory with smoke test | T1 | 15 min | ⬜ Not started |
| 6 | Fix `.gitignore` to actually exclude outputs, models, archives | T1 | 30 min | ⬜ Not started |
| 7 | Fix or remove empty `05_regression_finetuning.ipynb` | T1 | 30 min | ⬜ Not started |
| 8 | Rewrite README directory tree to match reality | T1 | 1 h | ⬜ Not started |
| 9 | Rotate `TABPFN_API_KEY` and confirm `.env` is not committed | T1 | 5 min | ⬜ Not started |
| 10 | Add `CHANGELOG.md` (PR template references it) | T1 | 5 min | ⬜ Not started |
| 11 | Add `notebooks/README.md` index | T2 | 1 h | ⬜ Not started |
| 12 | Add `docs/README.md` landing page | T2 | 1 h | ⬜ Not started |
| 13 | Move root-level audit/status `.md` files to `docs/status/` | T2 | 30 min | ⬜ Not started |
| 14 | Dedup `src/data_loader.py` vs `src/data_loader_class.py` | T2 | 1 h | ⬜ Not started |
| 15 | Make hardcoded `TabPFN-upstream/` path optional in scripts | T2 | 1 h | ⬜ Not started |
| 16 | Add `scripts/README.md` index | T2 | 1 h | ⬜ Not started |
| 17 | Extend `replication_config.json` with versions, SHA, hardware | T2 | 30 min | ⬜ Not started |
| 18 | Standardize random seed defaults via `baseline_config.py` | T2 | 30 min | ⬜ Not started |
| 19 | Unify output naming conventions | T2 | 1 h | ⬜ Not started |
| 20 | Move `data/processed/` outputs to `outputs/` | T2 | 15 min | ⬜ Not started |
| 21 | Add module docstrings to 5 scripts | T2 | 20 min | ⬜ Not started |
| 22 | Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | T2 | 30 min | ⬜ Not started |
| 23 | Add `data/README.md` with per-CSV schema | T2 | 1 h | ⬜ Not started |
| 24 | Populate `src/__init__.py` with `__version__` and `__all__` | T2 | 10 min | ⬜ Not started |
| 25 | Delete dead `BaselineConfig` wrapper in `src/baseline_utils.py` | T2 | 5 min | ⬜ Not started |
| 26 | Add `CITATION.cff` and citation block to README | T3 | 1 h | ⬜ Not started |
| 27 | Add `.python-version` and a minimal `Dockerfile` | T3 | 2 h | ⬜ Not started |
| 28 | Add `Makefile` with `install / test / reproduce-paper` | T3 | 1 h | ⬜ Not started |
| 29 | Add `pytest-cov` and coverage reporting | T3 | 1 h | ⬜ Not started |
| 30 | Parameterize the REPLICATION notebook (papermill) | T3 | 2 h | ⬜ Not started |
| 31 | Add notebook smoke test (headless run on a 100-row fixture) | T3 | 2 h | ⬜ Not started |
| 32 | Enable Dependabot auto-merge for patch updates | T3 | 1 h | ⬜ Not started |
| 33 | Move import-time I/O under `if __name__ == "__main__":` | T3 | 10 min | ⬜ Not started |
| 34 | Move admin docs (`funding_request.md`, `procurement_request_*`, `provisioning_gpu.md`) to `docs/admin/` | T3 | 10 min | ⬜ Not started |
| 35 | Delete superseded `TabPFN_Classifier_on_eudirectlapse_v1_0.ipynb` | T3 | 5 min | ⬜ Not started |
| 36 | Gitignore and untrack `notebooks/baseline_experiments/catboost_info/` | T3 | 10 min | ⬜ Not started |

**Totals:** T1 = ~6 h, T2 = ~10 h, T3 = ~10 h, **~26 h total** (≈ 3 focused work days).

**Tiers:**
- **T1** — Day-1 blockers. New contributor cannot onboard until these are fixed.
- **T2** — Onboarding friction. Code works, navigation is painful.
- **T3** — Production polish. Required for a true open-source release.

---

## Tier 1 — Critical (Day-1 Blockers)

### Issue 1 — Add `LICENSE` file

**Problem.** The repository has no `LICENSE` file at any level. The project cannot be legally reused or redistributed outside the project team. For an open-source / open-research release this is a day-one blocker.

**Evidence.**

```bash
$ find . -maxdepth 2 -iname 'license*' -not -path '*/.venv*' -not -path '*/.git/*'
$ # (no output)
```

**Proposed fix.**

1. Decide on a license. Suggested: **MIT** (simple, permissive, common for research code) or **Apache-2.0** (explicit patent grant, also common). If a license has already been agreed with the institutional sponsor, use that one.
2. Add `LICENSE` at the repo root with the full license text.
3. Add a one-line badge to the top of `README.md`: `[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)`
4. Add a `License` section under the project description block.

**Acceptance criteria.**

- [ ] `LICENSE` file exists at repo root
- [ ] README has a license badge and section
- [ ] Commit message references the chosen license choice

**Effort.** ~5 minutes (text paste + badge). Trivial unblocker.

**Depends on.** Nothing.

---

### Issue 2 — Untrack and gitignore `.venv/` and `.venv312/`

**Problem.** Two Python virtual environments are tracked in git: `.venv/` (628 files) and `.venv312/` (many more). This bloats the repository, breaks the pre-commit `check-added-large-files` rule, and signals the `.gitignore` is not being enforced.

**Evidence.**

```bash
$ git ls-files | grep -c '^\.venv/'
628
$ git ls-files | grep -c '^\.venv312/'
<many>
```

The current `.gitignore` only lists `.venv`, `env`, `venv` — not `.venv312` or `.venv*`.

**Proposed fix.**

1. Add to `.gitignore`:
   ```gitignore
   .venv/
   .venv*/
   venv/
   env/
   ENV/
   ```
2. Remove from git index:
   ```bash
   git rm -r --cached .venv .venv312
   ```
3. (Optional) `git gc --aggressive --prune=now` to shrink the packfile. Confirm with reviewer first.
4. Commit as a single `chore:` commit.

**Acceptance criteria.**

- [ ] `git ls-files | grep -c '^\.venv' ` returns 0
- [ ] `.gitignore` excludes `.venv/`, `.venv*/`, `venv/`, `env/`
- [ ] `git status` clean after the operation
- [ ] Fresh clone + `python -m venv .venv` works without conflicts

**Effort.** ~5 minutes of git commands + careful review (this rewrites local state visibly).

**Risk.** Medium. `git rm -r --cached` is local-only; the commit is reversible with `git revert`. Reviewer should confirm the diff contains only the venv removal.

---

### Issue 3 — Add `pyproject.toml` and pinned `requirements.txt`

**Problem.** No `pyproject.toml`, `requirements.txt`, `setup.py`, or any dependency manifest. Yet:

- `README.md` line 80 instructs `pip install -r requirements.txt` (will fail)
- `docs/REPLICATION_SETUP_GUIDE.md` line 92 says `pip install tabpfn` (unpinned)
- `.pre-commit-config.yaml` line 53 says "ruff version must be the same as in pyproject.toml" (file missing)
- `.github/workflows/pull_request.yml` lines 14, 19 use `version-file: pyproject.toml` for ruff (file missing)

A new contributor cannot install the environment, and CI is broken from day one.

**Proposed fix.**

**1. Create `pyproject.toml` at repo root:**

```toml
[project]
name = "tabpfn-adswp"
version = "0.1.0"
description = "TabPFN for actuarial / insurance use cases (ADSWP)"
readme = "README.md"
requires-python = ">=3.9,<3.14"
license = { file = "LICENSE" }
dependencies = [
    "tabpfn==2.6.0",
    "numpy>=1.24,<3.0",
    "pandas>=2.0,<3.0",
    "scikit-learn>=1.3,<2.0",
    "matplotlib>=3.7",
    "jupyter>=1.0",
]

[project.optional-dependencies]
finetune = ["torch>=2.0"]
regression = ["catboost>=1.2", "xgboost>=2.0"]
all = ["tabpfn-adswp[finetune,regression]"]

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
```

**2. Generate `requirements.txt` from pyproject:**

```bash
pip-compile pyproject.toml -o requirements.txt
```

**3. Update README install line** to: `pip install -e .[all]`.

**Acceptance criteria.**

- [ ] `pyproject.toml` exists with deps pinned to current working versions
- [ ] `requirements.txt` exists, byte-identical to a `pip-compile` output
- [ ] `pip install -e .` succeeds in a fresh venv
- [ ] `ruff check src/` reads version from pyproject
- [ ] README install line updated
- [ ] `REPLICATION_SETUP_GUIDE.md` updated

**Effort.** ~2 hours (test the install in a fresh venv, verify imports).

**Depends on.** Issue 1 (LICENSE).

---

### Issue 4 — Rewrite broken CI workflow

**Problem.** `.github/workflows/pull_request.yml` was copied from the upstream TabPFN repo and never adapted. It references:

- `./tests/` directory (does not exist)
- `pyproject.toml` (does not exist)
- `requirements.txt` (does not exist)
- `scripts/generate_dependencies.py` (does not exist)

Every PR will fail CI, no matter what the change is.

**Proposed fix.**

Replace `.github/workflows/pull_request.yml` with a minimal, working workflow:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v1
        with:
          version-file: pyproject.toml
          args: "check src/ scripts/ --output-format=github"
      - uses: astral-sh/ruff-action@v1
        with:
          version-file: pyproject.toml
          args: "format --check src/ scripts/"

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e .[all]
      - run: pytest tests/ -q
```

**Acceptance criteria.**

- [ ] Workflow file references only files that exist in the repo
- [ ] Lint job runs ruff and passes
- [ ] Test job runs `pytest tests/` and passes
- [ ] Tested on a sample PR — both jobs green

**Effort.** ~1 hour.

**Depends on.** Issue 3 (`pyproject.toml`), Issue 5 (`tests/`).

---

### Issue 5 — Add `tests/` directory with smoke test

**Problem.** No `tests/` directory, no `test_*.py` files, no `conftest.py`. CI calls `pytest tests/` and the directory is missing. Zero test coverage.

**Proposed fix.**

1. Create `tests/__init__.py` (empty) and `tests/test_smoke.py`:
   ```python
   """Smoke tests — fail loudly if the package is broken."""

   def test_imports():
       from src import data_loader, evaluation_metrics, model_training
       from src import baseline_config, baseline_utils, data_loader_class
       assert data_loader is not None
       assert evaluation_metrics is not None
       assert model_training is not None

   def test_baseline_config_constants():
       from src.baseline_config import RANDOM_SEED, DATA_DIR
       assert isinstance(RANDOM_SEED, int)
       assert RANDOM_SEED == 42
   ```
2. Add `conftest.py`:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
   ```

**Acceptance criteria.**

- [ ] `pytest tests/` returns 2 passing tests
- [ ] `tests/conftest.py` makes `src` importable

**Effort.** ~15 minutes.

---

### Issue 6 — Fix `.gitignore` to match what the README claims

**Problem.** `.gitignore` does not match the README's "gitignore Strategy" section. README claims `outputs/archive/`, `*.pkl`, and `data/` are excluded; none are. Tracked binary artifacts:

- 9 `.pkl` files in `outputs/archive/`
- 17 `.tabpfn_fit` files in `outputs/current/models/` (different extension, slips past `*.pkl`)
- 6 `.DS_Store` files in non-top-level directories
- 2 `__pycache__/` directories in `src/` and `scripts/`

**Proposed fix.**

Append to `.gitignore`:

```gitignore
# Output artifacts
outputs/archive/
*.pkl
*.tabpfn_fit
*.joblib
*.npy
*.npz

# OS / editor junk
**/.DS_Store
**/Thumbs.db

# CatBoost
**/catboost_info/

# Python build artifacts (if not already there)
**/__pycache__/
*.pyc
*.pyo
```

Then untrack:

```bash
git rm -r --cached outputs/archive/
git rm --cached '*.pkl' '*.tabpfn_fit' '*.DS_Store' 2>/dev/null
git rm -r --cached src/__pycache__ scripts/__pycache__ 2>/dev/null
```

**Acceptance criteria.**

- [ ] `git ls-files outputs/archive/` returns nothing
- [ ] `git ls-files '*.pkl' '*.tabpfn_fit'` returns nothing
- [ ] `.gitignore` matches the README's claims
- [ ] README is updated to reflect the new rules (Section: "gitignore Strategy")

**Effort.** ~30 minutes.

---

### Issue 7 — Fix or remove empty `05_regression_finetuning.ipynb`

**Problem.** `notebooks/baseline_experiments/05_regression_finetuning.ipynb` is a 119-byte JSON stub with no cells. But `docs/reports/REPORT_REGISTRY.md` lists it as a source workbook for **4 reports**: `INSURANCE_DOMAIN_FINETUNING_METHOD_PROTOCOL`, `COMBINED_TABPFN_CLASSIFIER_REGRESSOR_ANALYSIS`, `INSURANCE_SPECIFIC_FINETUNING_EVIDENCE`, `TABPFN_FINE_TUNING_LIMIT_STUDY`, `CLASSIFIER_HOMOGENEITY_HYPOTHESIS_METHOD`. Either the reports are ungrounded or the notebook needs filling.

**Proposed fix.**

Pick one:

- **Option A (preferred):** Fill the notebook with the regressor fine-tuning experiment scaffold, using the existing scripts (`run_small_finetune_regressor_trial.py`, `run_finetuned_tabpfn_regression_benchmark.py`). Connect each cell to the outputs in `outputs/current/tables/`.
- **Option B:** Remove the notebook and update `REPORT_REGISTRY.md` to remove it as a source for the affected reports (which must then list their actual sources).

**Acceptance criteria.**

- [ ] Notebook either contains a complete experiment with visible outputs, or is deleted
- [ ] `REPORT_REGISTRY.md` is consistent with the chosen path

**Effort.** ~30 min (Option B) to several hours (Option A).

---

### Issue 8 — Rewrite README directory tree to match reality

**Problem.** `README.md`'s directory diagram is significantly out of date:

- Lists 5 baseline notebooks; repo has 8
- Claims `outputs/shap/` and `outputs/catboost_info/` exist; they don't
- Claims `legacy/adswp_project_scripts/TabPFN_ausprivauto0405.R` exists; it doesn't
- `data/processed/` exists with files but isn't mentioned
- `outputs/replication/` exists (created 14 Jul 2026) but isn't in the tree
- Only lists 2 raw datasets in `data/raw/`; actual is 5
- Only references `data_loader.py` and `evaluation_metrics.py` in src/; actual is 6 modules

**Proposed fix.**

1. Generate the tree: `tree -L 3 -I '.venv*|.venv312*|.git|__pycache__|*.pyc' --dirsfirst`
2. Manually edit the README to insert the generated tree in place of the existing one.
3. Add a "Data" subsection listing all 5 raw CSVs.
4. Add an "Outputs" subsection listing `current/`, `archive/`, `replication/`.
5. Update the "Python Modules" list to include all 6 modules.

**Acceptance criteria.**

- [ ] README tree matches `tree -L 3` output
- [ ] All file references in the README resolve to existing files
- [ ] "Last updated" date is set to today's date

**Effort.** ~1 hour.

---

### Issue 9 — Rotate `TABPFN_API_KEY` and confirm `.env` is not committed

**Problem.** A real JWT-style API key is sitting in plaintext at `.env` at the repo root. The file is correctly listed in `.gitignore`, so it is not in the git tree, but:

- The key is exposed to anyone with filesystem access
- If the gitignore rule is ever weakened, the key leaks
- Best practice: rotate the key, regenerate, and re-establish `.env` from `.env.example`

**Proposed fix.**

1. In the TabPFN API console (or wherever the key was issued), revoke the current key and generate a new one.
2. Update local `.env` with the new key.
3. Verify `.env` is still in `.gitignore` and **not** in the git tree:
   ```bash
   git ls-files | grep -E '^\.env$'  # must return nothing
   ```
4. If `.env.example` exists, ensure it has only a placeholder like `TABPFN_API_KEY=your-key-here`.
5. Add a note to the README: "Never commit `.env`. Use `.env.example` as a template."

**Acceptance criteria.**

- [ ] Old key revoked in provider console
- [ ] New key generated and stored in local `.env`
- [ ] `.env` is not in `git ls-files`
- [ ] `.env.example` exists with placeholders only
- [ ] README has a `.env` warning

**Effort.** ~5 minutes (assuming access to the key issuance console).

---

### Issue 10 — Add `CHANGELOG.md` (PR template references it)

**Problem.** `.github/PULL_REQUEST_TEMPLATE.md` line 20 asks contributors to add a `CHANGELOG.md` entry, but no such file exists. Either remove the line or create the file.

**Proposed fix.**

Create `CHANGELOG.md` at repo root with the Keep a Changelog format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (TBD)

### Changed
- (TBD)

### Removed
- (TBD)
```

**Acceptance criteria.**

- [ ] `CHANGELOG.md` exists at repo root
- [ ] PR template line is now satisfiable
- [ ] First entry will be added with the next release

**Effort.** ~5 minutes.

---

## Tier 2 — Important (Onboarding Friction)

### Issue 11 — Add `notebooks/README.md` index

**Problem.** 14 notebooks across 2 directories, no landing page, no ordering, no map from notebook to output. `notebooks/baseline_experiments/REFACTORING_SUMMARY.md` is the only orientation doc and lives in the wrong place.

**Proposed fix.**

Create `notebooks/README.md` with a table:

| Notebook | Project | Status | Produces |
|----------|---------|--------|----------|
| `adswp_project/01_TabPFN_classifier_eudirectlapse.ipynb` | ADSWP | ✓ | `outputs/current/tables/Table1_Model_Performance.csv` |
| `adswp_project/02_TabPFN_regression_*.ipynb` | ADSWP | ✓ | ... |
| ... | | | |

Include a one-paragraph "Start here" pointer at the top.

**Acceptance criteria.**

- [ ] `notebooks/README.md` exists
- [ ] Every notebook in `notebooks/` is listed
- [ ] Each entry notes its output paths

**Effort.** ~1 hour.

---

### Issue 12 — Add `docs/README.md` landing page

**Problem.** 30+ markdown files across `docs/reports/`, `docs/papers/`, `docs/analyses/`, `docs/status/`, plus root-level `funding_request.md`, `procurement_request_A100-80.md`, `provisioning_gpu.md`, `REPLICATION_SETUP_GUIDE.md`. No orientation for a new reader.

**Proposed fix.**

Create `docs/README.md` with:
- One-paragraph project description
- Subdirectory map (`reports/`, `papers/`, `analyses/`, `status/`, `admin/`)
- 1-line description per subdirectory
- Link to `docs/reports/REPORT_REGISTRY.md` (which should be the master index of reports)

**Acceptance criteria.**

- [ ] `docs/README.md` exists and points to every subdirectory
- [ ] Every subdirectory is described in 1–2 lines

**Effort.** ~1 hour.

---

### Issue 13 — Move root-level audit/status `.md` files to `docs/status/`

**Problem.** 11 `.md` files at the repo root. Only `README.md` belongs there. The other 10 are process/audit artifacts and belong in `docs/status/`.

**Files to move.**

```
ADDED_FILES_COMPARISON.md
ADDED_FILES_COMPLETE_LIST.txt
CLEANUP_COMPLETE.md
CLEANUP_PLAN.md
FINAL_STATUS.md
NOTEBOOK_ANALYSIS.md
REFACTORING_PLAN.md
REORGANIZATION_COMPLETE.md
REPOSITORY_STRUCTURE_ANALYSIS.md
SESSION_SUMMARY_APRIL2_REGRESSOR_SCALING.md
STAGE_R1_PHASE1A_REPORT.md
```

**Proposed fix.**

```bash
mkdir -p docs/status
git mv ADDED_FILES_COMPARISON.md docs/status/
git mv ADDED_FILES_COMPLETE_LIST.txt docs/status/
# ... etc.
```

**Acceptance criteria.**

- [ ] Only `README.md` (and `CHANGELOG.md` if added) at repo root
- [ ] All audit files moved to `docs/status/`
- [ ] Internal links in any of these files that point to root-relative paths are updated

**Effort.** ~30 minutes.

---

### Issue 14 — Dedup `src/data_loader.py` and `src/data_loader_class.py`

**Problem.** Two competing data loaders. `data_loader.py` defines functions (`load_data`, `preprocess_data`); `data_loader_class.py` defines a class (`DataLoader`) that wraps and extends the function. `REFACTORING_SUMMARY.md` acknowledges the duplication as "Phase 2" work. Similarly, `src/model_training.py` has its own training functions and `src/baseline_utils.py` has `fit_and_evaluate` and `fit_and_eval` (the latter a near-duplicate added per `REFACTORING_SUMMARY.md:288–291`).

**Proposed fix.**

1. Keep `src/data_loader_class.py` (more complete).
2. Delete `src/data_loader.py`.
3. Update all callers (`scripts/*`, `notebooks/*`) to use `DataLoader` instead of the functions.
4. Delete `fit_and_eval` from `src/baseline_utils.py` (keep `fit_and_evaluate`).
5. Update callers accordingly.

**Acceptance criteria.**

- [ ] Only one data-loader module in `src/`
- [ ] Only one fit-and-evaluate function in `src/`
- [ ] All callers updated and the test suite (Issue 5) still passes
- [ ] At least one notebook re-runs end-to-end

**Effort.** ~1 hour.

---

### Issue 15 — Make hardcoded `TabPFN-upstream/` path optional

**Problem.** Five scripts hard-code a sibling-repo path: `Path(__file__).resolve().parents[2] / "TabPFN-upstream" / "src"`. No README warning, no check, no fallback.

**Files affected.**

- `scripts/run_small_finetune_classifier_trial.py:30`
- `scripts/check_saved_finetune_classifier_model.py:24`
- `scripts/diagnose_claimnb_finiteness.py:42`
- `scripts/run_finetuned_tabpfn_regression_benchmark.py:21`

**Proposed fix.**

Replace the hard-coded path with a CLI argument or environment variable:

```python
import os
from pathlib import Path

def resolve_upstream_src():
    """Locate the TabPFN upstream source. Set TABPFN_UPSTREAM_SRC env var to override."""
    override = os.environ.get("TABPFN_UPSTREAM_SRC")
    if override:
        return Path(override)
    default = Path(__file__).resolve().parents[2] / "TabPFN-upstream" / "src"
    if not default.exists():
        raise FileNotFoundError(
            f"TabPFN upstream src not found at {default}. "
            f"Either clone https://github.com/IFoA-ADSWP/TabPFN-upstream as a sibling, "
            f"or set TABPFN_UPSTREAM_SRC env var to the absolute path."
        )
    return default
```

**Acceptance criteria.**

- [ ] No hard-coded path to `TabPFN-upstream` outside the `resolve_upstream_src()` helper
- [ ] Scripts raise a clean error if the path is missing
- [ ] README documents the env-var override

**Effort.** ~1 hour.

---

### Issue 16 — Add `scripts/README.md` index

**Problem.** 19 scripts in `scripts/`, no orientation. A new contributor has no idea which to run, in what order, with what inputs.

**Proposed fix.**

Create `scripts/README.md` with a table:

| Script | Purpose | Inputs | Outputs | Notes |
|--------|---------|--------|---------|-------|
| `run_small_finetune_classifier_trial.py` | Single-config TabPFNClassifier fine-tune | CSV, target, rows | `outputs/current/tables/tabpfn_finetune_trial_results.csv` | Run from repo root |
| `run_domain_finetune_stage_a.py` | Leave-one-out domain fine-tune | Insurance datasets | `outputs/current/tables/domain_finetune_study_runs.csv` | Long-running |
| `download_datasets.py` | Pulls coil2000, ausprivauto0405, freMTPL2freq_binary | None | `data/raw/*.csv` | One-shot |
| ... | | | | |

Group by use case: "Quickstart", "Fine-tuning", "Regression benchmarking", "Diagnostics".

**Acceptance criteria.**

- [ ] `scripts/README.md` exists
- [ ] Every script in `scripts/*.py` is listed
- [ ] Each entry names inputs and outputs

**Effort.** ~1 hour.

---

### Issue 17 — Extend `replication_config.json` with versions, SHA, hardware

**Problem.** The current `replication_config.json` captures only: notebook, dataset, paper, authors, seed, split, sample counts, device, target rate, timestamp. Missing: library versions, dataset SHA, commit SHA, OS, hardware details, model checkpoint, calibration params, CV folds.

**Proposed fix.**

Extend the JSON schema:

```json
{
  "Notebook": "...",
  "Dataset": "...",
  "Paper_Title": "...",
  "Authors": "...",
  "Seed": 45,
  "Train_Test_Split": "70.0%/30.0%",
  "Stratified": true,
  "Samples_Train": 16142,
  "Samples_Test": 6918,
  "Features": 18,
  "Target_Lapse_Rate": 0.128,
  "Device": "cpu",
  "Timestamp": "2026-07-14T22:00:00Z",
  "Python_Version": "3.11.9",
  "Package_Versions": {
    "tabpfn": "2.6.0",
    "numpy": "1.26.4",
    "scikit-learn": "1.4.1.post1"
  },
  "Dataset_Sha256": "...",
  "Commit_Sha": "...",
  "Hardware": {
    "cpu": "Apple M2",
    "ram_gb": 16,
    "os": "macOS 14.5"
  },
  "Model_Checkpoint": "tabpfn-v2-classifier",
  "Calibration": "isotonic",
  "CV_Folds": null
}
```

Also update `REPLICATION_SETUP_GUIDE.md` to document the schema and how it's auto-populated.

**Acceptance criteria.**

- [ ] JSON includes the new fields
- [ ] The REPLICATION notebook writes the JSON on completion
- [ ] README references the new schema

**Effort.** ~30 minutes.

---

### Issue 18 — Standardize random seed defaults via `baseline_config.py`

**Problem.** `src/data_loader.py:9,32` defaults to `random_state=0`. `src/model_training.py:75,84,85,86,93,100` hard-codes `random_state=0`. Scripts and notebooks use 42. The REPLICATION uses 45. Three different seeds for "default" depending on where you start.

**Proposed fix.**

1. Add a centralized `RANDOM_SEED` to `src/baseline_config.py` (already there as `RANDOM_SEED = 42`).
2. Replace every `random_state=0` and `random_state=None` in `src/` with `random_state=baseline_config.RANDOM_SEED` (or import as `from .baseline_config import RANDOM_SEED`).
3. Document in `baseline_config.py` that REPLICATION uses 45 (a project-specific override).

**Acceptance criteria.**

- [ ] No literal `random_state=0` or `random_state=None` in `src/`
- [ ] `RANDOM_SEED` from `baseline_config.py` is the single source of truth
- [ ] Existing tests still pass

**Effort.** ~30 minutes.

---

### Issue 19 — Unify output naming conventions

**Problem.** Three different filename conventions in three subdirectories of `outputs/`:

- `outputs/archive/model_comparison_YYYYMMDD_HHMMSS.csv` (matches README)
- `outputs/current/models/YYYYMMDDTHHMMSSZ_*` (ISO-style, no README)
- `outputs/current/figures/FigureN_*.png` (matches README)

**Proposed fix.**

Pick one convention: `YYYYMMDD_HHMMSS_*` (matches README, separator is space-friendly). Apply across all `outputs/`. Rename existing files with `git mv`.

**Acceptance criteria.**

- [ ] All `outputs/` files use one timestamp format
- [ ] README documents the convention
- [ ] No `Z`-suffix files remain

**Effort.** ~1 hour.

---

### Issue 20 — Move `data/processed/` outputs to `outputs/`

**Problem.** `data/processed/` contains output-style artifacts (CSVs and PNGs from the multi-dataset benchmark) that belong in `outputs/`, not `data/`. `data/` should be inputs only.

**Files to move.**

```
data/processed/glm_vs_tabpfn_head_to_head.csv
data/processed/multi_dataset_benchmark_results.csv
data/processed/multi_dataset_regression_benchmark_results.csv
data/processed/multi_dataset_regression_rmse_comparison.png
data/processed/multi_dataset_roc_comparison.png
```

**Proposed fix.**

```bash
mkdir -p outputs/benchmarks
git mv data/processed/*.csv data/processed/*.png outputs/benchmarks/
rmdir data/processed
```

Update README and any notebook that references the old path.

**Acceptance criteria.**

- [ ] `data/processed/` is removed
- [ ] All files end up in `outputs/benchmarks/`
- [ ] All callers updated

**Effort.** ~15 minutes.

---

### Issue 21 — Add module docstrings to 5 scripts

**Problem.** Five scripts lack a top-of-file module docstring:

- `scripts/debug_preprocess.py`
- `scripts/diagnose_claimnb_finiteness.py`
- `scripts/run_finetuned_tabpfn_regression_benchmark.py`
- `scripts/run_raw_tabpfn_regression_benchmark.py`
- `scripts/summarize_classifier_homogeneity_smoke.py`

**Proposed fix.**

Add a 1-paragraph module docstring to each: what it does, what it produces, when to run it.

**Acceptance criteria.**

- [ ] All 5 scripts have module docstrings
- [ ] Ruff `D` rule (docstrings) passes on the new files

**Effort.** ~20 minutes.

---

### Issue 22 — Add `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`

**Problem.** None of the standard open-source community files exist. The repo's `PULL_REQUEST_TEMPLATE.md` and `ISSUE_TEMPLATE/` reference a `CHANGELOG.md` but no `CONTRIBUTING.md` is present.

**Proposed fix.**

1. Create `CONTRIBUTING.md` (use the [GitHub-standard template](https://github.com/github/.github/blob/main/CONTRIBUTING.md.template)). Cover: how to file issues, branch naming, commit style, PR process, local testing.
2. Create `CODE_OF_CONDUCT.md` based on the [Contributor Covenant](https://www.contributor-covenant.org/).
3. Create `SECURITY.md` with disclosure policy (e.g., "Report security issues to security@ifoa.example.org").

**Acceptance criteria.**

- [ ] All three files exist at repo root
- [ ] Linked from the README

**Effort.** ~30 minutes (mostly text).

---

### Issue 23 — Add `data/README.md` with per-CSV schema

**Problem.** Five CSVs in `data/raw/`, zero schema docs. New contributor has no idea which dataset has which target, how many rows, or where it came from.

**Proposed fix.**

Create `data/README.md` with one section per CSV:

```markdown
## eudirectlapse.csv
- **Source:** `CASdatasets` R package, accessed via `pyreadr`
- **Rows:** 23,160
- **Target:** `lapse` (binary, 12.8% positive rate)
- **Features:** 18 (mix of numeric and categorical)
- **Acquisition:** See `docs/REPLICATION_SETUP_GUIDE.md`
- **SHA256:** ...
```

Repeat for the other 4.

**Acceptance criteria.**

- [ ] `data/README.md` exists
- [ ] All 5 CSVs documented with rows, target, source, features

**Effort.** ~1 hour.

---

### Issue 24 — Populate `src/__init__.py`

**Problem.** `src/__init__.py` is a 0-byte empty file. The package is not initialized.

**Proposed fix.**

```python
"""ADSWP TabPFN experiments — actuarial / insurance use cases."""

__version__ = "0.1.0"
__all__ = [
    "data_loader",
    "data_loader_class",
    "evaluation_metrics",
    "model_training",
    "baseline_config",
    "baseline_utils",
    "cleanup_outputs",
]
```

**Acceptance criteria.**

- [ ] `src/__init__.py` has a module docstring, `__version__`, `__all__`
- [ ] `import src; src.__version__` works

**Effort.** ~10 minutes.

---

### Issue 25 — Delete dead `BaselineConfig` wrapper

**Problem.** `src/baseline_utils.py:22–32` defines a `BaselineConfig` class that is a single-line wrapper around `config_module.set_random_seeds`. Not referenced anywhere else.

**Proposed fix.**

Delete the class. If any caller uses it, replace with a direct call to `set_random_seeds()`.

**Acceptance criteria.**

- [ ] `BaselineConfig` removed from `src/baseline_utils.py`
- [ ] No callers broken
- [ ] Tests pass

**Effort.** ~5 minutes.

---

## Tier 3 — Polish (Production Quality)

### Issue 26 — Add `CITATION.cff` and citation block to README

**Problem.** No way to cite this work. `docs/REPLICATION_SETUP_GUIDE.md` has a BibTeX entry but the main README does not.

**Proposed fix.**

1. Create `CITATION.cff` at repo root (GitHub auto-renders this on the repo page):
   ```yaml
   cff-version: 1.2.0
   message: "If you use this work, please cite it as below."
   title: "TabPFN for Insurance / Actuarial Use Cases (ADSWP)"
   authors:
     - family-names: Hawes
       given-names: Scott
   type: software
   ```
2. Add a `## Citation` section to the README with the BibTeX entry.

**Acceptance criteria.**

- [ ] `CITATION.cff` exists and is valid (verified at https://citation-file-format.github.io/cff-initializer-javascript/)
- [ ] README has a `## Citation` block

**Effort.** ~1 hour.

---

### Issue 27 — Add `.python-version` and a minimal `Dockerfile`

**Problem.** No Python version pin, no containerized reproducibility. A new contributor's `python --version` may differ from a year-old commit, silently.

**Proposed fix.**

1. Create `.python-version`:
   ```
   3.11.9
   ```
2. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11.9-slim
   WORKDIR /app
   COPY pyproject.toml .
   COPY src/ src/
   COPY scripts/ scripts/
   COPY notebooks/ notebooks/
   COPY data/ data/
   RUN pip install --no-cache-dir -e .
   CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
   ```
3. Add a "Docker" section to the README.

**Acceptance criteria.**

- [ ] `.python-version` exists
- [ ] `Dockerfile` builds (`docker build -t tabpfn-adswp .`)
- [ ] README documents the Docker usage

**Effort.** ~2 hours.

---

### Issue 28 — Add `Makefile` for one-button reproduction

**Problem.** No `Makefile`, no `tox.ini`, no `noxfile.py`. Reproduction is ad-hoc.

**Proposed fix.**

Create `Makefile`:

```makefile
.PHONY: install test lint format reproduce-paper clean

install:
	pip install -e .[all]

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

format:
	ruff format src/ scripts/

reproduce-paper:
	jupyter execute notebooks/adswp_project/REPLICATION_There_Is_Life_in_the_Old_GLM_Yet.ipynb

clean:
	rm -rf .pytest_cache/ htmlcov/ .coverage
```

**Acceptance criteria.**

- [ ] `Makefile` exists with `install / test / lint / format / reproduce-paper / clean` targets
- [ ] Each target runs the documented command
- [ ] README documents `make install`, `make test`, `make reproduce-paper`

**Effort.** ~1 hour.

---

### Issue 29 — Add `pytest-cov` and coverage reporting

**Problem.** No coverage tooling. Once Issue 5 (smoke test) is in, we want a baseline and a way to track growth.

**Proposed fix.**

1. Add `pytest-cov` to `[project.optional-dependencies]` in `pyproject.toml`.
2. Add `[tool.coverage.run]` and `[tool.coverage.report]` to `pyproject.toml`.
3. Update the test job in CI to run `pytest tests/ --cov=src --cov-report=term-missing`.

**Acceptance criteria.**

- [ ] `pytest tests/ --cov=src` shows a coverage report
- [ ] CI uploads the report (or prints it)

**Effort.** ~1 hour.

---

### Issue 30 — Parameterize the REPLICATION notebook (papermill)

**Problem.** All notebooks have hardcoded paths and parameters. The REPLICATION notebook has 36 cells, no `parameters` tag.

**Proposed fix.**

1. Identify the hardcoded values in the REPLICATION notebook: dataset path, target column, seed, train/test split, device.
2. Convert the first cell into a `parameters` cell using the papermill convention (`# parameters` comment).
3. Add `papermill` to `pyproject.toml` optional dependencies.

**Acceptance criteria.**

- [ ] The REPLICATION notebook can be executed via `papermill execute REPLICATION_There_Is_Life_in_the_Old_GLM_Yet.ipynb output.ipynb -p seed 45 -p device cpu`

**Effort.** ~2 hours.

---

### Issue 31 — Add notebook smoke test (headless run)

**Problem.** 14 notebooks, zero automated smoke test. A typo in cell 1 breaks everything and nobody knows until a human opens it.

**Proposed fix.**

1. Add `nbval` or `pytest-notebook` to dev dependencies.
2. Write `tests/test_notebooks.py` that runs each notebook against a 100-row fixture and asserts successful execution.
3. Add this to the CI test job.

**Acceptance criteria.**

- [ ] `pytest tests/test_notebooks.py` runs all 14 notebooks and passes
- [ ] CI runs the smoke test on every PR

**Effort.** ~2 hours.

---

### Issue 32 — Enable Dependabot auto-merge for patch updates

**Problem.** `.github/dependabot.yml` is configured but not configured to auto-merge. Minor/patch PRs sit unmerged.

**Proposed fix.**

In `.github/dependabot.yml`, add:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    # Allow auto-merge
    labels:
      - "dependencies"
```

And enable auto-merge in the repo's Settings → General → Pull Requests → "Allow auto-merge".

**Acceptance criteria.**

- [ ] Dependabot PRs are auto-mergeable for patch updates
- [ ] `SECURITY.md` documents the policy

**Effort.** ~1 hour.

---

### Issue 33 — Move import-time I/O under `if __name__ == "__main__":`

**Problem.** `scripts/analyze_round3_results.py` calls `pd.read_csv(...)` at module top level. Importing the file triggers I/O. Surprising and untestable.

**Proposed fix.**

Wrap the top-level call in:

```python
if __name__ == "__main__":
    df = pd.read_csv(...)
    ...
```

**Acceptance criteria.**

- [ ] `import analyze_round3_results` no longer triggers I/O
- [ ] Running the script directly still produces the same output

**Effort.** ~10 minutes.

---

### Issue 34 — Move admin docs to `docs/admin/`

**Problem.** `docs/funding_request.md`, `docs/procurement_request_A100-80.md`, `docs/provisioning_gpu.md` are institutional admin docs mixed with technical docs.

**Proposed fix.**

```bash
mkdir -p docs/admin
git mv docs/funding_request.md docs/admin/
git mv docs/procurement_request_A100-80.md docs/admin/
git mv docs/provisioning_gpu.md docs/admin/
```

Update `docs/README.md` (Issue 12) to mention `admin/`.

**Acceptance criteria.**

- [ ] Three files moved to `docs/admin/`
- [ ] `docs/README.md` updated

**Effort.** ~10 minutes.

---

### Issue 35 — Delete superseded `TabPFN_Classifier_on_eudirectlapse_v1_0.ipynb`

**Problem.** `notebooks/adswp_project/TabPFN_Classifier_on_eudirectlapse_v1_0.ipynb` is a v1 of `01_TabPFN_classifier_eudirectlapse.ipynb`. Confusing versioning.

**Proposed fix.**

```bash
git rm notebooks/adswp_project/TabPFN_Classifier_on_eudirectlapse_v1_0.ipynb
```

**Acceptance criteria.**

- [ ] File is deleted
- [ ] No references to it elsewhere

**Effort.** ~5 minutes.

---

### Issue 36 — Gitignore `catboost_info/`

**Problem.** `notebooks/baseline_experiments/catboost_info/` is auto-generated CatBoost training metadata, committed to git. Bloat.

**Proposed fix.**

```bash
echo "**/catboost_info/" >> .gitignore
git rm -r --cached notebooks/baseline_experiments/catboost_info/
```

**Acceptance criteria.**

- [ ] `.gitignore` contains `**/catboost_info/`
- [ ] `git ls-files | grep catboost_info` returns nothing

**Effort.** ~10 minutes.

---

## Importing to GitHub Issues (when enabled)

Once Issues are re-enabled on the repo, each issue can be created with:

```bash
gh issue create --repo IFoA-ADSWP/TabPFN \
  --title "[chore] Add LICENSE file (MIT or Apache-2.0)" \
  --label "documentation" \
  --body-file issues/01-license.md
```

A helper script to bulk-create from this file is straightforward. Run:

```bash
# After enabling Issues:
python scripts/import_backlog_to_github.py docs/MAINTENANCE_BACKLOG.md
```

(A future enhancement.)

---

## Tracking this backlog

Recommended workflow:

1. **Pick an issue.** Copy its title into a new branch: `git checkout -b chore/add-license`
2. **Work it.** Follow the proposed fix and acceptance criteria.
3. **PR.** Reference the issue number (or this file's section) in the PR description.
4. **Mark done.** Update the status column in the summary table.
5. **Sequence.** Work Tier 1 first (unblocks everything), then Tier 2 (UX), then Tier 3 (polish).
