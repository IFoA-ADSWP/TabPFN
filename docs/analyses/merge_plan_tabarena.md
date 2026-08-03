# TabArena Benchmark Branch — Staged Merge Plan

Coordination/ops document — not a research report; no `docs/reports/REPORT_REGISTRY.md` row is added for this file.
Branch: `feat/tabarena-benchmark` vs `origin/main` — 35 commits, 79 files. Diff is fully additive: 76 files added, 3 modified (`README.md`, `scripts/README.md`, `docs/reports/REPORT_REGISTRY.md`) plus `TASKS.md`. ~+466K lines, bulk of which is 5 committed CSVs in `data/raw/` (~457K rows total). No conflict risk — two merge commits have already reconciled the branch with `origin/main`.

## 1. What this branch contains

All research work for issue #27 (TabArena insurance frontier benchmark).

- **Data** — `data/raw/` 5 new CSVs:

  | Dataset | Rows | Notes |
  |---|---|---|
  | `uslapseagent` | 29K | |
  | `bemtl97` | 163K | carries known label leak (`nclaims`/`amount`) — excluded from conclusions |
  | `bemtl16` | 59K | |
  | `ausautoBI8999` | 22K | |
  | `norauto` | 184K | |

  Plus `scripts/prepare_insurance_datasets.py`, which regenerates them from the CASdatasets `.rda` files.

- **Benchmark scripts** (`scripts/`):

  | Script | Purpose |
  |---|---|
  | `run_smoke_tabarena.py` | pipeline validation (coil2000) |
  | `run_lapse_benchmark.py` | lapse classification + regression |
  | `run_tabarena_insurance_benchmark.py` | v1 baseline: 9 tasks, 8 models (TabPFN / GBDTs / GLMs / RF) |
  | `run_tabarena_insurance_imbalance_pilot.py` | imbalance hypothesis; reuses v1 splits |
  | `run_home_turf_size_sweep.py` + `finish_home_turf_sweep_v2.py` | 3 datasets × 3 sizes × 5 folds |
  | `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py` | issue #27 deliverable — Pareto frontiers, `--regression` flag, D1–D5 |
  | `scripts/eval/insurance_benchmark_v1/rescore_focused_imbalance_logloss.py` | log-loss / Brier re-score of the pilot; post-run only — needs uncommitted `scripts/experiments/` caches |

- **Evidence** — `scripts/eval/{smoke_test, lapse_benchmark_v1, insurance_benchmark_v1, insurance_imbalance_pilot}/`: ~60 files — leaderboards, `results_per_split.csv`, 10 `frontier_results_<dataset>.csv` + PNG plots, Pareto PDFs/HTML explorers. `insurance_benchmark_v1` covers 9 task ids / 7 datasets; the frontier covers 10 datasets (6 classification + 4 regression), 9 methods for classification / 8 regressors.

- **Docs** — `docs/analyses/insurance_frontier_benchmark_spec.md` (D1–D5 design), `docs/analyses/tabpfn_vs_gbdt_baselines_finetuning.md` (master report, §1–§14.8), `docs/sessions/2026-07-28-tabarena-benchmark-setup.md`, a `REPORT_REGISTRY.md` row, and the `TASKS.md` #27 row.

## 2. Staged implementation plan (5 stages, in merge order)

| Stage | Contents | Rationale |
|---|---|---|
| 1 — Data + prep | 5 CSVs, prep script, `data/README.md` update | nothing runs without data |
| 2 — Core harness (v1) | smoke + lapse + insurance benchmark scripts, eval outputs, API-key docs | v1 is imported by the pilot |
| 3 — Hypothesis work | imbalance pilot + size sweep + results | sweep feeds Stage 4 |
| 4 — Frontier (issue #27 deliverable) | frontier script, results + plots, rescore script | depends on sweep results |
| 5 — Docs umbrella | spec, master report, session notes, registry row, `TASKS.md` | merge last so references don't dangle |

### 2.1 Stage 1 — Data + prep

- 5 CSVs in `data/raw/`, `scripts/prepare_insurance_datasets.py`, and a `data/README.md` update.
- Rationale: nothing runs without data.
- `data/README.md` currently documents neither the 5 datasets nor the beMTPL16-vs-`bemtl16` naming collision.

### 2.2 Stage 2 — Core harness (v1)

- `run_smoke_tabarena.py` (pipeline validation, coil2000), `run_lapse_benchmark.py` (lapse class+reg), `run_tabarena_insurance_benchmark.py` (v1 baseline: 9 tasks, 8 models — TabPFN/GBDTs/GLMs/RF), plus their eval outputs.
- README API-key section: `TABPFN_API_KEY` env / repo `.env` / `TABPFN_ENV_FILE`; scripts fail with a clear error if absent.
- Rationale: v1 is imported by the pilot (Stage 3).
- v1 verdict: 2 wins / 1 tie / 5 losses vs GBDT baselines; TabPFN 5–50× train, 100–1000×+ inference; `bemtl97` label leak identified here (§6).

### 2.3 Stage 3 — Hypothesis work

- Imbalance pilot (`run_tabarena_insurance_imbalance_pilot.py`) + results; size sweep (`run_home_turf_size_sweep.py` + `finish_home_turf_sweep_v2.py`) + finishers + `home_turf_sweep_results.csv`.
- Pilot reuses v1 splits; sweep feeds Stage 4.
- Findings: imbalance null on 1−AUC; `balance_probabilities` hurts calibration (log loss 0.4716 vs 0.2008 on coil2000); sweep TabPFN wins 8/9 cells at ≤5K rows.

### 2.4 Stage 4 — Frontier (issue #27 deliverable)

- `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py` (Pareto frontiers, `--regression` flag, D1–D5), 10 `frontier_results_*.csv` + plots, `rescore_focused_imbalance_logloss.py` + `focused_imbalance_logloss.csv`.
- Depends on sweep results: reuses power for cat/lgbm/xgb/tabpfn on bemtl97/coil2000/uslapseagent; fresh fits for the rest.
- Headlines: lead erodes at scale (norauto LGBM wins; ausprivauto0405 7-param GLMs beat 10M-param TabPFN).
- Regression Phase 2 (§14.8): wins ausautoBI8999, vehvalue within SE (v1's +67.2% not reproduced — protocol-specific), dominated at scale on bemtl97_amount + freMTPL2freq.

### 2.5 Stage 5 — Docs umbrella

- Spec, master report, session notes, registry row, `TASKS.md`.
- Merge last so references don't dangle.

## 3. Pre-merge fixes (colleagues would trip on these)

1. `scripts/README.md` lists only 2 of the 9 new scripts — add the other 7.
2. 5 run logs (`frontier_benchmark_run.log`, `frontier_norauto_run.log`, `frontier_regression_run.log`, `frontier_v1suite_run.log`, `home_turf_sweep_run.log`) are gitignored (`*.log`) but cited as evidence in §14.5–14.8 and the registry → decide: `git add -f` or mark them regenerable in the docs.
3. `rescore_focused_imbalance_logloss.py` needs uncommitted `scripts/experiments/` caches → add a "post-run only" docstring note.
4. `data/README.md` gap (Stage 1) + beMTPL16 vs `bemtl16` naming collision.
5. Vehvalue discrepancy (§4 +67.2% vs §14.8 within SE) — flag as protocol-specific, not a contradiction.
6. `TABPFN_API_KEY` required on fresh checkout for all TabPFN runs (documented in README; no secrets in repo — the only API-key string in history is a redacted placeholder).

## 4. What to communicate to colleagues (science summary)

- **10-dataset frontier** (6 classification + 4 regression), 9/8 methods; actuary takeaways in §14.4.
- **Home-turf sweep:** TabPFN wins 8/9 cells at ≤5K rows; the single loss is bemtl97@full (LGBM 0.3418 vs 0.3428); 3,045 s cold fit at 163K rows.
- **At-scale erosion:** norauto (LGBM wins), ausprivauto0405 (GLMs dominate), regression frontier (lead does not survive except ausautoBI8999).
- **bemtl97 label leak excluded;** imbalance null + calibration harm; fine-tuning degrades 3/4 targets (§5).
- **Overall verdict (§8):** do not adopt default TabPFN as a general insurance engine; fix the protocol before re-testing.
- **Operations:** `TABPFN_API_KEY` env setup; CPU runs; macOS libomp `DYLD_LIBRARY_PATH` workaround.

## 5. Merge mechanics

Documented flow (not executed):

```bash
git fetch origin
git merge origin/main     # currently resolves cleanly
# PR / merge to main
```

- Merge `origin/main` into the branch first, resolve any conflicts (currently none), then PR to `main`.
- `TASKS.md` #27 row currently says "unpushed" — update it to reflect the staged plan status.
- Low risk: additive-only diff, no existing source files modified.
