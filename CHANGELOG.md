# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- docs: human-voice cleanup pass — removed AI-assistant framing and phantom file references (papers/status/analyses/notebooks/data)
- docs: frontier AUC/Brier rescore — §14.11 addendum (2026-08-06): per-fold AUC/Brier in `frontier_results_*.csv`; ausprivauto0405 "DOMINATED" retracted to calibration tie + ranking edge; verdict reframed to best risk-ranking model (TabPFN AUC #1 of 9 methods on all 6 classification datasets); stale-verdict reconciliation across regime/portfolio/papers + wiki reframe (commits 4e7912c, 9037b26, cea967d, ced3ac9)
- docs: ranking-robustness addendum — §14.12 (2026-08-06): PR AUC #1 on all 6 (paired-significant 5/6), all AUC edges significant under paired fold tests, seed-stable across 3 seeds; top-decile lift dataset-dependent (4/6 rank 1) — triage claim scoped accordingly
### Added
- Metrics explainer — log loss vs AUC vs Brier, imbalance mechanism, insurance reading guide — `docs/analyses/metrics_explained.md` (wiki: Metrics-Explained)

## [v8.2.0] — 2026-08-06

Benchmark suite release: TabArena harness, parsimony frontier, lapse settlement, Spanish motor portfolio, regression Phase 2, adoption regime.

### Added
- TabArena benchmark harness + v1 suite: 6 classification + 3 regression insurance datasets, 10 models, 5-fold CV (seed 42), same-fold comparisons — `scripts/eval/insurance_benchmark_v1/`
- Parsimony frontier analysis: accuracy-per-complexity across 12 datasets, per-dataset verdicts — `scripts/eval/` frontier outputs
- Lapse settlement benchmark: eudirectlapse class + regression via TabArena bundle — `scripts/eval/lapse_benchmark_v1/`
- Spanish motor portfolio: frequency, lapse, severity variants + benchmark naming canon (issue #27a)
- Regression Phase 2: RMSE + Poisson-deviance frontiers, 4 datasets (issue #27 D4)
- Home-turf size sweep: 3 datasets × 3 sizes × TabPFN (n8/n1/default) + GBDT baselines — `home_turf_sweep_results.csv`
- Imbalance pilot — `scripts/eval/insurance_imbalance_pilot/`
- Benchmark summary one-pager for actuarial colleagues — `docs/reports/TABPFN_BENCHMARK_SUMMARY.md`
- Regime characterization + adoption guidance (issues #52/#53) — `docs/analyses/regime_characterization.md`
- TabPFN levers assessment: fine-tuning / HPO / ensembling (issue #54)
- Version-drift re-test policy (issue #55)
- `scripts/infra/make_notebook.py` — clone template notebooks (strips outputs) for new experiments
- (earlier, previously unversioned) `LICENSE` (MIT), `CHANGELOG.md`, `tests/` smoke test, wiki + backlog sync tooling

### Changed
- Scripts reorg: active benchmarks → `scripts/benchmarks/`, shared tooling → `scripts/infra/`, legacy fine-tuning → `scripts/legacy_finetuning/` (issue #27)
- Benchmark protocol: single 80/20 split → 5-fold CV; TabPFN `model_path` pinned to `v3_default`
- Replication notebook loads bundled `data/raw/eudirectlapse.csv` first — R/CASdatasets no longer required (fallback retained)
- API key removed from code; `TABPFN_API_KEY` env var with documented setup

### Removed
- Superseded `finish_home_turf_sweep.py` and `STATUS_REPORT_FINAL.md` (superseded by the report registry); `.venv/` artifacts untracked

### Fixed
- `bemtl97` label-leak dataset excluded from frontier results (leak-fixed version used, §6 master report)
- Stale and phantom file references removed across docs (human-voice cleanup, PR #64)
- Colab widget metadata stripped from notebooks

### Headline verdict
TabPFN is a small-data specialist, not a general engine: wins 8/9 size-sweep cells at ≤5K rows and the Spanish lapse + premium-regression cases, but sits off the parsimony frontier 5 of 12 times at scale. Adoption decision rule: `docs/analyses/regime_characterization.md`. One-page summary: `docs/reports/TABPFN_BENCHMARK_SUMMARY.md`.

## [0.0.0] — pre-release

Initial state. No versioned release. Pre-licence, pre-changelog, pre-tests.
