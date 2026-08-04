# TabPFN for Insurance — Benchmark Summary

One-page answer to the question: **Is TabPFN worth adopting for insurance modeling?** Written for actuarial colleagues — the full evidence and methods live in the technical report (references at the end).

**Verdict: TabPFN is a small-data specialist, not a general engine — adopt where data is scarce, not as a replacement.**

## Where TabPFN fits — at a glance

| Scenario | What we measured | Call |
|---|---|---|
| Small data (≤5K rows) | Wins 8/9 size-sweep cells | Use it |
| Lapse classification and premium regression | Spanish lapse AUC 0.7553 vs 0.7500 (5/5 folds); premium RMSE 19.07 vs 81.8 | Use it, validated on your own book |
| Large datasets (50K+ rows) | Off the parsimony frontier 5 of 12 — all at scale | Prefer GBDT/GLM |
| Regulator-facing simplicity | 21-param GLMs within noise of the 10M-param model | Prefer the simple model |

## How we tested it

We built a custom benchmark on real insurance datasets — claims, lapse, frequency and severity — with fair same-fold comparisons and insurance-native metrics (log loss, RMSE, Poisson deviance). Every model ran at default settings with no tuning, so the differences are the models', not the setup's. The core question is "best accuracy per unit of model complexity" — the parsimony frontier — because regulators and governance care about models an actuary can explain. Everything is re-runnable with one command, with results committed alongside the code.

## The evidence

- **Size sweep** — TabPFN wins 8/9 cells at ≤5K rows; the small-data regime is real.
- **Parsimony frontier** — on the frontier for 7 of 12 datasets, but off it 5× at scale; 21-param GLMs sit within noise of the 10M-param model.
- **Regression** — the lead does not survive at scale: wins `ausautoBI8999`, ties `vehvalue`, loses the rest.
- **Lapse** — a real win at 5-fold: AUC 0.7553 vs 0.7500 on Spanish motor (wins all 5 folds), plus the premium-regression win.

## Why TabPFN loses at scale

Earlier generations could effectively attend to only ~1K training rows, a context-window design limit. The v3 model's API now accepts up to 1M rows and still loses on accuracy-per-complexity and speed — so the measured pattern holds regardless of mechanism (see the §12.1 model-version correction in the master report).

## Caveats

- The `bemtl97` label-leak dataset is excluded; all frontier results use the leak-fixed version (§6 of the master report).
- The `vehvalue` discrepancy between the v1 baseline and the frontier run is protocol-specific, not a stable property of the model.
- The lapse leaderboard's ELO lead is partly carried by the premium-regression task — an artifact of the task mix, not pure lapse skill.

## Conclusion & adoption guidance

**Decision rule** (regime analysis `docs/analyses/regime_characterization.md` §3; evidence in master report §8, §14.9, §14.10):

> **Adopt default TabPFN (v3)** when data is scarce — ≤ ~5K training rows (8/9 size-sweep cells won, §13.2) — **or** on classification/lapse-style tasks where the linear floor is far from achievable: best LR/GLM ≥ ~3% behind the best model, or ≥ ~0.05 AUC gap. Every TabPFN win satisfies this — coil2000, bemtl16, uslapseagent, ausautoBI8999, vehvalue, and Spanish lapse (LR 0.684 vs TabPFN 0.7553, 5/5 folds, §14.10).
>
> **Otherwise, expect TabPFN to be dominated on the accuracy-per-complexity trade-off at scale** (≥ ~50K rows): every off-frontier point sits at ≥53.5K rows (money chart); no edge when the GLM floor is within ~2% of achievable (ausprivauto0405, eudirectlapse, norauto, bemtl97 — GLM gap ≈ 0, 0/4 wins); and no edge on frequency targets at scale even with a weak GLM floor, because the signal there is tree-extractable only (Spanish freq: LGBM 0.8916 vs TabPFN 0.9876, Poisson GLM at the 1.0123 null floor, §14.9; freMTPL2freq at 678K rows, §14.8).
>
> **Between the regimes, prefer the GLM**: the 11–86-param GLM family is never dominated on any dataset (§14.4), and it is statistically indistinguishable from TabPFN on frequency at 21 params (§14.9).

- The rule is predictive, not descriptive: it explains the Spanish motor flip at the same 53,502 rows — lapse wins (linear floor far, §14.10), frequency loses (tree-only signal, §14.9) — where a "small data only" rule would not (regime analysis §2).
- Re-test on model-version change: every number here is pinned to `v3_default` (tabpfn-client 0.3.3); re-run before trusting the rule on a new model version. The procedure — trigger, scope, diff, and addendum steps — is master report §15 ("Version-Drift Re-Test Policy", issue #55).

## What's next

The two open questions are settled: the 5-fold lapse re-run confirms the Spanish win, and the Spanish severity frontier is done. Future work: fine-tuned TabPFN inside the parsimony frontier; other foundation models (TabFM was dropped for out-of-memory — revisit on smaller data); and version re-runs — the model is pinned to `v3_default`, so re-runs are reproducible.

## References

- Master report: `docs/analyses/tabpfn_vs_gbdt_baselines_finetuning.md` (§1–§14.10; verdict §8)
- Regime analysis: `docs/analyses/regime_characterization.md` (decision rule §3, hypothesis test §2)
- Benchmark portfolio: `docs/analyses/benchmark_portfolio.md`
- Merge plan: `docs/analyses/merge_plan_tabarena.md`

### Source Workbooks

`scripts/benchmarks/run_smoke_tabarena.py`; `scripts/benchmarks/run_tabarena_insurance_benchmark.py`; `scripts/benchmarks/run_lapse_benchmark.py`; `scripts/benchmarks/run_tabarena_insurance_imbalance_pilot.py`; `scripts/benchmarks/run_home_turf_size_sweep.py`; `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py`

### Evidence Files

`scripts/eval/insurance_benchmark_v1/frontier_results_*.csv`; `scripts/eval/insurance_benchmark_v1/money_chart_tabpfn_relative.png`; `scripts/eval/lapse_benchmark_v1/results_per_split.csv`
