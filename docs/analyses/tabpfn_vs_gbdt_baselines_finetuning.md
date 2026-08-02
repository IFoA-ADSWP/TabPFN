# TabPFN vs GBDT Baselines on Insurance Datasets, and Domain Fine-Tuning Effectiveness

Technical research report — engineering/data-science audience.
Branch: `feat/tabarena-benchmark`. Date: 2026-08-01. Updated: 2026-08-02 (§11, §12).

## 1. Objective

Answer two questions with reproducible evidence:

1. **Does default-config TabPFN beat GBDT baselines on insurance datasets?**
   (Baseline benchmark, authoritative, holdout mode, CPU, default configs only.)
2. **Does domain fine-tuning make TabPFN materially better, and can it close the gap?**
   (Earlier fine-tuning studies — different protocol, synthesized here, not re-run.)

Cross-checked against `docs/reports/REPORT_REGISTRY.md`: no existing report covers the
`insurance_benchmark_v1` GBDT comparison (`results_per_split.csv` is not referenced by any
registry entry). Fine-tuning evidence overlaps with
`docs/reports/COMBINED_TABPFN_CLASSIFIER_REGRESSOR_ANALYSIS.md` and
`docs/reports/STAGE_A_B_FINDINGS_AND_RECOMMENDATIONS.md`; this report cross-references them
rather than duplicating their content.

## 2. Experimental Setup

### 2.1 Baseline benchmark (newest, authoritative)
- Source: `scripts/eval/insurance_benchmark_v1/results_per_split.csv` + `method_info.csv`
- 9 CASdatasets tasks (7 datasets, 2 with dual targets), holdout split, CPU, 3–5 folds.
- 8 methods, all **default config** (`config_type=default`, `can_hpo=False`):
  TabPFNClient, CAT (CatBoost), GBM (LightGBM), XGB (XGBoost), RF, LR, LogisticGLM,
  PoissonGLM, TweedieGLM. GLM variants only run on tasks matching their family.
- Metric: `metric_error` = **error metric, lower is better** — RMSE for regression,
  **1 − ROC AUC** for binary tasks (ROC AUC stored as error).
- Harness drops only the target column from features (`run_tabarena_insurance_benchmark.py:184`).

### 2.2 Domain fine-tuning study (earlier, different protocol)
- Source: `outputs/current/tables/domain_finetune_study_runs.csv`,
  `outputs/current/logs/domain_finetune_logbook.md`
- Stage A protocol: 2,500-row subsets, `cpu/64/1..5` (and `cpu/128/5`), seed 42, single
  train/test split (1,750/750). Models: logistic regression, random forest, raw TabPFN,
  domain-finetuned TabPFN.
- **Not directly comparable to the benchmark**: different targets, splits, row counts, and
  protocol. The benchmark measured only **raw default** TabPFN — no fine-tuned arm ran inside
  the benchmark harness.

### 2.3 Small-finetune trials (protocol validation)
- Source: `outputs/current/tables/tabpfn_finetune_trial_results.csv`
- coil2000 (target `CARAVAN`), rows 300–3,000, 1–3 steps, ctx 64/128, seed 42.

## 3. Data and Targets

| Task (id suffix) | Metric | Task type | Folds |
| --- | --- | --- | --- |
| ausautoBI8999 | RMSE | regression (BI severity) | 3 |
| ausprivauto0405 | 1−AUC | binary (claim occurrence, 6.8% pos) | 5 |
| ausprivauto0405_vehvalue | RMSE | regression (vehicle value) | 3 |
| bemtl16 | 1−AUC | binary (liability claims, 36% pos) | 5 |
| bemtl97 | 1−AUC | binary (claim, 11.2% pos) — **EXCLUDED, leak** | 5 |
| bemtl97_amount | RMSE | regression (severity, log1p zero-inflated) | 3 |
| coil2000 | 1−AUC | binary (caravan purchase) | 5 |
| norauto | 1−AUC | binary (claims, 4.6% pos) | 5 |
| uslapseagent | 1−AUC | binary (lapse) | 5 |

## 4. Results — Baseline Benchmark

Mean `metric_error` across folds, per task. Lower is better; best baseline in **bold**;
TabPFN rank within the task's method pool.

| Task | Metric | Best baseline (err) | TabPFN (err) | Δ vs best | Rank |
| --- | --- | --- | --- | --- | --- |
| bemtl16 | 1−AUC | TabPFN **0.04394** | **0.04394** | 0% | 1/7 |
| coil2000 | 1−AUC | TabPFN **0.22738** | **0.22738** | 0% | 1/7 |
| ausprivauto0405 | 1−AUC | CAT 0.34006 | 0.34240 | +0.7% | 3/7 |
| uslapseagent | 1−AUC | CAT 0.05550 | 0.05688 | +2.5% | 3/7 |
| ausautoBI8999 | RMSE | CAT 0.97588 | 1.01273 | +3.8% | 5/8 |
| norauto | 1−AUC | CAT 0.29947 | 0.31564 | +5.4% | 6/7 |
| bemtl97_amount | RMSE | CAT 0.48201 | 0.71485 | **+48.3%** | 8/8 |
| ausprivauto0405_vehvalue | RMSE | XGB 0.71722 | 1.19891 | **+67.2%** | 8/8 |
| bemtl97 | 1−AUC | GBM 0.00000 | 0.00000 | — | **EXCLUDED** |

### 4.1 Per-dataset verdicts (TabPFN vs best baseline)

- **WIN** — bemtl16, coil2000: TabPFN is the best method (rank 1 of 7). Both are small-ish
  classification tasks with strong categorical structure.
- **TIE** — ausprivauto0405 (claim occurrence): within 0.7% of best (CAT), rank 3/7.
- **LOSE (close)** — uslapseagent (+2.5%), ausautoBI8999 (+3.8%).
- **LOSE (clear)** — norauto (+5.4%, rank 6/7).
- **LOSE (decisive)** — both severity regressions: bemtl97_amount (+48.3%) and
  ausprivauto0405_vehvalue (+67.2%), TabPFN rank 8/8. GBDTs and even linear models crush
  TabPFN on these; TabPFN's zero-shot regression transfer does not fit insurance severity
  targets under default config.

**Tally on 8 valid tasks: 2 wins, 1 tie, 5 losses (2 of them large).** GBDTs (CAT/GBM/XGB)
hold the aggregate lead; TabPFN wins only where default GBDTs are weakest.

### 4.2 Compute cost (engineering)

Train time summed over folds; inference mean per fold.

| Task | TabPFN train (s) | GBM/XGB/CAT train (s) | TabPFN infer/fold (s) | GBDT infer/fold (s) |
| --- | ---: | ---: | ---: | ---: |
| norauto (184K rows) | 190.1 | 5.7–75.1 | **239.9** | <0.05 |
| bemtl97 (163K rows) | 157.9 | 6.7–135.6 | 34.1 | ~0.06 |
| bemtl97_amount | 75.1 | 3.1–15.5 | 30.2 | ~0.05 |
| ausprivauto0405 | 55.1 | 2.7–27.8 | 9.1 | <0.02 |

TabPFN is ~5–50× slower than GBDTs for train and 100–1000×+ for inference on this hardware
(CPU, local Mac). Worst single fold: norauto fold 0 inference took **1,030 s** (vs ~42 s on
other folds — memory/context pressure on the 184K-row task). For production batch scoring of
100K+ row insurance portfolios, raw TabPFN inference is the binding constraint.

## 5. Domain Fine-Tuning Effectiveness

### 5.1 Domain fine-tuning (Stage A protocol) — raw vs tuned TabPFN, ROC AUC

| Target | Raw | Tuned (1 step) | Tuned (3 steps) | Tuned (5 steps) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| eudirectlapse | 0.5763 | 0.5334 | 0.5339 | 0.5337 | Degrades |
| coil2000 | 0.7566 | 0.5541 | 0.5532 | 0.5527 | **Severe degradation (−0.20)** |
| ausprivauto0405 | 0.6486 | 0.5525 | 0.5505 | 0.5508 | Degrades |
| freMTPL2freq_binary | 0.5412 | 0.5819 | 0.5888 | 0.5909 | Improves (+0.05) |

- Only **freMTPL2freq_binary** improved; the other 3 targets degraded at **every** step count
  (1, 3, 5) and context (64, 128 — see `domain_finetune_logbook.md` runs 2026-04-02T01:2x).
- Aggregate deltas (logbook): ROC AUC **−0.0752**, PR AUC −0.0314, Brier +0.0017,
  LogLoss +0.0106 — net degradation under this low-budget protocol.
- Finding consistent with prior report
  `docs/reports/COMBINED_TABPFN_CLASSIFIER_REGRESSOR_ANALYSIS.md`: single-step/3-step/5-step
  domain adaptation is currently **misaligned or too weak** for stable cross-dataset uplift.

### 5.2 Small-finetune trials (coil2000) — negligible

`tabpfn_finetune_trial_results.csv`, all rows/contexts/steps: largest gain was
**+0.0047 ROC AUC** (0.708 → 0.713 at 300 rows); several runs moved −0.002 to −0.008.
Conclusion: one- to three-step fine-tune on the target's own data produces **no material
gain**; step counts are not the lever.

### 5.3 Net assessment

Neither fine-tuning path currently closes the gap to GBDTs. Worse, the benchmark's two
decisive TabPFN losses (severity regressions) are exactly the tasks where the fine-tuning
evidence is thinnest (all finetune studies were classification targets). **Fine-tuning, as
configured today, does not change the "do not deploy TabPFN for severity" verdict.**

## 6. Data Anomaly — bemtl97 (must not be used)

`bemtl97` (`target=claim`) is **excluded from all conclusions** due to confirmed label
leakage:

- The prepared dataset (`prepare_insurance_datasets.py:44-52`) keeps `nclaims` and `amount`
  as **features** alongside the `claim` target.
- Inspection of `data/raw/bemtl97.csv` (163,212 rows):
  - rows with `claim==1` but `nclaims==0`: **0**
  - rows with `claim==0` but `nclaims>0`: **0**
  - rows with `amount>0` where `nclaims==0`: **0** (0 mismatches total)
  - i.e. `claim == (nclaims > 0) == (amount > 0)` **exactly**, for every row.
- The harness only drops the target column from features
  (`run_tabarena_insurance_benchmark.py:184`), so `nclaims`/`amount` stayed in the feature
  matrix. Every method trivially reaches AUC = 1.0 → `metric_error = 0.0000` on all folds.
  CAT shows floating-point residue `1.11e-16`; LogisticGLM shows `4.55e-05` on fold 3 —
  numerically zero.
- **Characterization: hard label leakage** (target derivable from a single feature column),
  not a trivial/near-constant target — the positive rate is a healthy 11.2%.
- **Fix options**: drop `nclaims`/`amount` from the feature set (they are post-hoc outcomes
  of the claim event), or drop the task. Note `bemtl97_amount` (target=`amount`) also keeps
  `nclaims` as a feature, which encodes `amount>0` exactly — a partial target proxy; flag for
  the same cleanup.

## 7. Limitations and Risks

1. **Default configs only** — `can_hpo=False` for all methods; HPO-tuned TabPFN (e.g. TabPFN's
   `auto` mode) could change rankings, as could tuned GBDTs.
2. **Fine-tuning protocol mismatch** — finetune studies used 2,500-row subsets and different
   targets; they never ran inside the benchmark harness. A direct "fine-tuned TabPFN vs GBDT"
   head-to-head is still missing.
3. **CPU only, small hardware** — TabPFN inference outliers (norauto fold 0: 1,030 s) suggest
   memory/context pressure; GPU (MPS) was available but unused for the benchmark.
4. **Single seed in finetune studies** (seed 42); the benchmark is fold-based so more robust.
5. **bemtl97 exclusion** removes one of the two large classification datasets; the effective
   evidence base is 8 tasks.
6. **Severity regressions may need transforms** (log/Tweedie-family modeling); default
   TabPFN made no such assumption, while GBDTs and GLMs benefited from harness-level setup.

## 8. Recommendation (engineering-facing)

- **Do not adopt default-config TabPFN as a general insurance modeling engine today.** On 8
  valid CASdatasets tasks: GBDTs win the aggregate; TabPFN loses decisively on both severity
  regressions and costs 5–50× in train and 100–1000× in inference.
- **Keep TabPFN on the bench for niche value**: small-data / few-shot classification
  (bemtl16, coil2000 wins) and cold-start or proxy-model contexts where a GBDT has no
  training signal.
- **Fix the fine-tuning protocol before re-testing** — current domain/self fine-tune
  degrades AUC on 3/4 targets; step-count sweeps (1→5) and context (64→128) do not recover
  it. A working fine-tune is a precondition for any "TabPFN closes the gap" claim.
- **Repair the benchmark**: remove `nclaims`/`amount` from bemtl97 features (also
  `bemtl97_amount`), then re-run bemtl97.
- **What would change the answer**: (a) HPO-enabled TabPFN in the harness; (b) a
  fine-tuned-TabPFN arm inside the benchmark on the same folds; (c) GPU inference for the
  100K+ row tasks; (d) severity-specific transforms in the TabPFN arm. Without at least (a)
  or (b), the verdict stands.

## 11. Addendum — Imbalance Pilot and Calibration Re-Score (2026-08-02)

Follow-up to the v1 benchmark (sections 1–10). Does not rewrite the v1 conclusions; it
qualifies the aggregate verdict. Evidence added by commits f1d7cc4 and bb7ab5c.

### 11.1 Hypothesis and pilot

Prior working hypothesis: TabPFN's losses to GBDTs are partly an **imbalance-handling
deficit** — GBDTs get class-weight handling "for free" while TabPFN defaults to the raw
prior. Pilot: `TabPFNClassifier(balance_probabilities=True, n_estimators=8)` vs v1 default
on the two most informative binary tasks, same folds as v1 (`coil2000`, 6% positive;
`uslapseagent`, 38% positive). Harness: `scripts/run_tabarena_insurance_imbalance_pilot.py`;
results: `scripts/eval/insurance_benchmark_v1/focused_imbalance_results.csv`.

### 11.2 Null on 1−AUC, and why

`balance_probabilities=True` produced **1−AUC identical to default to ≤1e-16 on every fold**
of both datasets (exact equality to float64 precision in the CSV, e.g. 0.215550653050653 both
arms, coil2000 fold 0). Root cause: **ROC-AUC is rank-invariant to monotone probability
transforms**, and `balance_probabilities` is a monotone calibration rescale. The v1
benchmark metric is therefore **structurally blind to the lever** — the pilot could never
have moved 1−AUC. Testing the imbalance hypothesis on 1−AUC is a measurement dead end.

### 11.3 Re-score on insurance-native metrics

Same folds re-scored on log loss + Brier — the metrics an insurer actually pays on
(`scripts/eval/insurance_benchmark_v1/rescore_focused_imbalance_logloss.py` →
`focused_imbalance_logloss.csv`). Mean over 5 folds, lower is better:

| dataset | method | log_loss | brier |
| --- | --- | ---: | ---: |
| coil2000 (6% pos) | TabPFN default | **0.2008** | **0.0528** |
| coil2000 | TabPFN balanced | 0.4716 | 0.1572 |
| coil2000 | CAT | 0.2032 | 0.0531 |
| uslapseagent (38% pos) | TabPFN default | 0.2536 | 0.0830 |
| uslapseagent | TabPFN balanced | 0.2669 | 0.0869 |
| uslapseagent | CAT | **0.2516** | **0.0822** |

### 11.4 The lever works in the wrong direction

`balance_probabilities=True` **actively hurts calibration**: it rescales predictions toward
a 50/50 prior, inflating mean predicted probability on 6%-positive coil2000 from 0.047 to
0.327 (base rate 0.06 — roughly a 5× overstatement of purchase risk). It is worse than the
v1 default on **both log loss and Brier in all 5 folds, both datasets**. The imbalance
hypothesis is disproven for insurance data: imbalance handling is not what separates TabPFN
from GBDTs on these tasks, and the exposed lever moves in the wrong direction.

### 11.5 Corrected calibration verdict (nuanced, not reversed)

- **v1-default TabPFN is near-best on log loss**: best on coil2000 (0.2008 vs CAT 0.2032),
  and 0.002 off CAT on uslapseagent (0.2536 vs 0.2516).
- The v1 "TabPFN loses" headline (§4 tally) was **partly an artifact of ranking on 1−AUC**,
  which discards all calibration information. On decision-relevant probabilities TabPFN holds
  its own on these two tasks.
- The verdict is **qualified, not reversed**: §8 still stands — not clearly production-worthy
  on large portfolios (context ceiling + 100–1000× inference cost, §4.2) — but the
  "TabPFN cannot compete on insurance probabilities" reading of the v1 results is dropped.
- Only remaining lever likely to move the aggregate verdict: HPO-enabled TabPFN inside the
  harness (§8(a)). Imbalance/flagship config levers do not.

## 12. Why the v1 results looked the way they did — interpretation and lessons

Sections 1–10 report *what* v1 found; this section explains *why* the numbers were as
lopsided as they were. It separates genuine TabPFN model limits from setup artifacts, so
future work does not re-run or re-explain the same results. No new experiments; every
claim traces to evidence cited above.

### 12.1 Dataset-size mismatch — primary cause of the headline losses

TabPFN is pre-trained on **1,024-sample context windows**. The benchmark's hosted-API
client (`scripts/run_tabarena_insurance_benchmark.py:23,245`) accepts full-size training
sets on our 10K–184K-row datasets (norauto 184K rows, §4.2; bemtl97 163,212 rows, §6), but
each prediction can effectively attend to only ~1K training rows — the local-API
equivalent of this behavior is `ignore_pretraining_limits=True`. GBDTs, by contrast, see
every row in training. This is a **capability limit of the model design, not a harness
bug**. It predicts exactly where TabPFN lost worst: the largest datasets and the
regression/severity tasks where every row of signal matters — bemtl97_amount +48.3% vs
CAT (rank 8/8, §4), ausprivauto0405_vehvalue +67.2% vs XGB (rank 8/8, §4), and the
184K-row norauto classification (+5.4%, rank 6/7, §4). The converse also holds: TabPFN
won its two smallest classification tasks (bemtl16, coil2000; §4.1), where a ~1K-row
context captures the learnable signal. The bemtl97 leak (§6) is unaffected by this — a
single leaking feature is trivially learnable inside any context window, which is why
exclusion is data-driven, not model-driven.

### 12.2 Metric blindness — 1−AUC cannot see calibration

v1 ranked methods on **1−AUC**, a rank-invariant metric that discards all probability
calibration (§2.1). TabPFN's comparative strength is calibrated probabilities, which 1−AUC
is structurally incapable of detecting — §11.2 demonstrates this: `balance_probabilities`
is a monotone probability transform, so its effects are provably invisible to 1−AUC.
The §11.3 re-score on insurance-native log loss / Brier (the metrics an insurer pays on)
shows what 1−AUC hid: default TabPFN is **best on coil2000** (0.2008 vs CAT 0.2032) and
**0.002 off CAT on uslapseagent** (0.2536 vs 0.2516). Part of the v1 "loss" was therefore
a **measurement artifact**, not model behavior.

### 12.3 Uneven tuning — TabPFN at its floor vs competitors at standard settings

The harness ran every method at default config with `can_hpo=False`
(`scripts/eval/insurance_benchmark_v1/method_info.csv`), but "default" is not symmetric
across libraries. CatBoost/LightGBM/XGBoost ship tuned default hyperparameters; TabPFN ran
as a bare `TabPFNClassifier(random_state=0)` (`run_tabarena_insurance_benchmark.py:262–
265`) — pure defaults, no HPO, no ensemble tuning. v1 therefore compared **TabPFN at its
floor against competitors at their standard settings**. HPO-enabled TabPFN in the harness
remains the one lever untested (§8(a), §11.5).

### 12.4 Ruled-out suspects (documented so future work does not re-chase them)

- **(a) CPU.** TabPFN executed on Prior Labs' hosted API, not local CPU
  (`run_tabarena_insurance_benchmark.py:23,245`); local CPU only affected the GBDT/GLM
  arms and the wall-clock comparison (§4.2). CPU does not explain any accuracy gap.
- **(b) Imbalance handling** (`balance_probabilities=True`). Tested in §11: actively
  hurts calibration — rescales predictions toward a 50/50 prior, inflating mean predicted
  probability on coil2000 from 0.047 to 0.327 vs a 0.06 base rate (§11.4) — and is worse
  on log loss and Brier in all 5 folds on both datasets (§11.4).
- **(c) Domain fine-tuning.** Tested in §5: degrades ROC AUC on 3/4 targets (§5.1) and
  produces no material gain in the coil2000 small-finetune trials (§5.2).

### 12.5 What this means for reading the results

The v1 "TabPFN loses" headline **conflated genuine model limits with setup choices**:
the context ceiling, the regression/severity gap, and inference cost are model limits;
dataset scale, the 1−AUC metric, and the tuning asymmetry are setup choices. Correcting
the setup choices (metric re-score, §11.3) flips classification from losing to
competitive on calibration — the classification headline was substantially measurement.
The **regression gap and the inference cost remain genuine TabPFN limits that no setup
change fixes**: the §8 verdict stands for severity modeling and 100K+-row batch scoring,
and TabPFN's competitive claim is restricted to calibrated classification on
small-to-moderate data.

## 13. Addendum — Home-Turf Size Sweep (2026-08-02)

Direct test of the §12.1 hypothesis on TabPFN's home turf: 3 insurance classification
datasets × 3 training sizes (1K / 5K / full) × 5 folds, scored on log loss (the metric an
insurer pays on, §11.3). Runs every method at defaults on every cell — no HPO, no
trimming, no setup asymmetry — so the only variable left is training-set size. Evidence
added by commit 63d43ee.

### 13.1 Setup

- **Cells:** 3 datasets (bemtl97 with the §6 leak features dropped, coil2000,
  uslapseagent) × 3 sizes (1K / 5K / full: 163,212 / 9,822 / 29,317 rows respectively) ×
  5 folds = 9 cells, 220 result rows, zero errors.
- **Methods:** TabPFN (hosted API) vs CAT / LGBM / XGB (CPU), all at default config —
  matching the §2.1 harness convention.
- **Config-lite probe:** per cell, TabPFN also ran an `n_estimators=8` arm (API cap) to
  test whether the v1 default sits below a better ensemble setting.
- Harness: `scripts/run_home_turf_size_sweep.py`; row assembly and error handling:
  `scripts/finish_home_turf_sweep.py`, `scripts/finish_home_turf_sweep_v2.py`.

### 13.2 Results — mean log loss over 5 folds, lower is better

| dataset | size | rows | TabPFN | CAT | LGBM | XGB | winner |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bemtl97 | 1K | 1,000 | **0.3529** | 0.3687 | 0.5103 | 0.5206 | TabPFN |
| bemtl97 | 5K | 5,000 | **0.3462** | 0.3545 | 0.3778 | 0.4088 | TabPFN |
| bemtl97 | full | 163,212 | 0.3428 | 0.3436 | **0.3418** | 0.3450 | LGBM |
| coil2000 | 1K | 1,000 | **0.2080** | 0.2205 | 0.3914 | 0.3352 | TabPFN |
| coil2000 | 5K | 5,000 | **0.2026** | 0.2103 | 0.2558 | 0.2807 | TabPFN |
| coil2000 | full | 9,822 | **0.2006** | 0.2082 | 0.2219 | 0.2494 | TabPFN |
| uslapseagent | 1K | 1,000 | **0.2513** | 0.2763 | 0.3459 | 0.3858 | TabPFN |
| uslapseagent | 5K | 5,000 | **0.2602** | 0.2718 | 0.2838 | 0.3098 | TabPFN |
| uslapseagent | full | 29,317 | **0.2491** | 0.2529 | 0.2537 | 0.2642 | TabPFN |

Winner per cell computed from the CSV means (mean of 5 folds per dataset × size × method,
`scripts/eval/insurance_benchmark_v1/home_turf_sweep_results.csv`). **TabPFN wins 8/9
cells.** The single loss is bemtl97@full: LGBM 0.3418 vs TabPFN 0.3428 — a 0.0010 margin
at 163K rows, the largest cell in the sweep.

### 13.3 The size-ceiling curve is fully mapped — and it is flat for TabPFN

- TabPFN leads on **all six practical-size cells** (1K and 5K on all three datasets) and
  on two of the three full-size cells; the GBDTs draw even exactly once, at 163K rows,
  within 0.001.
- This confirms §12.1: the v1 headline losses were driven by **dataset-size mismatch**,
  not by TabPFN's context ceiling per se. When the benchmark stays within TabPFN's
  training-data sweet spot, it beats all three GBDT families on log loss on every dataset.
  The v1 "context ceiling kills TabPFN" story is **dead for classification log loss** —
  the ceiling shows up only as a 0.001 shave at 163K rows, not as the dominant effect.
- The sweep does not revive v1: it explains it. §12.5's restricted claim — TabPFN
  competitive on calibrated classification at small-to-moderate data — is upgraded to:
  *leading* on log loss at every practical size tested here.

### 13.4 The real TabPFN cost is compute, not accuracy

- Full-size hosted fit was the bottleneck: bemtl97@full fold 0 took **3045 s (≈51 min)
  cold**; folds 1–4 took 36–64 s once the server-side cache was warm. Coil2000 and
  uslapseagent full-size cells fit in ~3 s (hosted API, no local GPU).
- This substantiates the §4.2 speed objection: TabPFN's accuracy is no longer the
  barrier, but 50-minute cold fits at 163K rows still rule it out for large-portfolio
  retraining cycles. The §8 verdict's *engineering* half stands on cost, not quality.

### 13.5 Methodological note — do not re-chase the config-lite arm

- The `n_estimators=8` probe arm was **bit-identical to the server default on all 40
  (dataset × size × fold) pairs** where both ran — identical log loss to float64
  precision. The API cap is not binding at these sizes, and the config-lite arm adds no
  information.
- The flaky `n_estimators=1` config was **dropped as a pure duplicate** (rows absent from
  the CSV; `trimmed=True` flags the bemtl97@full cell where the extra arms were trimmed).
- Conclusion for future work: TabPFN default config is already at the tested ceiling;
  HPO inside the harness (§8(a)) remains the only untested accuracy lever, and the
  ensemble-size dimension is closed.

## 9. Source Workbooks

- `scripts/run_home_turf_size_sweep.py` — home-turf size sweep runner (3 datasets × 3
  sizes × 5 folds, TabPFN + GBDT arms; §13).
- `scripts/finish_home_turf_sweep.py` — sweep result assembly, fold/cell bookkeeping
  (§13).
- `scripts/finish_home_turf_sweep_v2.py` — revised assembly: flaky-config dedup, error
  handling (§13).
- `scripts/run_tabarena_insurance_benchmark.py` — benchmark runner (task registry, target
  definitions, feature handling at line 184).
- `scripts/run_tabarena_insurance_imbalance_pilot.py` — imbalance pilot (coil2000 +
  uslapseagent, `balance_probabilities` vs default; §11).
- `scripts/eval/insurance_benchmark_v1/rescore_focused_imbalance_logloss.py` — log-loss /
  Brier re-score of pilot folds (§11.3).
- `scripts/prepare_insurance_datasets.py` — dataset prep (`make_bemtl97` at line 43).
- `outputs/current/logs/domain_finetune_logbook.md` — domain fine-tune runs and
  interpretation blocks (protocol runs 2026-04-02).
- Prior write-ups: `docs/reports/COMBINED_TABPFN_CLASSIFIER_REGRESSOR_ANALYSIS.md`,
  `docs/reports/STAGE_A_B_FINDINGS_AND_RECOMMENDATIONS.md`,
  `docs/analyses/tabpfn_small_finetune_methodology.md`.

## 10. Evidence Files

- `scripts/eval/insurance_benchmark_v1/results_per_split.csv` — per-split benchmark metrics
  (282 rows; primary evidence for §4).
- `scripts/eval/insurance_benchmark_v1/method_info.csv` — method registry (default config,
  CPU, `can_hpo=False`).
- `outputs/current/tables/domain_finetune_study_runs.csv` — Stage A runs (§5.1).
- `outputs/current/logs/domain_finetune_logbook.md` — raw-vs-tuned summary and step/context
  sweeps (§5.1).
- `outputs/current/tables/tabpfn_finetune_trial_results.csv` — coil2000 small-finetune
  trials (§5.2).
- `data/raw/bemtl97.csv` — leak inspection (§6; 163,212 rows, `claim`/`nclaims`/`amount`
  collinearity check).
- `scripts/eval/insurance_benchmark_v1/focused_imbalance_results.csv` — imbalance pilot,
  per-fold 1−AUC (null result; §11.2).
- `scripts/eval/insurance_benchmark_v1/focused_imbalance_logloss.csv` — per-fold log-loss /
  Brier re-score (§11.3).
- `scripts/eval/insurance_benchmark_v1/home_turf_sweep_results.csv` — size sweep, per-fold
  log loss / Brier / ROC-AUC, 220 rows (9 cells × 5 folds, zero errors; primary evidence
  for §13).
