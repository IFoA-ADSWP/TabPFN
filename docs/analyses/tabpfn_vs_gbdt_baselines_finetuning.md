# TabPFN vs GBDT Baselines on Insurance Datasets, and Domain Fine-Tuning Effectiveness

Technical research report — engineering/data-science audience.
Branch: `feat/tabarena-benchmark`. Date: 2026-08-01. Updated: 2026-08-02 (§11, §12, §13, §14); 2026-08-03 (§14.6 norauto first extension, §14.7 v1-suite extension, §14.8 regression Phase 2).

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

**Model-version correction (2026-08-04).** The "~1K effective context" framing above
was written from a v2.5-era architecture understanding and does **not** describe the
model this benchmark actually ran. All hosted-API runs in this report used the
**hosted v3 model** (auto-selection; pinned post-hoc to `model_path="v3_default"`,
tabpfn-client 0.3.3 — see the §14.9 model version note). Live-verified API limits on
2026-08-04 via the client's `/tabpfn/get_model_limits` (ModelLimit): **v3 supports up
to 1,000,000 rows / 200,000,000 cells / 160 classes / 2,000 columns** (v2/v2.5 capped
at 50,000 rows, v2.6 at 100,000), so there is **no 1K context ceiling at the API
level** for v3. The server manages large training sets internally — plausibly a larger
v3 context window and/or internal subsampling/ensembling — but the mechanism is not
exposed by the client. What stands: the measured size-dependent pattern (TabPFN wins
small data, ties mid-size, drops off the frontier at scale) is empirical and
**mechanism-independent**. The operative v3-era limits are accuracy-per-parameter and
accuracy-per-second against GBDTs/GLMs, not a hard 1K window.

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
the context ceiling, the regression/severity gap, and inference cost are model limits
(context-ceiling item amended by the §12.1 model-version correction, 2026-08-04);
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

## 14. Addendum — Insurance Frontier Benchmark (2026-08-02)

Pareto-efficiency analysis for issue #27 (repo IFoA-ADSWP/TabPFN): predictive power
(log loss) vs parsimony (# params as a proxy for interpretability), per the agreed design
spec `docs/analyses/insurance_frontier_benchmark_spec.md` (commit 037d215, settled
decisions D1–D5). Each of the three home-turf datasets is an **independent frontier**
(separate table + plot; no cross-dataset Pareto comparison). Implementation committed in
`ed3e119` (script) with the spec's acceptance criteria 1–4 met; this addendum is
deliverable 6.

### 14.1 Setup

- **Spec:** `docs/analyses/insurance_frontier_benchmark_spec.md` — authoritative reading
  of issue #27; D1–D5 are settled, not re-derived (§14.3/14.4 check the spec's expected
  results (a) and (b) against the run).
- **D1/D2 combined pass, one script:** `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py`
  runs all three datasets, one pass each. D1 (Option B) re-runs LogisticGLM/LR,
  TweedieGLM, PoissonGLM and RandomForest (the fast families whose v1 folds did not match
  the sweep's splits); D2 (Option A) re-fits LightGBM/XGBoost/CatBoost at their recorded
  defaults purely to count parameters. GBDT/RF param count = `n_estimators × mean leaves
  per tree`; GLM/LR = post-encoding column count + 1 intercept (harness uses `cat.codes`
  → raw column count); count inputs recorded in
  `scripts/eval/insurance_benchmark_v1/method_info.csv`.
- **Reused sweep power (no re-run):** TabPFN / CAT / LGBM / XGB log loss is taken as-is
  from `scripts/eval/insurance_benchmark_v1/home_turf_sweep_results.csv` (§13), filtered
  to full size (`n_rows == len(X)`, `n_estimators.isna()`) — the new run uses the same
  splits, `StratifiedKFold(5, shuffle=True, random_state=42)`, so the D1 re-runs land on
  the identical folds.
- **New compute only:** GLMs/LR/RF power (5 folds) + leaf-count refits for all tree
  families. The GLM/LR/RF pass is cheap even at 163K rows (LR/GLMs ~1–7 s/fold on
  bemtl97; RF ~16–87 s/fold, per `frontier_benchmark_run.log`).
- **D3 Pareto rule (Option B, beyond SE):** model A is dominated iff some model B with
  strictly fewer params has `mean_B + SE_B < mean_A − SE_A`. Models within SE of each
  other both stay on the frontier; ties on both axes keep both models on. Rationale per
  spec §5: a point-estimate rule would let fold noise decide membership (bemtl97 LGBM
  0.3418 vs TabPFN 0.3428 is within fold noise).
- **TabPFN parameter count — settled non-decision (spec §5):** constant **10,000,000** per
  dataset regardless of training size — the top-right anchor (max power, min parsimony).
  It never changes frontier membership; the exact hosted-model figure is TBD from
  PriorLabs/TabPFN docs and is flagged as constant regardless.
- **Datasets:** bemtl97 (leak-fixed per §6: `claim` target, `nclaims`/`amount` dropped;
  163,212 rows, pos-rate 11.2%), coil2000 (`CARAVAN`; 9,822 rows, pos-rate ≈6%),
  uslapseagent (`surrender`; 29,317 rows). Note: the 184K-row dataset is `norauto` — the
  designated first extension per D5, **not** part of this v1 run.

### 14.2 Results — log loss mean ± SE over 5 folds, lower is better

**bemtl97** (163,212 rows, `claim`, leak-fixed):

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| lgbm | **0.34177** | 0.00053 | 3,100 | yes |
| lr | 0.34270 | 0.00040 | 11 | yes |
| logisticglm | 0.34270 | 0.00040 | 11 | yes |
| poissonglm | 0.34277 | 0.00040 | 11 | yes |
| tabpfn | 0.34279 | 0.00057 | 10,000,000 | yes |
| tweedieglm | 0.34280 | 0.00042 | 11 | yes |
| cat | 0.34363 | 0.00053 | 63,546 | no |
| xgb | 0.34503 | 0.00054 | 3,981 | no |
| rf | 0.59530 | 0.01219 | 2,245,565 | no |

**coil2000** (9,822 rows, `CARAVAN`):

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| tabpfn | **0.20059** | 0.00221 | 10,000,000 | yes |
| lr | 0.20554 | 0.00187 | 86 | yes |
| logisticglm | 0.20626 | 0.00188 | 86 | yes |
| cat | 0.20823 | 0.00244 | 63,764 | yes |
| poissonglm | 0.21093 | 0.00233 | 86 | yes |
| tweedieglm | 0.21779 | 0.00260 | 86 | yes |
| lgbm | 0.22193 | 0.00261 | 3,100 | no |
| xgb | 0.24937 | 0.00354 | 2,719 | no |
| rf | 0.47859 | 0.02993 | 71,763 | no |

**uslapseagent** (29,317 rows, `surrender`):

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| tabpfn | **0.24909** | 0.00518 | 10,000,000 | yes |
| cat | 0.25286 | 0.00459 | 63,394 | yes |
| lgbm | 0.25367 | 0.00446 | 3,100 | yes |
| xgb | 0.26419 | 0.00460 | 3,781 | no |
| rf | 0.27141 | 0.00325 | 270,218 | no |
| logisticglm | 0.27624 | 0.00506 | 11 | yes |
| lr | 0.27660 | 0.00505 | 11 | yes |
| poissonglm | 0.28584 | 0.00488 | 11 | yes |
| tweedieglm | 0.28781 | 0.00593 | 11 | yes |

Bold = best mean log loss per dataset (power winner). Frontier membership per the D3
beyond-SE rule, reproduced from `frontier_results_<dataset>.csv` (the run's self-check
asserts ≥1 frontier point per dataset).

### 14.3 Narrative per dataset

**bemtl97 — the spec's mid-complexity squeeze, confirmed.** The entire top of the table
is a 0.001-wide cluster: LGBM 0.34177, all four GLM variants and LR 0.34270–0.34280,
TabPFN 0.34279. LGBM's tiny edge puts it on the frontier; TabPFN and all six GLMs are on
because no 11-param model beats anyone else beyond SE and LGBM does not beat TabPFN beyond
SE (`0.34177+0.00053 = 0.34230` vs `0.34279−0.00057 = 0.34222` — within SE). This is
exactly the fold-noise trap the D3 rule exists to avoid. **CatBoost (63,546 params) and
XGBoost (3,981) are dominated** — beaten on power by LGBM/TabPFN *and* on parsimony by the
GLMs: a clean case of the spec's expected result (a), a real "don't deploy this" signal.
RandomForest is catastrophic (0.59530 ± 0.01219) — worst by a wide margin, dominated on
both axes, consistent with §13's sweep pattern. On expected result (b): TabPFN's §13
0.0010 lead over LGBM does **not** survive the parsimony constraint here — LGBM holds the
power edge and the GLMs hold parsimony; TabPFN survives the frontier only on the
beyond-SE tie.

**coil2000 — TabPFN leads, GLMs hold the low-complexity end, CatBoost beats LightGBM.**
TabPFN is the power winner (0.20059) and the top-right anchor. LR (0.20554) and
LogisticGLM (0.20626) at just 86 params are within ~0.006 of TabPFN — the most
interesting actuary trade-off on this dataset. CatBoost makes the frontier (0.20823, 63,764
params) because it beats LGBM (0.22193) and XGBoost (0.24937) on power; LGBM and XGBoost
are dominated (no power advantage, worse parsimony than the GLMs) — the usual GBDT pecking
order is inverted here. RandomForest again fails (0.47859 ± 0.02993). On expected result
(b): TabPFN's lead **survives** the frontier constraint — it is the best power on the
dataset, and parsimony pressure only closes the gap, it does not remove it.

**uslapseagent — the full family stack is on the frontier.** TabPFN (0.24909), CatBoost
(0.25286) and LightGBM (0.25367) are all within ~0.005 and all on the frontier, each the
most parsimonious model at its power tier (CatBoost and LightGBM dominate XGBoost: both
beat it on power with far fewer params). The GLM family anchors the parsimony end — all
four 11-param variants are on the frontier (logisticglm/lr at 0.27624/0.27660, then
poisson/tweedie within SE of them); nothing at 11 params beats them beyond SE. XGBoost and
RandomForest are dominated; RF (0.27141) beats the GLMs on power but is crushed on
parsimony (270,218 vs 11 params) — a textbook mid-complexity squeeze (expected result
(a)). On expected result (b): TabPFN's lead survives — it wins power outright, with
parsimony again closing but not removing the gap.

### 14.4 Interpretation — what an actuary should take away

- **The GLM family (11–86 params) is never dominated on any dataset.** On every frontier
  at least one GLM variant is on the curve — they anchor the low-complexity end with log
  loss within ~0.001 (bemtl97) to ~0.006 (coil2000) of the power winner. For
  governance-defensible deployment under a complexity budget, a GLM is always on the
  table. This is the single most robust result of the frontier.
- **TabPFN is on the frontier everywhere but never parsimonious.** It wins power on 2 of 3
  datasets (coil2000, uslapseagent) and is within fold-noise of the winner on the third,
  but at a constant 10,000,000 params it is always the top-right anchor — never a
  parsimony answer. The spec's expected result (b) resolves as: TabPFN's accuracy lead
  survives the frontier only where it is the outright power winner; where it merely ties
  (bemtl97), parsimony pressure removes the advantage.
- **RandomForest at default config is a frontier failure** — dominated on all three
  datasets, catastrophic on bemtl97 (0.59530), consistent with §13 and the v1 baseline.
  No further RF exploration is warranted at default settings.
- **The mid-complexity squeeze (expected result (a)) is real**: CatBoost/XGBoost are
  dominated on bemtl97, LGBM/XGBoost on coil2000, XGBoost/RF on uslapseagent — beaten on
  power by TabPFN and on parsimony by the GLMs. Mid-complexity models must earn their
  place by beating the GLMs by more than SE, and mostly they do not.
- **`norauto` (184K rows) is done (§14.6)** — the designated first extension per D5.
  LGBM takes the power edge at scale (0.17518 vs TabPFN 0.17619) and TabPFN survives the
  frontier only on the beyond-SE tie — the §12.1 size-ceiling hypothesis holds on the
  frontier axis. **The v1 classification suite is now complete (§14.6, §14.7)** —
  `ausprivauto0405` and `bemtl16` closed out the remaining post-`norauto` priority
  (spec §8). **The regression Phase 2 (D4) is now done (§14.8)** — the frontier covers
  10 datasets (6 classification + 4 regression). **Where to look next:** the spec's
  remaining open items are TabPFN's exact param count (non-decision, recorded), an
  exposure-weighted Poisson deviance / offsets refinement, and the fine-tuned TabPFN arm
  (§12.3 tuning asymmetry) — of these the highest-value is the fine-tuned TabPFN arm, or
  else close out issue #27.

### 14.5 Files & reproducibility

- **Script:** `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py` (committed
  `ed3e119`, branch `feat/tabarena-benchmark`; spec commit `037d215`).
- **Run command:** `source /tmp/tabarena/.venv-ta/bin/activate && python
  scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py`
- **Outputs** (same dir as the script): `frontier_results_bemtl97.csv`,
  `frontier_results_coil2000.csv`, `frontier_results_uslapseagent.csv`,
  `frontier_results_norauto.csv` (method | mean | se | n_params | on-frontier),
  `frontier_plot_{dataset}.png` (x = log10(n_params), y = mean
  log loss, ± SE bars, frontier red / dominated grey), `frontier_benchmark_run.log`,
  `frontier_norauto_run.log` (per-fold timings and log loss).
- **Inputs:** `data/raw/{bemtl97,coil2000,uslapseagent}.csv`;
  `home_turf_sweep_results.csv` (reused TabPFN/CAT/LGBM/XGB power, identical folds);
  `method_info.csv` (recorded defaults for the D2 leaf-count refits).
- **Self-check:** assert-based sanity checks at end of run — 5 sweep fold rows per reused
  method, no NaNs, unique methods, ≥1 frontier point, no train/test overlap within folds.
- Regression/severity frontier deliberately **not** in this pass (D4, classification-only
  v1); Phase 2 adds the RMSE/Poisson-deviance axis.

### 14.6 First extension — norauto (184K rows)

`norauto` (184,000 rows; 183,999 after dropna, 4.6% positive, target `NbClaim`, 5
features — the §6-style leak-fix is already baked into the prepared file, d170afc).
All 9 methods fit **fresh**: the home-turf sweep did not cover norauto, so there was no
reusable power (`run_frontier_benchmark.py` falls back to fresh fits per commit e93f3c0;
TabPFN via the hosted API, 5 folds, 504 s total).

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| lgbm | **0.17518** | 0.00047 | 3,100 | yes |
| tabpfn | 0.17619 | 0.00061 | 10,000,000 | yes |
| cat | 0.17676 | 0.00053 | 62,540 | no |
| xgb | 0.17739 | 0.00055 | 4,517 | no |
| logisticglm | 0.17846 | 0.00046 | 6 | yes |
| lr | 0.17846 | 0.00046 | 6 | yes |
| poissonglm | 0.18131 | 0.00035 | 6 | yes |
| tweedieglm | 0.18264 | 0.00159 | 6 | yes |
| rf | 0.73544 | 0.00929 | 593,042 | no |

Frontier = 6: LGBM, TabPFN, and all four 6-param GLMs (logisticglm/lr tied at 0.17846).
CatBoost and XGBoost are dominated — beaten on power by LGBM/TabPFN and on parsimony by
the GLMs; RF is catastrophic at scale (0.73544 ± 0.00929, 593,042 params), dominated on
both axes, consistent with §13 and §14.3. **D5 verdict — the size-ceiling hypothesis
transfers to the frontier axis:** LGBM takes the power edge at scale (0.17518 vs TabPFN
0.17619), and TabPFN survives the frontier **only** on the beyond-SE tie
(`lgbm mean+SE` = 0.17565 vs `tabpfn mean−SE` = 0.17558) — the same fold-noise
membership as bemtl97 at 163K rows (§14.3). Files:
`scripts/eval/insurance_benchmark_v1/frontier_results_norauto.csv`,
`frontier_plot_norauto.png`.

### 14.7 v1-suite extension — ausprivauto0405 + bemtl16

The remaining two v1 classification datasets, completing the post-`norauto` priority
(spec §8). Both fit **fresh** — neither was covered by the home-turf sweep, so there is
no reusable power; `run_frontier_benchmark.py` uses the same fallback path as norauto
(§14.6). Results committed in `b2d03a5`.

**ausprivauto0405** (67,856 rows, 6.8% positive, target `ClaimOcc`):

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| logisticglm | **0.23947** | 0.00038 | 7 | yes |
| lr | 0.23947 | 0.00038 | 7 | yes |
| poissonglm | 0.23956 | 0.00036 | 7 | yes |
| tweedieglm | 0.23983 | 0.00033 | 7 | yes |
| tabpfn | 0.24026 | 0.00037 | 10,000,000 | no |
| lgbm | 0.24164 | 0.00018 | 3,100 | no |
| cat | 0.24327 | 0.00047 | 62,604 | no |
| xgb | 0.24907 | 0.00051 | 4,368 | no |
| rf | 0.52831 | 0.00988 | 602,223 | no |

**bemtl16** (58,723 rows, 36.0% positive, target `number_of_liability_claims`):

| method | mean log loss | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| tabpfn | **0.23803** | 0.00129 | 10,000,000 | yes |
| lgbm | 0.23985 | 0.00112 | 3,100 | yes |
| cat | 0.24341 | 0.00101 | 63,778 | no |
| xgb | 0.25144 | 0.00114 | 4,253 | no |
| rf | 0.26121 | 0.00203 | 434,404 | no |
| logisticglm | 0.26315 | 0.00171 | 14 | yes |
| lr | 0.26319 | 0.00172 | 14 | yes |
| poissonglm | 0.30162 | 0.00383 | 14 | yes |
| tweedieglm | 0.65338 | 0.00001 | 14 | yes |

**ausprivauto0405 — the first outright TabPFN domination.** The frontier is exactly the
four 7-param GLMs (logisticglm/lr tied at 0.23947, then poissonglm 0.23956, tweedieglm
0.23983); every other method — tabpfn, lgbm, cat, xgb, rf — is dominated. This
confirms the spec's expected result (a) at its most decisive: the GLM family wins **both
axes**. TabPFN's 10,000,000-param power edge is gone — logisticglm at 7 params beats it
beyond SE (`mean_B + SE_B = 0.23984` vs `mean_A − SE_A = 0.23989`). It extends
the §12.1 "at scale" narrative one step further than norauto: at 67,856 rows / 6.8%
positive the problem is easy enough that a 7-param GLM beats 10M-param TabPFN outright.

**bemtl16 — TabPFN survives, and this time the edge is real.** TabPFN keeps the power
edge (0.23803) over lgbm (0.23985) and it is **beyond SE**, not a fold-noise tie:
`lgbm mean+SE = 0.24097` vs `tabpfn mean−SE = 0.23674`. The frontier-tie story of
§14.3/§14.6 (TabPFN surviving only on the beyond-SE tie) does **not** repeat here — at
36% positive, TabPFN's lead over the GBDTs holds on the power axis at 10M params.
Parsimony still narrows the practical gap: lgbm at 3,100 params is within 0.002 of
TabPFN's log loss, and the GLM family anchors the low-complexity end at 14 params
(logisticglm/lr 0.26315/0.26319, poisson/tweedie well behind — tweedieglm catastrophic
at 0.65338). But unlike §14.6, parsimony alone does not remove TabPFN from the frontier;
cat/xgb/rf are dominated. Files:
`scripts/eval/insurance_benchmark_v1/frontier_results_ausprivauto0405.csv`,
`frontier_results_bemtl16.csv`, `frontier_plot_ausprivauto0405.png`,
`frontier_plot_bemtl16.png`, `frontier_v1suite_run.log`.

### 14.8 Regression Phase 2 — RMSE + Poisson-deviance frontiers (D4)

The regression/severity axis the v1 classification pass deliberately deferred (spec §5
D4): severity mixes a different power metric and TabPFN's documented regression weakness
would have polluted v1's first delivery. Phase 2 runs **4 datasets** on their **stored
targets** — RMSE for `ausautoBI8999` (log-scale `AggClaim`), `ausprivauto0405_vehvalue`
(raw `VehValue`), `bemtl97_amount` (log1p amount, `claim` dropped as a leak) and Poisson
deviance for `freMTPL2freq` (log(`Exposure`) offset feature, `IDpol` dropped). All **8
regressors at defaults** (ols / poissonglm / tweedieglm / rf / cat / lgbm / xgb / tabpfn
via hosted API), **5-fold KFold seed 42**, all fits fresh. The §14.7 pointer named
`ausprivauto0405_vehvalue` + `freMTPL2freq` as candidates; `ausautoBI8999` +
`bemtl97_amount` were run as well. Results committed in `78c3df2`.

**ausautoBI8999** (22,036 rows, RMSE on stored log-scale `AggClaim`):

| method | mean RMSE | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| tabpfn | **0.96491** | 0.00868 | 10,000,000 | yes |
| cat | 0.96883 | 0.00756 | 63,944 | yes |
| lgbm | 0.97456 | 0.00822 | 3,100 | yes |
| xgb | 0.98945 | 0.00957 | 5,223 | yes |
| rf | 1.01260 | 0.00874 | 1,112,247 | no |
| ols | 1.07133 | 0.00928 | 12 | yes |
| poissonglm | 1.07881 | 0.00926 | 12 | yes |
| tweedieglm | 1.08867 | 0.00875 | 12 | yes |

**ausprivauto0405_vehvalue** (67,856 rows, RMSE on raw `VehValue`):

| method | mean RMSE | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| tabpfn | **0.71162** | 0.01246 | 10,000,000 | yes |
| lgbm | 0.71645 | 0.01258 | 3,100 | yes |
| cat | 0.72171 | 0.01164 | 63,912 | yes |
| xgb | 0.73784 | 0.00809 | 5,994 | no |
| rf | 0.82071 | 0.00893 | 2,599,078 | no |
| poissonglm | 1.00932 | 0.01609 | 7 | yes |
| ols | 1.04150 | 0.01545 | 7 | yes |
| tweedieglm | 1.05117 | 0.01548 | 7 | yes |

**bemtl97_amount** (163,212 rows, RMSE on stored log1p amount, `claim` dropped as leak):

| method | mean RMSE | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| lgbm | **0.48499** | 0.00431 | 3,100 | yes |
| cat | 0.48726 | 0.00412 | 63,226 | yes |
| xgb | 0.49149 | 0.00448 | 4,733 | yes |
| rf | 0.50457 | 0.00461 | 902,022 | no |
| ols | 0.70799 | 0.00651 | 12 | yes |
| tabpfn | 0.72825 | 0.01057 | 10,000,000 | no |
| tweedieglm | 1.84473 | 0.00626 | 12 | yes |
| poissonglm | 6.45906 | 2.83431 | 12 | yes |

**freMTPL2freq** (678,013 rows, Poisson deviance, log(`Exposure`) offset feature, `IDpol` dropped):

| method | mean Poisson deviance | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| lgbm | **0.29113** | 0.00138 | 3,100 | yes |
| cat | 0.30048 | 0.00184 | 63,788 | no |
| xgb | 0.31221 | 0.00208 | 5,747 | no |
| ols | 0.32097 | 0.00161 | 11 | yes |
| poissonglm | 0.32109 | 0.00161 | 11 | yes |
| tweedieglm | 0.32109 | 0.00161 | 11 | yes |
| tabpfn | 0.38770 | 0.00201 | 10,000,000 | no |
| rf | 0.47966 | 0.00525 | 3,337,930 | no |

Bold = best mean per dataset (power winner). Frontier membership per the D3 beyond-SE
rule (§14.1), reproduced from `frontier_results_<dataset>.csv`.

**ausautoBI8999 — TabPFN wins the power axis outright, and this time the lead is real.**
TabPFN is the power winner (0.96491) over cat (0.96883) and it is **beyond SE**, not a
fold-noise tie: `tabpfn mean−SE = 0.95623` vs `cat mean+SE = 0.97639`. The frontier is
tabpfn/cat/lgbm/xgb plus the 12-param GLM anchors; rf is the only dominated method (beaten
on power by the GBDTs *and* on parsimony by the GLMs). The GLMs trail by ~0.11–0.12
(1.07133/1.07881/1.08867) — the widest GLM gap of the regression pass.

**ausprivauto0405_vehvalue — TabPFN wins power within SE of lgbm — and a critical v1
discrepancy.** TabPFN leads (0.71162 vs lgbm 0.71645), but the intervals overlap, so it
shares the frontier with lgbm on the beyond-SE tie (`tabpfn mean+SE = 0.72408` vs
`lgbm mean−SE = 0.70387`); cat trails at 0.72171, xgb/rf are dominated. **This does NOT
reproduce the v1 verdict (§4.1: TabPFN +67.2% on vehvalue, rank 8/8, RMSE ≈ 1.20/fold in
`results_per_split.csv`).** The v1 figure came from the TabArena harness (3-fold, a
different client invocation); the frontier protocol — TabPFNRegressor (random_state=0) on
5-fold KFold seed 42 — gives 0.71. The +67.2% was protocol-specific, not a stable
property of the model on this dataset, and §12.1's citation of it as a size-ceiling data
point should be read accordingly.

**bemtl97_amount — TabPFN dominated at scale; the zero-inflation trap.** lgbm (0.48499) /
cat (0.48726) / xgb (0.49149) sweep the top; rf and tabpfn (0.72825 ± 0.01057) are
dominated — tabpfn beaten on power and on parsimony, consistent with v1's +48.3% verdict
(§12.1). The GLM end tells the zero-inflation story: poissonglm is catastrophic
(6.45906 ± 2.83431) — predicting >0 systematically misses the ~89% zero mass — and
tweedieglm is poor (1.84473); ols (0.70799) anchors the GLM family, and the 12-param
models all sit on the frontier because nothing has fewer parameters to dominate them.

**freMTPL2freq — the frequency axis is genuinely competitive for GLMs, and TabPFN's worst
size-ceiling case.** lgbm wins power (0.29113); the 11-param GLM family — Poisson, the
actuarial gold standard — lands within ~10% (0.32097/0.32109/0.32109), unlike
classification where GLM variants were placeholders. cat/xgb/tabpfn/rf are dominated;
tabpfn at 0.38770 is ~+33% behind lgbm on 678,013 rows ≈ 100× its ~1K-row home-turf
regime — the strongest §12.1 size-ceiling confirmation yet. Files:
`scripts/eval/insurance_benchmark_v1/frontier_results_{ausautoBI8999,ausprivauto0405_vehvalue,bemtl97_amount,freMTPL2freq}.csv`,
`frontier_plot_{ausautoBI8999,ausprivauto0405_vehvalue,bemtl97_amount,freMTPL2freq}.png`,
`frontier_regression_run.log`.

**D4 synthesis — TabPFN's regression lead does not survive the frontier.** Across the four
regression axes TabPFN wins power only at small N — beyond SE at 22,036 rows
(ausautoBI8999) and on the beyond-SE tie at 67,856 rows (vehvalue) — and is dominated at
scale: 163,212 rows (bemtl97_amount) and 678,013 rows (freMTPL2freq). This transfers the
§12.1 size-ceiling hypothesis to the regression axis: at ~1K effective context the model
holds its edge on small/medium severity problems and loses it where every row of signal
matters. (The "~1K effective context" mechanism is amended by the §12.1 model-version
correction, 2026-08-04; the size-dependent pattern stands.) It also flags a protocol
sensitivity worth recording — the frontier protocol's
TabPFNRegressor does not reproduce v1's vehvalue +67.2%, so v1's regression verdicts are
harness-specific. The GLMs are far more competitive on frequency than on classification;
on severity the GBDTs dominate outright.

### 14.9 Spanish motor portfolio — frequency frontier (53,502 rows)

Real-portfolio extension: Segura-Gisbert et al. (2024), "Dataset of an actual motor
vehicle insurance portfolio" (Spanish motor, prepared by `make_spanish_motor_freq` in
`scripts/prepare_insurance_datasets.py`). 53,502 policies, target `N_claims_year`
(int 0–18, 11.1% claims > 0), Poisson deviance, 5-fold KFold seed 42, all 8 regressors
fit **fresh**. The raw policy-year panel is collapsed to the last policy-year per ID
(bemtl16 precedent); `Length` NA (motorbikes) is imputed by `Type_risk` median;
exposure uniform ~1yr (no offset). Leak exclusions verified **pre-run**:
`N_claims_history` / `R_Claims_history` include CURRENT-year claims (leak AUC
0.76/0.92 vs claims > 0 — the history-variable leak, ~0.918, was caught before the
run) and `Cost_claims_year` is a sibling target; none are kept as features.

| method | mean Poisson deviance | ± SE | n_params | on frontier |
| --- | ---: | ---: | ---: | --- |
| lgbm | **0.89157** | 0.01238 | 3,100 | yes |
| tabpfn | 0.98764 | 0.01624 | 10,000,000 | no |
| ols | 0.98994 | 0.01854 | 21 | yes |
| cat | 0.99757 | 0.02372 | 63,892 | no |
| rf | 0.99808 | 0.01651 | 412,124 | no |
| poissonglm | 1.01250 | 0.01867 | 21 | yes |
| tweedieglm | 1.01250 | 0.01867 | 21 | yes |
| xgb | 1.35178 | 0.03642 | 5,350 | no |

**LGBM dominates beyond SE; TabPFN falls off the frontier for the fourth time at
scale; the GLM anchors sit within noise of TabPFN at 21 params.** lgbm wins power
(0.89157) and dominates TabPFN outright per the D3 beyond-SE rule (`lgbm mean+SE =
0.90395` vs `tabpfn mean−SE = 0.97140`), ~10.8% better deviance (0.98764 vs 0.89157).
ols / poissonglm / tweedieglm (21 params) hold the frontier anchors, all within SE of
TabPFN's mean (0.98994 / 1.01250 vs 0.98764) — parsimony closes a gap that power
leaves open, the §12.1 pattern again. Null deviance is 1.0123, so poissonglm /
tweedieglm at 1.01250 are effectively at the intercept-only floor: the portfolio's
frequency signal is thin, and only lgbm extracts it. cat (0.99757) / rf (0.99808) sit
just off the power edge, xgb is dominated (1.35178). This is the **fourth at-scale
frontier where TabPFN is off-frontier** — after norauto (surviving only on the
beyond-SE tie, §14.6), bemtl97_amount and freMTPL2freq (dominated, §14.8) — and
consistent with the D4 synthesis: at 53,502 rows the 10M-param model adds nothing a
3,100-param LGBM or a 21-param GLM does not already provide. Files:
`scripts/eval/insurance_benchmark_v1/frontier_results_spanish_motor_freq.csv`,
`frontier_plot_spanish_motor_freq.png`. Model version note: TabPFN served via the
hosted client 0.3.3; this run used auto-selection (resolves to v3_default), and the
benchmark scripts now pin `model_path="v3_default"` explicitly for reproducibility.

**Actuary takeaway:** on this real Spanish motor frequency book, a default LGBM
beats TabPFN by ~10.8% deviance with 1/3,200 of the parameters, and the standard
Poisson GLM is statistically indistinguishable from TabPFN — there is no actuarial
case for TabPFN on frequency at this scale. On lapse, the picture flips: TabPFN
edges LGBM (AUC 0.752 vs 0.745 on Spanish motor, 53.5K rows) and both far outrun
LR (0.684); TabPFN also leads the combined lapse leaderboard (elo 1128). The
2-fold caveat on that lapse gap is settled by the 5-fold re-run in §14.10.

## 14.10 Gap-closing addendum (2026-08-04)

**5-fold lapse re-run** (`scripts/run_lapse_benchmark.py`, both classification
entries now `n_splits: 5`; experiment cache cleared first — 0 cache_exists, so
both datasets including eudirectlapse were freshly refit; premium regression task
stays 2-fold). Mean AUC over 5 stratified folds, ±SE:

| Dataset | TabPFN | LGBM | Linear |
|---|---|---|---|
| spanish_motor_lapse (53.5K rows, 35.4% pos) | **0.7553 ± 0.0026** | 0.7500 ± 0.0022 | 0.6841 ± 0.0015 |
| eudirectlapse (23K rows, 12.8% pos) | 0.6101 ± 0.0049 | 0.6138 ± 0.0043 | **0.6260 ± 0.0037** |

The Spanish lapse gap **survives 5 folds and widens slightly** (0.7553 vs 0.7500,
~2 SE): TabPFN wins all five folds (0.7465–0.7621 vs 0.7416–0.7550). eudirectlapse
still goes to Linear (wins all 5 folds). Combined leaderboard (2×5-fold lapse +
2-fold premium): TabPFN elo 1099.0 > GBM 983.5 > LR 917.6 — TabPFN's lead is
carried by the premium RMSE task (19.07 vs 26.34 vs 81.84) plus Spanish lapse.

**Spanish severity frontier** (`data/raw/spanish_motor_severity.csv`: last
policy-year per ID, 53,502 rows × 20 features, target log1p(Cost_claims_year) in
place — bemtl97_amount precedent; N_claims_year excluded as a sibling target:
cost==0 iff count==0, so the count would give away the zero/non-zero split).
RMSE on the stored log1p scale, 5-fold KFold, 8 methods:

| Method | RMSE (mean ± SE) | n_params | Frontier |
|---|---|---|---|
| lgbm | **1.8372 ± 0.0100** | 3,100 | yes |
| cat | 1.8376 ± 0.0095 | 63,952 | yes |
| rf | 1.8702 ± 0.0100 | 477,610 | no |
| ols | 1.8781 ± 0.0104 | 21 | yes |
| xgb | 1.8786 ± 0.0100 | 5,459 | no |
| tabpfn | 1.8862 ± 0.0116 | 10,000,000 | no |
| poissonglm / tweedieglm | 1.8876 ± 0.0106 | 21 | yes |

TabPFN lands 5th of 8 (mid-pack) and off-frontier — 2.7% behind LGBM on RMSE.
Severity shows the same at-scale pattern as frequency: LGBM/Cat extract the
signal, TabPFN's 10M-param prior adds nothing. Files:
`scripts/eval/insurance_benchmark_v1/frontier_results_spanish_motor_severity.csv`,
`frontier_plot_spanish_motor_severity.png`.

## 9. Source Workbooks

- `scripts/run_home_turf_size_sweep.py` — home-turf size sweep runner (3 datasets × 3
  sizes × 5 folds, TabPFN + GBDT arms; §13).
- `scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py` — insurance frontier
  benchmark (D1/D2 combined pass, 3 datasets × 9 methods, D3 beyond-SE Pareto rule;
  §14, commit `ed3e119`).
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
- `scripts/eval/insurance_benchmark_v1/frontier_results_{bemtl97,coil2000,uslapseagent,norauto}.csv`
  — frontier benchmark, per-method mean log loss ± SE, n_params, on-frontier flag
  (primary evidence for §14.2, §14.6).
- `scripts/eval/insurance_benchmark_v1/frontier_plot_{bemtl97,coil2000,uslapseagent,norauto}.png`
  — frontier plots, x = log10(n_params), y = mean log loss ± SE (§14.2, §14.6).
- `scripts/eval/insurance_benchmark_v1/frontier_benchmark_run.log` — frontier run log,
  per-fold timings and log loss (§14.1).
- `scripts/eval/insurance_benchmark_v1/frontier_norauto_run.log` — norauto run log,
  per-fold timings and log loss (§14.6).
