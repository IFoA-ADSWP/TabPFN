# Session: TabArena Benchmark Setup — 28 Jul 2026

## Context

Set up the TabArena benchmark framework to evaluate TabPFN against tree-based and statistical models on insurance lapse data. This is the first step toward a full 7-dataset insurance benchmark.

## Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/run_smoke_tabarena.py` | Quick 3-model smoke test on coil2000 | ✅ Passed |
| `scripts/run_lapse_benchmark.py` | Lapse-only benchmark (classification + regression), 7 models | ⚠️ Runs, but too slow for full results on this machine |
| `scripts/run_tabarena_insurance_benchmark.py` | 7-dataset full benchmark | ⚠️ Untested, has same fixes applied |

## Environment

- Python 3.12 venv at `/tmp/tabarena/.venv-ta`
- 140 packages installed via uv (PyTorch 2.13, LightGBM, XGBoost, CatBoost, scikit-learn, TabPFN client 0.3.3)
- TabPFN hosted API authenticated with access key (`tabpfn_sk_...`)

## Script Fixes Applied

All three scripts received the same fixes:

1. **`ConfigGenerator(search_space={}, ...)`** — TabArena API changed; `search_space` is now a required first positional arg
2. **`("LR", 0) → ("Linear", 0)`** — `'LR'` is not registered in TabArena's model registry; the correct name is `'Linear'` (maps to AutoGluon's `LR` which uses sklearn LogisticRegression)
3. **`build_experiments(num_gpus=0)`** — prevents AutoGluon from selecting GPU mode on Mac (Apple Silicon Metal), which was causing LightGBM to train on GPU-emulation-mode and stall
4. **`import numpy as np`** — missing import in smoke test script
5. **`LightGBM device_type: "cpu"`** — per-model hyperparameter override to force CPU even when LightGBM's auto-detection picks GPU
6. **Object→category dtype conversion** — eudirectlapse.csv has 9 object-type categorical columns; TabArena/AutoGluon require non-object dtypes

## Smoke Test Results (coil2000, 2 folds)

| Method | Rank | ELO | Metric Error | Train Time |
|--------|------|-----|-------------|------------|
| TabPFN (hosted API) | #1 | 2190.2 | 0.2424 | 83.3s |
| LightGBM | #2.5 | 404.9 | 0.2688 | 14.1s |
| Linear (LR) | #2.5 | 404.9 | 0.2703 | 1.6s |

Results saved to `scripts/eval/smoke_test/`

## Lapse Benchmark Status

**Classification** (target=`lapse`, 23K rows, 12.8% positive rate):
- TabPFN achieved 0.6039 ROC-AUC (single fold on 11,530 training rows)
- Full 2-fold × 8-bag benchmark could not complete due to runtime constraints
- Models: TabPFN Client, LightGBM, XGBoost, CatBoost, RandomForest, Logistic GLM, Linear

**Regression** (target=`prem_pure`):
- Not run — Poisson GLM and Tweedie GLM were dropped from the lapse-only script as they're specialized for count/premium targets in the full benchmark

## Issues Found

### Performance: 8-fold bagging is too heavy on CPU
TabArena's default uses 8 bagged folds per model × 2 CV splits × 7 models × 2 tasks = 224 model fits. On this 8-core Mac (no GPU), each tree-based model fit takes 2-5 minutes, making a full run exceed reasonable timeouts.

**Possible mitigations:**
- Set `num_bag_folds` to 3 in the bundle config
- Reduce CV splits to 2
- Run only on a subset of models for quick comparisons
- Use a GPU machine for full benchmark

### LightGBM GPU auto-detection
On Apple Silicon, LightGBM auto-detects Metal GPU and trains in GPU-emulation mode, which is slower than CPU. Fixed via `device_type: "cpu"` in per-model hyperparameters and `num_gpus=0` in `build_experiments()`.

### TabArena API instability
- `ConfigGenerator` API changed (search_space now required)
- Model registry names differ from AutoGluon keys

### Dataset dtype
eudirectlapse.csv has 9 object-type categorical columns that must be converted to `category` dtype for AutoGluon compatibility.

## Files Changed

```
M scripts/run_smoke_tabarena.py          (fixes 1-4, 6)
M scripts/run_tabarena_insurance_benchmark.py  (fixes 1-3, 5, 6)
A scripts/run_lapse_benchmark.py         (new, focused lapse benchmark)
M scripts/README.md                      (updated with benchmark section)
A scripts/eval/smoke_test/               (smoke test results)
A TASKS.md                               (GitHub issue tracker)
```

## Remaining Work

1. Run full 7-dataset insurance benchmark (needs more time or GPU machine)
2. Complete lapse regression benchmark
3. Run TabFM locally (requires GPU or heavy CPU time)
4. Fine-tuning experiments on lapse data
5. Set up `.opencode/project/CONTEXT.md` for agent orientation
