# Scripts

One-off experiment scripts, infrastructure, and debug tools. Ordered by purpose.

## Data

| Script | Purpose |
|--------|---------|
| `download_datasets.py` | Download coil2000, ausprivauto0405, generate freMTPL2freq_binary from freMTPL2freq |

## Classification Pipeline

| Script | Purpose |
|--------|---------|
| `run_domain_finetune_stage_a.py` | Domain fine-tuning classifier comparison across 4 insurance datasets |
| `run_small_finetune_classifier_trial.py` | Local readiness smoke test for classifier fine-tuning |
| `check_saved_finetune_classifier_model.py` | Reload + validate a saved `.tabpfn_fit` artifact |
| `evaluate_classifier_homogeneity_proposal.py` | Evaluate homogeneity hypothesis from Stage A results |
| `summarize_classifier_homogeneity_smoke.py` | Summarise homogeneity smoke-test results |

## Regression Pipeline

| Script | Purpose |
|--------|---------|
| `run_raw_tabpfn_regression_benchmark.py` | Raw TabPFN regressor benchmark |
| `run_finetuned_tabpfn_regression_benchmark.py` | Fine-tuned TabPFN regressor benchmark |
| `run_small_finetune_regressor_trial.py` | Local readiness smoke test for regressor fine-tuning |
| `evaluate_regressor_stability_gate.py` | Evaluate whether regressor config is ready for full rerun |
| `diagnose_claimnb_finiteness.py` | Debug tool for TabPFN regressor finiteness issues |

## Fine-tuning Timing & Limits

| Script | Purpose |
|--------|---------|
| `pilot_timing.py` | Measure per-sample time for TabPFN fine-tuning loop |
| `finetune_pilot.py` | Fine-tuning pilot — time per sample under different configs |
| `debug_preprocess.py` | Debug TabPFN preprocessing pipeline |

## Analysis

| Script | Purpose |
|--------|---------|
| `analyze_round3_results.py` | Analyse Round 3 classifier fine-tuning results |

## Batch Job Launchers

| Script | Purpose |
|--------|---------|
| `run_finetune_first_batch.sh` | First batch of fine-tuning runs |
| `run_finetune_crossover_batch_3000.sh` | Crossover batch at 3000 rows |
| `run_finetune_stress_batch_2000.sh` | Stress batch at 2000 rows |

## Infrastructure

| Script | Purpose |
|--------|---------|
| `push_wiki.sh` | Sync `.wiki-content/` to GitHub Wiki |
| `import_backlog_to_github.py` | Import maintenance backlog into GitHub Issues (for when Issues are re-enabled) |
