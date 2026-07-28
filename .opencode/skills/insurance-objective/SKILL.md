---
name: insurance-objective
description: "Assess TabPFN performance for insurance modeling and evaluate domain-specific fine-tuning effectiveness. Use for baseline vs fine-tuned comparisons, classifier/regressor evidence synthesis, stability diagnostics, and recommendation reports."
argument-hint: "Dataset(s), target column(s), task type, baseline/fine-tune configs, and evaluation objective"
---

# Insurance Objective Evaluation

## When to Use
- Running the core project objective end-to-end
- Comparing raw TabPFN vs fine-tuned TabPFN on insurance datasets
- Producing one decision summary across classifier and regressor tracks
- Investigating regressors that fail with non-finite behavior

## Execution Mode
- This repository is research-first and run manually in the working environment.
- Do not require CI/CD gates for experiment progress.
- Treat `outputs/current/tables` and `outputs/current/logs` as the source of truth for decisions.

## Inputs to Confirm
1. Dataset path(s) and insurance task scope
2. Target(s) and task type (classification vs regression)
3. Baseline and fine-tune configuration (device, rows, context, steps, seed)
4. Primary metric(s):
   - Classification: ROC AUC, PR AUC, log loss
   - Regression: MSE, MAE, R2
5. Required artifact paths for evidence

## Workflow
1. Validate dataset and target suitability.
2. Run tiny smoke pass for each track before scaling.
3. Run raw baseline and fine-tuned variant with identical split/seed where possible.
4. Capture timing and memory for every run.
5. For regressor, always collect viability counters:
   - `finetune_steps_executed`
   - `skipped_preprocess_errors`
   - `skipped_nonfinite_target`
   - `skipped_nonfinite_loss`
6. Compare baseline vs fine-tuned results in one table.
7. Decide: scale, adjust transform/context, or escalate instability.

## Escalation Criteria (Regressor)
- If `finetune_steps_executed = 0` across tiny and scaled settings for claim targets while control targets execute steps, escalate as claim-target numerical instability.

## Required Evidence Files
- `outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv`
- `outputs/current/tables/tabpfn_finetune_trial_results.csv` (classifier when available)
- `outputs/current/logs/claimnb_finiteness_checkpoints.csv` (when instability diagnostics are run)

## Reporting Template
1. What was run (task, target, rows, context, device, seed)
2. Baseline vs fine-tuned metric table
3. Runtime/memory table
4. Regressor viability table (if applicable)
5. Decision and one next recommendation