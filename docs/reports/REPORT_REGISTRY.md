# Report Registry

Purpose: prevent duplicate report topics and keep reports traceable to source workbooks and evidence files.

## Usage

1. Before drafting a report, search this registry for similar topic keys.
2. If a matching topic exists, update that report unless a separate version is explicitly required.
3. Ensure each report includes "Source Workbooks" and "Evidence Files" sections.
4. Update the corresponding row after creating/updating a report.

## Registry Table

| report_path | topic_key | audience | status | source_workbooks | evidence_files | last_updated |
| --- | --- | --- | --- | --- | --- | --- |
| docs/reports/INSURANCE_DOMAIN_FINETUNING_METHOD_PROTOCOL.md | insurance-finetuning-method-protocol | technical | active | notebooks/baseline_experiments/05_regression_finetuning.ipynb; notebooks/baseline_experiments/07_multi_dataset_benchmark.ipynb | outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv; outputs/current/logs/tabpfn_finetune_regressor_logbook.md | 2026-04-04 |
| docs/reports/COMBINED_TABPFN_CLASSIFIER_REGRESSOR_ANALYSIS.md | combined-classifier-regressor-analysis | technical | active | notebooks/baseline_experiments/03_tabpfn_vs_glm_summary.ipynb; notebooks/baseline_experiments/08_multi_dataset_regression_benchmark.ipynb; notebooks/baseline_experiments/05_regression_finetuning.ipynb | data/processed/glm_vs_tabpfn_head_to_head.csv; data/processed/multi_dataset_regression_benchmark_results.csv; outputs/current/tables/raw_tabpfn_regression_revalidation.csv; outputs/current/tables/multi_dataset_regression_benchmark_with_tabpfn_revalidated.csv; outputs/current/tables/tabpfn_regression_finetune_vs_raw.csv; outputs/current/tables/multi_dataset_regression_benchmark_with_tabpfn_finetuned.csv; outputs/current/tables/domain_finetune_study_runs.csv; outputs/current/tables/classifier_round3_seed42_44_deltas.csv; outputs/current/tables/classifier_round3_policy_pooled.csv; outputs/current/logs/tabpfn_regression_finetune_vs_raw.md; outputs/current/logs/domain_finetune_logbook.md | 2026-04-07 |
| docs/reports/INSURANCE_SPECIFIC_FINETUNING_EVIDENCE.md | insurance-finetuning-evidence | technical | active | notebooks/baseline_experiments/05_regression_finetuning.ipynb | outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv; outputs/current/logs/claimnb_finiteness_checkpoints.csv | 2026-04-04 |
| docs/reports/STAGE_A_B_FINDINGS_AND_RECOMMENDATIONS.md | stage-a-b-findings-recommendations | non-technical | active | notebooks/baseline_experiments/07_multi_dataset_benchmark.ipynb | outputs/current/tables/domain_finetune_study_summary.csv | 2026-04-04 |
| docs/reports/TABPFN_FINE_TUNING_LIMIT_STUDY.md | finetuning-limit-study | technical | active | notebooks/baseline_experiments/05_regression_finetuning.ipynb | outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv | 2026-04-04 |
| docs/reports/TECHNICAL_COMPANION.md | technical-companion-reference | technical | active | notebooks/adswp_project/01_TabPFN_classifier_eudirectlapse.ipynb; notebooks/adswp_project/02_TabPFN_freMTPL.ipynb | outputs/current/tables/tabpfn_finetune_trial_results.csv; outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv | 2026-04-04 |
| docs/reports/MULTI_DATASET_GLM_VS_TABPFN_SUMMARY.md | glm-vs-tabpfn-summary | non-technical | active | notebooks/baseline_experiments/03_tabpfn_vs_glm_summary.ipynb | data/processed/glm_vs_tabpfn_head_to_head.csv | 2026-04-04 |
| docs/reports/POST_HOC_OPTIMISATION.md | post-hoc-optimisation | technical | active | notebooks/baseline_experiments/04_probability_calibration.ipynb | outputs/current/tables/tabpfn_finetune_trial_results.csv | 2026-04-04 |
| docs/reports/CLASSIFIER_HOMOGENEITY_HYPOTHESIS_METHOD.md | classifier-homogeneity-hypothesis-method | technical | active | notebooks/baseline_experiments/03_tabpfn_vs_glm_summary.ipynb; notebooks/baseline_experiments/05_regression_finetuning.ipynb | outputs/current/tables/domain_finetune_study_runs.csv; outputs/current/tables/classifier_homogeneity_smoke_seed42_summary.csv; outputs/current/tables/classifier_homogeneity_matrix_seed42_44_summary.csv; outputs/current/tables/classifier_similarity_round2_seed42_44_summary.csv; outputs/current/tables/classifier_round3_seed42_44_deltas.csv; outputs/current/tables/classifier_round3_policy_pooled.csv; outputs/current/logs/domain_finetune_logbook.md | 2026-04-07 |
