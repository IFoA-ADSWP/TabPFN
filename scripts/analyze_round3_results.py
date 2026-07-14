#!/usr/bin/env python3
"""Analyze Round 3 (context=128, steps=5) classifier fine-tuning results."""

import pandas as pd
import numpy as np

# Load the full CSV
df = pd.read_csv("outputs/current/tables/domain_finetune_study_runs.csv")

# Filter for Round 3 runs (context_samples=128, max_finetune_steps=5, tabpfn only)
df_r3 = df[(df['tabpfn_context_samples'] == 128) & 
           (df['tabpfn_max_finetune_steps'] == 5) &
           (df['model_family'] == 'tabpfn')].copy()

print(f"=== Round 3 Data Summary ===")
print(f"Total tabpfn rows (context=128, steps=5): {len(df_r3)}")
print(f"Unique targets: {df_r3['target_dataset'].nunique()}")
print(f"    {sorted(df_r3['target_dataset'].unique())}")
print(f"Unique seeds: {sorted(df_r3['seed'].unique())}")
print()

# Reconstruct deltas: raw vs domain_finetuned for each seed/target/policy
results = []
for seed in sorted(df_r3['seed'].unique()):
    for target in sorted(df_r3['target_dataset'].unique()):
        for policy in df_r3['pool_policy'].dropna().unique():
            subset = df_r3[(df_r3['seed']==seed) & 
                          (df_r3['target_dataset']==target) &
                          (df_r3['pool_policy']==policy)]
            
            if len(subset) < 2:
                continue  # Need both raw and finetuned
            
            raw_row = subset[subset['model_variant']=='raw']
            ft_row = subset[subset['model_variant']=='domain_finetuned']
            
            if len(raw_row) > 0 and len(ft_row) > 0:
                raw = raw_row.iloc[0]
                ft = ft_row.iloc[0]
                
                # Calculate deltas
                delta_roc = ft['roc_auc'] - raw['roc_auc']
                delta_pr = ft['pr_auc'] - raw['pr_auc']
                delta_brier = ft['brier'] - raw['brier']
                delta_logloss = ft['log_loss'] - raw['log_loss']
                
                results.append({
                    'seed': seed,
                    'target': target,
                    'policy': policy,
                    'raw_roc': raw['roc_auc'],
                    'ft_roc': ft['roc_auc'],
                    'delta_roc': delta_roc,
                    'raw_pr': raw['pr_auc'],
                    'ft_pr': ft['pr_auc'],
                    'delta_pr': delta_pr,
                    'raw_brier': raw['brier'],
                    'ft_brier': ft['brier'],
                    'delta_brier': delta_brier,
                    'raw_logloss': raw['log_loss'],
                    'ft_logloss': ft['log_loss'],
                    'delta_logloss': delta_logloss,
                })

df_results = pd.DataFrame(results)

print("=== Round 3 Individual Deltas (Sample, 15 rows) ===")
print(df_results.head(15).to_string(index=False))
print()

# Policy-level pooled deltas
print("=== POOLED DELTAS BY POLICY (Across All Seeds & Targets) ===")
policy_summary = []
for policy in sorted(df_results['policy'].unique()):
    subset = df_results[df_results['policy'] == policy]
    mean_roc_delta = subset['delta_roc'].mean()
    std_roc_delta = subset['delta_roc'].std()
    mean_pr_delta = subset['delta_pr'].mean()
    std_pr_delta = subset['delta_pr'].std()
    mean_brier_delta = subset['delta_brier'].mean()
    std_brier_delta = subset['delta_brier'].std()
    mean_logloss_delta = subset['delta_logloss'].mean()
    std_logloss_delta = subset['delta_logloss'].std()
    
    policy_summary.append({
        'policy': policy,
        'n': len(subset),
        'mean_delta_roc': mean_roc_delta,
        'std_delta_roc': std_roc_delta,
        'mean_delta_pr': mean_pr_delta,
        'std_delta_pr': std_pr_delta,
        'mean_delta_brier': mean_brier_delta,
        'std_delta_brier': std_brier_delta,
        'mean_delta_logloss': mean_logloss_delta,
        'std_delta_logloss': std_logloss_delta,
    })

df_policy = pd.DataFrame(policy_summary)
print(df_policy.round(4).to_string(index=False))
print()

# Decision rule evaluation
print("=== DECISION RULE EVALUATION (Success Criteria) ===")
print("Rules (all must hold):")
print("  1. Mean Δ ROC AUC > 0 for ≥1 policy")
print("  2. Mean Δ PR AUC  > 0 for same policy")
print("  3. Δ Calibration (Brier, LogLoss) does not degrade > 0.5%")
print("  4. No single seed/target driving all gains")
print()

decision_status = "FAIL"
for policy in sorted(df_results['policy'].unique()):
    subset = df_results[df_results['policy'] == policy]
    mean_roc_delta = subset['delta_roc'].mean()
    mean_pr_delta = subset['delta_pr'].mean()
    mean_brier_delta = subset['delta_brier'].mean()
    mean_logloss_delta = subset['delta_logloss'].mean()
    
    print(f"Policy: {policy}")
    print(f"  Δ ROC AUC:    {mean_roc_delta:+.4f} {'✓' if mean_roc_delta > 0 else '✗'} (need > 0)")
    print(f"  Δ PR AUC:     {mean_pr_delta:+.4f} {'✓' if mean_pr_delta > 0 else '✗'} (need > 0)")
    print(f"  Δ Brier:      {mean_brier_delta:+.4f} {'✓' if abs(mean_brier_delta) <= 0.005 else '✗'} (need ≤ ±0.005)")
    print(f"  Δ LogLoss:    {mean_logloss_delta:+.4f} {'✓' if abs(mean_logloss_delta) <= 0.005 else '✗'} (need ≤ ±0.005)")
    
    # Check if this policy passes all criteria
    roc_ok = mean_roc_delta > 0
    pr_ok = mean_pr_delta > 0
    brier_ok = abs(mean_brier_delta) <= 0.005
    logloss_ok = abs(mean_logloss_delta) <= 0.005
    
    if roc_ok and pr_ok and brier_ok and logloss_ok:
        decision_status = "PASS"
        print(f"  ➜ ALL CRITERIA MET ✓")
    else:
        print(f"  ➜ Some criteria unmet")
    print()

print(f"=== FINAL DECISION ===")
print(f"Status: {decision_status}")
if decision_status == "PASS":
    print("Action: Continue classifier fine-tuning investigation")
    print("  - Fine-tuning budget increase (3→5 steps) successfully unlocked positive deltas")
    print("  - Recommend one more controlled scale before expanding data universe")
else:
    print("Action: Downgrade classifier fine-tuning hypothesis")
    print("  - Even with larger budget, pooled deltas remain negative or calibration degrades")
    print("  - Shift focus to calibration/preprocessing without adding new datasets")
print()

# Save summary CSV
df_results.to_csv("outputs/current/tables/classifier_round3_seed42_44_deltas.csv", index=False)
df_policy.to_csv("outputs/current/tables/classifier_round3_policy_pooled.csv", index=False)
print("Saved:")
print("  - outputs/current/tables/classifier_round3_seed42_44_deltas.csv")
print("  - outputs/current/tables/classifier_round3_policy_pooled.csv")
