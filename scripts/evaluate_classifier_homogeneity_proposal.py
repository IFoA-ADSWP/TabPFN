"""Evaluate homogeneity hypothesis from existing Stage A classifier results.

Hypothesis under test:
- Fine-tuning on a more homogeneous domain pool should improve target performance
  relative to a heterogeneous pool.

This script does NOT run models. It reuses current results in:
- outputs/current/tables/domain_finetune_study_runs.csv

It computes per-target fine-tune deltas (domain_finetuned - raw) and compares those
against a simple homogeneity proxy based on class prevalence distance between target
and domain pool candidates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_MAP = {
    "eudirectlapse": ("data/raw/eudirectlapse.csv", "lapse"),
    "coil2000": ("data/raw/coil2000.csv", "CARAVAN"),
    "ausprivauto0405": ("data/raw/ausprivauto0405.csv", "ClaimOcc"),
    "freMTPL2freq_binary": ("data/raw/freMTPL2freq_binary.csv", "ClaimIndicator"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_prevalence(dataset: str) -> float:
    csv_path, target_col = TARGET_MAP[dataset]
    frame = pd.read_csv(repo_root() / csv_path)
    y = frame[target_col]
    return float((y == 1).mean())


def main() -> None:
    runs_path = repo_root() / "outputs/current/tables/domain_finetune_study_runs.csv"
    runs = pd.read_csv(runs_path)

    tab = runs[runs["model_family"] == "tabpfn"].copy()
    raw = tab[tab["model_variant"] == "raw"].copy()
    ft = tab[tab["model_variant"] == "domain_finetuned"].copy()

    key_cols = [
        "stage",
        "target_dataset",
        "seed",
        "target_rows",
        "tabpfn_context_samples",
        "tabpfn_max_finetune_steps",
    ]
    paired = raw.merge(ft, on=key_cols, suffixes=("_raw", "_ft"))

    paired["delta_roc_auc"] = paired["roc_auc_ft"] - paired["roc_auc_raw"]
    paired["delta_pr_auc"] = paired["pr_auc_ft"] - paired["pr_auc_raw"]
    paired["delta_brier"] = paired["brier_ft"] - paired["brier_raw"]
    paired["delta_log_loss"] = paired["log_loss_ft"] - paired["log_loss_raw"]

    prevalences = {name: load_prevalence(name) for name in TARGET_MAP}

    rows: list[dict[str, float | str]] = []
    all_names = sorted(prevalences)
    for target in all_names:
        others = [name for name in all_names if name != target]
        pool_prev_mean = float(np.mean([prevalences[name] for name in others]))
        prev_distance = abs(prevalences[target] - pool_prev_mean)
        rows.append(
            {
                "target_dataset": target,
                "target_positive_rate": prevalences[target],
                "pool_positive_rate_mean": pool_prev_mean,
                "prevalence_distance": prev_distance,
            }
        )

    hom_proxy = pd.DataFrame(rows)
    agg = (
        paired.groupby("target_dataset")
        [["delta_roc_auc", "delta_pr_auc", "delta_brier", "delta_log_loss"]]
        .mean()
        .reset_index()
    )

    joined = agg.merge(hom_proxy, on="target_dataset", how="left")

    # Convert Brier/LogLoss into "improvement" orientation (higher is better)
    joined["improvement_brier"] = -joined["delta_brier"]
    joined["improvement_log_loss"] = -joined["delta_log_loss"]

    print("=== Classifier Homogeneity Proposal: Existing-Evidence Evaluation ===")
    print(f"paired_tabpfn_runs={len(paired)}")
    print("\nPer-target mean fine-tune deltas (domain_finetuned - raw):")
    print(
        joined[
            [
                "target_dataset",
                "delta_roc_auc",
                "delta_pr_auc",
                "delta_brier",
                "delta_log_loss",
                "prevalence_distance",
            ]
        ]
        .sort_values("prevalence_distance")
        .to_string(index=False)
    )

    metric_cols = [
        ("delta_roc_auc", "ROC AUC"),
        ("delta_pr_auc", "PR AUC"),
        ("improvement_brier", "Brier improvement"),
        ("improvement_log_loss", "LogLoss improvement"),
    ]
    print("\nCorrelation with homogeneity proxy (smaller prevalence_distance ~= more homogeneous):")
    for col, label in metric_cols:
        corr = float(np.corrcoef(joined["prevalence_distance"], joined[col])[0, 1])
        print(f"corr(prevalence_distance, {label}) = {corr:+.3f}")

    print("\nInterpretation guardrails:")
    print("- This is a post-hoc proxy check, not a causal test.")
    print("- n=4 targets is very small; use only for planning the next controlled run.")


if __name__ == "__main__":
    main()
