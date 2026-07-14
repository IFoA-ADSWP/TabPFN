import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize classifier homogeneity checklist runs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to include")
    parser.add_argument("--target-rows", type=int, default=800)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--max-finetune-steps", type=int, default=1)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["homogeneous", "heterogeneous", "all"],
        help="Pool policies to include",
    )
    parser.add_argument("--compare-a", type=str, default="homogeneous")
    parser.add_argument("--compare-b", type=str, default="heterogeneous")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/current/tables/classifier_homogeneity_smoke_seed42_summary.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs_path = Path("outputs/current/tables/domain_finetune_study_runs.csv")
    runs = pd.read_csv(runs_path)

    flt = runs[
        (runs["stage"] == "A")
        & (runs["seed"].isin(args.seeds))
        & (runs["target_rows"] == args.target_rows)
        & (runs["tabpfn_context_samples"] == args.context)
        & (runs["tabpfn_max_finetune_steps"] == args.max_finetune_steps)
        & (runs["pool_policy"].isin(args.policies))
    ]

    rows: list[dict[str, object]] = []
    for (seed, target, policy), group in flt.groupby(["seed", "target_dataset", "pool_policy"]):
        raw = group[(group["model_family"] == "tabpfn") & (group["model_variant"] == "raw")].tail(1)
        tuned = group[
            (group["model_family"] == "tabpfn") & (group["model_variant"] == "domain_finetuned")
        ].tail(1)
        if raw.empty or tuned.empty:
            continue

        r = raw.iloc[0]
        t = tuned.iloc[0]
        rows.append(
            {
                "seed": seed,
                "target_dataset": target,
                "pool_policy": policy,
                "selected_pool_datasets": t.get("selected_pool_datasets", ""),
                "selected_pool_mean_prev_distance": t.get("selected_pool_mean_prev_distance", ""),
                "raw_roc_auc": r["roc_auc"],
                "ft_roc_auc": t["roc_auc"],
                "delta_roc_auc": t["roc_auc"] - r["roc_auc"],
                "raw_pr_auc": r["pr_auc"],
                "ft_pr_auc": t["pr_auc"],
                "delta_pr_auc": t["pr_auc"] - r["pr_auc"],
                "raw_brier": r["brier"],
                "ft_brier": t["brier"],
                "delta_brier": t["brier"] - r["brier"],
                "raw_log_loss": r["log_loss"],
                "ft_log_loss": t["log_loss"],
                "delta_log_loss": t["log_loss"] - r["log_loss"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["seed", "target_dataset", "pool_policy"])
    out_path = args.out
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path} rows={len(out)}")
    print(out.to_string(index=False))

    means = (
        out.groupby("pool_policy")[["delta_roc_auc", "delta_pr_auc", "delta_brier", "delta_log_loss"]]
        .mean()
        .reset_index()
    )
    print("\npolicy_means")
    print(means.to_string(index=False))

    # Checklist decision support: compare-a versus compare-b (higher ROC/PR better).
    if {args.compare_a, args.compare_b}.issubset(set(out["pool_policy"].unique())):
        pivot = out[out["pool_policy"].isin([args.compare_a, args.compare_b])].pivot_table(
            index=["seed", "target_dataset"],
            columns="pool_policy",
            values=["delta_roc_auc", "delta_pr_auc"],
            aggfunc="mean",
        )
        roc_win = (pivot[("delta_roc_auc", args.compare_a)] > pivot[("delta_roc_auc", args.compare_b)]).sum()
        pr_win = (pivot[("delta_pr_auc", args.compare_a)] > pivot[("delta_pr_auc", args.compare_b)]).sum()
        total = len(pivot)
        print("\nhead_to_head_counts")
        print(f"{args.compare_a}_wins_roc={int(roc_win)}/{total}")
        print(f"{args.compare_a}_wins_pr={int(pr_win)}/{total}")


if __name__ == "__main__":
    main()
