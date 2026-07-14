"""Evaluate whether a regressor fine-tuning config is ready for Stage R3 reruns.

The gate is evaluated across a small seed panel using previously logged rows from
the regressor trial ledger. A configuration is considered ready only when the
latest run for each required seed executes at least the requested number of
fine-tune steps and the observed last-step losses remain finite and non-extreme.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cross-seed regressor stability gate")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv"),
        help="CSV ledger path, relative to TabPFN-work-scott root unless absolute",
    )
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument(
        "--target-transform",
        type=str,
        default="none",
        help="Target transform to evaluate, for example none or claimfreq_raw",
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--context-samples", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--max-finetune-steps", type=int, default=1)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 2025],
        help="Seed panel that must all satisfy the gate",
    )
    parser.add_argument(
        "--min-steps-executed",
        type=int,
        default=1,
        help="Minimum executed fine-tune steps required for every seed",
    )
    parser.add_argument(
        "--max-abs-loss",
        type=float,
        default=1e6,
        help="Maximum allowed absolute last-step loss for every seed",
    )
    parser.add_argument(
        "--max-loss-range",
        type=float,
        default=100.0,
        help="Maximum allowed range between min and max last-step loss across seeds",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        default=False,
        help="Exit non-zero when the gate is not satisfied",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def main() -> int:
    args = parse_args()
    log_path = resolve_path(args.log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"Regressor trial ledger not found: {log_path}")

    df = pd.read_csv(log_path)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["rows"] = pd.to_numeric(df["rows"], errors="coerce")
    df["context_samples"] = pd.to_numeric(df["context_samples"], errors="coerce")
    df["n_estimators"] = pd.to_numeric(df["n_estimators"], errors="coerce")
    df["max_finetune_steps"] = pd.to_numeric(df["max_finetune_steps"], errors="coerce")
    df["finetune_steps_executed"] = pd.to_numeric(df["finetune_steps_executed"], errors="coerce")

    filtered = df[
        (df["target_col"] == args.target_col)
        & (df["target_transform"].fillna("") == args.target_transform)
        & (df["rows"] == args.rows)
        & (df["context_samples"] == args.context_samples)
        & (df["device"] == args.device)
        & (df["n_estimators"] == args.n_estimators)
        & (df["max_finetune_steps"] == args.max_finetune_steps)
        & (df["seed"].isin(args.seeds))
    ].copy()

    filtered = filtered.sort_values("timestamp_utc")
    latest_by_seed = filtered.groupby("seed", as_index=False).tail(1).copy()
    latest_by_seed["last_step_loss_parsed"] = latest_by_seed["last_step_loss"].map(parse_float)
    latest_by_seed = latest_by_seed.sort_values("seed")

    observed_seeds = latest_by_seed["seed"].dropna().astype(int).tolist()
    missing_seeds = [seed for seed in args.seeds if seed not in observed_seeds]

    executed_ok = False
    finite_loss_ok = False
    non_extreme_loss_ok = False
    loss_range_ok = False
    loss_range = None

    if not latest_by_seed.empty and not missing_seeds:
        executed_ok = bool((latest_by_seed["finetune_steps_executed"] >= args.min_steps_executed).all())
        losses = latest_by_seed["last_step_loss_parsed"]
        finite_loss_ok = bool(losses.notna().all())
        if finite_loss_ok:
            non_extreme_loss_ok = bool((losses.abs() <= args.max_abs_loss).all())
            loss_range = float(losses.max() - losses.min())
            loss_range_ok = bool(loss_range <= args.max_loss_range)

    gate_ready = (
        not missing_seeds
        and executed_ok
        and finite_loss_ok
        and non_extreme_loss_ok
        and loss_range_ok
    )

    print("=== Regressor Stability Gate ===")
    print(f"log_path={log_path}")
    print(f"target_col={args.target_col}")
    print(f"target_transform={args.target_transform}")
    print(f"rows={args.rows} context_samples={args.context_samples} device={args.device}")
    print(f"seeds_required={args.seeds}")
    print(f"seeds_observed={observed_seeds}")
    print(f"missing_seeds={missing_seeds}")
    print(f"min_steps_executed={args.min_steps_executed}")
    print(f"max_abs_loss={args.max_abs_loss}")
    print(f"max_loss_range={args.max_loss_range}")
    print(f"executed_ok={executed_ok}")
    print(f"finite_loss_ok={finite_loss_ok}")
    print(f"non_extreme_loss_ok={non_extreme_loss_ok}")
    print(f"loss_range={loss_range if loss_range is not None else 'n/a'}")
    print(f"loss_range_ok={loss_range_ok}")
    print(f"gate_ready={gate_ready}")

    if not latest_by_seed.empty:
        display_columns = [
            "timestamp_utc",
            "seed",
            "finetune_steps_executed",
            "skipped_preprocess_errors",
            "skipped_nonfinite_target",
            "skipped_nonfinite_loss",
            "last_step_loss",
            "post_r2",
        ]
        renamed = latest_by_seed.rename(columns={"post_step_r2": "post_r2"})
        print("\nLatest matching rows:")
        print(renamed[display_columns].to_string(index=False))

    if args.strict_exit and not gate_ready:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())