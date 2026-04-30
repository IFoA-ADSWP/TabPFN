from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ClaimNb finiteness issues in tiny TabPFN regressor pass")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/freMTPL2freq.csv"))
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--context-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/current/logs/claimnb_finiteness_checkpoints.csv"),
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def configure_import_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    upstream_src = repo_root.parent / "TabPFN-upstream" / "src"
    sys.path.insert(0, str(upstream_src))
    return upstream_src


def build_target(df: pd.DataFrame, transform: str) -> np.ndarray:
    if transform == "none":
        return df["ClaimNb"].to_numpy(dtype=np.float32)
    if transform == "claimfreq_log1p":
        exposure = df["Exposure"].replace(0, np.nan)
        freq = (df["ClaimNb"] / exposure).fillna(0.0).to_numpy(dtype=np.float32)
        return np.log1p(freq)
    raise ValueError(f"Unknown transform {transform}")


def append_rows(log_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def run_transform(df_full: pd.DataFrame, transform: str, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    from tabpfn import TabPFNRegressor
    from tabpfn.utils import meta_dataset_collator

    checkpoints: list[dict[str, object]] = []

    y_all = build_target(df_full, transform)
    X_all = pd.get_dummies(df_full.drop(columns=["ClaimNb"]), drop_first=False).to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        random_state=args.seed,
    )

    reg = TabPFNRegressor(
        ignore_pretraining_limits=True,
        device=args.device,
        n_estimators=2,
        random_state=args.seed,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=False,
    )

    splitter = lambda a, b: train_test_split(a, b, test_size=0.3, random_state=args.seed)
    datasets = reg.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        min(args.context_samples, len(X_train)),
    )
    finite_batch_count = 0
    first_failure_reason = ""

    max_checks = min(args.max_batches, len(datasets))
    for i in range(1, max_checks + 1):
        row: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "transform": transform,
            "batch_idx": i,
            "rows_total": len(df_full),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "y_train_min": float(np.min(y_train)),
            "y_train_max": float(np.max(y_train)),
            "y_train_mean": float(np.mean(y_train)),
            "y_test_std_finite": False,
            "logits_finite": False,
            "loss_vector_finite": False,
            "loss_mean_finite": False,
            "loss_mean": "",
            "skip_reason": "",
            "exception": "",
        }

        try:
            sample = datasets[i - 1]
        except Exception as exc:
            row["skip_reason"] = "preprocess_exception"
            row["exception"] = str(exc)
            if not first_failure_reason:
                first_failure_reason = str(row["skip_reason"])
            checkpoints.append(row)
            continue

        batch = meta_dataset_collator([sample])

        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            normalized_bardist,
            _,
            _,
            _,
        ) = batch

        row["y_test_std_finite"] = bool(torch.isfinite(y_test_standardized).all())

        if not row["y_test_std_finite"]:
            row["skip_reason"] = "nonfinite_target_standardized"
            if not first_failure_reason:
                first_failure_reason = str(row["skip_reason"])
            checkpoints.append(row)
            continue

        reg.normalized_bardist_ = normalized_bardist[0]
        reg.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
        pred_logits, _, _ = reg.forward(X_tests_preprocessed)

        row["logits_finite"] = bool(torch.isfinite(pred_logits).all())
        if not row["logits_finite"]:
            row["skip_reason"] = "nonfinite_logits"
            if not first_failure_reason:
                first_failure_reason = str(row["skip_reason"])
            checkpoints.append(row)
            continue

        loss_vec = normalized_bardist[0](pred_logits, y_test_standardized.to(args.device))
        row["loss_vector_finite"] = bool(torch.isfinite(loss_vec).all())

        loss_mean = loss_vec.mean()
        row["loss_mean_finite"] = bool(torch.isfinite(loss_mean))
        row["loss_mean"] = float(loss_mean.detach().cpu().item()) if row["loss_mean_finite"] else ""

        if not row["loss_vector_finite"] or not row["loss_mean_finite"]:
            row["skip_reason"] = "nonfinite_loss"
            if not first_failure_reason:
                first_failure_reason = str(row["skip_reason"])
        else:
            finite_batch_count += 1

        checkpoints.append(row)

    summary = {
        "transform": transform,
        "batches_checked": max_checks,
        "finite_batches": finite_batch_count,
        "first_failure_reason": first_failure_reason or "none",
    }
    return checkpoints, summary


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    upstream_src = configure_import_path()
    data_path = resolve_repo_path(args.data_path)
    log_path = resolve_repo_path(args.log_path)

    df = pd.read_csv(data_path).sample(n=min(args.rows, len(pd.read_csv(data_path))), random_state=args.seed)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for transform in ["none", "claimfreq_log1p"]:
        rows, summary = run_transform(df, transform, args)
        all_rows.extend(rows)
        summaries.append(summary)

    append_rows(log_path, all_rows)

    wall_time = time.perf_counter() - start
    print("=== ClaimNb Tiny Finiteness Diagnostic ===")
    print(f"data_path={data_path}")
    print(f"rows={len(df)} context={args.context_samples} max_batches={args.max_batches} device={args.device}")
    print(f"upstream_src={upstream_src}")
    print(f"checkpoint_log={log_path}")
    for s in summaries:
        print(
            "transform={transform} batches_checked={batches_checked} "
            "finite_batches={finite_batches} first_failure_reason={first_failure_reason}".format(**s)
        )
    print(f"wall_time_sec={wall_time:.4f}")


if __name__ == "__main__":
    main()
