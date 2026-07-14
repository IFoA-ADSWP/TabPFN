"""Run a small TabPFN regressor fine-tuning smoke test on a real dataset.

This script mirrors the classifier smoke harness for quick local readiness checks.
It prefers the local upstream TabPFN source tree so fine-tuning APIs are available
when installed package versions differ.
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam


def configure_import_path(prefer_upstream_src: bool) -> Path | None:
    work_repo_root = Path(__file__).resolve().parents[1]
    upstream_src = work_repo_root.parent / "TabPFN-upstream" / "src"
    if prefer_upstream_src and upstream_src.exists():
        sys.path.insert(0, str(upstream_src))
        return upstream_src
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small TabPFN regressor fine-tune smoke test")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/freMTPL2freq.csv"),
        help="CSV dataset path, relative to TabPFN-work-scott root unless absolute",
    )
    parser.add_argument("--target-col", type=str, default="ClaimNb")
    parser.add_argument(
        "--target-transform",
        type=str,
        choices=["none", "log1p", "claimfreq_raw", "claimfreq_log1p"],
        default="none",
        help="Optional target transform for regression stability experiments",
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--context-samples", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--max-finetune-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path for the saved fine-tuned model (.tabpfn_fit). Defaults to outputs/current/models/<timestamp>_...tabpfn_fit",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/current/tables/tabpfn_finetune_regressor_trial_results.csv"),
        help="CSV file to append structured trial results to",
    )
    parser.add_argument(
        "--logbook-path",
        type=Path,
        default=Path("outputs/current/logs/tabpfn_finetune_regressor_logbook.md"),
        help="Markdown logbook file to append per-run summaries",
    )
    parser.add_argument("--prefer-upstream-src", action="store_true", default=True)
    parser.add_argument("--no-prefer-upstream-src", dest="prefer_upstream_src", action="store_false")
    parser.add_argument(
        "--positive-claims-pool",
        action="store_true",
        default=False,
        help="Restrict fine-tune training pool to rows with ClaimNb > 0 (Stage R2 ablation test)",
    )
    return parser.parse_args()


def resolve_data_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def resolve_log_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def resolve_logbook_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def resolve_save_path(path: Path | None, target_col: str, device: str, rows: int) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if path is not None:
        return path if path.is_absolute() else repo_root / path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_name = f"{timestamp}_tabpfn_finetune_regressor_{target_col}_{device}_{rows}.tabpfn_fit"
    return repo_root / "outputs" / "current" / "models" / file_name


def build_target(df: pd.DataFrame, target_col: str, target_transform: str) -> np.ndarray:
    if target_transform in {"claimfreq_raw", "claimfreq_log1p"}:
        if "ClaimNb" not in df.columns or "Exposure" not in df.columns:
            raise ValueError("claimfreq transforms require 'ClaimNb' and 'Exposure' columns")
        exposure = df["Exposure"].replace(0, np.nan)
        y = (df["ClaimNb"] / exposure).fillna(0.0).to_numpy(dtype=np.float32)
        if target_transform == "claimfreq_log1p":
            y = np.log1p(y)
        return y

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col].to_numpy(dtype=np.float32)
    if target_transform == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p target transform requires non-negative target values")
        y = np.log1p(y)
    return y


def load_subset(
    data_path: Path,
    target_col: str,
    rows: int,
    seed: int,
    target_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)

    sampled = df.sample(n=min(rows, len(df)), random_state=seed)
    y = build_target(sampled, target_col, target_transform)
    X = pd.get_dummies(sampled.drop(columns=[target_col]), drop_first=False).to_numpy(dtype=np.float32)
    return X, y


def evaluate_model(
    regressor: Any,
    regressor_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    context_samples: int,
) -> tuple[float, float, float]:
    from tabpfn import TabPFNRegressor
    from tabpfn.finetune_utils import clone_model_for_evaluation

    eval_config = {
        **regressor_config,
        "inference_config": {"SUBSAMPLE_SAMPLES": context_samples},
    }
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)
    predictions = eval_regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2


def append_result_row(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with log_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = list(reader)
        existing_fieldnames = reader.fieldnames or []

    merged_fieldnames = list(existing_fieldnames)
    for key in row.keys():
        if key not in merged_fieldnames:
            merged_fieldnames.append(key)

    if merged_fieldnames != existing_fieldnames:
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                normalized_row = {field: existing_row.get(field, "") for field in merged_fieldnames}
                writer.writerow(normalized_row)

    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        normalized_row = {field: row.get(field, "") for field in merged_fieldnames}
        writer.writerow(normalized_row)


def append_logbook_entry(logbook_path: Path, row: dict[str, Any]) -> None:
    logbook_path.parent.mkdir(parents=True, exist_ok=True)
    if not logbook_path.exists():
        with logbook_path.open("w") as handle:
            handle.write("# TabPFN Regressor Fine-Tune Trial Logbook\n\n")

    with logbook_path.open("a") as handle:
        handle.write(f"## Run {row['timestamp_utc']}\n\n")
        handle.write("### Configuration\n")
        handle.write(f"- Script: {row['script_name']}\n")
        handle.write(f"- Data path: {row['data_path']}\n")
        handle.write(f"- Target: {row['target_col']}\n")
        handle.write(f"- Target transform: {row['target_transform']}\n")
        handle.write(f"- Rows train/finetune/test: {row['train_rows']}/{row['finetune_rows']}/{row['test_rows']}\n")
        if row.get('positive_claims_pool'):
            handle.write(f"- **Ablation: Positive-claims-only fine-tune pool**\n")
        handle.write(f"- Device: {row['device']}\n")
        handle.write(f"- Context samples: {row['context_samples']}\n")
        handle.write(f"- Max fine-tune steps: {row['max_finetune_steps']}\n")
        handle.write(f"- Seed: {row['seed']}\n")
        handle.write("\n")
        handle.write("### Outcomes\n")
        handle.write(f"- Fine-tune steps executed: {row['finetune_steps_executed']}\n")
        handle.write(f"- Skipped preprocess errors: {row['skipped_preprocess_errors']}\n")
        handle.write(f"- Skipped non-finite target: {row['skipped_nonfinite_target']}\n")
        handle.write(f"- Skipped non-finite loss/logits: {row['skipped_nonfinite_loss']}\n")
        handle.write(f"- Last step loss: {row['last_step_loss'] or 'n/a'}\n")
        handle.write(
            "- Initial metrics (MSE/MAE/R2): "
            f"{row['initial_mse']}/{row['initial_mae']}/{row['initial_r2']}\n"
        )
        handle.write(
            "- Post-step metrics (MSE/MAE/R2): "
            f"{row['post_step_mse']}/{row['post_step_mae']}/{row['post_step_r2']}\n"
        )
        handle.write(f"- Wall time sec: {row['wall_time_sec']}\n")
        handle.write(f"- Max RSS bytes: {row['max_rss_bytes']}\n")
        handle.write(f"- Saved model path: {row['save_path']}\n\n")


def main() -> None:
    args = parse_args()
    wall_start = time.perf_counter()
    upstream_src = configure_import_path(args.prefer_upstream_src)

    from tabpfn import TabPFNRegressor, save_fitted_tabpfn_model
    from tabpfn.utils import meta_dataset_collator

    data_path = resolve_data_path(args.data_path)
    log_path = resolve_log_path(args.log_path)
    logbook_path = resolve_logbook_path(args.logbook_path)
    save_path = resolve_save_path(args.save_path, args.target_col, args.device, args.rows)

    # Load data as dataframes for potential filtering before conversion to numpy
    df = pd.read_csv(data_path)
    df_sampled = df.sample(n=min(args.rows, len(df)), random_state=args.seed)

    # Train/test split at dataframe level
    df_train, df_test = train_test_split(
        df_sampled,
        test_size=0.3,
        random_state=args.seed,
    )

    # Stage R2 ablation: optionally restrict fine-tune pool to positive claims only
    df_train_finetune = df_train
    finetune_pool_note = ""
    if args.positive_claims_pool and "ClaimNb" in df_train.columns:
        df_train_finetune = df_train[df_train["ClaimNb"] > 0].copy()
        n_before = len(df_train)
        n_after = len(df_train_finetune)
        finetune_pool_note = f" (positive-claims-only: {n_after}/{n_before} rows retained)"

    # Build one shared one-hot feature table to keep train/test columns aligned.
    encoded_features = pd.get_dummies(
        df_sampled.drop(columns=[args.target_col]),
        drop_first=False,
    )

    # Convert dataframes to numpy arrays
    y_train = build_target(df_train, args.target_col, args.target_transform)
    y_train_finetune = build_target(df_train_finetune, args.target_col, args.target_transform)
    y_test = build_target(df_test, args.target_col, args.target_transform)

    X_train = encoded_features.loc[df_train.index].to_numpy(dtype=np.float32)
    X_train_finetune = encoded_features.loc[df_train_finetune.index].to_numpy(dtype=np.float32)
    X_test = encoded_features.loc[df_test.index].to_numpy(dtype=np.float32)

    # Use full X, y for the overall row count
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "inference_precision": torch.float32,
    }

    regressor = TabPFNRegressor(
        **regressor_config,
        fit_mode="batched",
        differentiable_input=False,
    )

    splitter = lambda features, target: train_test_split(features, target, test_size=0.3, random_state=args.seed)
    # Use finetune pool for fine-tune dataset generation, evaluation pool for evaluation
    training_datasets = regressor.get_preprocessed_datasets(
        X_train_finetune,
        y_train_finetune,
        splitter,
        min(args.context_samples, len(X_train_finetune)),
    )
    optimizer = Adam(regressor.model_.parameters(), lr=1e-5)

    mse_before, mae_before, r2_before = evaluate_model(
        regressor,
        regressor_config,
        X_train,
        y_train,
        X_test,
        y_test,
        args.context_samples,
    )

    steps_executed = 0
    last_loss = None
    skipped_preprocess_errors = 0
    skipped_nonfinite_target = 0
    skipped_nonfinite_loss = 0

    for idx in range(len(training_datasets)):
        try:
            sample = training_datasets[idx]
        except Exception:
            skipped_preprocess_errors += 1
            continue

        data_batch = meta_dataset_collator([sample])
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
        ) = data_batch

        if not torch.isfinite(y_test_standardized).all():
            skipped_nonfinite_target += 1
            continue

        optimizer.zero_grad()
        regressor.normalized_bardist_ = normalized_bardist[0]
        regressor.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
        pred_logits, _, _ = regressor.forward(X_tests_preprocessed)

        if not torch.isfinite(pred_logits).all():
            skipped_nonfinite_loss += 1
            continue

        loss_fn = normalized_bardist[0]
        loss = loss_fn(pred_logits, y_test_standardized.to(args.device)).mean()
        if not torch.isfinite(loss):
            skipped_nonfinite_loss += 1
            continue

        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        steps_executed += 1
        if steps_executed >= args.max_finetune_steps:
            break

    mse_after, mae_after, r2_after = evaluate_model(
        regressor,
        regressor_config,
        X_train,
        y_train,
        X_test,
        y_test,
        args.context_samples,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fitted_tabpfn_model(regressor, save_path)

    wall_time_sec = time.perf_counter() - wall_start
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    result_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "data_path": str(data_path),
        "target_col": args.target_col,
        "target_transform": args.target_transform,
        "rows": len(X),
        "train_rows": len(X_train),
        "finetune_rows": len(X_train_finetune),
        "test_rows": len(X_test),
        "positive_claims_pool": args.positive_claims_pool,
        "device": args.device,
        "prefer_upstream_src": args.prefer_upstream_src,
        "upstream_src": str(upstream_src) if upstream_src is not None else "",
        "context_samples": args.context_samples,
        "n_estimators": args.n_estimators,
        "max_finetune_steps": args.max_finetune_steps,
        "seed": args.seed,
        "save_path": str(save_path),
        "finetune_steps_executed": steps_executed,
        "skipped_preprocess_errors": skipped_preprocess_errors,
        "skipped_nonfinite_target": skipped_nonfinite_target,
        "skipped_nonfinite_loss": skipped_nonfinite_loss,
        "last_step_loss": f"{last_loss:.6f}" if last_loss is not None else "",
        "initial_mse": f"{mse_before:.6f}",
        "initial_mae": f"{mae_before:.6f}",
        "initial_r2": f"{r2_before:.6f}",
        "post_step_mse": f"{mse_after:.6f}",
        "post_step_mae": f"{mae_after:.6f}",
        "post_step_r2": f"{r2_after:.6f}",
        "wall_time_sec": f"{wall_time_sec:.6f}",
        "max_rss_bytes": max_rss_bytes,
    }
    append_result_row(log_path, result_row)
    append_logbook_entry(logbook_path, result_row)

    print("=== Small TabPFN Regressor Fine-Tune Trial ===")
    print(f"data_path={data_path}")
    print(f"target_col={args.target_col}")
    print(f"target_transform={args.target_transform}")
    print(f"rows={len(X)} train_rows={len(X_train)} finetune_rows={len(X_train_finetune)} test_rows={len(X_test)}")
    if args.positive_claims_pool:
        print(f"** ABLATION: positive-claims-only fine-tune pool (retained {len(X_train_finetune)}/{len(X_train)} rows)**")
    print(f"device={args.device}")
    print(f"prefer_upstream_src={args.prefer_upstream_src}")
    if upstream_src is not None:
        print(f"upstream_src={upstream_src}")
    print(f"log_path={log_path}")
    print(f"logbook_path={logbook_path}")
    print(f"save_path={save_path}")
    print(f"context_samples={args.context_samples} n_estimators={args.n_estimators}")
    print(f"initial_eval mse={mse_before:.4f} mae={mae_before:.4f} r2={r2_before:.4f}")
    print(f"finetune_steps_executed={steps_executed}")
    print(
        "skips "
        f"preprocess_errors={skipped_preprocess_errors} "
        f"nonfinite_target={skipped_nonfinite_target} "
        f"nonfinite_loss={skipped_nonfinite_loss}"
    )
    if last_loss is not None:
        print(f"last_step_loss={last_loss:.4f}")
    print(f"post_step_eval mse={mse_after:.4f} mae={mae_after:.4f} r2={r2_after:.4f}")
    print(f"wall_time_sec={wall_time_sec:.4f}")
    print(f"max_rss_bytes={max_rss_bytes}")


if __name__ == "__main__":
    main()
