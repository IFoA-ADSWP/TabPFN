from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.optim import Adam

WORK_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = WORK_ROOT.parent / "TabPFN-upstream" / "src"
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation  # type: ignore[import-not-found]
from tabpfn.utils import meta_dataset_collator

DATA_DIR = WORK_ROOT / "data" / "raw"
BASELINE_RESULTS_PATH = WORK_ROOT / "data" / "processed" / "multi_dataset_regression_benchmark_results.csv"
RAW_TABPFN_RESULTS_PATH = WORK_ROOT / "outputs" / "current" / "tables" / "raw_tabpfn_regression_revalidation.csv"
TABLE_DIR = WORK_ROOT / "outputs" / "current" / "tables"
LOG_DIR = WORK_ROOT / "outputs" / "current" / "logs"

RANDOM_SEED = 42
TEST_SIZE = 0.20
GLOBAL_MAX_TRAIN = 300
GLOBAL_MAX_TEST = 2000
N_ESTIMATORS = 4
MAX_PRED_BATCH = 300
DEVICE = "cpu"

CONTEXT_SAMPLES = 64
MAX_FINETUNE_STEPS = 1
FINETUNE_LR = 1e-5

DATASETS = [
    ("freMTPL2freq.csv", "ClaimNb", "freMTPL2 Frequency (FR)"),
    ("eudirectlapse.csv", "prem_pure", "EU Direct Premium (pure)"),
    ("ausprivauto0405.csv", "VehValue", "AUS Auto Vehicle Value"),
]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scl", StandardScaler()),
                ]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def tabpfn_predict_batched(model: TabPFNRegressor, X: np.ndarray, batch_size: int = MAX_PRED_BATCH) -> np.ndarray:
    if X.shape[0] <= batch_size:
        return model.predict(X)
    batches = []
    for start in range(0, X.shape[0], batch_size):
        batches.append(model.predict(X[start : start + batch_size]))
    return np.concatenate(batches)


def evaluate_model(
    regressor: TabPFNRegressor,
    regressor_config: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    context_samples: int,
) -> dict[str, float]:
    eval_config = {
        **regressor_config,
        "inference_config": {"SUBSAMPLE_SAMPLES": context_samples},
    }
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)
    predictions = eval_regressor.predict(X_test)
    return regression_metrics(y_test, predictions)


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def run_one_step_finetune(
    regressor: TabPFNRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    context_samples: int,
    max_steps: int,
    seed: int,
) -> dict[str, float | int | str]:
    steps_executed = 0
    skipped_preprocess_errors = 0
    skipped_nonfinite_target = 0
    skipped_nonfinite_loss = 0
    last_step_loss = ""

    splitter = lambda features, target: train_test_split(features, target, test_size=0.3, random_state=seed)
    training_datasets = regressor.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        min(context_samples, len(X_train)),
    )

    optimizer = Adam(regressor.model_.parameters(), lr=FINETUNE_LR)

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
        loss = loss_fn(pred_logits, y_test_standardized.to(DEVICE)).mean()
        if not torch.isfinite(loss):
            skipped_nonfinite_loss += 1
            continue

        loss.backward()
        optimizer.step()

        last_step_loss = f"{float(loss.item()):.6f}"
        steps_executed += 1
        if steps_executed >= max_steps:
            break

    return {
        "finetune_steps_executed": steps_executed,
        "skipped_preprocess_errors": skipped_preprocess_errors,
        "skipped_nonfinite_target": skipped_nonfinite_target,
        "skipped_nonfinite_loss": skipped_nonfinite_loss,
        "last_step_loss": last_step_loss,
    }


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    started_at = pd.Timestamp.utcnow().isoformat()
    results = []
    dataset_meta = []

    print("Using upstream src:", UPSTREAM_SRC)
    print("Device:", DEVICE)

    for fname, target_col, label in DATASETS:
        df = pd.read_csv(DATA_DIR / fname)
        X = df.drop(columns=[target_col])
        y = pd.to_numeric(df[target_col], errors="coerce")
        keep = y.notna()
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
        )
        if len(X_train) > GLOBAL_MAX_TRAIN:
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=GLOBAL_MAX_TRAIN,
                random_state=RANDOM_SEED,
            )
        if len(X_test) > GLOBAL_MAX_TEST:
            _, X_test, _, y_test = train_test_split(
                X_test,
                y_test,
                test_size=GLOBAL_MAX_TEST,
                random_state=RANDOM_SEED,
            )

        preprocessor = build_preprocessor(X_train)
        X_train_arr = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
        X_test_arr = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
        y_train_arr = y_train.to_numpy(dtype=np.float32)
        y_test_arr = y_test.to_numpy(dtype=np.float32)

        rss_before = max_rss_mb()

        fit_start = time.time()
        regressor_config: dict[str, object] = {
            "device": DEVICE,
            "random_state": RANDOM_SEED,
            "n_estimators": N_ESTIMATORS,
            "ignore_pretraining_limits": True,
            "inference_precision": torch.float32,
        }
        regressor = TabPFNRegressor(
            **regressor_config,
            fit_mode="batched",
            differentiable_input=False,
        )
        regressor.fit(X_train_arr, y_train_arr)
        raw_metrics = evaluate_model(
            regressor,
            regressor_config,
            X_train_arr,
            y_train_arr,
            X_test_arr,
            y_test_arr,
            CONTEXT_SAMPLES,
        )
        fit_eval_time = time.time() - fit_start

        ft_start = time.time()
        ft_stats = run_one_step_finetune(
            regressor,
            X_train_arr,
            y_train_arr,
            context_samples=CONTEXT_SAMPLES,
            max_steps=MAX_FINETUNE_STEPS,
            seed=RANDOM_SEED,
        )
        ft_metrics = evaluate_model(
            regressor,
            regressor_config,
            X_train_arr,
            y_train_arr,
            X_test_arr,
            y_test_arr,
            CONTEXT_SAMPLES,
        )
        ft_eval_time = time.time() - ft_start

        rss_after = max_rss_mb()

        row = {
            "dataset": label,
            "target_col": target_col,
            "rows_total": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "n_features_after_preprocess": int(X_train_arr.shape[1]),
            "device": DEVICE,
            "seed": RANDOM_SEED,
            "n_estimators": N_ESTIMATORS,
            "context_samples": CONTEXT_SAMPLES,
            "max_finetune_steps": MAX_FINETUNE_STEPS,
            "fit_eval_time_s": float(fit_eval_time),
            "finetune_eval_time_s": float(ft_eval_time),
            "wall_time_total_s": float(fit_eval_time + ft_eval_time),
            "max_rss_mb": max(rss_before, rss_after),
            "raw_MAE": raw_metrics["MAE"],
            "raw_RMSE": raw_metrics["RMSE"],
            "raw_R2": raw_metrics["R2"],
            "ft_MAE": ft_metrics["MAE"],
            "ft_RMSE": ft_metrics["RMSE"],
            "ft_R2": ft_metrics["R2"],
            "delta_MAE": ft_metrics["MAE"] - raw_metrics["MAE"],
            "delta_RMSE": ft_metrics["RMSE"] - raw_metrics["RMSE"],
            "delta_R2": ft_metrics["R2"] - raw_metrics["R2"],
            **ft_stats,
        }
        results.append(row)

        dataset_meta.append(
            {
                "dataset_file": fname,
                "dataset": label,
                "target_col": target_col,
                "rows_total": int(len(df)),
                "rows_train": int(len(X_train)),
                "rows_test": int(len(X_test)),
                "n_features_after_preprocess": int(X_train_arr.shape[1]),
            }
        )

        print(
            f"{label:<30} raw(MAE={row['raw_MAE']:.4f}, RMSE={row['raw_RMSE']:.4f}, R2={row['raw_R2']:.4f}) | "
            f"ft(MAE={row['ft_MAE']:.4f}, RMSE={row['ft_RMSE']:.4f}, R2={row['ft_R2']:.4f}) | "
            f"steps={row['finetune_steps_executed']} skips(loss={row['skipped_nonfinite_loss']})"
        )

    results_df = pd.DataFrame(results)

    detailed_out = TABLE_DIR / "tabpfn_regression_finetune_vs_raw.csv"
    results_df.to_csv(detailed_out, index=False)

    fine_model_rows = results_df[["dataset", "ft_MAE", "ft_RMSE", "ft_R2", "wall_time_total_s"]].rename(
        columns={
            "ft_MAE": "MAE",
            "ft_RMSE": "RMSE",
            "ft_R2": "R2",
            "wall_time_total_s": "time_s",
        }
    )
    fine_model_rows.insert(1, "model", "TabPFNRegressorFineTuned")

    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    raw_df = pd.read_csv(RAW_TABPFN_RESULTS_PATH)
    combined_df = pd.concat([baseline_df, raw_df.drop(columns=["max_rss_mb"]), fine_model_rows], ignore_index=True)

    combined_out = TABLE_DIR / "multi_dataset_regression_benchmark_with_tabpfn_finetuned.csv"
    combined_df.to_csv(combined_out, index=False)

    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": pd.Timestamp.utcnow().isoformat(),
        "device": DEVICE,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "global_max_train": GLOBAL_MAX_TRAIN,
        "global_max_test": GLOBAL_MAX_TEST,
        "n_estimators": N_ESTIMATORS,
        "max_pred_batch": MAX_PRED_BATCH,
        "context_samples": CONTEXT_SAMPLES,
        "max_finetune_steps": MAX_FINETUNE_STEPS,
        "finetune_lr": FINETUNE_LR,
        "ignore_pretraining_limits": True,
        "upstream_src": str(UPSTREAM_SRC),
        "datasets": dataset_meta,
    }
    summary_out = TABLE_DIR / "tabpfn_regression_finetune_vs_raw_run_meta.json"
    summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log_out = LOG_DIR / "tabpfn_regression_finetune_vs_raw.md"
    log_lines = [
        "# TabPFN Regressor Fine-Tuned vs Raw (Benchmark Revalidation)",
        "",
        f"- Started: {summary['started_at_utc']}",
        f"- Finished: {summary['finished_at_utc']}",
        f"- Device: {DEVICE}",
        f"- Random seed: {RANDOM_SEED}",
        f"- Global max train rows: {GLOBAL_MAX_TRAIN}",
        f"- Global max test rows: {GLOBAL_MAX_TEST}",
        f"- n_estimators: {N_ESTIMATORS}",
        f"- context_samples: {CONTEXT_SAMPLES}",
        f"- max_finetune_steps: {MAX_FINETUNE_STEPS}",
        "",
        "## Results",
        "",
    ]
    for row in results:
        log_lines.append(
            f"- {row['dataset']}: raw(MAE={row['raw_MAE']:.4f}, RMSE={row['raw_RMSE']:.4f}, R2={row['raw_R2']:.4f}) "
            f"-> ft(MAE={row['ft_MAE']:.4f}, RMSE={row['ft_RMSE']:.4f}, R2={row['ft_R2']:.4f}), "
            f"delta_R2={row['delta_R2']:.6f}, steps={row['finetune_steps_executed']}, "
            f"skips[p={row['skipped_preprocess_errors']}, t={row['skipped_nonfinite_target']}, l={row['skipped_nonfinite_loss']}], "
            f"last_loss={row['last_step_loss'] or 'n/a'}"
        )
    log_lines.extend(
        [
            "",
            f"- Detailed table: {detailed_out}",
            f"- Combined benchmark table: {combined_out}",
            f"- Run metadata: {summary_out}",
        ]
    )
    log_out.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("\nSaved detailed results to", detailed_out)
    print("Saved combined results to", combined_out)
    print("Saved metadata to", summary_out)
    print("Saved log to", log_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
