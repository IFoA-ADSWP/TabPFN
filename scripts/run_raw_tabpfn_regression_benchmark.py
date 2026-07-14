from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

WORK_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = WORK_ROOT.parent / 'TabPFN-upstream' / 'src'
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from tabpfn import TabPFNRegressor

DATA_DIR = WORK_ROOT / 'data' / 'raw'
BASELINE_RESULTS_PATH = WORK_ROOT / 'data' / 'processed' / 'multi_dataset_regression_benchmark_results.csv'
TABLE_DIR = WORK_ROOT / 'outputs' / 'current' / 'tables'
LOG_DIR = WORK_ROOT / 'outputs' / 'current' / 'logs'

RANDOM_SEED = 42
TEST_SIZE = 0.20
GLOBAL_MAX_TRAIN = 300    # match typical CPU-feasible TabPFN scale (~diabetes example)
GLOBAL_MAX_TEST = 2000    # evaluate on broader test set, batched
N_ESTIMATORS = 4          # reasonable ensemble
MAX_PRED_BATCH = 300      # keep (train+batch) ≈ 600 rows for manageable CPU attn
DEVICE = 'cpu'

DATASETS = [
    ('freMTPL2freq.csv', 'ClaimNb', 'freMTPL2 Frequency (FR)'),
    ('eudirectlapse.csv', 'prem_pure', 'EU Direct Premium (pure)'),
    ('ausprivauto0405.csv', 'VehValue', 'AUS Auto Vehicle Value'),
]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(
            (
                'num',
                Pipeline([
                    ('imp', SimpleImputer(strategy='median')),
                    ('scl', StandardScaler()),
                ]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                'cat',
                Pipeline([
                    ('imp', SimpleImputer(strategy='most_frequent')),
                    ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ]),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder='drop')


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': rmse(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred)),
    }


def tabpfn_predict_batched(model: TabPFNRegressor, X: np.ndarray, batch_size: int = MAX_PRED_BATCH) -> np.ndarray:
    if X.shape[0] <= batch_size:
        return model.predict(X)
    batches = []
    for start in range(0, X.shape[0], batch_size):
        batches.append(model.predict(X[start:start + batch_size]))
    return np.concatenate(batches)


def max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    started_at = pd.Timestamp.utcnow().isoformat()
    results = []
    dataset_meta = []

    print('Using upstream src:', UPSTREAM_SRC)
    print('Device:', DEVICE)

    for fname, target_col, label in DATASETS:
        df = pd.read_csv(DATA_DIR / fname)
        X = df.drop(columns=[target_col])
        y = pd.to_numeric(df[target_col], errors='coerce')
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

        rss_before = max_rss_mb()
        t0 = time.time()
        regressor = TabPFNRegressor(
            device=DEVICE,
            random_state=RANDOM_SEED,
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
        )
        regressor.fit(X_train_arr, y_train.to_numpy(dtype=np.float32))
        predictions = tabpfn_predict_batched(regressor, X_test_arr)
        elapsed = time.time() - t0
        rss_after = max_rss_mb()

        metrics = regression_metrics(y_test, predictions)
        row = {
            'dataset': label,
            'model': 'TabPFNRegressor',
            'time_s': elapsed,
            'max_rss_mb': max(rss_before, rss_after),
            **metrics,
        }
        results.append(row)
        dataset_meta.append(
            {
                'dataset_file': fname,
                'dataset': label,
                'target_col': target_col,
                'rows_total': int(len(df)),
                'rows_train': int(len(X_train)),
                'rows_test': int(len(X_test)),
                'n_features_after_preprocess': int(X_train_arr.shape[1]),
            }
        )
        print(
            f"{label:<30} MAE={row['MAE']:.4f} RMSE={row['RMSE']:.4f} "
            f"R2={row['R2']:.4f} ({row['time_s']:.1f}s, {row['max_rss_mb']:.1f} MB)"
        )

    results_df = pd.DataFrame(results)
    raw_out = TABLE_DIR / 'raw_tabpfn_regression_revalidation.csv'
    results_df.to_csv(raw_out, index=False)

    combined_out = TABLE_DIR / 'multi_dataset_regression_benchmark_with_tabpfn_revalidated.csv'
    baseline_df = pd.read_csv(BASELINE_RESULTS_PATH)
    combined_df = pd.concat([baseline_df, results_df.drop(columns=['max_rss_mb'])], ignore_index=True)
    combined_df.to_csv(combined_out, index=False)

    summary = {
        'started_at_utc': started_at,
        'finished_at_utc': pd.Timestamp.utcnow().isoformat(),
        'device': DEVICE,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'global_max_train': GLOBAL_MAX_TRAIN,
        'global_max_test': GLOBAL_MAX_TEST,
        'n_estimators': N_ESTIMATORS,
        'max_pred_batch': MAX_PRED_BATCH,
        'ignore_pretraining_limits': True,
        'upstream_src': str(UPSTREAM_SRC),
        'datasets': dataset_meta,
    }
    summary_out = TABLE_DIR / 'raw_tabpfn_regression_revalidation_run_meta.json'
    summary_out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    log_out = LOG_DIR / 'raw_tabpfn_regression_revalidation.md'
    log_lines = [
        '# Raw TabPFN Regressor Revalidation',
        '',
        f'- Started: {summary["started_at_utc"]}',
        f'- Finished: {summary["finished_at_utc"]}',
        f'- Device: {DEVICE}',
        f'- Random seed: {RANDOM_SEED}',
        f'- Split: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test',
        f'- Global max train rows: {GLOBAL_MAX_TRAIN}',
        '- ignore_pretraining_limits: True',
        '',
        '## Results',
        '',
    ]
    for row in results:
        log_lines.append(
            f"- {row['dataset']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R2={row['R2']:.4f}, time_s={row['time_s']:.3f}, max_rss_mb={row['max_rss_mb']:.1f}"
        )
    log_lines.extend(
        [
            '',
            f'- Raw results table: {raw_out}',
            f'- Combined comparison table: {combined_out}',
            f'- Run metadata: {summary_out}',
        ]
    )
    log_out.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')

    print('\nSaved raw results to', raw_out)
    print('Saved combined results to', combined_out)
    print('Saved metadata to', summary_out)
    print('Saved log to', log_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
