"""Re-score cached TabPFN/GBDT fold predictions on log loss + Brier.

The v1 insurance metric (1-auc) is rank-invariant to probability calibration,
so balance_probabilities cannot show a signal there. Insurance-native metrics
(log loss / Brier) measure calibration directly. This walks the per-fold
result caches (test-split probas, matching v1's reported metric) and rescues
the evidence without any refits.

Usage: /tmp/tabarena/.venv-ta/bin/python scripts/eval/insurance_benchmark_v1/rescore_focused_imbalance_logloss.py

Post-run only: requires scripts/experiments/ task caches from completed v1 + pilot
runs (not committed).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

REPO = Path(__file__).resolve().parents[3]
V1_CACHE = REPO / "scripts/experiments/insurance_benchmark_v1" / "data"
PILOT_CACHE = REPO / "scripts/experiments/insurance_imbalance_pilot" / "data"
OUT = REPO / "scripts/eval/insurance_benchmark_v1" / "focused_imbalance_logloss.csv"

DATASETS = ["coil2000-75f8a6315d24", "uslapseagent-7021b667ce68"]
FOLDS = 5
SPLIT = "test"  # v1's reported metric is test-split (verified against results_per_split.csv)

# cache dir -> (display method, params)
METHODS = {
    "CatBoost_c1_default_HOLDOUT": ("CAT (default)", {"balance_probabilities": False, "n_estimators": None}),
    "LightGBM_c1_default_HOLDOUT": ("GBM (default)", {"balance_probabilities": False, "n_estimators": None}),
    "XGBoost_c1_default_HOLDOUT": ("XGB (default)", {"balance_probabilities": False, "n_estimators": None}),
    "TabPFNClient_c1_default_HOLDOUT": ("TabPFNClient (default)", {"balance_probabilities": False, "n_estimators": 8}),
    "TabPFNClient-balanced_c1_default_HOLDOUT": ("TabPFNClient-balanced (default)", {"balance_probabilities": True, "n_estimators": 8}),
}
CACHE_ROOT = {
    "CatBoost_c1_default_HOLDOUT": V1_CACHE,
    "LightGBM_c1_default_HOLDOUT": V1_CACHE,
    "XGBoost_c1_default_HOLDOUT": V1_CACHE,
    "TabPFNClient_c1_default_HOLDOUT": V1_CACHE,
    "TabPFNClient-balanced_c1_default_HOLDOUT": PILOT_CACHE,
}


def main() -> None:
    rows = []
    for ds in DATASETS:
        for model_key, (display, params) in METHODS.items():
            for fold in range(FOLDS):
                p = CACHE_ROOT[model_key] / model_key / f"{ds}/0_{fold}/results.pkl"
                if not p.exists():
                    print(f"MISSING {ds} {model_key} fold {fold}")
                    continue
                obj = joblib.load(p)
                sa = obj["simulation_artifacts"]
                proba = sa[f"pred_proba_dict_{SPLIT}"][model_key]
                y = sa[f"y_{SPLIT}"]
                rows.append({
                    "dataset": ds,
                    "method": display,
                    "fold": fold,
                    "log_loss": log_loss(y, proba, labels=[0, 1]),
                    "brier": brier_score_loss(y, proba),
                    **params,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")

    # quick summary
    summ = (
        df.groupby(["dataset", "method"])[["log_loss", "brier"]]
        .mean()
        .round(5)
    )
    print(summ.to_string())


if __name__ == "__main__":
    main()
