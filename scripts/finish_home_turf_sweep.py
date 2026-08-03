"""Finisher: complete missing cells of home_turf_sweep_results.csv after the API degraded.

Reads the partial CSV, computes missing (dataset, n_rows, method, fold) combos for the
bemtl97 cells, runs them with a 300s per-fit cap (thread timeout), records failures,
and appends. Same preprocessing/methods/fold logic as run_home_turf_size_sweep.py.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_RAW = REPO / "data" / "raw"
EVAL_DIR = REPO / "scripts" / "eval" / "insurance_benchmark_v1"
OUT_CSV = EVAL_DIR / "home_turf_sweep_results.csv"
PER_FIT_TIMEOUT = 300.0

os.environ["TABPFN_CLIENT_TIMEOUT"] = "300"


def _load_api_key() -> None:
    """TABPFN_API_KEY env, else the first TABPFN_API_KEY= line from the repo-root
    .env or the file pointed to by TABPFN_ENV_FILE (if set)."""
    if os.environ.get("TABPFN_API_KEY"):
        return
    candidates = [
        Path.cwd() / ".env",
        Path(os.environ["TABPFN_ENV_FILE"]) if os.environ.get("TABPFN_ENV_FILE") else None,
    ]
    for p in candidates:
        if p and p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("TABPFN_API_KEY="):
                    os.environ["TABPFN_API_KEY"] = line.split("=", 1)[1].strip()
                    return
    raise RuntimeError(
        "TABPFN_API_KEY not set: export TABPFN_API_KEY=... or add a "
        "TABPFN_API_KEY=... line to a .env file in the repo root "
        "(or set TABPFN_ENV_FILE to point at one)"
    )


_load_api_key()


def load_Xy(name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = {
        "coil2000": ("coil2000.csv", "CARAVAN", []),
        "uslapseagent": ("uslapseagent.csv", "surrender", []),
        "bemtl97": ("bemtl97.csv", "claim", ["nclaims", "amount"]),
    }[name]
    file, target, drop = spec
    df = pd.read_csv(DATA_RAW / file).dropna(subset=[target]).reset_index(drop=True)
    y = df[target].to_numpy(dtype=np.int64)
    X = df.drop(columns=[target] + drop)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    return X.fillna(0.0).astype(np.float32).to_numpy(dtype=np.float32), y


def slice_data(X, y, n):
    if n is None or len(X) <= n:
        return X, y
    from sklearn.model_selection import train_test_split
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=42)
    return Xs, ys


def fit_predict(mname: str, mkw: dict, Xtr, ytr, Xte):
    if mname == "tabpfn":
        from tabpfn_client import TabPFNClassifier
        model = TabPFNClassifier(random_state=0, **mkw)
    elif mname == "cat":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=0, verbose=0)
    elif mname == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=0)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=0, device_type="cpu", verbose=-1)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)


def main() -> None:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    existing = pd.read_csv(OUT_CSV)
    have = {(r.dataset, r.n_rows, r.method, r.fold) for r in existing.itertuples()}

    # Full intended method matrix per cell
    def methods_for(n_rows, name):
        m = [("tabpfn", {"n_estimators": None})]
        if not (name == "bemtl97" and n_rows > 50000):  # trimmed largest cell
            m.append(("tabpfn", {"n_estimators": 1}))
        m += [("cat", {}), ("xgb", {}), ("lgbm", {})]
        return m

    new_rows = []
    for name in ["bemtl97"]:
        Xf, yf = load_Xy(name)
        for n in [5000, None]:
            X, y = slice_data(Xf, yf, n)
            n_rows = len(X)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                for mname, mkw in methods_for(n_rows, name):
                    if (name, n_rows, mname, fold) in have:
                        continue
                    row = {"dataset": name, "n_rows": n_rows, "method": mname, "fold": fold,
                           "n_estimators": mkw.get("n_estimators"),
                           "trimmed": bool(name == "bemtl97" and n_rows > 50000),
                           "error": "", "train_s": np.nan, "infer_s": np.nan,
                           "log_loss": np.nan, "brier": np.nan, "roc_auc": np.nan}
                    t0 = time.time()
                    try:
                        with ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(fit_predict, mname, mkw, X[tr], y[tr], X[te])
                            p = fut.result(timeout=PER_FIT_TIMEOUT)
                        row["train_s"] = round(time.time() - t0, 2)
                        pp = p if p.shape[1] > 1 else np.column_stack([1 - p[:, 0], p[:, 0]])
                        row["log_loss"] = log_loss(y[te], pp)
                        row["brier"] = brier_score_loss(y[te], pp[:, 1])
                        row["roc_auc"] = roc_auc_score(y[te], pp[:, 1])
                        print(f"OK {name} n={n_rows} {mname}{mkw} f{fold} ll={row['log_loss']:.4f} auc={row['roc_auc']:.4f} ({row['train_s']}s)", flush=True)
                    except FutTimeout:
                        row["error"] = f"TimeoutError: fit+pred >{PER_FIT_TIMEOUT}s"
                        print(f"TIMEOUT {name} n={n_rows} {mname}{mkw} f{fold}", flush=True)
                    except Exception as e:
                        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")
                        print(f"FAILED {name} n={n_rows} {mname}{mkw} f{fold}: {row['error']}", flush=True)
                    new_rows.append(row)
                    pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True).to_csv(OUT_CSV, index=False)

    print(f"finisher done: +{len(new_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
