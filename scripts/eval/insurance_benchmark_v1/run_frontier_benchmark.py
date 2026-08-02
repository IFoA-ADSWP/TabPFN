"""Insurance frontier benchmark — full v1 scope (3 independent frontiers).

Generalizes the validated pilot (run_frontier_pilot_bemtl97.py) to all three
home-turf sweep datasets per docs/analyses/insurance_frontier_benchmark_spec.md
(commit 037d215, branch feat/tabarena-benchmark), D5 Option A:

  bemtl97       target=claim,    drop=["nclaims","amount"]  (leak-fixed)
  coil2000      target=CARAVAN,  drop=[]
  uslapseagent  target=surrender, drop=[]

Each dataset is an INDEPENDENT frontier (separate table + plot, no cross-dataset
Pareto comparison).

  D1 (Option B): re-run LogisticGLM/LR, TweedieGLM, PoissonGLM + RandomForest on the
                 home-turf sweep's splits (StratifiedKFold(5, shuffle=True, random_state=42)),
                 log loss, 5 folds. Tweedie/Poisson regressors on the binary target:
                 predicted mean treated as P(y=1), clipped to [1e-6, 1-1e-6].
  D2 (Option A): re-fit LightGBM/XGBoost/CatBoost at recorded defaults purely to count
                 parameters: n_estimators x average leaves per tree (fold-0 train only;
                 tree structure is seed-insensitive for counting). RF leaves counted from
                 the fitted fold-0 RF. GLM/LR params = post-encoding column count + 1
                 (harness uses cat.codes -> raw column count). TabPFN = constant
                 10,000,000 (settled non-decision, spec §5 — do not research).
  D3 (Option B): Pareto frontier under the beyond-SE rule — model A is dominated iff some B
                 with strictly fewer params has mean_B + SE_B < mean_A - SE_A.
                 Models within SE of each other both stay on the frontier.

TabPFN/CAT/LGBM/XGB log-loss values are REUSED from home_turf_sweep_results.csv
(identical splits — no re-running), filtered to dataset + full size
(n_rows == len(X), n_estimators.isna()): 5 fold rows per method.

Usage:
    source /tmp/tabarena/.venv-ta/bin/activate
    python scripts/eval/insurance_benchmark_v1/run_frontier_benchmark.py

Outputs (same dir as this script), per dataset:
    frontier_results_<dataset>.csv   method | mean log loss | SE | n_params | on-frontier
    frontier_plot_<dataset>.png      x = log10(n_params), y = mean log loss, +/- SE bars,
                                     frontier red / dominated grey

Self-check: per-dataset assert-based sanity checks (5 fold rows per reused method,
no NaNs, unique methods, >=1 frontier point, no train/test overlap).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DATA_RAW = REPO / "data" / "raw"

# D5 Option A: the three home-turf sweep datasets. Load configs identical to
# scripts/run_home_turf_size_sweep.py (target + drops).
DATASETS = [
    dict(name="bemtl97", file="bemtl97.csv", target="claim", drop=["nclaims", "amount"]),
    dict(name="coil2000", file="coil2000.csv", target="CARAVAN", drop=[]),
    dict(name="uslapseagent", file="uslapseagent.csv", target="surrender", drop=[]),
]
N_FOLDS = 5
SWEEP_CSV = HERE / "home_turf_sweep_results.csv"
REUSED_METHODS = ["cat", "lgbm", "xgb", "tabpfn"]  # log loss reused as-is from the sweep
FAST_METHODS = ["lr", "logisticglm", "tweedieglm", "poissonglm", "rf"]  # D1 Option B, new compute

# TabPFN parameter count — settled non-decision (spec §5): constant per dataset, orders of
# magnitude above the GBDTs, so its precise value never changes frontier membership.
# ponytail: fixed constant, exact hosted-model figure to be confirmed from PriorLabs/TabPFN
# docs later — do not research or block on it now.
TABPFN_N_PARAMS = 10_000_000

# One-hot style: the harness encodes with cat.codes, so post-encoding column count == raw
# column count; GLM params = that + 1 intercept (spec §5 counting-rule table).


def load_Xy(ds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Identical to scripts/run_home_turf_size_sweep.py load_Xy — same split input."""
    df = pd.read_csv(DATA_RAW / ds["file"])
    df = df.dropna(subset=[ds["target"]]).reset_index(drop=True)
    y = df[ds["target"]].to_numpy(dtype=np.int64)
    X = df.drop(columns=[ds["target"]] + ds["drop"])
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    X = X.fillna(0.0).astype(np.float32)
    return X.to_numpy(dtype=np.float32), y


# ---------------------------------------------------------------------------
# D1 models — hyperparameters match the v1 harness (run_tabarena_insurance_benchmark.py)
# ---------------------------------------------------------------------------
def make_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(random_state=0, max_iter=500)


def make_logisticglm():
    from sklearn.linear_model import LogisticRegression
    # v1 harness LogisticGlmModel: penalty=None, lbfgs, max_iter=500, random_state=0
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=500, random_state=0)


def make_tweedieglm():
    from sklearn.linear_model import TweedieRegressor
    # v1 harness TweedieGlmModel: power=1.5, alpha=0.0, max_iter=500
    return TweedieRegressor(power=1.5, alpha=0.0, max_iter=500)


def make_poissonglm():
    from sklearn.linear_model import PoissonRegressor
    # v1 harness PoissonGlmModel: alpha=0.0, max_iter=500
    return PoissonRegressor(alpha=0.0, max_iter=500)


def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=0)


# ---------------------------------------------------------------------------
# D2 models — refit at sweep defaults purely to count parameters (fold-0 train only;
# tree structure is seed-insensitive for counting, spec §5 D2)
# ---------------------------------------------------------------------------
def make_cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=0, verbose=0)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=0)


def make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=0, device_type="cpu", verbose=-1)


# ---------------------------------------------------------------------------
# Parameter counting (spec §5 rule table)
# ---------------------------------------------------------------------------
def count_lgbm_leaves(model) -> int:
    """n_estimators x mean leaves per tree, from the fitted booster."""
    trees = model.booster_.dump_model()["tree_info"]
    leaves = [t["tree_structure"] for t in trees]
    counts = []

    def walk(node):
        if "leaf_index" in node:
            return 1
        return walk(node["left_child"]) + walk(node["right_child"])

    for t in leaves:
        counts.append(walk(t))
    return int(round(len(counts) * float(np.mean(counts))))


def count_xgb_leaves(model) -> int:
    """n_estimators x mean leaves per tree via trees_to_dataframe."""
    df = model.get_booster().trees_to_dataframe()
    per_tree = df[df["Feature"] == "Leaf"].groupby("Tree").size()
    return int(round(len(per_tree) * float(per_tree.mean())))


def count_cat_leaves(model) -> int:
    """n_estimators x mean leaves per tree via get_tree_leaf_counts (per-tree leaf counts)."""
    counts = model.get_tree_leaf_counts()  # (n_trees,) — leaf count per tree
    return int(round(len(counts) * float(counts.mean())))


def count_rf_leaves(model) -> int:
    """RandomForest counted under the same GBDT leaf rule: n_estimators x mean leaves."""
    leaves = [int(t.tree_.n_leaves) for t in model.estimators_]
    return int(round(len(leaves) * float(np.mean(leaves))))


# ---------------------------------------------------------------------------
# Frontier rule (spec D3: Option B)
# ---------------------------------------------------------------------------
def pareto_frontier(rows: list[dict]) -> list[str]:
    """Dominated iff some other method is strictly more parsimonious AND better on power
    beyond SE: mean_B + SE_B < mean_A - SE_A (lower log loss is better)."""
    frontier = []
    for a in rows:
        dominated = False
        for b in rows:
            if b is a:
                continue
            if b["n_params"] < a["n_params"] and b["mean"] + b["se"] < a["mean"] - a["se"]:
                dominated = True
                break
        if not dominated:
            frontier.append(a["method"])
    return frontier


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def run_dataset(ds: dict, out_csv: Path, out_png: Path) -> pd.DataFrame:
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    X, y = load_Xy(ds)
    n_cols = X.shape[1]
    say(f"dataset={ds['name']} shape={X.shape} pos_rate={y.mean():.4f}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    # ---- 1. Reused power: sweep CSV rows for <dataset>@full, default config only ----
    sweep = pd.read_csv(SWEEP_CSV)
    sweep_full = sweep[(sweep.dataset == ds["name"]) & (sweep.n_rows == len(X)) & (sweep.n_estimators.isna())]
    rows: list[dict] = []
    for m in REUSED_METHODS:
        r = sweep_full[sweep_full.method == m].sort_values("fold")
        assert len(r) == N_FOLDS, f"sweep missing fold rows for {m}"
        ll = r["log_loss"].to_numpy(dtype=float)
        rows.append({"method": m, "mean": ll.mean(), "se": ll.std(ddof=1) / np.sqrt(len(ll))})

    # ---- 2. D1: fit GLMs/LR/RF on the same 5 folds (new compute) ----
    d1_makers = {
        "lr": make_lr,
        "logisticglm": make_logisticglm,
        "tweedieglm": make_tweedieglm,
        "poissonglm": make_poissonglm,
        "rf": make_rf,
    }
    for m in FAST_METHODS:
        maker = d1_makers[m]
        fold_ll = []
        t1 = time.time()
        for fold, (tr, te) in enumerate(folds):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            model = maker()
            model.fit(Xtr, ytr)
            if m in ("tweedieglm", "poissonglm"):
                # regressor on a binary target: predicted mean treated as P(y=1), clipped
                mu = np.clip(model.predict(Xte), 1e-6, 1 - 1e-6)
                pp = np.column_stack([1 - mu, mu])
            else:
                pp = model.predict_proba(Xte)
            fold_ll.append(log_loss(yte, pp))
            say(f"  {m} f{fold} ll={fold_ll[-1]:.4f} ({time.time() - t1:.0f}s)")
        ll = np.array(fold_ll)
        rows.append({"method": m, "mean": ll.mean(), "se": ll.std(ddof=1) / np.sqrt(len(ll))})

    # ---- 3. D2: refit GBDTs + RF on fold-0 train purely to count leaves ----
    # (tree structure is seed-insensitive for counting, spec §5 D2 — one fold suffices)
    tr0, _ = folds[0]
    d2_counts: dict[str, int] = {}
    for m, maker in {"cat": make_cat, "xgb": make_xgb, "lgbm": make_lgbm}.items():
        t1 = time.time()
        model = maker()
        model.fit(X[tr0], y[tr0])
        counter = {"cat": count_cat_leaves, "xgb": count_xgb_leaves, "lgbm": count_lgbm_leaves}[m]
        d2_counts[m] = counter(model)
        say(f"  count {m}: {d2_counts[m]} params ({time.time() - t1:.0f}s)")
    t1 = time.time()
    rf = make_rf()
    rf.fit(X[tr0], y[tr0])
    d2_counts["rf"] = count_rf_leaves(rf)
    say(f"  count rf: {d2_counts['rf']} params ({time.time() - t1:.0f}s)")

    # ---- 4. n_params for every method ----
    glm_n = n_cols + 1  # post-encoding column count + 1 intercept (harness uses cat.codes)
    for r in rows:
        m = r["method"]
        if m in ("lr", "logisticglm", "tweedieglm", "poissonglm"):
            r["n_params"] = glm_n
        elif m == "tabpfn":
            r["n_params"] = TABPFN_N_PARAMS
        else:
            r["n_params"] = d2_counts[m]
    for r in rows:
        assert isinstance(r["n_params"], int) and r["n_params"] > 0

    on_frontier = set(pareto_frontier(rows))
    for r in rows:
        r["on_frontier"] = "yes" if r["method"] in on_frontier else "no"
    table = pd.DataFrame(rows)[["method", "mean", "se", "n_params", "on_frontier"]].sort_values(
        "mean", ignore_index=True
    )
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 6. Plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in table.iterrows():
        color = "#d62728" if r["on_frontier"] == "yes" else "#7f7f7f"
        ax.errorbar(np.log10(r["n_params"]), r["mean"], yerr=r["se"], fmt="o",
                    color=color, capsize=4, markersize=7)
        ax.annotate(r["method"], (np.log10(r["n_params"]), r["mean"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)
    ax.set_xlabel("log10(n_params)")
    ax.set_ylabel("mean log loss")
    ax.set_title(f"Insurance frontier — {ds['name']} (full, 5 folds, mean ± SE)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- 7. Self-check (repo assert convention) ----
    sanity_check(sweep_full, folds, table, on_frontier)
    return table


def sanity_check(sweep_full: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], table: pd.DataFrame, on_frontier: set) -> None:
    """Assert-based sanity checks, per repo convention (run at end of main)."""
    # (a) sweep has exactly 5 non-NaN fold rows per reused method; split is a valid
    # 5-fold partition of the full index set with no train/test overlap within a fold.
    n = sweep_full[sweep_full.method == "lgbm"]["fold"].nunique()
    assert n == N_FOLDS, f"expected {N_FOLDS} sweep folds, got {n}"
    assert sweep_full["log_loss"].notna().all(), "sweep reused log-loss has NaNs"
    n_total = sum(len(te) for _, te in folds)
    assert len(np.unique(np.concatenate([te for _, te in folds]))) == n_total, "test folds overlap"
    for tr, te in folds:
        assert not np.isin(te, tr).any(), "train/test overlap within fold"
    # (b) exactly one n_params per method, no NaNs in emitted table
    assert table["n_params"].notna().all() and table["mean"].notna().all() and table["se"].notna().all(), "NaN in table"
    assert table["method"].nunique() == len(table), "duplicate methods in table"
    # (c) frontier rule returns at least one on-frontier model
    assert len(on_frontier) >= 1, "frontier is empty"
    print(f"SELF-CHECK OK: folds={N_FOLDS}, methods={len(table)}, on-frontier={len(on_frontier)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    for ds in DATASETS:
        name = ds["name"]
        out_csv = HERE / f"frontier_results_{name}.csv"
        out_png = HERE / f"frontier_plot_{name}.png"
        print(f"\n{'=' * 70}\nDATASET {name}\n{'=' * 70}", flush=True)
        run_dataset(ds, out_csv, out_png)
        print(f"[{time.strftime('%H:%M:%S')}] {name} done -> {out_csv}, {out_png}", flush=True)
    print(f"\nALL DATASETS DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
