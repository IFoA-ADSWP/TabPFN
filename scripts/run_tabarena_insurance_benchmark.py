"""TabArena insurance benchmark — foundation vs tree-based vs statistical models.

Usage:
    # 1. Clone and install TabArena (one-time)
    git clone https://github.com/autogluon/tabarena.git /tmp/tabarena
    uv venv --seed --python 3.12 .venv-ta
    source .venv-ta/bin/activate
    uv pip install --prerelease=allow -e "/tmp/tabarena/packages/tabarena[benchmark,tabfm]"

    # 2. Run
    python scripts/run_tabarena_insurance_benchmark.py

What it does:
    Wraps 4 classification + 3 regression insurance datasets from data/raw/
    as TabArena UserTasks, runs 10 models across 3 families, and outputs
    a leaderboard table comparing them.

    Two benchmark tiers:
        Full datasets  (7) → TabPFN (hosted API), tree-based, statistical
        Small subsets (1K) → + TabFM (PyTorch on CPU)

    Datasets:   eudirectlapse, coil2000, ausprivauto0405, freMTPL2freq_binary  (class)
                freMTPL2freq, eudirectlapse-premium, ausprivauto0405-vehvalue  (reg)

    Models:
        Foundational  -> TabPFN (via tabpfn-client hosted API)   ← no GPU needed
                          TabFM (via PyTorch, 1K-row subsets)    ← CPU-feasible at this size
        Tree-based    -> LightGBM, XGBoost, CatBoost, RandomForest
        Statistical   -> LogisticRegression, Poisson GLM, Tweedie GLM, LinearRegression
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel
from sklearn.model_selection import StratifiedKFold, KFold

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.metadata import (
    TabArenaTaskMetadata,
    TaskMetadataCollection,
)
from tabarena.benchmark.task.user_task import from_sklearn_splits_to_user_task_splits
from tabarena.contexts import AbstractArenaContext

if TYPE_CHECKING:
    from tabarena.utils.config_utils import ConfigGenerator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_RAW = REPO / "data" / "raw"

RUN_NAME = "insurance_benchmark_v1"
RESULTS_DIR = str(HERE / "experiments" / RUN_NAME)
EVAL_DIR = HERE / "eval" / RUN_NAME
TASK_CACHE_DIR = HERE / "task_cache" / RUN_NAME

# ---------------------------------------------------------------------------
# Dataset definitions — full
# ---------------------------------------------------------------------------
CLASSIFICATION_DATASETS = {
    "eudirectlapse": {
        "file": "eudirectlapse.csv",
        "target": "lapse",
        "n_splits": 5,
        "desc": "EU direct insurance lapse prediction, 23K rows, 12.8% positive",
    },
    "coil2000": {
        "file": "coil2000.csv",
        "target": "CARAVAN",
        "n_splits": 5,
        "desc": "CoIL 2000 caravan insurance, 9.8K rows, 6.0% positive",
    },
    "ausprivauto0405": {
        "file": "ausprivauto0405.csv",
        "target": "ClaimOcc",
        "n_splits": 5,
        "desc": "Australian vehicle insurance claim occurrence, 68K rows, 6.8% positive",
    },
    "freMTPL2freq_binary": {
        "file": "freMTPL2freq_binary.csv",
        "target": "ClaimIndicator",
        "n_splits": 5,
        "desc": "French motor TPL (binarised), 50K rows, 5.0% positive",
    },
}

REGRESSION_DATASETS = {
    "freMTPL2freq": {
        "file": "freMTPL2freq.csv",
        "target": "ClaimNb",
        "n_splits": 3,
        "desc": "Claim frequency regression, 678K rows, Poisson target",
    },
    "eudirectlapse_premium": {
        "file": "eudirectlapse.csv",
        "target": "prem_pure",
        "n_splits": 3,
        "desc": "Pure premium regression, 23K rows, continuous target",
    },
    "ausprivauto0405_vehvalue": {
        "file": "ausprivauto0405.csv",
        "target": "VehValue",
        "n_splits": 3,
        "desc": "Vehicle value regression, 68K rows, continuous target",
    },
}

# ---------------------------------------------------------------------------
# Small subsets for GPU-bound models like TabFM (1K rows, CPU-feasible)
# ---------------------------------------------------------------------------
TABFM_SMALL_DATASETS = {
    "eudirectlapse_1k": {
        "file": "eudirectlapse.csv",
        "target": "lapse",
        "n_splits": 3,
        "n_rows": 1000,
        "desc": "EU lapse (1K subset for TabFM)",
    },
    "coil2000_1k": {
        "file": "coil2000.csv",
        "target": "CARAVAN",
        "n_splits": 3,
        "n_rows": 1000,
        "desc": "CoIL 2000 (1K subset for TabFM)",
    },
    "freMTPL2freq_binary_1k": {
        "file": "freMTPL2freq_binary.csv",
        "target": "ClaimIndicator",
        "n_splits": 3,
        "n_rows": 1000,
        "desc": "French motor TPL (1K subset for TabFM)",
    },
}


def _load_raw(info: dict) -> pd.DataFrame:
    path = DATA_RAW / info["file"]
    df = pd.read_csv(path)
    target = info["target"]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {path}")
    # Convert object columns to categorical (TabArena metadata requires non-object dtypes)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")
    return df.dropna(subset=[target])


def _sample_balanced(df: pd.DataFrame, target: str, n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Stratified sample down to n_rows preserving class balance."""
    from sklearn.model_selection import train_test_split
    df_out, _ = train_test_split(
        df, train_size=n_rows, stratify=df[target], random_state=seed
    )
    return df_out


def make_classification_task(
    name: str, info: dict, task_cache_dir: Path
) -> tuple[UserTask, TabArenaTaskMetadata]:
    """Build a k-fold classification UserTask from a raw CSV."""
    dataset = _load_raw(info)
    target = info["target"]
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(
            n_splits=info["n_splits"], shuffle=True, random_state=42
        ).split(dataset.drop(columns=[target]), dataset[target]),
        n_splits=info["n_splits"],
    )
    task = UserTask(task_name=name, task_cache_path=task_cache_dir)
    wrapper = task.create_task(
        dataset=dataset,
        target_feature=target,
        problem_type="classification",
        splits=splits,
    )
    task.save_task(wrapper)
    return task, wrapper.metadata


def make_regression_task(
    name: str, info: dict, task_cache_dir: Path
) -> tuple[UserTask, TabArenaTaskMetadata]:
    """Build a k-fold regression UserTask from a raw CSV."""
    dataset = _load_raw(info)
    target = info["target"]
    splits = {}
    for fold, (train_idx, test_idx) in enumerate(
        KFold(n_splits=info["n_splits"], shuffle=True, random_state=42).split(dataset)
    ):
        splits[fold] = {0: (train_idx.tolist(), test_idx.tolist())}
    task = UserTask(task_name=name, task_cache_path=task_cache_dir)
    wrapper = task.create_task(
        dataset=dataset,
        target_feature=target,
        problem_type="regression",
        splits=splits,
    )
    task.save_task(wrapper)
    return task, wrapper.metadata


def make_tabfm_task(
    name: str, info: dict, task_cache_dir: Path
) -> tuple[UserTask, TabArenaTaskMetadata]:
    """Build a 3-fold classification UserTask from a 1K stratified subset."""
    dataset = _load_raw(info)
    target = info["target"]
    dataset = _sample_balanced(dataset, target, n_rows=info["n_rows"])
    splits = from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(
            n_splits=info["n_splits"], shuffle=True, random_state=42
        ).split(dataset.drop(columns=[target]), dataset[target]),
        n_splits=info["n_splits"],
    )
    task = UserTask(task_name=name, task_cache_path=task_cache_dir)
    wrapper = task.create_task(
        dataset=dataset,
        target_feature=target,
        problem_type="classification",
        splits=splits,
    )
    task.save_task(wrapper)
    return task, wrapper.metadata


# ---------------------------------------------------------------------------
# Custom model: TabPFN via hosted API  (no GPU required)
# ---------------------------------------------------------------------------

class TabPFNClientModel(AbstractModel):
    """TabPFN via tabpfn-client hosted API.

    Uses Prior Labs' cloud inference — no local GPU needed. Requires
    ``TABPFN_API_KEY`` env var or interactive login via ``tabpfn_client.init()``.
    """

    ag_key = "TabPFNClient"
    ag_name = "TabPFNClient"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        Xp = self.preprocess(X, y=y, is_train=True)
        if self.problem_type == "regression":
            from tabpfn_client import TabPFNRegressor
            self.model = TabPFNRegressor(random_state=0)
        else:
            from tabpfn_client import TabPFNClassifier
            self.model = TabPFNClassifier(random_state=0)
        self.model.fit(Xp, y)

    def _preprocess(self, X, is_train=False, **kwargs):
        X = super()._preprocess(X, **kwargs)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _set_default_params(self) -> None:
        pass

    def _get_default_auxiliary_params(self) -> dict:
        default = super()._get_default_auxiliary_params()
        default.update({"valid_raw_types": ["int", "float"]})
        return default

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.utils.config_utils import ConfigGenerator
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{}])


# ---------------------------------------------------------------------------
# Custom GLM models
# ---------------------------------------------------------------------------

class PoissonGlmModel(AbstractModel):
    """GLM with Poisson family — for claim count regression."""
    ag_key = "PoissonGLM"
    ag_name = "PoissonGLM"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from sklearn.linear_model import PoissonRegressor
        Xp = self.preprocess(X, y=y, is_train=True)
        self.model = PoissonRegressor(alpha=0.0, max_iter=500)
        self.model.fit(Xp, y)

    def _preprocess(self, X, is_train=False, **kwargs):
        X = super()._preprocess(X, **kwargs)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.utils.config_utils import ConfigGenerator
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{}])


class TweedieGlmModel(AbstractModel):
    """GLM with Tweedie family (p=1.5) — for pure premium regression."""
    ag_key = "TweedieGLM"
    ag_name = "TweedieGLM"

    def __init__(self, tweedie_p: float = 1.5, **kwargs):
        self.tweedie_p = tweedie_p
        super().__init__(**kwargs)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from sklearn.linear_model import TweedieRegressor
        Xp = self.preprocess(X, y=y, is_train=True)
        self.model = TweedieRegressor(power=self.tweedie_p, alpha=0.0, max_iter=500)
        self.model.fit(Xp, y)

    def _preprocess(self, X, is_train=False, **kwargs):
        X = super()._preprocess(X, **kwargs)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["regression"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.utils.config_utils import ConfigGenerator
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{"tweedie_p": 1.5}])


class LogisticGlmModel(AbstractModel):
    """GLM with Binomial family (logistic) — for binary classification."""
    ag_key = "LogisticGLM"
    ag_name = "LogisticGLM"

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from sklearn.linear_model import LogisticRegression
        Xp = self.preprocess(X, y=y, is_train=True)
        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500, random_state=0)
        self.model.fit(Xp, y)

    def _preprocess(self, X, is_train=False, **kwargs):
        X = super()._preprocess(X, **kwargs)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _set_default_params(self) -> None:
        pass

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary"]

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.utils.config_utils import ConfigGenerator
        return ConfigGenerator(search_space={}, model_cls=cls, manual_configs=[{}])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build full-size datasets ----
    print("Building classification tasks...")
    full_tasks, full_meta = [], []
    for name, info in CLASSIFICATION_DATASETS.items():
        task, meta = make_classification_task(name, info, TASK_CACHE_DIR)
        full_tasks.append(task)
        full_meta.append(meta)
        print(f"  ✓ {name} ({info['desc']})")

    print("Building regression tasks...")
    for name, info in REGRESSION_DATASETS.items():
        task, meta = make_regression_task(name, info, TASK_CACHE_DIR)
        full_tasks.append(task)
        full_meta.append(meta)
        print(f"  ✓ {name} ({info['desc']})")

    # ---- 2. Build 1K-row subsets for TabFM ----
    print("Building 1K-row subsets for TabFM...")
    tabfm_tasks, tabfm_meta = [], []
    for name, info in TABFM_SMALL_DATASETS.items():
        task, meta = make_tabfm_task(name, info, TASK_CACHE_DIR)
        tabfm_tasks.append(task)
        tabfm_meta.append(meta)
        print(f"  ✓ {name} ({info['desc']})")

    all_tasks = full_tasks + tabfm_tasks
    all_meta = full_meta + tabfm_meta
    task_collection = TaskMetadataCollection.from_source(all_meta)

    for task, meta in zip(all_tasks, all_meta):
        task.with_task_metadata(meta).load().validate_metadata()

    # ---- 3. Define models ----
    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            # --- Foundation (hosted API, no GPU) ---
            (TabPFNClientModel.config_generator(), 0),
            # --- Foundation (PyTorch, small subsets only) ---
            ("TabFM", 0),
            # --- Tree-based (registry, all CPU) ---
            ("LightGBM", 0, {"device_type": "cpu"}),
            ("XGBoost", 0),
            ("CatBoost", 0),
            ("RandomForest", 0),
            # --- Statistical (custom GLMs + registry LR, all CPU) ---
            (LogisticGlmModel.config_generator(), 0),
            (PoissonGlmModel.config_generator(), 0),
            (TweedieGlmModel.config_generator(), 0),
            ("Linear", 0),
        ],
        # ponytail: holdout = no bagging (1 fit per model per split). The default
        # 8-bag folds would be hundreds of fits across 7 datasets on an 8-core M1;
        # point estimates for this first full pass, re-enable bagging on GPU.
        holdout_experiments=True,
    ).build_experiments(num_gpus=0)

    # ---- 4. Run benchmark ----
    context = AbstractArenaContext(task_metadata=task_collection, methods=[])

    print(f"\nRunning {len(experiments)} experiments across {len(all_tasks)} datasets...")
    context.build_and_run_jobs(
        experiments,
        expname=RESULTS_DIR,
        user_tasks=all_tasks,
        new_result_prefix="[Insurance] ",
        debug_mode=True,
    )

    # ---- 5. Output leaderboard ----
    leaderboard = context.compare(output_dir=EVAL_DIR)
    print("\n=== Insurance Benchmark Leaderboard ===")
    print(leaderboard.to_markdown())
    print(f"\nResults saved to {EVAL_DIR}")
