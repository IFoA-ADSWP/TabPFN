"""Prepare CASdatasets .rda files as clean CSVs for the TabArena insurance benchmark.

Reads .rda files (default /tmp/opencode/datasets/) and writes cleaned CSVs to
data/raw/. Categorical columns are written as strings; binary targets are int.

Usage:
    python scripts/prepare_insurance_datasets.py [SRC_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "data" / "raw"
SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/opencode/datasets")


def _to_str(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = df[c].astype(str)


def make_uslapseagent(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["duration", "acc.death.rider", "gender", "premium.frequency", "risk.state",
            "underwriting.age", "living.place", "annual.premium", "DJIA",
            "surrender", "allcause"]
    cats = ["acc.death.rider", "gender", "premium.frequency", "risk.state",
            "underwriting.age", "living.place"]
    df = df[keep].copy()
    df["surrender"] = df["surrender"].astype(int)
    df["allcause"] = df["allcause"].astype(int)
    _to_str(df, cats)
    return df, ["surrender", "allcause", "duration"]


def make_bemtl97(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["expo", "coverage", "ageph", "sex", "bm", "power", "agec", "fuel",
            "use", "fleet", "claim", "nclaims", "amount"]
    cats = ["coverage", "sex", "fuel", "use"]
    df = df[keep].copy()
    df["claim"] = df["claim"].astype(int)
    df["fleet"] = df["fleet"].astype(int)
    df["amount"] = np.log1p(df["amount"].fillna(0.0))
    _to_str(df, cats)
    return df, ["claim", "nclaims", "amount"]


def make_bemtl16(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["policy_year", "exposure", "insured_birth_year", "vehicle_age",
            "policy_holder_age", "driver_license_age", "vehicle_brand",
            "vehicle_model", "mileage", "vehicle_power", "catalog_value",
            "claim_responsibility_rate", "driving_training_label",
            "number_of_liability_claims"]
    cats = ["vehicle_brand", "vehicle_model", "driving_training_label"]
    # Panel data: one row per policy-year; keep the latest year per contract.
    df = (df.sort_values("policy_year")
            .groupby("insurance_contract", as_index=False).tail(1))[keep]
    df["number_of_liability_claims"] = df["number_of_liability_claims"].astype(int)
    _to_str(df, cats)
    return df, ["number_of_liability_claims"]


def make_ausauto(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["AccMth", "ReportMth", "FinMth", "OpTime", "InjType1", "InjType2",
            "InjType3", "InjType4", "InjType5", "InjNb", "Legal", "AggClaim"]
    cats = ["InjType1", "InjType2", "InjType3", "InjType4", "InjType5", "Legal"]
    df = df[keep].copy()
    df["AggClaim"] = np.log(df["AggClaim"])
    _to_str(df, cats)
    return df, ["AggClaim"]


def make_norauto(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = ["Male", "Young", "DistLimit", "GeoRegion", "Expo", "NbClaim"]
    cats = ["DistLimit", "GeoRegion"]
    df = df[keep].copy()
    # Binarise claim count -> has-claim (4.6% positive)
    df["NbClaim"] = (df["NbClaim"] > 0).astype(int)
    _to_str(df, cats)
    return df, ["NbClaim"]


DATASETS = [
    ("uslapseagent", "uslapseagent.csv", make_uslapseagent),
    ("beMTPL97", "bemtl97.csv", make_bemtl97),
    ("beMTPL16", "bemtl16.csv", make_bemtl16),
    ("ausautoBI8999", "ausautoBI8999.csv", make_ausauto),
    ("norauto", "norauto.csv", make_norauto),
]


def _print_stats(df: pd.DataFrame, targets: list[str]) -> None:
    for t in targets:
        s = df[t]
        n = len(s)
        if s.dtype.kind in "iu" and s.nunique() <= 2:
            pos = int((s == 1).sum())
            print(f"      target {t}: binary, positive rate {pos}/{n} = {pos / n:.1%}")
        else:
            print(f"      target {t}: numeric, min {s.min():.4g} mean {s.mean():.4g} max {s.max():.4g}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, out_name, maker in DATASETS:
        src = SRC_DIR / f"{stem}.rda"
        df_raw = list(pyreadr.read_r(src).values())[0]
        df, targets = maker(df_raw)
        out = OUT_DIR / out_name
        df.to_csv(out, index=False)
        print(f"{out.name}: {df.shape[0]} rows x {df.shape[1]} cols")
        _print_stats(df, targets)


if __name__ == "__main__":
    main()
