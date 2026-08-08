"""Generate the charts embedded in docs/LEARNING-PATH-JUNIOR.md.

Real-data charts read outputs/current/tables/*.csv (source of truth);
the reliability chart is a labeled schematic for teaching.

Run from repo root: .venv312/bin/python scripts/infra/make_learning_path_charts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

TABLES = Path("outputs/current/tables")
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Discrimination: Table1 ROC/PR AUC per model
t1 = pd.read_csv(TABLES / "Table1_Model_Performance.csv")
x = np.arange(len(t1)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, t1["roc_auc"], w, label="ROC AUC", color="#4C72B0")
ax.bar(x + w/2, t1["pr_auc"], w, label="PR AUC", color="#DD8452")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="ROC chance (0.5)")
ax.axhline(0.128, color="gray", ls=":", lw=1, label="PR chance = prevalence (0.128)")
ax.set_xticks(x); ax.set_xticklabels(t1["model"], rotation=15)
ax.set_ylim(0, 0.7); ax.set_ylabel("score")
ax.set_title("Table1: discrimination on eudirectlapse (real results)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lp_discrimination.png", dpi=150); plt.close(fig)

# 2. Calibration: Table4 Brier vs constant-predictor floor
t4 = pd.read_csv(TABLES / "Table4_Brier_Scores.csv")
floor = 0.1281 * (1 - 0.1281)  # pi*(1-pi): Brier of predicting the base rate for everyone
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(t4["Method"], t4["Brier Score"], color="#55A868", width=0.5)
ax.axhline(floor, color="crimson", ls="--", lw=1.5,
           label=f"constant-predictor floor = {floor:.4f}")
ax.set_ylabel("Brier (lower is better)")
ax.set_title("Table4: Brier on eudirectlapse (real results)")
ax.set_ylim(0, 0.14); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lp_brier.png", dpi=150); plt.close(fig)

# 3. Class balance: Table3 full vs stratified 10K subset
t3 = pd.read_csv(TABLES / "Table3_Class_Balance.csv")
x = np.arange(2); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w/2, t3["Full Dataset %"], w, label="Full dataset", color="#4C72B0")
ax.bar(x + w/2, t3["Subset (10K) %"], w, label="10K subset", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(["no lapse (0)", "lapse (1)"])
ax.set_ylabel("% of rows")
ax.set_title("Table3: stratified subset holds the 12.8% balance (real results)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lp_class_balance.png", dpi=150); plt.close(fig)

# 4. Schematic: how to read a reliability diagram (teaching, not evidence)
p = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], color="green", lw=2, label="perfectly calibrated")
ax.plot(p, p**1.6, color="crimson", lw=2, label="overconfident (schematic)")
ax.plot(p, p**0.6, color="#4C72B0", lw=2, label="underconfident (schematic)")
ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
ax.set_title("Reliability diagram — how to read it")
ax.legend(fontsize=8); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT / "lp_reliability_schematic.png", dpi=150); plt.close(fig)

print("wrote:", sorted(p.name for p in OUT.glob("lp_*.png")))
