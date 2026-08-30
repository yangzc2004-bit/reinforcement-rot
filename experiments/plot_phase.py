"""Regenerate the L2 phase-diagram figures from the decoded CSVs.

Usage:
    python results/decode.py        # first, restores the CSVs
    python experiments/plot_phase.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

phase = pd.read_csv(RES / "l2_phase_results.csv")
verify = pd.read_csv(RES / "l2_verify_results.csv")

# --- Figure 1: mill probability over the (lambda, delta) grid ---------------
pv = (
    phase.assign(mill=(phase.category == "mill").astype(float))
    .pivot_table(index="lam", columns="delta", values="mill", aggfunc="mean")
    .sort_index(ascending=True)
)
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pv.values, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(pv.columns)))
ax.set_xticklabels(pv.columns)
ax.set_yticks(range(len(pv.index)))
ax.set_yticklabels(pv.index)
ax.set_xlabel("detour length delta")
ax.set_ylabel("evaporation lambda")
ax.set_title("L2 mill probability over (lambda, delta)")
fig.colorbar(im, label="P(mill)")
fig.tight_layout()
fig.savefig(RES / "L2_phase_diagram.png", dpi=200)
plt.close(fig)

# --- Figure 2: robustness verification (mill fraction by task) --------------
summary = (
    verify.assign(mill=(verify.category == "mill").astype(float))
    .groupby("task")["mill"]
    .agg(["mean", "count"])
)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(summary.index.astype(str), summary["mean"])
ax.set_xlabel("verification task")
ax.set_ylabel("P(mill)")
ax.set_title("L2 robustness verification: mill fraction by task")
for i, (m, n) in enumerate(zip(summary["mean"], summary["count"])):
    ax.text(i, m, f"n={n}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(RES / "L2_verification.png", dpi=200)
plt.close(fig)

print("wrote", RES / "L2_phase_diagram.png")
print("wrote", RES / "L2_verification.png")
