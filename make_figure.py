"""
Plot reproduced vs. reported test accuracy (run after run_experiments.py).

    python make_figure.py                          # uses results/summary_paper_random.csv
    python make_figure.py --summary results/summary_cv_random.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OURS, PAPER = "#eb6834", "#2a78d6"          # validated categorical pair (CVD-safe)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
NAMES = {"SVM": "SVM", "UELM": "Unified ELM", "SELM": "Sparse ELM"}

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default="results/summary_paper_random.csv")
ap.add_argument("--out", default=None)
args = ap.parse_args()
df = pd.read_csv(args.summary)
out = args.out or os.path.splitext(args.summary)[0] + ".png"

datasets = list(dict.fromkeys(df.dataset))
fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 5.4), sharey=True)
axes = [axes] if len(datasets) == 1 else axes
for ax, ds in zip(axes, datasets):
    d = df[df.dataset == ds].reset_index(drop=True)
    labels = [f"{NAMES[m]} · {k}" for m, k in zip(d.method, d.kind)]
    ys = list(range(len(d)))[::-1]
    ax.errorbar(d.test_acc, [y - 0.13 for y in ys], xerr=d.test_std, fmt="o", ms=8,
                color=OURS, ecolor=OURS, elinewidth=2, capsize=0, label="This reproduction (mean ± std, 20 trials)")
    ax.scatter(d.paper_test_acc, [y + 0.13 for y in ys], marker="D", s=56, color=PAPER,
               edgecolor="#fcfcfb", linewidth=2, zorder=3, label="Reported in paper (Tables IV–VII)")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.set_xlim(88, 101)
    ax.set_xlabel("Test accuracy (%)", color=MUTED)
    ax.set_title(ds.capitalize(), color=INK, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper center", ncol=2, frameon=False, fontsize=9.5,
           labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Sparse ELM (Bai et al., IEEE T-CYB 2014): reproduced vs. reported",
             color=INK, fontsize=12.5, y=1.05)
fig.patch.set_facecolor("#fcfcfb")
for ax in axes:
    ax.set_facecolor("#fcfcfb")
fig.tight_layout()
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved", out)
