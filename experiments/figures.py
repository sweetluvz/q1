"""Paper figures from results/summary.json and results/*/staf_analysis.json -> docs/figures/*.png"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

INTERNAL, EXTERNAL, NEUTRAL = "#2a78d6", "#eb6834", "#8a8984"
POS, NEG = "#2a78d6", "#e34948"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["lr", "lgbm", "lgbm_noproc", "gru", "staf_h", "staf", "staf_nomeas", "staf_nostale", "staf_notemp"]
LABEL = {"lr": "LR", "lgbm": "LightGBM", "lgbm_noproc": "LightGBM − meas. features", "gru": "GRU", "staf": "STAF", "staf_h": "STAF+GRU",
         "staf_nomeas": "STAF − meas. terms", "staf_nostale": "STAF − staleness", "staf_notemp": "STAF − temporal"}
OUT = "docs/figures"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
                     "savefig.bbox": "tight", "figure.constrained_layout.use": True})


def dumbbell(S, metric, fname, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        models = [m for m in ORDER if m in S[d]["models"]]
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models):
            a = S[d]["models"][m]["src_test"][metric][0]
            b = S[d]["models"][m]["tgt_test"][metric][0]
            ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
            ax.scatter([a], [yi], s=40, color=INTERNAL, zorder=2, edgecolor="#fcfcfb", lw=1.5,
                       label="Internal (source test)" if yi == y[0] else None)
            ax.scatter([b], [yi], s=40, color=EXTERNAL, zorder=2, edgecolor="#fcfcfb", lw=1.5,
                       label="External (other hospital)" if yi == y[0] else None)
        ax.set_yticks(y, [LABEL[m] for m in models])
        ax.set_title(f"Train {d[0]} → test {d[-1]}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def coverage(S, fname, model="staf"):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        M = S[d]["models"][model]
        fs = M["fewshot"]
        n = np.array(sorted(int(k) for k in fs))
        mean = np.array([fs[str(k)]["coverage_pos"][0] for k in n])
        lo = np.array([fs[str(k)]["coverage_pos"][1] for k in n])
        hi = np.array([fs[str(k)]["coverage_pos"][2] for k in n])
        ax.fill_between(n, lo, hi, color=EXTERNAL, alpha=0.18, lw=0)
        ax.plot(n, mean, color=EXTERNAL, lw=2, marker="o", ms=5, label="Calibrated on n target patients (10–90%)")
        ax.axhline(M["conformal"]["standard"]["coverage_pos"], color=NEUTRAL, lw=1.5, ls="--",
                   label="Calibrated on source hospital")
        ax.axhline(M["conformal"]["in_domain"]["coverage_pos"], color=INTERNAL, lw=1.5, ls=":",
                   label="In-domain reference")
        ax.axhline(0.9, color=INK2, lw=0.8)
        ax.text(n[0], 0.893, "nominal 90%", ha="left", va="top", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(n, [str(k) for k in n])
        ax.set_xlabel("Labeled target patients for calibration")
        ax.set_title(f"Train {d[0]} → deploy {d[-1]}", color=INK, fontsize=9, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("Coverage of septic hours")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, fontsize=8)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def shift_attribution(fname, k=12):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        path = os.path.join("results", d, "staf_analysis.json")
        if not os.path.exists(path):
            continue
        sd = json.load(open(path))["seeds"]["0"]["shift_decomposition"]
        rows = sd["top_terms"][:k][::-1]
        c = [r["contribution"] for r in rows]
        ax.barh(range(len(rows)), c, color=[POS if v > 0 else NEG for v in c], height=0.7)
        ax.set_yticks(range(len(rows)), [r["term"] for r in rows], fontsize=7.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Contribution to Δ mean logit")
        ax.set_title(f"{d[0]} → {d[-1]}: total Δ = {sd['actual_mean_logit_change']:+.2f}", color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open("results/summary.json"))
    dumbbell(S, "auroc", "fig_auroc_internal_external.png", "AUROC (hourly)")
    dumbbell(S, "utility", "fig_utility_internal_external.png", "Normalized utility (Challenge metric)")
    coverage(S, "fig_conformal_fewshot.png")
    shift_attribution("fig_shift_attribution.png")


if __name__ == "__main__":
    main()
