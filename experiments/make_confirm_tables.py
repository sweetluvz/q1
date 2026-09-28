"""Manuscript tables (markdown) and figures from results_confirm/summary.json and fs_full_decomposition.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = "results_confirm"
FIG = "docs/figures"
NAME = {"lr": "Logistic regression", "lgbm": "LightGBM", "gru": "GRU", "fs": "Fuzzy scorecard"}
DIR = {"AtoB": "A→B", "BtoA": "B→A"}
FULL, PF, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
                     "savefig.bbox": "tight", "figure.constrained_layout.use": True})


def fmt_ci(c, d=3):
    return f"{c[0]:+.{d}f} ({c[1]:+.{d}f} to {c[2]:+.{d}f})"


def fmt_p(p):
    return "<0.01" if p < 0.01 else f"{p:.2f}"


def tables(S):
    out = []
    P = S["primary"]
    out.append("**Table 2. Discrimination on the target hospital's test patients and the primary estimand.** "
               "Local = model trained at the target hospital; transferred = model trained at the other hospital. "
               "DiD = (local − transferred)_FULL − (local − transferred)_PF; 95% CI from 1,000 patient-level bootstrap "
               "resamples; p two-sided bootstrap, Holm-adjusted across the 8 tests.\n")
    out.append("| Direction | Model | FULL local | FULL transferred | PF local | PF transferred | DiD (95% CI) | p | p (Holm) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for c in ("lr", "lgbm", "gru", "fs"):
            r = P[d][c]
            a = r["auroc"]
            out.append(f"| {DIR[d]} | {NAME[c]} | {a['full_loc']:.3f} | {a['full_ext']:.3f} | {a['pf_loc']:.3f} | {a['pf_ext']:.3f} | "
                       f"{fmt_ci([r['did_auroc']['point']] + r['did_auroc']['ci'][1:])} | {fmt_p(r['did_auroc']['p'])} | {fmt_p(r['did_auroc']['p_holm'])} |")
    n = {d: (P[d]["lr"]["n_patients"], P[d]["lr"]["n_septic"]) for d in P}
    out.append(f"\nTarget test sets: hospital B {n['AtoB'][0]} patients ({n['AtoB'][1]} septic); hospital A {n['BtoA'][0]} patients ({n['BtoA'][1]} septic).\n")

    out.append("**Table 3. Secondary estimands (exploratory, unadjusted).** ΔExt = FULL − PF for the transferred model; "
               "utility uses the threshold that maximizes utility on source validation.\n")
    out.append("| Direction | Model | ΔExt AUROC (95% CI) | ΔExt utility (95% CI) | DiD utility (95% CI) |")
    out.append("|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for c in ("lr", "lgbm", "gru", "fs"):
            r = P[d][c]
            out.append(f"| {DIR[d]} | {NAME[c]} | {fmt_ci(r['dext_auroc']['ci'])} | {fmt_ci(r['dext_utility']['ci'])} | {fmt_ci(r['did_utility']['ci'])} |")

    Sec = S["secondary"]
    out.append("\n**Table 4. Calibration and patient-level conformal prediction (α = 0.1) of transferred models.** "
               "Internal = source test set; external = target test set. Conformal coverage = probability that a uniformly drawn "
               "hour of a patient has its true label in the prediction set (septic hours / non-septic hours).\n")
    out.append("| Direction | Model | Variant | O/E int → ext | Slope int → ext | Coverage septic int → ext | Coverage non-septic ext | Ambiguous sets ext |")
    out.append("|---|---|---|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for c in ("lr", "lgbm", "gru", "fs"):
            for v in ("full", "pf"):
                r = Sec[d][f"{c}_{v}"]
                s, t = r["src_test"], r["tgt_test"]
                out.append(f"| {DIR[d]} | {NAME[c]} | {v.upper()} | {s['oe']:.2f} → {t['oe']:.2f} | {s['slope']:.2f} → {t['slope']:.2f} | "
                           f"{s['conformal']['coverage_pos']:.3f} → {t['conformal']['coverage_pos']:.3f} | {t['conformal']['coverage_neg']:.3f} | "
                           f"{t['conformal']['ambiguous_rate']:.2f} |")

    out.append("\n**Table 5. Label-free estimation of the target prevalence of positive hours.**\n")
    out.append("| Direction | Model | Variant | True | BBSE | EM |")
    out.append("|---|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for c in ("lr", "lgbm", "gru", "fs"):
            for v in ("full", "pf"):
                p = Sec[d][f"{c}_{v}"]["prevalence"]
                out.append(f"| {DIR[d]} | {NAME[c]} | {v.upper()} | {p['true_target']:.4f} | {p['bbse']:.4f} | {p['em']:.4f} |")

    out.append("\n**Table 6. Sensitivity analyses of the primary estimand (DiD in AUROC, 95% CI, Holm-adjusted p).**\n")
    out.append("| Direction | Model | Primary | S1: exclude positive-at-first-hour | S2: no ICULOS (seed 0) |")
    out.append("|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for c in ("lr", "lgbm", "gru", "fs"):
            cells = []
            for k in ("primary", "S1_exclude_positive_first_hour", "S2_no_iculos_seed0"):
                r = S[k][d][c]["did_auroc"]
                cells.append(f"{fmt_ci([r['point']] + r['ci'][1:])}; p={fmt_p(r['p_holm'])}")
            out.append(f"| {DIR[d]} | {NAME[c]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def forest(S, path):
    keys = [("primary", "Primary"), ("S1_exclude_positive_first_hour", "S1"), ("S2_no_iculos_seed0", "S2")]
    colors = {"primary": INK, "S1_exclude_positive_first_hour": "#2a78d6", "S2_no_iculos_seed0": "#eb6834"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rows = [(d, c) for d in ("AtoB", "BtoA") for c in ("lr", "lgbm", "gru", "fs")]
    for i, (d, c) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, (k, lab) in enumerate(keys):
            r = S[k][d][c]["did_auroc"]
            yy = y + 0.22 * (1 - j)
            ax.plot(r["ci"][1:], [yy, yy], color=colors[k], lw=1.5)
            ax.scatter([r["point"]], [yy], s=22, color=colors[k], zorder=3, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)), [f"{DIR[d]}  {NAME[c]}" for d, c in rows][::-1])
    ax.set_xlabel("DiD in AUROC: extra transfer loss attributable to process features")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def coverage_fig(S, path):
    Sec = S["secondary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, d, key, title in ((axes[0], "AtoB", "coverage_pos", "A→B: septic hours"),
                              (axes[1], "BtoA", "coverage_neg", "B→A: non-septic hours")):
        x = np.arange(4)
        for off, v, col in ((-0.17, "full", FULL), (0.17, "pf", PF)):
            vals = [Sec[d][f"{c}_{v}"]["tgt_test"]["conformal"][key] for c in ("lr", "lgbm", "gru", "fs")]
            ax.bar(x + off, vals, width=0.32, color=col, label="FULL" if v == "full" else "Process-free")
        ax.axhline(0.9, color=INK2, lw=0.9, ls="--")
        ax.set_xticks(x, [NAME[c].replace("Logistic regression", "LR") for c in ("lr", "lgbm", "gru", "fs")], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("External conformal coverage (nominal 0.90)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="outside upper center", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def decomposition_fig(path, k=12):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    for ax, d in zip(axes, ("AtoB", "BtoA")):
        r = json.load(open(os.path.join(ROOT, d, "fs_full_decomposition.json")))["seeds"]["0"]
        rows = r["top_contributions"][:k][::-1]
        c = [x["contribution"] for x in rows]
        ax.barh(range(len(rows)), c, color=[FULL if v > 0 else "#e34948" for v in c], height=0.7)
        ax.set_yticks(range(len(rows)), [x["term"] for x in rows], fontsize=7.5)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel("Contribution to Δ mean logit")
        ax.set_title(f"{DIR[d]}: Δ = {r['actual_mean_logit_change']:+.2f}; process terms {100 * r['share_abs_contribution_measurement_terms']:.0f}%",
                     loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(path)
    plt.close(fig)


def main():
    S = json.load(open(os.path.join(ROOT, "summary.json")))
    os.makedirs(FIG, exist_ok=True)
    with open("docs/manuscript_tables.md", "w") as f:
        f.write(tables(S) + "\n")
    forest(S, os.path.join(FIG, "confirm_fig1_did_forest.png"))
    coverage_fig(S, os.path.join(FIG, "confirm_fig2_conformal.png"))
    decomposition_fig(os.path.join(FIG, "confirm_fig3_decomposition.png"))


if __name__ == "__main__":
    main()
