"""Confirmatory evaluation following docs/analysis_plan.md -> results_confirm/summary.json

Primary: per model class and direction s->t, on the same target-test patients,
DiD = [AUROC_loc(full) - AUROC_ext(full)] - [AUROC_loc(pf) - AUROC_ext(pf)],
1,000 patient-level bootstrap resamples, two-sided bootstrap p, Holm over 8 tests.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")

from staf.calibration import bbse_prior, calibration_stats, em_prior_shift, patient_mondrian  # noqa: E402
from staf.metrics import best_threshold, bootstrap_patients, normalized_utility  # noqa: E402

ROOT = os.environ.get("CONFIRM_DIR", "results_confirm")
CLASSES = ("lr", "lgbm", "gru", "fs")
N_BOOT = int(os.environ.get("N_BOOT", 1000))
REV = {"AtoB": "BtoA", "BtoA": "AtoB"}


def auroc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def load(root, d):
    lab = dict(np.load(os.path.join(root, d, "labels.npz")))
    ens, per_seed = {}, {}
    for cls in CLASSES:
        for v in ("full", "pf"):
            fs = sorted(glob.glob(os.path.join(root, d, f"{cls}_{v}_s[0-9].npz")))
            if not fs:
                continue
            runs = [dict(np.load(f)) for f in fs]
            ens[f"{cls}_{v}"] = {k: np.mean([r[k] for r in runs], 0) for k in runs[0]}
            per_seed[f"{cls}_{v}"] = runs
    return lab, ens, per_seed


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def boot_p(samples):
    s = np.asarray(samples)
    return float(min(1.0, 2 * min((np.sum(s <= 0) + 1) / (len(s) + 1), (np.sum(s >= 0) + 1) / (len(s) + 1))))


def ci(s):
    return [float(np.mean(s)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def primary(root, labs, enss, keep_patients=None, n_boot=N_BOOT):
    """DiD in AUROC (and utility, secondary) for each class and direction; optional patient filter."""
    out = {}
    for d in ("AtoB", "BtoA"):
        lab, ens = labs[d], enss[d]
        lab_r, ens_r = labs[REV[d]], enss[REV[d]]
        assert np.array_equal(lab["pid_tgt_test"], lab_r["pid_src_test"]), "target test patients must match"
        y, off = lab["y_tgt_test"], lab["off_tgt_test"]
        preds, thr = {}, {}
        for cls in CLASSES:
            for v in ("full", "pf"):
                cv = f"{cls}_{v}"
                if cv not in ens or cv not in ens_r:
                    continue
                preds[(cv, "ext")] = ens[cv]["tgt_test"]
                preds[(cv, "loc")] = ens_r[cv]["src_test"]
                thr[(cv, "ext")] = best_threshold(lab["y_src_val"], ens[cv]["src_val"], lab["off_src_val"])[0]
                thr[(cv, "loc")] = best_threshold(lab_r["y_src_val"], ens_r[cv]["src_val"], lab_r["off_src_val"])[0]
        if keep_patients is not None:
            keep = keep_patients[d]
            lens = np.diff(off)
            rows = np.concatenate([np.arange(off[i], off[i + 1]) for i in np.flatnonzero(keep)])
            off = np.r_[0, np.cumsum(lens[keep])]
            y = y[rows]
            preds = {k: p[rows] for k, p in preds.items()}

        def stats(yy, oo, idx):
            a = {k: auroc(yy, p[idx]) for k, p in preds.items()}
            u = {k: normalized_utility(yy, p[idx] >= thr[k], oo) for k, p in preds.items()}
            return a, u

        a0, u0 = stats(y, off, np.arange(len(y)))
        boots = {cls: {"did_auroc": [], "did_utility": [], "dext_auroc": [], "dext_utility": []} for cls in CLASSES}
        for rows, noff in bootstrap_patients(off, n_boot, seed=1):
            yy = y[rows]
            if yy.sum() == 0:
                continue
            a, u = stats(yy, noff, rows)
            for cls in CLASSES:
                f, p = f"{cls}_full", f"{cls}_pf"
                if (f, "ext") not in a or (p, "ext") not in a:
                    continue
                boots[cls]["did_auroc"].append((a[(f, "loc")] - a[(f, "ext")]) - (a[(p, "loc")] - a[(p, "ext")]))
                boots[cls]["did_utility"].append((u[(f, "loc")] - u[(f, "ext")]) - (u[(p, "loc")] - u[(p, "ext")]))
                boots[cls]["dext_auroc"].append(a[(f, "ext")] - a[(p, "ext")])
                boots[cls]["dext_utility"].append(u[(f, "ext")] - u[(p, "ext")])
        res = {}
        for cls in CLASSES:
            f, p = f"{cls}_full", f"{cls}_pf"
            if (f, "ext") not in a0 or (p, "ext") not in a0:
                continue
            res[cls] = dict(
                auroc={f"{v}_{w}": float(a0[(f"{cls}_{v}", w)]) for v in ("full", "pf") for w in ("loc", "ext")},
                utility={f"{v}_{w}": float(u0[(f"{cls}_{v}", w)]) for v in ("full", "pf") for w in ("loc", "ext")},
                did_auroc=dict(point=float((a0[(f, "loc")] - a0[(f, "ext")]) - (a0[(p, "loc")] - a0[(p, "ext")])),
                               ci=ci(boots[cls]["did_auroc"]), p=boot_p(boots[cls]["did_auroc"])),
                did_utility=dict(ci=ci(boots[cls]["did_utility"]), p=boot_p(boots[cls]["did_utility"])),
                dext_auroc=dict(ci=ci(boots[cls]["dext_auroc"]), p=boot_p(boots[cls]["dext_auroc"])),
                dext_utility=dict(ci=ci(boots[cls]["dext_utility"]), p=boot_p(boots[cls]["dext_utility"])),
                n_patients=int(len(off) - 1), n_septic=int(sum(y[off[i]:off[i + 1]].any() for i in range(len(off) - 1))),
                n_boot=len(boots[cls]["did_auroc"]))
        out[d] = res
    keys = [(d, c) for d in out for c in out[d]]
    adj = holm(np.array([out[d][c]["did_auroc"]["p"] for d, c in keys]))
    for (d, c), pa in zip(keys, adj):
        out[d][c]["did_auroc"]["p_holm"] = float(pa)
    return out


def secondary(labs, enss, per_seed):
    out = {}
    for d in ("AtoB", "BtoA"):
        lab, ens = labs[d], enss[d]
        res = {}
        for cv, pr in ens.items():
            y_v, off_v = lab["y_src_val"], lab["off_src_val"]
            thr, _ = best_threshold(y_v, pr["src_val"], off_v)
            r = {}
            for part in ("src_test", "tgt_test"):
                y, off = lab[f"y_{part}"], lab[f"off_{part}"]
                r[part] = dict(
                    auroc=float(auroc(y, pr[part])), auprc=float(average_precision_score(y, pr[part])),
                    utility=float(normalized_utility(y, pr[part] >= thr, off)),
                    utility_oracle_thr=float(best_threshold(y, pr[part], off)[1]),
                    auroc_seed_sd=float(np.std([auroc(y, s[part]) for s in per_seed[d][cv]])),
                    n_seeds=len(per_seed[d][cv]),
                    **calibration_stats(pr[part], y),
                    conformal=patient_mondrian(expit(pr["src_val"]), y_v, off_v, expit(pr[part]), y, off, alpha=0.1),
                )
            pi_s = float(y_v.mean())
            r["prevalence"] = dict(true_target=float(lab["y_tgt_test"].mean()), source=pi_s,
                                   bbse=bbse_prior(expit(pr["src_val"]), y_v, expit(pr["tgt_test"])),
                                   em=em_prior_shift(pr["tgt_test"], pi_s)[0])
            res[cv] = r
        out[d] = res
    return out


def main():
    labs, enss, seeds = {}, {}, {}
    for d in ("AtoB", "BtoA"):
        labs[d], enss[d], seeds[d] = load(ROOT, d)
    summary = {"n_boot": N_BOOT, "primary": primary(ROOT, labs, enss)}
    keep = {d: labs[d]["firstpos_tgt_test"] == 0 for d in labs}
    summary["S1_exclude_positive_first_hour"] = primary(ROOT, labs, enss, keep_patients=keep)
    summary["secondary"] = secondary(labs, enss, seeds)
    s2 = ROOT + "_noiculos"
    if all(os.path.exists(os.path.join(s2, d, "labels.npz")) for d in labs):
        l2, e2 = {}, {}
        for d in labs:
            l2[d], e2[d], _ = load(s2, d)
        summary["S2_no_iculos_seed0"] = primary(s2, l2, e2)
    with open(os.path.join(ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for d, res in summary["primary"].items():
        for c, r in res.items():
            print(d, c, "DiD AUROC %.4f [%.4f, %.4f] p=%.4f holm=%.4f" % (
                r["did_auroc"]["point"], *r["did_auroc"]["ci"][1:], r["did_auroc"]["p"], r["did_auroc"]["p_holm"]))


if __name__ == "__main__":
    main()
