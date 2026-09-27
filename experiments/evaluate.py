"""Evaluate cached logits: discrimination, utility, calibration, conformal coverage, bootstrap CIs.

Usage: python experiments/evaluate.py  (reads results/AtoB, results/BtoA; writes results/summary.json)
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")

from staf.calibration import (calibration_stats, density_ratio, em_prior_shift, fit_temperature,  # noqa: E402
                              mondrian_sets, set_report)
from staf.data import load_site, patient_split, row_index  # noqa: E402
from staf.metrics import best_threshold, bootstrap_patients, normalized_utility, summarize  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

OUT = "results"
ALPHA = 0.1
N_BOOT = 200


def load_direction(d):
    lab = np.load(os.path.join(OUT, d, "labels.npz"))
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, d, "*_s*.npz"))):
        name, seed = os.path.basename(f)[:-4].rsplit("_s", 1)
        runs.setdefault(name, {})[int(seed)] = dict(np.load(f))
    return lab, runs


def domain_weights(src, tgt):
    """Density ratio p_B(x)/p_A(x) on engineered features; target-train rows are used unlabeled."""
    import lightgbm as lgb
    ss, ts = patient_split(src_site := load_site("data/raw", src), 0), patient_split(tgt_site := load_site("data/raw", tgt), 0)
    Fs = np.load(f"data/processed/site{src}_feat.npz")["F"]
    Ft = np.load(f"data/processed/site{tgt}_feat.npz")["F"]
    usable = ~np.all(np.isnan(Fs[row_index(src_site["offsets"], ss["train"])]), axis=0)
    Fs, Ft = Fs[:, usable], Ft[:, usable]
    rng = np.random.default_rng(0)
    sv = row_index(src_site["offsets"], ss["val"])
    tte = row_index(tgt_site["offsets"], ts["test"])
    s_fit = rng.choice(row_index(src_site["offsets"], ss["train"]), size=150_000, replace=False)
    t_fit = rng.choice(row_index(tgt_site["offsets"], ts["train"]), size=150_000, replace=False)

    def clf_fit(X, d):
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                                  verbose=-1, random_state=0).fit(X, d)

    (w_val, w_test), clf = density_ratio(clf_fit, Fs[s_fit], Ft[t_fit], [Fs[sv], Ft[tte]])
    X_auc = np.vstack([Fs[sv], Ft[tte]])
    d_auc = np.r_[np.zeros(len(sv)), np.ones(len(tte))]
    auc = roc_auc_score(d_auc, clf.predict_proba(X_auc)[:, 1])
    ess = w_val.sum() ** 2 / (w_val ** 2).sum()
    return w_val, w_test, dict(domain_auc=float(auc), ess=float(ess), n_cal=int(len(w_val)))


def evaluate_run(lab, pred):
    y_v, y_s, y_t = lab["y_src_val"], lab["y_src_test"], lab["y_tgt_test"]
    thr_logit, _ = best_threshold(y_v, pred["src_val"], lab["off_src_val"])
    T = fit_temperature(pred["src_val"], y_v)
    res = {"T": T}
    for part, y in (("src_test", y_s), ("tgt_test", y_t)):
        r = summarize(y, expit(pred[part]), lab[f"off_{part}"], expit(thr_logit))
        r.update(calibration_stats(pred[part], y))
        r["utility_oracle_thr"] = best_threshold(y, pred[part], lab[f"off_{part}"])[1]
        res[part] = r
    pi_t, shift = em_prior_shift(pred["tgt_test"], float(y_v.mean()))
    adj = pred["tgt_test"] + shift
    em = calibration_stats(adj, y_t)
    em.update(prior_est=pi_t, prior_true=float(y_t.mean()),
              utility=float(normalized_utility(y_t, adj >= thr_logit, lab["off_tgt_test"])))
    res["tgt_test_em"] = em
    return res


def fewshot(lab, pred, T, sizes=(25, 50, 100, 200, 400), reps=30):
    """Conformal calibration on n labeled target patients, evaluated on the remaining target-test patients."""
    y, off, logit = lab["y_tgt_test"], lab["off_tgt_test"], pred["tgt_test"] / T
    n_pat = len(off) - 1
    rng = np.random.default_rng(0)
    out = {}
    for n in sizes:
        cov_pos, amb, npos = [], [], []
        for _ in range(reps):
            perm = rng.permutation(n_pat)
            cal, ev = np.sort(perm[:n]), np.sort(perm[n:])
            rc, re = row_index(off, cal), row_index(off, ev)
            npos.append(int(sum(y[off[i]:off[i + 1]].any() for i in cal)))
            sets = mondrian_sets(expit(logit[rc]), y[rc], expit(logit[re]), ALPHA)
            r = set_report(sets, y[re])
            cov_pos.append(r["coverage_pos"]); amb.append(r["ambiguous_rate"])
        out[n] = dict(coverage_pos=[float(np.mean(cov_pos)), float(np.percentile(cov_pos, 10)), float(np.percentile(cov_pos, 90))],
                      ambiguous_rate=float(np.mean(amb)), septic_patients_in_cal=float(np.mean(npos)))
    return out


def conformal(lab, pred, T, w_val=None, w_test=None):
    p_cal = expit(pred["src_val"] / T)
    p_t = expit(pred["tgt_test"] / T)
    y_cal, y_t = lab["y_src_val"], lab["y_tgt_test"]
    out = {"standard": set_report(mondrian_sets(p_cal, y_cal, p_t, ALPHA), y_t)}
    if w_val is not None:
        out["weighted"] = set_report(mondrian_sets(p_cal, y_cal, p_t, ALPHA, w_val, w_test), y_t)
    p_s = expit(pred["src_test"] / T)
    out["in_domain"] = set_report(mondrian_sets(p_cal, y_cal, p_s, ALPHA), lab["y_src_test"])
    return out


def paired_bootstrap(lab, pa, pb, part, thr_a, thr_b):
    y, off = lab[f"y_{part}"], lab[f"off_{part}"]
    d_auc, d_util = [], []
    for rows, noff in bootstrap_patients(off, N_BOOT):
        yy = y[rows]
        if yy.sum() == 0:
            continue
        d_auc.append(roc_auc_score(yy, pa[rows]) - roc_auc_score(yy, pb[rows]))
        d_util.append(normalized_utility(yy, pa[rows] >= thr_a, noff) - normalized_utility(yy, pb[rows] >= thr_b, noff))
    ci = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(d_auroc=ci(d_auc), d_utility=ci(d_util))


def seed_mean(pred_by_seed):
    """Seed-ensemble: average logits across seeds (reported alongside per-seed mean +- sd)."""
    keys = next(iter(pred_by_seed.values())).keys()
    return {k: np.mean([p[k] for p in pred_by_seed.values()], 0) for k in keys}


def main():
    summary = {}
    for d in ("AtoB", "BtoA"):
        if not os.path.exists(os.path.join(OUT, d, "labels.npz")):
            continue
        lab, runs = load_direction(d)
        src, tgt = d[0], d[-1]
        w_val, w_test, shift = domain_weights(src, tgt)
        S = {"shift": shift, "models": {}, "paired": {}}
        ens = {}
        for name, by_seed in runs.items():
            per_seed = [evaluate_run(lab, p) for p in by_seed.values()]
            agg = {}
            for part in ("src_test", "tgt_test", "tgt_test_em"):
                agg[part] = {m: [float(np.mean([r[part][m] for r in per_seed])), float(np.std([r[part][m] for r in per_seed]))]
                             for m in per_seed[0][part]}
            ens[name] = seed_mean(by_seed)
            ev = evaluate_run(lab, ens[name])
            agg["ensemble"] = ev
            agg["conformal"] = conformal(lab, ens[name], ev["T"], w_val, w_test)
            agg["fewshot"] = fewshot(lab, ens[name], ev["T"])
            agg["n_seeds"] = len(by_seed)
            S["models"][name] = agg
            print(d, name, "src AUROC %.3f tgt AUROC %.3f | src U %.3f tgt U %.3f" % (
                agg["src_test"]["auroc"][0], agg["tgt_test"]["auroc"][0], agg["src_test"]["utility"][0],
                agg["tgt_test"]["utility"][0]), flush=True)
        thr = {n: best_threshold(lab["y_src_val"], p["src_val"], lab["off_src_val"])[0] for n, p in ens.items()}
        pairs = [("staf", o) for o in ens if o != "staf"] + [("staf_nomeas", "lgbm"), ("lgbm_noproc", "lgbm"),
                                                             ("staf_nomeas", "lgbm_noproc")]
        for a, b in pairs:
            if a in ens and b in ens:
                S["paired"][f"{a}_vs_{b}"] = {
                    part: paired_bootstrap(lab, ens[a][part], ens[b][part], part, thr[a], thr[b])
                    for part in ("src_test", "tgt_test")}
        summary[d] = S
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
