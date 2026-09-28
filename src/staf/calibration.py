import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_temperature(logits, y):
    """Scalar temperature T minimizing NLL of sigmoid(logit / T) on source validation data."""
    def nll(logT):
        p = np.clip(expit(logits / np.exp(logT[0])), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.exp(minimize(nll, [0.0], method="L-BFGS-B").x[0]))


def calibration_stats(logit, y):
    """O/E ratio, calibration-in-the-large (intercept with slope fixed at 1) and calibration slope."""
    p = expit(logit)

    def nll(params, slope_fixed):
        a, b = (params[0], 1.0) if slope_fixed else params
        q = np.clip(expit(a + b * logit), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    citl = minimize(nll, [0.0], args=(True,), method="L-BFGS-B").x[0]
    slope = minimize(nll, [0.0, 1.0], args=(False,), method="L-BFGS-B").x[1]
    return dict(oe=float(y.sum() / p.sum()), citl=float(citl), slope=float(slope))


def em_prior_shift(logit_target, prior_source, n_iter=100, tol=1e-7):
    """Saerens et al. (2002) EM estimate of the target prevalence from unlabeled target predictions.

    Returns (estimated prior, logit offset to add to source logits).
    """
    base = np.log(prior_source / (1 - prior_source))
    pi = prior_source
    for _ in range(n_iter):
        shift = np.log(pi / (1 - pi)) - base
        new = float(expit(logit_target + shift).mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi, float(np.log(pi / (1 - pi)) - base)


def _weighted_quantile_per_test(scores, w_cal, w_test, alpha):
    """Tibshirani et al. (2019) weighted conformal quantile, vectorised over test points.

    q(x) = inf{s : sum_{i: s_i <= s} w_i >= (1 - alpha) * (sum_i w_i + w(x))}, +inf if unreachable.
    """
    o = np.argsort(scores)
    s, cw = scores[o], np.cumsum(w_cal[o])
    target = (1 - alpha) * (cw[-1] + w_test)
    k = np.searchsorted(cw, target, side="left")
    q = np.full(len(w_test), np.inf)
    ok = k < len(s)
    q[ok] = s[k[ok]]
    return q


def mondrian_sets(p_cal, y_cal, p_test, alpha=0.1, w_cal=None, w_test=None):
    """Class-conditional split-conformal sets for binary labels; nonconformity = 1 - p(label).

    Returns boolean (n_test, 2): column c says whether label c is in the set. With weights, uses
    covariate-shift weighted quantiles per class.
    """
    sets = np.zeros((len(p_test), 2), bool)
    for c in (0, 1):
        m = y_cal == c
        sc = 1 - (p_cal[m] if c == 1 else 1 - p_cal[m])
        st = 1 - (p_test if c == 1 else 1 - p_test)
        if w_cal is None:
            n = m.sum()
            rank = int(np.ceil((n + 1) * (1 - alpha)))
            q = np.inf if rank > n else np.sort(sc)[rank - 1]
            sets[:, c] = st <= q
        else:
            sets[:, c] = st <= _weighted_quantile_per_test(sc, w_cal[m], w_test, alpha)
    return sets


def set_report(sets, y):
    cov = sets[np.arange(len(y)), y.astype(int)]
    size = sets.sum(1)
    return dict(
        coverage=float(cov.mean()),
        coverage_pos=float(cov[y == 1].mean()),
        coverage_neg=float(cov[y == 0].mean()),
        ambiguous_rate=float((size == 2).mean()),
        empty_rate=float((size == 0).mean()),
        alarm_rate=float((sets[:, 1] & ~sets[:, 0]).mean()),
    )


def density_ratio(clf_fit, X_src, X_tgt, X_eval_list, clip=(0.05, 20.0)):
    """w(x) = p_tgt(x)/p_src(x) from a probabilistic domain classifier (label 1 = target)."""
    X = np.vstack([X_src, X_tgt])
    d = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
    clf = clf_fit(X, d)
    prior = len(X_src) / len(X_tgt)
    out = []
    for Xe in X_eval_list:
        pr = np.clip(clf.predict_proba(Xe)[:, 1], 1e-6, 1 - 1e-6)
        out.append(np.clip(pr / (1 - pr) * prior, *clip))
    return out, clf


def bbse_prior(p_src, y_src, p_tgt, prior_src_quantile=None):
    """Black-box shift estimation (Lipton et al., 2018) for a binary label.

    Hard predictions flag the top prevalence-fraction of source scores as positive; solving
    C w = mu_t (C = source joint of prediction and label) gives importance weights w and the
    target prevalence w_1 * pi_s.
    """
    pi_s = float(y_src.mean())
    thr = np.quantile(p_src, 1 - pi_s) if prior_src_quantile is None else prior_src_quantile
    yh_s, yh_t = p_src >= thr, p_tgt >= thr
    C = np.array([[np.mean(~yh_s & (y_src == 0)), np.mean(~yh_s & (y_src == 1))],
                  [np.mean(yh_s & (y_src == 0)), np.mean(yh_s & (y_src == 1))]])
    mu = np.array([np.mean(~yh_t), np.mean(yh_t)])
    w = np.clip(np.linalg.solve(C, mu), 0, None)
    return float(np.clip(w[1] * pi_s, 0, 1))


def _one_row_per_patient(rows, patient, rng):
    """Uniformly pick one of the given rows for each patient that has any."""
    if len(rows) == 0:
        return rows
    pat = patient[rows]
    order = np.lexsort((rng.random(len(rows)), pat))
    last = np.r_[pat[order][1:] != pat[order][:-1], True]
    return rows[order][last]


def patient_mondrian(p_cal, y_cal, off_cal, p_test, y_test, off_test, alpha=0.1, draws=200, seed=0):
    """Class-conditional split conformal with one uniformly drawn hour per patient and class.

    Calibration units (patients) are exchangeable, so the finite-sample guarantee applies to a
    uniformly drawn hour of a new patient from the same distribution. Test coverage and the rate
    of ambiguous {0,1} sets are averaged over `draws` random hour draws per test patient.
    """
    rng = np.random.default_rng(seed)
    pat_c = np.repeat(np.arange(len(off_cal) - 1), np.diff(off_cal))
    pat_t = np.repeat(np.arange(len(off_test) - 1), np.diff(off_test))
    q, n_units = {}, {}
    for c in (0, 1):
        r = _one_row_per_patient(np.flatnonzero(y_cal == c), pat_c, rng)
        n_units[c] = len(r)
        sc = np.sort(1 - p_cal[r] if c == 1 else p_cal[r])
        rank = int(np.ceil((len(sc) + 1) * (1 - alpha)))
        q[c] = np.inf if rank > len(sc) else sc[rank - 1]
    in1 = (1 - p_test) <= q[1]
    in0 = p_test <= q[0]
    cov = {0: [], 1: []}
    amb = []
    for _ in range(draws):
        for c in (0, 1):
            r = _one_row_per_patient(np.flatnonzero(y_test == c), pat_t, rng)
            cov[c].append((in1 if c == 1 else in0)[r].mean())
        r = _one_row_per_patient(np.arange(len(p_test)), pat_t, rng)
        amb.append((in0 & in1)[r].mean())
    return dict(coverage_pos=float(np.mean(cov[1])), coverage_neg=float(np.mean(cov[0])),
                ambiguous_rate=float(np.mean(amb)), n_cal_pos=n_units[1], n_cal_neg=n_units[0],
                q_pos=float(q[1]), q_neg=float(q[0]))
