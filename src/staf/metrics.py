import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP = 1.0, -2.0, -0.05


def _time_and_onset(y, offsets):
    lens = np.diff(offsets)
    pat = np.repeat(np.arange(len(lens)), lens)
    t = np.arange(len(y)) - offsets[pat]
    first_pos = np.full(len(lens), np.inf)
    pos = np.flatnonzero(y)
    np.minimum.at(first_pos, pat[pos], t[pos])
    t_sep = first_pos - DT_OPTIMAL
    return pat, t, t_sep, lens


def _row_utility(pred, t, t_sep):
    septic = np.isfinite(t_sep)
    dt = np.where(septic, t - t_sep, 0.0)
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY); b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL); b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL); b3 = -m3 * DT_OPTIMAL
    early = dt <= DT_OPTIMAL
    u_tp = np.where(early, np.maximum(m1 * dt + b1, U_FP), m2 * dt + b2)
    u_fn = np.where(early, 0.0, m3 * dt + b3)
    u = np.where(septic, np.where(pred, u_tp, u_fn), np.where(pred, U_FP, 0.0))
    return np.where(septic & (dt > DT_LATE), 0.0, u)


def normalized_utility(y, pred, offsets):
    """Vectorised re-implementation of the official PhysioNet 2019 normalized utility."""
    pat, t, t_sep, lens = _time_and_onset(y, offsets)
    ts = t_sep[pat]
    best = np.isfinite(ts) & (t >= ts + DT_EARLY) & (t <= ts + DT_LATE)
    obs = _row_utility(pred.astype(bool), t, ts).sum()
    bst = _row_utility(best, t, ts).sum()
    ina = _row_utility(np.zeros_like(best), t, ts).sum()
    return (obs - ina) / (bst - ina)


def best_threshold(y, p, offsets, grid=None):
    grid = np.quantile(p, np.linspace(0.80, 0.995, 60)) if grid is None else grid
    scores = [normalized_utility(y, p >= g, offsets) for g in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def ece(y, p, n_bins=15, adaptive=True):
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    conf = np.bincount(b, weights=p, minlength=n_bins)
    acc = np.bincount(b, weights=y, minlength=n_bins)
    m = cnt > 0
    return float(np.sum(np.abs(acc[m] - conf[m])) / len(p))


def summarize(y, p, offsets, thr):
    return dict(
        auroc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)),
        utility=float(normalized_utility(y, p >= thr, offsets)),
        ece=ece(y, p),
        brier=float(np.mean((p - y) ** 2)),
        mean_p=float(p.mean()),
        prevalence=float(y.mean()),
    )


def bootstrap_patients(offsets, n_boot=200, seed=0):
    """Yield (row_idx, new_offsets) for patient-level bootstrap resamples."""
    rng = np.random.default_rng(seed)
    n = len(offsets) - 1
    lens = np.diff(offsets)
    for _ in range(n_boot):
        pats = rng.integers(0, n, n)
        l = lens[pats]
        starts = offsets[pats]
        rows = np.repeat(starts - np.concatenate([[0], np.cumsum(l)[:-1]]), l) + np.arange(l.sum())
        yield rows, np.concatenate([[0], np.cumsum(l)])
