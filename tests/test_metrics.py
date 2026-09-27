import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "evaluation_2019"))

from evaluate_sepsis_score import compute_prediction_utility  # noqa: E402
from staf.metrics import DT_EARLY, DT_LATE, DT_OPTIMAL, normalized_utility  # noqa: E402


def _official_normalized(y, pred, offsets):
    obs = best = ina = 0.0
    for a, b in zip(offsets[:-1], offsets[1:]):
        lab, pr = y[a:b], pred[a:b]
        n = b - a
        bp = np.zeros(n)
        if lab.any():
            ts = int(np.argmax(lab)) - DT_OPTIMAL
            bp[max(0, ts + DT_EARLY): min(ts + DT_LATE + 1, n)] = 1
        obs += compute_prediction_utility(lab, pr)
        best += compute_prediction_utility(lab, bp)
        ina += compute_prediction_utility(lab, np.zeros(n))
    return (obs - ina) / (best - ina)


def _synthetic(rng, n_pat=300):
    ys, offs = [], [0]
    for _ in range(n_pat):
        n = int(rng.integers(5, 80))
        lab = np.zeros(n, dtype=int)
        if rng.random() < 0.3:
            lab[int(rng.integers(0, n)):] = 1
        ys.append(lab)
        offs.append(offs[-1] + n)
    return np.concatenate(ys), np.array(offs)


def test_utility_matches_official():
    rng = np.random.default_rng(0)
    y, off = _synthetic(rng)
    for rate in (0.0, 0.05, 0.3, 1.0):
        pred = (rng.random(len(y)) < rate).astype(int)
        assert np.isclose(normalized_utility(y, pred, off), _official_normalized(y, pred, off))


def test_utility_bounds():
    rng = np.random.default_rng(1)
    y, off = _synthetic(rng)
    assert np.isclose(normalized_utility(y, np.zeros_like(y), off), 0.0)
    pat = np.repeat(np.arange(len(off) - 1), np.diff(off))
    t = np.arange(len(y)) - off[pat]
    first = {p: t[(pat == p) & (y == 1)].min() for p in np.unique(pat[y == 1])}
    best = np.array([p in first and first[p] - DT_OPTIMAL + DT_EARLY <= ti <= first[p] - DT_OPTIMAL + DT_LATE
                     for p, ti in zip(pat, t)])
    assert np.isclose(normalized_utility(y, best, off), 1.0)
