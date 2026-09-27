import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from staf.calibration import fit_temperature, mondrian_sets, set_report  # noqa: E402


def _draw(rng, n):
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(2 * x - 2)))
    return p, (rng.random(n) < p).astype(int)


def test_mondrian_class_conditional_coverage():
    rng = np.random.default_rng(0)
    pc, yc = _draw(rng, 20000)
    pt, yt = _draw(rng, 20000)
    r = set_report(mondrian_sets(pc, yc, pt, alpha=0.1), yt)
    assert abs(r["coverage_pos"] - 0.9) < 0.02 and abs(r["coverage_neg"] - 0.9) < 0.02


def test_unit_weights_match_unweighted():
    rng = np.random.default_rng(1)
    pc, yc = _draw(rng, 5000)
    pt, _ = _draw(rng, 3000)
    a = mondrian_sets(pc, yc, pt, alpha=0.1)
    b = mondrian_sets(pc, yc, pt, alpha=0.1, w_cal=np.ones(len(pc)), w_test=np.ones(len(pt)))
    assert (a != b).mean() < 0.01


def test_temperature_recovers_scale():
    rng = np.random.default_rng(2)
    logit = rng.normal(scale=2, size=50000)
    y = (rng.random(len(logit)) < 1 / (1 + np.exp(-logit))).astype(float)
    assert abs(fit_temperature(3 * logit, y) - 3) < 0.15


def test_em_recovers_label_shift_prior():
    from staf.calibration import em_prior_shift
    rng = np.random.default_rng(3)
    n = 200000
    y = rng.random(n) < 0.03
    x = rng.normal(np.where(y, 1.0, -1.0), 1.0)
    logit_src = np.log(0.1 / 0.9) + 2 * x
    pi, _ = em_prior_shift(logit_src, 0.1)
    assert abs(pi - 0.03) < 0.003
