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


def test_bbse_recovers_label_shift_prior():
    from staf.calibration import bbse_prior
    rng = np.random.default_rng(4)

    def draw(n, prior):
        y = (rng.random(n) < prior).astype(int)
        x = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
        return 1 / (1 + np.exp(-(np.log(0.1 / 0.9) + 2 * x))), y

    ps, ys = draw(200000, 0.1)
    pt, _ = draw(200000, 0.03)
    assert abs(bbse_prior(ps, ys, pt) - 0.03) < 0.004


def test_patient_mondrian_coverage_exchangeable():
    from staf.calibration import patient_mondrian
    rng = np.random.default_rng(5)

    def cohort(n):
        lens = rng.integers(5, 40, n)
        off = np.r_[0, np.cumsum(lens)]
        y = np.concatenate([(np.arange(L) >= L - 5) & (rng.random() < 0.3) for L in lens]).astype(int)
        p = 1 / (1 + np.exp(-(rng.normal(size=len(y)) + 2 * y - 2)))
        return p, y, off

    pc, yc, oc = cohort(3000)
    pt, yt, ot = cohort(3000)
    r = patient_mondrian(pc, yc, oc, pt, yt, ot, alpha=0.1, draws=50)
    assert abs(r["coverage_pos"] - 0.9) < 0.03 and abs(r["coverage_neg"] - 0.9) < 0.03
