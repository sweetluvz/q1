import numpy as np
import pandas as pd

from .data import DYNAMIC, STATIC, VITALS


def engineered(Xf, D, S, iculos, offsets, window=6):
    """Causal hand-crafted features for tree/linear baselines (one row per patient-hour)."""
    lens = np.diff(offsets)
    pid = np.repeat(np.arange(len(lens)), lens)
    vit = pd.DataFrame(Xf[:, [DYNAMIC.index(v) for v in VITALS]], columns=VITALS)
    g = vit.groupby(pid)
    roll = g.rolling(window, min_periods=1)
    parts = [
        pd.DataFrame(Xf, columns=DYNAMIC),
        pd.DataFrame(np.log1p(np.minimum(D, 72)), columns=[f"{c}_dt" for c in DYNAMIC]),
        roll.mean().reset_index(level=0, drop=True).add_suffix("_mean6"),
        roll.min().reset_index(level=0, drop=True).add_suffix("_min6"),
        roll.max().reset_index(level=0, drop=True).add_suffix("_max6"),
        roll.std().reset_index(level=0, drop=True).add_suffix("_std6"),
        (vit - g.shift(window)).add_suffix("_diff6"),
        pd.DataFrame(S, columns=STATIC),
        pd.DataFrame({"ICULOS": iculos, "n_labs_seen": (D[:, len(VITALS):] < 1e3).sum(1)}),
    ]
    F = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return F.astype(np.float32)
