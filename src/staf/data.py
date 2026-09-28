import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
        "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
DYNAMIC = VITALS + LABS
STATIC = ["Age", "Gender", "HospAdmTime"]
NEVER_SEEN = 1e3
RAW_DIR = "data/raw_official"


def _read(path):
    d = pd.read_csv(path, sep="|")
    return (d[DYNAMIC].to_numpy(np.float32), d[STATIC].to_numpy(np.float32),
            d["ICULOS"].to_numpy(np.float32), d["SepsisLabel"].to_numpy(np.int8))


def load_site(raw_dir, site, cache_dir="data/processed"):
    """Ragged per-hour arrays for one hospital system; rows of patient i are offsets[i]:offsets[i+1]."""
    cache = os.path.join(cache_dir, f"site{site}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=False))
    files = sorted(glob.glob(os.path.join(raw_dir, f"training_set{site}", "*.psv")))
    with Pool(os.cpu_count()) as p:
        parts = p.map(_read, files, chunksize=256)
    lens = np.array([len(x[3]) for x in parts])
    out = dict(
        X=np.concatenate([x[0] for x in parts]),
        S=np.concatenate([x[1] for x in parts]),
        iculos=np.concatenate([x[2] for x in parts]),
        y=np.concatenate([x[3] for x in parts]),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        pid=np.array([os.path.basename(f)[:-4] for f in files]),
    )
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def ffill_and_delta(X, offsets):
    """Causal last-observation-carried-forward and hours since last observation (NEVER_SEEN if none)."""
    Xf = np.empty_like(X)
    D = np.empty_like(X)
    for a, b in zip(offsets[:-1], offsets[1:]):
        x = X[a:b]
        obs = ~np.isnan(x)
        t = np.arange(b - a)[:, None]
        last = np.where(obs, t, -1)
        last = np.maximum.accumulate(last, axis=0)
        Xf[a:b] = np.take_along_axis(np.where(obs, x, 0), np.maximum(last, 0), axis=0)
        Xf[a:b][last < 0] = np.nan
        D[a:b] = np.where(last < 0, NEVER_SEEN, t - last)
    return Xf, D


def patient_split(site_data, seed=0, frac=(0.70, 0.15, 0.15)):
    """Patient-level split stratified by whether the patient ever becomes septic."""
    off, y = site_data["offsets"], site_data["y"]
    septic = np.array([y[a:b].any() for a, b in zip(off[:-1], off[1:])])
    rng = np.random.default_rng(seed)
    idx = {"train": [], "val": [], "test": []}
    for cls in (False, True):
        p = rng.permutation(np.flatnonzero(septic == cls))
        n1 = int(frac[0] * len(p))
        n2 = n1 + int(frac[1] * len(p))
        idx["train"].append(p[:n1]); idx["val"].append(p[n1:n2]); idx["test"].append(p[n2:])
    return {k: np.sort(np.concatenate(v)) for k, v in idx.items()}


def row_index(offsets, patients):
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in patients])


def subset(site_data, patients):
    """Restrict ragged arrays to a set of patients, re-basing offsets."""
    off = site_data["offsets"]
    rows = row_index(off, patients)
    lens = off[patients + 1] - off[patients]
    out = {k: v[rows] for k, v in site_data.items() if k not in ("offsets", "pid")}
    out["offsets"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out["pid"] = site_data["pid"][patients]
    return out


class Normalizer:
    """Robust per-feature scaling fitted on source-hospital training rows only."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        self.scale = np.where(q75 - q25 > 1e-6, q75 - q25, np.nanstd(X, axis=0) + 1e-6)
        return self

    def __call__(self, X):
        return np.clip((X - self.med) / self.scale, -10, 10)
