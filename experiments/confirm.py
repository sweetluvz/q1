"""Confirmatory run following docs/analysis_plan.md.

Four model classes x {full, pf} (process-free). Equal-budget grid on source-validation log-loss
(seed 0), then the selected configuration refit with seeds 0-4. Caches logits on source-val,
source-test and target-test per variant/config/seed.

Usage: python experiments/confirm.py --src A --tgt B [--no-iculos] [--seeds 0,1,2,3,4]
"""
import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from run import prepare_site  # noqa: E402
from staf.data import DYNAMIC, STATIC, Normalizer, row_index, subset  # noqa: E402
from staf.model import STAF, GRUBaseline, init_from_data  # noqa: E402
from staf.train import fit, nn_inputs, predict_logits, val_logloss  # noqa: E402

# Engineered layout (features.py): 0-33 LOCF values, 34-67 time-since-measured, 68-107 vital windows,
# 108-110 static, 111 ICULOS, 112 n_labs_seen.
VALUE_COLS = np.r_[0:34, 68:112]
ICULOS_COL = 111
GRIDS = {
    "lr": [dict(C=c) for c in (0.01, 0.1, 1.0)],
    "lgbm": [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mc)
             for nl, lr, mc in itertools.product((15, 63), (0.03, 0.1), (100, 500))],
    "gru": [dict(hidden=h, lr=lr) for h, lr in itertools.product((32, 64), (1e-3, 3e-3))],
    "fs": [dict(lr=lr, reg=r) for lr, r in itertools.product((3e-3, 1e-2), (1e-4, 1e-3))],
}
VARIANTS = [f"{c}_{v}" for c in ("lr", "lgbm", "gru", "fs") for v in ("full", "pf")]


def tabular_design(F, rows, variant, no_iculos, med=None):
    cols = np.arange(F.shape[1]) if variant == "full" else VALUE_COLS
    if no_iculos:
        cols = cols[cols != ICULOS_COL]
    X = F[rows][:, cols]
    if variant == "pf":
        X = np.where(np.isnan(X), med[cols], X)
    return X


def fit_tabular(cls, variant, cfg, seed, Xtr, ytr, X_eval, yval):
    if cls == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=3000, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                               reg_lambda=1.0, random_state=seed, verbose=-1, **cfg)
        m.fit(Xtr, ytr, eval_set=[(X_eval["src_val"], yval)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        return {k: m.predict(v, raw_score=True) for k, v in X_eval.items()}, int(m.best_iteration_)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=(variant == "full")),
                      StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=3000))
    m.fit(Xtr, ytr)
    return {k: m.decision_function(v) for k, v in X_eval.items()}, None


def to_process_free(inp):
    out = dict(inp)
    out["seen"] = np.ones_like(inp["seen"])
    out["delta"] = np.zeros_like(inp["delta"])
    return out


def drop_iculos(inp):
    out = dict(inp)
    out["static"] = inp["static"][:, :3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="A")
    ap.add_argument("--tgt", default="B")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no-iculos", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    root = os.environ.get("RESULTS_DIR", "results_confirm" + ("_noiculos" if a.no_iculos else ""))
    out_dir = os.path.join(root, f"{a.src}to{a.tgt}")
    os.makedirs(out_dir, exist_ok=True)

    src, sp = prepare_site(a.src)
    tgt, tp = prepare_site(a.tgt)
    parts = {"src_val": (src, sp["val"]), "src_test": (src, sp["test"]), "tgt_test": (tgt, tp["test"])}
    tr_rows = row_index(src["offsets"], sp["train"])
    rows = {k: row_index(d["offsets"], p) for k, (d, p) in parts.items()}
    labels = {k: d["y"][rows[k]] for k, (d, _) in parts.items()}
    offs = {k: subset(d, p)["offsets"] for k, (d, p) in parts.items()}
    first_pos = {k: np.array([d["y"][d["offsets"][i]] for i in p]) for k, (d, p) in parts.items()}
    np.savez_compressed(os.path.join(out_dir, "labels.npz"), **{f"y_{k}": v for k, v in labels.items()},
                        **{f"off_{k}": v for k, v in offs.items()}, **{f"firstpos_{k}": v for k, v in first_pos.items()},
                        **{f"pid_{k}": d["pid"][p] for k, (d, p) in parts.items()})

    med = np.nanmedian(src["F"][tr_rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nd = Normalizer().fit(src["Xf"][tr_rows])
    ns = Normalizer().fit(np.column_stack([src["S"], np.log1p(src["iculos"])])[tr_rows])
    nn_cache = {}

    def nn_data(variant):
        if variant not in nn_cache:
            base = {}
            for k, (d, p) in [("train", (src, sp["train"]))] + list(parts.items()):
                s = subset(d, p)
                x = nn_inputs(s, s["Xf"], s["D"], nd, ns)
                x = to_process_free(x) if variant == "pf" else x
                base[k] = drop_iculos(x) if a.no_iculos else x
            nn_cache[variant] = base
        return nn_cache[variant]

    def train_one(cls, variant, cfg, seed):
        if cls in ("lr", "lgbm"):
            Xtr = tabular_design(src["F"], tr_rows, variant, a.no_iculos, med)
            X_eval = {k: tabular_design(d["F"], rows[k], variant, a.no_iculos, med) for k, (d, _) in parts.items()}
            preds, it = fit_tabular(cls, variant, cfg, seed, Xtr, src["y"][tr_rows], X_eval, labels["src_val"])
            return preds, {"best_iteration": it}
        data = nn_data(variant)
        tr = data["train"]
        static_names = STATIC + ([] if a.no_iculos else ["ICULOS"])
        torch.manual_seed(seed)
        if cls == "fs":
            obs = nn_data("full")["train"]
            init = init_from_data(obs["z"], obs["seen"], obs["delta"], tr["static"])
            model = STAF(*init, DYNAMIC, static_names, n_rules=0, seed=seed,
                         staleness=(variant == "full"), measurement_terms=(variant == "full"))
            lr, reg = cfg["lr"], cfg["reg"]
        else:
            model = GRUBaseline(len(DYNAMIC), tr["static"].shape[1], hidden=cfg["hidden"])
            lr, reg = cfg["lr"], 0.0
        model, best = fit(model, tr, data["src_val"], epochs=a.epochs, lr=lr, reg=reg, seed=seed, patience=5,
                          criterion="logloss", log=lambda *_: None)
        return {k: predict_logits(model, data[k]) for k in parts}, {"val_logloss_best_epoch": best,
                                                                     "state": model.state_dict()}

    seeds = [int(s) for s in a.seeds.split(",")]
    for variant_name in a.variants.split(","):
        cls, variant = variant_name.split("_")
        tune_path = os.path.join(out_dir, f"{variant_name}_tuning.json")
        if os.path.exists(tune_path):
            tuning = json.load(open(tune_path))
        else:
            tuning = []
            for i, cfg in enumerate(GRIDS[cls][:int(os.environ.get("GRID_LIMIT", 99))]):
                t0 = time.time()
                preds, info = train_one(cls, variant, cfg, 0)
                ll = val_logloss(labels["src_val"].astype(np.float32), preds["src_val"].astype(np.float32))
                tuning.append(dict(cfg=cfg, val_logloss=ll, best_iteration=info.get("best_iteration")))
                np.savez_compressed(os.path.join(out_dir, f"{variant_name}_cfg{i}_s0.npz"), **preds)
                if "state" in info:
                    torch.save(info["state"], os.path.join(out_dir, f"{variant_name}_cfg{i}_s0.pt"))
                print(f"[{a.src}->{a.tgt}] tune {variant_name} {cfg} val_logloss={ll:.5f} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(tuning, open(tune_path, "w"), indent=1)
        best_i = int(np.argmin([t["val_logloss"] for t in tuning]))
        cfg = tuning[best_i]["cfg"]
        for seed in (seeds if cls != "lr" else [0]):
            path = os.path.join(out_dir, f"{variant_name}_s{seed}.npz")
            if os.path.exists(path):
                continue
            cached = os.path.join(out_dir, f"{variant_name}_cfg{best_i}_s0.npz")
            if seed == 0 and os.path.exists(cached):
                preds = dict(np.load(cached))
                if os.path.exists(cached.replace(".npz", ".pt")):
                    torch.save(torch.load(cached.replace(".npz", ".pt")), path.replace(".npz", ".pt"))
            else:
                t0 = time.time()
                preds, info = train_one(cls, variant, cfg, seed)
                if "state" in info:
                    torch.save(info["state"], path.replace(".npz", ".pt"))
                print(f"[{a.src}->{a.tgt}] final {variant_name} {cfg} seed {seed} ({time.time() - t0:.0f}s)", flush=True)
            np.savez_compressed(path, **preds)
        json.dump(dict(selected=cfg, selected_index=best_i), open(os.path.join(out_dir, f"{variant_name}_selected.json"), "w"))


if __name__ == "__main__":
    main()
