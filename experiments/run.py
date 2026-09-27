"""Train all models for one transfer direction and cache logits on source-val, source-test, target-test.

Usage: python experiments/run.py --src A --tgt B --models lgbm,lr,gru,staf,staf_h,staf_nostale,staf_notemp,staf_nomeas --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
warnings.filterwarnings("ignore")

from staf.data import DYNAMIC, STATIC, Normalizer, ffill_and_delta, load_site, patient_split, row_index, subset  # noqa: E402
from staf.features import engineered  # noqa: E402
from staf.interpret import fuzzy_sets_original_units  # noqa: E402
from staf.model import STAF, GRUBaseline, init_from_data  # noqa: E402
from staf.train import fit, nn_inputs, predict_logits  # noqa: E402

RAW, OUT = "data/raw", os.environ.get("RESULTS_DIR", "results")
NN_CFG = {
    "staf": dict(kind="staf", kw=dict(), lr=1e-2),
    "staf_h": dict(kind="staf", kw=dict(hybrid_hidden=64), lr=3e-3),
    "staf_nostale": dict(kind="staf", kw=dict(staleness=False), lr=1e-2),
    "staf_notemp": dict(kind="staf", kw=dict(temporal=False), lr=1e-2),
    "staf_nomeas": dict(kind="staf", kw=dict(measurement_terms=False), lr=1e-2),
    "gru": dict(kind="gru", kw=dict(hidden=64), lr=3e-3),
}


def prepare_site(site):
    d = load_site(RAW, site)
    cache = f"data/processed/site{site}_feat.npz"
    Xf, D = ffill_and_delta(d["X"], d["offsets"])
    if os.path.exists(cache):
        Fe = np.load(cache)["F"]
    else:
        Fe = engineered(Xf, D, d["S"], d["iculos"], d["offsets"]).to_numpy()
        np.savez_compressed(cache, F=Fe)
    d.update(Xf=Xf, D=D, F=Fe)
    return d, patient_split(d, seed=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="A")
    ap.add_argument("--tgt", default="B")
    ap.add_argument("--models", default="lgbm,lr,gru,staf,staf_nomeas,staf_nostale,staf_notemp,staf_h")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    out_dir = os.path.join(OUT, f"{a.src}to{a.tgt}")
    os.makedirs(out_dir, exist_ok=True)

    src, sp = prepare_site(a.src)
    tgt, tp = prepare_site(a.tgt)
    parts = {"src_val": (src, sp["val"]), "src_test": (src, sp["test"]), "tgt_test": (tgt, tp["test"])}
    tr_rows = row_index(src["offsets"], sp["train"])
    labels = {k: d["y"][row_index(d["offsets"], p)] for k, (d, p) in parts.items()}
    offs = {k: subset(d, p)["offsets"] for k, (d, p) in parts.items()}
    np.savez_compressed(os.path.join(out_dir, "labels.npz"), **{f"y_{k}": v for k, v in labels.items()},
                        **{f"off_{k}": v for k, v in offs.items()})

    nd = Normalizer().fit(src["Xf"][tr_rows])
    ns = Normalizer().fit(np.column_stack([src["S"], np.log1p(src["iculos"])])[tr_rows])

    def nn_part(d, p):
        s = subset(d, p)
        return nn_inputs(s, s["Xf"], s["D"], nd, ns)

    nn_data = None
    for name in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")] if name != "lr" else [0]:
            path = os.path.join(out_dir, f"{name}_s{seed}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            print(f"[{a.src}->{a.tgt}] {name} seed {seed}", flush=True)
            if name in ("lgbm", "lr"):
                preds = fit_tabular(name, seed, src["F"][tr_rows], src["y"][tr_rows],
                                    {k: d["F"][row_index(d["offsets"], p)] for k, (d, p) in parts.items()},
                                    labels["src_val"])
            else:
                if nn_data is None:
                    nn_data = {"train": nn_part(src, sp["train"]), **{k: nn_part(d, p) for k, (d, p) in parts.items()}}
                cfg = NN_CFG[name]
                if cfg["kind"] == "staf":
                    tr = nn_data["train"]
                    init = init_from_data(tr["z"], tr["seen"], tr["delta"], tr["static"])
                    model = STAF(*init, DYNAMIC, STATIC + ["ICULOS"], seed=seed, **cfg["kw"])
                else:
                    model = GRUBaseline(len(DYNAMIC), nn_data["train"]["static"].shape[1], **cfg["kw"])
                model, _ = fit(model, nn_data["train"], nn_data["src_val"], epochs=a.epochs, lr=cfg["lr"],
                               seed=seed, patience=8)
                preds = {k: predict_logits(model, nn_data[k]) for k in parts}
                torch.save(model.state_dict(), path.replace(".npz", ".pt"))
                if cfg["kind"] == "staf":
                    with open(path.replace(".npz", "_sets.json"), "w") as f:
                        json.dump(fuzzy_sets_original_units(model, nd), f, indent=1)
            np.savez_compressed(path, **preds)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)


def fit_tabular(name, seed, Xtr, ytr, X_eval, yval):
    if name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                               random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(X_eval["src_val"], yval)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return {k: m.predict(v, raw_score=True) for k, v in X_eval.items()}
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=2000))
    m.fit(Xtr, ytr)
    return {k: m.decision_function(v) for k, v in X_eval.items()}


if __name__ == "__main__":
    main()
