"""Interpretability analyses for trained STAF models (run after experiments/run.py).

Outputs results/{dir}/staf_analysis.json: rules, scorecard, rule-truncation fidelity, seed stability of
the top terms, learned half-lives / fuzzy sets, and an exact additive decomposition of the source->target
change in mean logit into per-term and per-rule contributions.
"""
import json
import os
import sys
import warnings

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from run import OUT, prepare_site  # noqa: E402
from staf.data import DYNAMIC, STATIC, Normalizer, row_index, subset  # noqa: E402
from staf.interpret import rule_table, scorecard, sparsify_scorecard, truncate_rules  # noqa: E402
from staf.metrics import best_threshold, normalized_utility  # noqa: E402
from staf.model import STAF, init_from_data  # noqa: E402
from staf.train import batches, nn_inputs, predict_logits  # noqa: E402


@torch.no_grad()
def term_means(model, inp):
    sT, sF, n = 0.0, 0.0, 0
    for _, b, mask in batches(inp, 256, shuffle=False):
        _, T, f = model(b["z"], b["delta"], b["seen"], b["static"], return_parts=True)
        m = mask.unsqueeze(-1)
        sT = sT + (T * m).sum((0, 1)); sF = sF + (f * m).sum((0, 1)); n += mask.sum()
    return (sT / n).numpy(), (sF / n).numpy()


def perf(model, inp, thr):
    p = predict_logits(model, inp)
    return dict(auroc=float(roc_auc_score(inp["y"], p)),
                utility=float(normalized_utility(inp["y"], p >= thr, inp["offsets"])))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    for src_name, tgt_name in (("A", "B"), ("B", "A")):
        d = os.path.join(OUT, f"{src_name}to{tgt_name}")
        if not os.path.exists(os.path.join(d, "staf_s0.pt")):
            continue
        src, sp = prepare_site(src_name)
        tgt, tp = prepare_site(tgt_name)
        tr_rows = row_index(src["offsets"], sp["train"])
        nd = Normalizer().fit(src["Xf"][tr_rows])
        ns = Normalizer().fit(np.column_stack([src["S"], np.log1p(src["iculos"])])[tr_rows])

        def part(site, pats):
            s = subset(site, pats)
            return nn_inputs(s, s["Xf"], s["D"], nd, ns)

        tr, va, te_s, te_t = part(src, sp["train"]), part(src, sp["val"]), part(src, sp["test"]), part(tgt, tp["test"])
        init = init_from_data(tr["z"], tr["seen"], tr["delta"], tr["static"])
        del tr
        report = {"seeds": {}}
        top_sets = []
        for seed in (0, 1, 2):
            path = os.path.join(d, f"staf_s{seed}.pt")
            if not os.path.exists(path):
                continue
            model = STAF(*init, DYNAMIC, STATIC + ["ICULOS"], seed=seed)
            model.load_state_dict(torch.load(path))
            model.eval()
            thr, _ = best_threshold(va["y"], predict_logits(model, va), va["offsets"])
            mT_s, mF_s = term_means(model, va)
            mT_t, mF_t = term_means(model, te_t)
            v, beta = model.v.detach().numpy(), model.beta.detach().numpy()
            d_terms = v * (mT_t - mT_s)
            d_rules = beta * (mF_t - mF_s)
            order = np.argsort(-np.abs(d_terms))[:15]
            actual = float(predict_logits(model, te_t).mean() - predict_logits(model, va).mean())
            sc = scorecard(model, mT_s, n=20)
            top_sets.append({r["term"] for r in sc})
            fid = {"full": {"src_test": perf(model, te_s, thr), "tgt_test": perf(model, te_t, thr)}}
            for k in (1, 2, 3, 5):
                mk = truncate_rules(model, k)
                fid[f"rules_top{k}"] = {"src_test": perf(mk, te_s, thr), "tgt_test": perf(mk, te_t, thr)}
            for n_keep in (20, 50):
                mk = sparsify_scorecard(truncate_rules(model, 3), n_keep)
                fid[f"rules_top3_scorecard{n_keep}"] = {"src_test": perf(mk, te_s, thr), "tgt_test": perf(mk, te_t, thr)}
            nz = int((np.abs(v) > 1e-2).sum())
            report["seeds"][seed] = dict(
                threshold_prob=float(expit(thr)),
                n_terms=len(model.term_names), n_nonzero_scorecard=nz,
                rules=rule_table(model, mF_s, k=3)[:10],
                scorecard=sc,
                fidelity=fid,
                shift_decomposition=dict(
                    actual_mean_logit_change=actual,
                    scorecard_total=float(d_terms.sum()), rules_total=float(d_rules.sum()),
                    top_terms=[dict(term=model.term_names[i], weight=float(v[i]), mean_src=float(mT_s[i]),
                                    mean_tgt=float(mT_t[i]), contribution=float(d_terms[i])) for i in order]),
                halflife_h={f: float(h) for f, h in zip(DYNAMIC, model.halflife().detach().numpy())},
                halflife_init_h={f: float(h) for f, h in zip(DYNAMIC, init[2])},
            )
            print(d, "seed", seed, "fidelity", json.dumps({k: round(v["tgt_test"]["auroc"], 4) for k, v in fid.items()}), flush=True)
        if len(top_sets) > 1:
            jac = [len(a & b) / len(a | b) for i, a in enumerate(top_sets) for b in top_sets[i + 1:]]
            report["top20_scorecard_jaccard"] = [float(np.mean(jac)), float(np.min(jac)), float(np.max(jac))]
        with open(os.path.join(d, "staf_analysis.json"), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
