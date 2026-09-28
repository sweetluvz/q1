"""Secondary estimand 5 (docs/analysis_plan.md): exact additive decomposition of the source->target
change in mean logit for the FULL fuzzy scorecard, with top-20 term stability across seeds.
Output: results_confirm/{dir}/fs_full_decomposition.json
"""
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from analyze_staf import term_means  # noqa: E402
from run import prepare_site  # noqa: E402
from staf.data import DYNAMIC, STATIC, Normalizer, row_index, subset  # noqa: E402
from staf.model import STAF, init_from_data  # noqa: E402
from staf.train import nn_inputs, predict_logits  # noqa: E402

ROOT = os.environ.get("CONFIRM_DIR", "results_confirm")


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", 4)))
    for s, t in (("A", "B"), ("B", "A")):
        d = os.path.join(ROOT, f"{s}to{t}")
        src, sp = prepare_site(s)
        tgt, tp = prepare_site(t)
        tr_rows = row_index(src["offsets"], sp["train"])
        nd = Normalizer().fit(src["Xf"][tr_rows])
        ns = Normalizer().fit(np.column_stack([src["S"], np.log1p(src["iculos"])])[tr_rows])

        def part(site, pats):
            x = subset(site, pats)
            return nn_inputs(x, x["Xf"], x["D"], nd, ns)

        tr = part(src, sp["train"])
        init = init_from_data(tr["z"], tr["seen"], tr["delta"], tr["static"])
        del tr
        va, te = part(src, sp["val"]), part(tgt, tp["test"])
        seeds, tops = {}, []
        for seed in range(5):
            path = os.path.join(d, f"fs_full_s{seed}.pt")
            if not os.path.exists(path):
                continue
            m = STAF(*init, DYNAMIC, STATIC + ["ICULOS"], n_rules=0, seed=seed)
            m.load_state_dict(torch.load(path))
            m.eval()
            mT_s, _ = term_means(m, va)
            mT_t, _ = term_means(m, te)
            v = m.v.detach().numpy()
            contrib = v * (mT_t - mT_s)
            actual = float(predict_logits(m, te).mean() - predict_logits(m, va).mean())
            imp = np.abs(v) * mT_s
            top20 = [m.term_names[i] for i in np.argsort(-imp)[:20]]
            tops.append(set(top20))
            is_meas = np.array(["measured" in n for n in m.term_names])
            order = np.argsort(-np.abs(contrib))
            seeds[seed] = dict(
                actual_mean_logit_change=actual, decomposition_total=float(contrib.sum()),
                share_abs_contribution_measurement_terms=float(np.abs(contrib[is_meas]).sum() / np.abs(contrib).sum()),
                share_abs_contribution_measurement_top15=float(np.abs(contrib[order[:15]][is_meas[order[:15]]]).sum()
                                                               / np.abs(contrib[order[:15]]).sum()),
                n_measurement_terms_in_top20=int(sum("measured" in n for n in top20)),
                n_nonzero_terms=int((np.abs(v) > 1e-2).sum()), n_terms=len(v),
                top_contributions=[dict(term=m.term_names[i], weight=float(v[i]), mean_src=float(mT_s[i]),
                                        mean_tgt=float(mT_t[i]), contribution=float(contrib[i])) for i in order[:15]],
                top20_by_importance=top20)
            print(d, "seed", seed, "Δ=%.3f sum=%.3f meas-share=%.2f meas-in-top20=%d" % (
                actual, contrib.sum(), seeds[seed]["share_abs_contribution_measurement_terms"],
                seeds[seed]["n_measurement_terms_in_top20"]), flush=True)
        jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        out = dict(seeds=seeds, top20_jaccard=[float(np.mean(jac)), float(np.min(jac)), float(np.max(jac))])
        json.dump(out, open(os.path.join(d, "fs_full_decomposition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
