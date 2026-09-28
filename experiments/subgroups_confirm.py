"""Descriptive subgroup performance (TRIPOD+AI item 14): AUROC by sex and age band on internal and
external test sets for each ensemble variant. Not part of the pre-specified analysis plan.
Output: results_confirm/subgroups.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_confirm import ROOT, auroc, load  # noqa: E402
from staf.data import RAW_DIR, load_site, patient_split, row_index  # noqa: E402


def covariates(site, part):
    d = load_site(RAW_DIR, site)
    rows = row_index(d["offsets"], patient_split(d, 0)[part])
    return d["S"][rows, 0], d["S"][rows, 1]


def main():
    out = {}
    for d in ("AtoB", "BtoA"):
        lab, ens, _ = load(ROOT, d)
        res = {}
        for part, site in (("src_test", d[0]), ("tgt_test", d[-1])):
            age, sex = covariates(site, "test")
            y = lab[f"y_{part}"]
            groups = {"female": sex == 0, "male": sex == 1, "age<65": age < 65, "age>=65": age >= 65}
            res[part] = {g: dict(n_hours=int(m.sum()), n_positive_hours=int(y[m].sum()),
                                 auroc={cv: float(auroc(y[m], p[part][m])) for cv, p in ens.items()})
                         for g, m in groups.items()}
        out[d] = res
    json.dump(out, open(os.path.join(ROOT, "subgroups.json"), "w"), indent=1)
    for d in out:
        for g, r in out[d]["tgt_test"].items():
            print(d, "external", g, r["n_positive_hours"], {k: round(v, 3) for k, v in r["auroc"].items() if k.endswith("full")})


if __name__ == "__main__":
    main()
