import copy

import numpy as np
import torch

from .data import DYNAMIC


def truncate_rules(model, k):
    """Copy whose rules keep only their top-k antecedents and whose scorecard keeps its top-k*R terms."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        keep = torch.topk(m.W, k, dim=1).indices
        W = torch.full_like(m.W, -30.0)
        W.scatter_(1, keep, m.W.gather(1, keep))
        m.W.copy_(W)
    return m


def sparsify_scorecard(model, n_keep):
    m = copy.deepcopy(model)
    with torch.no_grad():
        keep = torch.topk(m.v.abs(), n_keep).indices
        v = torch.zeros_like(m.v)
        v[keep] = m.v[keep]
        m.v.copy_(v)
    return m


@torch.no_grad()
def rule_table(model, mean_fire, k=3):
    A = model.antecedents().numpy()
    beta = model.beta.numpy()
    rows = []
    for r in np.argsort(-np.abs(beta) * mean_fire):
        top = np.argsort(-A[r])[:k]
        rows.append(dict(rule=int(r), beta=float(beta[r]), mean_fire=float(mean_fire[r]),
                         importance=float(abs(beta[r]) * mean_fire[r]),
                         antecedents=[(model.term_names[i], float(A[r, i])) for i in top]))
    return rows


@torch.no_grad()
def scorecard(model, mean_term, n=25):
    v = model.v.numpy()
    imp = np.abs(v) * mean_term
    return [dict(term=model.term_names[i], weight=float(v[i]), mean_membership=float(mean_term[i]),
                 importance=float(imp[i])) for i in np.argsort(-imp)[:n]]


@torch.no_grad()
def fuzzy_sets_original_units(model, norm):
    c = model.c.numpy() * norm.scale[:, None] + norm.med[:, None]
    s = model.log_s.exp().numpy() * norm.scale[:, None]
    return {f: dict(centers=c[j].round(2).tolist(), widths=s[j].round(2).tolist(),
                    halflife_h=float(model.halflife()[j]))
            for j, f in enumerate(DYNAMIC)}
