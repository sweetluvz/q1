import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def nn_inputs(site, Xf, D, norm_dyn, norm_static):
    """Per-row model inputs: normalized LOCF value (0 if never seen), capped delta, seen flag, static.

    Features never observed in source training (NaN median) are masked as never-seen at any hospital.
    """
    usable = ~np.isnan(norm_dyn.med)
    seen = ((~np.isnan(Xf)) & usable).astype(np.float32)
    z = np.nan_to_num(norm_dyn(Xf), nan=0.0).astype(np.float32) * seen
    delta = np.where(seen > 0, np.minimum(D, 1e3), 1e3).astype(np.float32)
    st = np.column_stack([site["S"], np.log1p(site["iculos"])])
    static = np.nan_to_num(norm_static(st), nan=0.0).astype(np.float32)
    return dict(z=z, delta=delta, seen=seen, static=static, y=site["y"].astype(np.float32),
                offsets=site["offsets"])


def batches(inp, batch_size=128, shuffle=True, rng=None, max_len=None):
    off = inp["offsets"]
    lens = np.diff(off)
    order = np.argsort(lens, kind="stable")
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    for pats in chunks:
        T = int(lens[pats].max()) if max_len is None else min(int(lens[pats].max()), max_len)
        B = len(pats)
        out = {k: np.zeros((B, T) + inp[k].shape[1:], np.float32) for k in ("z", "delta", "seen", "static", "y")}
        mask = np.zeros((B, T), np.float32)
        for i, p in enumerate(pats):
            a, n = off[p], min(lens[p], T)
            for k in out:
                out[k][i, :n] = inp[k][a:a + n]
            mask[i, :n] = 1
        yield pats, {k: torch.from_numpy(v) for k, v in out.items()}, torch.from_numpy(mask)


@torch.no_grad()
def predict_logits(model, inp, batch_size=256):
    model.eval()
    off = inp["offsets"]
    out = np.zeros(off[-1], np.float32)
    for pats, b, mask in batches(inp, batch_size, shuffle=False):
        logit = model(b["z"], b["delta"], b["seen"], b["static"]).numpy()
        for i, p in enumerate(pats):
            n = off[p + 1] - off[p]
            out[off[p]:off[p + 1]] = logit[i, :n]
    return out


def val_logloss(y, logit):
    return float(F.binary_cross_entropy_with_logits(torch.from_numpy(logit), torch.from_numpy(y)).item())


def fit(model, tr, va, epochs=25, lr=1e-2, reg=1e-4, batch_size=32, patience=4, seed=0, max_len=None,
        criterion="auprc", log=print):
    """Train with early stopping on validation AUPRC (higher is better) or log-loss (lower is better)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sign = -1.0 if criterion == "logloss" else 1.0
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, b, mask in batches(tr, batch_size, True, rng, max_len):
            logit = model(b["z"], b["delta"], b["seen"], b["static"])
            loss = (F.binary_cross_entropy_with_logits(logit, b["y"], reduction="none") * mask).sum() / mask.sum()
            obj = loss + reg * model.penalty()
            opt.zero_grad()
            obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * mask.sum().item(); n += mask.sum().item()
        sched.step()
        p = predict_logits(model, va)
        score = val_logloss(va["y"], p) if criterion == "logloss" else average_precision_score(va["y"], p)
        log(f"  ep {ep:02d} loss {tot / n:.4f} val_{criterion} {score:.5f} ({time.time() - t0:.0f}s)")
        if sign * score > best:
            best, best_state, bad = sign * score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, sign * best
