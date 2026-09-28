import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = ("LOW", "NORMAL", "HIGH")
TEMPORAL = ("now", "sustained", "rising")


def _inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def _gauss_partition(x, c, log_s):
    mu = torch.exp(-0.5 * ((x.unsqueeze(-1) - c) / log_s.exp()) ** 2)
    return mu / (mu.sum(-1, keepdim=True) + 1e-8)


class STAF(nn.Module):
    """Staleness-Aware Temporal Fuzzy network.

    Per dynamic feature j: a Gaussian fuzzy partition (LOW/NORMAL/HIGH) of the last observed value is
    blended toward a learned population prior as the observation ages, with reliability
    rho_j = exp(-lambda_j * hours_since_obs). Temporal terms (now / sustained / rising) come from two
    learnable EMAs of those memberships; rho_j ("recently measured") and an EMA of measurement events
    ("frequently measured") are terms too, as are fuzzy partitions of the static covariates.
    The logit is additive in (i) the terms (a fuzzy scorecard, weights v) and (ii) rule firings
    f_r = exp(-sum_i a_ri (1 - T_i)), a soft conjunction where a_ri = 0 means "don't care".
    L1 on a and v keeps both sparse.
    """

    def __init__(self, centers, widths, halflife, static_centers, feat_names, static_names, n_rules=24,
                 staleness=True, temporal=True, measurement_terms=True, hybrid_hidden=0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        J, K = centers.shape
        S = static_centers.shape[0]
        self.J, self.K, self.S = J, K, S
        self.staleness, self.temporal, self.measurement_terms = staleness, temporal, measurement_terms
        self.c = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.log_s = nn.Parameter(torch.log(torch.tensor(widths, dtype=torch.float32)))
        self.lam_raw = nn.Parameter(_inv_softplus(math.log(2) / torch.tensor(halflife, dtype=torch.float32)))
        self.prior_logit = nn.Parameter(torch.zeros(J, K))
        self.slow_raw = nn.Parameter(torch.full((J,), math.log(12.0)))
        self.fast_raw = nn.Parameter(torch.full((J,), math.log(2.0)))
        self.log_kappa = nn.Parameter(torch.full((J, K), math.log(10.0)))
        self.sc = nn.Parameter(torch.tensor(static_centers, dtype=torch.float32))
        self.log_ss = nn.Parameter(torch.full((S, K), math.log(0.8)))

        self.term_names = [f"{f} is {LEVELS[k]} ({TEMPORAL[t]})" for f in feat_names
                           for t in range(3 if temporal else 1) for k in range(K)]
        if measurement_terms:
            self.term_names += [f"{f} recently measured" for f in feat_names]
            self.term_names += [f"{f} frequently measured" for f in feat_names]
        self.term_names += [f"{s} is {LEVELS[k]}" for s in static_names for k in range(K)]
        n_terms = len(self.term_names)

        W = torch.full((n_rules, n_terms), -6.0)
        for r in range(n_rules):
            W[r, torch.randperm(n_terms, generator=g)[:2]] = 2.0
        self.W = nn.Parameter(W)
        self.beta = nn.Parameter(torch.zeros(n_rules))
        self.v = nn.Parameter(torch.zeros(n_terms))
        self.bias = nn.Parameter(torch.tensor(-4.0))
        self.hybrid = None
        if hybrid_hidden:
            self.hybrid = nn.GRU(3 * J + S, hybrid_hidden, batch_first=True)
            self.hybrid_out = nn.Linear(hybrid_hidden, 1)
            nn.init.zeros_(self.hybrid_out.weight)
            nn.init.zeros_(self.hybrid_out.bias)

    def antecedents(self):
        return F.softplus(self.W)

    def halflife(self):
        return math.log(2) / F.softplus(self.lam_raw)

    def reliability(self, delta, seen):
        if not self.staleness:
            return seen
        return seen * torch.exp(-F.softplus(self.lam_raw) * delta)

    def _ema(self, x, a, init):
        s, out = init, []
        for t in range(x.shape[1]):
            s = a * s + (1 - a) * x[:, t]
            out.append(s)
        return torch.stack(out, 1)

    def terms(self, z, delta, seen, static):
        B, T, J = z.shape
        rho = self.reliability(delta, seen)
        prior = torch.softmax(self.prior_logit, -1)
        m = rho.unsqueeze(-1) * _gauss_partition(z, self.c, self.log_s) + (1 - rho.unsqueeze(-1)) * prior
        parts = [m]
        if self.temporal:
            a_s = torch.exp(-1.0 / self.slow_raw.exp()).view(1, J, 1)
            a_f = torch.exp(-1.0 / self.fast_raw.exp()).view(1, J, 1)
            init = prior.expand(B, -1, -1)
            S, Fs = self._ema(m, a_s, init), self._ema(m, a_f, init)
            parts += [S, torch.sigmoid(self.log_kappa.exp() * (Fs - S) - 3.0)]
        out = [torch.cat(parts, -1).flatten(2)]
        if self.measurement_terms:
            event = seen * (delta == 0).float()
            a_s = torch.exp(-1.0 / self.slow_raw.exp()).view(1, J)
            out += [rho, self._ema(event, a_s, torch.zeros(B, J))]
        out.append(_gauss_partition(static, self.sc, self.log_ss).flatten(2))
        return torch.cat(out, -1), rho

    def forward(self, z, delta, seen, static, return_parts=False):
        T, rho = self.terms(z, delta, seen, static)
        fire = torch.exp(-(1 - T) @ self.antecedents().T)
        logit = self.bias + T @ self.v + fire @ self.beta
        if self.hybrid is not None:
            h, _ = self.hybrid(torch.cat([z, rho, seen, static], -1))
            logit = logit + self.hybrid_out(h).squeeze(-1)
        if return_parts:
            return logit, T, fire
        return logit

    def penalty(self):
        rules = self.antecedents().sum(-1).mean() if self.W.shape[0] else 0.0
        return rules + self.v.abs().sum()


class GRUBaseline(nn.Module):
    """GRU over [value, observed-ever mask, log time-since-observation, static] (GRU-simple style)."""

    def __init__(self, n_feat, n_static, hidden=64):
        super().__init__()
        self.gru = nn.GRU(3 * n_feat + n_static, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, z, delta, seen, static):
        h, _ = self.gru(torch.cat([z, seen, torch.log1p(torch.clamp(delta, max=72.0)) / 4.3, static], -1))
        return self.out(h).squeeze(-1)

    def penalty(self):
        return torch.tensor(0.0)


def init_from_data(z_train, seen_train, delta_train, static_train, K=3):
    """Centers at 10/50/90% quantiles of observed normalized values; half-life = 2x mean sampling gap."""
    J = z_train.shape[1]
    centers = np.zeros((J, K), np.float32)
    widths = np.zeros((J, K), np.float32)
    halflife = np.zeros(J, np.float32)
    for j in range(J):
        obs = z_train[(seen_train[:, j] > 0) & (delta_train[:, j] == 0), j]
        if len(obs) < 50:
            centers[j], widths[j], halflife[j] = [-1.0, 0.0, 1.0], 0.8, 24.0
            continue
        q = np.quantile(obs, [0.10, 0.50, 0.90]) + np.array([-1e-2, 0, 1e-2])
        centers[j] = q
        widths[j] = max((q[2] - q[0]) / 2.5, 0.1)
        halflife[j] = float(np.clip(2.0 * (seen_train[:, j] > 0).sum() / len(obs), 1.0, 72.0))
    static_centers = np.quantile(static_train, [0.10, 0.50, 0.90], axis=0).T.astype(np.float32)
    static_centers += np.array([-1e-2, 0, 1e-2], np.float32)
    return centers, widths, halflife, static_centers
