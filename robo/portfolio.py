"""Strategic model portfolios: resampled mean-variance optimisation under an investment
policy (IPS) constraint set, one model per risk tier.

Tier t (0..10) fixes the equity share at e = min(t/10, 0.98). Within that, the optimiser
chooses the equity mix and the defensive/diversifier mix by maximising
    w'mu - (lambda/2) w' Sigma w
subject to policy bounds. Michaud-style resampling: the optimisation is repeated on
expected returns drawn from their estimation-error distribution and the weights are
averaged, which removes the corner solutions plain MVO produces from CMAs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .cma import CMA, EQUITY_KEYS, ALT_KEYS, DATA, load_cma, portfolio_stats

TIERS = list(range(11))
TIER_NAMES = {
    0: "Capital Preservation", 1: "Conservative", 2: "Conservative", 3: "Moderately Conservative",
    4: "Moderately Conservative", 5: "Moderate", 6: "Moderate", 7: "Moderate Growth",
    8: "Growth", 9: "Aggressive Growth", 10: "Maximum Growth",
}
MODEL_FILE = DATA / "model_portfolios.csv"


def tier_equity(t: int) -> float:
    return min(t / 10, 0.98)


def tier_lambda(t: int) -> float:
    """Risk-aversion used inside the optimiser; smooth in the equity share (Merton-consistent)."""
    return float(np.clip(1.9 / (tier_equity(t) + 0.15), 2.0, 12.0))


def tier_for_equity(e: float) -> int:
    return int(np.clip(round(e * 10), 0, 10))


def _bounds(cma: CMA, t: int) -> tuple[list, list]:
    """Per-asset bounds and extra linear inequality constraints (A w >= b) for tier t."""
    e = tier_equity(t)
    ub = {k: 1.0 for k in cma.keys}
    lb = {k: 0.0 for k in cma.keys}
    lb["cash"] = 0.02
    defensive = 1 - e
    allow_alts = 3 <= t <= 9
    allow_credit = t >= 3
    ub.update({
        "us_hy": 0.08 if allow_credit else 0.0,
        "ust_long": 0.08 if allow_credit else 0.0,
        "us_ig": min(0.20, defensive),
        "tips": 0.30 * defensive + 0.02,
        "commod": 0.03 if allow_alts else 0.0,
        "gold": 0.04 if allow_alts else 0.0,
        "mgd_fut": 0.06 if allow_alts else 0.0,
    })
    if t == 0:
        for k in ("us_ig",):
            ub[k] = 0.0
    bnds = [(lb[k], ub[k]) for k in cma.keys]

    eq = cma.mask(EQUITY_KEYS).astype(float)
    i = cma.idx
    cons = [{"type": "eq", "fun": lambda w, eq=eq, e=e: w @ eq - e},
            {"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if e > 0:
        def share(k, lo=None, hi=None):
            if lo is not None:
                cons.append({"type": "ineq", "fun": lambda w, j=i(k), lo=lo, e=e: w[j] - lo * e})
            if hi is not None:
                cons.append({"type": "ineq", "fun": lambda w, j=i(k), hi=hi, e=e: hi * e - w[j]})
        share("us_lc", lo=0.45)
        share("intl_dev", lo=0.20, hi=0.35)
        share("em", hi=0.12)
        share("us_sc", hi=0.10)
        share("reit", hi=0.06)
    if allow_alts:
        alt = cma.mask(ALT_KEYS).astype(float)
        cons.append({"type": "ineq", "fun": lambda w, alt=alt: 0.10 - w @ alt})
    return bnds, cons


def _solve(mu: np.ndarray, cov: np.ndarray, lam: float, bnds, cons, x0) -> np.ndarray:
    obj = lambda w: -(w @ mu - 0.5 * lam * w @ cov @ w)
    jac = lambda w: -(mu - lam * cov @ w)
    res = minimize(obj, x0, jac=jac, bounds=bnds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else x0


def _feasible_start(cma: CMA, t: int, bnds) -> np.ndarray:
    e = tier_equity(t)
    w = np.zeros(cma.n)
    if e > 0:
        mix = {"us_lc": 0.6, "us_sc": 0.05, "intl_dev": 0.25, "em": 0.07, "reit": 0.03}
        for k, s in mix.items():
            w[cma.idx(k)] = s * e
    rest = 1 - e
    w[cma.idx("cash")] = 0.02
    w[cma.idx("us_agg")] = rest - 0.02
    return w


def build_models(cma: CMA | None = None, n_resample: int = 150, t_eff: float = 25.0,
                 seed: int = 7) -> pd.DataFrame:
    cma = cma or load_cma()
    rng = np.random.default_rng(seed)
    mu, cov = cma.mu_arith, cma.cov
    rows = []
    for t in TIERS:
        e = tier_equity(t)
        lam = tier_lambda(t)
        bnds, cons = _bounds(cma, t)
        x0 = _feasible_start(cma, t, bnds)
        base = _solve(mu, cov, lam, bnds, cons, x0)
        draws = rng.multivariate_normal(mu, cov / t_eff, size=n_resample)
        ws = [_solve(m, cov, lam, bnds, cons, base) for m in draws]
        w = np.mean(ws, axis=0)
        w = np.clip(w, 0, None)
        w = np.round(w / w.sum() / 0.005) * 0.005  # implementable: 0.5% increments
        w[cma.idx("cash")] += 1 - w.sum()
        rows.append(pd.Series(w, index=cma.keys, name=t))
    models = pd.DataFrame(rows)
    models.index.name = "tier"
    return models


def load_models(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not MODEL_FILE.exists():
        m = build_models()
        m.to_csv(MODEL_FILE, float_format="%.4f")
        return m
    return pd.read_csv(MODEL_FILE, index_col="tier")


def model_weights(t: int, models: pd.DataFrame | None = None) -> np.ndarray:
    models = load_models() if models is None else models
    return models.loc[int(t)].to_numpy(float)


def blend_for_equity(e: float, models: pd.DataFrame | None = None) -> np.ndarray:
    """Linear interpolation between adjacent tier models for a continuous equity target."""
    models = load_models() if models is None else models
    x = float(np.clip(e, 0, 0.98)) * 10
    lo, hi = int(np.floor(x)), int(min(np.ceil(x), 10))
    if lo == hi:
        return models.loc[lo].to_numpy(float)
    a = x - lo
    return (1 - a) * models.loc[lo].to_numpy(float) + a * models.loc[hi].to_numpy(float)


def models_summary(cma: CMA | None = None, models: pd.DataFrame | None = None) -> pd.DataFrame:
    cma = cma or load_cma()
    models = load_models() if models is None else models
    out = []
    for t, row in models.iterrows():
        s = portfolio_stats(row.to_numpy(float), cma)
        out.append({"tier": t, "name": TIER_NAMES[t], "equity": tier_equity(t), **s})
    return pd.DataFrame(out).set_index("tier")


def efficient_frontier(cma: CMA | None = None, n: int = 30) -> pd.DataFrame:
    """Unconstrained-long-only frontier (for the methodology chart)."""
    cma = cma or load_cma()
    mu, cov = cma.mu_arith, cma.cov
    out = []
    for target in np.linspace(cma.cash_rate + 0.002, mu.max() - 0.002, n):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w, np.full(cma.n, 1 / cma.n), bounds=[(0, 1)] * cma.n,
                       constraints=cons, method="SLSQP")
        if res.success:
            out.append({"exp_return_arith": target, "vol": float(np.sqrt(res.fun))})
    return pd.DataFrame(out)
