"""Risk profiling: willingness (tolerance), ability (capacity) and need.

1. Willingness  - psychometric questionnaire score -> CRRA gamma, cross-checked with the
                  Barsky-Juster-Kimball-Shapiro (1997) income gamble; plus the stated maximum
                  one-year loss, enforced against 2008 and 2022 stress scenarios.
2. Ability      - household balance sheet (Merton 1969 with human capital; Bodie, Merton &
                  Samuelson 1992; Ibbotson et al. 2007): future savings and guaranteed income
                  (Social Security, pensions) are bond-like assets, so the financial portfolio
                  can hold more equity when they are large relative to financial wealth.
                  A capacity score (horizon, funded status, liquidity, income stability,
                  leverage) caps the result.
3. Need         - real return required to fund the goals (IRR of the cash-flow plan).

Recommended long-horizon equity = min(balance-sheet Merton share, stress-loss cap, capacity cap).
Need never raises risk above that; a gap triggers advice to change the plan instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .cma import CMA, EQUITY_KEYS, load_cma, portfolio_stats, stress_returns
from .goals import cashflow_schedule, goal_columns, liability_duration, net_flows, pv
from .kyc import MAX_LOSS_MAP, ClientProfile, gamble_gamma, questionnaire_score
from .portfolio import TIER_NAMES, load_models, models_summary, tier_equity, tier_for_equity

REAL_SAFE_RATE = 0.019        # TIPS-like real yield: JPM 10y UST 4.6% less 2.5% inflation (~2%)
STABILITY = {  # (equity beta of labour income, extra discount spread, capacity points)
    "very stable": (0.05, 0.005, 100), "stable": (0.15, 0.015, 75),
    "variable": (0.35, 0.03, 45), "volatile": (0.60, 0.05, 20),
}
GAMMA_MAX, GAMMA_MIN = 12.0, 1.5
CAPACITY_WEIGHTS = {"Horizon (liability duration)": .35, "Funded status": .25,
                    "Liquidity (emergency fund)": .15, "Income stability / guaranteed income": .15,
                    "Leverage": .10}


@dataclass
class RiskProfile:
    tolerance_score: float
    n_answered: int
    gamma_questionnaire: float
    gamma_gamble: float | None
    gamble_note: str
    gamma: float
    max_loss: float
    merton_equity: float              # financial wealth only
    human_capital: float              # PV future portfolio contributions
    guaranteed_pv: float              # PV Social Security + pension
    beta_hc: float
    hc_equity: float                  # balance-sheet adjusted
    drawdown_equity: float            # highest model whose worst stress loss <= stated max loss
    capacity_score: float
    capacity_parts: dict
    capacity_equity: float
    need_real_return: float | None
    need_equity: float | None
    funded_ratio: float
    recommended_equity: float
    tier: int
    tier_name: str
    binding: str
    flags: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("capacity_parts", "flags")}


# ---------------------------------------------------------------- willingness
def gamma_from_score(score: float) -> float:
    """Log-linear map: score 0 -> gamma 12 (very averse), 50 -> ~4.2, 100 -> 1.5."""
    return float(np.exp(np.log(GAMMA_MAX) - (np.log(GAMMA_MAX) - np.log(GAMMA_MIN)) * score / 100))


def equity_sleeve_stats(cma: CMA) -> tuple[float, float]:
    """Arithmetic excess return and volatility of the policy equity mix (tier-10 equity sleeve)."""
    w = load_models().loc[10].to_numpy(float) * cma.mask(EQUITY_KEYS)
    s = portfolio_stats(w / w.sum(), cma)
    return s["exp_return_arith"] - cma.cash_rate, s["vol"]


def merton_share(gamma: float, cma: CMA) -> float:
    erp, sig = equity_sleeve_stats(cma)
    return erp / (gamma * sig ** 2)


def model_stress_table(cma: CMA | None = None) -> pd.DataFrame:
    cma = cma or load_cma()
    rows = {t: stress_returns(w.to_numpy(float), cma) for t, w in load_models().iterrows()}
    df = pd.DataFrame(rows).T
    df["worst"] = df.min(axis=1)
    df.index.name = "tier"
    return df


def stress_cap_equity(max_loss: float, cma: CMA) -> float:
    """Highest model equity whose worst historical-stress loss is within the stated max loss."""
    st = model_stress_table(cma)
    ok = st[st["worst"] >= -max_loss - 1e-9]
    return float(max(tier_equity(int(t)) for t in ok.index)) if len(ok) else 0.0


# ---------------------------------------------------------------- ability
def capacity_parts(p: ClientProfile, df: pd.DataFrame, funded: float, t: int = 0) -> dict:
    """Capacity components (0-100) evaluated t years from now (horizon/stability evolve)."""
    f = p.finances
    sub = df.iloc[t:].reset_index(drop=True)
    dur = liability_duration(sub, p)
    retired = p.identity.employment_status == "retired" or (p.household.age + t) >= f.retirement_age
    if retired:
        essential = f.retirement_spending * f.essential_share_of_spending
        g0 = float((sub["ss_client"] + sub["ss_spouse"] + sub["pension"]).iloc[0]) if len(sub) else 0.0
        stab = float(np.clip(100 * g0 / max(essential, 1), 0, 100))
    else:
        stab = float(STABILITY[f.income_stability][2])
    emerg = f.cash / max(f.monthly_essential_expenses, 1)
    lev = f.other_debt / max(f.liquid_net_worth + f.other_debt, 1)
    return {
        "Horizon (liability duration)": float(np.clip(dur / 20, 0, 1) * 100),
        "Funded status": float(np.clip((funded - 0.8) / 0.7, 0, 1) * 100),
        "Liquidity (emergency fund)": float(np.clip(emerg / 6, 0, 1) * 100),
        "Income stability / guaranteed income": stab,
        "Leverage": float(np.clip(1 - lev / 0.5, 0, 1) * 100),
    }


def capacity_cap(parts: dict) -> tuple[float, float]:
    score = float(sum(parts[k] * w for k, w in CAPACITY_WEIGHTS.items()))
    return score, float(min(1.0, 0.10 + 0.90 * score / 100))


def _required_return(flows: np.ndarray, assets: float) -> float | None:
    """Real return r at which assets + PV(net flows) = 0. None if unreachable below 20%."""
    f = lambda r: assets + pv(flows, r)
    lo, hi = -0.06, 0.20
    if f(lo) >= 0:
        return lo
    if f(hi) < 0:
        return None
    return float(brentq(f, lo, hi))


def _balance_sheet(p: ClientProfile, df: pd.DataFrame) -> tuple[float, float, float, float]:
    f = p.finances
    _, spread, _ = STABILITY[f.income_stability]
    contrib = (df["employee_savings"] + df["employer_match"]).to_numpy(float)
    H = pv(contrib, REAL_SAFE_RATE + spread) - contrib[0]
    G = pv((df["ss_client"] + df["ss_spouse"] + df["pension"]).to_numpy(float), REAL_SAFE_RATE)
    outflows = df["retirement_spending"].to_numpy(float).copy()
    for c in goal_columns(df):
        outflows += df[c].to_numpy(float)
    pv_out = pv(outflows, REAL_SAFE_RATE)
    F = max(f.investable, 1.0)
    funded = (F + pv(contrib, REAL_SAFE_RATE) + G) / pv_out if pv_out > 0 else 9.99
    return F, H, G, funded


# ---------------------------------------------------------------- assessment
def assess(p: ClientProfile, cma: CMA | None = None, apply_ss_shortfall: bool = True) -> RiskProfile:
    cma = cma or load_cma()
    f = p.finances
    flags: list[str] = []

    score, n = questionnaire_score(p.risk_answers)
    g_q = gamma_from_score(score)
    g_g, g_note = gamble_gamma(p.gamble_answers)
    # questionnaire weighted 2/3: psychometric scales are more reliable than one hypothetical gamble
    gamma = float(np.exp((2 * np.log(g_q) + np.log(g_g)) / 3)) if g_g else g_q
    if g_g and abs(np.log(g_q / g_g)) > np.log(2.5):
        flags.append("Questionnaire and income-gamble answers disagree strongly; an advisor should "
                     "review the risk answers with the client before investing.")
    max_loss = MAX_LOSS_MAP[p.risk_answers["q5_maxloss"]] if "q5_maxloss" in p.risk_answers else 0.20
    e_merton = merton_share(gamma, cma)

    df = cashflow_schedule(p, apply_ss_shortfall)
    beta_h = STABILITY[f.income_stability][0]
    F, H, G, funded = _balance_sheet(p, df)
    e_hc = float(np.clip((e_merton * (F + H + G) - beta_h * H) / F, 0.0, 1.0))

    parts = capacity_parts(p, df, funded)
    cap_score, e_capacity = capacity_cap(parts)
    e_draw = stress_cap_equity(max_loss, cma)

    flows = net_flows(df, p)
    r_need = _required_return(flows, F)
    summ = models_summary(cma)
    e_need = None
    if r_need is None:
        flags.append("Goals cannot be funded at any plausible return: the plan must change "
                     "(save more, retire later, or reduce goals).")
    else:
        ok = summ[summ["real_return_geo"] >= r_need]
        if len(ok):
            e_need = float(ok["equity"].min())
        else:
            flags.append(f"Goals need a {r_need:.1%} real return, above even the most aggressive model "
                         f"({summ['real_return_geo'].max():.1%}); investment risk alone cannot close the gap.")

    cands = {"balance-sheet willingness (Merton + human capital)": e_hc,
             "stated maximum loss (stress test)": e_draw, "risk capacity": e_capacity}
    binding = min(cands, key=cands.get)
    e_rec = float(min(cands.values()))
    tier = tier_for_equity(e_rec)

    if r_need is None or (e_need is not None and e_need > e_rec + 0.05) or (r_need is not None and e_need is None):
        need_txt = f"{e_need:.0%} equity" if e_need is not None else "more than any model offers"
        flags.append(f"Risk need ({need_txt}) exceeds the risk the client is willing and able to take "
                     f"({e_rec:.0%}). We do not raise risk to close the gap; see the plan levers.")
    if e_need is not None and e_need < e_rec - 0.30 and funded > 1.2:
        flags.append(f"Goals are well funded: a {e_need:.0%}-equity portfolio would meet them. The client "
                     "could take less risk and lock in the plan (liability matching).")
    emerg = f.cash / max(f.monthly_essential_expenses, 1)
    if emerg < 3:
        flags.append(f"Emergency fund covers {emerg:.1f} months of essential expenses; build it to "
                     "3-6 months in cash before investing new money.")
    if f.other_debt > 0.25 * max(f.household_income, 1):
        flags.append("High non-mortgage debt: paying down debt above ~7% interest beats expected portfolio returns.")
    if f.concentrated_stock > 0.10 * max(f.liquid_net_worth, 1):
        flags.append(f"Single-stock position is {f.concentrated_stock / f.liquid_net_worth:.0%} of liquid "
                     "net worth (above the 10% concentration threshold).")
    if e_hc > e_rec + 0.20:
        flags.append("Future savings and Social Security would justify more equity than the client's "
                     f"stated limits allow ({e_hc:.0%} vs {e_rec:.0%}); the recommendation respects the limits.")

    return RiskProfile(
        tolerance_score=score, n_answered=n, gamma_questionnaire=g_q, gamma_gamble=g_g, gamble_note=g_note,
        gamma=gamma, max_loss=max_loss, merton_equity=float(e_merton), human_capital=float(H),
        guaranteed_pv=float(G), beta_hc=beta_h, hc_equity=e_hc, drawdown_equity=e_draw,
        capacity_score=cap_score, capacity_parts=parts, capacity_equity=e_capacity,
        need_real_return=r_need, need_equity=e_need, funded_ratio=float(funded),
        recommended_equity=e_rec, tier=tier, tier_name=TIER_NAMES[tier], binding=binding, flags=flags,
    )


def retirement_glidepath(p: ClientProfile, rp: RiskProfile, cma: CMA | None = None,
                         apply_ss_shortfall: bool = True) -> np.ndarray:
    """Equity share of the retirement sleeve for each future year (t = 0..T).

    Each year the balance-sheet Merton share and the capacity cap are recomputed along the
    expected path. As future savings (human capital) turn into financial wealth and the
    horizon shortens, equity falls: the household's own numbers produce the glide path
    instead of a generic target-date curve. The stated stress-loss limit applies throughout;
    equity is floored at 20% (inflation protection over long retirements), held at its
    lowest level from 5 years before to 5 years after retirement (sequence-risk zone), never
    rises again after the zone starts, and never moves more than 5 points a year.
    """
    cma = cma or load_cma()
    f = p.finances
    df = cashflow_schedule(p, apply_ss_shortfall)
    T = len(df) - 1
    beta_h, spread, _ = STABILITY[f.income_stability]
    contrib = (df["employee_savings"] + df["employer_match"]).to_numpy(float)
    guaranteed = (df["ss_client"] + df["ss_spouse"] + df["pension"]).to_numpy(float)
    spend = df["retirement_spending"].to_numpy(float)
    floor = min(0.20, rp.recommended_equity)

    r_exp = 0.035
    F = max(f.investable, 1.0)
    path = np.zeros(T + 1)
    for t in range(T + 1):
        H = pv(contrib[t + 1:], REAL_SAFE_RATE + spread)
        G = pv(guaranteed[t + 1:], REAL_SAFE_RATE)
        e = (rp.merton_equity * (F + H + G) - beta_h * H) / max(F, 1.0)
        _, cap_t = capacity_cap(capacity_parts(p, df, rp.funded_ratio, t))
        path[t] = np.clip(e, floor, min(rp.drawdown_equity, cap_t))
        F = max(F * (1 + r_exp) + contrib[t] + guaranteed[t] - spend[t], 1.0)

    ret_t = max(f.retirement_age - p.household.age, 0)
    lo, hi = max(ret_t - 5, 0), min(ret_t + 5, T)
    if ret_t <= T:
        path[lo:hi + 1] = path[lo:hi + 1].min()
    path[0] = rp.recommended_equity
    for t in range(1, T + 1):
        path[t] = np.clip(path[t], path[t - 1] - 0.05, path[t - 1] + 0.05)
        if t > lo:   # from the start of the sequence-risk zone on, equity never rises again
            path[t] = min(path[t], path[t - 1])
    return path
