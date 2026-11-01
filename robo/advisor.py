"""Orchestrates the full advice pipeline and writes the client-facing advice text.

KYC profile -> risk profile -> goal buckets + retirement glide path -> ETF implementation
and asset location -> suitability-gated overlays -> Monte Carlo projections -> plan levers.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cma import CMA, load_cma, portfolio_stats, stress_returns
from .derivatives import Market
from .goals import cashflow_schedule, goal_equity_path, planning_age, social_security
from .kyc import ClientProfile
from .portfolio import TIER_NAMES, blend_for_equity, load_models, tier_for_equity
from .retirement import (claim_factor, estimate_pia, full_retirement_age, retirement_contribution_cap,
                         rmd_start_age, birth_year, SS_DEPLETION_YEAR, SS_PAYABLE_AFTER)
from .risk import RiskProfile, assess, retirement_glidepath
from .simulate import SimConfig, SimResult, fund_goals, run, sustainable_spending
from .suitability import Eligibility, OverlayPlan, eligibility, recommend_overlays, reg_bi_checklist

# Approximate expense ratios of the primary ETFs (issuer sites, 2025); used for the cost line only.
EXPENSE_RATIO = {"SGOV": 0.0009, "VGIT": 0.0003, "SCHP": 0.0003, "BND": 0.0003, "VCIT": 0.0003,
                 "USHY": 0.0008, "VGLT": 0.0003, "VOO": 0.0003, "VB": 0.0005, "VEA": 0.0003,
                 "VWO": 0.0007, "VNQ": 0.0013, "PDBC": 0.0059, "IAU": 0.0025, "DBMF": 0.0085}
# Asset location: higher = less tax-efficient (belongs in tax-deferred accounts first)
TAX_INEFFICIENCY = {"us_hy": 10, "us_ig": 9, "reit": 9, "tips": 8, "us_agg": 8, "ust_long": 7, "ust_int": 7,
                    "commod": 7, "mgd_fut": 7, "gold": 6, "cash": 5, "em": 3, "us_sc": 2, "intl_dev": 1,
                    "us_lc": 0}
INCOME_RESERVE_YEARS = 5


@dataclass
class Recommendation:
    profile: ClientProfile
    risk: RiskProfile
    elig: Eligibility
    overlays: OverlayPlan
    glide: np.ndarray
    schedule: pd.DataFrame
    buckets: pd.DataFrame
    weights_now: pd.Series                 # household allocation (fraction of investable)
    implementation: pd.DataFrame
    location: pd.DataFrame
    stats: dict
    stress: dict
    sim: SimResult
    sim_life: SimResult
    sim_overlay: SimResult | None
    sustainable: float
    levers: pd.DataFrame | None
    advice: list[str]
    checklist: list
    market: Market
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------- building blocks
def build_buckets(p: ClientProfile, rp: RiskProfile, glide: np.ndarray, sched: pd.DataFrame,
                  cma: CMA) -> tuple[pd.DataFrame, pd.Series]:
    models = load_models()
    funding, left_taxable = fund_goals(p, sched)
    T = len(sched) - 1
    rows, total_w = [], pd.Series(0.0, index=cma.keys)
    for g in p.goals:
        if g.kind in ("retirement", "bequest"):
            continue
        e0 = float(goal_equity_path(g, rp.recommended_equity, T)[0])
        w = blend_for_equity(e0, models)
        amt = funding[g.name]["fund"]
        total_w += w * amt
        rows.append({"Bucket": g.name, "Priority": g.priority, "Horizon (yrs)": g.years_until,
                     "Target (today's $)": g.amount, "Funded today": amt,
                     "Annual top-up": funding[g.name]["sink"], "Equity now": e0,
                     "Model": TIER_NAMES[tier_for_equity(e0)]})
    ret_assets = left_taxable + p.finances.traditional + p.finances.roth
    e_ret = float(glide[0])
    retired = p.household.age >= p.finances.retirement_age
    if retired and ret_assets > 0:
        net_need = max(float(sched["retirement_spending"].iloc[0] - sched["ss_client"].iloc[0]
                             - sched["ss_spouse"].iloc[0] - sched["pension"].iloc[0]), 0.0)
        reserve = min(INCOME_RESERVE_YEARS * net_need, ret_assets * (1 - e_ret / 0.98))
        growth = ret_assets - reserve
        e_growth = min(e_ret * ret_assets / max(growth, 1), 0.98)
        w_res, w_gro = blend_for_equity(0.0, models), blend_for_equity(e_growth, models)
        total_w += w_res * reserve + w_gro * growth
        rows.append({"Bucket": f"Income reserve ({INCOME_RESERVE_YEARS} yrs of withdrawals)", "Priority": "need",
                     "Horizon (yrs)": 0, "Target (today's $)": reserve, "Funded today": reserve,
                     "Annual top-up": 0.0, "Equity now": 0.0, "Model": TIER_NAMES[0]})
        rows.append({"Bucket": "Long-term growth (spending + legacy)", "Priority": "want",
                     "Horizon (yrs)": T, "Target (today's $)": np.nan, "Funded today": growth,
                     "Annual top-up": 0.0, "Equity now": e_growth, "Model": TIER_NAMES[tier_for_equity(e_growth)]})
    else:
        total_w += blend_for_equity(e_ret, models) * ret_assets
        rows.append({"Bucket": "Retirement (401k / IRA / Roth + taxable)", "Priority": "need",
                     "Horizon (yrs)": max(p.finances.retirement_age - p.household.age, 0),
                     "Target (today's $)": p.finances.retirement_spending, "Funded today": ret_assets,
                     "Annual top-up": float(sched["employee_savings"].iloc[0] + sched["employer_match"].iloc[0]
                                            - sum(v["sink"] for v in funding.values())),
                     "Equity now": e_ret, "Model": TIER_NAMES[tier_for_equity(e_ret)]})
    inv = max(p.finances.investable, 1)
    return pd.DataFrame(rows), total_w / inv


def implementation_table(w: pd.Series, investable: float, cma: CMA, prices: dict | None = None) -> pd.DataFrame:
    prices = prices or {}
    rows = []
    for k, wt in w.items():
        if wt < 1e-4:
            continue
        i = cma.idx(k)
        etf = cma.etf[i]
        amt = wt * investable
        rows.append({"Asset class": cma.names[i], "ETF": etf, "Alternative": cma.alt_etf[i], "Weight": wt,
                     "Amount": amt, "Price": prices.get(etf, np.nan),
                     "Shares": np.floor(amt / prices[etf]) if etf in prices and prices[etf] > 0 else np.nan,
                     "Expense ratio": EXPENSE_RATIO.get(etf, np.nan), "Key": k})
    return pd.DataFrame(rows).sort_values("Weight", ascending=False).reset_index(drop=True)


def asset_location(w_ret: pd.Series, p: ClientProfile, left_taxable: float, cma: CMA) -> pd.DataFrame:
    """Greedy asset location: Roth gets the highest-expected-return assets (tax-free growth),
    traditional gets the least tax-efficient assets (bonds, REITs, HY), taxable gets the rest
    (broad equity: qualified dividends, foreign tax credit, step-up at death)."""
    f = p.finances
    total = left_taxable + f.traditional + f.roth
    dollars = (w_ret * total).to_dict()
    cap = {"Roth": f.roth, "Traditional": f.traditional, "Taxable": left_taxable}
    alloc = {a: {k: 0.0 for k in dollars} for a in cap}
    mu = dict(zip(cma.keys, cma.mu_geo))
    for acct, order in (("Roth", sorted(dollars, key=lambda k: -mu[k])),
                        ("Traditional", sorted(dollars, key=lambda k: -TAX_INEFFICIENCY[k]))):
        room = cap[acct]
        for k in order:
            put = min(room, dollars[k])
            alloc[acct][k] += put; dollars[k] -= put; room -= put
            if room <= 1e-6:
                break
    for k in dollars:
        alloc["Taxable"][k] += dollars[k]
    df = pd.DataFrame(alloc)
    df.index = [cma.names[cma.idx(k)] for k in df.index]
    df = df[(df.sum(axis=1) > 1)]
    return df


def plan_levers(p: ClientProfile, rp: RiskProfile, cfg: SimConfig, cma: CMA) -> pd.DataFrame:
    """Success probability under the standard levers for an underfunded plan."""
    f, h = p.finances, p.household
    variants = {"Current plan": lambda q: q}

    def save_more(q):
        q.finances.savings_rate = min(f.savings_rate + 0.08, retirement_contribution_cap(h.age) / max(f.household_income, 1))
        return q

    def retire_later(q):
        q.finances.retirement_age = f.retirement_age + 3
        for g in q.goals:
            if g.kind == "retirement":
                g.years_until += 3
        return q

    def claim_70(q):
        if not f.already_claiming_ss:
            fra = full_retirement_age(h.age)
            if q.finances.social_security is None:
                q.finances.social_security = estimate_pia(f.annual_income) * claim_factor(70, fra)
            else:
                q.finances.social_security *= claim_factor(70, fra) / claim_factor(f.ss_claim_age, fra)
            q.finances.ss_claim_age = 70
        return q

    def spend_less(q):
        q.finances.retirement_spending *= 0.9
        for g in q.goals:
            if g.kind == "retirement":
                g.amount *= 0.9
        return q

    if h.age < f.retirement_age:
        variants[f"Save 8 pts more of income ({f.savings_rate:.0%} -> {min(f.savings_rate + .08, 1):.0%})"] = save_more
        variants[f"Retire 3 years later (at {f.retirement_age + 3})"] = retire_later
        if not f.already_claiming_ss and f.ss_claim_age < 70:
            variants["Claim Social Security at 70"] = claim_70
    variants["Spend 10% less in retirement"] = spend_less
    if h.age < f.retirement_age:
        variants["All of the above"] = lambda q: spend_less(claim_70(retire_later(save_more(q))))

    rows = []
    for name, fn in variants.items():
        q = fn(copy.deepcopy(p))
        rq = assess(q, cma, cfg.apply_ss_shortfall)
        r = run(q, rq, cfg, cma)
        rows.append({"Lever": name, "P(success, plan to age 95)": r.success,
                     "Median wealth at retirement": float(np.nanmedian(r.retire_wealth)),
                     "Recommended equity": rq.recommended_equity})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- orchestration
def advise(p: ClientProfile, market: Market | None = None, cfg: SimConfig | None = None,
           cma: CMA | None = None, prices: dict | None = None, with_levers: bool = True,
           with_overlay_sim: bool = True) -> Recommendation:
    cma = cma or load_cma()
    market = market or Market()
    cfg = cfg or SimConfig()
    rp = assess(p, cma, cfg.apply_ss_shortfall)
    el = eligibility(p, rp)
    sched = cashflow_schedule(p, cfg.apply_ss_shortfall)
    glide = retirement_glidepath(p, rp, cma, cfg.apply_ss_shortfall)
    buckets, w_now = build_buckets(p, rp, glide, sched, cma)
    impl = implementation_table(w_now, p.finances.investable, cma, prices)
    _, left_tax = fund_goals(p, sched)
    w_ret = pd.Series(blend_for_equity(float(glide[0])), index=cma.keys)
    loc = asset_location(w_ret, p, left_tax, cma)
    us_lc_value = float(w_now["us_lc"] * p.finances.investable)
    T = len(sched) - 1
    plan = recommend_overlays(p, rp, el, market, us_lc_value, horizon_t=T)

    sim_cfg = copy.copy(cfg); sim_cfg.mortality = False; sim_cfg.overlays = []
    sim = run(p, rp, sim_cfg, cma, glide=glide)
    life_cfg = copy.copy(cfg); life_cfg.mortality = True; life_cfg.overlays = []
    sim_life = run(p, rp, life_cfg, cma, glide=glide)
    sim_ov = None
    if with_overlay_sim and any(o.kind != "csp" for o, _, _ in plan.overlays):
        ov_cfg = copy.copy(sim_cfg); ov_cfg.overlays = plan.overlays
        sim_ov = run(p, rp, ov_cfg, cma, glide=glide)
    sust = sustainable_spending(p, rp, 0.85, SimConfig(n_paths=min(cfg.n_paths, 1500), seed=cfg.seed,
                                                       spending_rule=cfg.spending_rule,
                                                       apply_ss_shortfall=cfg.apply_ss_shortfall), cma)
    levers = None
    if with_levers and (sim.success < 0.85 or (rp.need_equity is None) or
                        (rp.need_equity is not None and rp.need_equity > rp.recommended_equity + 0.05)):
        lev_cfg = SimConfig(n_paths=min(cfg.n_paths, 1500), seed=cfg.seed, spending_rule=cfg.spending_rule,
                            apply_ss_shortfall=cfg.apply_ss_shortfall)
        levers = plan_levers(p, rp, lev_cfg, cma)

    w_arr = w_now.to_numpy(float)
    stats = portfolio_stats(w_arr, cma)
    stats["expense_ratio"] = float(np.nansum(impl["Weight"] * impl["Expense ratio"]))
    stats["equity"] = float(w_now[[k for k in cma.keys if k in ("us_lc", "us_sc", "intl_dev", "em", "reit")]].sum())
    rec = Recommendation(profile=p, risk=rp, elig=el, overlays=plan, glide=glide, schedule=sched,
                         buckets=buckets, weights_now=w_now, implementation=impl, location=loc, stats=stats,
                         stress=stress_returns(w_arr, cma), sim=sim, sim_life=sim_life, sim_overlay=sim_ov,
                         sustainable=sust, levers=levers, advice=[], checklist=reg_bi_checklist(p, rp, el),
                         market=market)
    rec.advice = write_advice(rec)
    return rec


# ---------------------------------------------------------------- narrative
def _m(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


def write_advice(r: Recommendation) -> list[str]:
    p, rp, f, h = r.profile, r.risk, r.profile.finances, r.profile.household
    out = []
    emerg = f.cash / max(f.monthly_essential_expenses, 1)
    out.append(f"**Liquidity first.** Keep {_m(f.cash)} in bank cash outside the portfolio "
               f"({emerg:.1f} months of essential expenses; target 3-6).")
    for _, b in r.buckets.iterrows():
        if b["Bucket"].startswith(("Retirement", "Income reserve", "Long-term")):
            continue
        prob = r.sim.goal_success.get(b["Bucket"], np.nan)
        top = f" plus {_m(b['Annual top-up'])} a year" if b["Annual top-up"] > 0 else ""
        if b["Equity now"] <= 0.10:
            mix = (f"({b['Equity now']:.0%} equity: T-bills, short Treasuries and TIPS). Money needed within "
                   "about 2 years should not be in stocks")
        else:
            mix = (f"({b['Equity now']:.0%} equity today, stepping down each year so the bucket is in cash and "
                   "short bonds by the time the money is needed)")
        out.append(f"**{b['Bucket']} ({b['Priority']}, {b['Horizon (yrs)']} yrs).** Fund {_m(b['Funded today'])} "
                   f"today{top} in the *{b['Model']}* model {mix}. Probability the goal is fully funded: {prob:.0%}.")
    ret_t = f.retirement_age - h.age
    g = r.glide
    if ret_t > 0:
        at_ret = g[min(ret_t, len(g) - 1)]
        later = g[min(ret_t + 15, len(g) - 1)]
        out.append(f"**Retirement portfolio.** Invest retirement money in the *{rp.tier_name}* model "
                   f"({g[0]:.0%} equity), gliding to {at_ret:.0%} at retirement (age {f.retirement_age}) and "
                   f"{later:.0%} by age {f.retirement_age + 15}. Binding constraint: {rp.binding}.")
        contrib = r.schedule["employee_savings"].iloc[0] + r.schedule["employer_match"].iloc[0]
        out.append(f"**Contributions.** Save {f.savings_rate:.0%} of pay plus the {f.employer_match:.0%} employer "
                   f"match ({_m(contrib)} a year). Never leave the match unclaimed; {f.contrib_to_roth_share:.0%} goes "
                   f"to Roth (tax diversification), and anything above the 2026 limits goes to the taxable account.")
        out.append(f"**Projection.** Median portfolio at retirement {_m(np.nanmedian(r.sim.retire_wealth))} "
                   f"(10th percentile {_m(np.nanpercentile(r.sim.retire_wealth, 10))}) in today's dollars. "
                   f"Probability of funding {_m(f.retirement_spending)}/yr to age {planning_age(p)}: "
                   f"{r.sim.success:.0%} ({r.sim.config.spending_rule} spending); over a simulated lifetime: "
                   f"{r.sim_life.success:.0%}. Spending sustainable with 85% confidence: {_m(r.sustainable)}/yr.")
    else:
        out.append(f"**Retirement portfolio.** Hold {g[0]:.0%} equity overall, split into an income reserve "
                   f"({INCOME_RESERVE_YEARS} years of portfolio withdrawals in T-bills, short Treasuries and TIPS) "
                   f"and a long-term growth bucket. Equity drifts to {g[min(15, len(g) - 1)]:.0%} by age {h.age + 15}.")
        out.append(f"**Projection.** Probability of funding {_m(f.retirement_spending)}/yr to age {planning_age(p)}: "
                   f"{r.sim.success:.0%}; over simulated lifetimes of both spouses: {r.sim_life.success:.0%}. "
                   f"Median estate at second death {_m(np.median(r.sim_life.terminal))}. Spending sustainable "
                   f"with 85% confidence: {_m(r.sustainable)}/yr.")
    ss1, ss2, _ = social_security(p)
    if not f.already_claiming_ss:
        fra = full_retirement_age(h.age)
        at70 = ss1 / claim_factor(f.ss_claim_age, fra) * claim_factor(70, fra)
        out.append(f"**Social Security.** Estimated {_m(ss1)}/yr at {f.ss_claim_age} (today's $); waiting to 70 "
                   f"raises it to {_m(at70)}/yr. The projection assumes the 2026 Trustees' {SS_PAYABLE_AFTER:.0%} "
                   f"payable rate from {SS_DEPLETION_YEAR}." if r.sim.config.apply_ss_shortfall else
                   f"**Social Security.** Estimated {_m(ss1)}/yr at {f.ss_claim_age}; {_m(at70)}/yr if claimed at 70.")
    if r.levers is not None:
        best = r.levers.sort_values("P(success, plan to age 95)", ascending=False).iloc[0]
        base = r.levers.iloc[0]["P(success, plan to age 95)"]
        out.append(f"**The plan needs to change, not the risk level.** Goals need about "
                   f"{(rp.need_real_return or 0):.1%} real a year, more than the {rp.recommended_equity:.0%}-equity "
                   f"portfolio the client accepts. Success today is {base:.0%}; the strongest lever is "
                   f"'{best['Lever']}' ({best['P(success, plan to age 95)']:.0%}). See the levers table.")
    rmd_age = rmd_start_age(h.age)
    if f.traditional > 0 and h.age < rmd_age + 5:
        yr = birth_year(h.age) + rmd_age
        if h.age >= 60:
            out.append(f"**Taxes.** RMDs start at {rmd_age} ({yr}). Use the years before then for partial Roth "
                       f"conversions up to the top of the current bracket, and give to charity by QCD from the IRA "
                       f"(allowed from age 70.5). Withdraw taxable first, then traditional, then Roth.")
        else:
            out.append(f"**Taxes.** Hold bonds and REITs in the 401(k)/IRA, the highest-growth assets in the Roth, "
                       f"and broad equity ETFs in the taxable account (see asset location). RMDs start at {rmd_age} ({yr}).")
    for _, why in enumerate(r.overlays.rationale):
        out.append(f"**Derivatives.** {why}")
    if not r.overlays.overlays and not r.overlays.futures:
        out.append(f"**Derivatives.** Options level {r.elig.options_level}; futures "
                   f"{'approved' if r.elig.futures_ok else 'not approved'}. {r.overlays.not_recommended[0]}")
    for fl in rp.flags:
        out.append(f"**Flag.** {fl}")
    return out
