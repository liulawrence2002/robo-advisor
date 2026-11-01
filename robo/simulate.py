"""Monte Carlo projection of accumulation and decumulation (real, today's dollars).

Per path and year:
  returns   correlated asset returns, lognormal with Student-t(5) shocks (fat tails, shocks
            capped at 6 standard deviations), calibrated so the median compound return equals
            the JPM 2026 CMA; net of the weighted ETF expense ratio and the advisory fee.
            Optionally the first years replay a historical episode (2008-09, 2000-02, 2022,
            1973-74) before the stochastic draws resume.
  sleeves   one sleeve per non-retirement goal (its own glide to the goal date) and the
            retirement sleeve (taxable / traditional / Roth accounts, household glide path)
  inflows   employer match -> 401(k) (capped by the IRC 415(c) limit); employee savings ->
            goal sinking funds, then retirement accounts up to 2026 IRS limits (Roth share as
            elected), then taxable
  outflows  goal payments; retirement spending net of Social Security and pension; an
            optional long-term-care shock late in life
  taxes     RMDs (SECURE 2.0 ages, Uniform Lifetime Table) are taken whether or not the client
            still works; ordinary income is taxed with the 2026 federal brackets and standard
            deduction (85% of Social Security included) plus a flat state rate; the taxable
            account pays capital-gains tax on the gain share (basis tracked); withdrawal order
            taxable -> traditional -> Roth; optional Roth conversions to the top of a chosen
            bracket in the years before RMDs
  longevity stochastic death ages for client and spouse (Gompertz fit to SSA 2022 table);
            survivor keeps the larger Social Security benefit, spending drops to 75%
  spending  fixed real, or Guyton-Klinger style guardrails (cut 10% if the withdrawal rate,
            measured on the start-of-year portfolio, rises 20% above its starting level, raise
            10% if it falls 20% below; never below the essential share; no cuts in the last 15
            years of the plan)
  overlays  option/buffer payoffs replace part of the US large-cap return in active years
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cma import CMA, load_cma, stress_path_returns
from .derivatives import Overlay, overlay_return
from .goals import cashflow_schedule, goal_equity_path, planning_age
from .kyc import ClientProfile
from .mortality import sample_death_ages
from .portfolio import blend_for_equity, load_models
from .retirement import rmd_divisor, rmd_start_age
from .risk import RiskProfile, retirement_glidepath
from .tax import SS_INCLUSION, bracket_top, gross_up, incremental_tax, standard_deduction

PCTS = (5, 25, 50, 75, 95)
GUARDRAIL_MIN_WR = 0.04
GUARDRAIL_NO_CUT_YEARS = 15          # Guyton-Klinger: no spending cuts within 15 years of the plan horizon
SHOCK_CAP = 6.0                      # standardized return shocks are capped at +/- 6 sd
MATCH_CAP_415C = 72_000              # 2026 IRC 415(c) limit on total annual additions per person
LTC_ANNUAL_COST = 120_000            # private nursing-home room, today's $ (Genworth 2024 ~$127K)
LTC_YEARS = 3                        # typical paid-care episode
LTC_LIFETIME_PROB = 0.55             # share of 65-year-olds who will need paid long-term care
LTC_INSURED_SHARE = 0.70             # share of the cost a typical LTC policy covers


@dataclass
class SimConfig:
    n_paths: int = 3000
    seed: int = 2026
    fat_tails: bool = True
    t_df: int = 5
    spending_rule: str = "guardrails"         # or "fixed"
    apply_ss_shortfall: bool = True
    mortality: bool = False                   # False: plan to planning age; True: stochastic lifetimes
    survivor_spending: float = 0.75
    overlays: list[tuple[Overlay, int, int]] = field(default_factory=list)   # (overlay, t_start, t_end)
    fee: float = 0.0                          # annual drag: weighted expense ratio + advisory fee
    scenario: str | None = None               # name in cma.STRESS_PATHS replayed as the first years
    roth_conversion_bracket: float | None = None   # e.g. 0.24: convert to the top of the 24% bracket pre-RMD
    ltc_shock: bool = False                   # late-life long-term-care cost shock


@dataclass
class SimResult:
    ages: np.ndarray
    wealth: np.ndarray                        # (n, T+1) total financial wealth at start of each year
    pct: pd.DataFrame                         # wealth percentiles by age
    success: float                            # P(spending never unmet while household alive)
    goal_success: dict[str, float]
    spending: np.ndarray                      # (n, T+1) actual spending
    shortfall_age: np.ndarray                 # age at first shortfall (nan if none)
    terminal: np.ndarray                      # wealth at end (or at household death if mortality)
    retire_wealth: np.ndarray
    cut_prob: float                           # P(spending cut >10% below plan at some point)
    accounts_end: dict[str, float]
    config: SimConfig
    taxes_paid: np.ndarray = None             # (n,) lifetime income tax paid in retirement, today's $
    converted: np.ndarray = None              # (n,) total Roth conversions
    ltc_hit: float = 0.0                      # share of paths with a long-term-care episode
    success_full: float = 0.0                 # P(plan paid in full with no spending cut beyond 10%)
    terminal_after_tax: np.ndarray = None     # estate net of the income tax heirs owe on pre-tax balances


# ---------------------------------------------------------------- returns
def simulate_returns(cma: CMA, n: int, T: int, rng: np.random.Generator, fat_tails: bool = True,
                     t_df: int = 5) -> np.ndarray:
    """Nominal simple returns, shape (n, T, N)."""
    L = np.linalg.cholesky(cma.corr)
    z = rng.standard_normal((n, T, cma.n)) @ L.T
    if fat_tails:
        chi = rng.chisquare(t_df, size=(n, T, 1))
        z = z / np.sqrt(chi / t_df) * np.sqrt((t_df - 2) / t_df)
        z = np.clip(z, -SHOCK_CAP, SHOCK_CAP)      # a 6-sd year is a -60% S&P year; beyond that is noise
    return np.expm1(cma.log_mu + cma.log_sigma * z)


def _weights_path(e_path: np.ndarray, models: pd.DataFrame) -> np.ndarray:
    return np.vstack([blend_for_equity(e, models) for e in e_path])


def fund_goals(p: ClientProfile, sched: pd.DataFrame) -> tuple[dict, float]:
    """Earmark taxable assets for non-retirement goals (needs first) with a margin of safety:
    needs are funded at a 0% real return plus a 3% buffer, wants at 1%, wishes at 2%. Any
    shortfall becomes an annual sinking-fund contribution. Surplus returns to the taxable
    account once the goal is paid. Returns ({goal: {pv, fund, sink}}, taxable left over)."""
    T = len(sched) - 1
    left = float(p.finances.taxable)
    out = {}
    goals = [g for g in p.goals if g.kind not in ("retirement", "bequest")]
    for g in sorted(goals, key=lambda g: {"need": 0, "want": 1, "wish": 2}[g.priority]):
        r_f, buf = {"need": (0.0, 1.03), "want": (0.01, 1.0), "wish": (0.02, 1.0)}[g.priority]
        flows = sched[f"goal:{g.name}"].to_numpy(float)
        pv_goal = buf * float(np.sum(flows / (1 + r_f) ** np.arange(T + 1)))
        fund = min(left, pv_goal) if g.funded_from_taxable else 0.0
        left -= fund
        short, yrs = pv_goal - fund, max(min(g.years_until, T), 0)
        out[g.name] = {"pv": pv_goal, "fund": float(fund),
                       "sink": (short / yrs) if (short > 0 and yrs >= 1) else 0.0}
    return out, left


# ---------------------------------------------------------------- core
def run(p: ClientProfile, rp: RiskProfile, cfg: SimConfig | None = None, cma: CMA | None = None,
        glide: np.ndarray | None = None, returns: np.ndarray | None = None) -> SimResult:
    cfg = cfg or SimConfig()
    cma = cma or load_cma()
    models = load_models()
    f, h = p.finances, p.household
    rng = np.random.default_rng(cfg.seed)
    sched = cashflow_schedule(p, cfg.apply_ss_shortfall)
    T = len(sched) - 1
    n = cfg.n_paths
    ages = sched["age"].to_numpy(int)
    infl = np.full(T + 1, cma.inflation)
    status = p.filing_status
    married = h.marital_status == "married" and bool(h.spouse_age)

    glide = retirement_glidepath(p, rp, cma, cfg.apply_ss_shortfall) if glide is None else glide
    W_ret = _weights_path(glide, models)
    i_lc = cma.idx("us_lc")

    R = simulate_returns(cma, n, T + 1, rng, cfg.fat_tails, cfg.t_df) if returns is None else returns[:, :T + 1].copy()
    if cfg.scenario:
        Rs, infl_s = stress_path_returns(cfg.scenario, cma)
        k = min(len(Rs), T + 1)
        R[:, :k, :] = Rs[None, :k, :]
        infl[:k] = infl_s[:k]
    R = (1 + R) * (1 - cfg.fee) - 1                       # fees come off every sleeve every year
    R_real = (1 + R) / (1 + infl)[None, :, None] - 1
    r_ret = np.einsum("ntk,tk->nt", R_real, W_ret)
    rf_nom = cma.cash_rate
    for ov, t0, t1 in cfg.overlays:
        if ov.kind == "csp":
            continue
        ro = overlay_return(ov, R[:, :, i_lc], rf_nom)
        diff = ((1 + ro) / (1 + infl)[None, :] - 1) - R_real[:, :, i_lc]
        active = np.zeros(T + 1); active[max(t0, 0):min(t1, T) + 1] = 1
        r_ret += diff * (W_ret[:, i_lc] * ov.fraction * active)[None, :]

    # goal sleeves --------------------------------------------------------------
    goals = [g for g in p.goals if g.kind not in ("retirement", "bequest")]
    g_cols = {g.name: sched[f"goal:{g.name}"].to_numpy(float) for g in goals}
    funding, taxable0 = fund_goals(p, sched)
    g_ret, g_bal, g_sink = {}, {}, {}
    for g in goals:
        e_g = goal_equity_path(g, rp.recommended_equity, T)
        g_ret[g.name] = np.einsum("ntk,tk->nt", R_real, _weights_path(e_g, models))
        g_bal[g.name] = np.full(n, funding[g.name]["fund"])
        g_sink[g.name] = funding[g.name]["sink"]
    g_paid = {g.name: np.zeros(n) for g in goals}
    g_need = {g.name: float(g_cols[g.name].sum()) for g in goals}

    # accounts ---------------------------------------------------------------
    tax = np.full(n, float(taxable0))
    basis = tax * f.taxable_cost_basis_pct
    trad = np.full(n, float(f.traditional))
    roth = np.full(n, float(f.roth))

    # mortality ----------------------------------------------------------------
    big = 10_000
    if cfg.mortality:
        d_c = sample_death_ages(h.sex, h.age, n, rng, h.health)
        if married:
            d_s = sample_death_ages(h.spouse_sex or "male", h.spouse_age, n, rng, h.spouse_health) - h.spouse_age + h.age
        else:
            d_s = np.full(n, -1)
    else:
        d_c = np.full(n, big)
        d_s = np.full(n, big if married else -1)

    # long-term-care episode: one per household, age drawn around 84, cost for LTC_YEARS ---
    if cfg.ltc_shock:
        hit = rng.random(n) < LTC_LIFETIME_PROB
        ltc_age = np.where(hit, np.clip(np.round(rng.normal(84, 6, n)), max(h.age, 65), 105), big)
        ltc_cost = LTC_ANNUAL_COST * (1 - LTC_INSURED_SHARE if h.long_term_care_insured else 1.0)
    else:
        ltc_age, ltc_cost = np.full(n, big), 0.0

    wealth = np.zeros((n, T + 1))
    spend_rec = np.zeros((n, T + 1))
    short_age = np.full(n, np.nan)
    m_mult = np.ones(n)
    wr0 = np.full(n, np.nan)
    ever_cut = np.zeros(n, bool)
    terminal = np.full(n, np.nan)
    trad_at_end = np.full(n, np.nan)
    retire_wealth = np.full(n, np.nan)
    taxes_paid = np.zeros(n)
    converted = np.zeros(n)
    ret_age = p.effective_retirement_age
    rmd_age = rmd_start_age(h.age)
    ess = f.essential_share_of_spending
    state = f.state_tax_rate
    conv_top = bracket_top(status, cfg.roth_conversion_bracket) if cfg.roth_conversion_bracket else None

    sal = sched["salary"].to_numpy(float)
    emp = sched["employee_savings"].to_numpy(float)
    mat = sched["employer_match"].to_numpy(float)
    ssc = sched["ss_client"].to_numpy(float)
    sss = sched["ss_spouse"].to_numpy(float)
    ssc_full = sched["ss_client_full"].to_numpy(float)
    sss_full = sched["ss_spouse_full"].to_numpy(float)
    pen = sched["pension"].to_numpy(float)
    spd = sched["retirement_spending"].to_numpy(float)
    cap = sched["retirement_cap"].to_numpy(float)
    plan_end_age = int(ages[-1])

    for t in range(T + 1):
        a = ages[t]
        alive_c = d_c >= a
        alive_s = d_s >= a
        alive = alive_c | alive_s
        both = alive_c & alive_s
        just_died = ~alive & np.isnan(terminal) & (t > 0)
        if cfg.mortality:
            terminal = np.where(just_died, wealth[:, t - 1] if t > 0 else terminal, terminal)
            trad_at_end = np.where(just_died, trad, trad_at_end)
        sp_age = (h.spouse_age + t) if married else 0
        deduction = standard_deduction(status, (a, sp_age) if married else (a,))

        def household_ss(c, s):
            if not married:
                return c * alive_c
            return np.where(both, c + s, np.where(alive, max(c, s), 0.0))   # survivor keeps the larger benefit

        pension_t = np.where(alive_c, pen[t], np.where(alive_s, 0.5 * pen[t], 0))
        ss_t = household_ss(ssc[t], sss[t])
        other_income = SS_INCLUSION * ss_t + pension_t + sal[t] * alive_c

        # contributions (while working, client alive) ---------------------------------
        if sal[t] > 0:
            working = alive_c.astype(float)
            avail = emp[t] * working
            for g in goals:
                if t < g.years_until and g_sink[g.name] > 0:
                    put = np.minimum(avail, g_sink[g.name])
                    g_bal[g.name] += put
                    avail = avail - put
            to_ret = np.minimum(avail, cap[t])
            roth += to_ret * f.contrib_to_roth_share
            trad += to_ret * (1 - f.contrib_to_roth_share)
            trad += np.minimum(mat[t], max(MATCH_CAP_415C * (2 if married and f.spouse_income > 0 else 1) - cap[t], 0)) * working
            tax += avail - to_ret
            basis += avail - to_ret
        if a == ret_age or (t == 0 and a > ret_age):
            retire_wealth = tax + trad + roth + sum(g_bal.values())

        # required minimum distributions: due from the RMD age whether or not the client works ---
        rmd = np.where((a >= rmd_age) & alive_c, trad / rmd_divisor(a), 0.0)
        trad -= rmd
        rmd_tax = incremental_tax(other_income, rmd, status, deduction, state)
        taxes_paid += rmd_tax
        rmd_cash = rmd - rmd_tax
        income_after_rmd = other_income + rmd

        # Roth conversions in the window before RMDs (paid from the taxable account) -----------
        if conv_top is not None and a >= max(ret_age, 59) and a < rmd_age:
            room = np.maximum(conv_top + deduction - income_after_rmd, 0.0)
            conv = np.minimum(np.minimum(room, trad), np.where(alive_c, np.inf, 0.0))
            conv_tax = incremental_tax(income_after_rmd, conv, status, deduction, state)
            pay_from_tax = np.minimum(conv_tax, tax)
            short_tax = conv_tax - pay_from_tax            # if taxable is empty, tax comes out of the conversion
            frac = np.where(tax > 0, pay_from_tax / np.maximum(tax, 1e-9), 0)
            basis -= basis * frac
            tax -= pay_from_tax
            trad -= conv
            roth += conv - short_tax
            converted += conv
            taxes_paid += conv_tax
            income_after_rmd = income_after_rmd + conv

        # goal payments ----------------------------------------------------------------
        for g in goals:
            need = g_cols[g.name][t]
            if need > 0:
                pay = np.minimum(g_bal[g.name], need)
                g_bal[g.name] -= pay
                rest = need - pay
                top = np.minimum(tax, rest)          # family tops up from the taxable account
                frac = np.where(tax > 0, top / np.maximum(tax, 1e-9), 0)
                basis -= basis * frac
                tax -= top
                g_paid[g.name] += (pay + top) * alive
            if t == min(g.years_until, T) + max(g.duration_years, 1) - 1 or (t == T):
                tax += g_bal[g.name]; basis += g_bal[g.name]; g_bal[g.name][:] = 0

        # retirement spending ---------------------------------------------------------
        if a >= ret_age:
            port_start = tax + trad + roth + rmd_cash
            base = spd[t] * np.where(married & ~both & alive, cfg.survivor_spending, 1.0)
            S = base * (m_mult if cfg.spending_rule == "guardrails" else 1.0)
            S = np.maximum(S, base * ess) * alive
            ltc = np.where((a >= ltc_age) & (a < ltc_age + LTC_YEARS) & alive, ltc_cost, 0.0)
            income = ss_t + pension_t
            need = np.maximum(S + ltc - income, 0)
            # long-run need (Social Security as if claimed) drives the guardrails, so a planned
            # bridge to a later claiming age is not mistaken for overspending
            need_lr = np.maximum(S - household_ss(ssc_full[t], sss_full[t]) - pension_t, 0)
            surplus = np.maximum(income - S - ltc, 0) * alive
            tax += surplus; basis += surplus

            excess = np.maximum(rmd_cash - need, 0)
            tax += excess; basis += excess
            rem = np.maximum(need - rmd_cash, 0)
            gain_frac = np.clip(1 - basis / np.maximum(tax, 1e-9), 0, 1)
            gross_tax_acct = np.minimum(rem / (1 - f.cap_gains_rate * gain_frac), tax)
            frac = np.where(tax > 0, gross_tax_acct / np.maximum(tax, 1e-9), 0)
            basis -= basis * frac
            tax -= gross_tax_acct
            cg_tax = gross_tax_acct * f.cap_gains_rate * gain_frac
            taxes_paid += cg_tax
            rem = np.maximum(rem - (gross_tax_acct - cg_tax), 0)
            gross_trad = np.minimum(gross_up(rem, income_after_rmd, status, deduction, state), trad)
            trad_tax = incremental_tax(income_after_rmd, gross_trad, status, deduction, state)
            taxes_paid += trad_tax
            trad -= gross_trad
            rem = np.maximum(rem - (gross_trad - trad_tax), 0)
            from_roth = np.minimum(rem, roth)
            roth -= from_roth
            rem -= from_roth
            unmet = (rem > 0.01 * np.maximum(S, 1)) & alive
            short_age = np.where(unmet & np.isnan(short_age), a, short_age)
            spend_rec[:, t] = S - np.minimum(rem, S)

            if cfg.spending_rule == "guardrails":
                wr = need_lr / np.maximum(port_start, 1)
                wr0 = np.where(np.isnan(wr0) & (need_lr > 0), wr, wr0)
                can_cut = a <= plan_end_age - GUARDRAIL_NO_CUT_YEARS
                hi = (wr > 1.2 * wr0) & (wr > GUARDRAIL_MIN_WR) & can_cut   # low withdrawal rates never trigger cuts
                lo = wr < 0.8 * wr0
                m_mult = np.where(hi, np.maximum(m_mult * 0.9, ess), np.where(lo, np.minimum(m_mult * 1.1, 1.25), m_mult))
                ever_cut |= (m_mult < 0.9) & alive
        else:
            tax += rmd_cash; basis += rmd_cash            # RMD while still working goes to the taxable account

        wealth[:, t] = tax + trad + roth + sum(g_bal.values())

        # returns ---------------------------------------------------------------------
        gr = 1 + r_ret[:, t]
        tax *= gr; trad *= gr; roth *= gr                  # basis is not clamped: unrealised losses carry forward
        for g in goals:
            g_bal[g.name] *= 1 + g_ret[g.name][:, t]

    end = tax + trad + roth + sum(g_bal.values())
    terminal = np.where(np.isnan(terminal), end, terminal)
    trad_at_end = np.where(np.isnan(trad_at_end), trad, trad_at_end)
    heir_rate = max(f.ordinary_tax_rate, 0.22)          # heirs pay ordinary tax on inherited pre-tax balances (10-year rule)
    terminal_after_tax = terminal - trad_at_end * heir_rate
    pct = pd.DataFrame({f"p{q}": np.percentile(wealth, q, axis=0) for q in PCTS}, index=ages)
    pct.index.name = "age"
    success = float(np.mean(np.isnan(short_age)))
    gs = {g: float(np.mean(g_paid[g] >= 0.99 * g_need[g])) if g_need[g] > 0 else 1.0 for g in g_paid}
    bq = next((g for g in p.goals if g.kind == "bequest" and g.amount > 0), None)
    if bq is not None:
        gs[bq.name] = float(np.mean(terminal >= bq.amount))
    return SimResult(ages=ages, wealth=wealth, pct=pct, success=success, goal_success=gs, spending=spend_rec,
                     shortfall_age=short_age, terminal=terminal, retire_wealth=retire_wealth,
                     cut_prob=float(np.mean(ever_cut)),
                     accounts_end={"taxable": float(np.median(tax)), "traditional": float(np.median(trad)),
                                   "roth": float(np.median(roth))}, config=cfg,
                     taxes_paid=taxes_paid, converted=converted,
                     ltc_hit=float(np.mean(ltc_age < big)) if cfg.ltc_shock else 0.0,
                     success_full=float(np.mean(np.isnan(short_age) & ~ever_cut)),
                     terminal_after_tax=terminal_after_tax)


def sustainable_spending(p: ClientProfile, rp: RiskProfile, target: float = 0.85, cfg: SimConfig | None = None,
                         cma: CMA | None = None, iters: int = 10) -> float:
    """Highest real retirement spending that is paid in full, with no cut, in `target` of paths
    (bisection with common random numbers). Measured with fixed real spending so the figure is
    not flattered by guardrail cuts."""
    import copy
    cfg = copy.copy(cfg) if cfg else SimConfig(n_paths=1500)
    cfg.spending_rule = "fixed"
    cma = cma or load_cma()
    rng = np.random.default_rng(cfg.seed)
    T = planning_age(p) - p.household.age
    R = simulate_returns(cma, cfg.n_paths, T + 1, rng, cfg.fat_tails, cfg.t_df)
    glide = retirement_glidepath(p, rp, cma, cfg.apply_ss_shortfall)
    lo, hi = 0.0, max(p.finances.retirement_spending * 3, 50_000)
    for _ in range(iters):
        mid = (lo + hi) / 2
        q = copy.deepcopy(p)
        q.finances.retirement_spending = mid
        for g in q.goals:
            if g.kind == "retirement":
                g.amount = mid
        if run(q, rp, cfg, cma, glide=glide, returns=R).success >= target:
            lo = mid
        else:
            hi = mid
    return lo
