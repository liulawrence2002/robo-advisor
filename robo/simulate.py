"""Monte Carlo projection of accumulation and decumulation (real, today's dollars).

Per path and year:
  returns   correlated asset returns, lognormal with Student-t(5) shocks (fat tails),
            calibrated so the median compound return equals the JPM 2026 CMA
  sleeves   one sleeve per non-retirement goal (its own glide to the goal date) and the
            retirement sleeve (taxable / traditional / Roth accounts, household glide path)
  inflows   employer match -> 401(k); employee savings -> goal sinking funds, then
            retirement accounts up to 2026 IRS limits (Roth share as elected), then taxable
  outflows  goal payments; retirement spending net of Social Security and pension
  taxes     RMDs (SECURE 2.0 ages, Uniform Lifetime Table) first, then taxable (capital-gains
            tax on the gain share, basis tracked), then traditional (ordinary rate), then Roth
  longevity stochastic death ages for client and spouse (Gompertz fit to SSA 2022 table);
            survivor keeps the larger Social Security benefit, spending drops to 75%
  spending  fixed real, or Guyton-Klinger style guardrails (cut 10% if the withdrawal rate
            rises 20% above its starting level, raise 10% if it falls 20% below; never below
            the essential share)
  overlays  option/buffer payoffs replace part of the US large-cap return in active years
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cma import CMA, load_cma
from .derivatives import Overlay, overlay_return
from .goals import cashflow_schedule, goal_equity_path, planning_age
from .kyc import ClientProfile
from .mortality import sample_death_ages
from .portfolio import blend_for_equity, load_models
from .retirement import rmd_divisor, rmd_start_age
from .risk import RiskProfile, retirement_glidepath

PCTS = (5, 25, 50, 75, 95)
GUARDRAIL_MIN_WR = 0.04


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


# ---------------------------------------------------------------- returns
def simulate_returns(cma: CMA, n: int, T: int, rng: np.random.Generator, fat_tails: bool = True,
                     t_df: int = 5) -> np.ndarray:
    """Nominal simple returns, shape (n, T, N)."""
    L = np.linalg.cholesky(cma.corr)
    z = rng.standard_normal((n, T, cma.n)) @ L.T
    if fat_tails:
        chi = rng.chisquare(t_df, size=(n, T, 1))
        z = z / np.sqrt(chi / t_df) * np.sqrt((t_df - 2) / t_df)
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
        short, yrs = pv_goal - fund, max(g.years_until, 0)
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
    infl = cma.inflation

    glide = retirement_glidepath(p, rp, cma, cfg.apply_ss_shortfall) if glide is None else glide
    W_ret = _weights_path(glide, models)
    i_lc = cma.idx("us_lc")

    R = simulate_returns(cma, n, T + 1, rng, cfg.fat_tails, cfg.t_df) if returns is None else returns[:, :T + 1]
    R_real = (1 + R) / (1 + infl) - 1
    r_ret = np.einsum("ntk,tk->nt", R_real, W_ret)
    rf_nom = cma.cash_rate
    for ov, t0, t1 in cfg.overlays:
        if ov.kind == "csp":
            continue
        ro = overlay_return(ov, R[:, :, i_lc], rf_nom)
        diff = ((1 + ro) / (1 + infl) - 1) - R_real[:, :, i_lc]
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
        if h.marital_status == "married" and h.spouse_age:
            d_s = sample_death_ages(h.spouse_sex or "male", h.spouse_age, n, rng, h.spouse_health) - h.spouse_age + h.age
        else:
            d_s = np.full(n, -1)
    else:
        d_c = np.full(n, big)
        d_s = np.full(n, big if (h.marital_status == "married" and h.spouse_age) else -1)
    married = h.marital_status == "married" and bool(h.spouse_age)

    wealth = np.zeros((n, T + 1))
    spend_rec = np.zeros((n, T + 1))
    short_age = np.full(n, np.nan)
    m_mult = np.ones(n)
    wr0 = np.full(n, np.nan)
    ever_cut = np.zeros(n, bool)
    terminal = np.full(n, np.nan)
    retire_wealth = np.full(n, np.nan)
    ret_age = f.retirement_age
    rmd_age = rmd_start_age(h.age)
    ess = f.essential_share_of_spending

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

    for t in range(T + 1):
        a = ages[t]
        alive_c = d_c >= a
        alive_s = d_s >= a
        alive = alive_c | alive_s
        just_died = ~alive & np.isnan(terminal) & (t > 0)
        if cfg.mortality:
            terminal = np.where(just_died, wealth[:, t - 1] if t > 0 else terminal, terminal)

        # contributions (pre-retirement, client alive) -------------------------------
        if a < ret_age:
            working = alive_c.astype(float)
            trad += mat[t] * working
            avail = emp[t] * working
            for g in goals:
                if t < g.years_until and g_sink[g.name] > 0:
                    put = np.minimum(avail, g_sink[g.name])
                    g_bal[g.name] += put
                    avail = avail - put
            to_ret = np.minimum(avail, cap[t])
            roth += to_ret * f.contrib_to_roth_share
            trad += to_ret * (1 - f.contrib_to_roth_share)
            tax += avail - to_ret
            basis += avail - to_ret
        if a == ret_age or (t == 0 and a > ret_age):
            retire_wealth = tax + trad + roth + sum(g_bal.values())

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
            if t == g.years_until + max(g.duration_years, 1) - 1 or (t == T):
                tax += g_bal[g.name]; basis += g_bal[g.name]; g_bal[g.name][:] = 0

        # retirement spending ---------------------------------------------------------
        if a >= ret_age:
            both = alive_c & alive_s
            base = spd[t] * np.where(married & ~both & alive, cfg.survivor_spending, 1.0)
            S = base * (m_mult if cfg.spending_rule == "guardrails" else 1.0)
            S = np.maximum(S, base * ess) * alive
            def household_ss(c, s):
                if not married:
                    return c * alive_c
                # survivor keeps the larger of the two benefits
                return np.where(both, c + s, np.where(alive, max(c, s), 0.0))
            pension_t = np.where(alive_c, pen[t], np.where(alive_s, 0.5 * pen[t], 0))
            income = household_ss(ssc[t], sss[t]) + pension_t
            need = np.maximum(S - income, 0)
            # long-run need (Social Security as if claimed) drives the guardrails, so a planned
            # bridge to a later claiming age is not mistaken for overspending
            need_lr = np.maximum(S - household_ss(ssc_full[t], sss_full[t]) - pension_t, 0)
            surplus = np.maximum(income - S, 0) * alive
            tax += surplus; basis += surplus

            rmd = np.where((a >= rmd_age) & alive, trad / rmd_divisor(a), 0.0)
            trad -= rmd
            cash = rmd * (1 - f.ordinary_tax_rate)
            excess = np.maximum(cash - need, 0)
            tax += excess; basis += excess
            rem = np.maximum(need - cash, 0)
            gain_frac = np.clip(1 - basis / np.maximum(tax, 1e-9), 0, 1)
            gross_tax_acct = np.minimum(rem / (1 - f.cap_gains_rate * gain_frac), tax)
            frac = np.where(tax > 0, gross_tax_acct / np.maximum(tax, 1e-9), 0)
            basis -= basis * frac
            tax -= gross_tax_acct
            rem = np.maximum(rem - gross_tax_acct * (1 - f.cap_gains_rate * gain_frac), 0)
            gross_trad = np.minimum(rem / (1 - f.ordinary_tax_rate), trad)
            trad -= gross_trad
            rem = np.maximum(rem - gross_trad * (1 - f.ordinary_tax_rate), 0)
            from_roth = np.minimum(rem, roth)
            roth -= from_roth
            rem -= from_roth
            unmet = (rem > 0.01 * np.maximum(S, 1)) & alive
            short_age = np.where(unmet & np.isnan(short_age), a, short_age)
            spend_rec[:, t] = S - rem

            if cfg.spending_rule == "guardrails":
                port = tax + trad + roth
                wr = need_lr / np.maximum(port, 1)
                wr0 = np.where(np.isnan(wr0) & (need_lr > 0), wr, wr0)
                hi = (wr > 1.2 * wr0) & (wr > GUARDRAIL_MIN_WR)   # low withdrawal rates never trigger cuts
                lo = wr < 0.8 * wr0
                m_mult = np.where(hi, np.maximum(m_mult * 0.9, ess), np.where(lo, np.minimum(m_mult * 1.1, 1.25), m_mult))
                ever_cut |= (m_mult < 0.9) & alive

        wealth[:, t] = tax + trad + roth + sum(g_bal.values())

        # returns ---------------------------------------------------------------------
        gr = 1 + r_ret[:, t]
        tax *= gr; basis = np.minimum(basis, tax); trad *= gr; roth *= gr
        for g in goals:
            g_bal[g.name] *= 1 + g_ret[g.name][:, t]

    end = tax + trad + roth + sum(g_bal.values())
    terminal = np.where(np.isnan(terminal), end, terminal)
    alive_mask_pct = wealth
    pct = pd.DataFrame({f"p{q}": np.percentile(alive_mask_pct, q, axis=0) for q in PCTS}, index=ages)
    pct.index.name = "age"
    success = float(np.mean(np.isnan(short_age)))
    gs = {g: float(np.mean(g_paid[g] >= 0.99 * g_need[g])) if g_need[g] > 0 else 1.0 for g in g_paid}
    return SimResult(ages=ages, wealth=wealth, pct=pct, success=success, goal_success=gs, spending=spend_rec,
                     shortfall_age=short_age, terminal=terminal, retire_wealth=retire_wealth,
                     cut_prob=float(np.mean(ever_cut)),
                     accounts_end={"taxable": float(np.median(tax)), "traditional": float(np.median(trad)),
                                   "roth": float(np.median(roth))}, config=cfg)


def sustainable_spending(p: ClientProfile, rp: RiskProfile, target: float = 0.85, cfg: SimConfig | None = None,
                         cma: CMA | None = None, iters: int = 14) -> float:
    """Highest real retirement spending with P(success) >= target (bisection, common random numbers)."""
    import copy
    cfg = cfg or SimConfig(n_paths=1500)
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
