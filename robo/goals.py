"""Goals-based planning: deterministic cash-flow schedule (today's dollars), planning
horizon, and per-goal equity paths (the 'buckets').

Goals-based wealth management (Chhabra 2005; Das, Markowitz, Scheid & Statman 2010)
treats each goal as its own mental account with its own horizon and required
probability of success. Near-term 'needs' are liability-matched (cash/short bonds);
long-horizon goals take the risk the client is willing and able to bear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kyc import ClientProfile, Goal
from .mortality import joint_survival, survival_table
from .retirement import (CURRENT_YEAR, full_retirement_age, claim_factor, estimate_pia,
                         retirement_contribution_cap, spousal_claim_factor, ss_payable_factor)

PLAN_FLOOR_AGE = 95


def planning_age(p: ClientProfile) -> int:
    """Plan to the later of age 95 or the age at which (joint) survival falls to 25%."""
    h = p.household
    if h.marital_status == "married" and h.spouse_age:
        js = joint_survival(h.sex, h.age, h.spouse_sex or "male", h.spouse_age, h.health, h.spouse_health)
        age25 = h.age + js["years_25pct"]
    else:
        age25 = survival_table(h.sex, h.age, h.health)["age_25pct_survive"]
    return int(max(PLAN_FLOOR_AGE, age25))


def social_security(p: ClientProfile) -> tuple[float, float, str]:
    """(client annual benefit, spouse annual benefit, note), today's $, at claim ages."""
    f, h = p.finances, p.household
    note = []
    if p.identity.citizenship == "Non-resident alien" and f.social_security is None:
        note.append("non-resident alien without an SSA statement: no benefit assumed")
        return 0.0, 0.0, "; ".join(note)
    married = h.marital_status == "married" and bool(h.spouse_age)
    own1, own2 = estimate_pia(f.annual_income), estimate_pia(f.spouse_income) if married else 0.0
    fra1 = full_retirement_age(h.age)
    if f.social_security is not None:
        ss1 = f.social_security
        note.append("client benefit from SSA statement")
    else:
        if f.annual_income <= 0 and not f.already_claiming_ss:
            note.append("no current earnings and no SSA statement: client benefit assumed zero; enter the statement")
        # own record, or the spousal benefit on the spouse's record (whichever is larger)
        ss1 = max(own1 * claim_factor(f.ss_claim_age, fra1),
                  0.5 * own2 * spousal_claim_factor(f.ss_claim_age, fra1) if married else 0.0)
        note.append(f"client benefit estimated from income at claim age {f.ss_claim_age}")
    ss2 = 0.0
    if married:
        if f.spouse_social_security is not None:
            ss2 = f.spouse_social_security
        else:
            fra2 = full_retirement_age(h.spouse_age)
            ss2 = max(own2 * claim_factor(f.spouse_ss_claim_age, fra2),
                      0.5 * own1 * spousal_claim_factor(f.spouse_ss_claim_age, fra2))
            note.append("spouse benefit estimated (own record or 50% spousal)")
    return ss1, ss2, "; ".join(note)


def retirement_spending(p: ClientProfile) -> float:
    g = p.retirement_goal
    return g.amount if g else p.finances.retirement_spending


def cashflow_schedule(p: ClientProfile, apply_ss_shortfall: bool = True) -> pd.DataFrame:
    """Expected annual cash flows from now (t=0) to the planning age. Real (today's) dollars.

    Contributions follow a bank-style waterfall: employer match -> 401(k); employee savings
    fund near-term 'need' goals first (sinking fund), then retirement accounts up to the
    2026 IRS limits, then the taxable account.
    """
    f, h = p.finances, p.household
    age0 = h.age
    T = planning_age(p) - age0
    t = np.arange(T + 1)
    age = age0 + t
    ret_age = p.effective_retirement_age
    # the client's own earnings stop at the effective retirement age; a stated retiree has none.
    # A working client past their 'retirement age' keeps earning until they stop (employment status).
    client_working = (age < ret_age) if p.identity.employment_status != "retired" else np.zeros_like(age, bool)
    if p.identity.employment_status != "retired" and h.age >= f.retirement_age and f.annual_income > 0:
        client_working = age < h.age + 1       # still working this year; retirement assumed from next year
    growth = (1 + f.income_real_growth) ** t
    married = h.marital_status == "married" and bool(h.spouse_age)
    # the spouse's earnings run to the spouse's own age at the household retirement age
    sp_working = (t < max(ret_age - h.age, 0)) if (married and f.spouse_income > 0) else np.zeros_like(age, bool)
    salary = np.where(client_working, f.annual_income * growth, 0.0) + np.where(sp_working, f.spouse_income * growth, 0.0)
    employee = salary * f.savings_rate
    match = salary * f.employer_match

    ss1, ss2, _ = social_security(p)
    claim1 = max(h.age, 62) if f.already_claiming_ss else f.ss_claim_age
    ss_client = np.where(age >= claim1, ss1, 0.0)
    if married:
        sp_age = h.spouse_age + t
        claim2 = h.spouse_age if (f.already_claiming_ss and h.spouse_age >= 62) else max(f.spouse_ss_claim_age, 62)
        ss_spouse = np.where(sp_age >= claim2, ss2, 0.0)
    else:
        ss_spouse = np.zeros_like(t, dtype=float)
    haircut = np.array([ss_payable_factor(CURRENT_YEAR + int(x), apply_ss_shortfall) for x in t])
    ss_client, ss_spouse = ss_client * haircut, ss_spouse * haircut
    # benefit as if already claimed: the long-run income the spending plan relies on
    ss_client_full = ss1 * haircut
    ss_spouse_full = (ss2 * haircut) if married else np.zeros(len(t))
    pension = np.where(age >= ret_age, f.pension, 0.0)
    spend = np.where(age >= ret_age, retirement_spending(p), 0.0)

    df = pd.DataFrame({"t": t, "age": age, "year": CURRENT_YEAR + t, "salary": salary,
                       "employee_savings": employee, "employer_match": match,
                       "ss_client": ss_client, "ss_spouse": ss_spouse, "pension": pension,
                       "ss_client_full": ss_client_full, "ss_spouse_full": ss_spouse_full,
                       "retirement_spending": spend})
    for g in p.goals:
        if g.kind in ("retirement", "bequest"):
            continue
        col = np.zeros(T + 1)
        start = min(max(int(g.years_until), 0), T)      # goals past the horizon are paid in the last year
        for k in range(start, min(start + max(g.duration_years, 1), T + 1)):
            col[k] = g.amount / max(g.duration_years, 1)
        df[f"goal:{g.name}"] = col
    df["retirement_cap"] = [retirement_contribution_cap(a) * (2 if h.marital_status == "married" and f.spouse_income > 0 else 1)
                            for a in age]
    return df


def goal_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("goal:")]


def net_flows(df: pd.DataFrame, p: ClientProfile, include_wishes: bool = False) -> np.ndarray:
    """Net flows into the portfolio (+ savings/income, - spending/goals)."""
    net = (df["employee_savings"] + df["employer_match"] + df["ss_client"] + df["ss_spouse"]
           + df["pension"] - df["retirement_spending"]).to_numpy(float)
    by_name = {f"goal:{g.name}": g for g in p.goals}
    for c in goal_columns(df):
        g = by_name[c]
        if g.priority != "wish" or include_wishes:
            net -= df[c].to_numpy(float)
    bq = next((g for g in p.goals if g.kind == "bequest"), None)
    if bq and (bq.priority != "wish" or include_wishes):
        net[-1] -= bq.amount
    return net


def pv(flows: np.ndarray, r: float) -> float:
    t = np.arange(len(flows))
    return float(np.sum(flows / (1 + r) ** t))


def liability_duration(df: pd.DataFrame, p: ClientProfile, r: float = 0.02) -> float:
    out = df["retirement_spending"].to_numpy(float).copy()
    for c in goal_columns(df):
        out += df[c].to_numpy(float)
    t = np.arange(len(out))
    disc = out / (1 + r) ** t
    return float((t * disc).sum() / disc.sum()) if disc.sum() > 0 else 0.0


# ---------------------------------------------------------------- bucket equity rules
def goal_equity_path(g: Goal, e_long: float, T: int) -> np.ndarray:
    """Equity share for a goal sleeve as a function of years remaining to the goal.

    needs: 0% within 2 years, then +7%/yr of horizon; wants: +9%/yr after year 1;
    wishes: +12%/yr. Always capped at the client's long-horizon equity e_long.
    Once spending starts (education), the sleeve stays at its floor.
    """
    slope, lag = {"need": (0.07, 2), "want": (0.09, 1), "wish": (0.12, 0)}[g.priority]
    years_left = np.maximum(g.years_until - np.arange(T + 1), 0)
    return np.minimum(e_long, np.maximum(0.0, (years_left - lag) * slope))
