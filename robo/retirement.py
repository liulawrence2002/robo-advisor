"""Retirement rules: Social Security, RMDs, 2026 contribution limits, withdrawal rules."""
from __future__ import annotations

CURRENT_YEAR = 2026

# 2026 Social Security parameters (SSA, announced Oct 2025)
SS_BEND_1, SS_BEND_2 = 1_286, 7_749       # monthly AIME bend points, first eligible in 2026
SS_WAGE_BASE = 184_500
SS_MAX_AT_FRA_MONTHLY = 4_152
# 2026 Trustees Report: OASI reserves depleted 2032 Q4; ~78% of scheduled benefits payable after.
SS_DEPLETION_YEAR, SS_PAYABLE_AFTER = 2033, 0.78

# 2026 IRS limits (Notice 2025-67)
LIMIT_401K, CATCHUP_50, CATCHUP_60_63 = 24_500, 8_000, 11_250
LIMIT_IRA, IRA_CATCHUP = 7_500, 1_100

# IRS Uniform Lifetime Table (Treas. Reg. 1.401(a)(9)-9, effective 2022)
UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2,
    81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
    90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3,
    99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9,
}


def birth_year(age: int) -> int:
    return CURRENT_YEAR - age


def full_retirement_age(age_now: int) -> float:
    by = birth_year(age_now)
    if by <= 1954:
        return 66.0
    if by >= 1960:
        return 67.0
    return 66.0 + (by - 1954) * 2 / 12


def claim_factor(claim_age: float, fra: float) -> float:
    """Benefit as a multiple of PIA when claiming at claim_age (SSA early/delayed rules)."""
    months = round((claim_age - fra) * 12)
    if months >= 0:
        return 1.0 + 0.08 * min(months, round((70 - fra) * 12)) / 12
    early = -months
    if early <= 36:
        return 1.0 - early * 5 / 900
    return 1.0 - 0.20 - (early - 36) * 5 / 1200


def estimate_pia(annual_income: float) -> float:
    """Primary Insurance Amount (annual, today's $) assuming a 35-year career at today's real wage,
    capped at the 2026 maximum benefit at full retirement age."""
    aime = min(annual_income, SS_WAGE_BASE) / 12
    pia = 0.9 * min(aime, SS_BEND_1) + 0.32 * max(0.0, min(aime, SS_BEND_2) - SS_BEND_1) \
        + 0.15 * max(0.0, aime - SS_BEND_2)
    return 12 * min(pia, SS_MAX_AT_FRA_MONTHLY)


def estimate_benefit(annual_income: float, age_now: int, claim_age: int) -> float:
    return estimate_pia(annual_income) * claim_factor(claim_age, full_retirement_age(age_now))


def spousal_claim_factor(claim_age: float, fra: float) -> float:
    """Spousal benefit (50% of the worker's PIA) reduced 25/36% per month for the first 36 months
    before FRA and 5/12% after; no delayed-retirement credits (SSA POMS RS 00615.201)."""
    months = round((fra - claim_age) * 12)
    if months <= 0:
        return 1.0
    return 1.0 - min(months, 36) * 25 / 3600 - max(months - 36, 0) * 5 / 1200


def claiming_analysis(pia: float, fra: float, survival: dict, discount: float = 0.019) -> list[dict]:
    """Expected present value of lifetime benefits (today's $) for each claiming age 62-70, using the
    client's own survival curve. Breakeven ages compare each option with claiming at 62."""
    ages, surv = survival["ages"], survival["survival"]
    out = []
    for claim in range(62, 71):
        f = claim_factor(claim, fra)
        pv, cum = 0.0, []
        for a, s in zip(ages, surv):
            if a >= claim:
                pv += pia * f * s / (1 + discount) ** (a - ages[0])
            cum.append(pv)
        out.append({"claim_age": claim, "factor": f, "annual": pia * f, "epv": pv, "cum": cum})
    base = out[0]["cum"]
    for row in out[1:]:
        be = next((a for a, c0, c1 in zip(ages, base, row["cum"]) if c1 >= c0 and a > row["claim_age"]), None)
        row["breakeven_age"] = be
    out[0]["breakeven_age"] = None
    return out


def rmd_start_age(age_now: int) -> int:
    """SECURE 2.0: 73 for those born 1951-1959, 75 for 1960+."""
    return 75 if birth_year(age_now) >= 1960 else 73


def rmd_divisor(age: int) -> float:
    if age < 72:
        return float("inf")
    return UNIFORM_LIFETIME.get(min(age, 115), 2.9)


def retirement_contribution_cap(age: int) -> float:
    """Employee 401(k) deferral + IRA limit for one person in 2026."""
    catch = CATCHUP_60_63 if 60 <= age <= 63 else (CATCHUP_50 if age >= 50 else 0)
    return LIMIT_401K + catch + LIMIT_IRA + (IRA_CATCHUP if age >= 50 else 0)


def ss_payable_factor(year: int, apply_shortfall: bool) -> float:
    return SS_PAYABLE_AFTER if (apply_shortfall and year >= SS_DEPLETION_YEAR) else 1.0
