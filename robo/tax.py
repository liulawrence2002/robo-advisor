"""Federal income tax on ordinary income: 2026 brackets and standard deduction, vectorised.

Used in the projection for RMDs, traditional-account withdrawals and Roth conversions,
in place of a single flat marginal rate. Social Security is included at the 85% maximum
inclusion rate (the provisional-income phase-in is not modelled). State tax is a flat
rate from the profile. Brackets: IRS Rev. Proc. 2025-32 (Oct 2025), standard deduction
after the 2025 reconciliation act, plus the age-65 additional deduction.
"""
from __future__ import annotations

import numpy as np

# (upper bound of taxable income, rate) for 2026
BRACKETS = {
    "single": [(12_400, .10), (50_400, .12), (105_700, .22), (201_775, .24), (256_225, .32), (640_600, .35), (np.inf, .37)],
    "mfj":    [(24_800, .10), (100_800, .12), (211_400, .22), (403_550, .24), (512_450, .32), (768_700, .35), (np.inf, .37)],
}
STD_DEDUCTION = {"single": 16_100, "mfj": 32_200}
EXTRA_65 = {"single": 2_050, "mfj": 1_650}       # per qualifying person
SS_INCLUSION = 0.85                                # share of Social Security that is taxable at most
IRMAA_FIRST_THRESHOLD = {"single": 109_000, "mfj": 218_000}   # 2026 MAGI above which Medicare premiums rise


def standard_deduction(status: str, ages: tuple[int, ...] = ()) -> float:
    return STD_DEDUCTION[status] + EXTRA_65[status] * sum(a >= 65 for a in ages)


def tax_due(taxable_income, status: str):
    """Federal tax on taxable income (after deductions). Works on floats or numpy arrays."""
    x = np.asarray(taxable_income, dtype=float)
    tax = np.zeros_like(x)
    lo = 0.0
    for hi, rate in BRACKETS[status]:
        tax += np.clip(x - lo, 0, hi - lo) * rate
        lo = hi
    return tax


def marginal_rate(taxable_income, status: str):
    x = np.asarray(taxable_income, dtype=float)
    out = np.full_like(x, BRACKETS[status][-1][1])
    lo = 0.0
    for hi, rate in BRACKETS[status]:
        out = np.where((x >= lo) & (x < hi), rate, out)
        lo = hi
    return out


def bracket_top(status: str, rate: float) -> float:
    """Taxable income at the top of the bracket with the given rate."""
    for hi, r in BRACKETS[status]:
        if abs(r - rate) < 1e-9:
            return hi
    raise ValueError(rate)


def incremental_tax(base_income, extra, status: str, deduction: float, state_rate: float = 0.0):
    """Federal + state tax caused by adding `extra` of ordinary income on top of `base_income`
    (both gross; the deduction is applied once)."""
    t0 = tax_due(np.maximum(np.asarray(base_income, float) - deduction, 0), status)
    t1 = tax_due(np.maximum(np.asarray(base_income, float) + np.asarray(extra, float) - deduction, 0), status)
    return t1 - t0 + state_rate * np.asarray(extra, float)


def gross_up(net_needed, base_income, status: str, deduction: float, state_rate: float = 0.0, iters: int = 8):
    """Gross withdrawal from a pre-tax account that leaves `net_needed` after federal + state tax.
    Fixed-point iteration on the marginal rate (converges in 2-3 steps for piecewise-linear tax)."""
    net = np.asarray(net_needed, float)
    gross = net / (1 - state_rate - marginal_rate(np.maximum(np.asarray(base_income, float) - deduction, 0), status))
    for _ in range(iters):
        t = incremental_tax(base_income, gross, status, deduction, state_rate)
        gross = net + t
    return np.maximum(gross, 0.0)


def effective_rate(taxable_income, status: str):
    x = np.asarray(taxable_income, float)
    return np.where(x > 0, tax_due(x, status) / np.maximum(x, 1), 0.0)
