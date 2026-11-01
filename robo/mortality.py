"""Longevity model: Gompertz-Makeham hazard calibrated to the SSA 2022 period life table.

Calibration targets (remaining life expectancy, SSA Actuarial Life Table, 2022 period,
as used in the 2025 Trustees Report):
    age 30: male 46.51, female 51.25
    age 55: male 24.94, female 28.34
    age 70: male 14.09, female 16.27

hazard(x) = A + B * exp(c * x). The constant A captures accident/background mortality;
the exponential term captures aging. A health adjustment multiplies the hazard.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares

SSA_TARGETS = {
    "male": {30: 46.51, 55: 24.94, 70: 14.09},
    "female": {30: 51.25, 55: 28.34, 70: 16.27},
}
HEALTH_MULT = {"excellent": 0.75, "good": 0.9, "average": 1.0, "fair": 1.3, "poor": 1.8}
MAX_AGE = 115


def _surv_curve(params, age: float, mult: float = 1.0, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = params
    t = np.arange(0, MAX_AGE - age + step, step)
    # integrated hazard from age to age+t
    H = a * t + (b / c) * (np.exp(c * (age + t)) - np.exp(c * age))
    return t, np.exp(-mult * H)


def life_expectancy(params, age: float, mult: float = 1.0) -> float:
    t, s = _surv_curve(params, age, mult)
    return float(np.trapezoid(s, t))


@lru_cache(maxsize=4)
def calibrate(sex: str) -> tuple[float, float, float]:
    targets = SSA_TARGETS[sex]

    def resid(p):
        a, logb, c = p
        return [life_expectancy((a, np.exp(logb), c), age) - e for age, e in targets.items()]

    sol = least_squares(resid, x0=[5e-4, np.log(2e-5), 0.095], bounds=([0, -20, 0.05], [0.01, -5, 0.15]))
    a, logb, c = sol.x
    return float(a), float(np.exp(logb)), float(c)


def annual_death_probs(sex: str, age: int, health: str = "average") -> np.ndarray:
    """q_x for ages age..MAX_AGE-1 (probability of dying within the year)."""
    a, b, c = calibrate(sex)
    mult = HEALTH_MULT.get(health, 1.0)
    x = np.arange(age, MAX_AGE)
    H1 = a + (b / c) * (np.exp(c * (x + 1)) - np.exp(c * x))
    q = 1 - np.exp(-mult * H1)
    q[-1] = 1.0
    return q


def sample_death_ages(sex: str, age: int, n: int, rng: np.random.Generator, health: str = "average") -> np.ndarray:
    """Age at death (integer, death occurs during that year of age) for n simulated lives."""
    q = annual_death_probs(sex, age, health)
    surv = np.concatenate([[1.0], np.cumprod(1 - q)])
    u = rng.random(n)
    # first year where survival drops below u
    idx = np.searchsorted(-surv, -u, side="left") - 1
    return age + np.clip(idx, 0, len(q) - 1)


def survival_table(sex: str, age: int, health: str = "average") -> dict:
    q = annual_death_probs(sex, age, health)
    s = np.cumprod(1 - q)
    ages = np.arange(age + 1, MAX_AGE + 1)
    out = {"ages": ages, "survival": s}
    out["life_expectancy"] = life_expectancy(calibrate(sex), age, HEALTH_MULT.get(health, 1.0))
    for p in (0.5, 0.25, 0.1):
        hit = ages[s <= p]
        out[f"age_{int(p*100)}pct_survive"] = int(hit[0]) if len(hit) else MAX_AGE
    return out


def joint_survival(sex1: str, age1: int, sex2: str, age2: int, health1="average", health2="average") -> dict:
    """Probability at least one of two partners is alive, indexed by years from today."""
    q1, q2 = annual_death_probs(sex1, age1, health1), annual_death_probs(sex2, age2, health2)
    n = min(len(q1), len(q2))
    s1, s2 = np.cumprod(1 - q1[:n]), np.cumprod(1 - q2[:n])
    either = 1 - (1 - s1) * (1 - s2)
    years = np.arange(1, n + 1)
    hit = years[either <= 0.25]
    return {"years": years, "either_alive": either,
            "years_25pct": int(hit[0]) if len(hit) else n}
