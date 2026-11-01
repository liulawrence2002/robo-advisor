"""Options and futures overlays: pricing, payoffs, sizing, and simulation transforms.

Pricing: Black-Scholes-Merton with continuous dividend yield. Implied volatility comes
from the VIX (live) blended toward a long-run 1-year level, with a linear skew in
log-moneyness so downside puts cost more than upside calls (as in listed SPX markets).
All overlays are written on the S&P 500 (SPY/SPX or the client's own large-cap ETF).

The robo only *recommends* risk-reducing or income overlays on positions the client
owns (covered calls, protective puts, collars, cash-secured puts used to buy the target
allocation) and short index futures as a hedge. Uncovered writing and speculative
futures are outside policy at every approval level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DIV_YIELD = 0.012        # S&P 500 dividend yield (approx.)
SKEW = 0.30              # IV change per unit of log-moneyness (puts richer, calls cheaper)
LONG_RUN_VOL = 0.18      # long-run 1-year S&P implied vol
TRADING_COST = 0.0015    # annual bid/ask + commissions on a rolled overlay
MES_MULT, ES_MULT = 5, 50  # Micro E-mini / E-mini S&P 500 contract multipliers ($ per index point)
BUFFER_ETF_FEE = 0.0075  # Morningstar: defined-outcome ETFs average 0.75%


# ---------------------------------------------------------------- pricing
def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_delta(S, K, T, r, q, sigma, kind) -> float:
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1))


def atm_vol(vix: float | None) -> float:
    """1-year ATM implied vol: VIX (30-day) blended 50/50 with the long-run level."""
    if vix is None or not np.isfinite(vix):
        return LONG_RUN_VOL
    return 0.5 * vix / 100 + 0.5 * LONG_RUN_VOL


def skew_vol(moneyness: float, sigma_atm: float) -> float:
    return max(0.05, sigma_atm - SKEW * np.log(moneyness))


@dataclass
class Market:
    spot: float = 6_800.0      # S&P 500 level (fallback if no live data)
    vix: float | None = 17.0
    rate: float = 0.031        # 1-year risk-free (JPM cash assumption fallback)
    div_yield: float = DIV_YIELD
    live: bool = False
    asof: str = "CMA defaults"
    sigma_override: float | None = None   # e.g. single-stock implied vol

    @property
    def sigma(self) -> float:
        return self.sigma_override if self.sigma_override else atm_vol(self.vix)


def option_cost(m: Market, moneyness: float, kind: str, T: float = 1.0) -> float:
    """Premium as a fraction of spot."""
    K = m.spot * moneyness
    return bs_price(m.spot, K, T, m.rate, m.div_yield, skew_vol(moneyness, m.sigma), kind) / m.spot


def zero_cost_call_strike(m: Market, put_moneyness: float, T: float = 1.0) -> float:
    """Call moneyness whose premium pays for the put (zero-cost collar)."""
    put = option_cost(m, put_moneyness, "put", T)
    f = lambda k: option_cost(m, k, "call", T) - put
    if f(1.0) < 0:     # put costs more than an ATM call: collar cannot be zero-cost above spot
        return 1.0
    return float(brentq(f, 1.0, 2.0))


# ---------------------------------------------------------------- overlay definitions
@dataclass
class Overlay:
    kind: str                    # covered_call | protective_put | collar | buffer_etf | csp | futures_hedge
    fraction: float              # share of the US large-cap sleeve covered (0..1)
    put_k: float = 0.0           # moneyness
    call_k: float = 0.0
    premium: float = 0.0         # net premium received (+) or paid (-), fraction of notional
    buffer: float = 0.0          # buffer ETF parameters
    cap: float = 0.0
    label: str = ""

    def describe(self) -> str:
        return self.label or self.kind


def make_covered_call(m: Market, fraction: float, call_k: float = 1.06) -> Overlay:
    prem = option_cost(m, call_k, "call")
    return Overlay("covered_call", fraction, call_k=call_k, premium=prem,
                   label=f"Covered call: sell 1-yr {call_k:.0%} calls on {fraction:.0%} of US large cap "
                         f"(premium {prem:.2%})")


def make_protective_put(m: Market, fraction: float, put_k: float = 0.90) -> Overlay:
    prem = option_cost(m, put_k, "put")
    return Overlay("protective_put", fraction, put_k=put_k, premium=-prem,
                   label=f"Protective put: buy 1-yr {put_k:.0%} puts on {fraction:.0%} of US large cap "
                         f"(cost {prem:.2%})")


def make_collar(m: Market, fraction: float, put_k: float = 0.90) -> Overlay:
    ck = zero_cost_call_strike(m, put_k)
    net = option_cost(m, ck, "call") - option_cost(m, put_k, "put")
    return Overlay("collar", fraction, put_k=put_k, call_k=ck, premium=net,
                   label=f"Zero-cost collar: long {put_k:.0%} put / short {ck:.1%} call on "
                         f"{fraction:.0%} of US large cap")


def make_buffer_etf(fraction: float, buffer: float = 0.10, cap: float = 0.146) -> Overlay:
    """Default terms: Innovator 10% buffer, 1-yr S&P 500 series, cap 14.6% (May 2026 list)."""
    return Overlay("buffer_etf", fraction, buffer=buffer, cap=cap,
                   label=f"Buffered ETF: {buffer:.0%} buffer / {cap:.1%} cap on {fraction:.0%} of US large cap")


def make_csp(m: Market, fraction: float, put_k: float = 0.95) -> Overlay:
    prem = option_cost(m, put_k, "put")
    return Overlay("csp", fraction, put_k=put_k, premium=prem,
                   label=f"Cash-secured put: sell 1-yr {put_k:.0%} puts against cash earmarked for equities "
                         f"(premium {prem:.2%})")


# ---------------------------------------------------------------- payoffs (annual, on total return)
def overlay_return(ov: Overlay, total_ret: np.ndarray, rf: float) -> np.ndarray:
    """Return of an overlaid S&P position given the underlying's total return (vectorised).

    Options settle on price return (total - dividend). The holder keeps dividends except
    in a buffer ETF (which holds options, not stock). Premiums earn the risk-free rate.
    """
    q = DIV_YIELD
    price = total_ret - q
    if ov.kind == "covered_call":
        return q + np.minimum(price, ov.call_k - 1) + ov.premium * (1 + rf) - TRADING_COST
    if ov.kind == "protective_put":
        return q + np.maximum(price, ov.put_k - 1) + ov.premium * (1 + rf) - TRADING_COST
    if ov.kind == "collar":
        return q + np.clip(price, ov.put_k - 1, ov.call_k - 1) + ov.premium * (1 + rf) - TRADING_COST
    if ov.kind == "buffer_etf":
        up = np.minimum(price, ov.cap)
        down = np.where(price >= -ov.buffer, 0.0, price + ov.buffer)
        return np.where(price >= 0, up, down) - BUFFER_ETF_FEE
    if ov.kind == "csp":
        # cash earning rf + short put: lose below strike, keep premium
        return rf + ov.premium * (1 + rf) + np.minimum(0.0, price - (ov.put_k - 1)) - TRADING_COST
    raise ValueError(ov.kind)


def payoff_curve(ov: Overlay, rf: float, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-0.45, 0.45, 181) if grid is None else grid
    return grid, overlay_return(ov, grid + DIV_YIELD, rf)


# ---------------------------------------------------------------- futures sizing
def futures_hedge(notional_to_hedge: float, m: Market, beta: float = 1.0, micro: bool = True) -> dict:
    """Short S&P 500 futures to cut equity exposure without selling (and realising gains)."""
    mult = MES_MULT if micro else ES_MULT
    per = m.spot * mult
    n = int(round(beta * notional_to_hedge / per))
    return {"contracts": n, "contract": "Micro E-mini S&P 500 (MES)" if micro else "E-mini S&P 500 (ES)",
            "notional_per_contract": per, "hedged_notional": n * per,
            "initial_margin_est": n * per * 0.06}   # ~6% of notional (exchange margins vary)
