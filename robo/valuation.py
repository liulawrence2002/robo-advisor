"""Fair value: single stocks and the equity market, computed from fetched inputs.

Single stock (Damodaran, *Investment Valuation*; Graham, *The Intelligent Investor*; Morningstar
equity methodology for the star mapping):
  1. analyst consensus target (median, low-high range, weighted by coverage)
  2. two-stage free-cash-flow DCF: CAPM cost of equity, stage-1 growth fading to 4% nominal GDP
     growth, +/-1% WACC and terminal-growth sensitivity corners
  3. Gordon dividend model when the payout is meaningful
  4. Graham formula V = EPS x (8.5 + 2g) as a growth-adjusted multiple, and the Graham number
     sqrt(22.5 x EPS x BV) for asset-heavy firms
  Blend -> fair value, a band, price/fair value, a Morningstar-style uncertainty rating from
  realised volatility and leverage, and 1-5 stars with 'consider buying / selling' prices.

Equity market: Shiller CAPE with Research Affiliates-style half mean reversion, the excess CAPE
yield, the Fed model spread, the Buffett indicator, and a bounded tilt (+/-1.5%) on the J.P.
Morgan 2026 LTCMA US large-cap assumption. Bonds: TIPS real yield vs breakeven inflation vs the
2.5% assumption. Options: VIX term structure and SKEW as a hedging-cost signal.
"""
from __future__ import annotations

import math

import numpy as np

TERMINAL_GROWTH = 0.04            # long-run nominal GDP growth (2% real + 2% inflation)
JPM_US_LC_RETURN = 0.067          # J.P. Morgan 2026 LTCMA, US large cap, compound nominal
CAPE_MEAN_1960 = 21.5             # mean Shiller CAPE since 1960 (Shiller data); 1881-2026 mean is ~17
BUFFETT_MEAN = 1.45               # nonfinancial corporate equity / GDP, mean 2000-2026 (FRED NCBEILQ027S / GDP)
FED_SPREAD_MEAN = 0.0             # earnings yield minus 10y yield, long-run mean
TILT_CAP = 0.015

UNCERTAINTY_BANDS = {   # Morningstar-style: uncertainty -> (5-star P/FV, 4-star, 2-star, 1-star)
    "Low": (0.80, 0.90, 1.10, 1.25), "Medium": (0.70, 0.85, 1.15, 1.35), "High": (0.60, 0.80, 1.25, 1.55),
    "Very High": (0.50, 0.70, 1.35, 1.75), "Extreme": (0.25, 0.50, 1.75, 4.00),
}


def _ok(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x)) and float(x) != 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- single stock
def dcf_value(snap: dict, rf: float, erp: float, wacc_shift: float = 0.0, g_shift: float = 0.0) -> float | None:
    fcf_hist = [x for x in snap.get("fcf_history") or [] if _ok(x)]
    shares = snap.get("shares_outstanding")
    if not fcf_hist or not _ok(shares):
        return None
    fcf0 = float(np.mean(fcf_hist[:2])) if len(fcf_hist) >= 2 else fcf_hist[0]
    if fcf0 <= 0:
        return None
    beta = float(np.clip(snap.get("beta") or 1.0, 0.6, 2.0))
    ke = rf + beta * erp
    debt, cash = float(snap.get("total_debt") or 0), float(snap.get("total_cash") or 0)
    mcap = float(snap.get("market_cap") or (snap["price"] * shares))
    kd = rf + 0.012
    wd = debt / max(debt + mcap, 1)
    wacc = (1 - wd) * ke + wd * kd * (1 - 0.21) + wacc_shift
    g1 = float(np.clip(snap.get("growth_1y") if _ok(snap.get("growth_1y")) else 0.06, -0.05, 0.15))
    g_t = min(TERMINAL_GROWTH + g_shift, rf + 0.01)
    n = 5
    pv, f = 0.0, fcf0
    for t in range(1, n + 1):
        g = g1 + (g_t - g1) * (t - 1) / max(n - 1, 1)
        f *= 1 + g
        pv += f / (1 + wacc) ** t
    tv = f * (1 + g_t) / max(wacc - g_t, 0.03)
    equity = pv + tv / (1 + wacc) ** n - debt + cash
    return equity / shares


def stock_fair_value(snap: dict, rf: float = 0.049, erp: float = 0.042, market_pe: float | None = None) -> dict:
    """Blend of valuation models for one stock. rf and erp are decimals; market_pe is the S&P 500
    multiple used for the market-multiple model (defaults to a long-run 18x)."""
    S = float(snap["price"])
    models: dict[str, dict] = {}
    # forward EPS is used where trailing EPS is distorted by one-offs (spin-offs, write-downs)
    eps_t, eps_f = snap.get("trailing_eps"), snap.get("forward_eps")
    eps = float(eps_f) if _ok(eps_f) and float(eps_f) > 0 else (float(eps_t) if _ok(eps_t) else None)
    g_raw = snap.get("growth_1y") if _ok(snap.get("growth_1y")) else 0.06
    # 1 analyst consensus
    tgt = snap.get("target_median") or snap.get("target_mean")
    if _ok(tgt):
        n = int(snap.get("n_analysts") or 0)
        ke = rf + float(np.clip(snap.get("beta") or 1.0, 0.6, 2.0)) * erp
        models["Analyst consensus (12-mo target, discounted)"] = {
            "value": float(tgt) / (1 + ke), "low": float(snap.get("target_low") or tgt) / (1 + ke),
            "high": float(snap.get("target_high") or tgt) / (1 + ke), "weight": 0.30 * min(n / 10, 1.0),
            "note": f"{n} analysts; rating {snap.get('recommendation', 'n/a')}"}
    # 2 DCF with sensitivity corners
    base = dcf_value(snap, rf, erp)
    if base is not None and base > 0:
        corners = [dcf_value(snap, rf, erp, dw, dg) for dw in (-0.01, 0.01) for dg in (-0.01, 0.01)]
        corners = [c for c in corners if c is not None and c > 0]
        lo, hi = min(corners + [base]), max(corners + [base])
        wide = (hi - lo) / base > 0.6                      # very rate-sensitive valuation: trust it less
        models["Discounted free cash flow (2-stage)"] = {
            "value": base, "low": lo, "high": hi, "weight": 0.15 if wide else 0.30,
            "note": f"FCF base ${np.mean([x for x in snap['fcf_history'][:2]]) / 1e9:,.1f}B, growth "
                    f"{float(np.clip(g_raw, -0.05, 0.15)):.0%} fading to {TERMINAL_GROWTH:.0%}"
                    + ("; wide sensitivity band, weight halved" if wide else "")}
    # market multiple: forward EPS at the S&P 500 multiple, scaled by relative growth (PEG-style, capped)
    if eps is not None and eps > 0:
        pe_mkt = float(market_pe) if _ok(market_pe) else 18.0
        rel = float(np.clip((g_raw if _ok(g_raw) else 0.06) / 0.08, 0.7, 1.3))
        v = eps * pe_mkt * rel
        models["Market multiple (fwd EPS x S&P P/E, growth-scaled)"] = {
            "value": v, "low": eps * pe_mkt * 0.8, "high": eps * pe_mkt * 1.3, "weight": 0.25,
            "note": f"EPS ${eps:.2f} x {pe_mkt:.1f} x {rel:.2f}"}
    # 3 dividend model
    d0, payout = float(snap.get("dividend_rate") or 0), snap.get("payout")
    if d0 > 0 and _ok(payout) and 0.2 <= float(payout) <= 0.8:
        g_d = min(TERMINAL_GROWTH + 0.01, max(0.0, (1 - float(payout)) * 0.10))
        ke = rf + float(np.clip(snap.get("beta") or 1.0, 0.6, 2.0)) * erp
        if ke - g_d > 0.01:
            v = d0 * (1 + g_d) / (ke - g_d)
            models["Gordon dividend model"] = {"value": v, "low": v * 0.8, "high": v * 1.25, "weight": 0.10,
                                               "note": f"dividend ${d0:.2f}, growth {g_d:.1%}, cost of equity {ke:.1%}"}
    # 4 Graham's revised growth formula V = EPS x (8.5 + 2g) x 4.4 / Y, and the Graham number
    if eps is not None and eps > 0:
        g_pct = float(np.clip((g_raw if _ok(g_raw) else 0.06) * 100, 0, 10))
        y_aaa = max(rf * 100 + 0.6, 3.0)                   # AAA corporate yield proxy
        v = eps * (8.5 + 2 * g_pct) * 4.4 / y_aaa
        models["Graham formula EPS x (8.5 + 2g) x 4.4/Y"] = {"value": v, "low": v * 0.85, "high": v * 1.15, "weight": 0.10,
                                                            "note": f"EPS ${eps:.2f}, g {g_pct:.0f}%, Y {y_aaa:.1f}%"}
        bv = snap.get("book_value")
        if _ok(bv) and float(bv) > 0 and _ok(snap.get("pb")) and float(snap["pb"]) < 3:
            gn = math.sqrt(22.5 * eps * float(bv))
            models["Graham number sqrt(22.5 x EPS x BV)"] = {"value": gn, "low": gn * 0.9, "high": gn * 1.1, "weight": 0.10,
                                                              "note": "asset-heavy firm (P/B < 3)"}
    if not models:
        return {"price": S, "fair_value": None, "models": {}, "note": "Not enough data to value this stock."}
    wsum = sum(m["weight"] for m in models.values())
    fv = sum(m["value"] * m["weight"] for m in models.values()) / wsum
    pool = []
    for m in models.values():
        pool += [m["low"], m["value"], m["high"]]
    low, high = float(np.percentile(pool, 25)), float(np.percentile(pool, 75))
    rv = snap.get("realized_vol")
    rv = float(rv) if _ok(rv) else 0.30
    unc = "Low" if rv < 0.20 else "Medium" if rv < 0.30 else "High" if rv < 0.45 else "Very High" if rv < 0.70 else "Extreme"
    order = list(UNCERTAINTY_BANDS)
    lev = float(snap.get("total_debt") or 0) / max(float(snap.get("ebitda") or 0), 1) if _ok(snap.get("ebitda")) else 0
    if lev > 4 or (high - low) / fv > 0.5:
        unc = order[min(order.index(unc) + 1, len(order) - 1)]
    b5, b4, b2, b1 = UNCERTAINTY_BANDS[unc]
    ratio = S / fv
    stars = 5 if ratio <= b5 else 4 if ratio <= b4 else 3 if ratio < b2 else 2 if ratio < b1 else 1
    return {"price": S, "fair_value": fv, "low": low, "high": high, "ratio": ratio, "stars": stars, "uncertainty": unc,
            "consider_buy": fv * b5, "consider_sell": fv * b1, "models": models, "realized_vol": rv,
            "rf": rf, "erp": erp}


# ---------------------------------------------------------------- equity market
def market_valuation(macro: dict, sigma_model_slope: float = -0.30) -> dict:
    fr, ix = macro["fred"], macro["indices"]
    cape = float(macro["cape"])
    dy = float(macro["sp_div_yield"]) / 100
    ey = float(macro["sp_earnings_yield"]) / 100
    r10 = float(fr.get("DGS10", ix.get("^TNX", 4.9))) / 100
    real10 = float(fr.get("DFII10", 2.6)) / 100
    bei = float(fr.get("T10YIE", 2.3)) / 100
    # Research Affiliates building blocks: yield + real growth + valuation change (half reversion over 10y)
    cape_target = cape - 0.5 * (cape - CAPE_MEAN_1960)
    val_change = (cape_target / cape) ** 0.1 - 1
    ra_real = dy + 0.015 + val_change
    ra_nominal = ra_real + bei
    excess_cape_yield = 1 / cape - real10
    fed_spread = ey - r10
    buffett = float(fr["NCBEILQ027S"]) / 1e3 / float(fr["GDP"]) if fr.get("NCBEILQ027S") and fr.get("GDP") else None
    z_cape = math.log(cape / CAPE_MEAN_1960) / 0.30                 # ~1 sd of ln(CAPE) since 1960
    z_fed = -(fed_spread - FED_SPREAD_MEAN) / 0.02
    z_buf = (math.log(buffett / BUFFETT_MEAN) / 0.30) if buffett else z_cape
    z = float(np.mean([z_cape, z_fed, z_buf]))
    tilt = float(np.clip(-0.0075 * z, -TILT_CAP, TILT_CAP))
    vix, v3, v9, v6 = ix.get("^VIX"), ix.get("^VIX3M"), ix.get("^VIX9D"), ix.get("^VIX6M")
    ratio = vix / v3 if vix and v3 else None
    regime = (None if ratio is None else "backwardation: stress, hedges expensive" if ratio > 1.0
              else "steep contango: hedges cheap, complacent" if ratio < 0.88 else "normal contango")
    return {
        "cape": cape, "cape_mean": CAPE_MEAN_1960, "cape_target": cape_target, "ra_real": ra_real, "ra_nominal": ra_nominal,
        "naive_real": 1 / cape, "excess_cape_yield": excess_cape_yield, "earnings_yield": ey, "ten_year": r10,
        "fed_spread": fed_spread, "buffett": buffett, "buffett_mean": BUFFETT_MEAN, "z": z, "tilt": tilt,
        "jpm_us_lc": JPM_US_LC_RETURN, "adjusted_us_lc": JPM_US_LC_RETURN + tilt,
        "real10": real10, "breakeven": bei, "model_inflation": 0.025, "tips_edge": bei - 0.025,
        "hy_spread": fr.get("BAMLH0A0HYM2"), "ig_spread": fr.get("BAMLC0A0CM"), "two_year": fr.get("DGS2"),
        "vix": vix, "vix9d": v9, "vix3m": v3, "vix6m": v6, "vix_ratio": ratio, "vix_regime": regime, "skew": ix.get("^SKEW"),
        "erp": macro["erp"], "put_call": macro["put_call"], "fear_greed": macro["fear_greed"],
    }
