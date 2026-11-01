"""Optional live market data (Yahoo Finance via yfinance), with a safe offline fallback.

Used for: S&P 500 level, VIX and the 13-week T-bill yield (option pricing and futures
sizing), and current ETF prices (share counts in the implementation table). The
long-term return assumptions never come from live prices.
"""
from __future__ import annotations

import warnings
from datetime import datetime

from .derivatives import Market

warnings.filterwarnings("ignore")


def live_market(timeout_ok: bool = True) -> Market:
    try:
        import yfinance as yf
        d = yf.download(["^GSPC", "^VIX", "^IRX"], period="10d", progress=False, auto_adjust=False)["Close"]
        d = d.dropna(how="all").ffill()
        last = d.iloc[-1]
        return Market(spot=float(last["^GSPC"]), vix=float(last["^VIX"]), rate=float(last["^IRX"]) / 100,
                      live=True, asof=f"Yahoo Finance close {d.index[-1].date()}")
    except Exception as exc:  # network down, rate limited, API change
        if not timeout_ok:
            raise
        return Market(asof=f"offline defaults ({type(exc).__name__})")


def etf_prices(tickers: list[str]) -> dict[str, float]:
    try:
        import yfinance as yf
        d = yf.download(tickers, period="10d", progress=False, auto_adjust=False)["Close"].ffill()
        last = d.iloc[-1]
        return {t: float(last[t]) for t in tickers if t in last and last[t] == last[t]}
    except Exception:
        return {}


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
