"""Live market data with safe offline fallbacks.

Sources (all keyless):
  Yahoo Finance via yfinance  S&P 500, VIX term structure, Treasury yields, ETF prices, option
                              chains (bid/ask -> our own implied vols), single-stock fundamentals
                              and analyst targets
  FRED (fredgraph.csv)        10y nominal and real yields, breakeven inflation, credit spreads,
                              GDP and corporate equity value (Buffett indicator)
  U.S. Treasury               daily par and real yield curves
  multpl.com                  S&P 500 Shiller CAPE, P/E, dividend yield
  Damodaran (NYU Stern)       implied equity risk premium
  CBOE                        daily put/call ratios
  CNN                         Fear & Greed index

Every fetch is wrapped: on any failure the caller gets None or the committed snapshot in
FALLBACK (values as of 2026-10-30) with `live=False`, so the app renders offline and when
Yahoo throttles shared cloud IPs. Long-term return assumptions never come from live prices.
"""
from __future__ import annotations

import datetime as dt
import io
import math
import warnings
import zipfile

import numpy as np
import pandas as pd

from .derivatives import Market, bs_delta, bs_price

warnings.filterwarnings("ignore")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"}
INDEX_TICKERS = ["^GSPC", "^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^SKEW", "^IRX", "^FVX", "^TNX", "^TYX", "ES=F", "MES=F"]

# Committed snapshot (close of 2026-10-30) used when a live source is unavailable
FALLBACK = {
    "indices": {"^GSPC": 7580.07, "^VIX": 19.42, "^VIX9D": 19.25, "^VIX3M": 20.78, "^VIX6M": 22.20, "^SKEW": 149.76,
                "^IRX": 3.973, "^FVX": 4.721, "^TNX": 4.929, "^TYX": 5.309, "ES=F": 7614.0, "MES=F": 7614.0},
    "fred": {"DGS10": 4.98, "DGS2": 4.61, "DFII10": 2.63, "T10YIE": 2.34, "T5YIE": 2.46, "BAMLH0A0HYM2": 3.26,
             "BAMLC0A0CM": 0.89, "DTB3": 4.00, "GDP": 32563.0, "NCBEILQ027S": 83050847.0},
    "treasury": {"1 Mo": 4.06, "3 Mo": 4.16, "6 Mo": 4.28, "1 Yr": 4.40, "2 Yr": 4.56, "5 Yr": 4.71, "10 Yr": 4.94,
                 "20 Yr": 5.34, "30 Yr": 5.31},
    "treasury_real": {"5 Yr": 2.25, "10 Yr": 2.60, "30 Yr": 3.08},
    "cape": 40.13, "sp_pe": 25.86, "sp_div_yield": 1.07, "sp_earnings_yield": 3.87,
    "erp": {"erp_t12m": 3.70, "erp_sustainable": 4.20, "erp_smoothed": 5.67, "tbond": 5.29, "expected_return": 8.99,
            "date": "2026-10-01"},
    "put_call": {"TOTAL": 0.90, "INDEX": 1.03, "EQUITY": 0.66, "VIX": 0.47, "date": "2026-10-30"},
    "fear_greed": {"score": 26.2, "rating": "fear", "previous_1_week": 16.7, "previous_1_month": 30.3},
    "asof": "snapshot 2026-10-30",
}


def _yf():
    import yfinance as yf
    try:
        yf.config.network.retries = 0      # fail fast on shared cloud IPs that Yahoo throttles; fallbacks take over
    except Exception:
        pass
    return yf


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- indices, yields, prices
def index_snapshot() -> dict:
    """Last closes of the index/yield/futures tickers in one request. Yields are in percent."""
    try:
        yf = _yf()
        d = yf.download(INDEX_TICKERS, period="10d", progress=False, auto_adjust=False, group_by="column")["Close"]
        last = d.ffill().iloc[-1]
        out = {k: float(last[k]) for k in INDEX_TICKERS if k in last and np.isfinite(last[k])}
        if "^GSPC" not in out or out["^GSPC"] <= 0:
            raise ValueError("no S&P close")
        out["asof"] = f"Yahoo Finance close {d.index[-1].date()}"
        out["live"] = True
        return out
    except Exception as exc:
        out = dict(FALLBACK["indices"]); out["asof"] = f"{FALLBACK['asof']} ({type(exc).__name__})"; out["live"] = False
        return out


def live_market(timeout_ok: bool = True) -> Market:
    snap = index_snapshot()
    if not snap["live"] and not timeout_ok:
        raise RuntimeError(snap["asof"])
    return Market(spot=snap["^GSPC"], vix=snap.get("^VIX"), rate=snap.get("^IRX", 3.97) / 100,
                  live=snap["live"], asof=snap["asof"])


def etf_prices(tickers: list[str]) -> dict[str, float]:
    try:
        yf = _yf()
        d = yf.download(tickers, period="10d", progress=False, auto_adjust=False)["Close"].ffill()
        last = d.iloc[-1]
        return {t: float(last[t]) for t in tickers if t in last and last[t] == last[t]}
    except Exception:
        return {}


# ---------------------------------------------------------------- option chains
def third_fridays(expiries) -> list[str]:
    return [e for e in expiries if (d := dt.date.fromisoformat(e)).weekday() == 4 and 15 <= d.day <= 21]


def nearest_expiry(expiries, months: float) -> str | None:
    if not expiries:
        return None
    target = dt.date.today() + dt.timedelta(days=round(30.44 * months))
    pool = third_fridays(expiries) or list(expiries)
    return min(pool, key=lambda e: abs((dt.date.fromisoformat(e) - target).days))


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, kind: str) -> float:
    from scipy.optimize import brentq
    try:
        return float(brentq(lambda s: bs_price(S, K, T, r, q, s, kind) - price, 1e-4, 5.0, xtol=1e-6))
    except ValueError:
        return float("nan")


def clean_chain(df: pd.DataFrame, S: float, T: float, r: float, q: float, kind: str,
                max_spread: float = 0.5, min_oi: int = 1) -> pd.DataFrame:
    """Keep quoted, liquid contracts and recompute implied vol from the bid/ask mid.
    Yahoo's own impliedVolatility column is stale or a 1e-5 placeholder on many strikes."""
    d = df.copy()
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d["spread_rel"] = (d["ask"] - d["bid"]) / d["mid"].replace(0, np.nan)
    intr = np.maximum((S - d["strike"]) if kind == "call" else (d["strike"] - S), 0)
    ok = ((d["bid"] > 0) & (d["ask"] > 0) & (d["spread_rel"] < max_spread) & (d["openInterest"].fillna(0) >= min_oi)
          & (d["mid"] >= intr * 0.98))
    d = d[ok].copy()
    d["iv_mid"] = [implied_vol(p, S, K, T, r, q, kind) for p, K in zip(d["mid"], d["strike"])]
    d = d.dropna(subset=["iv_mid"])
    d = d[(d["iv_mid"] > 0.03) & (d["iv_mid"] < 2.5)]
    d["delta"] = [bs_delta(S, K, T, r, q, s, kind) for K, s in zip(d["strike"], d["iv_mid"])]
    d["logm"] = np.log(d["strike"] / S)
    d["moneyness"] = d["strike"] / S
    return d.sort_values("strike")


def _interp(d: pd.DataFrame, x: str, y: str, target: float) -> float:
    if d.empty:
        return float("nan")
    d = d.sort_values(x)
    if not (d[x].min() <= target <= d[x].max()):
        return float("nan")
    return float(np.interp(target, d[x], d[y]))


def option_chain_summary(symbol: str = "SPY", months=(3, 6, 12), rate: float | None = None) -> dict | None:
    """Cleaned option chains at ~3, 6 and 12 months with ATM vol, 25-delta skew and the fitted
    smile slope d(IV)/d(ln K). Returns None when Yahoo is unavailable."""
    try:
        yf = _yf()
        t = yf.Ticker(symbol)
        exps = list(t.options)
        if not exps:
            return None
        info = {}
        try:
            info = t.fast_info or {}
            S = float(info["lastPrice"])
        except Exception:
            S = None
        picked, out = [], {"symbol": symbol, "expiries": [], "asof": stamp()}
        for m in months:
            e = nearest_expiry(exps, m)
            if e and e not in picked:
                picked.append(e)
        q = 0.0
        try:
            dy = t.info.get("dividendYield")
            if dy is not None:
                q = float(dy) / 100 if float(dy) > 0.5 else float(dy)      # Yahoo mixes percent and fraction units
        except Exception:
            q = 0.012 if symbol in ("SPY", "VOO", "IVV") else 0.0
        for e in picked:
            oc = t.option_chain(e)
            und = oc.underlying or {}
            if S is None:
                S = float(und.get("regularMarketPrice") or und.get("postMarketPrice"))
            T = max((dt.date.fromisoformat(e) - dt.date.today()).days, 1) / 365.0
            r = rate if rate is not None else 0.04
            calls = clean_chain(oc.calls, S, T, r, q, "call")
            puts = clean_chain(oc.puts, S, T, r, q, "put")
            atm = np.nanmean([_interp(calls, "strike", "iv_mid", S), _interp(puts, "strike", "iv_mid", S)])
            put25, call25 = _interp(puts, "delta", "iv_mid", -0.25), _interp(calls, "delta", "iv_mid", 0.25)
            otm = pd.concat([puts[puts["strike"] < S], calls[calls["strike"] > S]])
            otm = otm[np.abs(otm["logm"]) < 0.25]
            slope = float(np.polyfit(otm["logm"], otm["iv_mid"], 1)[0]) if len(otm) >= 6 else float("nan")
            out["expiries"].append({"expiry": e, "T": T, "atm_iv": float(atm), "put25_iv": put25, "call25_iv": call25,
                                    "skew25": put25 - call25 if np.isfinite(put25) and np.isfinite(call25) else float("nan"),
                                    "slope": slope, "n_calls": len(calls), "n_puts": len(puts),
                                    "calls": calls[["strike", "moneyness", "bid", "ask", "mid", "iv_mid", "delta", "openInterest", "volume"]],
                                    "puts": puts[["strike", "moneyness", "bid", "ask", "mid", "iv_mid", "delta", "openInterest", "volume"]]})
        out["spot"], out["q"] = S, q
        return out if out["expiries"] else None
    except Exception:
        return None


# ---------------------------------------------------------------- single stock
def stock_snapshot(ticker: str) -> dict | None:
    """Price, analyst targets, multiples, balance-sheet items, FCF history, growth, realised
    volatility and the ~6-month ATM implied vol for one stock."""
    if not ticker:
        return None
    try:
        yf = _yf()
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return None
        px = float(info.get("currentPrice") or info.get("regularMarketPrice"))
        hist = t.history(period="1y")["Close"]
        rv = float(np.log(hist).diff().std() * math.sqrt(252)) if len(hist) > 60 else float("nan")
        fcf_hist = []
        try:
            cf = t.cashflow
            if "Free Cash Flow" in cf.index:
                fcf_hist = [float(x) for x in cf.loc["Free Cash Flow"].dropna().tolist()[:4]]
        except Exception:
            pass
        growth = None
        try:
            ee = t.earnings_estimate
            growth = float(ee.loc["+1y", "growth"])
        except Exception:
            growth = info.get("earningsGrowth")
        dy = info.get("dividendYield")
        div_yield = (float(dy) / 100 if dy and float(dy) > 0.5 else float(dy or 0.0))
        snap = {
            "ticker": ticker.upper(), "name": info.get("shortName") or ticker, "price": px, "sector": info.get("sector"),
            "industry": info.get("industry"), "beta": info.get("beta"), "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"), "total_debt": info.get("totalDebt") or 0.0, "total_cash": info.get("totalCash") or 0.0,
            "trailing_pe": info.get("trailingPE"), "forward_pe": info.get("forwardPE"), "pb": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"), "trailing_eps": info.get("trailingEps"), "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"), "dividend_rate": info.get("dividendRate") or 0.0, "dividend_yield": div_yield,
            "payout": info.get("payoutRatio"), "fcf_history": fcf_hist, "growth_1y": growth,
            "target_mean": info.get("targetMeanPrice"), "target_median": info.get("targetMedianPrice"),
            "target_low": info.get("targetLowPrice"), "target_high": info.get("targetHighPrice"),
            "n_analysts": info.get("numberOfAnalystOpinions"), "recommendation": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"), "avg_volume": info.get("averageVolume"),
            "realized_vol": rv, "ebitda": info.get("ebitda"), "week52_low": info.get("fiftyTwoWeekLow"),
            "week52_high": info.get("fiftyTwoWeekHigh"), "asof": stamp(),
        }
        try:
            cal = t.calendar or {}
            ed = cal.get("Earnings Date")
            snap["next_earnings"] = str(ed[0]) if ed else None
        except Exception:
            snap["next_earnings"] = None
        chain = option_chain_summary(ticker, months=(6,))
        snap["implied_vol"] = chain["expiries"][0]["atm_iv"] if chain and chain["expiries"] else None
        snap["skew_slope"] = chain["expiries"][0]["slope"] if chain and chain["expiries"] else None
        return snap
    except Exception:
        return None


# ---------------------------------------------------------------- macro / valuation inputs
def fred(ids: list[str], start: str = "2024-01-01") -> dict | None:
    """Latest values from FRED's keyless CSV endpoint (needs a browser TLS fingerprint)."""
    try:
        from curl_cffi import requests as crq
        r = crq.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": ",".join(ids), "cosd": start},
                    impersonate="chrome", timeout=12)
        r.raise_for_status()
        raw = r.content

        def _csv(b):
            return pd.read_csv(io.BytesIO(b), na_values=".", parse_dates=["observation_date"]).set_index("observation_date")
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                df = pd.concat([_csv(z.read(n)) for n in z.namelist() if n.lower().endswith(".csv")], axis=1).sort_index()
        else:
            df = _csv(raw)
        last = df.ffill().iloc[-1]
        out = {k: float(last[k]) for k in ids if k in last and np.isfinite(last[k])}
        out["asof"] = str(df.index[-1].date())
        return out
    except Exception:
        return None


def treasury_curve(real: bool = False) -> dict | None:
    year = dt.date.today().year
    kind = "daily_treasury_real_yield_curve" if real else "daily_treasury_yield_curve"
    url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
           f"{year}/all?type={kind}&field_tdr_date_value={year}&page&_format=csv")
    try:
        import requests
        r = requests.get(url, headers=UA, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        row = df.iloc[0]
        out = {c.replace("YR", "Yr").replace("MO", "Mo").replace("  ", " ").strip(): float(row[c])
               for c in df.columns if c != "Date" and pd.notna(row[c])}
        out = {k.replace("1 Mo", "1 Mo"): v for k, v in out.items()}
        out["asof"] = str(row["Date"])
        return out
    except Exception:
        return None


def multpl_table(slug: str) -> tuple[float, str] | None:
    try:
        import requests
        r = requests.get(f"https://www.multpl.com/{slug}/table/by-month", headers=UA, timeout=10)
        r.raise_for_status()
        df = pd.read_html(io.StringIO(r.text))[0]
        val = str(df.iloc[0, 1]).replace("†", "").replace("%", "").strip()
        return float(val), str(df.iloc[0, 0])
    except Exception:
        return None


def damodaran_erp() -> dict | None:
    try:
        import requests
        r = requests.get("https://pages.stern.nyu.edu/~adamodar/pc/implprem/ERPbymonth.xlsx", headers=UA, timeout=15)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="Historical ERP", header=0)
        cols = {str(c).strip(): c for c in df.columns}
        first = df.columns[0]
        rows = df[pd.to_datetime(df[first], errors="coerce").notna()]
        last = rows.iloc[-1]
        pct = lambda name: float(last[cols[name]]) * 100 if name in cols and pd.notna(last[cols[name]]) else None
        return {"date": str(pd.to_datetime(last[first]).date()), "sp": float(last[cols["S&P 500"]]),
                "tbond": pct("T.Bond Rate"), "erp_t12m": pct("ERP (T12m)"),
                "erp_sustainable": pct("ERP (T12 m with sustainable payout)"), "erp_smoothed": pct("ERP (Smoothed)")}
    except Exception:
        return None


def cboe_put_call(max_back: int = 4) -> dict | None:
    try:
        import requests
        d = dt.date.today()
        for _ in range(max_back):
            r = requests.get(f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{d:%Y-%m-%d}_daily_options",
                             headers=UA, timeout=6)
            if r.ok:
                j = r.json()
                out = {x["name"].replace(" PUT/CALL RATIO", "").replace("CBOE VOLATILITY INDEX (VIX)", "VIX"): float(x["value"])
                       for x in j.get("ratios", [])}
                out["date"] = str(d)
                return out
            d -= dt.timedelta(days=1)
    except Exception:
        pass
    return None


def fear_greed() -> dict | None:
    try:
        import requests
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers=UA, timeout=8)
        r.raise_for_status()
        fg = r.json()["fear_and_greed"]
        return {"score": float(fg["score"]), "rating": fg["rating"], "previous_1_week": float(fg["previous_1_week"]),
                "previous_1_month": float(fg["previous_1_month"])}
    except Exception:
        return None


def macro_snapshot(use_live: bool = True) -> dict:
    """Everything the valuation dashboard needs, with per-source live flags and fallbacks."""
    out = {"sources": {}}
    def pick(name, fn, fallback):
        val = fn() if use_live else None
        out["sources"][name] = val is not None
        return val if val is not None else fallback
    out["indices"] = pick("Yahoo indices", lambda: (lambda s: s if s.get("live") else None)(index_snapshot()), FALLBACK["indices"])
    out["fred"] = pick("FRED", lambda: fred(["DGS10", "DGS2", "DFII10", "T10YIE", "T5YIE", "BAMLH0A0HYM2", "BAMLC0A0CM", "DTB3"]),
                       FALLBACK["fred"])
    gdp = pick("FRED GDP", lambda: fred(["GDP", "NCBEILQ027S"], "2020-01-01"), {"GDP": FALLBACK["fred"]["GDP"], "NCBEILQ027S": FALLBACK["fred"]["NCBEILQ027S"]})
    out["fred"].update({k: v for k, v in gdp.items() if k in ("GDP", "NCBEILQ027S")})
    out["treasury"] = pick("Treasury par curve", treasury_curve, FALLBACK["treasury"])
    out["treasury_real"] = pick("Treasury real curve", lambda: treasury_curve(real=True), FALLBACK["treasury_real"])
    cape = pick("multpl CAPE", lambda: multpl_table("shiller-pe"), (FALLBACK["cape"], "Oct 2026"))
    pe = pick("multpl P/E", lambda: multpl_table("s-p-500-pe-ratio"), (FALLBACK["sp_pe"], ""))
    dyld = pick("multpl dividend yield", lambda: multpl_table("s-p-500-dividend-yield"), (FALLBACK["sp_div_yield"], ""))
    out["cape"], out["cape_date"] = cape
    out["sp_pe"], out["sp_div_yield"] = pe[0], dyld[0]
    out["sp_earnings_yield"] = 100 / pe[0] if pe[0] else FALLBACK["sp_earnings_yield"]
    out["erp"] = pick("Damodaran ERP", damodaran_erp, FALLBACK["erp"])
    out["put_call"] = pick("CBOE put/call", cboe_put_call, FALLBACK["put_call"])
    out["fear_greed"] = pick("CNN Fear & Greed", fear_greed, FALLBACK["fear_greed"])
    out["asof"] = stamp() if use_live else FALLBACK["asof"]
    return out
