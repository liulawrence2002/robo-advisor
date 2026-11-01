"""Capital market assumptions (CMAs).

Expected returns: J.P. Morgan Asset Management, 2026 Long-Term Capital Market
Assumptions (10-15 year horizon, compound/geometric, USD, published Oct 20 2025).
Volatility: JPM's published risk/return chart where available, otherwise the
historical volatility of a liquid proxy (2010-2026 monthly). Correlations: historical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

INFLATION = 0.025  # JPM 2026 LTCMA, US inflation assumption
CASH_RATE = 0.031  # JPM 2026 LTCMA, USD cash

EQUITY_KEYS = ("us_lc", "us_sc", "intl_dev", "em", "reit")
ALT_KEYS = ("commod", "gold", "mgd_fut")


@dataclass(frozen=True)
class CMA:
    keys: tuple[str, ...]
    names: tuple[str, ...]
    groups: tuple[str, ...]
    etf: tuple[str, ...]
    alt_etf: tuple[str, ...]
    mu_geo: np.ndarray          # expected compound (geometric) nominal return
    vol: np.ndarray             # annual volatility
    corr: np.ndarray
    sources: tuple[str, ...]
    vol_source: tuple[str, ...]
    inflation: float = INFLATION
    cash_rate: float = CASH_RATE
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.keys)

    @property
    def cov(self) -> np.ndarray:
        return np.outer(self.vol, self.vol) * self.corr

    @property
    def mu_arith(self) -> np.ndarray:
        """Arithmetic mean consistent with a lognormal whose median growth is mu_geo."""
        return (1 + self.mu_geo) * np.exp(self.log_sigma ** 2 / 2) - 1

    @property
    def log_mu(self) -> np.ndarray:
        return np.log1p(self.mu_geo)

    @property
    def log_sigma(self) -> np.ndarray:
        # sd of log returns for a lognormal with the given arithmetic sd (approx. around mu_geo)
        return np.sqrt(np.log1p((self.vol / (1 + self.mu_geo)) ** 2))

    def idx(self, key: str) -> int:
        return self.keys.index(key)

    def mask(self, keys) -> np.ndarray:
        return np.array([k in keys for k in self.keys])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Asset class": self.names, "Key": self.keys, "ETF": self.etf, "Alt ETF": self.alt_etf,
            "Exp. return (geo)": self.mu_geo, "Exp. return (arith)": self.mu_arith,
            "Volatility": self.vol, "Return source": self.sources, "Vol source": self.vol_source,
        })


def _nearest_psd(c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    c2 = v @ np.diag(np.clip(w, eps, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


@lru_cache(maxsize=4)
def load_cma(corr_file: str = "correlation.csv", hist_file: str = "hist_stats.csv") -> CMA:
    df = pd.read_csv(DATA / "cma_jpm_2026.csv")
    hist = pd.read_csv(DATA / hist_file).set_index("key")
    corr = pd.read_csv(DATA / corr_file, index_col=0)
    keys = tuple(df["key"])
    corr = corr.loc[list(keys), list(keys)].to_numpy()

    vol, vsrc = [], []
    for _, r in df.iterrows():
        if pd.notna(r["vol_jpm"]):
            vol.append(float(r["vol_jpm"])); vsrc.append("JPM 2026 LTCMA (Exh. 8A)")
        else:
            vol.append(float(hist.loc[r["key"], "hist_vol"])); vsrc.append(f"History {r['history_proxy']} 2010-26")
    return CMA(
        keys=keys, names=tuple(df["name"]), groups=tuple(df["group"]),
        etf=tuple(df["etf"]), alt_etf=tuple(df["alt_etf"]),
        mu_geo=df["exp_return"].to_numpy(float), vol=np.array(vol), corr=_nearest_psd(corr),
        sources=tuple(df["return_source"]), vol_source=tuple(vsrc),
        meta={"hist_start": str(hist["start"].iloc[0]), "hist_end": str(hist["end"].iloc[0]),
              "hist_months": int(hist["months"].iloc[0])},
    )


# Historical stress scenarios: approximate calendar-year total returns of the index each
# asset class tracks (S&P 500, Russell 2000, MSCI EAFE/EM, FTSE NAREIT, Bloomberg US Agg/
# Treasury/TIPS/Corporate/HY, Bloomberg Commodity, gold, SG/Barclay CTA indices).
STRESS_SCENARIOS = {
    "2008 global financial crisis": {
        "cash": 0.015, "ust_int": 0.114, "tips": -0.024, "us_agg": 0.052, "us_ig": -0.049, "us_hy": -0.262,
        "ust_long": 0.240, "us_lc": -0.370, "us_sc": -0.338, "intl_dev": -0.434, "em": -0.533,
        "reit": -0.377, "commod": -0.357, "gold": 0.058, "mgd_fut": 0.140},
    "2022 stock-bond inflation shock": {
        "cash": 0.015, "ust_int": -0.077, "tips": -0.118, "us_agg": -0.130, "us_ig": -0.158, "us_hy": -0.112,
        "ust_long": -0.293, "us_lc": -0.181, "us_sc": -0.204, "intl_dev": -0.145, "em": -0.201,
        "reit": -0.249, "commod": 0.161, "gold": -0.003, "mgd_fut": 0.201},
}


# Multi-year historical episodes replayed as the first years of a projection (nominal calendar-year
# total returns of the asset-class indices, approximate; CPI inflation for the same years). Where
# an index did not exist (TIPS before 1997, EM before 1988, HY before 1983, CTA before 1980) the
# value is a documented proxy. Sources: S&P/Ibbotson SBBI, MSCI, FTSE Nareit, Bloomberg indices,
# S&P GSCI / BCOM, LBMA gold, BarclayHedge CTA index, BLS CPI-U.
STRESS_PATHS: dict[str, dict] = {
    "Dot-com bust 2000-02": {"inflation": [0.034, 0.016, 0.024], "years": [
        {"cash": 0.060, "ust_int": 0.103, "tips": 0.132, "us_agg": 0.116, "us_ig": 0.094, "us_hy": -0.059, "ust_long": 0.203,
         "us_lc": -0.091, "us_sc": -0.030, "intl_dev": -0.142, "em": -0.306, "reit": 0.264, "commod": 0.318, "gold": -0.054, "mgd_fut": 0.079},
        {"cash": 0.038, "ust_int": 0.076, "tips": 0.079, "us_agg": 0.084, "us_ig": 0.104, "us_hy": 0.053, "ust_long": 0.042,
         "us_lc": -0.119, "us_sc": 0.025, "intl_dev": -0.214, "em": -0.026, "reit": 0.139, "commod": -0.195, "gold": 0.025, "mgd_fut": 0.008},
        {"cash": 0.017, "ust_int": 0.096, "tips": 0.166, "us_agg": 0.103, "us_ig": 0.101, "us_hy": -0.014, "ust_long": 0.168,
         "us_lc": -0.221, "us_sc": -0.205, "intl_dev": -0.159, "em": -0.062, "reit": 0.038, "commod": 0.259, "gold": 0.248, "mgd_fut": 0.124}]},
    "Global financial crisis 2008-09": {"inflation": [0.001, 0.027], "years": [
        {"cash": 0.015, "ust_int": 0.114, "tips": -0.024, "us_agg": 0.052, "us_ig": -0.049, "us_hy": -0.262, "ust_long": 0.240,
         "us_lc": -0.370, "us_sc": -0.338, "intl_dev": -0.434, "em": -0.533, "reit": -0.377, "commod": -0.357, "gold": 0.058, "mgd_fut": 0.140},
        {"cash": 0.002, "ust_int": -0.014, "tips": 0.114, "us_agg": 0.059, "us_ig": 0.187, "us_hy": 0.582, "ust_long": -0.129,
         "us_lc": 0.265, "us_sc": 0.272, "intl_dev": 0.318, "em": 0.785, "reit": 0.280, "commod": 0.189, "gold": 0.244, "mgd_fut": -0.001}]},
    "Inflation shock 2022": {"inflation": [0.065], "years": [
        {"cash": 0.015, "ust_int": -0.077, "tips": -0.118, "us_agg": -0.130, "us_ig": -0.158, "us_hy": -0.112, "ust_long": -0.293,
         "us_lc": -0.181, "us_sc": -0.204, "intl_dev": -0.145, "em": -0.201, "reit": -0.249, "commod": 0.161, "gold": -0.003, "mgd_fut": 0.201}]},
    "Stagflation 1973-74": {"inflation": [0.088, 0.122], "years": [
        {"cash": 0.069, "ust_int": 0.046, "tips": 0.050, "us_agg": 0.011, "us_ig": 0.011, "us_hy": -0.050, "ust_long": -0.011,
         "us_lc": -0.147, "us_sc": -0.309, "intl_dev": -0.149, "em": -0.250, "reit": -0.155, "commod": 0.750, "gold": 0.730, "mgd_fut": 0.200},
        {"cash": 0.080, "ust_int": 0.057, "tips": 0.080, "us_agg": -0.031, "us_ig": -0.031, "us_hy": -0.100, "ust_long": 0.044,
         "us_lc": -0.265, "us_sc": -0.199, "intl_dev": -0.222, "em": -0.300, "reit": -0.214, "commod": 0.395, "gold": 0.660, "mgd_fut": 0.200}]},
}


def stress_path_returns(name: str, cma: CMA) -> tuple[np.ndarray, np.ndarray]:
    """(nominal returns (years, assets), inflation (years,)) for a named historical episode."""
    ep = STRESS_PATHS[name]
    R = np.array([[yr[k] for k in cma.keys] for yr in ep["years"]])
    return R, np.array(ep["inflation"])


def stress_returns(w: np.ndarray, cma: CMA) -> dict[str, float]:
    return {name: float(sum(w[cma.idx(k)] * r for k, r in shocks.items()))
            for name, shocks in STRESS_SCENARIOS.items()}


def portfolio_stats(w: np.ndarray, cma: CMA) -> dict:
    mu_a = float(w @ cma.mu_arith)
    var = float(w @ cma.cov @ w)
    vol = var ** 0.5
    geo = (1 + mu_a) * np.exp(-np.log1p(var / (1 + mu_a) ** 2) / 2) - 1
    return {"exp_return_arith": mu_a, "exp_return_geo": float(geo), "vol": vol,
            "sharpe": (mu_a - cma.cash_rate) / vol if vol > 0 else 0.0,
            "real_return_geo": (1 + geo) / (1 + cma.inflation) - 1}
