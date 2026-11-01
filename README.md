# Robo-Advisor

A goals-based robo-advisor that follows how large wealth managers handle **know-your-customer (KYC) and
suitability**. It turns a client profile into a **goals-based ETF portfolio**, adds **options and futures
overlays only where the client's account approval allows**, and projects outcomes with a **Monte Carlo** model
that includes taxes, required minimum distributions, Social Security and longevity.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Live market data (S&P 500, VIX, T-bill yield, ETF prices) comes from Yahoo Finance. If it is unavailable, or
**Live market data** is off in the sidebar, the app uses stored values.

## What it does

- **Know the client:** identity and AML checks, then the FINRA 2111 / Reg BI investment profile, a trusted
  contact (FINRA 4512), a 10-question risk questionnaire and an income-gamble ladder.
- **Risk:** willingness (questionnaire + gamble), ability (Merton share with human capital, a 2008/2022
  stress-loss cap and a capacity score) and need (the return the plan requires). The recommendation is the
  lowest of the three; need never raises risk.
- **Portfolio:** 11 model portfolios built by resampled mean-variance optimization on J.P. Morgan's 2026
  capital market assumptions; goal buckets (needs, wants, wishes) and a household glide path.
- **Options and futures:** options approval levels 0-4 and futures eligibility gate collars, covered calls,
  protective puts, buffer ETFs, cash-secured puts and Micro E-mini hedges, priced with Black-Scholes using
  VIX-based volatility and skew.
- **Projections:** fat-tailed (Student-t) returns, 2026 IRS contribution limits, RMDs, taxes, Social Security
  with the projected 2033 trust-fund shortfall, Gompertz-Makeham mortality and Guyton-Klinger guardrails.
- **Compliance:** a Reg BI checklist and a downloadable JSON audit file.

## Three sample clients

| | 30-year-old saver | 55-year-old, $200K | 70-year-old, $2M |
|---|---|---|---|
| Retirement portfolio | 70% equity, gliding to 55% at 65 | 50% equity | 50% equity, split into a 5-year income reserve and a growth bucket |
| Near-term goals | $58.5K house fund in T-bills/short Treasuries | n/a | Grandchildren's college bucket |
| P(success, plan to 95) | ~100% | **66%**: plan must change | ~100% |
| Key advice | Max the match; 50% Roth; no derivatives (not approved) | Don't add risk; save 18%, retire at 70, claim Social Security at 70 or spend 10% less | Covered calls; zero-cost collar on concentrated employer stock; Micro E-mini shorts to trim equity without realizing gains; Roth conversions before RMDs |

Every input beyond each client's headline facts is a documented assumption in [`robo/cases.py`](robo/cases.py)
and editable in the app.

## Code layout

| File | What it does |
|---|---|
| `app.py` | Streamlit front end: Overview, 7-step Profile intake, Risk, Portfolio, Projections, Options & Futures, Compliance, Compare, Methodology |
| `ui.py`, `.streamlit/config.toml` | Visual system: theme, fonts, card and chart components |
| `robo/kyc.py` | Client data model, risk questionnaire and income-gamble ladder |
| `robo/risk.py` | Willingness, ability and need; stress-loss cap; capacity score; glide path |
| `robo/portfolio.py` | Model portfolios by resampled mean-variance optimization |
| `robo/goals.py` | Cash-flow schedule, planning horizon, per-goal equity rules |
| `robo/derivatives.py` | Option pricing, overlay strategies, futures hedge sizing |
| `robo/suitability.py` | Options approval levels, futures eligibility, overlay selection, Reg BI checklist |
| `robo/simulate.py` | Monte Carlo engine |
| `robo/mortality.py` | Gompertz-Makeham mortality fitted to SSA 2022 life expectancies |
| `robo/advisor.py` | Orchestration, ETF implementation, asset location, plan levers, advice text |
| `robo/market_data.py` | Live market data via yfinance, with offline fallback |
| `data/` | Capital market assumptions, correlations, historical statistics, model portfolios |

## Data sources

- **Expected returns:** J.P. Morgan Asset Management, *2026 Long-Term Capital Market Assumptions*.
- **Volatility and correlations:** J.P. Morgan where published; otherwise 200 months of ETF history (2010-2026).
- **Rules:** 2026 IRS limits (Notice 2025-67); SSA 2026 bend points and COLA; 2026 Social Security Trustees
  Report; SSA 2022 period life table; IRS Uniform Lifetime Table; SECURE 2.0 RMD ages.

*Educational project, not investment advice.*
