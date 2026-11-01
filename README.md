# Robo-Advisor

A goals-based robo-advisor that follows how large wealth managers handle **account opening, know-your-customer (KYC)
and suitability**. It turns a client profile into a **goals-based ETF portfolio**, adds **options and futures overlays
only where the client's account approval allows**, prices them against the **live option chain**, and projects outcomes
with a **Monte Carlo** model that includes fees, 2026 tax brackets, required minimum distributions, Social Security,
longevity, long-term care and replays of past crises.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Live data (S&P 500, VIX term structure, Treasury yields, ETF prices, option chains, stock fundamentals, FRED, Treasury
curves, Shiller CAPE, Damodaran ERP, CBOE put/call, CNN Fear & Greed) is fetched without API keys. If a source is
unavailable, or **Live market data** is off in the sidebar, the app uses a committed snapshot.

## What it does

- **Open the account:** identity verification (CIP), OFAC screening, account registration and beneficial owners,
  source of wealth, politically exposed persons with enhanced due diligence, FATCA/CRS tax residency, trusted contact,
  36-month profile refresh. A failed check blocks the account; a pending review holds trading.
- **Know the client:** the FINRA 2111 / Reg BI investment profile, experience by product (FINRA 2360), disclosures,
  insider status (Rule 144 / 10b5-1), a 10-question risk questionnaire and an income-gamble ladder, with consistency
  checks on contradictory answers and a client override that is honoured only when more conservative.
- **Risk:** willingness (questionnaire + gamble), ability (Merton share with human capital, a 2008/2022 stress-loss cap
  and a capacity score) and need (the return the plan requires). The recommendation is the lowest of the three; need
  never raises risk.
- **Portfolio:** 11 model portfolios built by resampled mean-variance optimization on J.P. Morgan's 2026 capital market
  assumptions; a T-bill/short-Treasury/TIPS sleeve for money needed within two years; goal buckets (needs, wants,
  wishes); a household glide path; asset location by tax wrapper; tax-loss-harvesting pairs.
- **Options and futures:** approval levels 0-4 and futures eligibility gate collars, covered calls, protective puts,
  buffer ETFs, cash-secured puts and beta-sized Micro E-mini hedges. The SPY option chain is pulled live, implied vols
  are recomputed from bid/ask mid-prices and compared with the model skew, and a collar is priced from the chain. A
  concentrated single-stock position gets live implied vol, a model fair value with Morningstar-style stars, a staged
  sale plan, Rule 144 limits and 10b5-1 / exchange-fund / donor-advised-fund notes.
- **Projections:** fat-tailed (Student-t) returns net of fees, 2026 IRS contribution limits, RMDs, federal brackets and
  state tax, a Roth-conversion lever, Social Security with the projected 2033 trust-fund shortfall and a claiming-age
  optimiser, Gompertz-Makeham mortality for both spouses, Guyton-Klinger guardrails, a long-term-care shock, and replays
  of 1973-74, 2000-02, 2008-09 and 2022 as the opening years of the plan.
- **Markets & valuation:** Shiller CAPE building blocks, excess CAPE yield, Fed model, Buffett indicator, implied equity
  risk premium, Treasury par and real curves, breakevens, credit spreads, VIX term structure, SKEW, put/call ratios and
  sentiment, with a bounded valuation tilt shown next to the J.P. Morgan anchor.
- **Compliance:** a Reg BI checklist with evidence, reasonably-available-alternatives comparison, account notes, a JSON
  audit record and a downloadable client report (investment policy statement).

## Three sample clients

| | 30-year-old saver | 55-year-old, $200K | 70-year-old, $2M |
|---|---|---|---|
| Retirement portfolio | 70% equity, gliding to 51% at 65 | 50% equity | 50% equity, split into a 5-year income reserve and a growth bucket |
| Near-term goals | $58.5K house fund in T-bills, short Treasuries and TIPS | n/a | Grandchildren's college bucket |
| P(success, plan to 95), after fees | ~100% | **54%**: plan must change | ~100% |
| Key advice | Max the match; 50% Roth; no derivatives (not approved) | Don't add risk; save 18%, retire at 70, claim Social Security at 70 or spend 10% less | Covered calls; zero-cost collar on concentrated stock priced off the live chain; Micro E-mini shorts to trim equity without realizing gains |

Every input beyond each client's headline facts is a documented assumption in [`robo/cases.py`](robo/cases.py)
and editable in the app.

## Code layout

| File | What it does |
|---|---|
| `app.py` | Streamlit front end: Overview, 7-step Profile, Risk, Portfolio, Projections, Options & Futures, Markets & Valuation, Compliance, Compare, Methodology |
| `ui.py`, `.streamlit/config.toml` | Visual system: theme, fonts, card and chart components |
| `robo/kyc.py` | Client data model, account gate, risk questionnaire, consistency checks |
| `robo/risk.py` | Willingness, ability and need; stress-loss cap; capacity score; client override; glide path |
| `robo/portfolio.py` | Model portfolios by resampled mean-variance optimization |
| `robo/goals.py` | Cash-flow schedule, planning horizon, Social Security, per-goal equity rules |
| `robo/derivatives.py` | Option pricing, overlay strategies, beta-sized futures hedges |
| `robo/suitability.py` | Options approval levels, futures eligibility, insider rules, concentrated-stock plan, Reg BI checklist |
| `robo/simulate.py` | Monte Carlo engine (fees, taxes, RMDs, Roth conversions, guardrails, LTC shock, historical replays) |
| `robo/tax.py` | 2026 federal brackets and gross-up |
| `robo/mortality.py` | Gompertz-Makeham mortality fitted to SSA 2022 life expectancies |
| `robo/retirement.py` | Social Security rules and claiming analysis, RMDs, contribution limits |
| `robo/advisor.py` | Orchestration, ETF implementation, asset location, plan levers, scenario table, advice text |
| `robo/market_data.py` | Live data via yfinance, FRED, Treasury, multpl, Damodaran, CBOE and CNN, with offline fallbacks |
| `robo/valuation.py` | Single-stock fair value (analyst, DCF, dividend, Graham, market multiple) and market valuation |
| `robo/report.py` | Downloadable HTML client report |
| `data/` | Capital market assumptions, correlations, historical statistics, model portfolios |

## Data sources

- **Expected returns:** J.P. Morgan Asset Management, *2026 Long-Term Capital Market Assumptions*.
- **Volatility and correlations:** J.P. Morgan where published; otherwise 200 months of ETF history (2010-2026).
- **Rules:** 2026 IRS limits and brackets; SSA 2026 bend points and COLA; 2026 Social Security Trustees Report; SSA 2022
  period life table; IRS Uniform Lifetime Table; SECURE 2.0 RMD ages; FINRA 2111/2360/4512/2165; Reg BI; SEC IM 2017-02.
- **Live inputs:** Yahoo Finance (yfinance), FRED, U.S. Treasury, multpl.com, Damodaran (NYU Stern), CBOE, CNN.

*Educational project, not investment advice.*
