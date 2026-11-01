"""Keel & Co. goals-based robo-advisor (Streamlit, single-scroll).  Run:  streamlit run app.py"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find the robo package however the app is launched

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from robo.advisor import LIMITATIONS, Recommendation, advise
from robo.cases import CASES
from robo.cma import STRESS_PATHS, load_cma, portfolio_stats, stress_returns
from robo.derivatives import Market, futures_hedge, option_cost, payoff_curve, skew_vol, zero_cost_call_strike
from robo.goals import planning_age
from robo.kyc import (GAMBLE_QUESTIONS, QUESTIONS, ClientProfile, Experience, Finances, Goal, Household,
                      Identity, profile_from_dict)
from robo.market_data import (FALLBACK, etf_prices, index_snapshot, macro_snapshot, option_chain_summary,
                              stock_snapshot)
from robo.mortality import survival_table
from robo.portfolio import TIER_NAMES, blend_for_equity, efficient_frontier, load_models, models_summary
from robo.report import build_report
from robo.retirement import CURRENT_YEAR
from robo.risk import model_stress_table
from robo.simulate import SimConfig
from robo.valuation import market_valuation, stock_fair_value

st.set_page_config(page_title="Keel & Co. | Goals-Based Advice", page_icon=":material/anchor:", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject()

CMA = load_cma()
NAME = dict(zip(CMA.keys, CMA.names))
C = ui.ASSET_COLORS
CLIENTS = {"A": "Early-career saver", "B": "Pre-retiree", "C": "Retired couple", "Custom": "New client"}
CLIENT_CAPTIONS = {"A": "Age 30 · house in 2 yrs · retire at 65", "B": "Age 55 · $200K saved · retire at 67",
                   "C": "Age 70 · $2M · married", "Custom": "Blank profile and questionnaire"}
HEADLINES = {"A": "Buy the house, <em>then</em> build the retirement",
             "B": "Twelve years to <em>close the gap</em>",
             "C": "Turning two million into <em>income for life</em>",
             "Custom": "A plan built from <em>your answers</em>"}
STARS = {5: "★★★★★", 4: "★★★★☆", 3: "★★★☆☆", 2: "★★☆☆☆", 1: "★☆☆☆☆"}


def money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else (f"${x / 1e3:,.0f}K" if abs(x) >= 1e4 else f"${x:,.0f}")


def pct(x, d: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}%}"


# ======================================================================== cached data and engine calls
@st.cache_data(ttl=1800, show_spinner=False)
def get_indices(use_live: bool) -> dict:
    if not use_live:
        d = dict(FALLBACK["indices"]); d["asof"] = FALLBACK["asof"]; d["live"] = False
        return d
    return index_snapshot()


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(use_live: bool) -> dict:
    return etf_prices(list(CMA.etf)) if use_live else {}


@st.cache_data(ttl=900, show_spinner=False)
def get_chain(symbol: str, rate: float):
    return option_chain_summary(symbol, rate=rate)


@st.cache_data(ttl=900, show_spinner=False)
def get_stock(ticker: str):
    return stock_snapshot(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro(use_live: bool) -> dict:
    return macro_snapshot(use_live)


@st.cache_data(show_spinner=False, max_entries=48)
def cached_advice(profile_json: str, cfg_json: str, market_json: str, prices_json: str,
                  stock_vol: float | None, stock_info_json: str, light: bool = False) -> Recommendation:
    """light=True (the Compare section) skips the scenario replays and overlay simulation and caps the
    path count, so the three side-by-side cases do not triple the first-load time on a shared server."""
    p = profile_from_dict(json.loads(profile_json))
    cfg = SimConfig(**json.loads(cfg_json))
    if light:
        cfg.n_paths = min(cfg.n_paths, 1500)
    return advise(p, Market(**json.loads(market_json)), cfg, prices=json.loads(prices_json),
                  stock_vol=stock_vol, stock_info=json.loads(stock_info_json) or None,
                  with_scenarios=not light, with_overlay_sim=not light)


@st.cache_data(show_spinner=False)
def cached_methodology():
    return models_summary(), load_models(), efficient_frontier(), model_stress_table()


# ======================================================================== profile (KYC) form
def _default_profile(k: str) -> ClientProfile:
    if k in CASES:
        return CASES[k]()
    return ClientProfile(label="New client", identity=Identity(full_name="New client"),
                         household=Household(age=40, sex="female"),
                         finances=Finances(taxable=50_000, traditional=100_000, roth=20_000, cash=20_000),
                         goals=[], experience=Experience())


def _idx(options, value):
    return options.index(value) if value in options else 0


def _num(v, default):
    try:
        x = float(v)
        return default if np.isnan(x) else x
    except (TypeError, ValueError):
        return default


def kyc_form(k: str) -> ClientProfile:
    b = _default_profile(k)
    i, h, f, x = b.identity, b.household, b.finances, b.experience
    K = lambda s: f"{k}_{s}"
    HEALTH = ["excellent", "good", "average", "fair", "poor"]
    MARITAL = ["single", "married", "divorced", "widowed"]
    blank = k == "Custom"
    steps = st.tabs(["01  Identity & AML", "02  Household", "03  Balance sheet", "04  Retirement & tax", "05  Goals",
                     "06  Experience & approvals", "07  Risk questionnaire"])

    with steps[0], st.container(key="card_kyc_identity"):
        st.caption("Why we ask: identity is verified under the Customer Identification Program (31 CFR 1023.220), screened "
                   "against OFAC, and risk-rated under FinCEN's customer due diligence rule before an account can open. A "
                   "failed check blocks the account; a pending review puts trading on hold. This demo does not collect an SSN.")
        c = st.columns(4)
        full_name = c[0].text_input("Legal name", i.full_name, key=K("name"))
        cit_opts = ["US citizen", "US resident alien", "Non-resident alien"]
        citizenship = c[1].selectbox("Citizenship", cit_opts, index=_idx(cit_opts, i.citizenship), key=K("cit"))
        res_opts = ["US only", "US and another country", "Outside the US"]
        tax_res = c[2].selectbox("Tax residency (FATCA / CRS)", res_opts, index=_idx(res_opts, i.tax_residency), key=K("res"))
        state = c[3].text_input("State of residence", i.state, key=K("state"))
        c = st.columns(4)
        acct_opts = ["individual", "joint", "IRA only", "trust", "LLC / corporation"]
        acct = c[0].selectbox("Account registration", acct_opts, index=_idx(acct_opts, i.account_type), key=K("acct"))
        emp_opts = ["employed", "self-employed", "retired", "not employed"]
        employment = c[1].selectbox("Employment status", emp_opts, index=_idx(emp_opts, i.employment_status), key=K("emp"))
        occupation = c[2].text_input("Occupation", i.occupation, key=K("occ"))
        employer = c[3].text_input("Employer", i.employer, key=K("employer"))
        c = st.columns(4)
        sow_opts = ["salary and savings", "business ownership", "inheritance or gift", "sale of a business or property",
                    "legal settlement", "crypto or trading gains", "other"]
        sow = c[0].selectbox("Source of wealth", sow_opts, index=_idx(sow_opts, i.source_of_wealth), key=K("sow"))
        funding = c[1].number_input("Expected initial funding ($)", 0, 1_000_000_000, int(i.expected_initial_funding), 5_000, key=K("fund"))
        idm_opts = ["documentary", "non-documentary", "pending"]
        id_method = c[2].selectbox("CIP verification", idm_opts, index=_idx(idm_opts, i.id_method), key=K("idm"))
        pdate = c[3].date_input("Profile confirmed on", date.fromisoformat(i.profile_date) if i.profile_date else date.today(),
                                key=K("pdate"), help="SEC Rule 17a-3: the account record must be refreshed at least every 36 months.")
        c = st.columns(3)
        trusted = c[0].text_input("Trusted contact (FINRA 4512)", i.trusted_contact, key=K("tc"))
        tc_declined = c[1].checkbox("Client declined to name a trusted contact", i.trusted_contact_declined, key=K("tcd"))
        poa = c[2].text_input("Power of attorney on file", i.power_of_attorney, key=K("poa"))
        c = st.columns(6)
        idv = c[0].checkbox("ID verified", i.identity_verified, key=K("idv"))
        ofac = c[1].checkbox("OFAC clear", i.ofac_clear, key=K("ofac"))
        bo = c[2].checkbox("Beneficial owners verified", i.beneficial_owners_verified, key=K("bo"),
                           help="Entities only: owners of 25%+ and one control person (FinCEN CDD rule).")
        bd = c[3].checkbox("Works at a broker-dealer", i.employed_by_broker_dealer, key=K("bd"))
        crs = c[4].checkbox("Form CRS acknowledged", i.form_crs_acknowledged, key=K("crs"))
        pep = c[5].checkbox("Politically exposed", i.politically_exposed, key=K("pep"))
        pep_detail, edd = i.pep_detail, i.edd_cleared
        if pep:
            c = st.columns([2, 1])
            pep_detail = c[0].text_input("PEP detail (office, country, self / family / associate)", i.pep_detail, key=K("pepd"))
            edd = c[1].checkbox("Enhanced due diligence approved", i.edd_cleared, key=K("edd"))
        ctrl = st.checkbox("Public-company insider (officer, director or 10% holder)", i.control_person_public_co, key=K("ctrl"))
        ins_tk, plan105, hedge_ok = i.insider_ticker, i.has_10b5_1_plan, i.issuer_permits_hedging
        if ctrl:
            c = st.columns(3)
            ins_tk = c[0].text_input("Issuer ticker", i.insider_ticker, key=K("instk"))
            plan105 = c[1].checkbox("Rule 10b5-1 plan in place", i.has_10b5_1_plan, key=K("105"))
            hedge_ok = c[2].checkbox("Issuer policy permits hedging", i.issuer_permits_hedging, key=K("hedgeok"))

    with steps[1], st.container(key="card_kyc_household"):
        st.caption("Why we ask: age, health and a spouse's details set the planning horizon, life tables, survivor "
                   "benefits and the age required withdrawals begin. Insurance in force changes what the portfolio must cover.")
        c = st.columns(5)
        age = c[0].number_input("Age", 18, 100, h.age, key=K("age"))
        sex = c[1].selectbox("Sex (life tables)", ["female", "male"], index=_idx(["female", "male"], h.sex), key=K("sex"))
        health = c[2].selectbox("Health", HEALTH, index=_idx(HEALTH, h.health), key=K("health"))
        marital = c[3].selectbox("Marital status", MARITAL, index=_idx(MARITAL, h.marital_status), key=K("mar"))
        deps = c[4].number_input("Dependents", 0, 10, h.dependents, key=K("deps"))
        sp_age, sp_sex, sp_health = None, None, "average"
        if marital == "married":
            c = st.columns(5)
            sp_age = c[0].number_input("Spouse age", 18, 100, h.spouse_age or age, key=K("spage"))
            sp_sex = c[1].selectbox("Spouse sex", ["female", "male"], index=_idx(["female", "male"], h.spouse_sex or "male"),
                                    key=K("spsex"))
            sp_health = c[2].selectbox("Spouse health", HEALTH, index=_idx(HEALTH, h.spouse_health), key=K("sphealth"))
        c = st.columns(5)
        life_ins = c[0].checkbox("Life insurance in force", h.life_insured, key=K("life"))
        ltc_ins = c[1].checkbox("Long-term-care insurance", h.long_term_care_insured, key=K("ltci"))

    with steps[2], st.container(key="card_kyc_money"):
        st.caption("Why we ask: income, savings and assets by tax treatment drive the household balance sheet, the "
                   "contribution plan, asset location and how much risk the household can carry.")
        c = st.columns(4)
        income = c[0].number_input("Your gross income ($/yr)", 0, 5_000_000, int(f.annual_income), 5_000, key=K("inc"))
        sp_income = c[1].number_input("Spouse income ($/yr)", 0, 5_000_000, int(f.spouse_income), 5_000, key=K("spinc"))
        growth = c[2].number_input("Real income growth (%/yr)", -5.0, 10.0, f.income_real_growth * 100, 0.5, key=K("g")) / 100
        STAB = ["very stable", "stable", "variable", "volatile"]
        stab = c[3].selectbox("Income stability", STAB, index=_idx(STAB, f.income_stability), key=K("stab"),
                              help="How bond-like future earnings are (the equity beta of human capital).")
        c = st.columns(4)
        sav = c[0].number_input("Savings rate (% of income)", 0.0, 80.0, f.savings_rate * 100, 1.0, key=K("sav")) / 100
        match = c[1].number_input("Employer contribution (%)", 0.0, 25.0, f.employer_match * 100, 0.5, key=K("match")) / 100
        roth_sh = c[2].number_input("Retirement savings to Roth (%)", 0, 100, int(f.contrib_to_roth_share * 100), 10, key=K("rothsh")) / 100
        essential_m = c[3].number_input("Essential expenses ($/month)", 0, 200_000, int(f.monthly_essential_expenses), 250, key=K("ess"))
        st.markdown("**Investable assets by tax treatment**")
        c = st.columns(5)
        taxable = c[0].number_input("Taxable brokerage", 0, 100_000_000, int(f.taxable), 5_000, key=K("tax"))
        trad = c[1].number_input("401(k) / traditional IRA", 0, 100_000_000, int(f.traditional), 5_000, key=K("trad"))
        roth = c[2].number_input("Roth IRA / 401(k)", 0, 100_000_000, int(f.roth), 5_000, key=K("roth"))
        cash = c[3].number_input("Bank cash (emergency)", 0, 100_000_000, int(f.cash), 1_000, key=K("cash"))
        basis = c[4].number_input("Taxable cost basis (%)", 0, 100, int(f.taxable_cost_basis_pct * 100), 5, key=K("basis")) / 100
        st.markdown("**Other assets, debts and liquidity**")
        c = st.columns(5)
        conc = c[0].number_input("Single stock (inside taxable)", 0, 100_000_000, int(f.concentrated_stock), 5_000, key=K("conc"))
        conc_tk = c[1].text_input("Its ticker", f.concentrated_ticker, key=K("conctk"), help="Used for live implied volatility and fair value.")
        home = c[2].number_input("Home value", 0, 100_000_000, int(f.home_value), 10_000, key=K("home"))
        mort = c[3].number_input("Mortgage balance", 0, 100_000_000, int(f.mortgage), 10_000, key=K("mort"))
        liq12 = c[4].number_input("Withdrawals planned in 12 months", 0, 100_000_000, int(f.liquidity_need_12m), 1_000, key=K("liq"))
        c = st.columns(5)
        debt = c[0].number_input("Other debt", 0, 10_000_000, int(f.other_debt), 1_000, key=K("debt"))
        debt_rate = c[1].number_input("Interest rate on it (%)", 0.0, 40.0, f.other_debt_rate * 100, 0.5, key=K("debtr")) / 100
        fee = c[2].number_input("Advisory fee (%/yr)", 0.0, 2.0, f.advisory_fee * 100, 0.05, key=K("fee"),
                                help="Charged on managed assets and deducted from projected returns (Form CRS).") / 100

    with steps[3], st.container(key="card_kyc_retire"):
        st.caption("Why we ask: retirement timing, spending and guaranteed income set the size of the biggest goal. "
                   "Without an SSA statement, benefits are estimated with the 2026 PIA formula. Projected taxes use the 2026 "
                   "federal brackets for your filing status plus a flat state rate.")
        c = st.columns(4)
        ret_age = c[0].number_input("Retirement age", 40, 90, f.retirement_age, key=K("retage"))
        ret_spend = c[1].number_input("Retirement spending ($/yr, today's $)", 0, 5_000_000, int(f.retirement_spending), 1_000, key=K("spend"))
        ess_share = c[2].number_input("Essential share of spending (%)", 20, 100, int(f.essential_share_of_spending * 100), 5, key=K("essh")) / 100
        pension = c[3].number_input("Pension ($/yr from retirement)", 0, 1_000_000, int(f.pension), 1_000, key=K("pen"))
        c = st.columns(4)
        claiming = c[0].checkbox("Already receiving Social Security", f.already_claiming_ss, key=K("claiming"))
        use_stmt = c[1].checkbox("Use benefit from SSA statement", f.social_security is not None, key=K("stmt"))
        ss = c[2].number_input("Your annual benefit ($)", 0, 100_000, int(f.social_security or 0), 500, key=K("ss"), disabled=not use_stmt)
        claim = c[3].number_input("Your claiming age", 62, 70, f.ss_claim_age, key=K("claim"), disabled=claiming)
        sp_ss, sp_claim = None, 67
        if marital == "married":
            c = st.columns(4)
            sp_ss = c[2].number_input("Spouse annual benefit ($)", 0, 100_000, int(f.spouse_social_security or 0), 500,
                                      key=K("spss"), disabled=not use_stmt)
            sp_claim = c[3].number_input("Spouse claiming age", 62, 70, f.spouse_ss_claim_age, key=K("spclaim"), disabled=claiming)
        c = st.columns(4)
        ord_rate = c[0].number_input("Marginal tax rate today, ordinary (%)", 0, 60, int(f.ordinary_tax_rate * 100), key=K("ord")) / 100
        cg_rate = c[1].number_input("Capital gains rate (%)", 0, 40, int(f.cap_gains_rate * 100), key=K("cg")) / 100
        state_rate = c[2].number_input("State income tax (%)", 0.0, 15.0, f.state_tax_rate * 100, 0.5, key=K("st")) / 100

    with steps[4], st.container(key="card_kyc_goals"):
        st.caption("Each goal becomes its own bucket. **Needs** are funded first and with a safety margin; **wishes**, "
                   "such as a legacy, are not counted when measuring how much return the plan requires. Retirement comes "
                   "from step 04. Goals dated beyond the planning horizon are moved to its last year and flagged.")
        g_df = pd.DataFrame([dataclasses.asdict(g) for g in b.goals if g.kind != "retirement"],
                            columns=["name", "kind", "amount", "years_until", "duration_years", "priority"])
        g_df = g_df[["name", "kind", "amount", "years_until", "duration_years", "priority"]]
        edited = st.data_editor(g_df, num_rows="dynamic", width="stretch", key=K("goals"), column_config={
            "name": st.column_config.TextColumn("Goal", required=True),
            "kind": st.column_config.SelectboxColumn("Type", options=["home", "education", "bequest", "other"], required=True),
            "amount": st.column_config.NumberColumn("Amount (today's $)", min_value=0, step=1000, format="$%d"),
            "years_until": st.column_config.NumberColumn("Years until needed", min_value=0, max_value=60),
            "duration_years": st.column_config.NumberColumn("Spread over (yrs)", min_value=1, max_value=30),
            "priority": st.column_config.SelectboxColumn("Priority", options=["need", "want", "wish"], required=True)})

    with steps[5], st.container(key="card_kyc_exp"):
        st.caption("Why we ask: options require written approval under FINRA Rule 2360 (experience by product, objectives, "
                   "financial information) and futures a risk disclosure under NFA Rule 2-30 / CFTC Rule 1.55. A more "
                   "conservative choice than the engine's is honoured automatically; a more aggressive one needs an advisor.")
        c = st.columns(4)
        yrs = c[0].number_input("Years investing", 0, 80, x.years_investing, key=K("yrs"))
        KN = ["none", "limited", "good", "extensive"]
        know = c[1].selectbox("Investment knowledge", KN, index=_idx(KN, x.knowledge), key=K("know"))
        OB = ["preservation", "income", "growth", "speculation"]
        obj = c[2].selectbox("Primary objective", OB, index=_idx(OB, x.objective), key=K("obj"))
        stock_yrs = c[3].number_input("Years in stocks / ETFs", 0, 80, x.stock_years, key=K("stkyrs"))
        c = st.columns(4)
        bond_yrs = c[0].number_input("Years in bonds", 0, 80, x.bond_years, key=K("bondyrs"))
        opt_yrs = c[1].number_input("Years trading options", 0, 60, x.options_years, key=K("optyrs"))
        opt_tr = c[2].number_input("Option trades per year", 0, 1000, x.options_trades_per_year, key=K("opttr"))
        fut_yrs = c[3].number_input("Years trading futures", 0, 60, x.futures_years, key=K("futyrs"))
        c = st.columns(4)
        odd = c[0].checkbox("Received options disclosure", x.options_disclosure_ack, key=K("odd"))
        fdis = c[1].checkbox("Signed futures disclosure", x.futures_disclosure_ack, key=K("fdis"))
        margin = c[2].checkbox("Margin agreement", x.wants_margin, key=K("margin"))
        nod = c[3].checkbox("Exclude derivatives", x.exclude_derivatives, key=K("nod"))
        c = st.columns([1, 2])
        ov = c[0].slider("Your own adjustment to the recommended risk tier", -3, 3, x.override_tiers, key=K("ov"),
                         help="Negative = more conservative (applied). Positive = more aggressive (flagged for an advisor).")
        ov_reason = c[1].text_input("Reason for the adjustment", x.override_reason, key=K("ovr"))

    with steps[6], st.container(key="card_kyc_risk"):
        st.caption("Ten questions in the Grable-Lytton / FinaMetrica tradition, then an income gamble from the Health "
                   "and Retirement Study that estimates risk aversion directly. There are no right answers. Contradictory "
                   "answers are scored conservatively and flagged for review (SEC robo-adviser guidance 2017-02).")
        answers = {}
        for n, q in enumerate(QUESTIONS, 1):
            default = None if blank else b.risk_answers.get(q["id"], len(q["options"]) // 2)
            pick = st.radio(f"{n}. {q['text']}", q["options"], index=default, horizontal=True, key=K(q["id"]))
            if pick is not None:
                answers[q["id"]] = q["options"].index(pick)
        st.markdown("**Income gamble.** Imagine a job that pays your current income for life.")
        gamble = {}
        g1 = st.radio(GAMBLE_QUESTIONS[0]["text"], ["No", "Yes"], index=None if blank else b.gamble_answers.get("g1", 0),
                      horizontal=True, key=K("g1"))
        if g1 is not None:
            gamble["g1"] = ["No", "Yes"].index(g1)
            if g1 == "Yes":
                g2 = st.radio(GAMBLE_QUESTIONS[1]["text"], ["No", "Yes"], index=b.gamble_answers.get("g2_if_yes", 0),
                              horizontal=True, key=K("g2y"))
                gamble["g2_if_yes"] = ["No", "Yes"].index(g2)
            else:
                g2 = st.radio(GAMBLE_QUESTIONS[2]["text"], ["No", "Yes"], index=b.gamble_answers.get("g2_if_no", 0),
                              horizontal=True, key=K("g2n"))
                gamble["g2_if_no"] = ["No", "Yes"].index(g2)

    goals = [Goal("Retirement", "retirement", float(ret_spend), max(int(ret_age) - int(age), 0), 30, "need")]
    for _, r in edited.iterrows():
        name = str(r.get("name") or "").strip()
        if name and name.lower() != "nan":
            goals.append(Goal(name, r.get("kind") or "other", _num(r.get("amount"), 0.0), int(_num(r.get("years_until"), 0)),
                              int(_num(r.get("duration_years"), 1)), r.get("priority") or "want"))
    return ClientProfile(
        label=b.label,
        identity=Identity(full_name=full_name, citizenship=citizenship, tax_residency=tax_res, state=state, account_type=acct,
                          beneficial_owners_verified=bo, employment_status=employment, occupation=occupation, employer=employer,
                          source_of_wealth=sow, expected_initial_funding=float(funding), employed_by_broker_dealer=bd,
                          control_person_public_co=ctrl, insider_ticker=ins_tk.upper(), has_10b5_1_plan=plan105,
                          issuer_permits_hedging=hedge_ok, politically_exposed=pep, pep_detail=pep_detail, edd_cleared=edd,
                          identity_verified=idv, id_method=id_method, ofac_clear=ofac, trusted_contact=trusted,
                          trusted_contact_declined=tc_declined, power_of_attorney=poa, profile_date=pdate.isoformat(),
                          form_crs_acknowledged=crs),
        household=Household(age=int(age), sex=sex, health=health, marital_status=marital,
                            spouse_age=int(sp_age) if sp_age else None, spouse_sex=sp_sex, spouse_health=sp_health,
                            dependents=int(deps), long_term_care_insured=ltc_ins, life_insured=life_ins),
        finances=Finances(
            annual_income=float(income), spouse_income=float(sp_income) if marital == "married" else 0.0,
            income_real_growth=growth, income_stability=stab, savings_rate=sav, employer_match=match,
            contrib_to_roth_share=roth_sh, taxable=float(taxable), traditional=float(trad), roth=float(roth),
            cash=float(cash), taxable_cost_basis_pct=basis, concentrated_stock=float(min(conc, taxable)),
            concentrated_ticker=conc_tk.upper().strip(), home_value=float(home), mortgage=float(mort),
            other_debt=float(debt), other_debt_rate=debt_rate, monthly_essential_expenses=float(essential_m),
            liquidity_need_12m=float(liq12), retirement_age=int(ret_age), retirement_spending=float(ret_spend),
            essential_share_of_spending=ess_share, social_security=float(ss) if use_stmt else None,
            spouse_social_security=float(sp_ss) if (use_stmt and sp_ss is not None) else None,
            ss_claim_age=int(claim), spouse_ss_claim_age=int(sp_claim), already_claiming_ss=claiming,
            pension=float(pension), ordinary_tax_rate=ord_rate, cap_gains_rate=cg_rate, state_tax_rate=state_rate,
            advisory_fee=fee),
        goals=goals,
        experience=Experience(years_investing=int(yrs), knowledge=know, stock_years=int(stock_yrs), bond_years=int(bond_yrs),
                              options_years=int(opt_yrs), options_trades_per_year=int(opt_tr), futures_years=int(fut_yrs),
                              objective=obj, options_disclosure_ack=odd, futures_disclosure_ack=fdis, wants_margin=margin,
                              exclude_derivatives=nod, override_tiers=int(ov), override_reason=ov_reason),
        risk_answers=answers, gamble_answers=gamble,
    )


# ======================================================================== charts
def donut(w: pd.Series, center: str, height: int = 300, legend: bool = True) -> go.Figure:
    w = w[w > 1e-4]
    fig = go.Figure(go.Pie(labels=[NAME[k] for k in w.index], values=w.values, hole=0.66, sort=False, direction="clockwise",
                           marker=dict(colors=[C[k] for k in w.index], line=dict(color=ui.CARD, width=2)),
                           textinfo="none", hovertemplate="%{label}<br>%{percent}<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(family="Instrument Serif, serif", size=24, color=ui.INK))
    fig.update_layout(height=height, showlegend=legend, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)))
    return fig


def fan(sim, profile, overlay=None, height: int = 420, title: str = "") -> go.Figure:
    pc = sim.pct
    fig = go.Figure()
    band = lambda lo, hi, a, name: [
        go.Scatter(x=pc.index, y=pc[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=pc.index, y=pc[lo], fill="tonexty", fillcolor=f"rgba(30,54,88,{a})", line=dict(width=0), name=name,
                   hoverinfo="skip")]
    for tr in band("p5", "p95", 0.10, "5th-95th percentile") + band("p25", "p75", 0.22, "25th-75th percentile"):
        fig.add_trace(tr)
    fig.add_trace(go.Scatter(x=pc.index, y=pc["p50"], line=dict(color=ui.NAVY, width=2.6), name="Median",
                             hovertemplate="Age %{x}<br>Median %{y:$,.0f}<extra></extra>"))
    if overlay is not None:
        fig.add_trace(go.Scatter(x=pc.index, y=overlay.pct["p5"], line=dict(color=ui.AMBER, dash="dot", width=2),
                                 name="5th pct with overlay"))
    if profile.household.age < profile.effective_retirement_age:
        fig.add_vline(x=profile.effective_retirement_age, line=dict(color=ui.AMBER, width=1, dash="dot"))
        fig.add_annotation(x=profile.effective_retirement_age, y=1, yref="paper", text="Retire", showarrow=False, yshift=8,
                           font=dict(size=11, color=ui.AMBER))
    for g in profile.goals:
        if g.kind not in ("retirement", "bequest") and g.years_until > 0:
            a = profile.household.age + g.years_until
            fig.add_vline(x=a, line=dict(color="#6B4E71", width=1, dash="dash"))
            fig.add_annotation(x=a, y=1, yref="paper", text=g.name, showarrow=False, yshift=8, xanchor="left",
                               font=dict(size=11, color="#6B4E71"))
    fig.update_layout(title=title, height=height, yaxis=dict(tickprefix="$", tickformat="~s", rangemode="tozero"),
                      xaxis=dict(title="Age"), hovermode="x unified")
    return fig


def bar_h(labels, values, colors, rng=(0, 1.12), fmt=".0%", vline=None, vlabel="", height=300) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=[format(v, fmt) for v in values], textposition="outside",
                           textfont=dict(family="JetBrains Mono", size=11, color=ui.INK2), cliponaxis=False))
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=ui.INK, dash="dash", width=1.2))
        fig.add_annotation(x=vline, y=1.06, yref="paper", text=vlabel, showarrow=False, font=dict(size=11, color=ui.INK))
    fig.update_layout(height=height, bargap=0.38, xaxis=dict(range=list(rng), tickformat=fmt, showgrid=False),
                      yaxis=dict(autorange="reversed"), margin=dict(l=8, r=30, t=34, b=8))
    return fig


def stop_panel(el) -> None:
    ui.notes(["The account cannot be opened, so no recommendation is implemented. Fix the items below in the profile "
              "(step 01) or escalate to compliance."] + el.kyc_reasons, kind="Account blocked", tone="bad")


# ======================================================================== page chrome: top bar and controls
idx_live_default = True
with st.container(key="controls"):
    c1, c2, c3 = st.columns([3.2, 1, 1.1], vertical_alignment="center")
    choice = c1.radio("Client", list(CLIENTS), format_func=CLIENTS.get, horizontal=True, label_visibility="collapsed")
    use_live = c2.toggle("Live market data", value=True,
                         help="S&P 500, VIX term structure, Treasury yields, option chains, ETF prices, stock fundamentals and "
                              "macro valuation inputs. Long-term return assumptions never come from live prices.")
    with c3.popover("Simulation settings", width="stretch"):
        n_paths = st.select_slider("Market scenarios", [1000, 2000, 3000, 5000, 10000], value=2000)
        spending_rule = st.radio("Retirement spending", ["guardrails", "fixed"],
                                 format_func=lambda s: "Guardrails (adjusts with markets)" if s == "guardrails" else "Fixed in real terms")
        ss_short = st.checkbox("Social Security trust-fund cut (78% from 2033)", value=True)
        fat = st.checkbox("Fat-tailed returns", value=True)
        ltc = st.checkbox("Long-term-care cost shock", value=False,
                          help="55% chance of a 3-year, $120K/yr care episode around age 84 (70% covered if insured).")
        seed = st.number_input("Random seed", value=2026, step=1)
    st.caption(f"{CLIENT_CAPTIONS[choice]} · {n_paths:,} scenarios · {'guardrail' if spending_rule == 'guardrails' else 'fixed'} spending"
               + (" · LTC shock on" if ltc else "") + " · edit anything in the profile below and every section updates.")

cfg = SimConfig(n_paths=int(n_paths), seed=int(seed), fat_tails=fat, spending_rule=spending_rule,
                apply_ss_shortfall=ss_short, ltc_shock=ltc)
idx = get_indices(use_live)
market = Market(spot=idx["^GSPC"], vix=idx.get("^VIX"), rate=idx.get("^IRX", 3.97) / 100, live=idx["live"], asof=idx["asof"])
mkt_dict = dataclasses.asdict(market)
yahoo_ok = use_live and bool(idx["live"])     # if the first Yahoo call failed (throttled IP), skip the rest
prices = get_prices(yahoo_ok)
ui.topbar(market.spot, market.vix, market.rate, market.live)
ui.page_runtime()           # near the top: Streamlit lazy-loads component frames, so it must start in view
hero_slot = st.container()

# ------------------------------------------------------------------------ 01 Profile
ui.section("Client profile", "Seven short steps: the same information a bank collects when it opens an advisory account. "
           "Cases A-C are pre-filled with documented assumptions. Change anything and every section below updates.",
           eyebrow="Know your customer", anchor="profile", num="01")
profile = kyc_form(choice)

# live single-stock data for the concentrated position (implied vol prices the collar; shares/volume for Rule 144)
stock = get_stock(profile.finances.concentrated_ticker) if (yahoo_ok and profile.finances.concentrated_ticker and
                                                           profile.finances.concentrated_stock > 0) else None
stock_vol = stock.get("implied_vol") if stock else None
stock_info = {"shares_outstanding": stock.get("shares_outstanding"), "price": stock.get("price"),
              "avg_volume": stock.get("avg_volume")} if stock else {}
cfg_json = json.dumps({k: v for k, v in dataclasses.asdict(cfg).items() if k != "overlays"})
with st.spinner(f"Profiling risk and running {cfg.n_paths:,} market scenarios..."):
    rec = cached_advice(json.dumps(profile.to_dict()), cfg_json, json.dumps(mkt_dict), json.dumps(prices),
                        stock_vol, json.dumps(stock_info))
rp, p, f, h = rec.risk, rec.profile, rec.profile.finances, rec.profile.household
sim, life, el = rec.sim, rec.sim_life, rec.elig
PA = planning_age(p)
retired = p.is_retired
blocked = el.kyc_status == "blocked"
tone, verdict = ui.verdict_for(sim.success)
advice_main = [a for a in rec.advice if not a.startswith(("**Flag", "**Review"))]
advice_flags = [ui.split_title(a)[1] for a in rec.advice if a.startswith("**Flag")]
advice_review = [ui.split_title(a)[1] for a in rec.advice if a.startswith("**Review")]

with hero_slot:
    hh = "Married" + (f", spouse {h.spouse_age}" if h.spouse_age else "") if h.marital_status == "married" else h.marital_status.capitalize()
    chips = [("Age", str(h.age)), ("Household", hh), ("Invested", money(f.investable)), ("Bank cash", money(f.cash)),
             ("Retirement", "Retired" if retired else f"at {p.effective_retirement_age}"),
             ("Goals", str(len([g for g in p.goals if g.kind != "retirement"]) + 1)), ("Risk tier", f"{rp.tier} of 10"),
             ("Account", el.kyc_status.upper())]
    if blocked:
        sub = "**Account blocked.** Identity, sanctions or beneficial-ownership checks failed; nothing is implemented until compliance clears them."
    else:
        sub = (f"**{verdict}.** The plan works in {sim.success:.0%} of {cfg.n_paths:,} simulated markets ({sim.success_full:.0%} "
               f"with no spending cut). Recommended: the *{rp.tier_name}* portfolio, {rec.glide[0]:.0%} in stocks for long-term money."
               + (" **Trading on hold** pending compliance review." if el.kyc_status == "hold" else ""))
    ui.hero(f"Client plan  ·  {CLIENTS[choice]}  ·  {date.today():%B %d, %Y}", HEADLINES[choice], sub, chips, sim.success,
            f"of simulated markets fund the plan to age {PA}, after {rec.fee_total:.2%} a year in fees and taxes")

# ------------------------------------------------------------------------ 02 Plan overview
ui.section("The plan", "What the engine recommends and how confident it is, before the detail.", eyebrow="Overview",
           anchor="overview", num="02")
if blocked:
    stop_panel(el)
else:
    if el.kyc_status == "hold":
        ui.notes(el.kyc_reasons, kind="On hold", tone="bad")
    if advice_review:
        ui.notes(advice_review, kind="Advisor review required", tone="bad")
    left, right = st.columns([1.3, 1], gap="large")
    with left, st.container(key="card_statement"):
        spend = money(f.retirement_spending)
        if rec.levers is not None and sim.success < 0.85:
            col = "P(success, plan to age 95)"
            best = rec.levers.iloc[1:].sort_values(col, ascending=False).iloc[0]
            sentence = (f"In {sim.success:.0%} of simulated markets the plan pays **{spend} a year** to age {PA}. The gap is a "
                        f"planning problem, not an investing one: *{best['Lever'].lower()}* lifts it to **{best[col]:.0%}**.")
        elif retired:
            sentence = (f"**{spend} a year** of spending lasts to age {PA} in {sim.success:.0%} of simulated markets "
                        f"({sim.success_full:.0%} with no cut), with a median estate of **{money(float(np.median(life.terminal)))}** "
                        "at the second death.")
        else:
            sentence = (f"Saving as planned funds **{spend} a year** from age {p.effective_retirement_age} to {PA} in "
                        f"{sim.success:.0%} of simulated markets, after {rec.fee_total:.2%} a year in fees.")
        third = (("Median estate", money(float(np.median(life.terminal)))) if retired
                 else ("Median at retirement", money(float(np.nanmedian(sim.retire_wealth)))))
        ui.plan_statement(sim.success, sentence, [
            ("No spending cut", pct(sim.success_full)), ("Over simulated lifetimes", pct(life.success)),
            ("Sustainable spending (85%)", f"{money(rec.sustainable)}/yr"), third])
    with right:
        ui.kpis([
            {"label": "Risk profile", "value": rp.tier_name, "sub": f"Tier {rp.tier} of 10", "tone": "info"},
            {"label": "Stocks, long-term money", "value": pct(rec.glide[0]), "sub": f"{pct(rec.stats['equity'])} across all buckets", "tone": "info"},
            {"label": "Expected return", "value": pct(rec.stats["exp_return_geo"], 1),
             "sub": f"{pct(rec.stats['vol'], 1)} volatility · {rec.fee_total:.2%} fees", "tone": "good"},
            {"label": "In a 2008-style year", "value": pct(rec.stress["2008 global financial crisis"]),
             "sub": f"2022-style: {pct(rec.stress['2022 stock-bond inflation shock'])}", "tone": "bad"},
        ])
        with st.container(key="card_alloc_mini"):
            ui.eyebrow("Allocation today, all buckets")
            ui.alloc_split(rec.weights_now.to_dict())
            st.caption(f"{len(rec.implementation)} ETFs · weighted expense ratio {rec.stats['expense_ratio']:.2%}. "
                       "Full breakdown in the Portfolio section.")
    ui.subsection("What to do next", "The most important actions, in order. The Portfolio section explains each one in full.")
    ui.actions(advice_main[:6])
    ui.subsection("Your goals", "Each goal is its own bucket with its own investments and its own chance of being met.")
    gcards = [{"name": "Retirement", "priority": "need", "prob": sim.success,
               "meta": f"{money(f.retirement_spending)}/yr " + ("now" if retired else f"from age {p.effective_retirement_age}") + f" to {PA}"}]
    for _, bk in rec.buckets.iterrows():
        if bk["Bucket"] in sim.goal_success:
            tgt_ = money(bk["Target (today's $)"])
            gcards.append({"name": bk["Bucket"], "priority": bk["Priority"], "prob": sim.goal_success[bk["Bucket"]],
                           "meta": f"{tgt_} in {bk['Horizon (yrs)']} yrs · {bk['Equity now']:.0%} stocks today"})
    for g in p.goals:
        if g.kind == "bequest" and g.amount > 0:
            gcards.append({"name": g.name, "priority": g.priority, "prob": float(np.mean(life.terminal >= g.amount)),
                           "meta": f"{money(g.amount)} estate target · simulated lifetimes"})
    ui.goal_cards(gcards)
    ui.subsection("The range of outcomes", "Financial wealth in today's dollars. The dark band holds the middle half of "
                  "outcomes; the light band holds 90% of them.")
    with st.container(key="card_fan_overview"):
        ui.chart(fan(sim, p, height=360))
    if advice_flags:
        ui.notes(advice_flags)

# ------------------------------------------------------------------------ 03 Risk
ui.section("How much risk is right", "We measure three different things. The recommendation is the lower of what "
           "you are willing and able to take; what your goals need is never used to push risk up.", eyebrow="Risk profile",
           anchor="risk", num="03")
if blocked:
    stop_panel(el)
else:
    if rp.review_flags:
        ui.notes(rp.review_flags, kind="Inconsistent answers · advisor review", tone="bad")
    need_txt = pct(rp.need_real_return, 1) if rp.need_real_return is not None else "unreachable"
    ui.pillars([
        {"key": "Willingness", "title": f"{rp.tolerance_score:.0f} / 100 tolerance",
         "rows": [("Risk aversion (gamma)", f"{rp.gamma:.2f}"), ("From questionnaire", f"{rp.gamma_questionnaire:.2f}"),
                  ("From income gamble", f"{rp.gamma_gamble:.2f}" if rp.gamma_gamble else "n/a"),
                  ("Largest loss you accept", pct(rp.max_loss))],
         "note": f"{rp.n_answered} of 10 items answered. Gamble: {rp.gamble_note}."},
        {"key": "Ability", "title": f"{rp.capacity_score:.0f} / 100 capacity",
         "rows": [("Balance-sheet equity", pct(rp.hc_equity)), ("Stress-loss cap", pct(rp.drawdown_equity)),
                  ("Capacity cap", pct(rp.capacity_equity)), ("Funded ratio", f"{rp.funded_ratio:.2f}x")],
         "note": "Future savings and Social Security act like bonds you already own, so they raise how much stock the "
                 "portfolio can hold."},
        {"key": "Need", "title": f"{need_txt} real return",
         "rows": [("Stocks needed for goals", pct(rp.need_equity) if rp.need_equity is not None else "above 98%"),
                  ("Recommended stocks", pct(rp.recommended_equity)), ("Binding limit", rp.binding.split(" (")[0]),
                  ("Worst stress loss, this mix", pct(rp.model_worst_loss, 1))],
         "note": "The return that makes assets plus future savings and income exactly pay for the goals."
                 + (f" Client override applied: {rp.override_applied:+d} tier(s) (engine said {rp.engine_equity:.0%})."
                    if rp.override_applied else "")},
    ])
    c1, c2 = st.columns([1.45, 1], gap="large")
    with c1, st.container(key="card_caps"):
        labels = ["Merton share, financial wealth only", "Merton + human capital & Social Security",
                  f"Stress-loss cap (max loss {rp.max_loss:.0%})", f"Capacity cap (score {rp.capacity_score:.0f})",
                  "Need (to fund goals)"]
        vals = [min(rp.merton_equity, 1.0), rp.hc_equity, rp.drawdown_equity, rp.capacity_equity,
                rp.need_equity if rp.need_equity is not None else 1.0]
        binding_i = {"balance": 1, "stated": 2, "risk capacity": 3}
        bi = next((v for k_, v in binding_i.items() if rp.binding.startswith(k_)), 2)
        cols = ["#DAD6CC"] * 5
        cols[bi] = ui.GREEN
        cols[4] = "#B9A6BF"
        fig = bar_h(labels, vals, cols, vline=rp.recommended_equity, vlabel=f"Recommended {rp.recommended_equity:.0%}", height=320)
        fig.update_layout(title="Equity limits from each dimension")
        ui.chart(fig)
    with c2, st.container(key="card_capacity"):
        cp = pd.Series(rp.capacity_parts)
        fig = bar_h(list(cp.index), list(cp.values / 100), [ui.SAGE] * len(cp), height=320)
        fig.update_layout(title="Capacity components (0-100)")
        ui.chart(fig)
    ui.subsection("Household balance sheet", "Today's dollars. Human capital is the present value of future portfolio "
                  "contributions; guaranteed income is Social Security and pensions.")
    with st.container(key="card_bs"):
        ui.table(["Item", "Value", "How it is used"], [
            ["Financial assets", money(f.investable), "The portfolio the robo manages"],
            ["Human capital", money(rp.human_capital), "Bond-like with equity beta " + f"{rp.beta_hc:.2f}"],
            ["Social Security + pension (PV)", money(rp.guaranteed_pv), "Bond-like, inflation-linked"],
            ["Funded ratio", f"{rp.funded_ratio:.2f}x", "Assets + future savings + income vs goals, at ~2% real"],
            ["Bank cash", money(f.cash), f"{f.cash / max(f.monthly_essential_expenses, 1):.1f} months of essentials"],
            ["Non-mortgage debt", money(f.other_debt), f"at {f.other_debt_rate:.1%}: pay-down is a risk-free return"],
        ], num_cols={1})
    ui.notes(rp.flags + rec.goal_warnings)

# ------------------------------------------------------------------------ 04 Portfolio
ui.section(f"The {rp.tier_name} portfolio", f"{rec.glide[0]:.0%} stocks for long-term money, built from 15 low-cost "
           f"ETFs (weighted expense ratio {rec.stats['expense_ratio']:.2%}, all-in {rec.fee_total:.2%} with the advisory fee). "
           "Near-term goals hold their own, safer mix.", eyebrow="Recommendation", anchor="portfolio", num="04")
if blocked:
    stop_panel(el)
else:
    ui.actions(advice_main)
    ui.notes(advice_flags)
    ui.subsection("Goal buckets", "Money is earmarked by goal, then invested to match that goal's date and priority.")
    with st.container(key="card_buckets"):
        rows = []
        for _, bk in rec.buckets.iterrows():
            tgt = bk["Target (today's $)"]
            rows.append([f"**{bk['Bucket']}**", bk["Priority"], f"{bk['Horizon (yrs)']}",
                         money(tgt) if pd.notna(tgt) else "-", money(bk["Funded today"]),
                         money(bk["Annual top-up"]) if bk["Annual top-up"] > 0 else "-", pct(bk["Equity now"]), bk["Model"]])
        ui.table(["Bucket", "Priority", "Years", "Target", "Funded today", "Top-up / yr", "Stocks", "Model"], rows,
                 num_cols={2, 3, 4, 5, 6})
    c1, c2 = st.columns([1, 1.55], gap="large")
    with c1, st.container(key="card_donut"):
        ui.eyebrow("Household allocation")
        ui.chart(donut(rec.weights_now, f"{rec.stats['equity']:.0%}<br><span style='font-size:11px'>stocks</span>",
                       height=430, legend=False))
        st.caption(f"2008-style year {rec.stress['2008 global financial crisis']:.1%} · 2022-style "
                   f"{rec.stress['2022 stock-bond inflation shock']:.1%}")
    with c2, st.container(key="card_etfs"):
        ui.eyebrow("ETF implementation")
        if len(rec.implementation):
            impl = rec.implementation.drop(columns=["Key"]).copy()
            impl["Amount"] = impl["Amount"].map(lambda v: f"${v:,.0f}")
            impl["Shares"] = impl["Shares"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
            st.dataframe(impl, width="stretch", hide_index=True, height=430,
                         column_config={"Weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=0.6, format="percent"),
                                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                                        "Expense ratio": st.column_config.NumberColumn(format="percent")})
        else:
            st.caption("Nothing to invest yet: the target mix is shown in the donut; the trade list appears once assets are funded.")
        st.caption("The *Alternative* column is the tax-loss-harvesting pair: sell the primary ETF at a loss, hold the alternative "
                   "for 31 days to avoid a wash sale, then switch back. Rebalance when a sleeve drifts 5 points or 25% from target.")
    ui.subsection("Glide path", "How the retirement portfolio's mix changes with age. It comes from your own balance sheet, "
                  "is held steady around retirement (the sequence-of-returns window) and never rises again after that.")
    with st.container(key="card_glide"):
        ages = rec.schedule["age"].to_numpy()
        W = np.vstack([blend_for_equity(e) for e in rec.glide])
        fig = go.Figure()
        for j, k_ in enumerate(CMA.keys):
            if W[:, j].max() > 0.002:
                fig.add_trace(go.Scatter(x=ages, y=W[:, j], stackgroup="one", name=NAME[k_], mode="lines",
                                         line=dict(width=0.4, color=C[k_]), fillcolor=C[k_],
                                         hovertemplate=f"{NAME[k_]}: " + "%{y:.0%}<extra></extra>"))
        if not retired:
            fig.add_vline(x=p.effective_retirement_age, line=dict(color=ui.INK, dash="dot", width=1))
        fig.update_layout(height=400, yaxis=dict(tickformat=".0%", range=[0, 1]), xaxis_title="Age",
                          legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        ui.chart(fig)
    ui.subsection("Asset location", "Which account holds what. Tax-free Roth growth goes to the highest-return assets, "
                  "tax-deferred accounts hold the least tax-efficient ones, and broad stock ETFs sit in the taxable account.")
    with st.container(key="card_location"):
        loc = rec.location
        fig = go.Figure()
        for acct, col in zip(loc.columns, ["#6B4E71", ui.GREEN, ui.NAVY]):
            fig.add_trace(go.Bar(y=loc.index, x=loc[acct], name=acct, orientation="h", marker_color=col,
                                 hovertemplate=f"{acct}: " + "%{x:$,.0f}<extra></extra>"))
        fig.update_layout(barmode="stack", height=440, xaxis=dict(tickprefix="$", tickformat="~s"),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08, x=0))
        ui.chart(fig)
        st.caption("Existing appreciated holdings should be moved gradually, not sold just to relocate them.")

# ------------------------------------------------------------------------ 05 Projections
ui.section("Projections", f"{cfg.n_paths:,} simulated markets with fat-tailed returns, {rec.fee_total:.2%} annual fees, 2026 federal "
           f"tax brackets, required withdrawals, Social Security{' (with the 2033 trust-fund cut)' if ss_short else ''}"
           f"{', a long-term-care shock' if ltc else ''} and life expectancy. All values in today's dollars.", eyebrow="Monte Carlo",
           anchor="projections", num="05")
if blocked:
    stop_panel(el)
else:
    ui.kpis([
        {"label": f"Success to age {PA}", "value": pct(sim.success), "sub": f"{pct(sim.success_full)} with no cut", "tone": tone},
        {"label": "Over simulated lifetimes", "value": pct(life.success), "tone": ui.verdict_for(life.success)[0]},
        {"label": "Median at retirement" if not retired else "Median in 10 years",
         "value": money(np.nanmedian(sim.retire_wealth)) if not retired else money(float(sim.pct["p50"].iloc[min(10, len(sim.pct) - 1)])),
         "sub": None if retired else f"10th pct {money(np.nanpercentile(sim.retire_wealth, 10))}", "tone": "info"},
        {"label": "Sustainable spending", "value": money(rec.sustainable),
         "sub": f"{rec.sustainable / max(f.retirement_spending, 1) - 1:+.0%} vs planned · 85%, no cuts", "tone": "info"},
        {"label": "Median estate", "value": money(float(np.median(life.terminal))),
         "sub": f"after heirs' tax {money(float(np.median(life.terminal_after_tax)))}", "tone": "info"},
        {"label": "Lifetime income tax", "value": money(float(np.median(sim.taxes_paid))), "sub": "median, in retirement", "tone": "info"},
    ])
    if sim.goal_success:
        st.caption("Goal funding probability: " + " · ".join(f"{g}: {v:.0%}" for g, v in sim.goal_success.items()))
    with st.container(key="card_fan_full"):
        ui.chart(fan(sim, p, rec.sim_overlay, height=470, title=f"Financial wealth to age {PA}"))
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_spend"):
        ret_idx = np.where(sim.ages >= p.effective_retirement_age)[0]
        if len(ret_idx):
            sp = pd.DataFrame({q: np.percentile(sim.spending[:, ret_idx], q, axis=0) for q in (10, 50, 90)}, index=sim.ages[ret_idx])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sp.index, y=sp[90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sp.index, y=sp[10], fill="tonexty", fillcolor="rgba(14,107,82,.14)",
                                     line=dict(width=0), name="10th-90th percentile"))
            fig.add_trace(go.Scatter(x=sp.index, y=sp[50], line=dict(color=ui.GREEN, width=2.4), name="Median"))
            fig.add_hline(y=f.retirement_spending, line=dict(color=ui.AMBER, dash="dash", width=1.2))
            fig.update_layout(title="Spending actually achieved", height=330, yaxis=dict(tickprefix="$", tickformat="~s"), xaxis_title="Age")
            ui.chart(fig)
            st.caption(f"Amber line: planned spending. Chance of a cut of more than 10% at some point: {sim.cut_prob:.0%}."
                       + (f" Long-term-care episode in {sim.ltc_hit:.0%} of paths." if ltc else ""))
    with c2, st.container(key="card_estate"):
        cap99 = max(float(np.percentile(life.terminal, 98)) / 1e6, 0.01)
        est = np.clip(life.terminal / 1e6, None, cap99)
        fig = go.Figure(go.Histogram(x=est, nbinsx=40, marker=dict(color="#6B4E71", line=dict(width=0)),
                                     opacity=0.85, hovertemplate="$%{x:.2f}M<br>%{y} scenarios<extra></extra>"))
        fig.add_vline(x=float(np.median(life.terminal)) / 1e6, line=dict(color=ui.INK, dash="dash", width=1))
        fig.update_layout(title="Estate at death (simulated lifetimes)", height=330,
                          xaxis=dict(title="$ millions", tickprefix="$"), yaxis=dict(title="Scenarios"), bargap=0.05)
        ui.chart(fig)
        st.caption(f"Dashed line: median. The top 2% of outcomes are grouped at ${cap99:,.1f}M.")

    if rec.scenarios is not None:
        ui.subsection("If history repeats", "Each episode's actual asset-class returns and inflation are replayed as the first "
                      "years of the plan; the random draws resume afterwards. This is the stress test behind the income reserve, "
                      "the guardrails and the overlays.")
        sc = rec.scenarios
        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1, st.container(key="card_scen_chart"):
            vals = list(sc["P(success)"])
            fig = bar_h(list(sc["Scenario"]), vals, [ui.TONE[ui.verdict_for(v)[0]] for v in vals], height=300)
            fig.update_layout(title="Probability of success by opening scenario")
            ui.chart(fig)
        with c2, st.container(key="card_scen_tbl"):
            ui.table(["Opening scenario", "Yrs", "Wealth after 3 yrs", "Success", "No cut", "Cut >10%", "Median estate"],
                     [[r_["Scenario"], r_["Years"], money(r_["Wealth after 3 yrs (median)"]), pct(r_["P(success)"]),
                       pct(r_["P(no cut)"]), pct(r_["P(cut >10%)"]), money(r_["Median estate"])] for _, r_ in sc.iterrows()],
                     num_cols={1, 2, 3, 4, 5, 6})
            st.caption("Returns are nominal index returns for each calendar year (approximate for pre-1990 asset classes) with "
                       "that year's CPI inflation, so 1973-74 and 2022 also test the plan against an inflation shock.")

    if rec.levers is not None:
        ui.subsection("Plan levers", "Each change is re-profiled and re-simulated. Raising risk is not on the list: the "
                      "client's limits bind. These are the conversations an advisor would have.")
        lv = rec.levers
        c1, c2 = st.columns([1.1, 1.3], gap="large")
        with c1, st.container(key="card_levers"):
            vals = list(lv["P(success, plan to age 95)"])
            cols = [ui.verdict_for(v)[0] for v in vals]
            fig = bar_h(list(lv["Lever"]), vals, [ui.TONE[c_] for c_ in cols], height=max(300, 42 * len(lv)))
            fig.update_layout(title="Probability of success")
            ui.chart(fig)
        with c2, st.container(key="card_levers_tbl"):
            ui.table(["Lever", "Success", "No cut", "At retirement", "Estate after tax", "Lifetime tax"],
                     [[r_["Lever"], pct(r_["P(success, plan to age 95)"]), pct(r_["P(no cut)"]),
                       money(r_["Median wealth at retirement"]), money(r_["Median estate after tax"]),
                       money(r_["Median lifetime tax"])] for _, r_ in lv.iterrows()], num_cols={1, 2, 3, 4, 5})
            st.caption("Roth conversions move money from the traditional account to the Roth each year before RMDs, up to the "
                       "top of the 24% bracket, paying the tax from the taxable account. The estate is shown net of the income "
                       "tax heirs owe on inherited pre-tax balances.")

    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_ss_claim"):
        if rec.ss_claiming:
            rows_ = rec.ss_claiming
            best = max(rows_, key=lambda r_: r_["epv"])
            fig = go.Figure(go.Bar(x=[r_["claim_age"] for r_ in rows_], y=[r_["epv"] for r_ in rows_],
                                   marker=dict(color=[ui.GREEN if r_ is best else "#DAD6CC" for r_ in rows_], line=dict(width=0)),
                                   hovertemplate="Claim at %{x}<br>EPV %{y:$,.0f}<extra></extra>"))
            fig.update_layout(title="Social Security: expected lifetime benefits by claiming age", height=320,
                              yaxis=dict(tickprefix="$", tickformat="~s"), xaxis=dict(title="Claiming age", dtick=1))
            ui.chart(fig)
            st.caption(f"Present value on the client's own survival curve at a 1.9% real discount rate. Best: {best['claim_age']} "
                       f"({money(best['annual'])}/yr). Breakeven vs claiming at 62: " +
                       ", ".join(f"{r_['claim_age']} → age {r_['breakeven_age']}" for r_ in rows_[1:] if r_.get("breakeven_age")) + ".")
        else:
            ui.eyebrow("Social Security claiming")
            st.caption("Already claiming, or no benefit on record: the claiming-age analysis does not apply.")
    with c2, st.container(key="card_seq"):
        mu_r = rec.stats["exp_return_geo"] - CMA.inflation
        sig = rec.stats["vol"]
        draws = np.random.default_rng(11).normal(mu_r, sig, 25)
        w0 = float(np.nanmedian(sim.retire_wealth)) if not retired else f.investable
        s0 = rec.schedule["retirement_spending"].to_numpy()
        r0 = np.where(rec.schedule["age"].to_numpy() >= p.effective_retirement_age)[0]
        need0 = float(s0[r0[0]] - rec.schedule["ss_client_full"].iloc[r0[0]] - rec.schedule["ss_spouse_full"].iloc[r0[0]]
                      - rec.schedule["pension"].iloc[r0[0]]) if len(r0) else 0.04 * w0
        need0 = max(need0, 0.045 * w0)
        bad_first = np.sort(draws)
        def path(rs):
            w, out = w0, [w0]
            for r_ in rs:
                w = max((w - need0) * (1 + r_), 0.0); out.append(w)
            return out
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=path(bad_first), name="Worst years first", line=dict(color=ui.RED, width=2.4)))
        fig.add_trace(go.Scatter(y=path(bad_first[::-1]), name="Same years, best first", line=dict(color=ui.GREEN, width=2.4)))
        fig.update_layout(title="Sequence-of-returns risk: identical returns, opposite order", height=320,
                          xaxis_title="Years into retirement", yaxis=dict(tickprefix="$", tickformat="~s", rangemode="tozero"))
        ui.chart(fig)
        st.caption(f"Starting wealth {money(w0)}, withdrawals {money(need0)}/yr, the same 25 annual returns (mean {mu_r:.1%} real, "
                   f"volatility {sig:.0%}) in opposite order. The average return is identical; the order decides whether the money "
                   "lasts. That is why the five-year income reserve, the collar window and the guardrails exist.")
    with st.expander("Cash-flow schedule (expected path, today's dollars)"):
        st.dataframe(rec.schedule.drop(columns=["ss_client_full", "ss_spouse_full"]).round(0), width="stretch", hide_index=True)

# ------------------------------------------------------------------------ 06 Options & Futures
plan = rec.overlays
ui.section("Options & futures", "Derivatives are used only to reduce risk or earn income on positions you already "
           "own, and only at the approval level your profile supports. Prices use the live S&P 500, VIX, T-bill rate and, "
           "where available, the listed option chain.", eyebrow="Overlays", anchor="options", num="06")
if blocked:
    stop_panel(el)
else:
    ui.kpis([
        {"label": "Options approval", "value": f"Level {el.options_level}", "sub": "FINRA Rule 2360", "tone": "good" if el.options_level else "bad"},
        {"label": "Futures account", "value": "Approved" if el.futures_ok else "Not approved", "sub": "Hedging only (NFA 2-30)",
         "tone": "good" if el.futures_ok else "bad"},
        {"label": "S&P 500 / VIX", "value": f"{market.spot:,.0f}", "sub": f"VIX {market.vix:.1f} · T-bill {market.rate:.2%}", "tone": "info"},
        {"label": "1-yr implied vol (model)", "value": pct(market.sigma, 1), "sub": "VIX blended with an 18% long-run level", "tone": "info"},
    ])
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1, st.container(key="card_elig"):
        ui.eyebrow("Why this approval level")
        for r_ in el.options_reasons + el.futures_reasons:
            st.markdown(f"- {r_}")
    with c2, st.container(key="card_levels"):
        ui.eyebrow("House approval ladder")
        ui.table(["Level", "Strategies", "Requirements"], [
            ["0", "None", "-"],
            ["1", "Collars, protective puts, covered calls on shares owned", "Disclosure received, knowledge limited+, any objective, $50K liquid"],
            ["2", "+ long options, cash-secured puts", "Knowledge good+, 1+ yr options, objective income+, $100K liquid"],
            ["3", "+ spreads (margin; never in IRAs)", "2+ yrs, objective growth+, $250K, margin agreement"],
            ["4", "+ uncovered writing", "Never recommended by this robo"],
        ])
    ui.subsection("Recommended overlays")
    if not plan.overlays and not plan.futures and not plan.single_stock_collar:
        ui.notes([plan.not_recommended[0]], kind="No overlay", tone="info")
    ui.notes(plan.rationale, kind="Rationale", tone="info")
    for n_ov, (ov, t0, t1) in enumerate(plan.overlays):
        grid, y = payoff_curve(ov, market.rate)
        end_age = min(h.age + t1, PA)
        with st.container(key=f"card_overlay_{n_ov}"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=grid + 0.012, name="S&P 500 (total return)", line=dict(color="#B9B5AC", dash="dash")))
            fig.add_trace(go.Scatter(x=grid, y=y, name="With overlay", line=dict(color=ui.NAVY, width=3),
                                     hovertemplate="Index %{x:.0%} → overlay %{y:.1%}<extra></extra>"))
            fig.update_layout(title=f"{ov.describe()}", height=330,
                              xaxis=dict(tickformat=".0%", title=f"S&P 500 price return over 1 year · active ages {h.age + t0}-{end_age}"),
                              yaxis=dict(tickformat=".0%", title="Overlay return"))
            ui.chart(fig)
    if rec.sim_overlay is not None:
        base, ovs = rec.sim, rec.sim_overlay
        h_age = min(p.effective_retirement_age + 5 if not retired else h.age + 10, int(base.ages[-1]))
        i = list(base.ages).index(h_age)
        with st.container(key="card_ov_impact"):
            ui.eyebrow(f"Monte Carlo impact · wealth at age {h_age}, same random paths")
            qs = (5, 25, 50, 75, 95)
            ui.table(["", "5th pct", "25th", "Median", "75th", "95th", "Success"], [
                ["Core portfolio"] + [money(np.percentile(base.wealth[:, i], q)) for q in qs] + [pct(base.success)],
                ["With overlay"] + [money(np.percentile(ovs.wealth[:, i], q)) for q in qs] + [pct(ovs.success)],
            ], num_cols={1, 2, 3, 4, 5, 6})
            st.caption("Overlays give up some expected return in exchange for a narrower range of outcomes. That trade "
                       "suits the years around retirement and income-focused retirees.")

    if plan.concentrated_plan or stock:
        ui.subsection("Concentrated stock", f"{money(f.concentrated_stock)} in {f.concentrated_ticker or 'a single stock'} inside the "
                      "taxable account: hedge it, sell it down on a schedule, or both.")
        kp = []
        if stock:
            mac_ = get_macro(use_live)
            rf_ = float(mac_["fred"].get("DGS10", 4.9)) / 100
            erp_ = float(mac_["erp"].get("erp_sustainable") or 4.2) / 100
            fv = stock_fair_value(stock, rf=rf_, erp=erp_, market_pe=mac_.get("sp_pe"))
            kp += [{"label": f"{stock['ticker']} price", "value": f"${stock['price']:,.2f}",
                    "sub": f"{stock.get('name', '')} · 52w {stock.get('week52_low', 0):,.0f}-{stock.get('week52_high', 0):,.0f}", "tone": "info"},
                   {"label": "Implied vol (6-mo ATM)", "value": pct(stock.get("implied_vol"), 1),
                    "sub": f"realised 1-yr {pct(stock.get('realized_vol'), 1)} · beta {stock.get('beta') or 'n/a'}", "tone": "info"},
                   {"label": "Analyst target", "value": f"${stock['target_median']:,.0f}" if stock.get("target_median") else "n/a",
                    "sub": f"{stock.get('n_analysts', 0)} analysts · {stock.get('recommendation', '')}", "tone": "info"}]
            if fv.get("fair_value"):
                kp.append({"label": "Model fair value", "value": f"${fv['fair_value']:,.0f} {STARS[fv['stars']]}",
                           "sub": f"price/FV {fv['ratio']:.2f} · {fv['uncertainty'].lower()} uncertainty · band ${fv['low']:,.0f}-{fv['high']:,.0f}",
                           "tone": "good" if fv["stars"] >= 4 else "bad" if fv["stars"] <= 2 else "mid"})
        if plan.single_stock_collar:
            s_ = plan.single_stock_collar
            kp.append({"label": "Zero-cost collar", "value": f"{s_['put_strike']:.0%} / {s_['call_strike']:.0%}",
                       "sub": f"max loss {money(s_['max_loss'])}, max gain {money(s_['max_gain'])} · vol {s_['vol']:.0%}"
                              + (" live" if s_.get("live_vol") else " assumed"), "tone": "info"})
        if kp:
            ui.kpis(kp)
        cpn = plan.concentrated_plan
        if cpn:
            c1, c2 = st.columns([1.2, 1], gap="large")
            with c1, st.container(key="card_tranches"):
                ui.eyebrow("Staged sale · 8 quarterly tranches" + (" under a Rule 10b5-1 plan" if p.identity.control_person_public_co else ""))
                yr0 = CURRENT_YEAR
                ui.table(["Quarter", "Sell", "Gain realised", "Tax (est.)", "Remaining"],
                         [[f"Q{(k_ % 4) + 1} {yr0 + k_ // 4}", money(cpn["per_tranche"]), money(cpn["embedded_gain"] / cpn["tranches"]),
                           money(cpn["tax_per_tranche"]), money(cpn["value"] - cpn["per_tranche"] * (k_ + 1))] for k_ in range(cpn["tranches"])],
                         num_cols={1, 2, 3, 4})
            with c2, st.container(key="card_conc_notes"):
                ui.eyebrow("Rules that apply")
                notes_ = [f"Embedded gain {money(cpn['embedded_gain'])} ({1 - f.taxable_cost_basis_pct:.0%} of value); spreading sales over "
                          f"2-3 tax years keeps each year inside the {f.cap_gains_rate:.0%} capital-gains bracket."]
                if p.identity.control_person_public_co:
                    notes_.append(("Rule 144: sales limited to the greater of 1% of shares outstanding or the 4-week average volume per "
                                   "3 months" + (f" (about {money(cpn['rule144_quarterly_cap'])} per quarter at today's volume)." if cpn.get("rule144_quarterly_cap") else ".")))
                    notes_.append("Rule 10b5-1 plan " + ("in place: sales continue through blackout periods." if not cpn["needs_10b5_1"]
                                                         else "needed: adopt it in an open window; a 90-day cooling-off applies (2023 amendments)."))
                    notes_.append("Hedging " + ("permitted by the issuer." if p.identity.issuer_permits_hedging else
                                                "prohibited by the issuer's anti-hedging policy, so no collar or put."))
                else:
                    notes_.append("Collar band kept wide enough (90/110) to avoid a constructive sale under IRC 1259; straddle rules (IRC 1092) "
                                  "suspend the holding period while the collar is on.")
                if cpn["exchange_fund_eligible"]:
                    notes_.append("Accredited: an exchange fund (7-year lock-up, diversified basket in kind) or a charitable remainder trust "
                                  "defers the gain entirely; both are advisor referrals.")
                notes_.append("Donating appreciated shares to a donor-advised fund removes the gain and gives a deduction at market value.")
                for n_ in notes_:
                    st.markdown(f"- {n_}")

    chain = get_chain("SPY", market.rate) if yahoo_ok else None
    ui.subsection("Listed market vs the model", "Implied volatilities recomputed from bid/ask mid-prices of SPY options at about "
                  "3, 6 and 12 months, compared with the model's skew σ(K) = σ_ATM − 0.30·ln(K/S).")
    if chain:
        exps = chain["expiries"]
        S = chain["spot"]
        ui.kpis([{"label": f"ATM IV {e['expiry']}", "value": pct(e["atm_iv"], 1),
                  "sub": f"25Δ skew {e['skew25'] * 100:.1f} pts · slope {e['slope']:.2f} vs model −0.30 · {e['n_calls'] + e['n_puts']} quotes",
                  "tone": "info"} for e in exps])
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1, st.container(key="card_smile"):
            fig = go.Figure()
            for e, col in zip(exps, [ui.NAVY, ui.AMBER, ui.GREEN]):
                otm = pd.concat([e["puts"][e["puts"]["strike"] < S], e["calls"][e["calls"]["strike"] > S]])
                otm = otm[(otm["moneyness"] > 0.7) & (otm["moneyness"] < 1.25)]
                fig.add_trace(go.Scatter(x=otm["moneyness"], y=otm["iv_mid"], mode="markers", name=f"Market {e['expiry']}",
                                         marker=dict(size=5, color=col, opacity=0.75),
                                         hovertemplate="K/S %{x:.2f}<br>IV %{y:.1%}<extra></extra>"))
            grid = np.linspace(0.7, 1.25, 60)
            fig.add_trace(go.Scatter(x=grid, y=[skew_vol(m_, market.sigma) for m_ in grid], name="Model (1-yr)",
                                     line=dict(color=ui.INK, dash="dash", width=2)))
            fig.update_layout(title=f"Implied-volatility smile · SPY {S:,.0f}", height=380,
                              xaxis=dict(title="Strike / spot", tickformat=".2f"), yaxis=dict(title="Implied vol", tickformat=".0%"))
            ui.chart(fig)
        with c2, st.container(key="card_live_collar"):
            ui.eyebrow("Collar priced from the chain")
            e12 = exps[-1]
            puts_, calls_ = e12["puts"], e12["calls"]
            if len(puts_) and len(calls_):
                prow = puts_.iloc[(puts_["moneyness"] - 0.90).abs().argsort().iloc[0]]
                above = calls_[calls_["strike"] > S].sort_values("strike")
                k_call = float(np.interp(prow["mid"], above["mid"].to_numpy()[::-1], above["strike"].to_numpy()[::-1])) if len(above) > 2 else float("nan")
                model_k = zero_cost_call_strike(market, 0.90, T=e12["T"])
                ui.table(["", "Market chain", "Model"], [
                    ["Expiry", e12["expiry"], f"{e12['T']:.2f} yr"],
                    ["Put strike", f"{prow['strike']:,.0f} ({prow['moneyness']:.0%})", "90%"],
                    ["Put premium", f"${prow['mid']:.2f} ({prow['mid'] / S:.2%})", f"{option_cost(market, 0.90, 'put', e12['T']):.2%}"],
                    ["Zero-cost call strike", f"{k_call:,.0f} ({k_call / S:.1%})" if np.isfinite(k_call) else "n/a", f"{model_k:.1%}"],
                    ["Put IV (market)", pct(prow["iv_mid"], 1), pct(skew_vol(0.90, market.sigma), 1)],
                ], num_cols={1, 2})
                st.caption("The market smile is steeper than the model's −0.30 slope at short maturities, so listed downside puts cost "
                           "more than the model prices; the collar's call strike is correspondingly lower. The engine's overlays "
                           "are sized on the model; execution uses the chain.")
        term = [(e["expiry"], e["atm_iv"]) for e in exps]
        vixes = [("9-day", idx.get("^VIX9D")), ("30-day", idx.get("^VIX")), ("3-month", idx.get("^VIX3M")), ("6-month", idx.get("^VIX6M"))]
        st.caption("VIX term structure: " + " · ".join(f"{n_} {v_:.1f}" for n_, v_ in vixes if v_) +
                   (f" · SKEW {idx['^SKEW']:.0f}" if idx.get("^SKEW") else "") +
                   ". ATM term structure from the chain: " + " · ".join(f"{d_} {v_:.1%}" for d_, v_ in term) + ".")
    else:
        ui.notes(["Live option chain unavailable (offline mode, or Yahoo Finance is throttling this server's shared IP). "
                  "The model skew and the committed market snapshot are used for all pricing; turn Live market data off and on "
                  "again later to retry."], kind="Chain", tone="info")

    if plan.futures:
        fu = plan.futures
        ui.kpis([{"label": "Futures hedge", "value": f"Short {fu['contracts']} MES",
                  "sub": f"{money(fu['hedged_notional'])} notional · beta {fu['beta']:.2f} · defers ~{money(fu['tax_deferred'])} tax"
                         + (f" · MES {idx['MES=F']:,.0f}" if idx.get("MES=F") else ""), "tone": "info"}])
    ui.subsection("Pricing workbench", "Black-Scholes-Merton with a 1.2% dividend yield, the live T-bill rate and the model "
                  "volatility skew (downside puts cost more, as in listed S&P options).")
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1, st.container(key="card_chain"):
        rows = [[f"{k_:.0%}", f"{market.spot * k_:,.0f}", f"{skew_vol(k_, market.sigma):.1%}",
                 f"{option_cost(market, k_, 'put'):.2%}", f"{option_cost(market, k_, 'call'):.2%}"]
                for k_ in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)]
        ui.table(["Moneyness", "Strike", "Implied vol", "1-yr put", "1-yr call"], rows, num_cols={1, 2, 3, 4})
    with c2, st.container(key="card_calc"):
        pk = st.slider("Collar put strike (% of spot)", 0.80, 0.98, 0.90, 0.01)
        ck = zero_cost_call_strike(market, pk)
        notional = st.number_input("Equity to hedge with futures ($)", 0, 50_000_000, 250_000, 25_000)
        fh = futures_hedge(notional, market, micro=notional < 2_000_000)
        ui.kpis([{"label": "Zero-cost call strike", "value": f"{ck:.1%}", "sub": f"pays for the {pk:.0%} put", "tone": "info"},
                 {"label": "Contracts to short", "value": str(fh["contracts"]),
                  "sub": f"{fh['contract'].split(' (')[0]} · margin ~{money(fh['initial_margin_est'])}", "tone": "info"}])
    ui.notes(plan.not_recommended, kind="Out of scope", tone="bad")

# ------------------------------------------------------------------------ 07 Markets & Valuation
ui.section("Markets & valuation", "Where prices stand against fundamentals, from public sources. This section is context: "
           "it explains the risks in today's starting point and shows how far a valuation-aware forecaster would tilt the "
           "J.P. Morgan assumption. The engine keeps the published assumptions so the recommendation does not swing with "
           "the news.", eyebrow="Fair value", anchor="markets", num="07")
if blocked:
    stop_panel(el)
else:
    mac = get_macro(use_live)
    mv = market_valuation(mac)
    fg = mv["fear_greed"] or {}
    ui.kpis([
        {"label": "Shiller CAPE", "value": f"{mv['cape']:.1f}", "sub": f"mean since 1960: {mv['cape_mean']:.1f} · {mac.get('cape_date', '')}",
         "tone": "bad" if mv["cape"] > 30 else "mid" if mv["cape"] > 22 else "good"},
        {"label": "10-yr Treasury", "value": pct(mv["ten_year"], 2), "sub": f"real (TIPS) {pct(mv['real10'], 2)} · breakeven {pct(mv['breakeven'], 2)}", "tone": "info"},
        {"label": "Earnings yield − 10y", "value": f"{mv['fed_spread'] * 100:+.1f} pts", "sub": f"S&P P/E {mac['sp_pe']:.1f} · 'Fed model'",
         "tone": "bad" if mv["fed_spread"] < 0 else "good"},
        {"label": "Buffett indicator", "value": f"{mv['buffett'] * 100:.0f}% of GDP" if mv["buffett"] else "n/a",
         "sub": f"corporate equity / GDP · mean ~{mv['buffett_mean'] * 100:.0f}%", "tone": "bad" if (mv["buffett"] or 0) > 2 else "mid"},
        {"label": "VIX / 3-month VIX", "value": f"{mv['vix_ratio']:.2f}" if mv["vix_ratio"] else "n/a",
         "sub": mv["vix_regime"] or "", "tone": "bad" if (mv["vix_ratio"] or 0) > 1 else "good"},
        {"label": "Fear & Greed", "value": f"{fg.get('score', 0):.0f}" if fg else "n/a",
         "sub": f"{fg.get('rating', '')} · 1 month ago {fg.get('previous_1_month', 0):.0f}" if fg else "", "tone": "mid"},
    ])
    c1, c2 = st.columns([1.25, 1], gap="large")
    with c1, st.container(key="card_blocks"):
        ui.eyebrow("US large-cap expected return · building blocks")
        erp_ = mv["erp"] or {}
        ui.table(["Component", "Value", "Note"], [
            ["J.P. Morgan 2026 LTCMA (anchor)", pct(mv["jpm_us_lc"], 1), "10-15 yr compound nominal; what the engine uses"],
            ["Dividend yield", pct(mac["sp_div_yield"] / 100, 2), "S&P 500"],
            ["+ real earnings growth", "1.5%", "1871-2026 trend"],
            ["+ valuation change (CAPE half-reverts in 10 yrs)", pct((mv["cape_target"] / mv["cape"]) ** 0.1 - 1, 1),
             f"CAPE {mv['cape']:.1f} → {mv['cape_target']:.1f}"],
            ["= Research Affiliates-style real return", pct(mv["ra_real"], 1), f"+ breakeven inflation = {pct(mv['ra_nominal'], 1)} nominal"],
            ["Naive 1/CAPE real return", pct(mv["naive_real"], 1), "Campbell-Shiller"],
            ["Excess CAPE yield", pct(mv["excess_cape_yield"], 1), "1/CAPE − real 10y; lowest since 2000 when negative"],
            ["Implied equity risk premium", f"{erp_.get('erp_t12m', 0):.2f}%" if erp_.get("erp_t12m") else "n/a",
             f"Damodaran, {erp_.get('date', '')}; sustainable-payout {erp_.get('erp_sustainable') or 0:.2f}%"],
            ["Composite valuation z-score", f"{mv['z']:+.2f}", "CAPE, Fed spread, Buffett vs history"],
            ["Valuation tilt (capped ±1.5%)", f"{mv['tilt'] * 100:+.2f} pts", "−0.75% per z"],
            ["Valuation-adjusted expected return", pct(mv["adjusted_us_lc"], 1), "shown, not applied"],
        ], num_cols={1})
        st.caption("Applying the tilt would lower every equity-heavy model's expected return by about the same amount and raise the "
                   "required savings; it would not change the ranking of the recommended mixes because the recommendation is bound "
                   "by the client's loss limit, not by expected return.")
    with c2, st.container(key="card_curve"):
        tr, trr = mac["treasury"], mac["treasury_real"]
        order = ["1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "5 Yr", "7 Yr", "10 Yr", "20 Yr", "30 Yr"]
        xs = [k_ for k_ in order if k_ in tr]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=[tr[k_] for k_ in xs], name="Nominal par yield", line=dict(color=ui.NAVY, width=2.4), mode="lines+markers"))
        rx = [k_ for k_ in order if k_ in trr]
        fig.add_trace(go.Scatter(x=rx, y=[trr[k_] for k_ in rx], name="Real (TIPS) yield", line=dict(color=ui.GREEN, width=2.4, dash="dot"), mode="lines+markers"))
        fig.add_hline(y=2.5, line=dict(color=ui.AMBER, dash="dash", width=1))
        fig.update_layout(title=f"Treasury yield curve · {tr.get('asof', '')}", height=360, yaxis=dict(ticksuffix="%", title="Yield"),
                          xaxis=dict(title="Maturity"))
        ui.chart(fig)
        st.caption(f"Amber line: the model's 2.5% inflation. Breakeven inflation {pct(mv['breakeven'], 2)} vs 2.5% → TIPS edge "
                   f"{mv['tips_edge'] * 100:+.2f} pts. 2s10s {(mv['ten_year'] * 100 - (mv['two_year'] or 0)):+.2f} pts. "
                   f"Credit: IG spread {mv['ig_spread']:.2f}%, HY spread {mv['hy_spread']:.2f}% (tight → do not reach for credit).")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1, st.container(key="card_vixterm"):
        vt = [("9-day", mv["vix9d"]), ("30-day", mv["vix"]), ("3-month", mv["vix3m"]), ("6-month", mv["vix6m"])]
        vt = [(a_, b_) for a_, b_ in vt if b_]
        fig = go.Figure(go.Scatter(x=[a_ for a_, _ in vt], y=[b_ for _, b_ in vt], mode="lines+markers",
                                   line=dict(color=ui.RED, width=2.4), marker=dict(size=8)))
        fig.update_layout(title="VIX term structure", height=300, yaxis=dict(title="Implied vol (index points)"))
        ui.chart(fig)
        pc_ = mv["put_call"] or {}
        st.caption(f"SKEW {mv['skew']:.0f} (tail-risk pricing; >140 elevated). CBOE put/call {pc_.get('date', '')}: total "
                   f"{pc_.get('TOTAL', float('nan')):.2f}, equity {pc_.get('EQUITY', float('nan')):.2f}, index {pc_.get('INDEX', float('nan')):.2f}. "
                   f"Regime: {mv['vix_regime']}.")
    with c2, st.container(key="card_sources"):
        ui.eyebrow("Sources and freshness")
        ui.table(["Source", "Status", "Used for"], [
            ["Yahoo Finance (yfinance)", "live" if mac["sources"].get("Yahoo indices") else "snapshot", "S&P 500, VIX term structure, yields, futures, option chains, fundamentals"],
            ["FRED", "live" if mac["sources"].get("FRED") else "snapshot", "10y nominal/real yields, breakevens, credit spreads, GDP, corporate equity"],
            ["U.S. Treasury", "live" if mac["sources"].get("Treasury par curve") else "snapshot", "Par and real yield curves"],
            ["multpl.com", "live" if mac["sources"].get("multpl CAPE") else "snapshot", "Shiller CAPE, S&P P/E and dividend yield"],
            ["Damodaran (NYU Stern)", "live" if mac["sources"].get("Damodaran ERP") else "snapshot", "Implied equity risk premium"],
            ["CBOE", "live" if mac["sources"].get("CBOE put/call") else "snapshot", "Put/call ratios"],
            ["CNN", "live" if mac["sources"].get("CNN Fear & Greed") else "snapshot", "Fear & Greed index"],
        ])
        st.caption(f"As of {mac['asof']}. Snapshot values are the 2026-10-30 close committed with the code. Morningstar, GuruFocus and "
                   "Simply Wall St fair values are licensed data and are not scraped; the single-stock fair value above is computed "
                   "here from the same public inputs.")

# ------------------------------------------------------------------------ 08 Compliance
ui.section("Suitability file", "What a reviewer would check before this recommendation goes to the client: account "
           "opening (CIP, AML, customer due diligence), the Regulation Best Interest and fiduciary care obligations, "
           "and account protections.", eyebrow="Compliance", anchor="compliance", num="08")
status_tone = {"cleared": "good", "hold": "mid", "blocked": "bad"}[el.kyc_status]
ui.kpis([
    {"label": "Account status", "value": el.kyc_status.upper(), "sub": "CIP · OFAC · CDD", "tone": status_tone},
    {"label": "Options / futures", "value": f"L{el.options_level} / {'yes' if el.futures_ok else 'no'}", "sub": "approval on file", "tone": "info"},
    {"label": "Checklist", "value": f"{sum(1 for c_ in rec.checklist if c_[1])} / {len(rec.checklist)}", "sub": "items passing",
     "tone": "good" if all(c_[1] for c_ in rec.checklist) else "bad"},
    {"label": "Profile age", "value": f"{p.profile_age_months} mo", "sub": "refresh every 36 months", "tone": "good" if p.profile_age_months <= 36 else "bad"},
    {"label": "Advisor review items", "value": str(len(rp.review_flags)), "sub": "inconsistent answers / overrides", "tone": "bad" if rp.review_flags else "good"},
])
if el.kyc_reasons:
    ui.notes(el.kyc_reasons, kind="Account gate", tone="bad" if el.kyc_status != "cleared" else "mid")
c1, c2 = st.columns([1.5, 1], gap="large")
with c1:
    ui.checklist(rec.checklist)
with c2:
    with st.container(key="card_notes"):
        ui.eyebrow("Account notes")
        for n_ in el.notes:
            st.markdown(f"- {n_}")
    audit = {"generated": f"{CURRENT_YEAR} (model run)", "firm": "Keel & Co. (demo)", "profile": p.to_dict(),
             "kyc_status": el.kyc_status, "kyc_reasons": el.kyc_reasons, "risk": rp.summary(),
             "capacity_parts": rp.capacity_parts, "flags": rp.flags, "review_flags": rp.review_flags,
             "allocation": rec.weights_now.round(4).to_dict(), "glide_path": [round(float(x), 3) for x in rec.glide],
             "options_level": el.options_level, "options_reasons": el.options_reasons, "futures_ok": el.futures_ok,
             "rop_approval": {"principal": "ROP-demo", "date": str(date.today()), "verification_letter_sent": str(date.today())},
             "overlays": [o.describe() for o, _, _ in rec.overlays.overlays], "futures": rec.overlays.futures,
             "concentrated_plan": rec.overlays.concentrated_plan, "fees": {"advisory": f.advisory_fee, "etf": rec.stats["expense_ratio"], "total": rec.fee_total},
             "success_plan": rec.sim.success, "success_no_cut": rec.sim.success_full, "success_lifetime": rec.sim_life.success,
             "sustainable_spending_85": rec.sustainable, "checklist": [{"item": a, "pass": b, "evidence": c_} for a, b, c_ in rec.checklist],
             "levers": rec.levers.to_dict("records") if rec.levers is not None else None,
             "scenarios": rec.scenarios.to_dict("records") if rec.scenarios is not None else None,
             "limitations": dict(LIMITATIONS), "market": mkt_dict, "stock": stock_info,
             "cma": "J.P. Morgan 2026 LTCMA; history 2010-2026", "sim_config": json.loads(cfg_json)}
    st.download_button("Download audit record (JSON)", json.dumps(audit, indent=2, default=float),
                       file_name=f"robo_audit_{choice}.json", mime="application/json", width="stretch")
    st.download_button("Download client report (HTML, print to PDF)",
                       build_report(rec, sim_label=f"{cfg.n_paths:,} paths, {cfg.spending_rule} spending, seed {cfg.seed}"),
                       file_name=f"keel_client_report_{choice}.html", mime="text/html", type="primary", width="stretch")
with st.container(key="card_disclosures"):
    ui.eyebrow("Disclosures")
    st.markdown(
        "Projections are hypothetical, do not reflect actual results and are not guarantees. Returns use J.P. Morgan "
        "Asset Management's 2026 Long-Term Capital Market Assumptions (10-15 years, passive, before fees) less the stated "
        "fees; actual returns, inflation and taxes will differ. The risk questionnaire is not the sole basis of the "
        "recommendation: financial capacity, goals and stated loss limits are weighed with it, and inconsistent answers "
        "are referred to an advisor. Options involve risk and are not suitable for all investors; read *Characteristics "
        "and Risks of Standardized Options*. Futures trading involves substantial risk of loss. Buffered ETFs protect "
        "only if held for the full outcome period. Fair values in the Markets section are model estimates from public data. "
        "This tool is an educational project.")

# ------------------------------------------------------------------------ 09 Compare
ui.section("Three clients, three answers", "The assignment's clients side by side, on their base-case assumptions. "
           "Pick a client at the top of the page to edit their profile.", eyebrow="Case comparison", anchor="compare", num="09")
recs = {}
for k_, fn in CASES.items():
    pk_ = fn()
    sv_ = stock_vol if (stock and pk_.finances.concentrated_ticker == stock.get("ticker")) else None
    si_ = stock_info if sv_ is not None else {}
    recs[k_] = cached_advice(json.dumps(pk_.to_dict()), cfg_json, json.dumps(mkt_dict), json.dumps(prices), sv_, json.dumps(si_), light=True)
cards = []
for k_, r_ in recs.items():
    t_, v_ = ui.verdict_for(r_.sim.success)
    cards.append({"key": f"Case {k_} · {v_}", "title": CLIENTS[k_],
                  "rows": [("Plan succeeds", pct(r_.sim.success)), ("No spending cut", pct(r_.sim.success_full)),
                           ("Over lifetimes", pct(r_.sim_life.success)),
                           ("Long-term stocks", pct(r_.glide[0])), ("Risk tier", f"{r_.risk.tier} · {r_.risk.tier_name}"),
                           ("Expected return", pct(r_.stats["exp_return_geo"], 1)),
                           ("2008-style year", pct(r_.stress["2008 global financial crisis"])),
                           ("Sustainable spend", money(r_.sustainable)),
                           ("Options / futures", f"Level {r_.elig.options_level} / {'yes' if r_.elig.futures_ok else 'no'}")],
                  "note": f"Binding limit: {r_.risk.binding.split(' (')[0]}."})
ui.pillars(cards)
for k_, r_ in recs.items():
    with st.expander(f"Case {k_}: {r_.profile.label.split(': ', 1)[-1]} · full advice", expanded=False):
        ui.actions([a for a in r_.advice if not a.startswith(("**Flag", "**Review"))])
        ui.notes([ui.split_title(a)[1] for a in r_.advice if a.startswith("**Flag")])

# ------------------------------------------------------------------------ 10 Methodology
summ, models, front, stress_tbl = cached_methodology()
ui.section("How the robo decides", "The same sequence a bank advisor follows, made explicit and repeatable. The full "
           "derivation and literature review are in the accompanying research paper.", eyebrow="Methodology",
           anchor="method", num="10")
ui.steps([
    ("Open the account", "CIP identity check, OFAC screening, customer due diligence; a failure blocks, a pending review holds."),
    ("Know the client", "The FINRA 2111 / Reg BI investment profile, product-level experience, trusted contact, insider status."),
    ("Measure risk three ways", "Willingness (gamma, loss limit, consistency checks), ability (balance sheet, stress test, capacity) and need."),
    ("Bucket the goals", "Each goal gets its own horizon-matched mix; retirement gets a household glide path."),
    ("Build the models", "Resampled mean-variance on J.P. Morgan 2026 assumptions under investment-policy limits."),
    ("Gate the derivatives", "Options level 0-4, futures eligibility and issuer rules decide which overlays are allowed."),
    ("Stress the plan", "Monte Carlo with fat tails, fees, 2026 tax brackets, RMDs, Social Security, longevity, historical replays."),
    ("Document and review", "Reg BI checklist, audit record, client report, 36-month refresh and advisor-review flags."),
])
ui.subsection("From inputs to recommendations", "Every item the intake collects is used somewhere. This is the map.")
with st.container(key="card_map"):
    ui.table(["Input", "Where it is used"], [
        ["Age, sex, health, spouse", "Life tables and planning horizon; RMD age; Social Security full retirement age; survivor benefits; Rule 2165 senior protections"],
        ["Income, stability, growth", "Human capital (size and equity beta) in the Merton balance sheet; capacity score; Social Security estimate; contribution waterfall"],
        ["Savings rate, employer match, Roth share", "Cash-flow schedule; 2026 IRS limits and 415(c) cap; asset location; 'save more' lever"],
        ["Assets by tax wrapper, cost basis", "Investable base; asset location; capital-gains tax in withdrawals; futures hedge for embedded gains; Roth-conversion lever"],
        ["Single stock, ticker, insider status", "Concentration flag; collar priced at live implied vol; Rule 144 / 10b5-1 staged sale; issuer hedging ban; fair value"],
        ["Cash, essential expenses, 12-month withdrawals", "Liquidity check (3-6 months); liquidity flag; cash-secured-put eligibility; capacity score"],
        ["Debt and its rate", "Leverage in the capacity score; pay-down-before-investing flag"],
        ["Retirement age, spending, pension, Social Security", "Retirement goal and liability; need (required return); income reserve sizing; claiming analysis; guardrails"],
        ["Goals (amount, date, priority)", "Buckets with their own glide; needs funded first with a 3% buffer; wishes excluded from 'need'; bequest probability"],
        ["Objective, knowledge, experience by product, disclosures, margin", "Options approval level; futures eligibility; overlay choice (income vs protection)"],
        ["Risk questionnaire and income gamble", "Gamma (willingness); max-loss cap via 2008/2022 stress test; consistency checks and advisor-review flags"],
        ["Client override", "More conservative: applied and logged; more aggressive: flagged for advisor sign-off"],
        ["Account type, citizenship, tax residency, source of wealth, PEP", "Account gate (block/hold); IRA-only limits on margin and futures; non-US tax notes; EDD"],
        ["Advisory fee, tax rates, state", "Fee drag on every simulated return; federal brackets + state rate on withdrawals, RMDs and conversions"],
    ])
ui.subsection("Reasonably available alternatives", "Reg BI asks whether a simpler product would serve the client as well. "
              "The recommended mix next to the obvious alternatives.")
with st.container(key="card_raa"):
    def _alt(name, w, cost):
        s_ = portfolio_stats(np.asarray(w, float), CMA)
        sr_ = stress_returns(np.asarray(w, float), CMA)
        return [name, pct(sum(w[i_] for i_, k_ in enumerate(CMA.keys) if k_ in ("us_lc", "us_sc", "intl_dev", "em", "reit"))),
                pct(s_["exp_return_geo"], 1), pct(s_["vol"], 1), pct(sr_["2008 global financial crisis"], 1), pct(cost, 2)]
    tdf_eq = 0.90 if h.age <= 40 else max(0.30, 0.90 - 0.016 * (h.age - 40))
    ui.table(["Portfolio", "Stocks", "Exp. return", "Volatility", "2008-style year", "All-in cost"], [
        _alt(f"Recommended · {rp.tier_name}", rec.weights_now.to_numpy(float), rec.fee_total),
        _alt("Target-date fund (age-based glide)", blend_for_equity(tdf_eq), 0.0008 + f.advisory_fee),
        _alt("60 / 40 index mix", blend_for_equity(0.60), 0.0004 + f.advisory_fee),
        _alt("All cash and short Treasuries", blend_for_equity(0.0), 0.0005),
    ], num_cols={1, 2, 3, 4, 5})
    st.caption("A target-date fund is the right answer for many savers; this plan differs by holding near-term goals in their own "
               "safe buckets, respecting the client's loss limit, locating assets by tax wrapper, and adding protection only "
               "where the account approvals allow.")
ui.subsection("Capital market assumptions", "Expected returns are J.P. Morgan's 10-15 year forecasts. Volatility comes "
              "from their published chart where available, otherwise from ETF history.")
with st.container(key="card_cma"):
    st.dataframe(CMA.table(), width="stretch", hide_index=True, column_config={
        "Exp. return (geo)": st.column_config.NumberColumn(format="percent"),
        "Exp. return (arith)": st.column_config.NumberColumn(format="percent"),
        "Volatility": st.column_config.NumberColumn(format="percent")})
c1, c2 = st.columns(2, gap="large")
with c1, st.container(key="card_corr"):
    fig = go.Figure(go.Heatmap(z=CMA.corr, x=list(CMA.keys), y=list(CMA.keys), zmin=-1, zmax=1,
                               colorscale=[[0, ui.RED], [0.5, "#F8F7F3"], [1, ui.GREEN]],
                               text=np.round(CMA.corr, 2), texttemplate="%{text}", textfont=dict(size=8),
                               colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(title=f"Correlations, {CMA.meta['hist_start'][:4]}-{CMA.meta['hist_end'][:4]}", height=520,
                      yaxis=dict(autorange="reversed"))
    ui.chart(fig)
with c2, st.container(key="card_frontier"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=front["vol"], y=front["exp_return_arith"], name="Efficient frontier", line=dict(color="#B9B5AC", width=2)))
    fig.add_trace(go.Scatter(x=CMA.vol, y=CMA.mu_arith, mode="markers+text", text=list(CMA.keys), name="Asset classes",
                             textposition="bottom center", marker=dict(size=7, color=ui.AMBER), textfont=dict(size=9, color=ui.MUTED)))
    fig.add_trace(go.Scatter(x=summ["vol"], y=summ["exp_return_arith"], mode="lines+markers+text", name="Model tiers 0-10",
                             text=[str(t) for t in summ.index], textposition="top center",
                             marker=dict(size=9, color=ui.NAVY), line=dict(color=ui.NAVY, width=1.5)))
    fig.update_layout(title="Assets, frontier and the 11 models", height=520,
                      xaxis=dict(tickformat=".0%", title="Volatility"), yaxis=dict(tickformat=".1%", title="Expected return"))
    ui.chart(fig)
ui.subsection("Model portfolios", "Resampled mean-variance optimisation, policy-constrained, one model per 10% of equity. "
              "Tier 0 is a liability-matching sleeve (T-bills, short Treasuries, TIPS) for money needed within two years.")
with st.container(key="card_models"):
    fig = go.Figure()
    for k_ in CMA.keys:
        fig.add_trace(go.Bar(x=[f"{t} · {TIER_NAMES[t]}" for t in models.index], y=models[k_], name=NAME[k_],
                             marker=dict(color=C[k_], line=dict(width=0)), hovertemplate=f"{NAME[k_]}: " + "%{y:.1%}<extra></extra>"))
    fig.update_layout(barmode="stack", height=440, bargap=0.25, yaxis=dict(tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.35, font=dict(size=10)))
    ui.chart(fig)
    st.dataframe(pd.concat([summ[["name", "equity", "exp_return_geo", "vol", "sharpe", "real_return_geo"]], stress_tbl], axis=1),
                 width="stretch", column_config={
                     c_: st.column_config.NumberColumn(format="percent") for c_ in
                     ["equity", "exp_return_geo", "vol", "real_return_geo", "2008 global financial crisis",
                      "2022 stock-bond inflation shock", "worst"]} | {"sharpe": st.column_config.NumberColumn(format="%.2f")})
ui.subsection("Longevity", "Gompertz-Makeham mortality fitted exactly to SSA 2022 life expectancies at ages 30, 55 and 70.")
with st.container(key="card_longevity"):
    fig = go.Figure()
    for sx, col in (("female", ui.AMBER), ("male", ui.NAVY)):
        for a0 in (30, 55, 70):
            s_ = survival_table(sx, a0)
            fig.add_trace(go.Scatter(x=s_["ages"], y=s_["survival"], name=f"{sx.capitalize()} {a0} (LE {s_['life_expectancy']:.1f})",
                                     line=dict(color=col, dash={30: "dot", 55: "dash", 70: "solid"}[a0], width=2)))
    fig.update_layout(height=380, yaxis=dict(tickformat=".0%", title="Probability alive"), xaxis_title="Age",
                      legend=dict(orientation="h", y=-0.22))
    ui.chart(fig)
ui.subsection("What this tool does not do yet", "Known limits, so nobody mistakes a model for the world.")
with st.container(key="card_limits"):
    ui.table(["Area", "Not modelled"], [[a, b] for a, b in LIMITATIONS])
with st.expander("Glossary"):
    ui.table(["Term", "Meaning"], [
        ["Risk aversion (gamma)", "How much expected return you need to accept more volatility; higher means more cautious."],
        ["Human capital", "Today's value of the savings you will add from future paychecks."],
        ["Funded ratio", "Assets plus future savings and income, divided by the cost of your goals."],
        ["Glide path", "How the stock/bond mix changes with age."],
        ["Sequence-of-returns risk", "The damage a bad market does when it hits just before or after retirement."],
        ["Guardrails", "Spending rules that trim withdrawals after bad markets and raise them after good ones."],
        ["Collar", "Buying a put for downside protection, paid for by selling a call that caps the upside."],
        ["Covered call", "Selling a call on shares you own: income now in exchange for capped upside."],
        ["Buffered ETF", "A fund that absorbs the first part of a loss (e.g. 10%) and caps gains for a set period."],
        ["Implied volatility / skew", "The volatility an option price implies; skew means downside puts price in more volatility than upside calls."],
        ["CAPE", "Price divided by ten-year average inflation-adjusted earnings; a slow-moving valuation gauge."],
        ["Roth conversion", "Moving money from a pre-tax IRA to a Roth IRA, paying income tax now so later growth is tax-free."],
        ["RMD", "Required minimum distribution: the IRS-mandated annual withdrawal from pre-tax accounts from age 73 or 75."],
        ["Rule 10b5-1 plan", "A pre-set schedule that lets a company insider sell shares without trading on inside information."],
        ["Monte Carlo", "Running the plan through thousands of randomly generated market histories."],
    ])
ui.footer()
