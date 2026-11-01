"""Keel & Co. goals-based robo-advisor (Streamlit).  Run:  streamlit run app.py"""
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
from robo.advisor import Recommendation, advise
from robo.cases import CASES
from robo.cma import load_cma
from robo.derivatives import Market, futures_hedge, option_cost, payoff_curve, skew_vol, zero_cost_call_strike
from robo.goals import planning_age
from robo.kyc import (GAMBLE_QUESTIONS, QUESTIONS, ClientProfile, Experience, Finances, Goal, Household,
                      Identity, profile_from_dict)
from robo.market_data import etf_prices, live_market
from robo.mortality import survival_table
from robo.portfolio import TIER_NAMES, blend_for_equity, efficient_frontier, load_models, models_summary
from robo.retirement import CURRENT_YEAR
from robo.risk import model_stress_table
from robo.simulate import SimConfig

st.set_page_config(page_title="Keel & Co. | Goals-Based Advice", page_icon=":material/anchor:", layout="wide",
                   initial_sidebar_state="expanded")
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


def money(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else (f"${x / 1e3:,.0f}K" if abs(x) >= 1e4 else f"${x:,.0f}")


def pct(x, d: int = 0) -> str:
    return "n/a" if x is None else f"{x:.{d}%}"


# ======================================================================== cached engine calls
@st.cache_data(ttl=3600, show_spinner=False)
def get_market(use_live: bool) -> dict:
    return dataclasses.asdict(live_market() if use_live else Market())


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(use_live: bool) -> dict:
    return etf_prices(list(CMA.etf)) if use_live else {}


@st.cache_data(show_spinner=False, max_entries=48)
def cached_advice(profile_json: str, cfg_json: str, market_json: str, prices_json: str) -> Recommendation:
    p = profile_from_dict(json.loads(profile_json))
    return advise(p, Market(**json.loads(market_json)), SimConfig(**json.loads(cfg_json)), prices=json.loads(prices_json))


@st.cache_data(show_spinner=False)
def cached_methodology():
    return models_summary(), load_models(), efficient_frontier(), model_stress_table()


# ======================================================================== sidebar
ui.sidebar_brand()
ui.sidebar_heading("Client")
choice = st.sidebar.radio("Client", list(CLIENTS), format_func=CLIENTS.get,
                          captions=[CLIENT_CAPTIONS[k] for k in CLIENTS], label_visibility="collapsed")
ui.sidebar_heading("Settings")
use_live = st.sidebar.toggle("Live market data", value=True,
                             help="S&P 500, VIX and T-bill yield price the options and size futures; ETF prices give "
                                  "share counts. Long-term return assumptions never come from live prices.")
with st.sidebar.expander("Simulation"):
    n_paths = st.select_slider("Market scenarios", [1000, 2000, 3000, 5000, 10000], value=3000)
    spending_rule = st.radio("Retirement spending", ["guardrails", "fixed"],
                             format_func=lambda s: "Guardrails (adjusts with markets)" if s == "guardrails" else "Fixed in real terms")
    ss_short = st.checkbox("Social Security trust-fund cut (78% from 2033)", value=True)
    fat = st.checkbox("Fat-tailed returns", value=True)
    seed = st.number_input("Random seed", value=2026, step=1)
cfg = SimConfig(n_paths=int(n_paths), seed=int(seed), fat_tails=fat, spending_rule=spending_rule, apply_ss_shortfall=ss_short)
mkt_dict = get_market(use_live)
market = Market(**mkt_dict)
prices = get_prices(use_live)
ui.sidebar_market(market.spot, market.vix, market.rate, market.live, market.asof)
ui.sidebar_foot("Return assumptions: J.P. Morgan 2026 Long-Term Capital Market Assumptions. Volatility and "
                "correlation: ETF history 2010-2026.", "Educational project. Not investment advice.")


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


def kyc_form(k: str) -> ClientProfile:
    b = _default_profile(k)
    i, h, f, x = b.identity, b.household, b.finances, b.experience
    K = lambda s: f"{k}_{s}"
    HEALTH = ["excellent", "good", "average", "fair", "poor"]
    MARITAL = ["single", "married", "divorced", "widowed"]

    ui.section("Client profile", "Seven short steps: the same information a bank collects when it opens an advisory "
               "account. Cases A-C are pre-filled with documented assumptions. Change anything and every page updates.",
               eyebrow="Know your customer")
    steps = st.tabs(["01  Identity", "02  Household", "03  Balance sheet", "04  Retirement & tax", "05  Goals",
                     "06  Experience", "07  Risk questionnaire"])

    with steps[0], st.container(key="card_kyc_identity"):
        st.caption("Why we ask: identity is verified under the Customer Identification Program (31 CFR 1023.220) and "
                   "screened against OFAC sanctions lists before an account can open. This demo does not collect an SSN.")
        c = st.columns(3)
        full_name = c[0].text_input("Legal name", i.full_name, key=K("name"))
        cit_opts = ["US citizen", "US resident alien", "Non-resident alien"]
        citizenship = c[1].selectbox("Citizenship / tax status", cit_opts, index=_idx(cit_opts, i.citizenship), key=K("cit"))
        state = c[2].text_input("State of residence", i.state, key=K("state"))
        c = st.columns(3)
        emp_opts = ["employed", "self-employed", "retired", "not employed"]
        employment = c[0].selectbox("Employment status", emp_opts, index=_idx(emp_opts, i.employment_status), key=K("emp"))
        occupation = c[1].text_input("Occupation", i.occupation, key=K("occ"))
        trusted = c[2].text_input("Trusted contact (FINRA 4512)", i.trusted_contact, key=K("tc"))
        c = st.columns(5)
        idv = c[0].checkbox("ID verified", i.identity_verified, key=K("idv"))
        ofac = c[1].checkbox("OFAC clear", i.ofac_clear, key=K("ofac"))
        bd = c[2].checkbox("Works at a broker-dealer", i.employed_by_broker_dealer, key=K("bd"))
        ctrl = c[3].checkbox("Public-company insider", i.control_person_public_co, key=K("ctrl"))
        pep = c[4].checkbox("Politically exposed", i.politically_exposed, key=K("pep"))

    with steps[1], st.container(key="card_kyc_household"):
        st.caption("Why we ask: age, health and a spouse's details set the planning horizon, life tables, survivor "
                   "benefits and the age required withdrawals begin.")
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
        st.markdown("**Other assets and debts**")
        c = st.columns(4)
        conc = c[0].number_input("Single stock (inside taxable)", 0, 100_000_000, int(f.concentrated_stock), 5_000, key=K("conc"))
        home = c[1].number_input("Home value", 0, 100_000_000, int(f.home_value), 10_000, key=K("home"))
        mort = c[2].number_input("Mortgage balance", 0, 100_000_000, int(f.mortgage), 10_000, key=K("mort"))
        debt = c[3].number_input("Other debt", 0, 10_000_000, int(f.other_debt), 1_000, key=K("debt"))

    with steps[3], st.container(key="card_kyc_retire"):
        st.caption("Why we ask: retirement timing, spending and guaranteed income set the size of the biggest goal. "
                   "Without an SSA statement, benefits are estimated with the 2026 PIA formula.")
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
        ord_rate = c[0].number_input("Marginal tax rate, ordinary (%)", 0, 60, int(f.ordinary_tax_rate * 100), key=K("ord")) / 100
        cg_rate = c[1].number_input("Capital gains rate (%)", 0, 40, int(f.cap_gains_rate * 100), key=K("cg")) / 100

    with steps[4], st.container(key="card_kyc_goals"):
        st.caption("Each goal becomes its own bucket. **Needs** are funded first and with a safety margin; **wishes**, "
                   "such as a legacy, are not counted when measuring how much return the plan requires. Retirement comes "
                   "from step 04.")
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
        st.caption("Why we ask: options require written approval under FINRA Rule 2360 and futures a risk disclosure "
                   "under NFA Rule 2-30 / CFTC Rule 1.55. Experience and objectives set the approval level.")
        c = st.columns(4)
        yrs = c[0].number_input("Years investing", 0, 80, x.years_investing, key=K("yrs"))
        KN = ["none", "limited", "good", "extensive"]
        know = c[1].selectbox("Investment knowledge", KN, index=_idx(KN, x.knowledge), key=K("know"))
        OB = ["preservation", "income", "growth", "speculation"]
        obj = c[2].selectbox("Primary objective", OB, index=_idx(OB, x.objective), key=K("obj"))
        opt_yrs = c[3].number_input("Years trading options", 0, 60, x.options_years, key=K("optyrs"))
        c = st.columns(5)
        fut_yrs = c[0].number_input("Years trading futures", 0, 60, x.futures_years, key=K("futyrs"))
        odd = c[1].checkbox("Received options disclosure", x.options_disclosure_ack, key=K("odd"))
        fdis = c[2].checkbox("Signed futures disclosure", x.futures_disclosure_ack, key=K("fdis"))
        margin = c[3].checkbox("Margin agreement", x.wants_margin, key=K("margin"))
        nod = c[4].checkbox("Exclude derivatives", x.exclude_derivatives, key=K("nod"))

    with steps[6], st.container(key="card_kyc_risk"):
        st.caption("Ten questions in the Grable-Lytton / FinaMetrica tradition, then an income gamble from the Health "
                   "and Retirement Study that estimates risk aversion directly. There are no right answers.")
        answers = {}
        for n, q in enumerate(QUESTIONS, 1):
            default = b.risk_answers.get(q["id"], len(q["options"]) // 2)
            pick = st.radio(f"{n}. {q['text']}", q["options"], index=default, horizontal=True, key=K(q["id"]))
            answers[q["id"]] = q["options"].index(pick)
        st.markdown("**Income gamble.** Imagine a job that pays your current income for life.")
        gamble = {}
        g1 = st.radio(GAMBLE_QUESTIONS[0]["text"], ["No", "Yes"], index=b.gamble_answers.get("g1", 0), horizontal=True, key=K("g1"))
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
    for _, r in edited.dropna(subset=["name"]).iterrows():
        if str(r["name"]).strip():
            goals.append(Goal(str(r["name"]), r.get("kind") or "other", float(r.get("amount") or 0),
                              int(r.get("years_until") or 0), int(r.get("duration_years") or 1), r.get("priority") or "want"))
    return ClientProfile(
        label=b.label,
        identity=Identity(full_name=full_name, citizenship=citizenship, state=state, employment_status=employment,
                          occupation=occupation, employed_by_broker_dealer=bd, control_person_public_co=ctrl,
                          politically_exposed=pep, identity_verified=idv, ofac_clear=ofac, trusted_contact=trusted),
        household=Household(age=int(age), sex=sex, health=health, marital_status=marital,
                            spouse_age=int(sp_age) if sp_age else None, spouse_sex=sp_sex, spouse_health=sp_health,
                            dependents=int(deps)),
        finances=Finances(
            annual_income=float(income), spouse_income=float(sp_income) if marital == "married" else 0.0,
            income_real_growth=growth, income_stability=stab, savings_rate=sav, employer_match=match,
            contrib_to_roth_share=roth_sh, taxable=float(taxable), traditional=float(trad), roth=float(roth),
            cash=float(cash), taxable_cost_basis_pct=basis, concentrated_stock=float(min(conc, taxable)),
            home_value=float(home), mortgage=float(mort), other_debt=float(debt),
            monthly_essential_expenses=float(essential_m), retirement_age=int(ret_age),
            retirement_spending=float(ret_spend), essential_share_of_spending=ess_share,
            social_security=float(ss) if use_stmt else None,
            spouse_social_security=float(sp_ss) if (use_stmt and sp_ss is not None) else None,
            ss_claim_age=int(claim), spouse_ss_claim_age=int(sp_claim), already_claiming_ss=claiming,
            pension=float(pension), ordinary_tax_rate=ord_rate, cap_gains_rate=cg_rate),
        goals=goals,
        experience=Experience(years_investing=int(yrs), knowledge=know, options_years=int(opt_yrs),
                              futures_years=int(fut_yrs), objective=obj, options_disclosure_ack=odd,
                              futures_disclosure_ack=fdis, wants_margin=margin, exclude_derivatives=nod),
        risk_answers=answers, gamble_answers=gamble,
    )


# ======================================================================== charts
def donut(w: pd.Series, center: str, height: int = 300, legend: bool = True) -> go.Figure:
    w = w[w > 1e-4]
    fig = go.Figure(go.Pie(labels=[NAME[k] for k in w.index], values=w.values, hole=0.66, sort=False, direction="clockwise",
                           marker=dict(colors=[C[k] for k in w.index], line=dict(color=ui.CARD, width=2)),
                           textinfo="none", hovertemplate="%{label}<br>%{percent}<extra></extra>"))
    fig.add_annotation(text=center, showarrow=False, font=dict(family="Newsreader, serif", size=22, color=ui.INK))
    fig.update_layout(height=height, showlegend=legend, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11)))
    return fig


def fan(sim, profile, overlay=None, height: int = 420, title: str = "") -> go.Figure:
    pc = sim.pct
    fig = go.Figure()
    band = lambda lo, hi, a, name: [
        go.Scatter(x=pc.index, y=pc[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=pc.index, y=pc[lo], fill="tonexty", fillcolor=f"rgba(31,58,95,{a})", line=dict(width=0), name=name,
                   hoverinfo="skip")]
    for tr in band("p5", "p95", 0.10, "5th-95th percentile") + band("p25", "p75", 0.22, "25th-75th percentile"):
        fig.add_trace(tr)
    fig.add_trace(go.Scatter(x=pc.index, y=pc["p50"], line=dict(color=ui.NAVY, width=2.6), name="Median",
                             hovertemplate="Age %{x}<br>Median %{y:$,.0f}<extra></extra>"))
    if overlay is not None:
        fig.add_trace(go.Scatter(x=pc.index, y=overlay.pct["p5"], line=dict(color=ui.BRASS, dash="dot", width=2),
                                 name="5th pct with overlay"))
    f = profile.finances
    if profile.household.age < f.retirement_age:
        fig.add_vline(x=f.retirement_age, line=dict(color=ui.BRASS, width=1, dash="dot"))
        fig.add_annotation(x=f.retirement_age, y=1, yref="paper", text="Retire", showarrow=False, yshift=8,
                           font=dict(size=11, color=ui.BRASS))
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
                           textfont=dict(family="IBM Plex Mono", size=11, color=ui.INK2), cliponaxis=False))
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=ui.INK, dash="dash", width=1.2))
        fig.add_annotation(x=vline, y=1.06, yref="paper", text=vlabel, showarrow=False, font=dict(size=11, color=ui.INK))
    fig.update_layout(height=height, bargap=0.38, xaxis=dict(range=list(rng), tickformat=fmt, showgrid=False),
                      yaxis=dict(autorange="reversed"), margin=dict(l=8, r=30, t=34, b=8))
    return fig


# ======================================================================== layout
mast = st.container()
tabs = st.tabs(["Overview", "Profile", "Risk", "Portfolio", "Projections", "Options & Futures", "Compliance",
                "Compare", "Methodology"])

with tabs[1]:
    profile = kyc_form(choice)

cfg_json = json.dumps({k: v for k, v in dataclasses.asdict(cfg).items() if k != "overlays"})
with st.spinner(f"Profiling risk and running {cfg.n_paths:,} market scenarios..."):
    rec = cached_advice(json.dumps(profile.to_dict()), cfg_json, json.dumps(mkt_dict), json.dumps(prices))
rp, p, f, h = rec.risk, rec.profile, rec.profile.finances, rec.profile.household
sim, life = rec.sim, rec.sim_life
PA = planning_age(p)
retired = h.age >= f.retirement_age
tone, verdict = ui.verdict_for(sim.success)
advice_main = [a for a in rec.advice if not a.startswith("**Flag")]
advice_flags = [ui.split_title(a)[1] for a in rec.advice if a.startswith("**Flag")]

with mast:
    hh = "Married" + (f", spouse {h.spouse_age}" if h.spouse_age else "") if h.marital_status == "married" else h.marital_status.capitalize()
    chips = [("Age", str(h.age)), ("Household", hh), ("Invested", money(f.investable)), ("Bank cash", money(f.cash)),
             ("Retirement", "Retired" if retired else f"at {f.retirement_age}"),
             ("Goals", str(len([g for g in p.goals if g.kind != "retirement"]) + 1)), ("Risk tier", f"{rp.tier} of 10")]
    sub = (f"**{verdict}.** The plan works in {sim.success:.0%} of {cfg.n_paths:,} simulated markets. Recommended: the "
           f"*{rp.tier_name}* portfolio, {rec.glide[0]:.0%} in stocks for long-term money.")
    ui.masthead(f"Client plan  ·  {CLIENTS[choice]}  ·  {date.today():%B %d, %Y}", HEADLINES[choice], sub, chips)

# ------------------------------------------------------------------------ Overview
with tabs[0]:
    left, right = st.columns([1.3, 1], gap="large")
    with left, st.container(key="card_statement"):
        spend = money(f.retirement_spending)
        if rec.levers is not None:
            best = rec.levers.sort_values("P(success, plan to age 95)", ascending=False).iloc[0]
            sentence = (f"In {sim.success:.0%} of simulated markets the plan pays **{spend} a year** to age {PA}. The gap is a "
                        f"planning problem, not an investing one: *{best['Lever'].lower()}* lifts it to "
                        f"**{best['P(success, plan to age 95)']:.0%}**. See Projections.")
        elif retired:
            sentence = (f"Your **{spend} a year** of spending lasts to age {PA} in {sim.success:.0%} of simulated markets, "
                        f"with a median estate of **{money(float(np.median(life.terminal)))}** at the second death.")
        else:
            sentence = (f"Saving as planned funds **{spend} a year** from age {f.retirement_age} to {PA} in "
                        f"{sim.success:.0%} of simulated markets.")
        third = (("Median estate", money(float(np.median(life.terminal)))) if retired
                 else ("Median at retirement", money(float(np.nanmedian(sim.retire_wealth)))))
        ui.plan_statement(sim.success, sentence, [
            ("Over simulated lifetimes", pct(life.success)),
            ("Sustainable spending (85%)", f"{money(rec.sustainable)}/yr"),
            third,
            ("Spending rule", "Guardrails" if cfg.spending_rule == "guardrails" else "Fixed real")])
    with right:
        ui.kpis([
            {"label": "Risk profile", "value": rp.tier_name, "sub": f"Tier {rp.tier} of 10", "tone": "info"},
            {"label": "Stocks, long-term money", "value": pct(rec.glide[0]),
             "sub": f"{pct(rec.stats['equity'])} across all buckets", "tone": "info"},
            {"label": "Expected return", "value": pct(rec.stats["exp_return_geo"], 1),
             "sub": f"{pct(rec.stats['vol'], 1)} volatility, nominal", "tone": "good"},
            {"label": "In a 2008-style year", "value": pct(rec.stress["2008 global financial crisis"]),
             "sub": f"2022-style: {pct(rec.stress['2022 stock-bond inflation shock'])}", "tone": "bad"},
        ])
        with st.container(key="card_alloc_mini"):
            st.html('<div class="eyebrow">Allocation today, all buckets</div>')
            ui.alloc_split(rec.weights_now.to_dict())
            st.caption(f"{len(rec.implementation)} ETFs · weighted expense ratio {rec.stats['expense_ratio']:.2%}. "
                       "Full breakdown on the Portfolio page.")

    ui.section("What to do next", "The most important actions, in order. The Portfolio page explains each one in full.",
               eyebrow="Next steps")
    ui.actions(advice_main[:6])

    ui.section("Your goals", "Each goal is its own bucket with its own investments and its own chance of being met.",
               eyebrow="Goals")
    gcards = [{"name": "Retirement", "priority": "need", "prob": sim.success,
               "meta": f"{money(f.retirement_spending)}/yr " + ("now" if retired else f"from age {f.retirement_age}") + f" to {PA}"}]
    for _, bk in rec.buckets.iterrows():
        if bk["Bucket"] in sim.goal_success:
            target = money(bk["Target (today's $)"])
            gcards.append({"name": bk["Bucket"], "priority": bk["Priority"], "prob": sim.goal_success[bk["Bucket"]],
                           "meta": f"{target} in {bk['Horizon (yrs)']} yrs · {bk['Equity now']:.0%} stocks today"})
    for g in p.goals:
        if g.kind == "bequest" and g.amount > 0:
            gcards.append({"name": g.name, "priority": g.priority, "prob": float(np.mean(life.terminal >= g.amount)),
                           "meta": f"{money(g.amount)} estate target · simulated lifetimes"})
    ui.goal_cards(gcards)

    ui.section("The range of outcomes", "Financial wealth in today's dollars. The dark band holds the middle half of "
               "outcomes; the light band holds 90% of them.", eyebrow="Projection")
    with st.container(key="card_fan_overview"):
        ui.chart(fan(sim, p, height=360))
    if advice_flags:
        ui.notes(advice_flags)

# ------------------------------------------------------------------------ Risk
with tabs[2]:
    ui.section("How much risk is right", "We measure three different things. The recommendation is the lower of what "
               "you are willing and able to take; what your goals need is never used to push risk up.", eyebrow="Risk profile")
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
                  ("Recommended stocks", pct(rp.recommended_equity)), ("Binding limit", rp.binding.split(" (")[0])],
         "note": "The return that makes assets plus future savings and income exactly pay for the goals."},
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
        cols = ["#D9D0BE"] * 5
        cols[bi] = ui.SPRUCE
        cols[4] = "#B9A6BF"
        fig = bar_h(labels, vals, cols, vline=rp.recommended_equity, vlabel=f"Recommended {rp.recommended_equity:.0%}", height=320)
        fig.update_layout(title="Equity limits from each dimension")
        ui.chart(fig)
    with c2, st.container(key="card_capacity"):
        cp = pd.Series(rp.capacity_parts)
        fig = bar_h(list(cp.index), list(cp.values / 100), [ui.SAGE] * len(cp), height=320)
        fig.update_layout(title="Capacity components (0-100)")
        ui.chart(fig)
    ui.section("Household balance sheet", "Today's dollars. Human capital is the present value of future portfolio "
               "contributions; guaranteed income is Social Security and pensions.", eyebrow="Balance sheet")
    with st.container(key="card_bs"):
        ui.table(["Item", "Value", "How it is used"], [
            ["Financial assets", money(f.investable), "The portfolio the robo manages"],
            ["Human capital", money(rp.human_capital), "Bond-like with equity beta " + f"{rp.beta_hc:.2f}"],
            ["Social Security + pension (PV)", money(rp.guaranteed_pv), "Bond-like, inflation-linked"],
            ["Funded ratio", f"{rp.funded_ratio:.2f}x", "Assets + future savings + income vs goals, at ~2% real"],
            ["Bank cash", money(f.cash), f"{f.cash / max(f.monthly_essential_expenses, 1):.1f} months of essentials"],
        ], num_cols={1})
    ui.notes(rp.flags)

# ------------------------------------------------------------------------ Portfolio
with tabs[3]:
    ui.section(f"The {rp.tier_name} portfolio", f"{rec.glide[0]:.0%} stocks for long-term money, built from 15 low-cost "
               f"ETFs (weighted expense ratio {rec.stats['expense_ratio']:.2%}). Near-term goals hold their own, safer mix.",
               eyebrow="Recommendation")
    ui.actions(advice_main)
    ui.notes(advice_flags)
    ui.section("Goal buckets", "Money is earmarked by goal, then invested to match that goal's date and priority.",
               eyebrow="Buckets")
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
        st.html('<div class="eyebrow">Household allocation</div>')
        ui.chart(donut(rec.weights_now, f"{rec.stats['equity']:.0%}<br><span style='font-size:11px'>stocks</span>",
                       height=430, legend=False))
        st.caption(f"2008-style year {rec.stress['2008 global financial crisis']:.1%} · 2022-style "
                   f"{rec.stress['2022 stock-bond inflation shock']:.1%}")
    with c2, st.container(key="card_etfs"):
        st.html('<div class="eyebrow">ETF implementation</div>')
        impl = rec.implementation.drop(columns=["Key"]).copy()
        impl["Amount"] = impl["Amount"].map(lambda v: f"${v:,.0f}")
        impl["Shares"] = impl["Shares"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")
        st.dataframe(impl, width="stretch", hide_index=True, height=430,
                     column_config={"Weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=0.6, format="percent"),
                                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Expense ratio": st.column_config.NumberColumn(format="percent")})
    ui.section("Glide path", "How the retirement portfolio's mix changes with age. It comes from your own balance sheet, "
               "is held steady around retirement (the sequence-of-returns window) and never rises again after that.",
               eyebrow="Over time")
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
            fig.add_vline(x=f.retirement_age, line=dict(color=ui.INK, dash="dot", width=1))
        fig.update_layout(height=400, yaxis=dict(tickformat=".0%", range=[0, 1]), xaxis_title="Age",
                          legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        ui.chart(fig)
    ui.section("Asset location", "Which account holds what. Tax-free Roth growth goes to the highest-return assets, "
               "tax-deferred accounts hold the least tax-efficient ones, and broad stock ETFs sit in the taxable account.",
               eyebrow="Taxes")
    with st.container(key="card_location"):
        loc = rec.location
        fig = go.Figure()
        for acct, col in zip(loc.columns, ["#6B4E71", ui.SPRUCE, ui.NAVY]):
            fig.add_trace(go.Bar(y=loc.index, x=loc[acct], name=acct, orientation="h", marker_color=col,
                                 hovertemplate=f"{acct}: " + "%{x:$,.0f}<extra></extra>"))
        fig.update_layout(barmode="stack", height=440, xaxis=dict(tickprefix="$", tickformat="~s"),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08, x=0))
        ui.chart(fig)
        st.caption("Existing appreciated holdings should be moved gradually, not sold just to relocate them.")

# ------------------------------------------------------------------------ Projections
with tabs[4]:
    ui.section("Projections", f"{cfg.n_paths:,} simulated markets with fat-tailed returns, taxes, required withdrawals, "
               f"Social Security{' (with the 2033 trust-fund cut)' if ss_short else ''} and life expectancy. "
               "All values in today's dollars.", eyebrow="Monte Carlo")
    ui.kpis([
        {"label": f"Success to age {PA}", "value": pct(sim.success), "tone": tone},
        {"label": "Over simulated lifetimes", "value": pct(life.success), "tone": ui.verdict_for(life.success)[0]},
        {"label": "Median at retirement" if not retired else "Median in 10 years",
         "value": money(np.nanmedian(sim.retire_wealth)) if not retired else money(float(sim.pct["p50"].iloc[min(10, len(sim.pct) - 1)])),
         "sub": None if retired else f"10th pct {money(np.nanpercentile(sim.retire_wealth, 10))}", "tone": "info"},
        {"label": "Sustainable spending", "value": money(rec.sustainable),
         "sub": f"{rec.sustainable / max(f.retirement_spending, 1) - 1:+.0%} vs planned, 85% confidence", "tone": "info"},
        {"label": "Median estate", "value": money(float(np.median(life.terminal))), "sub": "at the last death", "tone": "info"},
    ])
    if sim.goal_success:
        st.caption("Goal funding probability: " + " · ".join(f"{g}: {v:.0%}" for g, v in sim.goal_success.items()))
    with st.container(key="card_fan_full"):
        ui.chart(fan(sim, p, rec.sim_overlay, height=470, title=f"Financial wealth to age {PA}"))
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_spend"):
        ret_idx = np.where(sim.ages >= f.retirement_age)[0]
        if len(ret_idx):
            sp = pd.DataFrame({q: np.percentile(sim.spending[:, ret_idx], q, axis=0) for q in (10, 50, 90)}, index=sim.ages[ret_idx])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sp.index, y=sp[90], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=sp.index, y=sp[10], fill="tonexty", fillcolor="rgba(30,77,64,.14)",
                                     line=dict(width=0), name="10th-90th percentile"))
            fig.add_trace(go.Scatter(x=sp.index, y=sp[50], line=dict(color=ui.SPRUCE, width=2.4), name="Median"))
            fig.add_hline(y=f.retirement_spending, line=dict(color=ui.BRASS, dash="dash", width=1.2))
            fig.update_layout(title="Spending actually achieved", height=330, yaxis=dict(tickprefix="$", tickformat="~s"),
                              xaxis_title="Age")
            ui.chart(fig)
            st.caption(f"Brass line: planned spending. Chance of a cut of more than 10% at some point: {sim.cut_prob:.0%}.")
    with c2, st.container(key="card_estate"):
        cap99 = float(np.percentile(life.terminal, 98)) / 1e6
        est = np.clip(life.terminal / 1e6, None, cap99)
        fig = go.Figure(go.Histogram(x=est, nbinsx=40, marker=dict(color="#6B4E71", line=dict(width=0)),
                                     opacity=0.85, hovertemplate="$%{x:.2f}M<br>%{y} scenarios<extra></extra>"))
        fig.add_vline(x=float(np.median(life.terminal)) / 1e6, line=dict(color=ui.INK, dash="dash", width=1))
        fig.update_layout(title="Estate at death (simulated lifetimes)", height=330,
                          xaxis=dict(title="$ millions", tickprefix="$"), yaxis=dict(title="Scenarios"), bargap=0.05)
        ui.chart(fig)
        st.caption(f"Dashed line: median. The top 2% of outcomes are grouped at ${cap99:,.1f}M.")
    if rec.levers is not None:
        ui.section("Plan levers", "Each change is re-profiled and re-simulated. Raising risk is not on the list: the "
                   "client's limits bind. These are the conversations an advisor would have.", eyebrow="What would change the outcome")
        lv = rec.levers
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1, st.container(key="card_levers"):
            vals = list(lv["P(success, plan to age 95)"])
            cols = [ui.verdict_for(v)[0] for v in vals]
            fig = bar_h(list(lv["Lever"]), vals, [ui.TONE[c_] for c_ in cols], height=320)
            fig.update_layout(title="Probability of success")
            ui.chart(fig)
        with c2, st.container(key="card_levers_tbl"):
            ui.table(["Lever", "Success", "At retirement"],
                     [[r_["Lever"], pct(r_["P(success, plan to age 95)"]), money(r_["Median wealth at retirement"])]
                      for _, r_ in lv.iterrows()], num_cols={1, 2})
    with st.expander("Cash-flow schedule (expected path, today's dollars)"):
        st.dataframe(rec.schedule.drop(columns=["ss_client_full", "ss_spouse_full"]).round(0), width="stretch", hide_index=True)

# ------------------------------------------------------------------------ Options & Futures
with tabs[5]:
    el, plan = rec.elig, rec.overlays
    ui.section("Options & futures", "Derivatives are used only to reduce risk or earn income on positions you already "
               "own, and only at the approval level your profile supports. Prices use the live S&P 500, VIX and T-bill rate.",
               eyebrow="Overlays")
    ui.kpis([
        {"label": "Options approval", "value": f"Level {el.options_level}", "sub": "FINRA Rule 2360", "tone": "good" if el.options_level else "bad"},
        {"label": "Futures account", "value": "Approved" if el.futures_ok else "Not approved", "sub": "Hedging only (NFA 2-30)",
         "tone": "good" if el.futures_ok else "bad"},
        {"label": "S&P 500 / VIX", "value": f"{market.spot:,.0f}", "sub": f"VIX {market.vix:.1f} · T-bill {market.rate:.2%}", "tone": "info"},
        {"label": "1-yr implied vol", "value": pct(market.sigma, 1), "sub": "VIX blended with an 18% long-run level", "tone": "info"},
    ])
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1, st.container(key="card_elig"):
        st.html('<div class="eyebrow">Why this approval level</div>')
        for r_ in el.options_reasons + el.futures_reasons:
            st.markdown(f"- {r_}")
    with c2, st.container(key="card_levels"):
        st.html('<div class="eyebrow">House approval ladder</div>')
        ui.table(["Level", "Strategies", "Requirements"], [
            ["0", "None", "-"],
            ["1", "Covered calls, protective puts, collars on shares owned", "Disclosure received, knowledge limited+, objective income+, $50K liquid"],
            ["2", "+ long options, cash-secured puts", "Knowledge good+, 1+ yr options, $100K liquid"],
            ["3", "+ spreads (margin; not in IRAs)", "2+ yrs, objective growth+, $250K, margin agreement"],
            ["4", "+ uncovered writing", "Never recommended by this robo"],
        ])
    ui.section("Recommended overlays", "", eyebrow="For this client")
    if not plan.overlays and not plan.futures and not plan.single_stock_collar:
        ui.notes([plan.not_recommended[0]], kind="No overlay", tone="info")
    ui.notes(plan.rationale, kind="Rationale", tone="info")
    for n_ov, (ov, t0, t1) in enumerate(plan.overlays):
        grid, y = payoff_curve(ov, market.rate)
        end_age = min(h.age + t1, PA)
        with st.container(key=f"card_overlay_{n_ov}"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=grid + 0.012, name="S&P 500 (total return)", line=dict(color="#B9B2A4", dash="dash")))
            fig.add_trace(go.Scatter(x=grid, y=y, name="With overlay", line=dict(color=ui.NAVY, width=3),
                                     hovertemplate="Index %{x:.0%} → overlay %{y:.1%}<extra></extra>"))
            fig.update_layout(title=f"{ov.describe()}", height=330,
                              xaxis=dict(tickformat=".0%", title=f"S&P 500 price return over 1 year · active ages {h.age + t0}-{end_age}"),
                              yaxis=dict(tickformat=".0%", title="Overlay return"))
            ui.chart(fig)
    if rec.sim_overlay is not None:
        base, ovs = rec.sim, rec.sim_overlay
        h_age = min(f.retirement_age + 5 if not retired else h.age + 10, int(base.ages[-1]))
        i = list(base.ages).index(h_age)
        with st.container(key="card_ov_impact"):
            st.html(f'<div class="eyebrow">Monte Carlo impact · wealth at age {h_age}, same random paths</div>')
            qs = (5, 25, 50, 75, 95)
            ui.table(["", "5th pct", "25th", "Median", "75th", "95th", "Success"], [
                ["Core portfolio"] + [money(np.percentile(base.wealth[:, i], q)) for q in qs] + [pct(base.success)],
                ["With overlay"] + [money(np.percentile(ovs.wealth[:, i], q)) for q in qs] + [pct(ovs.success)],
            ], num_cols={1, 2, 3, 4, 5, 6})
            st.caption("Overlays give up some expected return in exchange for a narrower range of outcomes. That trade "
                       "suits the years around retirement and income-focused retirees.")
    extra = []
    if plan.single_stock_collar:
        s_ = plan.single_stock_collar
        extra.append({"label": "Single-stock collar", "value": f"{s_['put_strike']:.0%} / {s_['call_strike']:.0%}",
                      "sub": f"On {money(s_['value'])}: max loss {money(s_['max_loss'])}, max gain {money(s_['max_gain'])}",
                      "tone": "info"})
    if plan.futures:
        fu = plan.futures
        extra.append({"label": "Futures hedge", "value": f"Short {fu['contracts']} MES",
                      "sub": f"{money(fu['hedged_notional'])} notional · defers ~{money(fu['tax_deferred'])} tax", "tone": "info"})
    if extra:
        ui.kpis(extra)
    ui.section("Pricing workbench", "Black-Scholes-Merton with a 1.2% dividend yield, the live T-bill rate and a "
               "volatility skew (downside puts cost more, as in listed S&P options).", eyebrow="Tools")
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

# ------------------------------------------------------------------------ Compliance
with tabs[6]:
    ui.section("Suitability file", "What a reviewer would check before this recommendation goes to the client: the "
               "Regulation Best Interest and fiduciary care obligations, identity and AML, and account protections.",
               eyebrow="Compliance")
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        ui.checklist(rec.checklist)
    with c2:
        with st.container(key="card_notes"):
            st.html('<div class="eyebrow">Account notes</div>')
            for n_ in rec.elig.notes:
                st.markdown(f"- {n_}")
        audit = {"generated": f"{CURRENT_YEAR} (model run)", "profile": p.to_dict(), "risk": rp.summary(),
                 "capacity_parts": rp.capacity_parts, "flags": rp.flags,
                 "allocation": rec.weights_now.round(4).to_dict(), "glide_path": [round(float(x), 3) for x in rec.glide],
                 "options_level": rec.elig.options_level, "futures_ok": rec.elig.futures_ok,
                 "overlays": [o.describe() for o, _, _ in rec.overlays.overlays], "futures": rec.overlays.futures,
                 "success_plan": rec.sim.success, "success_lifetime": rec.sim_life.success,
                 "sustainable_spending_85": rec.sustainable, "market": mkt_dict,
                 "cma": "J.P. Morgan 2026 LTCMA; history 2010-2026", "sim_config": json.loads(cfg_json)}
        st.download_button("Download audit record (JSON)", json.dumps(audit, indent=2, default=float),
                           file_name=f"robo_audit_{choice}.json", mime="application/json", type="primary", width="stretch")
    with st.container(key="card_disclosures"):
        st.html('<div class="eyebrow">Disclosures</div>')
        st.markdown(
            "Projections are hypothetical, do not reflect actual results and are not guarantees. Returns use J.P. Morgan "
            "Asset Management's 2026 Long-Term Capital Market Assumptions (10-15 years, passive, before fees); actual "
            "returns, inflation and taxes will differ. Options involve risk and are not suitable for all investors; read "
            "*Characteristics and Risks of Standardized Options*. Futures trading involves substantial risk of loss. "
            "Buffered ETFs protect only if held for the full outcome period. This tool is an educational project.")

# ------------------------------------------------------------------------ Compare
with tabs[7]:
    ui.section("Three clients, three answers", "The assignment's clients side by side, on their base-case assumptions. "
               "Choose a client in the sidebar to edit their profile.", eyebrow="Case comparison")
    recs = {}
    for k_, fn in CASES.items():
        pk_ = fn()
        recs[k_] = cached_advice(json.dumps(pk_.to_dict()), cfg_json, json.dumps(mkt_dict), json.dumps(prices))
    cards = []
    for k_, r_ in recs.items():
        t_, v_ = ui.verdict_for(r_.sim.success)
        cards.append({"key": f"Case {k_} · {v_}", "title": CLIENTS[k_],
                      "rows": [("Plan succeeds", pct(r_.sim.success)), ("Over lifetimes", pct(r_.sim_life.success)),
                               ("Long-term stocks", pct(r_.glide[0])), ("Risk tier", f"{r_.risk.tier} · {r_.risk.tier_name}"),
                               ("Expected return", pct(r_.stats["exp_return_geo"], 1)),
                               ("2008-style year", pct(r_.stress["2008 global financial crisis"])),
                               ("Sustainable spend", money(r_.sustainable)),
                               ("Options / futures", f"Level {r_.elig.options_level} / {'yes' if r_.elig.futures_ok else 'no'}")],
                      "note": f"Binding limit: {r_.risk.binding.split(' (')[0]}."})
    ui.pillars(cards)
    for k_, r_ in recs.items():
        with st.expander(f"Case {k_}: {r_.profile.label.split(': ', 1)[-1]} · full advice", expanded=(k_ == choice)):
            ui.actions([a for a in r_.advice if not a.startswith("**Flag")])
            ui.notes([ui.split_title(a)[1] for a in r_.advice if a.startswith("**Flag")])

# ------------------------------------------------------------------------ Methodology
with tabs[8]:
    summ, models, front, stress_tbl = cached_methodology()
    ui.section("How the robo decides", "The same sequence a bank advisor follows, made explicit and repeatable. The full "
               "derivation and literature review are in the accompanying research paper.", eyebrow="Methodology")
    ui.steps([
        ("Know the client", "Identity and AML checks, then the FINRA 2111 / Reg BI investment profile."),
        ("Measure risk three ways", "Willingness (gamma), ability (balance sheet, stress test, capacity) and need."),
        ("Bucket the goals", "Each goal gets its own horizon-matched mix; retirement gets a household glide path."),
        ("Build the models", "Resampled mean-variance on J.P. Morgan 2026 assumptions under investment-policy limits."),
        ("Gate the derivatives", "Options level 0-4 and futures eligibility decide which overlays are allowed."),
        ("Stress the plan", "Monte Carlo with fat tails, taxes, RMDs, Social Security and stochastic lifetimes."),
    ])
    ui.section("Capital market assumptions", "Expected returns are J.P. Morgan's 10-15 year forecasts. Volatility comes "
               "from their published chart where available, otherwise from ETF history.", eyebrow="Inputs")
    with st.container(key="card_cma"):
        st.dataframe(CMA.table(), width="stretch", hide_index=True, column_config={
            "Exp. return (geo)": st.column_config.NumberColumn(format="percent"),
            "Exp. return (arith)": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent")})
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(key="card_corr"):
        fig = go.Figure(go.Heatmap(z=CMA.corr, x=list(CMA.keys), y=list(CMA.keys), zmin=-1, zmax=1,
                                   colorscale=[[0, ui.CLARET], [0.5, "#F6F2E9"], [1, ui.SPRUCE]],
                                   text=np.round(CMA.corr, 2), texttemplate="%{text}", textfont=dict(size=8),
                                   colorbar=dict(thickness=10, outlinewidth=0)))
        fig.update_layout(title=f"Correlations, {CMA.meta['hist_start'][:4]}-{CMA.meta['hist_end'][:4]}", height=520,
                          yaxis=dict(autorange="reversed"))
        ui.chart(fig)
    with c2, st.container(key="card_frontier"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=front["vol"], y=front["exp_return_arith"], name="Efficient frontier",
                                 line=dict(color="#B9B2A4", width=2)))
        fig.add_trace(go.Scatter(x=CMA.vol, y=CMA.mu_arith, mode="markers+text", text=list(CMA.keys), name="Asset classes",
                                 textposition="bottom center", marker=dict(size=7, color=ui.BRASS), textfont=dict(size=9, color=ui.MUTED)))
        fig.add_trace(go.Scatter(x=summ["vol"], y=summ["exp_return_arith"], mode="lines+markers+text", name="Model tiers 0-10",
                                 text=[str(t) for t in summ.index], textposition="top center",
                                 marker=dict(size=9, color=ui.NAVY), line=dict(color=ui.NAVY, width=1.5)))
        fig.update_layout(title="Assets, frontier and the 11 models", height=520,
                          xaxis=dict(tickformat=".0%", title="Volatility"), yaxis=dict(tickformat=".1%", title="Expected return"))
        ui.chart(fig)
    ui.section("Model portfolios", "Resampled mean-variance optimisation, policy-constrained, one model per 10% of equity.",
               eyebrow="Models")
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
    ui.section("Longevity", "Gompertz-Makeham mortality fitted exactly to SSA 2022 life expectancies at ages 30, 55 and 70.",
               eyebrow="Life tables")
    with st.container(key="card_longevity"):
        fig = go.Figure()
        for sx, col in (("female", ui.BRASS), ("male", ui.NAVY)):
            for a0 in (30, 55, 70):
                s_ = survival_table(sx, a0)
                fig.add_trace(go.Scatter(x=s_["ages"], y=s_["survival"], name=f"{sx.capitalize()} {a0} (LE {s_['life_expectancy']:.1f})",
                                         line=dict(color=col, dash={30: "dot", 55: "dash", 70: "solid"}[a0], width=2)))
        fig.update_layout(height=380, yaxis=dict(tickformat=".0%", title="Probability alive"), xaxis_title="Age",
                          legend=dict(orientation="h", y=-0.22))
        ui.chart(fig)
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
            ["Monte Carlo", "Running the plan through thousands of randomly generated market histories."],
        ])
