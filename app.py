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
from robo import aml, sanctions
from robo.advisor import LIMITATIONS, Recommendation, advise
from robo.audit import RESULTS as AUDIT_RESULTS
from robo.audit import build_log
from robo.audit import summary as log_summary
from robo.audit import to_csv as log_csv
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
                   "C": "Age 70 · $2M · married", "Custom": "Blank application and questionnaire"}
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


@st.cache_resource(ttl=6 * 3600, show_spinner=False)
def get_sdn(use_live: bool):
    """OFAC SDN list with aliases and a token index; the built-in synthetic sample when offline."""
    return sanctions.load_list(use_live)


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
CIP_NOTICE = ("**Important information about procedures for opening a new account.** To help the government fight the funding "
              "of terrorism and money laundering activities, Federal law requires all financial institutions to obtain, verify, "
              "and record information that identifies each person who opens an account. What this means for you: when you open "
              "an account, we will ask for your name, address, date of birth and other information that will allow us to "
              "identify you. We may also ask to see your driver's license or other identifying documents.")
SAMPLE_SSN = "987-65-43"       # + two digits: the 987-65-432x range is reserved for advertising and never issued
STATUS_SUMMARY = {
    "cleared": "Every control passed and a principal accepted the account. It is open for funding and trading, and the "
               "recommendation below can be implemented.",
    "hold": "The application is in good order, but a firm review is still open. The account is restricted (no trading, no "
            "withdrawals) until the items below are cleared on the firm-review panel. The plan is shown; overlays wait.",
    "incomplete": "Some required items are missing, so nothing has been opened. The plan below is a **proposal only**, the "
                  "way a prospect sees one before applying. Complete the steps marked with a number.",
    "blocked": "A hard stop was hit. The account cannot be opened and no recommendation is implemented; the file goes to "
               "the AML officer.",
}


def _sample_identity() -> Identity:
    """A fictional applicant with a complete application; the firm review is left undone."""
    t = date.today()
    return Identity(full_name="Jordan A. Rivera", enforce_intake=True, date_of_birth=date(t.year - 38, 5, 9).isoformat(),
                    tin_type="SSN", tin_masked="XXX-XX-4325", tin_valid=True, address_line="88 Beacon Street, Unit 4",
                    address_city="Boston", state="MA", address_zip="02108", phone="(617) 555-0134",
                    email="jordan.rivera@example.com", id_doc_type="Driver's license", id_doc_masked="XXXX1342",
                    id_doc_issuer="MA", id_doc_expiry=date(t.year + 4, 5, 9).isoformat(), occupation="Product manager",
                    employer="Harborview Software", expected_initial_funding=60_000, expected_annual_deposits=18_000,
                    trusted_contact="Elena Rivera (sister)", trusted_contact_info="(617) 555-0190", tax_cert_signed=True,
                    cip_notice_ack=True, form_crs_acknowledged=True, privacy_notice_ack=True, esign_consent=True,
                    agreement_signed=True, signature_name="Jordan A. Rivera", id_verification=aml.ID_VERIFY[2])


def _default_profile(k: str) -> ClientProfile:
    if k in CASES:
        return CASES[k]()
    ident = _sample_identity() if st.session_state.get("custom_sample") else Identity(
        full_name="", enforce_intake=True, id_verification=aml.ID_VERIFY[2], form_crs_acknowledged=False)
    return ClientProfile(label="New client", identity=ident, household=Household(age=40, sex="female"),
                         finances=Finances(taxable=50_000, traditional=100_000, roth=20_000, cash=20_000),
                         goals=[], experience=Experience())


def _set_sample(on: bool) -> None:
    """Button callback: load or clear the sample applicant. Bumping the version gives the application
    widgets new keys, so they are rebuilt from the new defaults."""
    st.session_state["custom_sample"] = on
    st.session_state["custom_ver"] = st.session_state.get("custom_ver", 0) + 1


def _idx(options, value):
    return options.index(value) if value in options else 0


def _num(v, default):
    try:
        x = float(v)
        return default if np.isnan(x) else x
    except (TypeError, ValueError):
        return default


def _date(iso: str):
    return aml.parse_date(iso)


def _sample_raw(masked: str) -> str:
    """Pre-fill for the sample clients: a number from the reserved advertising range ending in the stored last four."""
    d = aml.digits(masked)
    return f"{SAMPLE_SSN}{d[-2:]}" if len(d) >= 2 else ""


@st.cache_data(show_spinner=False, max_entries=256)
def screen_parties(parties: tuple, list_stamp: str, _sl) -> list[dict]:
    """Sanctions hits for every party to the account, as plain dicts (cached per name set and list version)."""
    out = []
    for role, name, dob, is_entity in parties:
        for h in sanctions.screen(_sl, name, dob, party=role, entity=is_entity):
            d = dataclasses.asdict(h)
            d["summary"] = h.describe()
            out.append(d)
    return out


def kyc_form(k: str, sdn) -> ClientProfile:
    b = _default_profile(k)
    i, h, f, x = b.identity, b.household, b.finances, b.experience
    K = lambda s: f"{k}_{s}"
    ver = st.session_state.get("custom_ver", 0) if k == "Custom" else 0
    KA = lambda s: f"{k}{ver}_{s}"            # application and firm-review widgets (rebuilt when the sample is loaded)
    HEALTH = ["excellent", "good", "average", "fair", "poor"]
    MARITAL = ["single", "married", "divorced", "widowed"]
    blank = k == "Custom"
    today = date.today()

    # ------------------------------------------------------------------ A. account application
    ui.part("A", "Account application", "What the firm must collect, by law, before it can open an account. "
            "Four short steps; the status panel underneath shows what is still outstanding.")
    if blank:
        c = st.columns([1.25, 1.05, 3])
        c[0].button("Fill with a sample applicant", on_click=_set_sample, args=(True,), key="custom_fill", width="stretch",
                    help="Fills the four application steps with a fictional person so you can see the full workflow. "
                         "The firm-review panel is left for you.")
        c[1].button("Clear the application", on_click=_set_sample, args=(False,), key="custom_clear", width="stretch")
        c[2].caption("New client: the application starts empty. Until it is complete the plan below is shown as a proposal, "
                     "not an open account.")
    sa = st.tabs(["01  Applicant", "02  Account & funding", "03  Regulatory questions", "04  Agreements & signature"])

    with sa[0], st.container(key="card_kyc_applicant"):
        st.caption("Why we ask: the Customer Identification Program (31 CFR 1023.220) requires your legal name, date of birth, "
                   "home address and an identification number before an account opens, and that we verify them.")
        ui.group("Legal name and date of birth", first=True)
        parts = i.full_name.split()
        c = st.columns([1.2, 0.7, 1.2, 1])
        first = c[0].text_input("Legal first name", parts[0] if parts else "", key=KA("first"))
        middle = c[1].text_input("Middle", " ".join(parts[1:-1]) if len(parts) > 2 else "", key=KA("middle"))
        last = c[2].text_input("Legal last name", parts[-1] if len(parts) > 1 else "", key=KA("last"))
        dob = c[3].date_input("Date of birth", _date(i.date_of_birth), min_value=date(1900, 1, 1), max_value=today,
                              format="MM/DD/YYYY", key=KA("dob"))
        full_name = " ".join(s.strip() for s in (first, middle, last) if s.strip())

        ui.group("Citizenship and identification number")
        c = st.columns(4)
        citizenship = c[0].selectbox("Citizenship status", aml.CITIZENSHIP, index=_idx(aml.CITIZENSHIP, i.citizenship), key=KA("cit"))
        us_cit = citizenship == "US citizen"
        ctry_cit = c[1].selectbox("Country of citizenship", aml.COUNTRIES, index=_idx(aml.COUNTRIES, i.country_of_citizenship),
                                  key=KA("citctry"), disabled=us_cit)
        ctry_cit = aml.US if us_cit else ctry_cit
        tin_opts = aml.TIN_TYPES if citizenship == "Non-resident alien" else [t for t in aml.TIN_TYPES if not t.startswith("Foreign")]
        tin_type = c[2].selectbox("Identification number type", tin_opts, index=_idx(tin_opts, i.tin_type), key=KA("tintype"))
        tin_raw = c[3].text_input("Number", _sample_raw(i.tin_masked), type="password", key=KA("tin"),
                                  disabled=tin_type.startswith("Applied"), placeholder="9 digits",
                                  help="Demo: never type a real Social Security number. Use the sample 987-65-4321 (a range "
                                       "reserved for advertising). Only the last four digits are kept.")
        tin_valid, tin_why = aml.validate_tin(tin_raw, tin_type)
        tin_masked = aml.mask_tin(tin_raw, tin_type)
        if tin_type.startswith("Applied"):
            c[3].caption(":orange[Applied for: we need a copy of the application and the number within 60 days.]")
        elif tin_raw:
            c[3].caption(f":green[Valid format · only the last four digits ({tin_masked[-4:]}) are kept]" if tin_valid
                         else f":red[Not valid: {tin_why}.]")

        ui.group("Home address and contact")
        c = st.columns([2.2, 1.3, 0.8, 0.9, 1.4])
        street = c[0].text_input("Street address (not a P.O. box)", i.address_line, key=KA("street"))
        city = c[1].text_input("City", i.address_city, key=KA("city"))
        country = c[4].selectbox("Country", aml.COUNTRIES, index=_idx(aml.COUNTRIES, i.address_country), key=KA("country"))
        state = c[2].selectbox("State", aml.US_STATES, index=_idx(aml.US_STATES, i.state), key=KA("state"), disabled=country != aml.US)
        zip_ = c[3].text_input("ZIP code", i.address_zip, key=KA("zip"))
        c = st.columns([1, 1.4, 2])
        phone = c[0].text_input("Mobile phone", i.phone, key=KA("phone"))
        email = c[1].text_input("Email", i.email, key=KA("email"))
        mailing = c[2].text_input("Mailing address, if different", i.mailing_address, key=KA("mailing"))
        if street and aml.is_po_box(street):
            st.caption(":red[A P.O. box cannot be the home address. Put it under mailing address and give a street address here.]")

        ui.group("Government-issued photo ID")
        c = st.columns(4)
        id_type = c[0].selectbox("Document", aml.ID_DOC_TYPES, index=_idx(aml.ID_DOC_TYPES, i.id_doc_type), key=KA("idtype"))
        id_raw = c[1].text_input("Document number", f"D000{i.id_doc_masked[-4:]}" if i.id_doc_masked else "", key=KA("idnum"),
                                 help="Only the last four characters are kept.")
        id_issuer = c[2].text_input("Issuing state or country", i.id_doc_issuer, key=KA("idissuer"))
        id_exp = c[3].date_input("Expiration date", _date(i.id_doc_expiry), min_value=date(2000, 1, 1),
                                 max_value=date(today.year + 20, 12, 31), format="MM/DD/YYYY", key=KA("idexp"))
        if id_exp and id_exp < today:
            c[3].caption(":red[This ID has expired. An unexpired document is required.]")

        ui.group("Employment")
        c = st.columns(4)
        emp_opts = ["employed", "self-employed", "retired", "not employed"]
        employment = c[0].selectbox("Employment status", emp_opts, index=_idx(emp_opts, i.employment_status), key=KA("emp"))
        occupation = c[1].text_input("Occupation", i.occupation, key=KA("occ"))
        employer = c[2].text_input("Employer", i.employer, key=KA("employer"))
        industry = c[3].selectbox("Is your work or business in any of these?", aml.INDUSTRIES, index=_idx(aml.INDUSTRIES, i.industry),
                                  key=KA("industry"), help="Industries that handle a lot of cash or move money for others get closer review.")

    with sa[1], st.container(key="card_kyc_account"):
        st.caption("Why we ask: the customer due diligence rule (31 CFR 1010.230, 1023.210) requires us to understand why you "
                   "are opening the account, where the money comes from and what activity to expect, and to identify every owner.")
        ui.group("The account", first=True)
        c = st.columns(3)
        acct = c[0].selectbox("Account registration", aml.ACCOUNT_TYPES, index=_idx(aml.ACCOUNT_TYPES, i.account_type),
                              format_func=aml.ACCOUNT_LABEL.get, key=KA("acct"))
        purpose = c[1].selectbox("Purpose of the account", aml.PURPOSES, index=_idx(aml.PURPOSES, i.account_purpose), key=KA("purpose"))
        ira_ben = i.ira_beneficiary
        if acct == "IRA only":
            ira_ben = c[2].text_input("Primary beneficiary", i.ira_beneficiary, key=KA("iraben"),
                                      help="You may revoke a new IRA within 7 days of receiving the disclosure statement.")

        joint, entity, bos, bo_ok = dict(i.joint_owner), dict(i.entity), list(i.beneficial_owners), i.beneficial_owners_verified
        if acct == "joint":
            ui.group("Joint owner · each owner is identified separately")
            j = i.joint_owner
            c = st.columns([1.6, 1, 1, 1.6])
            j_name = c[0].text_input("Joint owner's legal name", j.get("name", ""), key=KA("jname"))
            j_dob = c[1].date_input("Date of birth", _date(j.get("dob", "")), min_value=date(1900, 1, 1), max_value=today,
                                    format="MM/DD/YYYY", key=KA("jdob"))
            j_tin = c[2].text_input("SSN or ITIN", _sample_raw(j.get("tin_masked", "")), type="password", key=KA("jtin"),
                                    help="Demo: use a sample such as 987-65-4322.")
            j_ten = c[3].selectbox("Form of ownership", aml.JOINT_TENANCY, index=_idx(aml.JOINT_TENANCY, j.get("tenancy", "")), key=KA("jten"))
            c = st.columns([1, 3])
            j_same = c[0].checkbox("Lives at the same address", j.get("address_same", True), key=KA("jsame"))
            j_addr = "" if j_same else c[1].text_input("Joint owner's home address", j.get("address", ""), key=KA("jaddr"))
            j_kind = "ITIN" if aml.digits(j_tin).startswith("9") and not aml.digits(j_tin).startswith(aml.TEST_SSN_PREFIX) else "SSN"
            joint = {"name": j_name.strip(), "dob": j_dob.isoformat() if j_dob else "", "tin_masked": aml.mask_tin(j_tin),
                     "tin_valid": aml.validate_tin(j_tin, j_kind)[0], "tenancy": j_ten, "address_same": j_same, "address": j_addr}
        elif acct in aml.ENTITY_TYPES:
            is_trust = acct == "trust"
            e = i.entity
            ui.group("The trust · you act as trustee" if is_trust else "The entity · you act as its authorized person")
            c = st.columns([1.8, 1, 1, 1.2])
            e_name = c[0].text_input("Name of the trust" if is_trust else "Legal name of the entity", e.get("legal_name", ""), key=KA("ename"))
            kinds = ["Revocable trust", "Irrevocable trust"] if is_trust else aml.ENTITY_KINDS
            e_kind = c[1].selectbox("Type", kinds, index=_idx(kinds, e.get("kind", "")), key=KA("ekind"))
            e_ein = c[2].text_input("Taxpayer ID (EIN)", f"00000{e['ein_masked'][-4:]}" if e.get("ein_masked") else "", key=KA("eein"), placeholder="9 digits",
                                    help="Demo: any 9 digits. A revocable trust may use the grantor's SSN.")
            e_formed = c[3].text_input("Governing state" if is_trust else "State or country of formation", e.get("formed_in", ""), key=KA("eformed"))
            docs = (["", "Certification of trust", "Trust agreement (title, trustee and signature pages)"] if is_trust else
                    ["", "Articles / certificate of formation and resolution", "Partnership agreement", "Business license"])
            c = st.columns([1.6, 1.6, 1.2])
            e_docs = c[0].selectbox("Document provided", docs, index=_idx(docs, e.get("documents", "")), key=KA("edocs"),
                                    format_func=lambda s: s or "None yet")
            e_biz = c[1].text_input("Purpose of the trust" if is_trust else "Nature of the business", e.get("business", ""), key=KA("ebiz"))
            e_exempt = aml.BO_EXEMPT[0]
            if not is_trust:
                e_exempt = c[2].selectbox("Excluded from beneficial-owner rule?", aml.BO_EXEMPT, index=_idx(aml.BO_EXEMPT, e.get("exempt", "")),
                                          key=KA("eexempt"), help="Listed companies, regulated financial institutions and a few others "
                                                                 "are excluded (31 CFR 1010.230(e)(2)).")
            entity = {"legal_name": e_name.strip(), "kind": e_kind, "ein_masked": aml.mask_doc(e_ein),
                      "ein_valid": len(aml.digits(e_ein)) == 9, "formed_in": e_formed.strip(), "documents": e_docs,
                      "business": e_biz, "exempt": e_exempt, "certified": False}
            if not is_trust and e_exempt == aml.BO_EXEMPT[0]:
                ui.group("Beneficial owners · everyone who owns 25% or more, plus one person who controls the entity")
                bo_df = pd.DataFrame(i.beneficial_owners or [{"name": "", "role": "Control person", "pct": None, "dob": "", "address": "", "id_last4": ""}],
                                     columns=["name", "role", "pct", "dob", "address", "id_last4"])
                bo_ed = st.data_editor(bo_df, num_rows="dynamic", width="stretch", key=KA("bos"), column_config={
                    "name": st.column_config.TextColumn("Full legal name", required=True),
                    "role": st.column_config.SelectboxColumn("Role", options=["Owner (25% or more)", "Control person", "Owner and control person"], required=True),
                    "pct": st.column_config.NumberColumn("Ownership %", min_value=0, max_value=100, step=1),
                    "dob": st.column_config.TextColumn("Date of birth (YYYY-MM-DD)", validate=r"^\d{4}-\d{2}-\d{2}$"),
                    "address": st.column_config.TextColumn("Home address"),
                    "id_last4": st.column_config.TextColumn("SSN / passport, last 4", validate=r"^[A-Za-z0-9]{4}$")})
                bos = [{kk: ("" if (isinstance(v, float) and np.isnan(v)) or v is None else v) for kk, v in r.items()}
                       for r in bo_ed.to_dict("records") if str(r.get("name") or "").strip() not in ("", "nan")]
                entity["certified"] = st.checkbox("I certify, to the best of my knowledge, that the beneficial-ownership information "
                                                  "above is complete and correct.", e.get("certified", False), key=KA("ecert"))
            else:
                bos = []

        ui.group("Where the money comes from")
        c = st.columns([1.05, 1.75, 0.9, 0.9])
        sow = c[0].selectbox("Source of wealth", aml.SOURCE_OF_WEALTH, index=_idx(aml.SOURCE_OF_WEALTH, i.source_of_wealth), key=KA("sow"),
                             help="How you built your assets over time.")
        fund_opts = list(aml.FUNDING)
        sof = c[1].selectbox("How will you fund the account?", fund_opts, index=_idx(fund_opts, i.source_of_funds), key=KA("sof"),
                             help="Where this deposit comes from. We do not accept cash; third-party payments need review.")
        funding = c[2].number_input("Initial deposit ($)", 0, 1_000_000_000, int(i.expected_initial_funding), 5_000, key=KA("fund"))
        annual_dep = c[3].number_input("Further deposits ($/yr)", 0, 1_000_000_000, int(i.expected_annual_deposits), 1_000, key=KA("annual"))
        c = st.columns([1.55, 1.6, 1.4], vertical_alignment="bottom")
        wd = c[0].selectbox("How often will you withdraw?", aml.WITHDRAWALS, index=_idx(aml.WITHDRAWALS, i.expected_withdrawals), key=KA("wd"))
        wires = c[1].checkbox("I expect to send or receive international wires", i.foreign_wires_expected, key=KA("wires"))
        wire_ctry = c[2].multiselect("Countries involved", [x_ for x_ in aml.COUNTRIES if x_ != aml.US], default=list(i.wire_countries),
                                     key=KA("wirectry")) if wires else []

    with sa[2], st.container(key="card_kyc_regulatory"):
        st.caption("Why we ask: tax law requires a certified taxpayer status; securities rules require us to ask about industry "
                   "and public-company affiliations, political exposure and a trusted contact.")
        ui.group("Tax status", first=True)
        c = st.columns([1, 1, 1.4], vertical_alignment="bottom")
        tax_res = c[0].selectbox("Tax residency", aml.TAX_RESIDENCY, index=_idx(aml.TAX_RESIDENCY, i.tax_residency), key=KA("res"))
        other_tax = ""
        if tax_res != "US only":
            foreign = [x_ for x_ in aml.COUNTRIES if x_ != aml.US]
            other_tax = c[1].selectbox("Other country of tax residence", foreign, index=_idx(foreign, i.other_tax_country), key=KA("taxctry"))
        backup = c[2].checkbox("The IRS has notified me that I am subject to backup withholding", i.backup_withholding, key=KA("backup"))
        if citizenship == "Non-resident alien":
            cert_text = ("Form W-8BEN: under penalties of perjury, I certify that I am the beneficial owner of the income, that I am "
                         "not a U.S. person, and that the information I have given is true and complete.")
        else:
            cert_text = ("Form W-9: under penalties of perjury, I certify that the number shown is my correct taxpayer identification "
                         "number, that I am a U.S. citizen or other U.S. person, and that I am "
                         + ("subject" if backup else "not subject") + " to backup withholding.")
        tax_cert = st.checkbox(cert_text, i.tax_cert_signed, key=KA("taxcert"))

        ui.group("Affiliations")
        bd = st.checkbox("I, my spouse or a dependent child work for a broker-dealer, a securities exchange or FINRA.",
                         i.employed_by_broker_dealer, key=KA("bd"))
        bd_firm = st.text_input("Name of the firm", i.bd_firm, key=KA("bdfirm")) if bd else ""
        ctrl = st.checkbox("I am a director, officer or 10% (or more) shareholder of a publicly traded company.",
                           i.control_person_public_co, key=KA("ctrl"))
        ins_tk, plan105, hedge_ok = i.insider_ticker, i.has_10b5_1_plan, i.issuer_permits_hedging
        if ctrl:
            c = st.columns(3)
            ins_tk = c[0].text_input("Company ticker", i.insider_ticker, key=KA("instk"))
            plan105 = c[1].checkbox("A Rule 10b5-1 trading plan is in place", i.has_10b5_1_plan, key=KA("105"))
            hedge_ok = c[2].checkbox("The company's policy permits hedging", i.issuer_permits_hedging, key=KA("hedgeok"))
        c = st.columns([1.6, 2])
        default_pep = i.pep_type if i.pep_type != aml.PEP_TYPES[0] else (aml.PEP_TYPES[1] if i.politically_exposed else aml.PEP_TYPES[0])
        pep_type = c[0].selectbox("Are you, a family member or a close associate a senior political or government official?",
                                  aml.PEP_TYPES, index=_idx(aml.PEP_TYPES, default_pep), key=KA("pep"))
        pep = pep_type != aml.PEP_TYPES[0]
        pep_detail = c[1].text_input("Office held, country and your relationship", i.pep_detail, key=KA("pepd")) if pep else ""

        ui.group("Trusted contact")
        st.caption("Someone 18 or older we may contact if we suspect financial exploitation, cannot reach you, or need to confirm "
                   "your health status or who holds a power of attorney. They cannot trade or see your balances (FINRA Rule 4512).")
        c = st.columns([1.5, 1.2, 1.3])
        trusted = c[0].text_input("Name and relationship", i.trusted_contact, key=KA("tc"))
        tc_info = c[1].text_input("Their phone or email", i.trusted_contact_info, key=KA("tcinfo"))
        poa = c[2].text_input("Power of attorney on file, if any", i.power_of_attorney, key=KA("poa"))
        tc_declined = st.checkbox("I decline to name a trusted contact", i.trusted_contact_declined, key=KA("tcd"))

    with sa[3], st.container(key="card_kyc_sign"):
        st.caption("Read each document, tick to acknowledge it, then sign. Nothing is opened until all five are ticked and "
                   "the signature matches the legal name on step 01.")
        ui.notes([CIP_NOTICE], kind="Notice", tone="info")
        ack_cip = st.checkbox("I have read the customer identification notice above.", i.cip_notice_ack, key=KA("ack_cip"))
        crs = st.checkbox("I received **Form CRS** (relationship summary) and **Form ADV Part 2A** (firm brochure): services, fees "
                          "and conflicts of interest.", i.form_crs_acknowledged, key=KA("crs"))
        ack_priv = st.checkbox("I received the **privacy notice** describing how my information is collected, shared and protected "
                               "(Regulation S-P).", i.privacy_notice_ack, key=KA("ack_priv"))
        ack_esign = st.checkbox("I **consent to electronic delivery** of statements, confirmations, tax forms and disclosures. I can "
                                "ask for paper copies or withdraw this consent at any time (E-SIGN Act).", i.esign_consent, key=KA("ack_esign"))
        ack_agree = st.checkbox("I agree to the **Advisory Agreement** and the **Account Agreement**, which contains a **pre-dispute "
                                "arbitration clause**: disputes go to FINRA arbitration, not to court.", i.agreement_signed, key=KA("ack_agree"))
        st.caption("Also delivered at account opening: SIPC information (FINRA Rule 2266), BrokerCheck (Rule 2267), business "
                   "continuity plan (Rule 4370), order-routing disclosure (Regulation NMS Rule 607), fee schedule"
                   + (", IRA disclosure statement with the 7-day right to revoke" if acct == "IRA only" else "") + ".")
        c = st.columns([2, 1, 1.4], vertical_alignment="bottom")
        signature = c[0].text_input("Type your full legal name to sign", i.signature_name, key=KA("sig"), placeholder=full_name or "Legal name")
        pdate = c[1].date_input("Date signed", _date(i.profile_date) or today, max_value=today, format="MM/DD/YYYY", key=KA("pdate"),
                                help="The account record must be re-confirmed at least every 36 months (SEA Rule 17a-3).")
        if signature and not aml.same_name(signature, full_name):
            c[2].caption(":red[The signature does not match the legal name on step 01.]")
        elif signature:
            c[2].caption(":green[Signed electronically.]")

    # ------------------------------------------------------------------ B. firm review
    party_list = tuple(aml.parties(Identity(full_name=full_name, date_of_birth=dob.isoformat() if dob else "",
                                           account_type=acct, joint_owner=joint, entity=entity, beneficial_owners=bos)))
    hits = screen_parties(party_list, sdn.asof, sdn)
    ui.part("B", "Firm review", "Filled in by the firm, never by the applicant: the verification vendor's result, list "
            "screening, due-diligence approvals and a principal's acceptance.")
    n_hits = len(hits)
    with st.expander(f"Compliance desk · {'no' if not n_hits else n_hits} potential sanctions match{'es' if n_hits != 1 else ''}"
                     f" · {sdn.count:,} listed parties screened", expanded=bool(n_hits) or i.politically_exposed):
        st.caption("In production these fields are written by systems and analysts, not typed here. They are editable so you can "
                   "see how each result moves the account between approved, pending and declined.")
        ui.group("Identity verification (CIP)", first=True)
        c = st.columns([1.2, 2, 1], vertical_alignment="bottom")
        idv = c[0].selectbox("Vendor result", aml.ID_VERIFY, index=_idx(aml.ID_VERIFY, i.id_verification), key=KA("idv"),
                             help="Documentary: unexpired government photo ID. Non-documentary: credit-bureau or public-database match.")
        flags = c[1].multiselect("Identity-theft red flags raised (Regulation S-ID)", aml.ID_THEFT_FLAGS, default=list(i.id_theft_flags), key=KA("flags"))
        flags_ok = c[2].checkbox("Red flags resolved", i.id_theft_resolved, key=KA("flagsok"), disabled=not flags)
        ui.group("Sanctions and list screening")
        if n_hits:
            ui.notes([hh["summary"] for hh in hits], kind="Potential match", tone="bad")
        else:
            st.caption(f":green[No potential match] for {len(party_list)} name(s) screened. List: {sdn.source}, {sdn.asof}."
                       + (" Try the name *Darius Embargo* to see the potential-match workflow." if blank else ""))
        c = st.columns([1.3, 1.3, 1.4], vertical_alignment="bottom")
        disp = c[0].selectbox("Analyst disposition of the match", aml.OFAC_DISPOSITIONS, index=_idx(aml.OFAC_DISPOSITIONS, i.ofac_disposition),
                              key=KA("ofacdisp"), disabled=not n_hits,
                              help="Compare date of birth, nationality and address. A similar name alone is not a match.")
        media = c[1].selectbox("Adverse-media search", aml.ADVERSE_MEDIA, index=_idx(aml.ADVERSE_MEDIA, i.adverse_media), key=KA("media"))
        m314 = c[2].checkbox("Positive match to a FinCEN 314(a) request", i.fincen_314a_match, key=KA("m314"))
        ui.group("Due diligence and approvals")
        c = st.columns(3)
        sow_doc = c[0].checkbox("Source of wealth / funds documents received", i.sow_documented, key=KA("sowdoc"))
        edd = c[1].checkbox("Enhanced due diligence approved", i.edd_cleared, key=KA("edd"))
        if acct in aml.ENTITY_TYPES:
            bo_ok = c[2].checkbox("Trustees verified" if acct == "trust" else "Beneficial owners verified", i.beneficial_owners_verified, key=KA("bo"))
        c = st.columns(3)
        edd_by = c[0].text_input("EDD approved by", i.edd_approver, key=KA("eddby"), disabled=not edd)
        analyst = c[1].text_input("Analyst", i.analyst or "K. Osei (AML analyst)", key=KA("analyst"))
        principal_name = c[2].text_input("Principal", i.principal_name or "D. Ramirez (Series 24)", key=KA("principal"))
        principal_ok = st.checkbox("A principal has reviewed the file and accepts the account (FINRA Rule 4512)", i.principal_approved, key=KA("accept"))
    status_slot = st.container()

    # ------------------------------------------------------------------ C. investment profile
    ui.part("C", "Investment profile", "What the recommendation is built from: household, balance sheet, retirement, goals, "
            "experience and attitude to risk. Change anything and every section below updates.")
    steps = st.tabs(["05  Household", "06  Balance sheet", "07  Retirement & tax", "08  Goals",
                     "09  Experience & approvals", "10  Risk questionnaire"])

    with steps[0], st.container(key="card_kyc_household"):
        st.caption("Why we ask: age, health and a spouse's details set the planning horizon, life tables, survivor "
                   "benefits and the age required withdrawals begin. Insurance in force changes what the portfolio must cover.")
        c = st.columns(5)
        dob_age = aml.age_on(dob.isoformat()) if dob else None
        if dob_age is not None:
            st.session_state[K("age")] = int(min(max(dob_age, 18), 100))
        age_help = "Taken from the date of birth on step 01 once it is entered."
        age = (c[0].number_input("Age", 18, 100, key=K("age"), disabled=True, help=age_help) if dob_age is not None
               else c[0].number_input("Age", 18, 100, h.age, key=K("age"), help=age_help))
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

    with steps[1], st.container(key="card_kyc_money"):
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

    with steps[2], st.container(key="card_kyc_retire"):
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

    with steps[3], st.container(key="card_kyc_goals"):
        st.caption("Each goal becomes its own bucket. **Needs** are funded first and with a safety margin; **wishes**, "
                   "such as a legacy, are not counted when measuring how much return the plan requires. Retirement comes "
                   "from step 07. Goals dated beyond the planning horizon are moved to its last year and flagged.")
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

    with steps[4], st.container(key="card_kyc_exp"):
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

    with steps[5], st.container(key="card_kyc_risk"):
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
    id_method = {aml.ID_VERIFY[0]: "documentary", aml.ID_VERIFY[1]: "non-documentary", aml.ID_VERIFY[2]: "pending"}.get(idv, "documentary")
    profile = ClientProfile(
        label=b.label,
        identity=Identity(
            full_name=full_name, citizenship=citizenship, tax_residency=tax_res, state=state, account_type=acct,
            beneficial_owners_verified=bo_ok, employment_status=employment, occupation=occupation, employer=employer,
            source_of_wealth=sow, expected_initial_funding=float(funding), employed_by_broker_dealer=bd,
            control_person_public_co=ctrl, insider_ticker=ins_tk.upper().strip(), has_10b5_1_plan=plan105,
            issuer_permits_hedging=hedge_ok, politically_exposed=pep, pep_detail=pep_detail, edd_cleared=edd,
            identity_verified=idv != aml.ID_VERIFY[3], id_method=id_method, ofac_clear=disp != aml.OFAC_DISPOSITIONS[2],
            trusted_contact=trusted, trusted_contact_declined=tc_declined, power_of_attorney=poa,
            profile_date=pdate.isoformat(), form_crs_acknowledged=crs,
            enforce_intake=True, date_of_birth=dob.isoformat() if dob else "", tin_type=tin_type, tin_masked=tin_masked,
            tin_valid=tin_valid, address_line=street, address_city=city, address_zip=zip_, address_country=country,
            mailing_address=mailing, phone=phone, email=email, country_of_citizenship=ctry_cit, other_tax_country=other_tax,
            id_doc_type=id_type, id_doc_masked=aml.mask_doc(id_raw), id_doc_issuer=id_issuer,
            id_doc_expiry=id_exp.isoformat() if id_exp else "", joint_owner=joint if acct == "joint" else {},
            entity=entity if acct in aml.ENTITY_TYPES else {}, beneficial_owners=bos if acct in aml.ENTITY_TYPES else [],
            ira_beneficiary=ira_ben if acct == "IRA only" else "", account_purpose=purpose, industry=industry,
            source_of_funds=sof, expected_annual_deposits=float(annual_dep), expected_withdrawals=wd,
            foreign_wires_expected=wires, wire_countries=list(wire_ctry), pep_type=pep_type, bd_firm=bd_firm,
            trusted_contact_info=tc_info, tax_cert_signed=tax_cert, backup_withholding=backup, cip_notice_ack=ack_cip,
            privacy_notice_ack=ack_priv, esign_consent=ack_esign, agreement_signed=ack_agree, signature_name=signature,
            id_verification=idv, id_theft_flags=list(flags), id_theft_resolved=flags_ok and bool(flags), ofac_hits=hits,
            ofac_disposition=disp if n_hits else aml.OFAC_DISPOSITIONS[0],
            screening_source=f"{'OFAC SDN list' if sdn.live else 'offline sample list'}, {sdn.asof}", adverse_media=media,
            fincen_314a_match=m314, sow_documented=sow_doc, edd_approver=edd_by if edd else "",
            principal_approved=principal_ok, principal_name=principal_name, analyst=analyst),
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

    status, reasons = profile.kyc_status()
    miss = aml.missing_items(profile)
    with status_slot:
        items = reasons if status != "incomplete" else miss
        ui.application_status(status, aml.STATUS_LABEL[status], aml.STATUS_TITLE[status], STATUS_SUMMARY[status],
                              [(s_, len(v)) for s_, v in miss.items()], items)
    return profile


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
    ui.notes(["The account cannot be opened, so no recommendation is implemented. The items below go to the AML "
              "officer; they are corrected on the application or the firm-review panel in section 01."] + el.kyc_reasons,
             kind="Account declined", tone="bad")


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
ui.section("Open the account", "The same sequence a broker-dealer follows: the client's application, the firm's own "
           "review, then the investment profile. Cases A-C are approved clients with documented assumptions; the new "
           "client starts from an empty application. Change anything and every section below updates.",
           eyebrow="Know your customer", anchor="profile", num="01")
with st.spinner("Loading the OFAC sanctions list..."):
    sdn = get_sdn(use_live)
profile = kyc_form(choice, sdn)

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
             ("Account", aml.STATUS_LABEL[el.kyc_status])]
    if blocked:
        sub = "**Account declined.** Identity, sanctions or beneficial-ownership checks failed; nothing is implemented until compliance clears them."
    else:
        sub = (f"**{verdict}.** The plan works in {sim.success:.0%} of {cfg.n_paths:,} simulated markets ({sim.success_full:.0%} "
               f"with no spending cut). Recommended: the *{rp.tier_name}* portfolio, {rec.glide[0]:.0%} in stocks for long-term money."
               + (" **Trading on hold** pending compliance review." if el.kyc_status == "hold" else "")
               + (" **Proposal only:** the account application is not complete yet." if el.kyc_status == "incomplete" else ""))
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
    elif el.kyc_status == "incomplete":
        ui.notes([f"This is a proposal, not an open account: {len(el.kyc_reasons)} item(s) are outstanding on the account "
                  "application in section 01. Nothing is traded and no derivative is recommended until it is approved."],
                 kind="Proposal only", tone="info")
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
    c1, c2 = st.columns([1, 1.7], gap="large")
    with c1, st.container(key="card_donut"):
        ui.eyebrow("Household allocation")
        ui.chart(donut(rec.weights_now, f"{rec.stats['equity']:.0%}<br><span style='font-size:11px'>stocks</span>",
                       height=620, legend=False))
        st.caption(f"Stress test: 2008-style year {rec.stress['2008 global financial crisis']:.1%} · 2022-style "
                   f"{rec.stress['2022 stock-bond inflation shock']:.1%}")
    with c2, st.container(key="card_etfs"):
        ui.eyebrow("ETF implementation")
        if len(rec.implementation):
            impl = rec.implementation.sort_values("Weight", ascending=False)
            ui.etf_table([{"asset_class": r["Asset class"], "etf": r["ETF"], "alt": r["Alternative"], "weight": float(r["Weight"]),
                           "amount": float(r["Amount"]), "price": None if pd.isna(r["Price"]) else float(r["Price"]),
                           "shares": None if pd.isna(r["Shares"]) else float(r["Shares"]),
                           "er": None if pd.isna(r["Expense ratio"]) else float(r["Expense ratio"])}
                          for _, r in impl.iterrows()],
                         total_amount=float(impl["Amount"].sum()), weighted_er=rec.stats["expense_ratio"])
        else:
            st.caption("Nothing to invest yet: the target mix is shown in the donut; the trade list appears once assets are funded.")
        st.caption("The grey ticker is the tax-loss-harvesting pair: sell the primary ETF at a loss, hold the pair for 31 days "
                   "to avoid a wash sale, then switch back. Shares are whole shares at the latest price. "
                   "Rebalance when a sleeve drifts 5 points or 25% from target.")
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
idn = p.identity
rating = aml.risk_rating(p)
log = build_log(p, el.kyc_status, rec.checklist)
log_sum = log_summary(log)
need_action = log_sum["OPEN"] + log_sum["REVIEW"] + log_sum["FAIL"]
ui.section("Compliance file", "What a reviewer checks before this recommendation goes to the client: the account-opening "
           "record (identification, due diligence, sanctions screening), the customer's AML risk rating, the Regulation "
           "Best Interest and fiduciary care obligations, and the audit log that ties them together.",
           eyebrow="Compliance", anchor="compliance", num="08")
status_tone = {"cleared": "good", "hold": "mid", "incomplete": "info", "blocked": "bad"}[el.kyc_status]
n_hits = len(idn.ofac_hits)
if not idn.ofac_clear:
    scr = ("Match", "confirmed: block and report", "bad")
elif n_hits and idn.ofac_disposition == aml.OFAC_DISPOSITIONS[0]:
    scr = (f"{n_hits} to review", "potential match, not yet dispositioned", "mid")
elif n_hits:
    scr = ("Cleared", f"{n_hits} false positive(s) dispositioned", "good")
else:
    scr = ("Clear", f"{len(aml.parties(idn))} name(s) · {'OFAC SDN list' if sdn.live else 'offline sample list'}", "good")
ui.kpis([
    {"label": "Account status", "value": aml.STATUS_LABEL[el.kyc_status], "sub": "CIP · CDD · OFAC · principal", "tone": status_tone},
    {"label": "AML risk rating", "value": rating.level.capitalize(), "sub": f"{rating.score}/100 · review every {rating.review_months} months",
     "tone": {"low": "good", "medium": "mid", "high": "bad"}[rating.level]},
    {"label": "Sanctions screening", "value": scr[0], "sub": scr[1], "tone": scr[2]},
    {"label": "Audit log", "value": f"{len(log)} lines", "sub": f"{need_action} need action" if need_action else "nothing needs action",
     "tone": "bad" if log_sum["FAIL"] else "mid" if need_action else "good"},
])
ui.kpis([
    {"label": "Suitability checklist", "value": f"{sum(1 for c_ in rec.checklist if c_[1])} / {len(rec.checklist)}", "sub": "items passing",
     "tone": "good" if all(c_[1] for c_ in rec.checklist) else "bad"},
    {"label": "Options / futures", "value": f"L{el.options_level} / {'yes' if el.futures_ok else 'no'}", "sub": "approval on file", "tone": "info"},
    {"label": "Profile age", "value": f"{p.profile_age_months} mo", "sub": "refresh every 36 months", "tone": "good" if p.profile_age_months <= 36 else "bad"},
    {"label": "Advisor review items", "value": str(len(rp.review_flags)), "sub": "inconsistent answers / overrides", "tone": "bad" if rp.review_flags else "good"},
])
if el.kyc_reasons:
    ui.notes(el.kyc_reasons, kind={"hold": "Pending review", "incomplete": "Outstanding", "blocked": "Declined"}.get(el.kyc_status, "Account gate"),
             tone={"hold": "mid", "incomplete": "info", "blocked": "bad"}.get(el.kyc_status, "mid"))

ui.subsection("Know-your-customer record", "What was collected and verified at account opening. Identification numbers are "
              "masked here; the full values stay with the custodian.")
c1, c2 = st.columns([1.3, 1], gap="large")
with c1, st.container(key="card_cip"):
    ui.eyebrow("Customer identification and due diligence")
    dob_age = aml.age_on(idn.date_of_birth)
    addr = ", ".join(x_ for x_ in (idn.address_line, idn.address_city, f"{idn.state} {idn.address_zip}".strip() if idn.address_country == aml.US
                                   else idn.address_zip, idn.address_country) if x_)
    others = []
    if idn.account_type == "joint" and idn.joint_owner.get("name"):
        others.append(f"Joint owner {idn.joint_owner['name']} ({idn.joint_owner.get('tin_masked', '')}); {str(idn.joint_owner.get('tenancy', '')).lower()}")
    if idn.account_type in aml.ENTITY_TYPES and idn.entity.get("legal_name"):
        others.append(f"{idn.entity['legal_name']} ({idn.entity.get('kind', '')}, EIN {idn.entity.get('ein_masked', '')})")
    others += [f"{b_['name']} ({b_.get('role', 'owner')})" for b_ in idn.beneficial_owners if b_.get("name")]
    acks = [n_ for v_, n_ in ((idn.cip_notice_ack, "CIP notice"), (idn.form_crs_acknowledged, "Form CRS / ADV 2A"),
                              (idn.privacy_notice_ack, "privacy notice"), (idn.esign_consent, "e-delivery"),
                              (idn.agreement_signed, "agreements")) if v_]
    ui.table(["Item", "On file", "Basis"], [
        ["Legal name", idn.full_name or "not provided", "CIP: 31 CFR 1023.220(a)(2)(i)"],
        ["Date of birth", f"{idn.date_of_birth} (age {dob_age})" if dob_age is not None else "not provided", "CIP"],
        ["Home address", addr or "not provided", "CIP: street address, not a P.O. box"],
        ["Identification number", f"{idn.tin_type} {idn.tin_masked}" + ("" if idn.tin_valid else " (not valid)") if idn.tin_masked else "not provided", "CIP; Form W-9"],
        ["Photo ID", f"{idn.id_doc_type} {idn.id_doc_masked}, {idn.id_doc_issuer}, expires {idn.id_doc_expiry}" if idn.id_doc_masked else "not provided",
         "Documentary verification"],
        ["Identity verification", idn.id_verification + ("; red flags: " + "; ".join(idn.id_theft_flags) if idn.id_theft_flags else ""),
         "31 CFR 1023.220(a)(2)(ii); Reg S-ID"],
        ["Citizenship and tax", f"{idn.citizenship} ({idn.country_of_citizenship}); tax residency: {idn.tax_residency}"
         + (f" + {idn.other_tax_country}" if idn.other_tax_country else "") + ("; backup withholding" if idn.backup_withholding else ""),
         "Form W-8BEN" if idn.citizenship == "Non-resident alien" else "Form W-9"],
        ["Account", f"{aml.ACCOUNT_LABEL.get(idn.account_type, idn.account_type)}; {idn.account_purpose.lower()}", "FINRA 4512; CDD purpose"],
        ["Other parties", "; ".join(others) or "none", "CIP per owner; 31 CFR 1010.230"],
        ["Employment", f"{idn.employment_status}; {idn.occupation or 'n/a'}" + (f" at {idn.employer}" if idn.employer else "")
         + (f"; {idn.industry.lower()}" if idn.industry != aml.INDUSTRIES[0] else ""), "FINRA 4512(a)(2); SEA 17a-3(a)(17)"],
        ["Source of wealth and funds", f"{idn.source_of_wealth}; funded by: {idn.source_of_funds}"
         + ("; documents received" if idn.sow_documented else ""), "CDD: 31 CFR 1023.210(b)(5)"],
        ["Expected activity", f"{money(idn.expected_initial_funding)} initial, {money(idn.expected_annual_deposits)}/yr after; withdrawals "
         f"{idn.expected_withdrawals.lower()}" + ("; international wires: " + (", ".join(idn.wire_countries) or "countries not given")
                                                  if idn.foreign_wires_expected else ""), "CDD baseline for monitoring"],
        ["Political exposure / media / 314(a)", f"{idn.pep_type if idn.politically_exposed else 'none'}; {idn.adverse_media.lower()}; "
         f"314(a) {'MATCH' if idn.fincen_314a_match else 'no match'}", "31 CFR 1010.620; 1010.520"],
        ["Trusted contact", (f"{idn.trusted_contact}" + (f", {idn.trusted_contact_info}" if idn.trusted_contact_info else "")) if idn.trusted_contact
         else ("declined" if idn.trusted_contact_declined else "not provided"), "FINRA 4512(a)(1)(F)"],
        ["Disclosures and signature", (", ".join(acks) or "none acknowledged") + (f"; signed '{idn.signature_name}' {idn.profile_date}" if idn.signature_name else "; not signed"),
         "Reg BI; Reg S-P; E-SIGN; FINRA 2268"],
        ["Accepted by", f"{idn.principal_name} (principal)" if idn.principal_approved else "not yet accepted", "FINRA 4512(a)(1)(D)"],
    ])
with c2:
    with st.container(key="card_rating"):
        ui.eyebrow(f"Customer AML risk rating · {rating.level} · {rating.score} / 100")
        ui.risk_rating(rating.score, rating.level)
        ui.table(["Factor", "Points", "Why"], rating.table(), num_cols={1})
        st.caption(f"House scoring model (no regulation prescribes one): under {aml.RISK_MEDIUM} low, {aml.RISK_MEDIUM}-{aml.RISK_HIGH - 1} "
                   f"medium, {aml.RISK_HIGH}+ high. Periodic review every {rating.review_months} months, sooner on a trigger event. "
                   + ("**Enhanced due diligence required:** " + "; ".join(rating.edd_reasons) + "." if rating.edd_required
                      else "Enhanced due diligence is not required.")
                   + f" FATF lists as of the {aml.FATF_ASOF}; sanctioned territories as of {aml.SANCTIONS_ASOF}.")
    with st.container(key="card_monitor"):
        ui.eyebrow("After opening · what surveillance watches for")
        for n_ in aml.monitoring_notes(p):
            st.markdown("- " + n_.replace("$", "\\$"))

ui.subsection("Suitability checklist", "The Regulation Best Interest and fiduciary care items the recommendation file must evidence.")
c1, c2 = st.columns([1.5, 1], gap="large")
with c1:
    ui.checklist(rec.checklist)
with c2, st.container(key="card_notes"):
    ui.eyebrow("Account notes")
    for n_ in el.notes:
        st.markdown("- " + n_.replace("$", "\\$"))

ui.subsection("Audit log", "One line per control, in the order an analyst works the file. **Click any line** to open its "
              "instructions: what to do now for that result, what the line records, the rule it evidences, who acts and "
              "how long the record is kept.")
ui.result_legend(AUDIT_RESULTS)
c = st.columns([1.4, 1.3, 2.3], vertical_alignment="center")
log_view = c[0].radio("Show", ["All lines", "Needs action"], horizontal=True, key="log_view", label_visibility="collapsed")
open_all = c[1].toggle("Open every instruction", value=False, key="log_open")
shown = [ln for ln in log if ln["result"] in ("OPEN", "REVIEW", "FAIL")] if log_view == "Needs action" else log
c[2].caption(f"{len(shown)} of {len(log)} lines · " + " · ".join(f"{v_} {k_}" for k_, v_ in log_sum.items() if v_))
if shown:
    ui.log_lines(shown, open_all=open_all)
else:
    ui.notes(["No line needs action: every control passed or is informational."], kind="All clear", tone="good")
st.caption(f"Evaluated {log[0]['time']} UTC. Each line carries a hash that covers the line before it, so an altered or deleted entry "
           f"breaks the chain (final hash `{log[-1]['hash']}`); production systems keep this trail in non-rewriteable storage "
           "(SEA Rule 17a-4(f)). The codes group as APP application, CIP identification, CDD due diligence, SAN sanctions and "
           "screening, TAX, DIS disclosures, ACC account protections, SUI suitability, MON monitoring, DEC decision, REC records.")

hist = st.session_state.setdefault("decision_history", {}).setdefault(choice, [])
if not hist or hist[-1][1] != el.kyc_status:
    hist.append((log[0]["time"], el.kyc_status, need_action))
if len(hist) > 1:
    with st.expander(f"Decision history this session · {len(hist)} status changes"):
        ui.table(["Time (UTC)", "Status", "Lines needing action"],
                 [[t_, aml.STATUS_LABEL[s_], str(n_)] for t_, s_, n_ in hist], num_cols={2})

audit = {"generated_utc": log[0]["time"], "firm": "Keel & Co. (demo)", "profile": p.to_dict(),
         "kyc_status": el.kyc_status, "kyc_status_label": aml.STATUS_LABEL[el.kyc_status], "kyc_reasons": el.kyc_reasons,
         "aml_risk_rating": {"score": rating.score, "level": rating.level, "review_months": rating.review_months,
                             "edd_required": rating.edd_required, "edd_reasons": rating.edd_reasons,
                             "factors": [{"factor": a_, "points": b_, "why": c_} for a_, b_, c_ in rating.factors]},
         "sanctions_screening": {"list": sdn.source, "list_asof": sdn.asof, "listed_parties": sdn.count,
                                 "names_screened": [n_ for _, n_, _, _ in aml.parties(idn)], "hits": idn.ofac_hits,
                                 "disposition": idn.ofac_disposition},
         "audit_log": log, "risk": rp.summary(),
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
c1, c2, c3 = st.columns(3)
c1.download_button("Download audit record (JSON)", json.dumps(audit, indent=2, default=float),
                   file_name=f"robo_audit_{choice}.json", mime="application/json", width="stretch")
c2.download_button("Download audit log with instructions (CSV)", log_csv(log),
                   file_name=f"keel_audit_log_{choice}.csv", mime="text/csv", width="stretch")
c3.download_button("Download client report (HTML, print to PDF)",
                   build_report(rec, sim_label=f"{cfg.n_paths:,} paths, {cfg.spending_rule} spending, seed {cfg.seed}"),
                   file_name=f"keel_client_report_{choice}.html", mime="text/html", type="primary", width="stretch")
with st.container(key="card_disclosures"):
    ui.eyebrow("Disclosures")
    st.markdown(
        "**IMPORTANT: The projections or other information generated by this tool regarding the likelihood of various "
        "investment outcomes are hypothetical in nature, do not reflect actual investment results and are not guarantees "
        "of future results.** Results vary with each use and over time. Returns use J.P. Morgan "
        "Asset Management's 2026 Long-Term Capital Market Assumptions (10-15 years, passive, before fees) less the stated "
        "fees; actual returns, inflation and taxes will differ. The universe is 15 broad index ETFs; no individual "
        "securities are selected. Recommendations are produced by an algorithm from the answers you give: it rebalances "
        "by rule regardless of market conditions, and an advisor reviews flagged cases rather than every account. The "
        "risk questionnaire is not the sole basis of the "
        "recommendation: financial capacity, goals and stated loss limits are weighed with it, and inconsistent answers "
        "are referred to an advisor. Options involve risk and are not suitable for all investors; read *Characteristics "
        "and Risks of Standardized Options*. Futures trading involves substantial risk of loss. Buffered ETFs protect "
        "only if held for the full outcome period. Fair values in the Markets section are model estimates from public data. "
        "Identity verification, adverse-media and 314(a) results on this page are entered by hand to show the workflow; "
        "sanctions screening uses the public OFAC list and is not a substitute for a licensed screening service. "
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
    ("Open the account", "Application (CIP data, tax certification, disclosures, signature), identity verification, OFAC "
                         "name screening, AML risk rating; missing items wait, a failure declines, a review holds."),
    ("Know the client", "The FINRA 2111 / Reg BI investment profile, product-level experience, trusted contact, insider status."),
    ("Measure risk three ways", "Willingness (gamma, loss limit, consistency checks), ability (balance sheet, stress test, capacity) and need."),
    ("Bucket the goals", "Each goal gets its own horizon-matched mix; retirement gets a household glide path."),
    ("Build the models", "Resampled mean-variance on J.P. Morgan 2026 assumptions under investment-policy limits."),
    ("Gate the derivatives", "Options level 0-4, futures eligibility and issuer rules decide which overlays are allowed."),
    ("Stress the plan", "Monte Carlo with fat tails, fees, 2026 tax brackets, RMDs, Social Security, longevity, historical replays."),
    ("Document and review", "Reg BI checklist, audit log with instructions per line, client report, risk-based refresh, advisor-review flags."),
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
        ["Name, date of birth, address, ID number, photo ID", "Customer Identification Program record; sanctions name screening; age for every projection; account gate"],
        ["Account type, citizenship, tax residency, source of wealth, PEP", "Account gate (decline/hold); IRA-only limits on margin and futures; non-US tax notes; EDD"],
        ["Funding method, expected deposits and withdrawals, industry, countries", "Customer AML risk rating and review cycle; source-of-funds review; monitoring baseline and reporting triggers"],
        ["Disclosures, tax certification, signature, firm review", "Application completeness (proposal vs open account); audit log; principal acceptance"],
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
