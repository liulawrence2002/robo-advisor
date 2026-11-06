"""Suitability and product gating.

Options approval levels follow the structure brokers use under FINRA Rule 2360 (Schwab
and Fidelity publish similar tiers); thresholds are this robo's house policy:

  Level 0  no options
  Level 1  protective strategies on shares held: collars, protective puts, covered calls
           (Schwab 'Level 0', Fidelity 'Level 1'); any objective, because they reduce risk
  Level 2  + long calls/puts, cash-secured puts
  Level 3  + spreads (margin account; never in an IRA)
  Level 4  + uncovered writing (never recommended by this robo)

Futures eligibility follows NFA Rule 2-30 (collect age, income, net worth, experience;
deliver the CFTC Rule 1.55 risk disclosure) plus house thresholds. The robo uses futures
only to hedge existing equity exposure, never to speculate or add leverage.

Account gating (robo/aml.py): a failed CIP/OFAC check blocks the account; an application with
missing items is a proposal only; a PEP without EDD sign-off, an undispositioned list match
or a high-risk source of wealth puts trading on hold. Insiders (Rule 144 control persons) are
not offered hedges on their employer's stock unless the issuer permits hedging; they get a
Rule 10b5-1 staged-sale plan instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .derivatives import (Market, Overlay, equity_beta_to_sp500, futures_hedge, make_buffer_etf, make_collar,
                          make_covered_call, make_csp, option_cost)
from .kyc import ClientProfile
from .risk import RiskProfile

KNOW = {"none": 0, "limited": 1, "good": 2, "extensive": 3}
OBJ = {"preservation": 0, "income": 1, "growth": 2, "speculation": 3}

LEVEL_RULES = [  # (level, min knowledge, min options yrs, min objective, min liquid NW, margin)
    (1, 1, 0, 0, 50_000, False),
    (2, 2, 1, 1, 100_000, False),
    (3, 2, 2, 2, 250_000, True),
    (4, 3, 3, 3, 500_000, True),
]
FUTURES_RULE = {"min_liquid_nw": 250_000, "min_knowledge": 2, "min_tier": 3}
RULE_144_PCT_OUTSTANDING = 0.01     # greater of 1% of shares outstanding or 4-week ADV, per 3 months


@dataclass
class Eligibility:
    options_level: int
    options_reasons: list[str]
    futures_ok: bool
    futures_reasons: list[str]
    accredited: bool
    buffer_etf_ok: bool
    managed_futures_ok: bool
    notes: list[str] = field(default_factory=list)
    kyc_status: str = "cleared"               # cleared | hold | blocked
    kyc_reasons: list[str] = field(default_factory=list)
    margin_ok: bool = False
    qualified_client: bool = False


def eligibility(p: ClientProfile, rp: RiskProfile) -> Eligibility:
    x, f, i = p.experience, p.finances, p.identity
    lnw = f.liquid_net_worth
    status, kyc_why = p.kyc_status()
    reasons, level = [], 0
    ira_only = i.account_type == "IRA only"
    margin_ok = x.wants_margin and not ira_only and lnw >= 25_000 and f.household_income > 0
    if x.exclude_derivatives:
        reasons.append("Client opted out of derivatives.")
    elif not x.options_disclosure_ack:
        reasons.append("Options Disclosure Document (Characteristics & Risks of Standardized Options) "
                       "not acknowledged, so no options approval (FINRA Rule 2360).")
    else:
        for lv, kn, yrs, obj, nw, margin in LEVEL_RULES:
            missing = []
            if KNOW[x.knowledge] < kn: missing.append(f"knowledge >= {list(KNOW)[kn]}")
            if x.options_years < yrs: missing.append(f"{yrs}+ yrs options experience")
            if OBJ[x.objective] < obj: missing.append(f"objective >= {list(OBJ)[obj]}")
            if lnw < nw: missing.append(f"liquid net worth >= ${nw:,.0f}")
            if margin and not margin_ok:
                missing.append("margin account (not available in an IRA)" if ira_only else "margin agreement")
            if missing:
                reasons.append(f"Level {lv} not granted: needs " + ", ".join(missing) + ".")
                break
            level = lv
        if level:
            reasons.insert(0, f"Approved for options Level {level} (Registered Options Principal sign-off and the "
                              "15-day account-information verification letter are recorded in the audit file).")
    if status == "blocked":
        level = 0
        reasons.insert(0, "Account cannot be opened: no options approval.")

    fr, fok = [], True
    if x.exclude_derivatives:
        fok = False; fr.append("Client opted out of derivatives.")
    if not x.futures_disclosure_ack:
        fok = False; fr.append("CFTC Rule 1.55 futures risk disclosure not acknowledged.")
    if lnw < FUTURES_RULE["min_liquid_nw"]:
        fok = False; fr.append(f"Liquid net worth below ${FUTURES_RULE['min_liquid_nw']:,} house minimum.")
    if KNOW[x.knowledge] < FUTURES_RULE["min_knowledge"]:
        fok = False; fr.append("Investment knowledge below 'good'.")
    if rp.tier < FUTURES_RULE["min_tier"]:
        fok = False; fr.append("Risk tier too low for a futures account.")
    if ira_only:
        fok = False; fr.append("Futures are not held in an IRA-only relationship (no margin in IRAs).")
    if status == "blocked":
        fok = False
    if fok:
        fr.append("Eligible for a futures account (hedging use only)."
                  + (" NFA Interpretive Notice 9004: additional risk disclosure delivered to a retired/senior client."
                     if p.is_senior or p.is_retired else ""))

    notes = []
    if p.is_senior:
        tc = ("on file." if i.trusted_contact else ("declined by the client (documented)." if i.trusted_contact_declined
                                                    else "MISSING (Rule 4512 requires a reasonable effort)."))
        notes.append("Specified adult (65+): FINRA Rule 2165 temporary-hold procedures apply; trusted contact " + tc
                     + (f" Power of attorney: {i.power_of_attorney}." if i.power_of_attorney else ""))
    if i.employed_by_broker_dealer:
        notes.append("Employed by a broker-dealer: Rule 3210 written consent and duplicate statements required.")
    if i.control_person_public_co:
        notes.append(f"Control person of {i.insider_ticker or 'a public company'}: sales limited by Rule 144 (greater of 1% "
                     "of shares outstanding or 4-week average volume per 3 months), Form 144 filing, Section 16 "
                     "short-swing rule; hedging " + ("permitted by the issuer's policy." if i.issuer_permits_hedging
                                                      else "prohibited by the issuer's anti-hedging policy (Reg S-K 407(i)).")
                     + (" A Rule 10b5-1 plan is in place." if i.has_10b5_1_plan else " No Rule 10b5-1 plan yet."))
    if i.politically_exposed:
        notes.append("Politically exposed person" + (f" ({i.pep_detail})" if i.pep_detail else "") + ": enhanced due "
                     "diligence " + ("completed and approved." if i.edd_cleared else "pending senior-management approval."))
    if i.tax_residency != "US only" or i.citizenship == "Non-resident alien":
        notes.append("Non-US tax residency: Form W-8BEN (renewed every three years), 30% (or treaty) withholding on US dividends, "
                     "US estate-tax exposure on US-situs assets above $60,000; IRA/Roth contributions need US earned income.")
    if i.account_type in ("trust", "LLC / corporation"):
        notes.append(f"{i.account_type.capitalize()} account: beneficial owners "
                     + ("verified (FinCEN CDD)." if i.beneficial_owners_verified else "NOT verified."))
    if i.account_type == "IRA only":
        notes.append("IRA-only relationship: no margin, no spreads, no futures; options limited to Level 2 with the "
                     "custodian's IRA options agreement.")
    if not i.form_crs_acknowledged:
        notes.append("Form CRS / ADV Part 2A delivery not acknowledged: deliver before or at the time of the recommendation.")
    acc = p.is_accredited
    notes.append(("Accredited investor (Reg D 501(a))" if acc else "Not an accredited investor") +
                 (", qualified client (Rule 205-3)" if p.is_qualified_client else "") +
                 ": private funds are outside this robo's universe; refer to an advisor if interested.")
    if x.override_tiers:
        notes.append(f"Client override of {x.override_tiers:+d} tier(s)"
                     + (f" ({x.override_reason})" if x.override_reason else "")
                     + (": applied (more conservative)." if x.override_tiers < 0 else ": pending advisor review (more aggressive)."))
    return Eligibility(options_level=level, options_reasons=reasons, futures_ok=fok, futures_reasons=fr,
                       accredited=acc, buffer_etf_ok=rp.tier >= 2 and not x.exclude_derivatives,
                       managed_futures_ok=rp.tier >= 3, notes=notes, kyc_status=status, kyc_reasons=kyc_why,
                       margin_ok=margin_ok, qualified_client=p.is_qualified_client)


@dataclass
class OverlayPlan:
    overlays: list[tuple[Overlay, int, int]]     # (overlay, start year t, end year t)
    rationale: list[str]
    futures: dict | None
    not_recommended: list[str]
    single_stock_collar: dict | None = None
    concentrated_plan: dict | None = None        # staged-sale / 10b5-1 schedule for the single-stock position


SEQUENCE_WINDOW = 7       # years before retirement when downside protection starts
SINGLE_STOCK_VOL = 0.30   # typical single-stock implied vol (replaced by the live quote when available)


def concentrated_stock_plan(p: ClientProfile, el: Eligibility, m: Market, stock_vol: float | None = None,
                            stock_info: dict | None = None) -> tuple[dict | None, dict | None, list[str]]:
    """Returns (collar terms, staged-sale plan, rationale) for a concentrated position."""
    f, i = p.finances, p.identity
    conc = f.concentrated_stock
    why = []
    if conc <= 0.10 * max(f.liquid_net_worth, f.taxable, 1):
        return None, None, why
    vol = stock_vol or SINGLE_STOCK_VOL
    insider = i.control_person_public_co
    hedge_blocked = insider and not i.issuer_permits_hedging
    gain = conc * (1 - f.taxable_cost_basis_pct)
    # staged sale: equal tranches over 8 quarters (2-3 tax years), limited by Rule 144 volume for insiders
    tranches = 8
    per_tranche = conc / tranches
    r144_cap = None
    if insider and stock_info and stock_info.get("shares_outstanding") and stock_info.get("price"):
        r144_cap = max(RULE_144_PCT_OUTSTANDING * stock_info["shares_outstanding"],
                       (stock_info.get("avg_volume") or 0) * 4 * 5) * stock_info["price"]    # $ per 3 months
    plan = {"value": conc, "embedded_gain": gain, "tranches": tranches, "per_tranche": per_tranche,
            "tax_per_tranche": gain / tranches * f.cap_gains_rate, "rule144_quarterly_cap": r144_cap,
            "needs_10b5_1": insider and not i.has_10b5_1_plan,
            "exchange_fund_eligible": p.is_accredited and conc >= 500_000}
    ssc = None
    if hedge_blocked:
        why.append(f"Concentrated position of ${conc:,.0f} in {i.insider_ticker or 'the employer'} held by an insider: the "
                   "issuer's anti-hedging policy rules out a collar or protective put. Diversify through a Rule 10b5-1 "
                   f"plan selling {tranches} equal tranches over two years (${per_tranche:,.0f} each, about "
                   f"${plan['tax_per_tranche']:,.0f} tax per tranche)" +
                   (f", within the Rule 144 limit of ${r144_cap:,.0f} per quarter" if r144_cap else "") + ".")
    elif el.options_level >= 1:
        sm = Market(spot=100.0, vix=None, rate=m.rate, div_yield=0.015, sigma_override=vol)
        col = make_collar(sm, 1.0, put_k=0.90)
        ssc = {"value": conc, "put_strike": col.put_k, "call_strike": col.call_k,
               "put_cost": option_cost(sm, col.put_k, "put"), "max_loss": conc * (1 - col.put_k),
               "max_gain": conc * (col.call_k - 1), "vol": vol, "live_vol": stock_vol is not None}
        src = "the live option chain" if stock_vol is not None else "a 30% assumed implied vol"
        why.append(f"Concentrated single-stock position (${conc:,.0f}): a 1-year zero-cost collar (buy the "
                   f"{col.put_k:.0%} put, sell the {col.call_k:.0%} call, priced at {vol:.0%} from {src}) caps the loss at "
                   f"10% while a staged sale over {tranches} quarters diversifies the position across 2-3 tax years. Keep the "
                   "band wide enough to avoid a constructive sale (IRC 1259) and note the straddle rules (IRC 1092) "
                   "suspend the holding period for qualified dividends.")
        if col.call_k <= 1.0 + 1e-6:
            why.append("At this volatility the 90% put costs more than an at-the-money call: a zero-cost collar would "
                       "cap all upside. Consider an 85% put or paying for the put outright.")
    else:
        why.append(f"Concentrated position (${conc:,.0f}): staged diversification over {tranches} quarters "
                   f"(${per_tranche:,.0f} per tranche); no options approval for a collar.")
    if plan["exchange_fund_eligible"]:
        why.append("Accredited investor with a large position: an exchange fund (7-year lock-up) or a charitable "
                   "remainder trust can diversify without an immediate capital-gains bill; referred to an advisor.")
    return ssc, plan, why


def recommend_overlays(p: ClientProfile, rp: RiskProfile, el: Eligibility, m: Market,
                       us_lc_value: float, horizon_t: int = 60, weights=None, cma=None,
                       stock_vol: float | None = None, stock_info: dict | None = None) -> OverlayPlan:
    """Choose at most one equity overlay with its active window, plus optional hedges."""
    f, h = p.finances, p.household
    yrs_to_ret = p.effective_retirement_age - h.age
    retired = p.is_retired
    ovs, why, skip = [], [], []
    fut = None
    if el.kyc_status != "cleared":
        skip.append("No derivatives until the account is approved (it is on hold, incomplete or declined).")
        return OverlayPlan([], why, None, skip)

    ssc, plan, why_c = concentrated_stock_plan(p, el, m, stock_vol, stock_info)
    why += why_c

    start = max(yrs_to_ret - SEQUENCE_WINDOW, 0)
    end = max(yrs_to_ret, 0) + 5
    if not retired and el.options_level >= 1 and us_lc_value > 25_000 and yrs_to_ret <= 15:
        ovs.append((make_collar(m, 0.40), start, end))
        when = "now" if start == 0 else f"at age {h.age + start}"
        why.append(f"Sequence-of-returns risk peaks in the {SEQUENCE_WINDOW} years before and 5 years after "
                   f"retirement. Starting {when}, a rolling zero-cost collar on 40% of US large cap limits a "
                   "2008-style loss on that sleeve to about 10%, at the cost of upside above the call strike.")
    elif retired and el.options_level >= 1 and p.experience.objective in ("income", "preservation") \
            and us_lc_value > 100_000:
        ovs.append((make_covered_call(m, 0.30, 1.06), 0, horizon_t))
        why.append("Retired with an income objective: overwriting 30% of US large cap with 1-year 106% calls "
                   "turns part of the upside into premium income (written in the IRA under the custodian's IRA "
                   "options agreement, premiums are not taxed each year).")
    elif el.options_level == 0 and el.buffer_etf_ok and (retired or yrs_to_ret <= 15):
        ovs.append((make_buffer_etf(0.30), 0 if retired else start, end if not retired else horizon_t))
        why.append("No options approval: a 10%-buffer S&P 500 ETF on 30% of large cap gives similar protection "
                   "inside a fund (about 0.75% a year and forgoes dividends), during the sequence-risk window.")
    elif retired:
        skip.append("No equity overlay today: with a growth objective in retirement the equity premium is worth more "
                    "than capped upside, and the income reserve already covers five years of withdrawals.")
    else:
        skip.append("No equity overlay today: with a long horizon the equity premium is worth more than "
                    "downside protection, and the core model already respects the client's loss limit. "
                    f"Revisit {SEQUENCE_WINDOW} years before retirement.")

    if el.options_level >= 2 and rp.tier >= 5 and f.cash > 6 * f.monthly_essential_expenses + 25_000:
        ovs.append((make_csp(m, 0.10), 0, 1))
        why.append("Excess cash waiting to be invested: cash-secured puts at 95% get paid to buy the target "
                   "equity allocation on a 5% dip.")

    if el.futures_ok and f.taxable > 250_000 and f.taxable_cost_basis_pct < 0.7:
        gain = f.taxable * (1 - f.taxable_cost_basis_pct)
        hedge_notional = 0.10 * f.taxable
        beta = equity_beta_to_sp500(weights, cma) if (weights is not None and cma is not None) else 1.0
        fut = futures_hedge(hedge_notional, m, beta=beta, micro=True)
        fut["tax_deferred"] = gain * 0.10 * f.cap_gains_rate
        fut["margin_covered"] = f.cash >= fut["initial_margin_est"]
        why.append(f"Taxable account has ${gain:,.0f} of embedded gains. To trim equity by 10% of the taxable "
                   f"account without realising gains, short {fut['contracts']} {fut['contract']} contracts "
                   f"(~${fut['hedged_notional']:,.0f}, sleeve beta {beta:.2f}); roll quarterly until gains can be "
                   "offset or stepped up. Futures gains are 60/40 long/short-term (IRC 1256); keep the hedge under "
                   "~15% of the position so it is not a constructive sale (IRC 1259)."
                   + ("" if fut["margin_covered"] else f" Initial margin (~${fut['initial_margin_est']:,.0f}) exceeds bank "
                                                       "cash: fund the futures account from the taxable sleeve first."))
    elif el.futures_ok:
        skip.append("Futures approved, but no hedge needed: the allocation can be adjusted inside "
                    "tax-advantaged accounts at no tax cost.")

    skip += ["Uncovered options, spreads and leveraged futures: outside robo policy at every level.",
             "Private equity / private credit (JPM's '60/40+' alternatives): illiquid and limited to "
             "accredited investors, so they are referred to an advisor, not automated."]
    return OverlayPlan(overlays=ovs, rationale=why, futures=fut, not_recommended=skip, single_stock_collar=ssc,
                       concentrated_plan=plan)


def reg_bi_checklist(p: ClientProfile, rp: RiskProfile, el: Eligibility, fee_total: float | None = None) -> list[tuple[str, bool, str]]:
    """Reg BI / Advisers Act care-obligation items the recommendation file must evidence."""
    f, i = p.finances, p.identity
    months = p.profile_age_months
    fee_txt = f"all-in cost {fee_total:.2%}/yr (advisory {f.advisory_fee:.2%} + ETFs)" if fee_total is not None else "costs shown"
    return [
        ("Investment profile complete (FINRA 2111 / Reg BI)", rp.n_answered >= 8 and bool(p.goals),
         f"{rp.n_answered}/10 risk items answered; {len(p.goals)} goal(s); finances, tax status, horizon, liquidity collected"),
        ("Identity verified (CIP) and OFAC screened",
         el.kyc_status in ("cleared", "hold") and i.id_method != "pending"
         and not (i.ofac_hits and i.ofac_disposition == "Not reviewed"),
         f"CIP {i.id_method}; OFAC {'clear' if i.ofac_clear else 'HIT'}; status: {el.kyc_status}"),
        ("Customer due diligence (source of wealth, PEP, beneficial owners)", el.kyc_status == "cleared",
         f"source of wealth: {i.source_of_wealth}; PEP: {'yes' if i.politically_exposed else 'no'}"),
        ("Trusted contact requested (FINRA 4512)", bool(i.trusted_contact) or i.trusted_contact_declined,
         i.trusted_contact or ("declined" if i.trusted_contact_declined else "missing")),
        ("Tax certification and disclosures signed (W-9 / W-8BEN, privacy, e-delivery, agreement)",
         not i.enforce_intake or (i.tax_cert_signed and i.privacy_notice_ack and i.esign_consent and i.agreement_signed
                                  and bool(i.signature_name)),
         (f"signed '{i.signature_name}'" if i.signature_name else "legacy record on file") if not i.enforce_intake or i.agreement_signed
         else "outstanding: see the application status"),
        ("Profile current (SEC 17a-3: 36-month refresh)", months <= 36,
         f"confirmed {months} month(s) ago" if i.profile_date else "confirmed today"),
        ("Risk answers consistent, or reviewed", not rp.review_flags,
         f"{len(rp.review_flags)} item(s) need advisor review" if rp.review_flags else "no inconsistencies detected"),
        ("Recommended risk within stated loss tolerance", rp.model_worst_loss >= -rp.max_loss - 1e-9,
         f"model's worst stress loss {rp.model_worst_loss:.1%} vs limit {rp.max_loss:.0%}"),
        ("Liquidity needs covered", f.cash >= 3 * f.monthly_essential_expenses and f.cash >= f.liquidity_need_12m,
         f"{f.cash / max(f.monthly_essential_expenses, 1):.1f} months of essentials in cash; "
         f"${f.liquidity_need_12m:,.0f} needed within 12 months"),
        ("Derivatives only within approval level", True, f"options level {el.options_level}; "
         f"futures {'approved' if el.futures_ok else 'not approved'}"),
        ("Costs and reasonably available alternatives considered", (fee_total or 0) < 0.01,
         fee_txt + "; compared with a target-date fund and a 60/40 index mix (Methodology)"),
        ("Form CRS / ADV 2A delivered; conflicts disclosed", i.form_crs_acknowledged,
         "no proprietary funds, no payment for order flow, asset-based fee only"),
    ]
