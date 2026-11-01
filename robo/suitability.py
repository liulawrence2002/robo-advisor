"""Suitability and product gating.

Options approval levels follow the structure brokers use under FINRA Rule 2360 (Schwab
and Fidelity publish similar tiers); thresholds are this robo's house policy:

  Level 0  no options
  Level 1  covered calls, protective puts, collars on shares held (Schwab 'Level 0',
           Fidelity 'Level 1')
  Level 2  + long calls/puts, cash-secured puts
  Level 3  + spreads (margin account; not in IRAs)
  Level 4  + uncovered writing (never recommended by this robo)

Futures eligibility follows NFA Rule 2-30 (collect age, income, net worth, experience;
deliver the CFTC Rule 1.55 risk disclosure) plus house thresholds. The robo uses futures
only to hedge existing equity exposure, never to speculate or add leverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .derivatives import (Market, Overlay, futures_hedge, make_buffer_etf, make_collar,
                          make_covered_call, make_csp, option_cost)
from .kyc import ClientProfile
from .risk import RiskProfile

KNOW = {"none": 0, "limited": 1, "good": 2, "extensive": 3}
OBJ = {"preservation": 0, "income": 1, "growth": 2, "speculation": 3}

LEVEL_RULES = [  # (level, min knowledge, min options yrs, min objective, min liquid NW, margin)
    (1, 1, 0, 1, 50_000, False),
    (2, 2, 1, 1, 100_000, False),
    (3, 2, 2, 2, 250_000, True),
    (4, 3, 3, 3, 500_000, True),
]
FUTURES_RULE = {"min_liquid_nw": 250_000, "min_knowledge": 2, "min_tier": 3}


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


def eligibility(p: ClientProfile, rp: RiskProfile) -> Eligibility:
    x, f = p.experience, p.finances
    lnw = f.liquid_net_worth
    reasons, level = [], 0
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
            if margin and not x.wants_margin: missing.append("margin agreement")
            if missing:
                reasons.append(f"Level {lv} not granted: needs " + ", ".join(missing) + ".")
                break
            level = lv
        if level:
            reasons.insert(0, f"Approved for options Level {level}.")

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
    if fok:
        fr.append("Eligible for a futures account (hedging use only).")

    notes = []
    if p.is_senior:
        notes.append("Specified adult (65+): FINRA Rule 2165 temporary-hold procedures apply; "
                     "trusted contact " + ("on file." if p.identity.trusted_contact else "MISSING (Rule 4512)."))
    if p.identity.employed_by_broker_dealer:
        notes.append("Employed by a broker-dealer: Rule 3210 written consent and duplicate statements required.")
    if p.identity.control_person_public_co:
        notes.append("Control person: sales of company stock subject to Rule 144 / 10b5-1 plan.")
    if p.identity.politically_exposed:
        notes.append("Politically exposed person: enhanced due diligence under the bank's AML program.")
    if not (p.identity.identity_verified and p.identity.ofac_clear):
        notes.append("CIP verification or OFAC screening incomplete: account cannot be opened.")
    acc = p.is_accredited
    notes.append(("Accredited investor (Reg D 501(a))" if acc else "Not an accredited investor") +
                 ": private funds are outside this robo's universe; refer to an advisor if interested.")
    return Eligibility(options_level=level, options_reasons=reasons, futures_ok=fok, futures_reasons=fr,
                       accredited=acc, buffer_etf_ok=rp.tier >= 2 and not x.exclude_derivatives,
                       managed_futures_ok=rp.tier >= 3, notes=notes)


@dataclass
class OverlayPlan:
    overlays: list[tuple[Overlay, int, int]]     # (overlay, start year t, end year t)
    rationale: list[str]
    futures: dict | None
    not_recommended: list[str]
    single_stock_collar: dict | None = None


SEQUENCE_WINDOW = 7       # years before retirement when downside protection starts
SINGLE_STOCK_VOL = 0.30   # typical single-stock implied vol (assumption; replace with a live quote)


def recommend_overlays(p: ClientProfile, rp: RiskProfile, el: Eligibility, m: Market,
                       us_lc_value: float, horizon_t: int = 60) -> OverlayPlan:
    """Choose at most one equity overlay with its active window, plus optional hedges."""
    f, h = p.finances, p.household
    yrs_to_ret = f.retirement_age - h.age
    retired = p.identity.employment_status == "retired" or yrs_to_ret <= 0
    ovs, why, skip = [], [], []
    fut, ssc = None, None

    conc = f.concentrated_stock
    if conc > 0.10 * max(f.liquid_net_worth, 1):
        if el.options_level >= 1:
            sm = Market(spot=100.0, vix=None, rate=m.rate, div_yield=0.015, sigma_override=SINGLE_STOCK_VOL)
            col = make_collar(sm, 1.0, put_k=0.90)
            ssc = {"value": conc, "put_strike": col.put_k, "call_strike": col.call_k,
                   "put_cost": option_cost(sm, col.put_k, "put"), "max_loss": conc * (1 - col.put_k),
                   "max_gain": conc * (col.call_k - 1), "vol": SINGLE_STOCK_VOL}
            why.append(f"Concentrated single-stock position (${conc:,.0f}): a 1-year zero-cost collar (buy the "
                       f"{col.put_k:.0%} put, sell the {col.call_k:.0%} call, priced at {SINGLE_STOCK_VOL:.0%} implied "
                       "vol) caps the loss at 10% while a staged sale diversifies the position over 2-3 tax years.")
        else:
            why.append("Concentrated position: staged diversification over 2-3 tax years (no options approval "
                       "for a collar).")

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
                   "turns part of the upside into premium income (written in the IRA, premiums are not taxed "
                   "each year).")
    elif el.options_level == 0 and el.buffer_etf_ok and (retired or yrs_to_ret <= 15):
        ovs.append((make_buffer_etf(0.30), 0 if retired else start, end if not retired else horizon_t))
        why.append("No options approval: a 10%-buffer S&P 500 ETF on 30% of large cap gives similar protection "
                   "inside a fund (about 0.75% a year and forgoes dividends), during the sequence-risk window.")
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
        fut = futures_hedge(hedge_notional, m, micro=True)
        fut["tax_deferred"] = gain * 0.10 * f.cap_gains_rate
        why.append(f"Taxable account has ${gain:,.0f} of embedded gains. To trim equity by 10% of the taxable "
                   f"account without realising gains, short {fut['contracts']} {fut['contract']} contracts "
                   f"(~${fut['hedged_notional']:,.0f}); roll quarterly until gains can be offset or step up.")
    elif el.futures_ok:
        skip.append("Futures approved, but no hedge needed: the allocation can be adjusted inside "
                    "tax-advantaged accounts at no tax cost.")

    skip += ["Uncovered options, spreads and leveraged futures: outside robo policy at every level.",
             "Private equity / private credit (JPM's '60/40+' alternatives): illiquid and limited to "
             "accredited investors, so they are referred to an advisor, not automated."]
    return OverlayPlan(overlays=ovs, rationale=why, futures=fut, not_recommended=skip, single_stock_collar=ssc)


def reg_bi_checklist(p: ClientProfile, rp: RiskProfile, el: Eligibility) -> list[tuple[str, bool, str]]:
    """Reg BI / Advisers Act care-obligation items the recommendation file must evidence."""
    f = p.finances
    return [
        ("Investment profile complete (FINRA 2111 / Reg BI)", rp.n_answered >= 8 and f.investable >= 0,
         f"{rp.n_answered}/10 risk items answered; finances, goals, tax status, horizon, liquidity collected"),
        ("Identity verified (CIP) and OFAC screened", p.identity.identity_verified and p.identity.ofac_clear, ""),
        ("Trusted contact requested (FINRA 4512)", bool(p.identity.trusted_contact), p.identity.trusted_contact or "missing"),
        ("Risk willingness, capacity and need reconciled", True,
         f"binding constraint: {rp.binding}"),
        ("Recommended risk within stated loss tolerance", rp.recommended_equity <= rp.drawdown_equity + 1e-9,
         f"2008/2022 stress loss limit {rp.max_loss:.0%}"),
        ("Liquidity needs covered", f.cash >= 3 * f.monthly_essential_expenses,
         f"{f.cash / max(f.monthly_essential_expenses, 1):.1f} months of essentials in cash"),
        ("Derivatives only within approval level", True, f"options level {el.options_level}; "
         f"futures {'approved' if el.futures_ok else 'not approved'}"),
        ("Costs considered (reasonably available alternatives)", True,
         "core ETFs average ~0.05% expense ratio; overlays shown with costs"),
        ("Conflicts disclosed", True, "no proprietary funds or payment for order flow in the model"),
    ]
