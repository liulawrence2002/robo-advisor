"""Know-Your-Customer data model and risk questionnaire.

The fields mirror the "investment profile" that FINRA Rule 2111 and Reg BI require a
broker-dealer to make reasonable efforts to obtain (age, other investments, financial
situation and needs, tax status, investment objectives, experience, time horizon,
liquidity needs, risk tolerance), plus the account-opening items banks collect under
the Customer Identification Program (31 CFR 1023.220), FinCEN's customer due diligence
rule (beneficial ownership, source of wealth, PEP screening), FATCA tax residency,
FINRA Rule 4512 (trusted contact), Rule 3210 (accounts at other broker-dealers), Rule
2165 (seniors), Rule 144 / 10b5-1 (insiders), and options/futures account approval
(FINRA Rule 2360, NFA Rule 2-30).

The account-opening controls themselves (application completeness, customer risk rating,
the gate) live in robo/aml.py; sanctions name screening in robo/sanctions.py; the audit
log and its practitioner guide in robo/audit.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from datetime import date
from typing import Literal

Priority = Literal["need", "want", "wish"]
GoalKind = Literal["retirement", "home", "education", "bequest", "other"]

PROFILE_MAX_AGE_MONTHS = 36      # SEC 17a-3(a)(17): account record refreshed at least every 36 months


# ---------------------------------------------------------------- account-opening KYC
@dataclass
class Identity:
    full_name: str = "Client"
    citizenship: Literal["US citizen", "US resident alien", "Non-resident alien"] = "US citizen"
    tax_residency: Literal["US only", "US and another country", "Outside the US"] = "US only"   # FATCA
    state: str = "NY"
    account_type: Literal["individual", "joint", "IRA only", "trust", "LLC / corporation"] = "individual"
    beneficial_owners_verified: bool = True      # FinCEN CDD rule: owners >= 25% and a control person (entities)
    employment_status: Literal["employed", "self-employed", "retired", "not employed"] = "employed"
    occupation: str = ""
    employer: str = ""
    source_of_wealth: Literal["salary and savings", "business ownership", "inheritance or gift",
                              "sale of a business or property", "legal settlement", "crypto or trading gains",
                              "other"] = "salary and savings"
    expected_initial_funding: float = 0.0        # $ the client plans to deposit (CDD expected activity)
    employed_by_broker_dealer: bool = False      # FINRA Rule 3210 / 3110 consent letter
    control_person_public_co: bool = False       # officer/director/10% holder -> Rule 144 limits
    insider_ticker: str = ""                     # issuer of the controlled stock
    has_10b5_1_plan: bool = False                # pre-arranged trading plan (Rule 10b5-1)
    issuer_permits_hedging: bool = False         # most issuer policies ban hedging by insiders (Reg S-K 407(i))
    politically_exposed: bool = False            # enhanced due diligence
    pep_detail: str = ""                         # office held, country, self / family / close associate
    edd_cleared: bool = False                    # senior-management sign-off on enhanced due diligence
    identity_verified: bool = True               # CIP documentary / non-documentary verification
    id_method: Literal["documentary", "non-documentary", "pending"] = "documentary"
    ofac_clear: bool = True                      # sanctions screening result
    trusted_contact: str = ""                    # FINRA Rule 4512
    trusted_contact_declined: bool = False       # client was asked and declined (the rule requires the ask)
    power_of_attorney: str = ""                  # POA / guardian on file (seniors, diminished capacity)
    profile_date: str = ""                       # ISO date the profile was last confirmed ("" = today)
    form_crs_acknowledged: bool = True           # Form CRS / ADV Part 2 delivered and acknowledged

    # ---- account application (collected by the intake form). Full ID numbers are never stored:
    # only a masked value and whether the number passed validation.
    enforce_intake: bool = False                 # set by the intake form: every required item must be present to clear
    date_of_birth: str = ""                      # ISO date (CIP data element)
    tin_type: str = "SSN"                        # SSN / ITIN / foreign passport or government ID / applied for
    tin_masked: str = ""                         # "XXX-XX-4321"
    tin_valid: bool = False
    address_line: str = ""                       # residential street address (CIP: not a P.O. box)
    address_city: str = ""
    address_zip: str = ""
    address_country: str = "United States"
    mailing_address: str = ""                    # only if different
    phone: str = ""
    email: str = ""
    country_of_citizenship: str = "United States"
    other_tax_country: str = ""                  # second / foreign country of tax residence
    id_doc_type: str = "Driver's license"        # government photo ID used for documentary verification
    id_doc_masked: str = ""
    id_doc_issuer: str = ""                      # issuing state or country
    id_doc_expiry: str = ""                      # ISO date; must be unexpired
    joint_owner: dict = field(default_factory=dict)        # name, dob, tin_masked, tin_valid, tenancy, address_same, address
    entity: dict = field(default_factory=dict)             # legal_name, kind, ein_masked, ein_valid, formed_in, business,
                                                           # exempt, documents, certified
    beneficial_owners: list = field(default_factory=list)  # dicts: name, role, pct, dob, address, id_last4
    ira_beneficiary: str = ""
    # ---- customer due diligence: nature and purpose of the relationship, expected activity
    account_purpose: str = "Retirement and long-term investing"
    industry: str = "None of these"              # higher-risk occupation / industry, if any
    source_of_funds: str = "ACH from a bank account in the applicant's name"   # how the first deposit arrives
    expected_annual_deposits: float = 0.0
    expected_withdrawals: str = "Rarely (retirement or the goal date)"
    foreign_wires_expected: bool = False
    wire_countries: list = field(default_factory=list)
    pep_type: str = "No"                         # senior foreign political figure / family / associate / domestic
    bd_firm: str = ""                            # FINRA member the applicant (or spouse / dependent) is associated with
    trusted_contact_info: str = ""               # phone or email of the trusted contact
    # ---- tax certification, disclosures and signature
    tax_cert_signed: bool = False                # W-9 (US person) or W-8BEN (foreign person) certification
    backup_withholding: bool = False             # IRS has notified the applicant they are subject to it
    cip_notice_ack: bool = False                 # CIP customer notice shown before opening
    privacy_notice_ack: bool = False             # Regulation S-P initial privacy notice
    esign_consent: bool = False                  # E-SIGN consent to electronic delivery
    agreement_signed: bool = False               # advisory + account agreement (with arbitration disclosure)
    signature_name: str = ""                     # typed signature
    # ---- firm review (entered by the compliance desk, never by the applicant)
    id_verification: str = "Verified: documentary"
    id_theft_flags: list = field(default_factory=list)     # Regulation S-ID red flags raised by the verification vendor
    id_theft_resolved: bool = False
    ofac_hits: list = field(default_factory=list)          # potential matches from robo/sanctions.py (dicts)
    ofac_disposition: str = "Not reviewed"
    screening_source: str = ""                   # which list was screened and when
    adverse_media: str = "No hits"
    fincen_314a_match: bool = False
    sow_documented: bool = False                 # source of wealth / funds documents received
    edd_approver: str = ""
    principal_approved: bool = False             # FINRA 4512(a)(1)(D): principal accepts the account
    principal_name: str = ""
    analyst: str = ""


@dataclass
class Household:
    age: int = 40
    sex: Literal["male", "female"] = "female"
    health: Literal["excellent", "good", "average", "fair", "poor"] = "average"
    marital_status: Literal["single", "married", "divorced", "widowed"] = "single"
    spouse_age: int | None = None
    spouse_sex: Literal["male", "female"] | None = None
    spouse_health: Literal["excellent", "good", "average", "fair", "poor"] = "average"
    dependents: int = 0
    long_term_care_insured: bool = False
    life_insured: bool = True                    # term cover in force (human-capital protection)


@dataclass
class Finances:
    annual_income: float = 100_000          # client's gross labor income
    spouse_income: float = 0.0              # spouse's gross labor income (household = sum)
    income_real_growth: float = 0.01        # above inflation, until retirement
    income_stability: Literal["very stable", "stable", "variable", "volatile"] = "stable"
    savings_rate: float = 0.10              # employee contributions + other savings, % of income
    employer_match: float = 0.03            # employer contribution, % of income
    contrib_to_roth_share: float = 0.3      # share of retirement contributions going to Roth
    taxable: float = 0.0                    # brokerage
    traditional: float = 0.0                # 401(k)/403(b)/traditional IRA (pre-tax)
    roth: float = 0.0
    cash: float = 0.0                       # bank deposits (emergency fund)
    taxable_cost_basis_pct: float = 0.7     # basis / market value in taxable account
    concentrated_stock: float = 0.0         # single-stock position held inside taxable (subset)
    concentrated_ticker: str = ""           # for live pricing / fair value of the position
    home_value: float = 0.0
    mortgage: float = 0.0
    other_debt: float = 0.0
    other_debt_rate: float = 0.07           # interest rate on non-mortgage debt (pay-down vs invest test)
    monthly_essential_expenses: float = 4_000
    liquidity_need_12m: float = 0.0         # planned withdrawals within 12 months (FINRA 2111 liquidity needs)
    retirement_age: int = 65
    retirement_spending: float = 60_000     # desired annual spending in retirement, today's $
    essential_share_of_spending: float = 0.7
    social_security: float | None = None    # annual benefit at claim age (today's $); None = estimate
    spouse_social_security: float | None = None
    ss_claim_age: int = 67
    spouse_ss_claim_age: int = 67
    already_claiming_ss: bool = False
    pension: float = 0.0                    # annual, today's $, starts at retirement
    ordinary_tax_rate: float = 0.22         # marginal rate today (fed + state), used for Roth/traditional advice
    cap_gains_rate: float = 0.15
    state_tax_rate: float = 0.05            # flat state rate on ordinary income in the projection
    advisory_fee: float = 0.0025            # annual advisory fee charged on managed assets (Form CRS)

    @property
    def household_income(self) -> float:
        return self.annual_income + self.spouse_income

    @property
    def investable(self) -> float:
        return self.taxable + self.traditional + self.roth

    @property
    def liquid_net_worth(self) -> float:
        return self.investable + self.cash - self.other_debt

    @property
    def net_worth(self) -> float:
        return self.liquid_net_worth + self.home_value - self.mortgage

    @property
    def net_worth_ex_home(self) -> float:
        """Reg D 501(a): primary residence and its mortgage (up to the home's value) are excluded."""
        return self.investable + self.cash - self.other_debt - max(self.mortgage - self.home_value, 0.0)


@dataclass
class Goal:
    name: str
    kind: GoalKind
    amount: float                 # today's $; for retirement this is annual spending
    years_until: int              # years until first withdrawal
    duration_years: int = 1       # years over which the amount is spent (education = 4)
    priority: Priority = "need"
    funded_from_taxable: bool = True


@dataclass
class Experience:
    years_investing: int = 0
    knowledge: Literal["none", "limited", "good", "extensive"] = "limited"
    stock_years: int = 0                    # FINRA 2360(b)(16): experience by product
    bond_years: int = 0
    options_years: int = 0
    options_trades_per_year: int = 0
    futures_years: int = 0
    objective: Literal["preservation", "income", "growth", "speculation"] = "growth"
    options_disclosure_ack: bool = False    # received Characteristics & Risks of Standardized Options
    futures_disclosure_ack: bool = False    # CFTC Rule 1.55 risk disclosure acknowledged
    wants_margin: bool = False
    exclude_derivatives: bool = False       # client preference
    override_tiers: int = 0                 # client's own adjustment to the recommended tier (<0 = safer)
    override_reason: str = ""


@dataclass
class ClientProfile:
    identity: Identity = field(default_factory=Identity)
    household: Household = field(default_factory=Household)
    finances: Finances = field(default_factory=Finances)
    goals: list[Goal] = field(default_factory=list)
    experience: Experience = field(default_factory=Experience)
    risk_answers: dict[str, int] = field(default_factory=dict)
    gamble_answers: dict[str, int] = field(default_factory=dict)
    label: str = "Custom client"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def retirement_goal(self) -> Goal | None:
        return next((g for g in self.goals if g.kind == "retirement"), None)

    @property
    def is_retired(self) -> bool:
        """One definition for every module: stated retired, or past the retirement age."""
        return self.identity.employment_status == "retired" or self.household.age >= self.finances.retirement_age

    @property
    def effective_retirement_age(self) -> int:
        """Age from which the plan treats the client as retired (a retiree at 60 with a planned
        'retirement age' of 65 is retired now)."""
        return min(self.finances.retirement_age, self.household.age) if self.is_retired else self.finances.retirement_age

    @property
    def filing_status(self) -> str:
        return "mfj" if self.household.marital_status == "married" else "single"

    @property
    def is_accredited(self) -> bool:
        """Reg D Rule 501(a): $1M net worth ex-primary residence, or income > $200k ($300k joint)."""
        f, h = self.finances, self.household
        joint = h.marital_status == "married"
        income = f.household_income if joint else f.annual_income
        return f.net_worth_ex_home >= 1_000_000 or income > (300_000 if joint else 200_000)

    @property
    def is_qualified_client(self) -> bool:
        """Advisers Act Rule 205-3: $2.7M net worth since 29 June 2026 (SEC Rel. IA-6961; was $2.2M).
        Performance fees are allowed only for qualified clients."""
        return self.finances.net_worth_ex_home >= 2_700_000

    @property
    def is_senior(self) -> bool:
        """FINRA Rule 2165 'specified adult' (65+): enables temporary holds for exploitation."""
        return self.household.age >= 65

    @property
    def profile_age_months(self) -> int:
        if not self.identity.profile_date:
            return 0
        try:
            d = date.fromisoformat(self.identity.profile_date)
        except ValueError:
            return 0
        today = date.today()
        return max((today.year - d.year) * 12 + today.month - d.month, 0)

    def kyc_status(self) -> tuple[str, list[str]]:
        """Account-opening gate (the rules are in robo/aml.py: account_gate).

        'blocked'     the account cannot be opened (identity could not be verified, sanctions hit or
                      sanctioned jurisdiction, a minor, beneficial owners unverified)
        'incomplete'  the application is not in good order: a required item is missing. Only raised
                      when the profile comes from the intake form (identity.enforce_intake)
        'hold'        the profile and risk analysis can be shown, but no trades until compliance clears it
                      (potential list match, PEP without EDD sign-off, high-risk source of wealth, ...)
        'cleared'     normal processing
        """
        from .aml import account_gate
        return account_gate(self)


# ---------------------------------------------------------------- risk tolerance questionnaire
# Ten scored items in the tradition of the Grable-Lytton (1999) 13-item scale and
# FinaMetrica: general risk attitude, behaviour under loss, loss/gain framing,
# max-loss tolerance and portfolio choice. Wording is original. Each option's score
# is on 0..4; the total is rescaled to 0..100.
QUESTIONS: list[dict] = [
    {"id": "q1_self", "text": "Compared with others, how would you rate your willingness to take financial risks?",
     "options": ["Extremely low", "Low", "Average", "High", "Extremely high"], "scores": [0, 1, 2, 3, 4]},
    {"id": "q2_word", "text": "When you think of the word 'risk' in a financial context, what comes to mind first?",
     "options": ["Loss", "Uncertainty", "Opportunity", "Thrill"], "scores": [0, 1, 3, 4]},
    {"id": "q3_drop", "text": "Your portfolio falls 20% in a month along with the market. What do you do?",
     "options": ["Sell everything", "Sell some", "Hold and wait", "Buy more"], "scores": [0, 1, 3, 4]},
    {"id": "q4_history", "text": "In the 2020 COVID crash or the 2022 stock-and-bond selloff, what did you actually do?",
     "options": ["Sold most investments", "Sold some", "Held", "Bought more", "Was not invested"],
     "scores": [0, 1, 3, 4, 2]},
    {"id": "q5_maxloss", "text": "What is the largest one-year decline you could tolerate without abandoning your plan?",
     "options": ["0-5%", "5-10%", "10-20%", "20-30%", "More than 30%"], "scores": [0, 1, 2, 3, 4]},
    {"id": "q6_portfolio", "text": "Which one-year range of outcomes on $100,000 would you choose?",
     "options": ["$97k to $108k", "$92k to $115k", "$85k to $122k", "$78k to $130k", "$70k to $140k"],
     "scores": [0, 1, 2, 3, 4]},
    # q7/q8: equal-expected-value gain and loss frames (as in Grable & Lytton 1999)
    {"id": "q7_sure", "text": "You are given $10,000. Choose: keep a sure extra $5,000, or a 50% chance of an extra $10,000 (otherwise nothing).",
     "options": ["The sure $5,000", "The 50% chance of $10,000"], "scores": [0, 4]},
    {"id": "q8_loss_frame", "text": "You are given $20,000. Choose: a sure loss of $5,000, or a 50% chance of losing $10,000 (otherwise nothing).",
     "options": ["The sure $5,000 loss", "The 50% chance of losing $10,000"], "scores": [0, 4]},
    {"id": "q9_job", "text": "You are offered a new job: 20% higher pay but a real chance the role is eliminated within two years.",
     "options": ["Definitely stay", "Probably stay", "Probably take it", "Definitely take it"], "scores": [0, 1, 3, 4]},
    {"id": "q10_volatility", "text": "How do you feel when your investments swing in value day to day?",
     "options": ["Very anxious; I check constantly", "Uncomfortable", "Neutral", "Unbothered, it's expected"],
     "scores": [0, 1, 3, 4]},
]

# Max-loss answers (q5): upper end of each bracket, used as a hard cap on the portfolio's
# loss in the worse of the 2008 / 2022 stress scenarios.
MAX_LOSS_MAP = [0.05, 0.10, 0.20, 0.30, 0.45]

# Barsky, Juster, Kimball & Shapiro (1997) income-gamble ladder. The client holds a job
# paying their current income and is offered a 50/50 gamble: double the income, or cut it by X%.
# Answers bracket the coefficient of relative risk aversion (CRRA gamma).
GAMBLE_QUESTIONS = [
    {"id": "g1", "text": "50% chance your lifetime income doubles, 50% chance it is cut by one-third. Take it?",
     "options": ["No", "Yes"]},
    {"id": "g2_if_yes", "text": "Same, but the downside is a cut of one-half. Take it?", "options": ["No", "Yes"]},
    {"id": "g2_if_no", "text": "Same, but the downside is a cut of one-fifth. Take it?", "options": ["No", "Yes"]},
]


def _construct(cls, d: dict):
    """Build a dataclass from a dict, ignoring unknown keys and filling new fields with defaults
    (so audit files and cached profiles written by older versions still load)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def profile_from_dict(d: dict) -> ClientProfile:
    """Inverse of ClientProfile.to_dict (used for caching and the audit file)."""
    return ClientProfile(
        identity=_construct(Identity, d["identity"]), household=_construct(Household, d["household"]),
        finances=_construct(Finances, d["finances"]), goals=[_construct(Goal, g) for g in d["goals"]],
        experience=_construct(Experience, d["experience"]), risk_answers=dict(d["risk_answers"]),
        gamble_answers=dict(d["gamble_answers"]), label=d.get("label", "Custom client"))


def questionnaire_score(answers: dict[str, int]) -> tuple[float, int]:
    """Return (0..100 score, number of answered items)."""
    pts, n = 0.0, 0
    for q in QUESTIONS:
        if q["id"] in answers and answers[q["id"]] is not None:
            pts += q["scores"][answers[q["id"]]]
            n += 1
    if n == 0:
        return 50.0, 0
    return 100.0 * pts / (4 * n), n


def consistency_checks(answers: dict[str, int], age: int, objective: str) -> tuple[dict[str, int], list[str]]:
    """SEC IM Guidance 2017-02 asks robo-advisers to detect and address inconsistent answers.

    Returns (adjusted answers used for scoring, review flags). Adjustments always move toward
    the more conservative answer; nothing here raises risk.
    """
    a = {k: v for k, v in answers.items() if v is not None}
    flags = []
    if a.get("q1_self", 0) >= 3 and a.get("q5_maxloss", 9) <= 1:
        flags.append("Self-rated risk willingness is high, but the largest acceptable loss is 10% or less. The loss "
                     "limit governs; an advisor should confirm which answer reflects the client.")
    if a.get("q3_drop", 0) == 3 and a.get("q4_history", 9) <= 1:
        a["q3_drop"] = a["q4_history"]       # revealed behaviour outweighs the hypothetical
        flags.append("Says they would buy more in a 20% fall, but sold in 2020 or 2022. Scored on what they actually did.")
    if "q5_maxloss" in a and a.get("q6_portfolio", 0) > a["q5_maxloss"] + 1:
        a["q6_portfolio"] = a["q5_maxloss"]  # chosen outcome range implies losses beyond the stated limit
        flags.append("The chosen $100,000 outcome range implies a bigger loss than the stated maximum. The more "
                     "conservative answer is used.")
    score, _ = questionnaire_score(a)
    if score > 75 and (age >= 70 or objective == "preservation"):
        flags.append(f"Tolerance score {score:.0f}/100 with " + ("age 70+" if age >= 70 else "a preservation objective") +
                     ": suitability review by an advisor before an aggressive allocation is implemented.")
    return a, flags


def gamble_gamma(gamble: dict[str, int]) -> tuple[float | None, str]:
    """Map the income-gamble ladder to a CRRA gamma (geometric midpoint of the BJKS bracket)."""
    if "g1" not in gamble:
        return None, "not answered"
    if gamble["g1"] == 1:
        if gamble.get("g2_if_yes") == 1:
            return 0.75, "accepts 50% cut: gamma < 1 (BJKS category IV)"
        return 1.4, "accepts 1/3 but not 1/2 cut: 1 < gamma < 2 (category III)"
    if gamble.get("g2_if_no") == 1:
        return 2.7, "rejects 1/3, accepts 1/5 cut: 2 < gamma < 3.76 (category II)"
    return 5.5, "rejects even 1/5 cut: gamma > 3.76 (category I)"


def answers_from_labels(labels: dict[str, str]) -> dict[str, int]:
    """Convenience for building cases from option text."""
    out = {}
    for q in QUESTIONS + GAMBLE_QUESTIONS:
        if q["id"] in labels:
            out[q["id"]] = q["options"].index(labels[q["id"]])
    return out


def validate_goals(goals: list[Goal], horizon_years: int) -> tuple[list[Goal], list[str]]:
    """Clean user-entered goals: drop blanks, de-duplicate names, clamp dates to the plan horizon.
    Returns (clean goals, warnings)."""
    import math
    out, seen, warn = [], {}, []
    for g in goals:
        amt = float(g.amount) if g.amount is not None and not (isinstance(g.amount, float) and math.isnan(g.amount)) else 0.0
        yrs = g.years_until if g.years_until is not None and not (isinstance(g.years_until, float) and math.isnan(g.years_until)) else 0
        dur = g.duration_years if g.duration_years is not None and not (isinstance(g.duration_years, float) and math.isnan(g.duration_years)) else 1
        name = str(g.name).strip()
        if not name or amt <= 0:
            continue
        yrs, dur = int(max(yrs, 0)), int(max(dur, 1))
        if g.kind != "retirement" and yrs > horizon_years:
            warn.append(f"'{name}' is dated {yrs} years out, beyond the planning horizon ({horizon_years} years); "
                        f"it is moved to year {horizon_years}.")
            yrs = horizon_years
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        out.append(Goal(name, g.kind, amt, yrs, dur, g.priority, g.funded_from_taxable))
    return out, warn
