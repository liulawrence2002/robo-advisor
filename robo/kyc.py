"""Know-Your-Customer data model and risk questionnaire.

The fields mirror the "investment profile" that FINRA Rule 2111 and Reg BI require a
broker-dealer to make reasonable efforts to obtain (age, other investments, financial
situation and needs, tax status, investment objectives, experience, time horizon,
liquidity needs, risk tolerance), plus the account-opening items banks collect under
the Customer Identification Program (31 CFR 1023.220), FINRA Rule 4512 (trusted
contact), Rule 3210 (accounts at other broker-dealers), and options/futures account
approval (FINRA Rule 2360, NFA Rule 2-30).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Priority = Literal["need", "want", "wish"]
GoalKind = Literal["retirement", "home", "education", "bequest", "other"]


# ---------------------------------------------------------------- account-opening KYC
@dataclass
class Identity:
    full_name: str = "Client"
    citizenship: Literal["US citizen", "US resident alien", "Non-resident alien"] = "US citizen"
    state: str = "NY"
    employment_status: Literal["employed", "self-employed", "retired", "not employed"] = "employed"
    occupation: str = ""
    employed_by_broker_dealer: bool = False      # FINRA Rule 3210 / 3110 consent letter
    control_person_public_co: bool = False       # officer/director/10% holder -> Rule 144 limits
    politically_exposed: bool = False            # enhanced due diligence
    identity_verified: bool = True               # CIP documentary / non-documentary verification
    ofac_clear: bool = True                      # sanctions screening result
    trusted_contact: str = ""                    # FINRA Rule 4512


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
    home_value: float = 0.0
    mortgage: float = 0.0
    other_debt: float = 0.0
    monthly_essential_expenses: float = 4_000
    retirement_age: int = 65
    retirement_spending: float = 60_000     # desired annual spending in retirement, today's $
    essential_share_of_spending: float = 0.7
    social_security: float | None = None    # annual benefit at claim age (today's $); None = estimate
    spouse_social_security: float | None = None
    ss_claim_age: int = 67
    spouse_ss_claim_age: int = 67
    already_claiming_ss: bool = False
    pension: float = 0.0                    # annual, today's $, starts at retirement
    ordinary_tax_rate: float = 0.22         # marginal federal+state on ordinary income
    cap_gains_rate: float = 0.15

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
    options_years: int = 0
    futures_years: int = 0
    objective: Literal["preservation", "income", "growth", "speculation"] = "growth"
    options_disclosure_ack: bool = False    # received Characteristics & Risks of Standardized Options
    futures_disclosure_ack: bool = False    # CFTC Rule 1.55 risk disclosure acknowledged
    wants_margin: bool = False
    exclude_derivatives: bool = False       # client preference


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
    def is_accredited(self) -> bool:
        """Reg D Rule 501(a): $1M net worth ex-primary residence, or income > $200k ($300k joint)."""
        f, h = self.finances, self.household
        joint = h.marital_status == "married"
        nw_ex_home = f.liquid_net_worth
        income = f.household_income if joint else f.annual_income
        return nw_ex_home >= 1_000_000 or income > (300_000 if joint else 200_000)

    @property
    def is_senior(self) -> bool:
        """FINRA Rule 2165 'specified adult' (65+): enables temporary holds for exploitation."""
        return self.household.age >= 65


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


def profile_from_dict(d: dict) -> ClientProfile:
    """Inverse of ClientProfile.to_dict (used for caching and the audit file)."""
    return ClientProfile(
        identity=Identity(**d["identity"]), household=Household(**d["household"]),
        finances=Finances(**d["finances"]), goals=[Goal(**g) for g in d["goals"]],
        experience=Experience(**d["experience"]), risk_answers=dict(d["risk_answers"]),
        gamble_answers=dict(d["gamble_answers"]), label=d.get("label", "Custom client"))


def questionnaire_score(answers: dict[str, int]) -> tuple[float, int]:
    """Return (0..100 score, number of answered items)."""
    pts, n = 0.0, 0
    for q in QUESTIONS:
        if q["id"] in answers:
            pts += q["scores"][answers[q["id"]]]
            n += 1
    if n == 0:
        return 50.0, 0
    return 100.0 * pts / (4 * n), n


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
