"""Account-opening controls under the Bank Secrecy Act: the Customer Identification Program,
customer due diligence, sanctions and list screening results, the customer AML risk rating and
the account gate that turns them into one decision.

The firm is modelled as an SEC-registered investment adviser with an affiliated introducing
broker-dealer, so the broker-dealer rules are the operative ones:

  31 CFR 1023.220   Customer Identification Program (name, date of birth, address, identification
                    number; documentary / non-documentary verification; customer notice; records)
  31 CFR 1010.230   beneficial owners of legal-entity customers (25% ownership prong, control prong)
  31 CFR 1023.210   AML program, including ongoing customer due diligence (customer risk profile,
                    monitoring, updating customer information) and FINRA Rule 3310
  31 CFR 1010.620   enhanced due diligence for senior foreign political figures (private banking)
  31 CFR part 501   OFAC blocking / rejecting, reports within 10 business days, 10-year records
  31 CFR 1023.320   suspicious activity reports ($5,000; 30 days; no tipping off)
  31 CFR 1010.520   FinCEN 314(a) information requests (search, report a match within 14 days)
  17 CFR 248.201    Regulation S-ID identity-theft red flags
  FINRA Rule 4512   customer account information; a principal's signature accepting the account

The adviser itself has no BSA program duty yet: FinCEN's investment-adviser AML rule (89 FR 72156)
was postponed from 1 January 2026 to 1 January 2028, and the adviser CIP proposal is not final.
Customer risk scoring and review cycles are industry convention, not a regulatory formula.

Nothing here contacts a verification vendor: identity verification, adverse-media and 314(a)
results are entered on the firm-review panel, the way an analyst would record a vendor's answer.
Sanctions name screening is real (robo/sanctions.py) and its hits are stored on the profile.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# ------------------------------------------------------------------ reference data
US_STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
             "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR",
             "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR", "VI", "GU", "AE / AP / AA (military)"]

US = "United States"
COUNTRIES = [US, "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
             "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
             "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "British Virgin Islands",
             "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Cayman Islands",
             "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Democratic Republic)",
             "Congo (Republic)", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czechia", "Denmark", "Djibouti",
             "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
             "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar",
             "Greece", "Grenada", "Guatemala", "Guernsey", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hong Kong",
             "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica",
             "Japan", "Jersey", "Jordan", "Kazakhstan", "Kenya", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
             "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Madagascar", "Malawi",
             "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Moldova", "Monaco",
             "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand",
             "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Panama",
             "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
             "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Saudi Arabia",
             "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
             "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
             "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago",
             "Tunisia", "Turkey", "Turkmenistan", "Uganda", "Ukraine", "Ukraine (Crimea, Donetsk or Luhansk region)",
             "United Arab Emirates", "United Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen",
             "Zambia", "Zimbabwe"]

# Jurisdictions under comprehensive (territory-wide) OFAC sanctions: no account can be opened for a
# person resident or located there. See SANCTIONS_ASOF for the date this list was checked.
SANCTIONED_JURISDICTIONS = {"Cuba", "Iran", "North Korea", "Ukraine (Crimea, Donetsk or Luhansk region)"}
SANCTIONS_ASOF = "November 2026"
# FATF public statements. Black list = call for action; grey list = increased monitoring.
FATF_BLACK = {"North Korea", "Iran", "Myanmar"}
FATF_GREY = {"Angola", "Bolivia", "Bosnia and Herzegovina", "Cameroon", "Haiti", "Iraq", "Kenya", "Kuwait", "Laos", "Lebanon",
             "Nepal", "Papua New Guinea", "South Sudan", "Syria", "Venezuela", "Vietnam", "British Virgin Islands", "Yemen"}
FATF_ASOF = "October 2026 plenary"
# Other jurisdictions the house policy treats as higher risk (targeted US sanctions programs or secrecy havens).
HOUSE_HIGH_RISK = {"Russia", "Belarus", "Afghanistan", "Libya", "Somalia", "Sudan", "Nicaragua", "Zimbabwe",
                   "Central African Republic", "Mali", "Eritrea"}

CITIZENSHIP = ["US citizen", "US resident alien", "Non-resident alien"]
TAX_RESIDENCY = ["US only", "US and another country", "Outside the US"]
ACCOUNT_TYPES = ["individual", "joint", "IRA only", "trust", "LLC / corporation"]
ACCOUNT_LABEL = {"individual": "Individual", "joint": "Joint (two owners)", "IRA only": "IRA (traditional or Roth)",
                 "trust": "Trust", "LLC / corporation": "LLC, corporation or partnership"}
ENTITY_TYPES = ("trust", "LLC / corporation")
TIN_TYPES = ["SSN", "ITIN", "Foreign passport / government ID", "Applied for (not yet issued)"]
ID_DOC_TYPES = ["Driver's license", "State ID card", "US passport", "Permanent resident card", "Foreign passport",
                "Other government photo ID"]
PURPOSES = ["Retirement and long-term investing", "Saving for a goal (home, education)", "Income in retirement",
            "Managing an inheritance or windfall", "Business or entity reserves", "Other"]
SOURCE_OF_WEALTH = ["salary and savings", "business ownership", "inheritance or gift", "sale of a business or property",
                    "legal settlement", "crypto or trading gains", "other"]
SOW_POINTS = {"salary and savings": 0, "business ownership": 5, "inheritance or gift": 5, "sale of a business or property": 5,
              "legal settlement": 12, "crypto or trading gains": 18, "other": 18}
SOW_NEEDS_DOCS = ("legal settlement", "crypto or trading gains", "other")
# how the first deposit arrives: (risk points, needs review before the deposit is accepted)
FUNDING = {
    "ACH from a bank account in the applicant's name": (0, False),
    "Transfer from another brokerage (ACATS)": (0, False),
    "401(k) or IRA rollover": (0, False),
    "Personal check": (4, False),
    "Domestic wire from the applicant's account": (6, False),
    "International wire": (18, False),
    "Sale of crypto assets through an exchange": (18, False),
    "Funds from a third party": (25, True),
    "Cash, money orders or cashier's checks": (30, True),
}
WITHDRAWALS = ["Rarely (retirement or the goal date)", "A few times a year", "Monthly income", "Frequent (weekly or more)"]
INDUSTRIES = ["None of these", "Cash-intensive business (restaurant, retail, car wash, laundromat)",
              "Money services business or crypto exchange / ATM", "Casino, gaming or card club", "Cannabis-related business",
              "Dealer in precious metals, stones, art or antiques", "Arms, defense or dual-use exports",
              "Import / export or freight forwarding", "Non-profit or charity operating abroad",
              "Law, accounting or company-formation services (client funds)"]
PEP_TYPES = ["No", "Senior foreign political figure", "Immediate family of a senior foreign political figure",
             "Close associate of a senior foreign political figure", "Senior US (domestic) public official"]
ID_VERIFY = ["Verified: documentary", "Verified: non-documentary", "Pending", "Unable to verify"]
OFAC_DISPOSITIONS = ["Not reviewed", "False positive: cleared", "True match: block and report"]
ADVERSE_MEDIA = ["No hits", "Hit: under review", "Hit: reviewed, not material", "Hit: material (financial crime)"]
ID_THEFT_FLAGS = ["Address does not match the credit file", "SSN issued before date of birth or belongs to a deceased person",
                  "Fraud alert or credit freeze on the file", "ID document appears altered or photo mismatch",
                  "Phone / email / device linked to other applicants", "Applicant cannot answer identity questions"]
JOINT_TENANCY = ["Joint tenants with right of survivorship", "Tenants in common", "Community property"]
ENTITY_KINDS = ["LLC", "Corporation", "Partnership", "Statutory trust"]
BO_EXEMPT = ["Not exempt", "Publicly traded company (or 51%+ owned subsidiary)", "Regulated financial institution",
             "SEC-registered investment company or adviser", "Government entity"]

RISK_MEDIUM, RISK_HIGH = 30, 60
REVIEW_CYCLE = {"low": 36, "medium": 24, "high": 12}      # months between periodic KYC refreshes
ADULT_AGE = 18
SAR_THRESHOLD, CTR_THRESHOLD, TRAVEL_RULE = 5_000, 10_000, 3_000
CIP_RETENTION_YEARS, OFAC_RETENTION_YEARS, BOOKS_RETENTION_YEARS = 5, 10, 6
TEST_SSN_PREFIX = "98765432"       # 987-65-4320..4329 are reserved for advertising: safe sample numbers


# ------------------------------------------------------------------ validation helpers
def digits(s: str) -> str:
    return re.sub(r"\D", "", str(s or ""))


def validate_tin(raw: str, kind: str = "SSN") -> tuple[bool, str]:
    """(valid, reason). SSN: 9 digits, area not 000/666/9xx, group not 00, serial not 0000.
    ITIN: 9 digits starting with 9 and a valid fourth/fifth-digit range. The advertising range
    987-65-432x is accepted as a sample number so the demo never needs a real SSN."""
    if kind.startswith("Foreign"):
        s = re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))
        return (len(s) >= 6, "" if len(s) >= 6 else "passport / ID number looks too short")
    if kind.startswith("Applied"):
        return True, ""
    d = digits(raw)
    if len(d) != 9:
        return False, "must be 9 digits"
    if d.startswith(TEST_SSN_PREFIX):
        return True, ""
    area, grp, serial = d[:3], d[3:5], d[5:]
    if kind == "ITIN":
        ok = area[0] == "9" and (50 <= int(grp) <= 65 or 70 <= int(grp) <= 88 or 90 <= int(grp) <= 92 or 94 <= int(grp) <= 99)
        return ok, "" if ok else "an ITIN starts with 9 and has 50-65, 70-88, 90-92 or 94-99 as its 4th-5th digits"
    if area in ("000", "666") or area[0] == "9":
        return False, "not a valid SSN area number (000, 666 and 900-999 are never issued)"
    if grp == "00" or serial == "0000":
        return False, "not a valid SSN (group 00 or serial 0000)"
    if len(set(d)) == 1 or d in ("123456789", "078051120"):
        return False, "a well-known invalid SSN"
    return True, ""


def mask_tin(raw: str, kind: str = "SSN") -> str:
    """Only the last four characters are kept anywhere in the app."""
    if kind.startswith("Applied"):
        return "applied for"
    s = re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))
    if len(s) < 4:
        return ""
    return f"XXX-XX-{s[-4:]}" if not kind.startswith("Foreign") else f"XXXXX{s[-4:]}"


def mask_doc(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", str(raw or ""))
    return f"XXXX{s[-4:]}" if len(s) >= 4 else ""


def parse_date(iso: str) -> date | None:
    try:
        return date.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return None


def age_on(dob_iso: str, today: date | None = None) -> int | None:
    d = parse_date(dob_iso)
    if d is None:
        return None
    t = today or date.today()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def is_po_box(addr: str) -> bool:
    return bool(re.search(r"\b(p\.?\s*o\.?\s*box|post office box|pmb)\b", str(addr or ""), flags=re.I))


def valid_zip(z: str, country: str = US) -> bool:
    return bool(re.fullmatch(r"\d{5}(-\d{4})?", str(z or "").strip())) if country == US else bool(str(z or "").strip())


def valid_email(e: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", str(e or "").strip()))


def valid_phone(p: str) -> bool:
    return 10 <= len(digits(p)) <= 15


def countries_of(i) -> list[tuple[str, str]]:
    """(role, country) for every country the customer is tied to."""
    out = [("residence", i.address_country or US), ("citizenship", i.country_of_citizenship or US)]
    if i.tax_residency != "US only" and i.other_tax_country:
        out.append(("tax residence", i.other_tax_country))
    for c in i.wire_countries or []:
        out.append(("expected wires", c))
    if i.entity.get("formed_in") and i.entity.get("formed_in") not in US_STATES:
        out.append(("entity formation", i.entity["formed_in"]))
    return out


def is_legal_entity_customer(i) -> bool:
    """31 CFR 1010.230(e): corporations, LLCs, partnerships and statutory trusts are legal-entity
    customers; an ordinary (non-statutory) trust is not, and neither is a natural person."""
    return i.account_type == "LLC / corporation"


def parties(i) -> list[tuple[str, str, str, bool]]:
    """Everyone screened against sanctions lists: (role, name, date of birth, is an entity)."""
    out = [("Applicant", i.full_name, i.date_of_birth, False)]
    if i.account_type == "joint" and i.joint_owner.get("name"):
        out.append(("Joint owner", i.joint_owner["name"], i.joint_owner.get("dob", ""), False))
    if i.account_type in ENTITY_TYPES and i.entity.get("legal_name"):
        out.append(("Trust" if i.account_type == "trust" else "Entity", i.entity["legal_name"], "", True))
    for b in i.beneficial_owners or []:
        if b.get("name"):
            out.append((f"{b.get('role') or 'Beneficial owner'}", b["name"], b.get("dob", ""), False))
    return out


# ------------------------------------------------------------------ application completeness
STEPS = ["Applicant", "Account & funding", "Regulatory questions", "Agreements & signature"]


def missing_items(p) -> dict[str, list[str]]:
    """What the application still needs, by intake step. Empty lists = in good order."""
    i = p.identity
    m: dict[str, list[str]] = {s: [] for s in STEPS}
    a = m["Applicant"]
    if len(str(i.full_name).split()) < 2:
        a.append("Full legal name (first and last)")
    age = age_on(i.date_of_birth)
    if age is None:
        a.append("Date of birth")
    if not i.tin_masked:
        a.append("Taxpayer identification number (SSN or ITIN)" if i.citizenship != "Non-resident alien"
                 else "Identification number (ITIN, or passport number and country)")
    elif not i.tin_valid:
        a.append(f"{i.tin_type}: the number entered is not valid")
    if i.citizenship != "Non-resident alien" and i.tin_type.startswith("Foreign"):
        a.append("A US person must give a taxpayer identification number (SSN or ITIN), not a foreign ID")
    if not i.address_line.strip() or not i.address_city.strip():
        a.append("Residential street address and city")
    elif is_po_box(i.address_line):
        a.append("A residential street address (a P.O. box can be the mailing address only)")
    if not valid_zip(i.address_zip, i.address_country):
        a.append("ZIP / postal code")
    if not valid_phone(i.phone):
        a.append("Phone number")
    if not valid_email(i.email):
        a.append("Email address")
    if i.id_verification != ID_VERIFY[1]:          # a photo ID is needed unless verified non-documentarily
        exp = parse_date(i.id_doc_expiry)
        if not i.id_doc_masked:
            a.append(f"{i.id_doc_type} number (documentary verification)")
        if exp is None:
            a.append("ID expiration date")
        elif exp < date.today():
            a.append("An unexpired government ID (the one on file has expired)")
    if not i.occupation.strip() and i.employment_status in ("employed", "self-employed"):
        a.append("Occupation")
    if not i.employer.strip() and i.employment_status == "employed":
        a.append("Employer")

    f = m["Account & funding"]
    if i.account_type == "joint":
        j = i.joint_owner
        if len(str(j.get("name", "")).split()) < 2:
            f.append("Joint owner: full legal name")
        if age_on(j.get("dob", "")) is None:
            f.append("Joint owner: date of birth")
        if not j.get("tin_masked") or not j.get("tin_valid"):
            f.append("Joint owner: valid SSN or ITIN")
        if not j.get("address_same") and not str(j.get("address", "")).strip():
            f.append("Joint owner: residential address")
    if i.account_type in ENTITY_TYPES:
        e = i.entity
        what = "Trust" if i.account_type == "trust" else "Entity"
        if not str(e.get("legal_name", "")).strip():
            f.append(f"{what}: legal name")
        if not e.get("ein_masked") or not e.get("ein_valid"):
            f.append(f"{what}: taxpayer identification number (EIN, 9 digits)")
        if not str(e.get("formed_in", "")).strip():
            f.append(f"{what}: state or country of formation" if what == "Entity" else "Trust: governing state")
        if not e.get("documents"):
            f.append("Trust certification or trust agreement pages (trustees and their powers)" if what == "Trust"
                     else "Formation documents (articles / certificate of formation and resolution naming authorized persons)")
    if is_legal_entity_customer(i) and i.entity.get("exempt", BO_EXEMPT[0]) == BO_EXEMPT[0]:
        bos = [b for b in (i.beneficial_owners or []) if str(b.get("name", "")).strip()]
        if not any("control" in str(b.get("role", "")).lower() for b in bos):
            f.append("Beneficial ownership: one control person (an executive or senior manager)")
        for b in bos:
            lack = [lab for k, lab in (("dob", "date of birth"), ("address", "address"), ("id_last4", "ID number"))
                    if not str(b.get(k, "")).strip()]
            if lack:
                f.append(f"Beneficial owner {b['name']}: " + ", ".join(lack))
        if not i.entity.get("certified"):
            f.append("Beneficial-ownership certification signed by the person opening the account")
    if i.account_type == "IRA only" and not i.ira_beneficiary.strip():
        f.append("IRA: primary beneficiary designation")
    if i.expected_initial_funding <= 0:
        f.append("Expected initial funding amount")

    r = m["Regulatory questions"]
    if not i.tax_cert_signed:
        r.append("Tax certification (Form W-9 for a US person, Form W-8BEN for a foreign person)")
    if i.tax_residency != "US only" and not i.other_tax_country:
        r.append("Other country of tax residence")
    if i.pep_type != PEP_TYPES[0] and not i.pep_detail.strip():
        r.append("Political exposure: office held, country and relationship")
    if i.control_person_public_co and not i.insider_ticker.strip():
        r.append("Public-company affiliation: the issuer's ticker")
    if i.employed_by_broker_dealer and not i.employer.strip():
        r.append("Broker-dealer affiliation: name of the employing firm")
    if not i.trusted_contact.strip() and not i.trusted_contact_declined:
        r.append("Trusted contact, or tick that you decline to name one")

    s = m["Agreements & signature"]
    for ok_, label in ((i.cip_notice_ack, "Customer identification notice"), (i.form_crs_acknowledged, "Form CRS and Form ADV Part 2A"),
                       (i.privacy_notice_ack, "Privacy notice"), (i.esign_consent, "Consent to electronic delivery"),
                       (i.agreement_signed, "Advisory and account agreement")):
        if not ok_:
            s.append(f"{label}: not yet acknowledged")
    if not i.signature_name.strip():
        s.append("Typed signature")
    elif not same_name(i.signature_name, i.full_name):
        s.append("The typed signature must match the legal name")
    return m


def same_name(a: str, b: str) -> bool:
    """A typed signature matches the legal name, ignoring case, spacing and punctuation."""
    norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())
    return bool(norm(a)) and norm(a) == norm(b)


# ------------------------------------------------------------------ customer risk rating
@dataclass
class RiskRating:
    score: int
    level: str                                  # low / medium / high
    factors: list[tuple[str, int, str]]         # (factor, points, why)
    edd_required: bool
    edd_reasons: list[str] = field(default_factory=list)
    review_months: int = 36

    def table(self) -> list[list[str]]:
        return [[f, f"+{pts}" if pts else "0", why] for f, pts, why in self.factors]


def risk_rating(p) -> RiskRating:
    """Additive customer risk score (0-100) over the factors regulators expect a risk-based program
    to weigh: customer type, geography, political exposure, occupation, source of wealth, how the
    account is funded and whether the expected activity fits the stated finances."""
    i, f = p.identity, p.finances
    fac: list[tuple[str, int, str]] = []
    edd: list[str] = []

    pts = {"individual": 0, "joint": 0, "IRA only": 0, "trust": 10, "LLC / corporation": 15}.get(i.account_type, 0)
    fac.append(("Customer type", pts, ACCOUNT_LABEL.get(i.account_type, i.account_type)
                + ("" if pts == 0 else ": ownership and control are less transparent than for a natural person")))

    pts = {"US citizen": 0, "US resident alien": 5, "Non-resident alien": 15}.get(i.citizenship, 0)
    fac.append(("Residency status", pts, i.citizenship))

    geo, geo_why = 0, []
    for role, c in countries_of(i):
        if c in FATF_BLACK or c in SANCTIONED_JURISDICTIONS:
            geo = max(geo, 40); geo_why.append(f"{c} ({role}): sanctioned or FATF call-for-action jurisdiction")
        elif c in FATF_GREY:
            geo = max(geo, 20); geo_why.append(f"{c} ({role}): FATF increased-monitoring list")
        elif c in HOUSE_HIGH_RISK:
            geo = max(geo, 20); geo_why.append(f"{c} ({role}): higher-risk jurisdiction under house policy")
        elif c != US:
            geo = max(geo, 5); geo_why.append(f"{c} ({role})")
    fac.append(("Geography", geo, "; ".join(dict.fromkeys(geo_why)) or "United States only"))
    if geo >= 20:
        edd.append("Ties to a higher-risk jurisdiction")

    if i.politically_exposed:
        foreign = i.pep_type != PEP_TYPES[4]
        pts = 35 if foreign else 15
        fac.append(("Political exposure", pts, (i.pep_type if i.pep_type != PEP_TYPES[0] else "Politically exposed person")
                    + (f": {i.pep_detail}" if i.pep_detail else "")))
        if foreign:
            edd.append("Senior foreign political figure, family member or close associate")
    else:
        fac.append(("Political exposure", 0, "None declared; screening found none"))

    pts = 15 if i.industry != INDUSTRIES[0] else 0
    fac.append(("Occupation / industry", pts, i.industry if pts else (i.occupation or i.employment_status)))

    pts = SOW_POINTS.get(i.source_of_wealth, 18)
    fac.append(("Source of wealth", pts, i.source_of_wealth + (" (documentation required)" if i.source_of_wealth in SOW_NEEDS_DOCS else "")))

    pts, _ = FUNDING.get(i.source_of_funds, (0, False))
    fac.append(("Funding method", pts, i.source_of_funds))

    means = max(f.net_worth, 0.0) + f.household_income
    pts, why = 0, "In line with stated income and net worth"
    if i.expected_initial_funding > 0 and means > 0 and i.expected_initial_funding > 1.5 * means:
        pts, why = 20, (f"Initial funding ${i.expected_initial_funding:,.0f} exceeds 1.5x stated net worth plus income "
                        f"(${means:,.0f})")
    elif i.expected_initial_funding > 0 and means <= 0:
        pts, why = 20, "Initial funding with no stated income or net worth"
    if i.foreign_wires_expected:
        pts += 10; why += "; international wires expected"
    if i.expected_withdrawals == WITHDRAWALS[3]:
        pts += 5; why += "; frequent withdrawals expected (little investment purpose)"
    fac.append(("Expected activity", pts, why))

    pts = {ADVERSE_MEDIA[0]: 0, ADVERSE_MEDIA[1]: 15, ADVERSE_MEDIA[2]: 5, ADVERSE_MEDIA[3]: 35}.get(i.adverse_media, 0)
    fac.append(("Adverse media", pts, i.adverse_media))
    if i.adverse_media == ADVERSE_MEDIA[3]:
        edd.append("Material adverse media")

    score = int(min(sum(x[1] for x in fac), 100))
    level = "high" if score >= RISK_HIGH else "medium" if score >= RISK_MEDIUM else "low"
    if level == "high" and "Overall rating is high" not in edd:
        edd.append("Overall rating is high")
    return RiskRating(score=score, level=level, factors=fac, edd_required=bool(edd), edd_reasons=edd,
                      review_months=REVIEW_CYCLE[level])


# ------------------------------------------------------------------ the account gate
STATUS_LABEL = {"cleared": "Approved", "hold": "Pending review", "incomplete": "Information needed", "blocked": "Declined"}
STATUS_TITLE = {"cleared": "Account approved and open for trading",
                "hold": "Application received: restricted until compliance clears it",
                "incomplete": "Application not in good order: items outstanding",
                "blocked": "Account cannot be opened"}


def unresolved_hits(i) -> list[dict]:
    return list(i.ofac_hits or []) if i.ofac_disposition == OFAC_DISPOSITIONS[0] else []


def account_gate(p) -> tuple[str, list[str]]:
    """One decision from every control.

    'blocked'     a hard stop: sanctions match or sanctioned jurisdiction, identity that could not be
                  verified, an applicant under 18, unverified beneficial owners of a legal entity
    'incomplete'  the application is not in good order (a required item is missing); nothing is opened,
                  the plan can still be shown as a proposal
    'hold'        opened but restricted: a review is pending (potential list match, enhanced due
                  diligence, source-of-funds documents, identity-theft red flags, stale profile)
    'cleared'     normal processing
    """
    i = p.identity
    why: list[str] = []
    # ---- hard stops
    if not i.ofac_clear or i.ofac_disposition == OFAC_DISPOSITIONS[2]:
        why.append("Sanctions screening hit (OFAC): account opening is prohibited. Block or reject the property, report "
                   "to OFAC within 10 business days and escalate to the AML officer (31 CFR 501.603-.604).")
    for role, c in countries_of(i):
        if c in SANCTIONED_JURISDICTIONS and role in ("residence", "citizenship", "entity formation"):
            why.append(f"{c} ({role}) is under comprehensive OFAC sanctions: the firm cannot open or service this account "
                       "without a specific license.")
    if not i.identity_verified or i.id_verification == ID_VERIFY[3]:
        why.append("Identity could not be verified (CIP, 31 CFR 1023.220(a)(2)): do not open the account; consider whether "
                   "a suspicious activity report is warranted.")
    elif i.id_method == "pending" and not i.enforce_intake:
        why.append("Identity not verified (CIP, 31 CFR 1023.220): no account until documentary or non-documentary "
                   "verification is complete.")
    age = age_on(i.date_of_birth)
    if age is not None and age < ADULT_AGE:
        why.append(f"Applicant is {age}: a minor cannot open this account (a custodial UTMA/UGMA account is opened by an adult).")
    if i.account_type in ENTITY_TYPES and not i.beneficial_owners_verified:
        why.append("Legal-entity account without verified beneficial owners (FinCEN CDD rule, 31 CFR 1010.230)."
                   if i.account_type != "trust" else
                   "Trust account whose trustees could not be identified and verified (CIP, 31 CFR 1023.220(a)(2)(ii)(C)).")
    if why:
        return "blocked", why

    # ---- not in good order
    if i.enforce_intake:
        miss = missing_items(p)
        for step, items in miss.items():
            why += [f"**{step}.** {t}" for t in items]
        if why:
            return "incomplete", why

    # ---- pending reviews
    status = "cleared"
    hits = unresolved_hits(i)
    if hits:
        status = "hold"
        why.append(f"{len(hits)} potential sanctions match(es) awaiting analyst review: "
                   + "; ".join(h.get("summary", h.get("listed_name", "")) for h in hits[:2])
                   + ". No funding or trading until each is cleared as a false positive or confirmed.")
    if i.enforce_intake and i.id_verification == ID_VERIFY[2]:
        status = "hold"
        why.append("Identity verification pending: the account is restricted (no trading, no withdrawals) until documentary "
                   "or non-documentary verification completes; close it if verification fails (31 CFR 1023.220(a)(2)(iii)).")
    if i.id_theft_flags and not i.id_theft_resolved:
        status = "hold"
        why.append("Identity-theft red flag(s) not yet resolved (Regulation S-ID, 17 CFR 248.201): "
                   + "; ".join(i.id_theft_flags) + ".")
    if i.fincen_314a_match:
        status = "hold"
        why.append("Positive match to a FinCEN 314(a) request: report the match to FinCEN through the Secure Information "
                   "Sharing System within 14 days, keep the request confidential and take no further step on the account "
                   "without the AML officer's decision (31 CFR 1010.520).")
    rr = risk_rating(p)
    if i.politically_exposed and not i.edd_cleared:
        status = "hold"
        why.append("Politically exposed person: enhanced due diligence and senior-management approval required before "
                   "the account is funded.")
    elif rr.edd_required and not i.edd_cleared and i.enforce_intake:
        status = "hold"
        why.append(f"Customer risk rating {rr.level} ({rr.score}/100): enhanced due diligence must be completed and approved "
                   f"({'; '.join(rr.edd_reasons)}).")
    means = max(p.finances.net_worth, 0.0) + p.finances.household_income
    if i.enforce_intake and i.expected_initial_funding > 1.5 * means and not i.sow_documented:
        status = "hold"
        why.append(f"Planned initial funding of ${i.expected_initial_funding:,.0f} is more than 1.5x the stated net worth plus "
                   f"income (${means:,.0f}): obtain a document showing where the money comes from before accepting it "
                   "(FINRA Regulatory Notice 19-18: activity inconsistent with the customer's financial profile).")
    if i.source_of_wealth in SOW_NEEDS_DOCS and not i.sow_documented:
        status = "hold"
        why.append(f"Source of wealth '{i.source_of_wealth}' is higher risk under the bank's customer risk rating: "
                   "document the source of funds before trading.")
    if FUNDING.get(i.source_of_funds, (0, False))[1] and not i.sow_documented:
        status = "hold"
        why.append(f"Funding method '{i.source_of_funds}' is not accepted without review: obtain the payer's identity and "
                   "the reason for the payment, or have the client fund from an account in their own name.")
    if i.adverse_media in (ADVERSE_MEDIA[1], ADVERSE_MEDIA[3]) and not i.edd_cleared:
        status = "hold"
        why.append(f"Adverse media: {i.adverse_media.lower()}. The AML officer decides whether the relationship is acceptable.")
    if i.citizenship == "Non-resident alien" and i.tax_residency == "US only":
        status = "hold"
        why.append("Non-resident alien with US-only tax residency is inconsistent: collect W-8BEN "
                   "and the country of tax residence (FATCA).")
    if i.enforce_intake and not i.principal_approved:
        status = "hold"
        why.append("Awaiting acceptance by a principal: FINRA Rule 4512(a)(1)(D) requires the signature of the partner, "
                   "officer or manager who accepts the account before it is opened for trading.")
    if p.is_senior and not i.trusted_contact and not i.trusted_contact_declined:
        why.append("Specified adult without a trusted contact on file: FINRA 4512 requires a reasonable effort to "
                   "obtain one (does not block the account).")
    from .kyc import PROFILE_MAX_AGE_MONTHS
    if p.profile_age_months > PROFILE_MAX_AGE_MONTHS:
        status = "hold"
        why.append(f"Investment profile is {p.profile_age_months} months old: refresh before "
                   f"relying on it (SEC Rule 17a-3: {PROFILE_MAX_AGE_MONTHS}-month cycle).")
    return status, why


def monitoring_notes(p) -> list[str]:
    """Reporting and recordkeeping triggers the funding and first activity will be watched for."""
    i = p.identity
    amt = i.expected_initial_funding
    out = [f"Expected activity baseline: ${amt:,.0f} initial funding ({i.source_of_funds}); "
           f"about ${i.expected_annual_deposits:,.0f} a year in further deposits; withdrawals {i.expected_withdrawals.lower()}. "
           "Activity well outside this baseline is reviewed."]
    if "wire" in i.source_of_funds.lower() and amt >= TRAVEL_RULE:
        out.append(f"Funds transfer of ${TRAVEL_RULE:,} or more: keep the originator and beneficiary record and pass it "
                   "along with the payment (recordkeeping and travel rules, 31 CFR 1010.410(e)-(f)).")
    if i.source_of_funds.startswith("Cash"):
        out.append(f"Currency over ${CTR_THRESHOLD:,} in one business day requires a Currency Transaction Report within 15 days "
                   f"(31 CFR 1010.311); monetary instruments of ${TRAVEL_RULE:,}-${CTR_THRESHOLD:,} bought with cash are a "
                   "structuring red flag. House policy: the firm does not accept cash.")
    out.append(f"Suspicious activity involving ${SAR_THRESHOLD:,} or more is reported on a SAR within 30 calendar days of "
               "detection (60 if no suspect is identified); the client is never told (31 CFR 1023.320).")
    return out
