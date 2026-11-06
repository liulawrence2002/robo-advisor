"""The three assignment clients. The assignment fixes age and the headline facts; every
other number below is an explicit, documented assumption (shown in the app and paper).

Each case also carries a complete, fictional account application (name, date of birth, address,
masked sample SSN from the 987-65-432x advertising range, ID document, disclosures, signature and
the firm's review results) so it passes the account gate the same way a real approved client would.
"""
from __future__ import annotations

from datetime import date

from .kyc import ClientProfile, Experience, Finances, Goal, Household, Identity, answers_from_labels


def _dob(age: int, month: int, day: int) -> str:
    """A date of birth that makes the client exactly `age` today, so the cases do not drift with the calendar."""
    t = date.today()
    return date(t.year - age - ((t.month, t.day) < (month, day)), month, day).isoformat()


def _application(last4: str, phone: str, email: str, street: str, city: str, zip_: str, id_state: str) -> dict:
    """Application fields every approved sample client has: valid sample SSN, unexpired driver's license,
    disclosures acknowledged, tax certification signed, identity verified and a principal's acceptance."""
    return dict(enforce_intake=True, tin_type="SSN", tin_masked=f"XXX-XX-{last4}", tin_valid=True,
                address_line=street, address_city=city, address_zip=zip_, phone=phone, email=email,
                id_doc_type="Driver's license", id_doc_masked=f"XXXX{last4}", id_doc_issuer=id_state,
                id_doc_expiry=date(date.today().year + 3, 6, 30).isoformat(),
                tax_cert_signed=True, cip_notice_ack=True, privacy_notice_ack=True, esign_consent=True,
                agreement_signed=True, id_verification="Verified: documentary", principal_approved=True,
                principal_name="D. Ramirez (Series 24)", analyst="K. Osei (AML analyst)")


def case_a() -> ClientProfile:
    """(a) 30-year-old saver: buy a house in 2 years, retire in 35 years."""
    return ClientProfile(
        label="A: 30-year-old saver (house in 2 yrs, retire in 35)",
        identity=Identity(full_name="Maya R. Thompson", occupation="Financial analyst", employer="Hudson Valley Bank",
                          state="NY", trusted_contact="Linda Thompson (mother)", trusted_contact_info="(518) 555-0117",
                          expected_initial_funding=35_000, expected_annual_deposits=14_000,
                          date_of_birth=_dob(30, 4, 18), signature_name="Maya R. Thompson",
                          account_purpose="Saving for a goal (home, education)",
                          **_application("4321", "(718) 555-0142", "maya.thompson@example.com",
                                         "214 Clinton Street, Apt 3B", "Brooklyn", "11201", "NY")),
        household=Household(age=30, sex="female", health="good", marital_status="single", life_insured=False),
        finances=Finances(
            annual_income=95_000, income_real_growth=0.015, income_stability="stable",
            savings_rate=0.15, employer_match=0.04, contrib_to_roth_share=0.5,
            taxable=35_000, traditional=22_000, roth=8_000, cash=15_000, taxable_cost_basis_pct=0.9,
            other_debt=12_000, other_debt_rate=0.055, monthly_essential_expenses=3_800,
            retirement_age=65, retirement_spending=70_000, essential_share_of_spending=0.65,
            ss_claim_age=67, ordinary_tax_rate=0.28, cap_gains_rate=0.15, state_tax_rate=0.06),
        goals=[
            Goal("Home down payment", "home", amount=58_500, years_until=2, priority="need"),
            Goal("Retirement", "retirement", amount=70_000, years_until=35, duration_years=30, priority="need"),
        ],
        experience=Experience(years_investing=4, knowledge="good", stock_years=4, objective="growth"),
        risk_answers=answers_from_labels({
            "q1_self": "High", "q2_word": "Opportunity", "q3_drop": "Hold and wait", "q4_history": "Held",
            "q5_maxloss": "20-30%", "q6_portfolio": "$78k to $130k", "q7_sure": "The 50% chance of $10,000",
            "q8_loss_frame": "The 50% chance of losing $10,000", "q9_job": "Probably take it",
            "q10_volatility": "Neutral"}),
        gamble_answers=answers_from_labels({"g1": "Yes", "g2_if_yes": "No"}),
    )


def case_b() -> ClientProfile:
    """(b) 55-year-old retirement investor with $200K saved."""
    return ClientProfile(
        label="B: 55-year-old with $200K saved",
        identity=Identity(full_name="Daniel J. Kowalski", occupation="Operations manager", employer="Buckeye Precision Parts",
                          state="OH", trusted_contact="Anna Kowalski (sister)", trusted_contact_info="(614) 555-0188",
                          expected_initial_funding=25_000, expected_annual_deposits=11_000,
                          date_of_birth=_dob(55, 8, 23), signature_name="Daniel J. Kowalski",
                          source_of_funds="Transfer from another brokerage (ACATS)",
                          **_application("4323", "(614) 555-0163", "dan.kowalski@example.com",
                                         "1847 Maple Ridge Drive", "Columbus", "43215", "OH")),
        household=Household(age=55, sex="male", health="average", marital_status="divorced"),
        finances=Finances(
            annual_income=110_000, income_real_growth=0.0, income_stability="stable",
            savings_rate=0.10, employer_match=0.04, contrib_to_roth_share=0.2,
            taxable=25_000, traditional=160_000, roth=15_000, cash=20_000, taxable_cost_basis_pct=0.8,
            home_value=380_000, mortgage=150_000, monthly_essential_expenses=4_500,
            retirement_age=67, retirement_spending=66_000, essential_share_of_spending=0.7,
            ss_claim_age=67, ordinary_tax_rate=0.24, cap_gains_rate=0.15, state_tax_rate=0.035),
        goals=[Goal("Retirement", "retirement", amount=66_000, years_until=12, duration_years=30, priority="need")],
        experience=Experience(years_investing=20, knowledge="good", stock_years=20, bond_years=10, options_years=2,
                              options_trades_per_year=6, objective="growth", options_disclosure_ack=True),
        risk_answers=answers_from_labels({
            "q1_self": "Average", "q2_word": "Uncertainty", "q3_drop": "Hold and wait", "q4_history": "Held",
            "q5_maxloss": "10-20%", "q6_portfolio": "$85k to $122k", "q7_sure": "The sure $5,000",
            "q8_loss_frame": "The 50% chance of losing $10,000", "q9_job": "Probably stay",
            "q10_volatility": "Uncomfortable"}),
        gamble_answers=answers_from_labels({"g1": "No", "g2_if_no": "Yes"}),
    )


def case_c() -> ClientProfile:
    """(c) 70-year-old retiree with $2 million saved."""
    return ClientProfile(
        label="C: 70-year-old retiree with $2M",
        identity=Identity(full_name="Walter E. Brandt", employment_status="retired", occupation="Retired engineer",
                          employer="Honeywell (former)", account_type="joint", state="FL",
                          trusted_contact="Karen Brandt-Lee (daughter)", trusted_contact_info="(239) 555-0171",
                          power_of_attorney="Spouse (durable POA)",
                          source_of_wealth="salary and savings", expected_initial_funding=2_000_000,
                          date_of_birth=_dob(70, 3, 30), signature_name="Walter E. Brandt",
                          account_purpose="Income in retirement", expected_withdrawals="Monthly income",
                          source_of_funds="Transfer from another brokerage (ACATS)",
                          joint_owner={"name": "Helen M. Brandt", "dob": _dob(68, 7, 19), "tin_masked": "XXX-XX-4322",
                                       "tin_valid": True, "tenancy": "Joint tenants with right of survivorship",
                                       "address_same": True, "address": ""},
                          **_application("4324", "(239) 555-0126", "walter.brandt@example.com",
                                         "5120 Gulf Shore Boulevard N", "Naples", "34103", "FL")),
        household=Household(age=70, sex="male", health="good", marital_status="married",
                            spouse_age=68, spouse_sex="female", spouse_health="good", long_term_care_insured=False),
        finances=Finances(
            annual_income=0, income_stability="very stable", savings_rate=0.0, employer_match=0.0,
            taxable=600_000, traditional=1_200_000, roth=200_000, cash=60_000, taxable_cost_basis_pct=0.55,
            concentrated_stock=250_000, concentrated_ticker="HON", home_value=900_000, monthly_essential_expenses=6_000,
            retirement_age=65, retirement_spending=110_000, essential_share_of_spending=0.6,
            social_security=42_000, spouse_social_security=24_000, already_claiming_ss=True,
            ordinary_tax_rate=0.24, cap_gains_rate=0.15, state_tax_rate=0.0),
        goals=[
            Goal("Retirement", "retirement", amount=110_000, years_until=0, duration_years=30, priority="need"),
            Goal("Grandchildren's college", "education", amount=120_000, years_until=6, duration_years=4,
                 priority="want"),
            Goal("Legacy to children", "bequest", amount=750_000, years_until=25, priority="wish"),
        ],
        experience=Experience(years_investing=40, knowledge="extensive", stock_years=40, bond_years=30,
                              options_years=10, options_trades_per_year=12, futures_years=3,
                              objective="income", options_disclosure_ack=True, futures_disclosure_ack=True),
        risk_answers=answers_from_labels({
            "q1_self": "Average", "q2_word": "Uncertainty", "q3_drop": "Hold and wait", "q4_history": "Held",
            "q5_maxloss": "10-20%", "q6_portfolio": "$85k to $122k", "q7_sure": "The sure $5,000",
            "q8_loss_frame": "The sure $5,000 loss", "q9_job": "Definitely stay",
            "q10_volatility": "Neutral"}),
        gamble_answers=answers_from_labels({"g1": "No", "g2_if_no": "No"}),
    )


CASES = {"A": case_a, "B": case_b, "C": case_c}
