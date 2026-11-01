"""The three assignment clients. The assignment fixes age and the headline facts; every
other number below is an explicit, documented assumption (shown in the app and paper)."""
from __future__ import annotations

from .kyc import ClientProfile, Experience, Finances, Goal, Household, Identity, answers_from_labels


def case_a() -> ClientProfile:
    """(a) 30-year-old saver: buy a house in 2 years, retire in 35 years."""
    return ClientProfile(
        label="A: 30-year-old saver (house in 2 yrs, retire in 35)",
        identity=Identity(full_name="Case A (age 30)", occupation="Financial analyst", state="NY",
                          trusted_contact="Parent (on file)"),
        household=Household(age=30, sex="female", health="good", marital_status="single"),
        finances=Finances(
            annual_income=95_000, income_real_growth=0.015, income_stability="stable",
            savings_rate=0.15, employer_match=0.04, contrib_to_roth_share=0.5,
            taxable=35_000, traditional=22_000, roth=8_000, cash=15_000, taxable_cost_basis_pct=0.9,
            other_debt=12_000, monthly_essential_expenses=3_800,
            retirement_age=65, retirement_spending=70_000, essential_share_of_spending=0.65,
            ss_claim_age=67, ordinary_tax_rate=0.28, cap_gains_rate=0.15),
        goals=[
            Goal("Home down payment", "home", amount=58_500, years_until=2, priority="need"),
            Goal("Retirement", "retirement", amount=70_000, years_until=35, duration_years=30, priority="need"),
        ],
        experience=Experience(years_investing=4, knowledge="good", objective="growth"),
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
        identity=Identity(full_name="Case B (age 55)", occupation="Operations manager", state="OH",
                          trusted_contact="Sibling (on file)"),
        household=Household(age=55, sex="male", health="average", marital_status="divorced"),
        finances=Finances(
            annual_income=110_000, income_real_growth=0.0, income_stability="stable",
            savings_rate=0.10, employer_match=0.04, contrib_to_roth_share=0.2,
            taxable=25_000, traditional=160_000, roth=15_000, cash=20_000, taxable_cost_basis_pct=0.8,
            home_value=380_000, mortgage=150_000, monthly_essential_expenses=4_500,
            retirement_age=67, retirement_spending=66_000, essential_share_of_spending=0.7,
            ss_claim_age=67, ordinary_tax_rate=0.24, cap_gains_rate=0.15),
        goals=[Goal("Retirement", "retirement", amount=66_000, years_until=12, duration_years=30, priority="need")],
        experience=Experience(years_investing=20, knowledge="good", options_years=2, objective="growth",
                              options_disclosure_ack=True),
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
        identity=Identity(full_name="Case C (age 70)", employment_status="retired", occupation="Retired engineer",
                          state="FL", trusted_contact="Adult child (on file)"),
        household=Household(age=70, sex="male", health="good", marital_status="married",
                            spouse_age=68, spouse_sex="female", spouse_health="good"),
        finances=Finances(
            annual_income=0, income_stability="very stable", savings_rate=0.0, employer_match=0.0,
            taxable=600_000, traditional=1_200_000, roth=200_000, cash=60_000, taxable_cost_basis_pct=0.55,
            concentrated_stock=250_000, home_value=900_000, monthly_essential_expenses=6_000,
            retirement_age=65, retirement_spending=110_000, essential_share_of_spending=0.6,
            social_security=42_000, spouse_social_security=24_000, already_claiming_ss=True,
            ordinary_tax_rate=0.24, cap_gains_rate=0.15),
        goals=[
            Goal("Retirement", "retirement", amount=110_000, years_until=0, duration_years=30, priority="need"),
            Goal("Grandchildren's college", "education", amount=120_000, years_until=6, duration_years=4,
                 priority="want"),
            Goal("Legacy to children", "bequest", amount=750_000, years_until=25, priority="wish"),
        ],
        experience=Experience(years_investing=40, knowledge="extensive", options_years=10, futures_years=3,
                              objective="income", options_disclosure_ack=True, futures_disclosure_ack=True),
        risk_answers=answers_from_labels({
            "q1_self": "Average", "q2_word": "Uncertainty", "q3_drop": "Hold and wait", "q4_history": "Held",
            "q5_maxloss": "10-20%", "q6_portfolio": "$85k to $122k", "q7_sure": "The sure $5,000",
            "q8_loss_frame": "The sure $5,000 loss", "q9_job": "Definitely stay",
            "q10_volatility": "Neutral"}),
        gamble_answers=answers_from_labels({"g1": "No", "g2_if_no": "No"}),
    )


CASES = {"A": case_a, "B": case_b, "C": case_c}
