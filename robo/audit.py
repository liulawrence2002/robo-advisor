"""Compliance audit log for one account application, with a practitioner guide for every line.

Each control the gate runs writes one log line: a code, the check, its result and the evidence.
GUIDE holds the instructions that go with each code: what the line records, the rule it evidences,
who acts on it, what to do for each possible result, and how long the record is kept. The app
shows the guide under each line; the JSON audit record and the client file carry the same text.

Results
  PASS    the control is satisfied; nothing to do
  INFO    a fact recorded for the file; nothing to do
  OPEN    the applicant still owes an item; the application is not in good order
  REVIEW  a person at the firm must act and record a decision; the decision line (DEC-900) says
          whether the account is restricted meanwhile
  FAIL    a hard stop; the account cannot be opened

Lines are hash-chained (each hash covers the previous one), the way an audit-trail system under
SEA Rule 17a-4(f)(2)(i)(A) shows that no entry was altered or removed after the fact.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from . import aml
from .kyc import PROFILE_MAX_AGE_MONTHS, ClientProfile

RESULTS = [("PASS", "control satisfied"), ("INFO", "recorded, no action"), ("OPEN", "applicant owes an item"),
           ("REVIEW", "the firm must act and record a decision"), ("FAIL", "hard stop; account not opened")]

R_CIP = "5 years after the account is closed (31 CFR 1023.220(a)(3)(ii))"
R_VER = "5 years after the record is made (31 CFR 1023.220(a)(3)(ii))"
R_BOOKS = "6 years after the account is closed or the record is superseded (SEA Rule 17a-4(c), (e)(5))"
R_OFAC = "10 years (31 CFR 501.601, as amended effective 12 March 2025)"
R_ADV = "5 years, the first 2 in the office (Advisers Act Rule 204-2)"

# code -> what the line records, the rule, who acts, retention, and the instruction for each result
GUIDE: dict[str, dict] = {
    "APP-001": dict(
        check="Application received",
        what="Opens the file: who applied, for which account registration, through which channel and for what purpose. Every later line refers back to this application.",
        rule="FINRA Rule 4512(a)(1) (customer account information); SEA Rule 17a-3(a)(17) (account record).",
        owner="System. No one needs to act.",
        retain=R_BOOKS,
        actions={"INFO": "Nothing to do. Check that the registration type is the one the client intended: it decides which later lines apply (joint owner, beneficial owners, IRA beneficiary)."}),
    "APP-002": dict(
        check="Account record fields",
        what="The contact and employment details the account record must carry beyond the identification items: phone, email, occupation and employer.",
        rule="FINRA Rule 4512(a)(2) (occupation, employer, association with another member, before the first trade settles); SEA Rule 17a-3(a)(17)(i)(A) (telephone number, employment status).",
        owner="Applicant supplies; operations checks.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do.",
                 "OPEN": "Ask the applicant for the items in the evidence. If a customer refuses to give employment or financial details, note the refusal in the file; a refusal is recorded, not ignored."}),
    "CIP-010": dict(
        check="Customer identification notice",
        what="The applicant was shown the notice that federal law requires the firm to obtain, verify and record identifying information, before the account was opened.",
        rule="31 CFR 1023.220(a)(5): adequate notice must be given before an account is opened.",
        owner="Applicant acknowledges; operations confirms the notice version shown.",
        retain=R_VER,
        actions={"PASS": "Nothing to do. The acknowledgement time and notice version are on file.",
                 "OPEN": "Do not open the account. Send the applicant back to Agreements & signature to acknowledge the notice; if onboarding by phone, read it aloud and note the date and time."}),
    "CIP-020": dict(
        check="Identifying information collected",
        what="The four data elements the Customer Identification Program requires before an account opens: legal name, date of birth, a residential street address and an identification number.",
        rule="31 CFR 1023.220(a)(2)(i)(A). A P.O. box is not a residential address; a US person gives a taxpayer identification number; a non-US person may give a passport number and country instead.",
        owner="Applicant supplies; operations checks completeness.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. All four elements are present for every owner.",
                 "OPEN": "Do not open the account. Ask the applicant for the items listed in the evidence. If the only gap is a street address, an APO/FPO number or the address of next of kin is acceptable.",
                 "FAIL": "The applicant is under 18. Decline the application; an adult may open a custodial (UTMA/UGMA) account for the minor instead."}),
    "CIP-030": dict(
        check="Identification number validated",
        what="A format and range check on the SSN, ITIN or foreign ID number. Only the last four digits are kept in this file; the full number sits with the custodian.",
        rule="31 CFR 1023.220(a)(2)(i)(A)(4); IRS Form W-9 (taxpayer identification number); Regulation S-ID red flags for numbers never issued.",
        owner="System validates; operations follows up on exceptions.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. A format check is not proof the number belongs to the applicant: that is line CIP-040.",
                 "OPEN": "Ask the applicant to re-enter the number. A number in a never-issued range is an identity-theft red flag: record it on CIP-050 if it happens twice.",
                 "REVIEW": "The applicant has applied for a number but does not have one yet. Obtain a copy of the application (Form SS-5 or W-7) before opening, diarise 60 days to collect the number, and restrict the account to closing transactions if it has not arrived (31 CFR 1023.220(a)(2)(i)(B))."}),
    "CIP-040": dict(
        check="Identity verification",
        what="Whether the firm formed a reasonable belief that it knows the applicant's true identity, and by which method: documentary (unexpired government photo ID) or non-documentary (credit bureau or public-database comparison, contacting the customer).",
        rule="31 CFR 1023.220(a)(2)(ii)-(iii). Verification must be completed within a reasonable time before or after opening; the firm's procedures say what the customer may do meanwhile, when to close, and when to consider a suspicious activity report.",
        owner="Verification vendor returns the result; the onboarding analyst records and resolves it.",
        retain=R_VER + ". Keep the document type, number, place of issue and issue / expiry dates, or the method and result of the non-documentary check, and how any discrepancy was resolved.",
        actions={"PASS": "Nothing to do. For an online account, a documentary pass should be paired with a non-documentary check because the customer was not seen in person.",
                 "OPEN": "The photo ID details are missing or the document has expired. Ask the applicant for an unexpired government-issued photo ID (number, issuer, expiration date), or verify by a non-documentary method and record it on the firm-review panel.",
                 "REVIEW": "Verification is still running. Keep the account restricted: deposits may be received, but no trading, withdrawals or third-party transfers. If it is not complete in 30 days, close the account and return funds to their source.",
                 "FAIL": "Identity could not be verified. Do not open the account (or close it and return funds to the source). Send the file to the AML officer to decide whether a suspicious activity report is warranted; do not tell the applicant a report is being considered."}),
    "CIP-050": dict(
        check="Identity-theft red flags",
        what="Warning signs raised during verification that someone may be using another person's identity: address or SSN discrepancies, a fraud alert or credit freeze, altered documents, shared phone or device.",
        rule="Regulation S-ID, 17 CFR 248.201 (identity theft prevention program for covered accounts).",
        owner="Onboarding analyst; escalate to the fraud team when a flag cannot be explained.",
        retain=R_VER,
        actions={"PASS": "Nothing to do. No red flags were raised, or each one was resolved and the resolution is on file.",
                 "REVIEW": "Do not fund or trade the account. Contact the applicant on a phone number from an independent source (not the one on the application), ask for a second form of ID, and record how each flag was resolved. If it cannot be resolved, decline and refer to the AML officer."}),
    "CIP-060": dict(
        check="Additional owners and authorized persons",
        what="Each joint owner is a customer in their own right, and a trust or company acts through trustees or authorized persons. This line records that every one of them was identified to the same standard as the applicant.",
        rule="31 CFR 1023.100(d) (each joint owner is a customer); 31 CFR 1023.220(a)(2)(ii)(C) (persons with authority or control over a non-individual account); FINRA Rule 4512(a)(1)(E).",
        owner="Applicant supplies; operations checks.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. All owners and authorized persons are identified and screened (see SAN-200).",
                 "OPEN": "Collect the missing items in the evidence for the joint owner, trustee or authorized person. No account until every owner is identified.",
                 "INFO": "Single-owner account: this line does not apply."}),
    "CDD-100": dict(
        check="Nature and purpose of the relationship",
        what="Why the client is opening the account and what activity to expect: purpose, initial funding, further deposits and how often money will leave. This is the baseline that later monitoring compares actual activity against.",
        rule="31 CFR 1023.210(b)(5)(i) (understand the nature and purpose of customer relationships to develop a customer risk profile); FINRA Rule 3310(f).",
        owner="Applicant supplies; the analyst checks it is plausible for this client.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. The expected activity is on file and fits the stated finances.",
                 "OPEN": "Ask for the expected initial funding and purpose. Without a baseline there is nothing to monitor against.",
                 "REVIEW": "The planned funding is large against the client's stated income and net worth. Ask where the money is coming from and obtain a document that shows it (bank or brokerage statement, sale contract, estate letter) before accepting the deposit."}),
    "CDD-110": dict(
        check="Source of wealth and source of funds",
        what="Source of wealth is how the client came to have their assets over time; source of funds is where this particular deposit comes from and how it arrives. Higher-risk answers need documents.",
        rule="31 CFR 1023.210(b)(5) (risk-based customer due diligence); FINRA Regulatory Notice 19-18 (red flags: refusal to identify a legitimate source of funds, third-party funding, cash equivalents).",
        owner="Onboarding analyst; AML officer for exceptions.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. The source is ordinary for the client's profile, or supporting documents were received.",
                 "REVIEW": "Obtain documents before any trade: for a legal settlement, the settlement statement; for crypto or trading gains, exchange statements showing acquisition and sale; for third-party funds, the payer's identity and relationship and the reason they are paying. The firm does not accept cash. Tick 'Source documents received' on the firm-review panel when done."}),
    "CDD-120": dict(
        check="Beneficial owners of a legal entity",
        what="For a company, LLC or partnership: each individual who owns 25% or more, plus one individual who controls it, identified and verified like a customer. The 25% figure also matters for sanctions, because an entity owned 50% or more by blocked persons is itself blocked.",
        rule="31 CFR 1010.230 (beneficial ownership requirements for legal-entity customers); certification on the Appendix A form or an equivalent. Ordinary trusts are not legal-entity customers; the trustee is identified under CIP instead.",
        owner="The person opening the account certifies; the analyst verifies each owner.",
        retain="Identification and certification: " + R_CIP + ". Verification records: " + R_VER,
        actions={"PASS": "Nothing to do. Owners and the control person are identified, verified and screened, and the certification is signed.",
                 "INFO": "Not a legal-entity customer (a natural person, or an ordinary trust), or an excluded entity such as a listed company or regulated financial institution. Record the exclusion basis and move on.",
                 "OPEN": "Collect the missing owner details in the evidence, name one control person, and have the person opening the account sign the certification.",
                 "FAIL": "An owner could not be verified. Do not open the account. If the entity will not disclose who owns or controls it, treat that as a red flag and refer to the AML officer."}),
    "CDD-130": dict(
        check="Customer risk rating",
        what="A score from 0 to 100 built from customer type, residency, geography, political exposure, occupation, source of wealth, funding method, expected activity and adverse media. Under 30 is low, 30 to 59 medium, 60 and over high. The rating sets how often the file is refreshed.",
        rule="31 CFR 1023.210(b)(5)(i) (customer risk profile); FINRA Rule 3310(f). The scoring weights and review cycles are house policy: no regulation prescribes them.",
        owner="System scores; the AML officer may override upward with a written reason, never downward without one.",
        retain=R_CIP,
        actions={"INFO": "Nothing to do now. Diarise the periodic review shown in the evidence (36 months for low, 24 for medium), and re-score at once on a trigger event: change of address abroad, new political exposure, unexpected wires, adverse media.",
                 "REVIEW": "High risk. Enhanced due diligence is required before the account is funded (line CDD-140), monitoring thresholds are tightened, and the file is reviewed every 12 months."}),
    "CDD-140": dict(
        check="Enhanced due diligence",
        what="The extra work a high-risk or politically exposed customer requires: documented source of wealth and funds, the reason for using this firm, adverse-media search, and approval by senior management.",
        rule="31 CFR 1010.620 (enhanced scrutiny of accounts for senior foreign political figures); 31 CFR 1023.210(b)(5); FinCEN / banking agencies joint statement of 21 August 2020 (due diligence on politically exposed persons is risk-based).",
        owner="AML analyst prepares the memo; a senior manager who is not the relationship owner approves.",
        retain=R_CIP,
        actions={"INFO": "Not required for this customer.",
                 "PASS": "Completed and approved; the approver is named in the evidence. Apply enhanced monitoring and review the file every 12 months.",
                 "REVIEW": "Prepare the memo: who the customer is, source of wealth with documents, source of this deposit, why the relationship makes sense, screening results. Obtain senior-management approval and record the approver on the firm-review panel. No funding or trading until then."}),
    "SAN-200": dict(
        check="OFAC sanctions name screening",
        what="Every owner, joint owner, beneficial owner and entity name was compared with OFAC's Specially Designated Nationals list (and its aliases) using fuzzy matching. A potential match is a similar name, not a finding that the applicant is the listed person.",
        rule="OFAC regulations, 31 CFR chapter V; blocked-property and rejected-transaction reports within 10 business days, 31 CFR 501.603-.604. This screen covers the SDN list; a production program also screens the Consolidated non-SDN list and re-screens when either list changes.",
        owner="System screens; an AML analyst dispositions every potential match; the AML officer handles a true match.",
        retain=R_OFAC + ". Keep the list version, the match details and the reason each potential match was cleared.",
        actions={"PASS": "Nothing to do. Either no name came close to a listed name, or an analyst compared the secondary identifiers and cleared each potential match as a false positive (the reason is in the evidence).",
                 "REVIEW": "Compare the applicant with the listed person on date of birth, place of birth, nationality, address and ID numbers. If they differ, choose 'False positive: cleared' on the firm-review panel and note which identifier ruled it out. If you cannot rule it out, call the OFAC hotline (1-800-540-6322) before doing anything else. No funding or trading while this is open.",
                 "FAIL": "True match. Do not open the account and do not return any funds already received: block them in a segregated interest-bearing account, file the blocked-property report with OFAC within 10 business days, tell the AML officer today, and do not process any instruction from the customer. Blocked property is also reported annually by 30 September."}),
    "SAN-210": dict(
        check="Sanctioned and higher-risk jurisdictions",
        what="The countries the customer is tied to (residence, citizenship, tax residence, entity formation, expected wires) checked against comprehensively sanctioned territories, the FATF call-for-action and increased-monitoring lists, and the house high-risk list.",
        rule="OFAC country programs (Cuba, Iran, North Korea, and the Crimea, Donetsk and Luhansk regions of Ukraine); FATF public statements; FINRA Regulatory Notice 19-18 (ties to high-risk or secrecy jurisdictions).",
        owner="System checks; AML officer decides on higher-risk ties.",
        retain=R_OFAC,
        actions={"PASS": "Nothing to do. No tie to a sanctioned or higher-risk jurisdiction.",
                 "REVIEW": "A tie to a higher-risk jurisdiction raises the risk rating and triggers enhanced due diligence (CDD-140). Find out why the customer has the tie and whether money will move to or from that country.",
                 "FAIL": "The customer resides in, is a citizen of, or is organised in a comprehensively sanctioned territory. The firm cannot open or service the account without a specific OFAC license. Decline, and refer to the AML officer."}),
    "SAN-220": dict(
        check="Political exposure",
        what="Whether the applicant, an immediate family member or a close associate is a senior foreign political figure (or, under house policy, a senior US official). Such customers are not prohibited, but the source of their wealth needs closer scrutiny for corruption risk.",
        rule="31 CFR 1010.605(p) (definition of senior foreign political figure); 31 CFR 1010.620; joint statement of 21 August 2020. FINRA's account application template asks about senior military, governmental or political officials of a non-US country.",
        owner="Applicant declares; screening vendor checks; AML officer approves.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. No political exposure declared or found; or exposure exists and enhanced due diligence was approved (see CDD-140).",
                 "OPEN": "The applicant declared political exposure but gave no detail. Obtain the office held, the country, the dates and whether it is the applicant, a family member or a close associate.",
                 "REVIEW": "Record the office held, the country and the relationship. Complete enhanced due diligence and obtain senior-management approval before the account is funded."}),
    "SAN-230": dict(
        check="Adverse media",
        what="A search of news and public records for links between the customer and financial crime, fraud, corruption or regulatory action.",
        rule="31 CFR 1023.210(b)(5) (risk-based due diligence); FINRA Regulatory Notice 19-18 (customer with a criminal, civil or regulatory history).",
        owner="Screening vendor returns hits; an AML analyst reads them.",
        retain=R_CIP,
        actions={"PASS": "Nothing to do. No hits, or a hit was read and judged immaterial (different person, old civil matter, unrelated to financial crime).",
                 "REVIEW": "Read the source articles, confirm they are about this person (age, location, middle name), and decide with the AML officer whether the relationship is acceptable. Record the decision and reason. Material financial-crime news normally means declining the account."}),
    "SAN-240": dict(
        check="FinCEN 314(a) request list",
        what="Whether the applicant matches a name law enforcement has asked financial institutions to search for through FinCEN.",
        rule="USA PATRIOT Act section 314(a); 31 CFR 1010.520 (search accounts of the past 12 months and transactions of the past 6 months; report a match within 14 days; keep the request confidential).",
        owner="AML officer only. The list is confidential and never discussed with the customer or with staff who do not need to know.",
        retain="5 years; keep the search record separate from the customer file.",
        actions={"PASS": "Nothing to do. No match in the current request.",
                 "REVIEW": "Report the match to FinCEN through the Secure Information Sharing System within 14 days. Do not tell the customer. A match does not by itself require closing or refusing the account: the AML officer decides, and considers a suspicious activity report."}),
    "TAX-300": dict(
        check="Tax certification",
        what="The applicant's certification, under penalty of perjury, of their taxpayer identification number and US-person status (Form W-9), or of foreign status (Form W-8BEN). It decides how income is reported and withheld.",
        rule="IRC 3406 and Form W-9 (backup withholding at 24% without a certified TIN); IRC 1441 and Form W-8BEN (30% or treaty-rate withholding for foreign persons; valid to the end of the third following calendar year).",
        owner="Applicant certifies; tax operations sets the withholding code.",
        retain="4 years after the last return to which the certificate relates; kept with the account record for " + R_BOOKS,
        actions={"PASS": "Nothing to do. Certification is on file. If the evidence shows backup withholding applies, check the 24% code is set on the account.",
                 "OPEN": "Do not open the account. The applicant must sign the certification on the Regulatory questions step."}),
    "TAX-310": dict(
        check="Tax residency consistency",
        what="The citizenship and residency status must agree with the tax residency declared: a non-resident alien cannot be tax-resident only in the US, and a second country of tax residence must be named.",
        rule="FATCA (IRC 1471-1474) and Forms W-9 / W-8BEN. The United States does not take part in the OECD Common Reporting Standard, so no CRS self-certification is collected.",
        owner="Tax operations.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. Status and tax residency agree.",
                 "REVIEW": "Ask the applicant which is right. A non-resident alien signs Form W-8BEN and names the country of tax residence; a US resident alien signs Form W-9. Trading stays on hold until corrected."}),
    "DIS-400": dict(
        check="Form CRS and Form ADV Part 2A delivered",
        what="The relationship summary and the firm brochure were delivered and acknowledged: the services offered, fees, conflicts of interest and disciplinary history.",
        rule="17 CFR 240.17a-14 and 275.204-5 (Form CRS: before or at the earliest of a recommendation, an order or account opening); Advisers Act Rule 204-3 (brochure before or at the advisory contract); Reg BI disclosure obligation.",
        owner="System delivers; applicant acknowledges.",
        retain=R_BOOKS + "; delivery dates also under " + R_ADV,
        actions={"PASS": "Nothing to do. The delivery date and document versions are on file.",
                 "OPEN": "No recommendation may be made until both documents are delivered. Send the applicant back to Agreements & signature."}),
    "DIS-410": dict(
        check="Privacy notice and electronic delivery consent",
        what="The initial privacy notice was provided, and the applicant consented to receive documents electronically after being told of the right to paper copies and how to withdraw consent.",
        rule="Regulation S-P, 17 CFR 248.4 (initial privacy notice no later than when the relationship is established); E-SIGN Act, 15 U.S.C. 7001(c) (consumer consent to electronic records).",
        owner="System delivers; applicant consents.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do.",
                 "OPEN": "The applicant must acknowledge the privacy notice. A digital-only adviser may make electronic delivery a condition of the account; if the applicant will not consent, the account cannot be opened online."}),
    "DIS-420": dict(
        check="Advisory and account agreement signed",
        what="The applicant signed the advisory agreement and the brokerage account agreement, including the highlighted pre-dispute arbitration disclosure, with a typed signature that matches the legal name.",
        rule="Advisers Act Rule 204-2(a)(10) (written agreements); FINRA Rule 2268 (arbitration disclosure, copy within 30 days); FINRA Rule 4512(a)(1); disclosures delivered at opening under FINRA Rules 2266 (SIPC), 2267 (BrokerCheck), 4370(e) (business continuity) and Regulation NMS Rule 607 (order routing).",
        owner="Applicant signs; operations sends the countersigned copy within 30 days.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. Send the copy of the agreement within 30 days and the account record within 30 days of opening (SEA Rule 17a-3(a)(17)(i)(B)).",
                 "OPEN": "No account without a signed agreement. The typed signature must match the legal name on the Applicant step exactly."}),
    "ACC-500": dict(
        check="Trusted contact person",
        what="A person aged 18 or over the firm may contact if it suspects financial exploitation, cannot reach the client, or needs to confirm their health status or who holds a power of attorney. The firm must ask; the client may decline.",
        rule="FINRA Rule 4512(a)(1)(F) and Supplementary Material .06 (reasonable efforts to obtain; written disclosure of when the firm may contact the person).",
        owner="Applicant; the advisor repeats the request at each periodic review.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. A contact is on file, or the client was asked and declined and the refusal is recorded.",
                 "OPEN": "Ask the applicant to name a contact or tick that they decline. The account can still be opened, but the request must be made and recorded. For a client aged 65 or over, call rather than email."}),
    "ACC-510": dict(
        check="Securities-industry affiliation",
        what="Whether the applicant, their spouse or a dependent child is associated with a FINRA member firm. Such a person needs their employer's written consent to hold an account elsewhere.",
        rule="FINRA Rule 3210 (accounts at other broker-dealers and financial institutions): prior written consent of the employer, and duplicate statements to the employer on request.",
        owner="Operations obtains the consent letter.",
        retain=R_BOOKS,
        actions={"INFO": "No affiliation declared. Nothing to do.",
                 "OPEN": "The applicant declared an affiliation but did not name the firm. Obtain the member firm's name before going further.",
                 "REVIEW": "Obtain the employer's written consent letter before the first trade and set up duplicate confirmations and statements to the employer's compliance department."}),
    "ACC-520": dict(
        check="Public-company insider",
        what="Whether the applicant is a director, officer or 10% shareholder of a listed company. Trading in that company's shares is restricted, and most issuers forbid insiders from hedging.",
        rule="Securities Act Rule 144 (volume limits, Form 144); Exchange Act section 16 (reporting, short-swing profits); Rule 10b5-1 (trading plans, 90-day cooling-off for officers and directors); Regulation S-K Item 407(i) (issuer hedging policies).",
        owner="Advisor and the firm's control room.",
        retain=R_BOOKS,
        actions={"INFO": "No insider status declared. Nothing to do.",
                 "OPEN": "The applicant declared insider status but did not name the company. Obtain the issuer and ticker.",
                 "REVIEW": "Add the issuer to the account's restricted list. Sales of that stock go through a Rule 10b5-1 plan or the issuer's pre-clearance window, inside the Rule 144 volume limit. Do not place a collar or protective put on it unless the issuer's policy permits hedging."}),
    "ACC-525": dict(
        check="IRA disclosure and beneficiary",
        what="For an IRA: the disclosure statement and custodial agreement were delivered, the 7-day right to revoke was explained, and a primary beneficiary was named.",
        rule="Treas. Reg. 1.408-6(d)(4) (disclosure statement at least 7 days before the IRA is established, or a 7-day right to revoke with a full refund); custodial agreement; beneficiary designation.",
        owner="Applicant names the beneficiary; operations delivers the disclosure statement.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. Diarise the end of the 7-day revocation period before investing the first contribution if the client asks to revoke.",
                 "OPEN": "Ask the applicant to name a primary beneficiary. Where a spouse is not the primary beneficiary in a community-property state, obtain the spouse's written consent."}),
    "ACC-530": dict(
        check="Senior investor protections",
        what="Whether the client is a 'specified adult' (65 or over). If the firm reasonably believes such a client is being financially exploited it may place a temporary hold on a disbursement or trade.",
        rule="FINRA Rule 2165 (hold of up to 15 business days, extendable by 10, and by 30 more if reported to a state agency or court; notify the trusted contact within 2 business days); Senior Safe Act (reporting immunity for trained staff).",
        owner="Advisor; escalates suspected exploitation to compliance the same day.",
        retain=R_BOOKS,
        actions={"INFO": "If the evidence says the client is a specified adult: make sure a trusted contact is on file (ACC-500), note any power of attorney, and treat unusual withdrawal requests, new third-party payees or a sudden change of beneficiary as reasons to pause and call the client. Otherwise nothing to do."}),
    "ACC-540": dict(
        check="Account record is current",
        what="How long ago the client confirmed the account record and investment profile. It must be sent to the client for confirmation at least every 36 months, and sooner for higher-risk customers.",
        rule="SEA Rule 17a-3(a)(17)(i)(B) (furnish the account record within 30 days of opening and at least every 36 months); 31 CFR 1023.210(b)(5)(ii) (update customer information on a risk basis).",
        owner="Advisor.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. Diarise the next confirmation at the review cycle set by the risk rating (CDD-130).",
                 "REVIEW": "The profile is stale. Re-confirm income, net worth, objectives and risk answers with the client before relying on the recommendation; trading is on hold until then."}),
    "SUI-600": dict(
        check="Investment profile complete",
        what="The facts a recommendation must rest on were collected: age, other investments, financial situation and needs, tax status, objectives, experience, time horizon, liquidity needs and risk tolerance.",
        rule="Reg BI care obligation, 17 CFR 240.15l-1(b)(2); FINRA Rule 2111(a); adviser fiduciary duty (SEC Release IA-5248); FINRA Rule 2090 (know your customer).",
        owner="Advisor.",
        retain=R_BOOKS + "; Reg BI records under SEA Rule 17a-3(a)(35)",
        actions={"PASS": "Nothing to do.",
                 "REVIEW": "Finish the risk questionnaire (at least 8 of 10 items) and record at least one goal. A recommendation built on a thin profile does not meet the care obligation."}),
    "SUI-610": dict(
        check="Risk answers consistent",
        what="The questionnaire answers were checked against each other and against what the client actually did in past sell-offs. Contradictions are scored conservatively and flagged.",
        rule="SEC IM Guidance Update 2017-02 (robo-advisers should flag or follow up on inconsistent questionnaire responses); Reg BI care obligation.",
        owner="Advisor.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do.",
                 "REVIEW": "Call the client, read back the conflicting answers listed on the Risk section, and record which one reflects them. The engine has already used the more conservative answer; raise risk only after that conversation."}),
    "SUI-620": dict(
        check="Recommended risk within stated loss tolerance",
        what="The recommended portfolio's loss in the worse of the 2008 and 2022 stress scenarios, compared with the largest loss the client said they could accept.",
        rule="Reg BI care obligation (customer-specific); FINRA Rule 2111.05(b).",
        owner="System enforces; advisor reviews exceptions.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. The stress loss is inside the client's limit.",
                 "REVIEW": "The stress loss exceeds the limit. Lower the equity tier or re-confirm the limit with the client in writing."}),
    "SUI-630": dict(
        check="Liquidity needs covered",
        what="Cash outside the portfolio against three months of essential expenses and any withdrawal planned in the next 12 months.",
        rule="Reg BI care obligation and FINRA Rule 2111 (liquidity needs are part of the investment profile).",
        owner="Advisor.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do.",
                 "REVIEW": "Build the cash reserve before investing: hold back the shortfall from the initial deposit in the cash sleeve, and tell the client why."}),
    "SUI-640": dict(
        check="Derivatives within approval level",
        what="The options approval level and futures eligibility the client's profile supports, and confirmation that no recommended overlay goes beyond them.",
        rule="FINRA Rule 2360(b)(16) (options account approval by a Registered Options Principal; disclosure document delivered; signed agreement within 15 days); NFA Compliance Rule 2-30 and CFTC Rule 1.55 (futures risk disclosure).",
        owner="Registered Options Principal approves; operations sends the verification letter.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do beyond the paperwork: if options are approved, send the account information for verification and collect the signed options agreement within 15 days, or the account goes to closing transactions only.",
                 "REVIEW": "An overlay exceeds the approval level. Remove it or obtain the higher approval first."}),
    "SUI-650": dict(
        check="Costs and reasonably available alternatives",
        what="The all-in annual cost and the comparison with simpler alternatives (a target-date fund, a 60/40 index mix, cash).",
        rule="Reg BI care obligation (consider costs and reasonably available alternatives); Form CRS fee disclosure; Form ADV Part 2A Item 5.",
        owner="Advisor.",
        retain=R_BOOKS,
        actions={"PASS": "Nothing to do. The comparison table is in the Method section and the client report.",
                 "REVIEW": "All-in cost is 1% a year or more. Document why this is still in the client's best interest against the cheaper alternatives, or reduce the fee."}),
    "SUI-660": dict(
        check="Client override of the recommended risk tier",
        what="The client asked for a different risk tier from the one recommended. A safer choice is applied automatically; a riskier one is not applied.",
        rule="SEC IM Guidance Update 2017-02 (explain a mismatch when the client selects a portfolio other than the one recommended); Reg BI care obligation.",
        owner="Advisor.",
        retain=R_BOOKS,
        actions={"INFO": "The client chose a safer tier. It was applied and the reason is on file; nothing to do.",
                 "REVIEW": "The client asked for more risk than the profile supports. It was not applied. Explain the mismatch, and change the tier only if the client's circumstances (not just their preference) justify it, with sign-off."}),
    "MON-700": dict(
        check="Monitoring baseline and reporting triggers",
        what="The activity the firm expects from this account, and the reports that funding or later activity could trigger.",
        rule="31 CFR 1023.320 (suspicious activity report: $5,000, 30 days, confidential); 31 CFR 1010.311 (currency transaction report over $10,000); 31 CFR 1010.410(e)-(f) (funds transfers of $3,000 or more); 31 CFR 1010.314 and 31 U.S.C. 5324 (structuring).",
        owner="AML surveillance.",
        retain="Suspicious activity reports and supporting documents: 5 years from filing.",
        actions={"INFO": "Nothing to do at opening. After funding, compare the first deposits with this baseline. Refer to the AML officer: funding from a source other than the one declared, a deposit followed quickly by a withdrawal to a third party, or amounts just under a reporting threshold. Never tell a customer that a report is being considered or was filed."}),
    "DEC-900": dict(
        check="Account decision",
        what="The single outcome of every line above, and the principal who accepted the account.",
        rule="FINRA Rule 4512(a)(1)(D) (signature of the partner, officer or manager accepting the account); FINRA Rule 3110 (supervision); FINRA Rule 3310 (AML program).",
        owner="A principal (Series 24) accepts; the AML officer must concur for any high-risk customer.",
        retain=R_BOOKS,
        actions={"PASS": "Approved. The account is open for funding and trading. Send the account record to the client within 30 days.",
                 "REVIEW": "Pending. The account is restricted: no trading, no withdrawals. Work the REVIEW lines above from the top; when they are all PASS or INFO a principal accepts the account on the firm-review panel.",
                 "OPEN": "Information needed. Nothing is opened yet. The plan is shown to the applicant as a proposal only. Send the outstanding-items list; applications idle for 30 days are withdrawn.",
                 "FAIL": "Declined. Do not open the account. Give the applicant a neutral reason (the firm is unable to open the account at this time); never mention a sanctions match or a suspicious activity report. Keep the full file."}),
    "REC-910": dict(
        check="Record retention schedule",
        what="How long each part of this file must be kept, so nothing is purged early.",
        rule="31 CFR 1023.220(a)(3) (CIP); 31 CFR 1010.230(i) (beneficial ownership); 31 CFR 501.601 (OFAC); SEA Rules 17a-3 and 17a-4; Advisers Act Rule 204-2.",
        owner="Records management.",
        retain="Identifying information and beneficial-ownership certification: 5 years after closing. Verification records: 5 years after made. Account record and agreements: 6 years after closing. Sanctions records: 10 years. Electronic records are kept with a time-stamped audit trail or in non-rewriteable storage (SEA Rule 17a-4(f)).",
        actions={"INFO": "Nothing to do. Do not delete any part of this file before the latest of the periods shown, and suspend deletion entirely if the account is under a legal hold or regulatory inquiry."}),
}

CHECKLIST_CODES = [("Investment profile complete", "SUI-600"), ("Risk answers consistent", "SUI-610"),
                   ("Recommended risk within", "SUI-620"), ("Liquidity needs covered", "SUI-630"),
                   ("Derivatives only within", "SUI-640"), ("Costs and reasonably available", "SUI-650")]


def _line(code: str, result: str, evidence: str) -> dict:
    g = GUIDE[code]
    action = g["actions"].get(result) or next(iter(g["actions"].values()))
    return {"code": code, "check": g["check"], "result": result, "evidence": evidence, "what": g["what"], "rule": g["rule"],
            "owner": g["owner"], "retain": g["retain"], "action": action}


def build_log(p: ClientProfile, kyc_status: str, checklist: list | None = None, now: dt.datetime | None = None) -> list[dict]:
    """One line per control, in the order an analyst works the file."""
    i, f = p.identity, p.finances
    miss = aml.missing_items(p) if i.enforce_intake else {s: [] for s in aml.STEPS}
    rr = aml.risk_rating(p)
    L: list[dict] = []
    add = lambda code, result, evidence: L.append(_line(code, result, evidence))
    has = lambda step, *words: [t for t in miss[step] if any(w.lower() in t.lower() for w in words)]

    add("APP-001", "INFO", f"{i.full_name}; {aml.ACCOUNT_LABEL.get(i.account_type, i.account_type)} account; online application; "
                           f"purpose: {i.account_purpose.lower()}")
    g = has("Applicant", "phone", "email", "occupation", "employer")
    add("APP-002", "OPEN" if g else "PASS", ("Missing: " + "; ".join(g)) if g else
        "Phone, email and employment on file" if i.enforce_intake else "Legacy record on file")
    add("CIP-010", "PASS" if i.cip_notice_ack or not i.enforce_intake else "OPEN",
        "Notice acknowledged by the applicant" if i.cip_notice_ack else
        ("Legacy record: notice given at original opening" if not i.enforce_intake else "Notice not yet acknowledged"))

    age = aml.age_on(i.date_of_birth)
    gaps = has("Applicant", "legal name", "date of birth", "identification number", "address", "zip", "taxpayer", "foreign ID")
    if age is not None and age < aml.ADULT_AGE:
        add("CIP-020", "FAIL", f"Applicant is {age} years old")
    elif gaps:
        add("CIP-020", "OPEN", "Missing: " + "; ".join(gaps))
    else:
        where = ", ".join(x for x in (i.address_city, i.state if i.address_country == aml.US else i.address_country) if x)
        add("CIP-020", "PASS", f"Name, date of birth{f' (age {age})' if age is not None else ''}, address"
                               f"{f' ({where})' if where else ''} and {i.tin_type} {i.tin_masked or 'on file'} recorded")
    if i.tin_type.startswith("Applied"):
        add("CIP-030", "REVIEW", "Taxpayer identification number applied for, not yet issued")
    elif i.tin_masked and not i.tin_valid:
        add("CIP-030", "OPEN", f"{i.tin_type} {i.tin_masked} failed the format and range check")
    elif not i.tin_masked and i.enforce_intake:
        add("CIP-030", "OPEN", "No identification number entered")
    else:
        add("CIP-030", "PASS", f"{i.tin_type} {i.tin_masked or 'on file'} passed the format and range check")

    doc = (f"{i.id_doc_type} {i.id_doc_masked}, issued by {i.id_doc_issuer or 'n/a'}, expires {i.id_doc_expiry or 'n/a'}"
           if i.id_doc_masked else "no document details recorded")
    if not i.identity_verified or i.id_verification == aml.ID_VERIFY[3]:
        add("CIP-040", "FAIL", f"Unable to verify identity; {doc}")
    elif has("Applicant", "documentary verification", "expiration", "unexpired"):
        add("CIP-040", "OPEN", "Missing: " + "; ".join(has("Applicant", "documentary verification", "expiration", "unexpired")))
    elif i.id_verification == aml.ID_VERIFY[2] or i.id_method == "pending":
        add("CIP-040", "REVIEW", f"Verification pending; {doc}")
    else:
        add("CIP-040", "PASS", f"{i.id_verification}; {doc}" + (f"; analyst {i.analyst}" if i.analyst else ""))
    if i.id_theft_flags and not i.id_theft_resolved:
        add("CIP-050", "REVIEW", "; ".join(i.id_theft_flags))
    else:
        add("CIP-050", "PASS", ("Resolved: " + "; ".join(i.id_theft_flags)) if i.id_theft_flags else "No red flags raised")

    if i.account_type == "joint":
        g = has("Account & funding", "joint owner")
        j = i.joint_owner
        add("CIP-060", "OPEN" if g else "PASS", ("Missing: " + "; ".join(g)) if g else
            f"Joint owner {j.get('name', '')}, SSN/ITIN {j.get('tin_masked', '')}; {str(j.get('tenancy', '')).lower()}")
    elif i.account_type in aml.ENTITY_TYPES:
        g = has("Account & funding", "trust", "entity:", "formation")
        e = i.entity
        add("CIP-060", "OPEN" if g else "PASS", ("Missing: " + "; ".join(g)) if g else
            f"{e.get('legal_name', '')} ({e.get('kind', i.account_type)}), EIN {e.get('ein_masked', '')}, formed in "
            f"{e.get('formed_in', '')}; documents: {e.get('documents', '')}; acts through {i.full_name}")
    else:
        add("CIP-060", "INFO", "Single owner; no additional parties")

    means = max(f.net_worth, 0.0) + f.household_income
    base = (f"{i.account_purpose}; initial funding ${i.expected_initial_funding:,.0f}; about ${i.expected_annual_deposits:,.0f}/yr "
            f"after; withdrawals {i.expected_withdrawals.lower()}")
    if i.enforce_intake and i.expected_initial_funding <= 0:
        add("CDD-100", "OPEN", "Expected initial funding not stated")
    elif i.enforce_intake and i.expected_initial_funding > 1.5 * means and not i.sow_documented:
        add("CDD-100", "REVIEW", base + f"; exceeds 1.5x stated net worth plus income (${means:,.0f})")
    else:
        add("CDD-100", "PASS", base)
    needs_docs = i.source_of_wealth in aml.SOW_NEEDS_DOCS or aml.FUNDING.get(i.source_of_funds, (0, False))[1]
    add("CDD-110", "REVIEW" if needs_docs and not i.sow_documented else "PASS",
        f"Wealth: {i.source_of_wealth}; funds: {i.source_of_funds}"
        + ("; supporting documents received" if i.sow_documented else ("; documents required" if needs_docs else "")))

    if aml.is_legal_entity_customer(i):
        exempt = i.entity.get("exempt", aml.BO_EXEMPT[0])
        g = has("Account & funding", "beneficial")
        bos = [b for b in i.beneficial_owners if b.get("name")]
        if exempt != aml.BO_EXEMPT[0]:
            add("CDD-120", "INFO", f"Excluded from the beneficial-ownership requirement: {exempt}")
        elif not i.beneficial_owners_verified:
            add("CDD-120", "FAIL", "Beneficial owners could not be verified")
        elif g:
            add("CDD-120", "OPEN", "Missing: " + "; ".join(g))
        else:
            add("CDD-120", "PASS", "; ".join(f"{b['name']} ({b.get('role', 'owner')}"
                                             + (f", {float(b.get('pct') or 0):.0f}%" if b.get("pct") else "") + ")" for b in bos)
                + "; certification signed")
    elif i.account_type == "trust":
        add("CDD-120", "FAIL" if not i.beneficial_owners_verified else "INFO",
            "Trustees could not be identified and verified" if not i.beneficial_owners_verified else
            "Ordinary trust: not a legal-entity customer; trustee identified under CIP (line CIP-060)")
    else:
        add("CDD-120", "INFO", "Natural-person account: not a legal-entity customer")

    top = sorted((x for x in rr.factors if x[1] > 0), key=lambda x: -x[1])[:3]
    add("CDD-130", "REVIEW" if rr.level == "high" and not i.edd_cleared else "INFO",
        f"{rr.level.capitalize()} ({rr.score}/100); next periodic review in {rr.review_months} months"
        + ("; drivers: " + ", ".join(f"{n.lower()} +{pts}" for n, pts, _ in top) if top else "; no risk factors scored"))
    if not rr.edd_required and not i.politically_exposed:
        add("CDD-140", "INFO", "Not required")
    elif i.edd_cleared:
        add("CDD-140", "PASS", "Approved" + (f" by {i.edd_approver}" if i.edd_approver else "") + "; reasons: "
            + ("; ".join(rr.edd_reasons) or "political exposure"))
    else:
        add("CDD-140", "REVIEW", "Required, not yet approved: " + ("; ".join(rr.edd_reasons) or "political exposure"))

    n_parties = len(aml.parties(i))
    src = i.screening_source or "sanctions list"
    hits = i.ofac_hits or []
    if not i.ofac_clear or i.ofac_disposition == aml.OFAC_DISPOSITIONS[2]:
        add("SAN-200", "FAIL", "True match confirmed" + (": " + hits[0].get("summary", "") if hits else "") + f" [{src}]")
    elif hits and i.ofac_disposition == aml.OFAC_DISPOSITIONS[0]:
        add("SAN-200", "REVIEW", f"{len(hits)} potential match(es): " + " | ".join(h.get("summary", "") for h in hits[:2]) + f" [{src}]")
    elif hits:
        add("SAN-200", "PASS", f"{len(hits)} potential match(es) cleared as false positive"
            + (f" by {i.analyst}" if i.analyst else "") + ": " + hits[0].get("summary", "") + f" [{src}]")
    else:
        add("SAN-200", "PASS", f"{n_parties} name(s) screened, no potential match [{src}]")

    ties = aml.countries_of(i)
    bad = [f"{c} ({r})" for r, c in ties if c in aml.SANCTIONED_JURISDICTIONS and r in ("residence", "citizenship", "entity formation")]
    watch = [f"{c} ({r})" for r, c in ties if c in aml.FATF_BLACK | aml.FATF_GREY | aml.HOUSE_HIGH_RISK
             or (c in aml.SANCTIONED_JURISDICTIONS and f"{c} ({r})" not in bad)]
    if bad:
        add("SAN-210", "FAIL", "Comprehensively sanctioned: " + ", ".join(bad))
    elif watch:
        add("SAN-210", "PASS" if i.edd_cleared else "REVIEW", "Higher-risk tie: " + ", ".join(dict.fromkeys(watch))
            + f" (FATF lists as of the {aml.FATF_ASOF})" + ("; accepted under enhanced due diligence" if i.edd_cleared else ""))
    else:
        add("SAN-210", "PASS", "Countries: " + ", ".join(dict.fromkeys(c for _, c in ties)))
    if i.politically_exposed and has("Regulatory questions", "political exposure"):
        add("SAN-220", "OPEN", f"{i.pep_type}: office, country and relationship not given")
    elif i.politically_exposed:
        add("SAN-220", "PASS" if i.edd_cleared else "REVIEW",
            (i.pep_type if i.pep_type != aml.PEP_TYPES[0] else "Politically exposed person") + (f": {i.pep_detail}" if i.pep_detail else "")
            + ("; enhanced due diligence approved" if i.edd_cleared else "; enhanced due diligence pending"))
    else:
        add("SAN-220", "PASS", "None declared; none found in screening")
    add("SAN-230", "REVIEW" if i.adverse_media in (aml.ADVERSE_MEDIA[1], aml.ADVERSE_MEDIA[3]) and not i.edd_cleared else "PASS",
        i.adverse_media)
    add("SAN-240", "REVIEW" if i.fincen_314a_match else "PASS",
        "Positive match to a current request" if i.fincen_314a_match else "No match")

    form = "Form W-8BEN" if i.citizenship == "Non-resident alien" else "Form W-9"
    add("TAX-300", "PASS" if i.tax_cert_signed or not i.enforce_intake else "OPEN",
        (f"{form} certification signed" if i.tax_cert_signed else (f"{form} on file (legacy record)" if not i.enforce_intake
                                                                    else f"{form} certification not signed"))
        + ("; subject to backup withholding (24%)" if i.backup_withholding else ""))
    incons = i.citizenship == "Non-resident alien" and i.tax_residency == "US only"
    add("TAX-310", "REVIEW" if incons else ("OPEN" if has("Regulatory questions", "country of tax residence") else "PASS"),
        f"{i.citizenship}; tax residency: {i.tax_residency}" + (f" ({i.other_tax_country})" if i.other_tax_country else "")
        + ("; inconsistent" if incons else ""))

    add("DIS-400", "PASS" if i.form_crs_acknowledged else "OPEN",
        "Form CRS and Form ADV Part 2A acknowledged" if i.form_crs_acknowledged else "Not yet acknowledged")
    ok = (i.privacy_notice_ack and i.esign_consent) or not i.enforce_intake
    add("DIS-410", "PASS" if ok else "OPEN",
        "Privacy notice and e-delivery consent on file" if ok else
        "Outstanding: " + ", ".join(n for v, n in ((i.privacy_notice_ack, "privacy notice"), (i.esign_consent, "e-delivery consent")) if not v))
    g = has("Agreements & signature", "agreement", "signature")
    add("DIS-420", "OPEN" if g else "PASS", ("Outstanding: " + "; ".join(g)) if g else
        (f"Signed electronically as '{i.signature_name}' on {i.profile_date or dt.date.today().isoformat()}" if i.signature_name
         else "Agreement on file (legacy record)"))

    if i.trusted_contact:
        add("ACC-500", "PASS", f"{i.trusted_contact}" + (f", {i.trusted_contact_info}" if i.trusted_contact_info else ""))
    elif i.trusted_contact_declined:
        add("ACC-500", "PASS", "Client was asked and declined; refusal recorded")
    else:
        add("ACC-500", "OPEN", "Not yet requested or answered" + ("; client is 65 or over" if p.is_senior else ""))
    add("ACC-510", "OPEN" if has("Regulatory questions", "broker-dealer affiliation") else "REVIEW" if i.employed_by_broker_dealer else "INFO",
        f"Associated with a FINRA member{f' ({i.bd_firm or i.employer})' if (i.bd_firm or i.employer) else ''}: employer consent letter required"
        if i.employed_by_broker_dealer else "None declared")
    add("ACC-520", "OPEN" if has("Regulatory questions", "public-company") else "REVIEW" if i.control_person_public_co else "INFO",
        (f"Director, officer or 10% holder of {i.insider_ticker or 'a listed company'}; 10b5-1 plan "
         f"{'in place' if i.has_10b5_1_plan else 'not in place'}; issuer {'permits' if i.issuer_permits_hedging else 'prohibits'} hedging")
        if i.control_person_public_co else "None declared")
    if i.account_type == "IRA only":
        g = has("Account & funding", "IRA")
        add("ACC-525", "OPEN" if g else "PASS", ("Missing: " + "; ".join(g)) if g else
            f"Disclosure statement delivered; primary beneficiary {i.ira_beneficiary or 'on file'}")
    add("ACC-530", "INFO", (f"Specified adult (age {p.household.age})" + (f"; power of attorney: {i.power_of_attorney}" if i.power_of_attorney else ""))
        if p.is_senior else f"Not a specified adult (age {p.household.age})")
    months = p.profile_age_months
    add("ACC-540", "REVIEW" if months > PROFILE_MAX_AGE_MONTHS else "PASS",
        (f"Confirmed {months} month(s) ago" if i.profile_date else "Confirmed today") + f"; limit {PROFILE_MAX_AGE_MONTHS} months")

    x = p.experience
    for name, ok_, ev in checklist or []:
        code = next((c for pre, c in CHECKLIST_CODES if name.startswith(pre)), None)
        if code:
            add(code, "PASS" if ok_ else "REVIEW", ev)
    if x.override_tiers:
        add("SUI-660", "INFO" if x.override_tiers < 0 else "REVIEW",
            f"{x.override_tiers:+d} tier(s)" + (f": {x.override_reason}" if x.override_reason else "; no reason given"))

    add("MON-700", "INFO", " ".join(aml.monitoring_notes(p)[:1]))
    res = {"cleared": "PASS", "hold": "REVIEW", "incomplete": "OPEN", "blocked": "FAIL"}[kyc_status]
    n_open = sum(1 for ln in L if ln["result"] in ("OPEN", "REVIEW", "FAIL"))
    add("DEC-900", res, f"{aml.STATUS_LABEL[kyc_status]}: {aml.STATUS_TITLE[kyc_status].lower()}"
        + (f"; accepted by {i.principal_name or 'a principal'}" if kyc_status == "cleared" and i.principal_approved else "")
        + (f"; {n_open} line(s) above need action" if n_open else ""))
    add("REC-910", "INFO", f"CIP {aml.CIP_RETENTION_YEARS} yrs after closing; account record {aml.BOOKS_RETENTION_YEARS} yrs; "
                           f"sanctions records {aml.OFAC_RETENTION_YEARS} yrs")

    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    prev = "0" * 12
    for n, ln in enumerate(L, 1):
        ln["seq"], ln["time"] = n, stamp
        prev = hashlib.sha256(f"{prev}|{n}|{ln['code']}|{ln['result']}|{ln['evidence']}".encode()).hexdigest()[:12]
        ln["hash"] = prev
    return L


def summary(lines: list[dict]) -> dict[str, int]:
    out = {k: 0 for k, _ in RESULTS}
    for ln in lines:
        out[ln["result"]] = out.get(ln["result"], 0) + 1
    return out


def to_csv(lines: list[dict]) -> str:
    """The log with its instructions, one row per line (opens in Excel)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["seq", "time_utc", "code", "check", "result", "evidence", "what_to_do_now", "what_this_line_records", "rule",
                "who_acts", "retention", "chain_hash"])
    for ln in lines:
        w.writerow([ln["seq"], ln["time"], ln["code"], ln["check"], ln["result"], ln["evidence"], ln["action"], ln["what"],
                    ln["rule"], ln["owner"], ln["retain"], ln["hash"]])
    return buf.getvalue()
