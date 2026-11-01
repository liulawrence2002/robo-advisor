"""Client report: a self-contained HTML document (investment policy statement + recommendation
file) that the app offers for download. Prints cleanly to PDF from any browser."""
from __future__ import annotations

import html
from datetime import date

import numpy as np

from .advisor import LIMITATIONS, Recommendation
from .goals import planning_age


def _m(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else (f"${x / 1e3:,.0f}K" if abs(x) >= 1e4 else f"${x:,.0f}")


def _p(x, d=0) -> str:
    return "n/a" if x is None else f"{x:.{d}%}"


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _md(s: str) -> str:
    import re
    s = _e(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return s


def _table(headers, rows) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_md(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
body{font-family:Georgia,'Times New Roman',serif;color:#18222F;max-width:900px;margin:40px auto;padding:0 24px;line-height:1.45}
h1{font-weight:500;font-size:30px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 8px;border-bottom:1px solid #DCD3C1;padding-bottom:4px}
.eyebrow{font-family:Helvetica,Arial,sans-serif;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#6B7480}
.meta{color:#6B7480;font-size:13px}table{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0 14px;font-family:Helvetica,Arial,sans-serif}
th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#6B7480;padding:6px 8px;border-bottom:1px solid #DCD3C1}
td{padding:6px 8px;border-bottom:1px solid #EEE7DA;vertical-align:top}.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}
.kpi{flex:1 1 150px;border:1px solid #DCD3C1;border-radius:8px;padding:10px 12px;font-family:Helvetica,Arial,sans-serif}
.kpi b{display:block;font-size:22px;font-weight:500}.kpi span{font-size:11px;color:#6B7480;text-transform:uppercase;letter-spacing:.08em}
ol li{margin:6px 0}.flag{border-left:3px solid #8C3B3B;padding:4px 10px;margin:6px 0;background:#FBF6EE}.review{border-left-color:#A8823F}
.stop{border:2px solid #8C3B3B;padding:10px 14px;background:#FBF0EE;margin:12px 0}.sig{margin-top:40px;display:flex;gap:40px}
.sig div{flex:1;border-top:1px solid #18222F;padding-top:6px;font-size:12px;color:#6B7480}.small{font-size:12px;color:#6B7480}
@media print{body{margin:0}h2{page-break-after:avoid}}
"""


def build_report(rec: Recommendation, firm: str = "Keel & Co.", sim_label: str = "") -> str:
    p, rp, f, h, el = rec.profile, rec.risk, rec.profile.finances, rec.profile.household, rec.elig
    sim, life = rec.sim, rec.sim_life
    PA = planning_age(p)
    today = date.today().strftime("%B %d, %Y")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{_e(firm)} client report</title><style>{CSS}</style></head><body>"]
    parts.append(f"<div class='eyebrow'>{_e(firm)} · Goals-based advice · Investment policy statement</div>"
                 f"<h1>{_e(p.identity.full_name)}</h1><div class='meta'>Prepared {today} · profile {_e(p.identity.profile_date or today)} · "
                 f"{_e(p.label)} · simulation {_e(sim_label)}</div>")
    if el.kyc_status != "cleared":
        parts.append(f"<div class='stop'><strong>Account status: {el.kyc_status.upper()}.</strong> " +
                     " ".join(_e(r) for r in el.kyc_reasons) + "</div>")

    parts.append("<h2>1. Summary</h2><div class='kpis'>"
                 f"<div class='kpi'><span>Plan success to {PA}</span><b>{_p(sim.success)}</b>{_p(sim.success_full)} with no cut</div>"
                 f"<div class='kpi'><span>Over simulated lifetimes</span><b>{_p(life.success)}</b></div>"
                 f"<div class='kpi'><span>Risk profile</span><b>{_e(rp.tier_name)}</b>tier {rp.tier} of 10 · {_p(rec.glide[0])} stocks</div>"
                 f"<div class='kpi'><span>Sustainable spending (85%)</span><b>{_m(rec.sustainable)}</b>per year, today's $</div>"
                 f"<div class='kpi'><span>All-in cost</span><b>{rec.fee_total:.2%}</b>advisory {f.advisory_fee:.2%} + ETFs {rec.stats['expense_ratio']:.2%}</div></div>")
    parts.append("<ol>" + "".join(f"<li>{_md(a)}</li>" for a in rec.advice if not a.startswith(("**Flag", "**Review"))) + "</ol>")

    parts.append("<h2>2. Client profile (FINRA 2111 / Reg BI investment profile)</h2>")
    hh = f"{h.marital_status}" + (f", spouse age {h.spouse_age}" if h.spouse_age else "") + f", {h.dependents} dependent(s)"
    parts.append(_table(["Item", "Value"], [
        ["Age / household", f"{h.age}, {hh}; health {h.health}"],
        ["Employment", f"{p.identity.employment_status}; {p.identity.occupation or 'n/a'}" + (f" at {p.identity.employer}" if p.identity.employer else "")],
        ["Account / tax status", f"{p.identity.account_type}; {p.identity.citizenship}; tax residency {p.identity.tax_residency}; filing {p.filing_status.upper()}"],
        ["Income", f"{_m(f.annual_income)} client" + (f" + {_m(f.spouse_income)} spouse" if f.spouse_income else "") + f"; {f.income_stability}; growth {f.income_real_growth:+.1%} real"],
        ["Savings", f"{_p(f.savings_rate)} of pay + {_p(f.employer_match)} employer; {_p(f.contrib_to_roth_share)} to Roth"],
        ["Investable assets", f"taxable {_m(f.taxable)} (basis {_p(f.taxable_cost_basis_pct)}), traditional {_m(f.traditional)}, Roth {_m(f.roth)}; bank cash {_m(f.cash)}"],
        ["Other assets / debts", f"home {_m(f.home_value)}, mortgage {_m(f.mortgage)}, other debt {_m(f.other_debt)} at {_p(f.other_debt_rate)}"
                                 + (f"; single stock {_m(f.concentrated_stock)} {f.concentrated_ticker}" if f.concentrated_stock else "")],
        ["Liquidity needs", f"{_m(f.monthly_essential_expenses)}/month essentials; {_m(f.liquidity_need_12m)} planned withdrawals in 12 months"],
        ["Retirement", f"age {f.retirement_age}; spending {_m(f.retirement_spending)}/yr ({_p(f.essential_share_of_spending)} essential); "
                       f"Social Security claim at {f.ss_claim_age}" + ("; already claiming" if f.already_claiming_ss else "") + f"; pension {_m(f.pension)}"],
        ["Objective / experience", f"{p.experience.objective}; {p.experience.years_investing} yrs investing, knowledge {p.experience.knowledge}; "
                                   f"options {p.experience.options_years} yrs ({p.experience.options_trades_per_year}/yr), futures {p.experience.futures_years} yrs"],
        ["Goals", "; ".join(f"{g.name} ({g.kind}, {g.priority}): {_m(g.amount)} in {g.years_until} yrs" for g in p.goals)],
        ["Source of wealth / funding", f"{p.identity.source_of_wealth}; expected initial funding {_m(p.identity.expected_initial_funding)}"],
        ["Trusted contact / POA", (p.identity.trusted_contact or ("declined" if p.identity.trusted_contact_declined else "not provided"))
                                  + (f"; POA {p.identity.power_of_attorney}" if p.identity.power_of_attorney else "")],
    ]))

    parts.append("<h2>3. Risk profile</h2>")
    parts.append(_table(["Dimension", "Measure", "Equity implied"], [
        ["Willingness", f"tolerance {rp.tolerance_score:.0f}/100 ({rp.n_answered}/10 answered); gamma {rp.gamma:.2f}; max loss {_p(rp.max_loss)}",
         f"{_p(rp.hc_equity)} (balance sheet) / {_p(rp.drawdown_equity)} (stress cap)"],
        ["Ability", f"capacity {rp.capacity_score:.0f}/100; funded ratio {rp.funded_ratio:.2f}x; human capital {_m(rp.human_capital)}; guaranteed income PV {_m(rp.guaranteed_pv)}",
         _p(rp.capacity_equity)],
        ["Need", f"required real return {_p(rp.need_real_return, 1) if rp.need_real_return is not None else 'unreachable'}",
         _p(rp.need_equity) if rp.need_equity is not None else "above any model"],
        ["Recommendation", f"binding: {rp.binding}" + (f"; client override {rp.override_applied:+d} tier(s)" if rp.override_applied else ""),
         f"<strong>{_p(rp.recommended_equity)}</strong> · {rp.tier_name}; worst stress loss {_p(rp.model_worst_loss, 1)}"],
    ]))
    for fl in rp.review_flags:
        parts.append(f"<div class='flag review'><strong>Advisor review.</strong> {_e(fl)}</div>")
    for fl in rp.flags + rec.goal_warnings:
        parts.append(f"<div class='flag'>{_e(fl)}</div>")

    parts.append("<h2>4. Portfolio and implementation</h2>")
    parts.append(_table(["Bucket", "Priority", "Years", "Target", "Funded today", "Top-up / yr", "Stocks", "Model"],
                        [[b["Bucket"], b["Priority"], f"{b['Horizon (yrs)']}", _m(b["Target (today's $)"]) if not np.isnan(b["Target (today's $)"]) else "-",
                          _m(b["Funded today"]), _m(b["Annual top-up"]) if b["Annual top-up"] > 0 else "-", _p(b["Equity now"]), b["Model"]]
                         for _, b in rec.buckets.iterrows()]))
    impl_rows = [[r["Asset class"], r["ETF"], r["Alternative"], _p(r["Weight"], 1), _m(r["Amount"]),
                  "" if np.isnan(r["Shares"]) else f"{r['Shares']:,.0f}", _p(r["Expense ratio"], 2) if not np.isnan(r["Expense ratio"]) else ""]
                 for _, r in rec.implementation.iterrows()]
    parts.append(_table(["Asset class", "ETF", "Alternative (TLH pair)", "Weight", "Amount", "Shares", "Expense"], impl_rows))
    parts.append(f"<p class='small'>Expected return {_p(rec.stats['exp_return_geo'], 1)} nominal, volatility {_p(rec.stats['vol'], 1)}; "
                 f"2008-style year {_p(rec.stress['2008 global financial crisis'], 1)}, 2022-style {_p(rec.stress['2022 stock-bond inflation shock'], 1)}. "
                 f"Glide path: {_p(rec.glide[0])} stocks today, {_p(rec.glide[min(10, len(rec.glide) - 1)])} in 10 years, "
                 f"{_p(rec.glide[min(25, len(rec.glide) - 1)])} in 25 years.</p>")
    loc = rec.location
    parts.append(_table(["Asset location"] + list(loc.columns), [[idx] + [_m(v) for v in row] for idx, row in loc.iterrows()]))

    parts.append("<h2>5. Options and futures</h2>")
    parts.append(f"<p>Options approval Level {el.options_level}; futures {'approved (hedging only)' if el.futures_ok else 'not approved'}.</p>")
    parts.append("<ul>" + "".join(f"<li>{_e(r)}</li>" for r in el.options_reasons + el.futures_reasons) + "</ul>")
    if rec.overlays.rationale:
        parts.append("<ul>" + "".join(f"<li>{_e(r)}</li>" for r in rec.overlays.rationale) + "</ul>")
    parts.append("<p class='small'>Not recommended: " + " ".join(_e(s) for s in rec.overlays.not_recommended) + "</p>")

    parts.append("<h2>6. Projections</h2>")
    pc = sim.pct
    idx = [i for i in range(0, len(pc), max(len(pc) // 8, 1))]
    parts.append(_table(["Age", "5th pct", "25th", "Median", "75th", "95th"],
                        [[f"{pc.index[i]}", _m(pc['p5'].iloc[i]), _m(pc['p25'].iloc[i]), _m(pc['p50'].iloc[i]), _m(pc['p75'].iloc[i]), _m(pc['p95'].iloc[i])] for i in idx]))
    if rec.levers is not None:
        parts.append(_table(["Lever", "P(success)", "P(no cut)", "Median at retirement", "Median estate (after tax)", "Lifetime tax"],
                            [[r["Lever"], _p(r["P(success, plan to age 95)"]), _p(r["P(no cut)"]), _m(r["Median wealth at retirement"]),
                              _m(r["Median estate after tax"]), _m(r["Median lifetime tax"])] for _, r in rec.levers.iterrows()]))
    if rec.scenarios is not None:
        parts.append(_table(["Historical replay", "P(success)", "P(no cut)", "Median wealth after 3 yrs", "Median estate"],
                            [[r["Scenario"], _p(r["P(success)"]), _p(r["P(no cut)"]), _m(r["Wealth after 3 yrs (median)"]), _m(r["Median estate"])]
                             for _, r in rec.scenarios.iterrows()]))

    parts.append("<h2>7. Suitability file (Reg BI / fiduciary care)</h2>")
    parts.append(_table(["Item", "Status", "Evidence"], [[n, "pass" if ok else "FAIL", ev] for n, ok, ev in rec.checklist]))
    parts.append("<ul>" + "".join(f"<li>{_e(n)}</li>" for n in el.notes) + "</ul>")

    parts.append("<h2>8. What this tool does not do</h2>")
    parts.append(_table(["Area", "Limitation"], [[a, b] for a, b in LIMITATIONS]))
    parts.append("<p class='small'>Projections are hypothetical, not guarantees. Return assumptions: J.P. Morgan Asset Management 2026 "
                 "Long-Term Capital Market Assumptions; volatility and correlation from ETF history 2010-2026; mortality SSA 2022; "
                 "2026 IRS limits and brackets. Options involve risk and are not suitable for all investors. Educational project, not "
                 "investment advice.</p>")
    parts.append("<div class='sig'><div>Client signature / date</div><div>Advisor / Registered Options Principal</div><div>Compliance review</div></div>")
    parts.append("</body></html>")
    return "".join(parts)
