"""Visual system for the robo-advisor app: palette, CSS, HTML components, page runtime and the
Plotly template.

Direction: a single-scroll editorial page in the manner of modern minimalist product sites. Bone
white canvas, ink type, one accent (keel green) with amber and red reserved for signals. Instrument
Serif for display lines and big figures, Instrument Sans for reading, JetBrains Mono for numbers
and labels. A fixed top bar with a scroll-progress line and section links; numbered sections with
faint numerals; cards on hairlines; a tight, consistent spacing scale. No animation library: the only
script highlights the section in view and draws the scroll-progress line.
"""
from __future__ import annotations

import html
import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

INK, INK2, MUTED = "#101418", "#3B424B", "#6E7580"
PAPER, CARD, HAIR, HAIR2 = "#F8F7F3", "#FFFFFF", "#E4E1D9", "#EEECE6"
GREEN, GREEN2, MINT = "#0E6B52", "#1F8A6C", "#DCEFE6"
AMBER, RED, NAVY, SLATE = "#B07A17", "#B0323F", "#1E3658", "#4E6B92"
# names kept for callers written against the previous palette
SPRUCE, SPRUCE2, SAGE, BRASS, BRASS2, CLARET = GREEN, GREEN2, "#8FB8A6", AMBER, "#C9A55C", RED

ASSET_COLORS = {
    "cash": "#C7C3B8", "ust_int": "#8FB8A6", "tips": "#B6D3C5", "us_agg": "#3F8A70", "us_ig": "#5EA089",
    "us_hy": "#9DB37A", "ust_long": GREEN, "us_lc": NAVY, "us_sc": SLATE, "intl_dev": "#7D98BC",
    "em": "#B4C5DB", "reit": "#B5764A", "commod": "#8E6A3A", "gold": "#C9A55C", "mgd_fut": "#6B4E71",
}
CASE_COLORS = {"A": NAVY, "B": AMBER, "C": GREEN}
TONE = {"good": GREEN, "mid": AMBER, "bad": RED, "info": NAVY}

NAV = [("profile", "Profile"), ("overview", "Plan"), ("risk", "Risk"), ("portfolio", "Portfolio"),
       ("projections", "Projections"), ("options", "Options"), ("markets", "Markets"), ("compliance", "Compliance"),
       ("compare", "Compare"), ("method", "Method")]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:ital,wght@0,400..700;1,400..700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root{
  --ink:#101418; --ink2:#3B424B; --muted:#6E7580; --paper:#F8F7F3; --card:#FFFFFF; --hair:#E4E1D9; --hair2:#EEECE6;
  --green:#0E6B52; --green2:#1F8A6C; --mint:#DCEFE6; --amber:#B07A17; --red:#B0323F; --navy:#1E3658;
  --serif:'Instrument Serif', Georgia, 'Times New Roman', serif;
  --sans:'Instrument Sans', 'Helvetica Neue', Arial, sans-serif;
  --mono:'JetBrains Mono', ui-monospace, Menlo, monospace;
  --shadow:0 1px 0 rgba(16,20,24,.02), 0 18px 40px -30px rgba(16,20,24,.28);
  --tb:64px;
}
/* ---------- canvas: no sidebar, no header, fixed top bar ---------- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
header[data-testid="stHeader"], [data-testid="stAppDeployButton"]{display:none !important;}
[data-testid="stAppViewContainer"]{background:
  radial-gradient(1200px 500px at 85% -10%, rgba(14,107,82,.07), transparent 60%),
  radial-gradient(900px 500px at -10% 20%, rgba(176,122,23,.05), transparent 55%), var(--paper);}
[data-testid="stMainBlockContainer"]{max-width:1200px; padding:calc(var(--tb) + .9rem) 2.4rem 4rem;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"]{gap:.7rem;}
[data-stale="true"]{opacity:.65 !important; transition:opacity .25s ease;}
.stMainBlockContainer p, .stMainBlockContainer li{line-height:1.6;}
h1, h2, h3, h4{font-family:var(--serif) !important; font-weight:400 !important; letter-spacing:-.01em; color:var(--ink);}
[data-testid="stCaptionContainer"]{color:var(--muted);}
html{scroll-behavior:smooth;}
::selection{background:var(--mint);}

/* ---------- top bar ---------- */
.tb{position:fixed; top:0; left:0; right:0; height:var(--tb); z-index:1000; background:rgba(248,247,243,.78);
  backdrop-filter:saturate(1.4) blur(14px); -webkit-backdrop-filter:saturate(1.4) blur(14px); border-bottom:1px solid var(--hair);}
.tb-progress{position:absolute; left:0; top:0; height:2px; width:0; background:var(--green); transition:width .08s linear;}
.tb-in{max-width:1200px; margin:0 auto; height:100%; padding:0 2.4rem; display:flex; align-items:center; gap:1.2rem; justify-content:space-between;}
.tb-in > *{min-width:0;}
.tb-brand{font-family:var(--serif); font-size:1.45rem; color:var(--ink); white-space:nowrap; display:flex; align-items:baseline; gap:.7rem;}
.tb-brand i{color:var(--green); font-style:italic;}
.tb-brand span{font-family:var(--mono); font-size:.6rem; letter-spacing:.2em; text-transform:uppercase; color:var(--muted);}
.tb-nav{display:flex; gap:.15rem; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:none;}
.tb-nav::-webkit-scrollbar{display:none;}
.tb-nav a{font-family:var(--sans); font-size:.76rem; font-weight:500; color:var(--muted); text-decoration:none; padding:.38rem .6rem;
  border-radius:999px; white-space:nowrap; transition:all .18s ease; cursor:pointer;}
.tb-nav a:hover{color:var(--ink); background:rgba(16,20,24,.05);}
.tb-nav a.active{color:var(--green); background:var(--mint);}
.tb-mkt{font-family:var(--mono); font-size:.72rem; color:var(--ink2); white-space:nowrap; display:flex; align-items:center; gap:.5rem;}
.dot{width:7px; height:7px; border-radius:50%; background:#B9B5AC; display:inline-block;}
.dot.live{background:var(--green2); box-shadow:0 0 0 3px rgba(31,138,108,.18); animation:pulse 2.4s ease-in-out infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 3px rgba(31,138,108,.18);} 50%{box-shadow:0 0 0 6px rgba(31,138,108,.06);}}
@media (max-width:1540px){.tb-brand span{display:none;}}
@media (max-width:1280px){.tb-mkt{display:none;}}
@media (max-width:760px){.tb-nav{display:none;}}

/* ---------- controls strip ---------- */
[class*="st-key-controls"]{background:var(--card); border:1px solid var(--hair); border-radius:14px; padding:.75rem 1.1rem .85rem; box-shadow:var(--shadow);}
[class*="st-key-controls"] [data-testid="stVerticalBlock"]{gap:.35rem;}
[class*="st-key-controls"] [data-testid="stRadio"] [role="radiogroup"]{gap:.35rem; flex-wrap:wrap;}
[class*="st-key-controls"] [data-testid="stRadio"] label[data-baseweb="radio"]{margin:0; padding:.42rem .9rem; border:1px solid var(--hair);
  border-radius:999px; background:var(--paper); transition:all .15s ease; cursor:pointer;}
[class*="st-key-controls"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover{border-color:var(--ink);}
[class*="st-key-controls"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){background:var(--ink); border-color:var(--ink);}
[class*="st-key-controls"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p{color:#fff;}
[class*="st-key-controls"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{display:none;}
[class*="st-key-controls"] [data-testid="stRadio"] label p{font-size:.84rem; font-weight:500; color:var(--ink2);}
[class*="st-key-controls"] [data-testid="stPopover"] button, [class*="st-key-controls"] .stButton button{border-radius:999px; font-weight:500;}
[class*="st-key-controls"] [data-testid="stCaptionContainer"]{margin:0;} [class*="st-key-controls"] [data-testid="stCaptionContainer"] p{margin:0; font-size:.8rem;}

/* ---------- intake stepper (tabs inside the profile section) ---------- */
.stTabs [data-baseweb="tab-list"]{gap:.45rem; border-bottom:none; flex-wrap:wrap; background:transparent;}
.stTabs [data-baseweb="tab"]{padding:.5rem 1rem; border:1px solid var(--hair); border-radius:999px; background:var(--card); height:auto; transition:all .15s ease;}
.stTabs [data-baseweb="tab"] p{font-family:var(--sans); font-size:.76rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
.stTabs [data-baseweb="tab"]:hover{border-color:var(--ink);}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:var(--green); border-color:var(--green);}
.stTabs [data-baseweb="tab"][aria-selected="true"] p{color:#fff;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem;}

/* ---------- cards ---------- */
[class*="st-key-card_"]{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:1rem 1.2rem 1.1rem;
  box-shadow:var(--shadow);}
[class*="st-key-card_"] [data-testid="stPlotlyChart"]{margin-top:-.2rem;}
[data-testid="stExpander"] details{background:var(--card); border:1px solid var(--hair); border-radius:12px;}
[data-testid="stExpander"] summary p{font-weight:600; color:var(--ink2);}
[data-testid="stDataFrame"]{border-radius:10px; overflow:hidden;}
.stDownloadButton button, .stButton button{font-weight:600; letter-spacing:.01em; border-radius:10px;}
[data-testid="stWidgetLabel"] p{color:var(--ink2); font-weight:500;}
div[data-baseweb="slider"] [role="slider"]{box-shadow:0 0 0 4px rgba(14,107,82,.14);}
[data-testid="stPlotlyChart"]{background:transparent;}

/* ---------- hero ---------- */
.hero{padding:1.1rem 0 1.5rem; border-bottom:1px solid var(--hair);}
.eyebrow{font-family:var(--mono); font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; color:var(--green);}
.hero-title{font-family:var(--serif); font-weight:400; font-size:clamp(2.3rem, 4vw, 3.5rem); line-height:1.04; letter-spacing:-.02em;
  color:var(--ink); margin:.45rem 0 .8rem; max-width:22ch;}
.hero-title em{font-style:italic; color:var(--green);}
.hero-title .w{display:inline-block; white-space:pre;}
.hero-grid{display:grid; grid-template-columns:1.4fr .8fr; gap:2rem; align-items:center;}
@media (max-width:900px){.hero-grid{grid-template-columns:1fr;}}
.hero-sub{color:var(--ink2); font-size:1rem; max-width:62ch; margin:0 0 .8rem;}
.chips{display:flex; flex-wrap:wrap; gap:.4rem;}
.chip{font-size:.76rem; color:var(--ink2); background:var(--card); border:1px solid var(--hair); border-radius:999px; padding:.24rem .7rem;}
.chip b{font-family:var(--mono); font-weight:500; color:var(--ink);}
.hero-num{display:flex; flex-direction:column; align-items:flex-start; gap:.35rem; border-left:1px solid var(--hair); padding-left:1.6rem;}
@media (max-width:900px){.hero-num{border-left:none; padding-left:0;}}
.hero-big{font-family:var(--serif); font-size:clamp(3.4rem, 6vw, 5rem); line-height:.9; letter-spacing:-.03em; color:var(--ink);
  font-variant-numeric:tabular-nums lining-nums;}
.hero-big small{font-size:.35em; color:var(--muted); margin-left:.05em; letter-spacing:0;}
.hero-cap{font-size:.82rem; color:var(--muted); max-width:30ch;}
.verdict{display:inline-flex; align-items:center; gap:.45rem; font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  padding:.34rem .8rem; border-radius:999px; color:#fff; background:var(--tone);}

/* ---------- section headers ---------- */
.sec{position:relative; margin:2.2rem 0 .6rem; padding-top:1.1rem; border-top:1px solid var(--hair); scroll-margin-top:calc(var(--tb) + 10px);}
.sec-num{position:absolute; right:0; top:.5rem; font-family:var(--serif); font-size:3.6rem; line-height:1; color:rgba(16,20,24,.08);
  letter-spacing:-.04em; pointer-events:none; user-select:none;}
.sec h3{font-family:var(--serif); font-weight:400; font-size:1.8rem; margin:.25rem 0 .25rem; color:var(--ink); letter-spacing:-.015em;}
.sec p{color:var(--muted); margin:0; max-width:78ch; font-size:.93rem;}
.sub{margin:1.1rem 0 .3rem;}
.sub h4{font-family:var(--serif); font-weight:400; font-size:1.25rem; margin:.1rem 0 .15rem; color:var(--ink);}
.sub p{color:var(--muted); margin:0; max-width:78ch; font-size:.88rem;}

/* ---------- KPI tiles ---------- */
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:.6rem;}
.kpi{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.8rem .95rem .8rem; box-shadow:var(--shadow);
  position:relative; overflow:hidden;}
.kpi:after{content:""; position:absolute; left:1.05rem; right:1.05rem; bottom:0; height:2px; background:var(--tone,var(--hair)); border-radius:2px 2px 0 0;}
.kpi-l{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted);}
.kpi-v{font-family:var(--serif); font-size:1.75rem; line-height:1.1; color:var(--ink); margin-top:.35rem; letter-spacing:-.01em;
  font-variant-numeric:tabular-nums lining-nums;}
.kpi-s{font-size:.78rem; color:var(--muted); margin-top:.25rem;}

/* ---------- plan statement & meter ---------- */
.stmt-text{font-family:var(--serif); font-size:1.35rem; line-height:1.3; color:var(--ink); margin:.35rem 0 .9rem; max-width:40ch; letter-spacing:-.01em;}
.stmt-text strong{font-weight:400; color:var(--green);}
.meter{margin:.3rem 0 .4rem;}
.meter-track{position:relative; height:10px; border-radius:999px; overflow:visible; display:flex; gap:2px;}
.meter-track .z{height:100%;}
.z1{background:rgba(176,50,63,.22); border-radius:999px 0 0 999px;} .z2{background:rgba(176,122,23,.28);} .z3{background:rgba(14,107,82,.32); border-radius:0 999px 999px 0;}
.needle{position:absolute; top:-6px; width:3px; height:22px; background:var(--ink); border-radius:2px; transform:translateX(-50%); box-shadow:0 0 0 3px var(--card);}
.meter-scale{position:relative; height:1.2rem; font-family:var(--mono); font-size:.64rem; color:var(--muted); margin-top:.35rem;}
.meter-scale span{position:absolute; transform:translateX(-50%);}
.meter-legend{display:flex; gap:1.1rem; font-size:.74rem; color:var(--muted); margin-top:.15rem; flex-wrap:wrap;}
.meter-legend i{display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:.35rem; vertical-align:-1px;}
.facts{display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.6rem 1.4rem; margin-top:1.1rem; padding-top:1rem; border-top:1px dashed var(--hair);}
.fact-l{font-size:.74rem; color:var(--muted);} .fact-v{font-family:var(--mono); font-size:.95rem; color:var(--ink); margin-top:.1rem;}

/* ---------- action cards ---------- */
.actions{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:.6rem;}
.act{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.9rem 1.05rem .95rem; box-shadow:var(--shadow); transition:border-color .2s ease;}
.act:hover{border-color:rgba(14,107,82,.4);}
.act-n{font-family:var(--mono); font-size:.66rem; color:var(--green); letter-spacing:.14em;}
.act-t{font-family:var(--serif); font-size:1.2rem; color:var(--ink); margin:.3rem 0 .35rem; line-height:1.15;}
.act-b{font-size:.9rem; color:var(--ink2); line-height:1.55;} .act-b strong{color:var(--ink);}

/* ---------- goal cards ---------- */
.goals{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.6rem;}
.goal{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.9rem 1rem; box-shadow:var(--shadow);}
.goal-h{display:flex; justify-content:space-between; align-items:center; gap:.5rem;}
.goal-n{font-family:var(--serif); font-size:1.22rem; color:var(--ink);}
.tag{font-family:var(--mono); font-size:.6rem; letter-spacing:.12em; text-transform:uppercase; padding:.16rem .5rem; border-radius:4px;
  border:1px solid var(--hair); color:var(--muted); white-space:nowrap;}
.tag.need{color:var(--green); border-color:rgba(14,107,82,.35);} .tag.want{color:var(--amber); border-color:rgba(176,122,23,.4);}
.tag.wish{color:#6B4E71; border-color:rgba(107,78,113,.35);}
.goal-m{font-size:.8rem; color:var(--muted); margin:.35rem 0 .7rem;}
.bar{height:6px; background:var(--hair2); border-radius:999px; overflow:hidden;}
.bar span{display:block; height:100%; background:var(--tone,var(--green)); border-radius:999px;}
.goal-p{display:flex; justify-content:space-between; font-size:.76rem; color:var(--muted); margin-top:.4rem;}
.goal-p b{font-family:var(--mono); color:var(--ink); font-weight:500;}

/* ---------- notes, pillars, checklist, tables, steps ---------- */
.note{display:flex; gap:.8rem; background:var(--card); border:1px solid var(--hair); border-left:3px solid var(--tone,var(--amber));
  border-radius:10px; padding:.65rem .95rem; margin:.35rem 0; font-size:.88rem; color:var(--ink2);}
.note-k{font-family:var(--mono); font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; color:var(--tone,var(--amber)); white-space:nowrap; padding-top:.2rem; min-width:88px;}
.pillars{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:.6rem;}
.pillar{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.95rem 1.1rem; box-shadow:var(--shadow);}
.pillar-k{font-family:var(--mono); font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; color:var(--green);}
.pillar-t{font-family:var(--serif); font-size:1.4rem; color:var(--ink); margin:.2rem 0 .6rem;}
.pillar-row{display:flex; justify-content:space-between; padding:.34rem 0; border-top:1px solid var(--hair2); font-size:.85rem; color:var(--ink2);}
.pillar-row b{font-family:var(--mono); font-weight:500; color:var(--ink);}
.pillar-q{font-size:.82rem; color:var(--muted); margin-top:.6rem; line-height:1.5;}
.checks{background:var(--card); border:1px solid var(--hair); border-radius:12px; box-shadow:var(--shadow); overflow:hidden;}
.check{display:grid; grid-template-columns:34px 1fr; gap:.2rem .6rem; padding:.75rem 1rem; border-top:1px solid var(--hair2);}
.check:first-child{border-top:none;}
.check-i{width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:700; color:#fff; background:var(--tone);}
.check-t{font-weight:600; color:var(--ink); font-size:.9rem;} .check-e{grid-column:2; font-size:.8rem; color:var(--muted);}
.tbl{width:100%; border-collapse:collapse; font-size:.85rem;}
.tbl th{font-family:var(--mono); font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); text-align:left; font-weight:500;
  padding:.5rem .6rem; border-bottom:1px solid var(--hair);}
.tbl td{padding:.55rem .6rem; border-bottom:1px solid var(--hair2); color:var(--ink2); vertical-align:top;}
.tbl td.num{font-family:var(--mono); text-align:right; color:var(--ink);}
.tbl tr:last-child td{border-bottom:none;}
.steps{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.6rem;}
@media (max-width:1000px){.steps{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media (max-width:640px){.steps{grid-template-columns:1fr;}}
.step{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.9rem .95rem;}
.step-n{font-family:var(--mono); font-size:.64rem; color:var(--green); letter-spacing:.14em;}
.step-t{font-family:var(--serif); font-size:1.12rem; color:var(--ink); margin:.2rem 0 .25rem;}
.step-b{font-size:.8rem; color:var(--muted); line-height:1.45;}
.badge{display:inline-block; font-family:var(--mono); font-size:.68rem; letter-spacing:.08em; padding:.2rem .6rem; border-radius:6px; color:#fff; background:var(--tone,var(--navy));}
.split{margin-top:.3rem;}
.split-bar{display:flex; height:12px; border-radius:999px; overflow:hidden; gap:2px; background:var(--hair2);}
.split-bar span{display:block; height:100%;}
.split-rows{display:grid; grid-template-columns:1fr 1fr; gap:.35rem 1.2rem; margin-top:.85rem;}
.split-row{display:flex; justify-content:space-between; font-size:.83rem; color:var(--ink2); border-bottom:1px solid var(--hair2); padding-bottom:.3rem;}
.split-row i{display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:.45rem;}
.split-row b{font-family:var(--mono); font-weight:500; color:var(--ink);}

/* ---------- intake: part headers, application status, field groups ---------- */
.part{display:flex; align-items:baseline; gap:.8rem; margin:1.3rem 0 .2rem;}
.part-k{font-family:var(--mono); font-size:.68rem; letter-spacing:.16em; text-transform:uppercase; color:#fff; background:var(--ink);
  border-radius:6px; padding:.2rem .55rem; white-space:nowrap;}
.part h4{font-family:var(--serif); font-weight:400; font-size:1.3rem; margin:0; color:var(--ink);}
.part p{color:var(--muted); margin:0; font-size:.86rem;}
.grp{font-family:var(--mono); font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; color:var(--green); margin:.7rem 0 -.1rem;
  padding-top:.7rem; border-top:1px dashed var(--hair);}
.grp.first{border-top:none; padding-top:0; margin-top:.2rem;}
.appst{background:var(--card); border:1px solid var(--hair); border-left:4px solid var(--tone); border-radius:12px; padding:.9rem 1.1rem 1rem;
  box-shadow:var(--shadow); margin:.5rem 0 .2rem;}
.appst-h{display:flex; align-items:center; gap:.8rem; flex-wrap:wrap;}
.appst-b{font-family:var(--mono); font-size:.68rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:#fff; background:var(--tone);
  border-radius:999px; padding:.3rem .8rem; white-space:nowrap;}
.appst-t{font-family:var(--serif); font-size:1.25rem; color:var(--ink);}
.appst-s{font-size:.86rem; color:var(--ink2); margin:.35rem 0 .1rem; max-width:90ch;}
.appst-steps{display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.7rem;}
.stepchip{font-size:.76rem; color:var(--ink2); border:1px solid var(--hair); border-radius:999px; padding:.22rem .7rem .22rem .5rem; background:var(--paper);
  display:inline-flex; align-items:center; gap:.4rem;}
.stepchip i{width:16px; height:16px; border-radius:50%; background:var(--tone); color:#fff; font-style:normal; font-size:.62rem; font-weight:700;
  display:inline-flex; align-items:center; justify-content:center;}
.appst-list{margin:.7rem 0 0; padding:0; list-style:none; display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:.3rem 1.4rem;}
.appst-list li{font-size:.84rem; color:var(--ink2); padding-left:1.05rem; position:relative; line-height:1.45;}
.appst-list li:before{content:""; position:absolute; left:0; top:.52em; width:6px; height:6px; border-radius:50%; background:var(--tone);}
.appst-list li b{font-weight:600; color:var(--ink);}
.appst-list.one{grid-template-columns:1fr; margin-top:.3rem;}
.appst-groups{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.9rem 1.6rem; margin-top:.8rem; padding-top:.8rem;
  border-top:1px dashed var(--hair);}
.appst-g{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--tone);}

/* ---------- customer risk rating ---------- */
.rr-track{position:relative; height:10px; border-radius:999px; display:flex; gap:2px; margin:.5rem 0 .3rem;}
.rr-track .z{height:100%;}
.rr-scale{display:flex; justify-content:space-between; font-family:var(--mono); font-size:.64rem; color:var(--muted);}

/* ---------- audit log ---------- */
.log{background:var(--card); border:1px solid var(--hair); border-radius:12px; overflow:hidden; box-shadow:var(--shadow);}
.log-head, .log summary{display:grid; grid-template-columns:28px 148px 86px minmax(0,1.1fr) 92px minmax(0,1.5fr); gap:.7rem; padding:.5rem 1rem; align-items:baseline;}
.log-head{font-family:var(--mono); font-size:.6rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--hair);
  background:var(--paper);}
.log details{border-top:1px solid var(--hair2);}
.log details:first-of-type{border-top:none;}
.log summary{cursor:pointer; list-style:none; font-size:.83rem; color:var(--ink2); transition:background .12s ease;}
.log summary::-webkit-details-marker{display:none;}
.log summary:hover{background:var(--paper);}
.log details[open] summary{background:var(--paper);}
.log-n, .log-t, .log-c{font-family:var(--mono); font-size:.7rem; color:var(--muted); white-space:nowrap;}
.log-c{color:var(--ink); font-weight:500;}
.log-k{font-weight:600; color:var(--ink);}
.log-e{color:var(--ink2); font-size:.8rem; overflow-wrap:anywhere;}
.log-r{font-family:var(--mono); font-size:.62rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#fff; background:var(--tone);
  border-radius:5px; padding:.14rem .45rem; justify-self:start; white-space:nowrap;}
.log-body{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem 1.6rem; padding:.3rem 1rem 1rem 3.2rem; background:var(--paper);}
.log-body div{font-size:.83rem; color:var(--ink2); line-height:1.5;}
.log-body b{display:block; font-family:var(--mono); font-size:.6rem; letter-spacing:.14em; text-transform:uppercase; color:var(--green); font-weight:500; margin-bottom:.15rem;}
.log-body .now{grid-column:1 / -1; background:var(--card); border:1px solid var(--hair); border-left:3px solid var(--tone); border-radius:8px; padding:.55rem .8rem;}
.log-body .now b{color:var(--tone);}
.log-body .log-evm{display:none;}
@media (max-width:900px){
  .log-head{display:none;}
  .log summary{grid-template-columns:28px 86px 1fr 92px;} .log-t, .log-e{display:none;}
  .log-body{grid-template-columns:1fr; padding-left:1rem;} .log-body .log-evm{display:block;}
}
.legend{display:flex; flex-wrap:wrap; gap:.5rem 1.2rem; font-size:.8rem; color:var(--ink2); margin:.2rem 0 .5rem;}
.legend span{display:inline-flex; align-items:center; gap:.4rem;}
.foot{margin-top:2.5rem; padding-top:1.1rem; border-top:1px solid var(--hair); display:flex; justify-content:space-between; gap:2rem; flex-wrap:wrap;
  font-size:.78rem; color:var(--muted);}
.foot b{font-family:var(--serif); font-size:1.1rem; color:var(--ink); font-weight:400;}
"""

# A small plain script injected into the page from a zero-height component frame (placed near the
# top of the page because Streamlit lazy-loads component frames): nav highlight, scroll-progress
# line and smooth scrolling. No animation library.
PAGE_JS = r"""
(function(){
  if (window.__keel) return;
  const D = document;
  const sc = D.querySelector('[data-testid="stMain"]') || D.scrollingElement;
  function prog(){ const b = D.querySelector('.tb-progress'); if(!b) return;
    const m = sc.scrollHeight - sc.clientHeight; b.style.width = (m > 0 ? sc.scrollTop / m * 100 : 0) + '%'; }
  sc.addEventListener('scroll', prog, {passive:true});
  D.addEventListener('click', e => { const a = e.target.closest('a[data-jump]'); if(!a) return; e.preventDefault();
    const t = D.getElementById(a.getAttribute('data-jump')); if(t) t.scrollIntoView({behavior:'smooth', block:'start'}); });
  const nav = new IntersectionObserver(es => { es.forEach(en => { if(!en.isIntersecting) return; const id = en.target.id;
    D.querySelectorAll('.tb-nav a').forEach(a => a.classList.toggle('active', a.dataset.jump === id)); }); },
    {rootMargin:'-15% 0px -75% 0px'});
  function scan(){ D.querySelectorAll('.sec[id]:not([data-nv])').forEach(el => { el.dataset.nv = 1; nav.observe(el); }); prog(); }
  window.__keel = {scan};
  scan();
  new MutationObserver(() => { clearTimeout(window.__keelT); window.__keelT = setTimeout(scan, 200); }).observe(D.body, {childList:true, subtree:true});
})();
"""

PAGE_BOOT = """
<script>
(function(){
  const P = window.parent, D = P.document;
  if (P.__keel) { P.__keel.scan(); return; }
  const s = D.createElement('script'); s.textContent = PAGE_CODE; D.body.appendChild(s);
})();
</script>
""".replace("PAGE_CODE", __import__("json").dumps(PAGE_JS))


# ============================================================================ setup
def inject() -> None:
    st.html(f"<style>{CSS}</style>")
    _register_plotly()


def page_runtime() -> None:
    """Mounts the nav/progress script from a zero-height component frame."""
    components.html(PAGE_BOOT, height=0)


def _register_plotly() -> None:
    if "keel" in pio.templates:
        return
    axis = dict(gridcolor="#ECEAE4", linecolor="#E4E1D9", zerolinecolor="#E4E1D9", tickcolor="#E4E1D9",
                title=dict(font=dict(size=11, color=MUTED)), tickfont=dict(size=11, color=INK2))
    pio.templates["keel"] = go.layout.Template(layout=dict(
        font=dict(family="Instrument Sans, Helvetica Neue, Arial, sans-serif", size=12, color=INK2),
        title=dict(font=dict(family="Instrument Serif, Georgia, serif", size=19, color=INK), x=0, xanchor="left", y=0.97),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        colorway=[NAVY, AMBER, GREEN, RED, SLATE, "#6B4E71", "#8FB8A6", "#B5764A"],
        xaxis=axis, yaxis=dict(axis, linecolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color=PAPER, family="JetBrains Mono, monospace", size=12)),
        legend=dict(font=dict(size=11, color=INK2), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, x=0),
        margin=dict(l=8, r=8, t=46, b=8),
    ))
    pio.templates.default = "plotly_white+keel"


def chart(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(paper_bgcolor=CARD, plot_bgcolor=CARD)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displayModeBar": False, "displaylogo": False, "responsive": True})


# ============================================================================ text helpers
def md(s: str) -> str:
    """Inline markdown (**bold**, *italic*) to safe HTML."""
    s = html.escape(str(s), quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<em>\1</em>", s)
    return s


def split_title(s: str) -> tuple[str, str]:
    m = re.match(r"\*\*(.+?)\*\*\s*(.*)", s, flags=re.S)
    if m:
        return m.group(1).rstrip("."), m.group(2)
    return "", s


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def verdict_for(p: float) -> tuple[str, str]:
    if p >= 0.85:
        return "good", "On track"
    if p >= 0.70:
        return "mid", "Within reach"
    return "bad", "Needs attention"


# ============================================================================ page chrome
def topbar(spot: float, vix: float | None, rate: float, live: bool) -> None:
    links = "".join(f'<a data-jump="sec-{k}">{esc(v)}</a>' for k, v in NAV)
    v = f"{vix:.1f}" if vix is not None else "n/a"
    st.html(f'<div class="tb"><div class="tb-progress"></div><div class="tb-in">'
            f'<div class="tb-brand">Keel <i>&amp;</i> Co.<span>Goals-based advice</span></div>'
            f'<nav class="tb-nav">{links}</nav>'
            f'<div class="tb-mkt"><span class="dot {"live" if live else ""}"></span>S&amp;P {spot:,.0f} &nbsp;·&nbsp; VIX {v} '
            f'&nbsp;·&nbsp; T-bill {rate:.2%}</div></div></div>')


def hero(eyebrow: str, title_html: str, sub: str, chips: list[tuple[str, str]], prob: float, cap: str) -> None:
    tone, label = verdict_for(prob)
    ch = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    st.html(f'''<section class="hero">
<div class="eyebrow">{esc(eyebrow)}</div>
<h1 class="hero-title">{title_html}</h1>
<div class="hero-grid">
  <div><p class="hero-sub">{md(sub)}</p><div class="chips">{ch}</div></div>
  <div class="hero-num"><span class="verdict" style="--tone:{TONE[tone]}">{label}</span>
    <div class="hero-big">{prob * 100:.0f}<small>%</small></div>
    <div class="hero-cap">{esc(cap)}</div></div>
</div></section>''')


def section(title: str, lede: str = "", eyebrow: str = "", anchor: str | None = None, num: str | None = None) -> None:
    eb = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    ld = f"<p>{md(lede)}</p>" if lede else ""
    nm = f'<div class="sec-num">{esc(num)}</div>' if num else ""
    idattr = f' id="sec-{anchor}"' if anchor else ""
    st.html(f'<div class="sec"{idattr}>{nm}{eb}<h3>{esc(title)}</h3>{ld}</div>')


def subsection(title: str, lede: str = "") -> None:
    ld = f"<p>{md(lede)}</p>" if lede else ""
    st.html(f'<div class="sub"><h4>{esc(title)}</h4>{ld}</div>')


def footer() -> None:
    st.html('<div class="foot"><div><b>Keel <i style="color:var(--green)">&amp;</i> Co.</b><br>Goals-based digital advice · educational project, not investment advice.</div>'
            '<div>Return assumptions: J.P. Morgan 2026 Long-Term Capital Market Assumptions.<br>Volatility and correlation: ETF history 2010-2026. '
            'Live data: Yahoo Finance, FRED, U.S. Treasury, multpl, Damodaran, CBOE, CNN.</div></div>')


# ============================================================================ components
def kpis(items: list[dict]) -> None:
    out = []
    for it in items:
        tone = TONE.get(it.get("tone", ""), it.get("tone", HAIR))
        sub = f'<div class="kpi-s">{md(it["sub"])}</div>' if it.get("sub") else ""
        out.append(f'<div class="kpi" style="--tone:{tone}"><div class="kpi-l">{esc(it["label"])}</div>'
                   f'<div class="kpi-v">{esc(it["value"])}</div>{sub}</div>')
    st.html(f'<div class="kpis">{"".join(out)}</div>')


def plan_statement(p: float, sentence: str, facts: list[tuple[str, str]], eyebrow: str = "Plan confidence") -> None:
    pos = max(1.0, min(99.0, p * 100))
    facts_html = "".join(f'<div><div class="fact-l">{esc(k)}</div><div class="fact-v">{esc(v)}</div></div>' for k, v in facts)
    st.html(f"""<div class="blk">
<div class="eyebrow">{esc(eyebrow)}</div>
<p class="stmt-text">{md(sentence)}</p>
<div class="meter">
  <div class="meter-track"><span class="z z1" style="width:70%"></span><span class="z z2" style="width:15%"></span>
  <span class="z z3" style="width:15%"></span><span class="needle" style="left:{pos:.1f}%"></span></div>
  <div class="meter-scale"><span style="left:0%">0</span><span style="left:50%">50</span><span style="left:70%">70</span>
  <span style="left:85%">85</span><span style="left:100%">100</span></div>
  <div class="meter-legend"><span><i style="background:rgba(176,50,63,.35)"></i>Needs attention &lt;70</span>
  <span><i style="background:rgba(176,122,23,.45)"></i>Within reach 70-85</span>
  <span><i style="background:rgba(14,107,82,.5)"></i>On track 85+</span></div>
</div>
<div class="facts">{facts_html}</div></div>""")


def actions(advice: list[str], start: int = 1) -> None:
    cards = []
    for i, a in enumerate(advice, start):
        t, b = split_title(a)
        cards.append(f'<div class="act"><div class="act-n">{i:02d}</div><div class="act-t">{esc(t)}</div>'
                     f'<div class="act-b">{md(b)}</div></div>')
    st.html(f'<div class="actions">{"".join(cards)}</div>')


def goal_cards(goals: list[dict]) -> None:
    out = []
    for g in goals:
        p = g["prob"]
        tone, _ = verdict_for(p)
        out.append(f"""<div class="goal"><div class="goal-h"><div class="goal-n">{esc(g['name'])}</div>
<span class="tag {esc(g['priority'])}">{esc(g['priority'])}</span></div>
<div class="goal-m">{md(g['meta'])}</div>
<div class="bar" style="--tone:{TONE[tone]}"><span style="width:{max(2, p * 100):.0f}%"></span></div>
<div class="goal-p"><span>Probability funded</span><b>{p:.0%}</b></div></div>""")
    st.html(f'<div class="goals">{"".join(out)}</div>')


def notes(items: list[str], kind: str = "Advisor note", tone: str = "mid") -> None:
    if not items:
        return
    st.html('<div class="blk">' + "".join(f'<div class="note" style="--tone:{TONE[tone]}"><div class="note-k">{esc(kind)}</div>'
                                         f'<div>{md(t)}</div></div>' for t in items) + "</div>")


def pillars(items: list[dict]) -> None:
    out = []
    for it in items:
        rows = "".join(f'<div class="pillar-row"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in it["rows"])
        out.append(f'<div class="pillar"><div class="pillar-k">{esc(it["key"])}</div><div class="pillar-t">{esc(it["title"])}</div>'
                   f'{rows}<div class="pillar-q">{md(it["note"])}</div></div>')
    st.html(f'<div class="pillars">{"".join(out)}</div>')


def checklist(rows: list[tuple[str, bool, str]]) -> None:
    out = []
    for name, ok, ev in rows:
        tone = GREEN if ok else RED
        out.append(f'<div class="check"><div class="check-i" style="--tone:{tone}">{"&#10003;" if ok else "&#10005;"}</div>'
                   f'<div class="check-t">{esc(name)}</div><div class="check-e">{esc(ev)}</div></div>')
    st.html(f'<div class="checks">{"".join(out)}</div>')


def table(headers: list[str], rows: list[list], num_cols: set[int] | None = None) -> None:
    num_cols = num_cols or set()
    right = ' style="text-align:right"'
    th = "".join(f"<th{right if i in num_cols else ''}>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{md(c)}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    st.html(f'<table class="tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


def steps(items: list[tuple[str, str]]) -> None:
    out = "".join(f'<div class="step"><div class="step-n">{i:02d}</div><div class="step-t">{esc(t)}</div>'
                  f'<div class="step-b">{md(b)}</div></div>' for i, (t, b) in enumerate(items, 1))
    st.html(f'<div class="steps">{out}</div>')


def badge(text: str, tone: str = "info") -> str:
    return f'<span class="badge" style="--tone:{TONE.get(tone, tone)}">{esc(text)}</span>'


def html_block(s: str) -> None:
    st.html(s)


# ============================================================================ intake and compliance file
STATUS_TONE = {"cleared": GREEN, "hold": AMBER, "incomplete": NAVY, "blocked": RED}
RESULT_TONE = {"PASS": GREEN, "INFO": SLATE, "REVIEW": AMBER, "FAIL": RED, "OPEN": NAVY}


def part(key: str, title: str, lede: str = "") -> None:
    """Header for one part of the intake (A. Account application, B. Investment profile, ...)."""
    ld = f"<p>{md(lede)}</p>" if lede else ""
    st.html(f'<div class="part"><span class="part-k">{esc(key)}</span><div><h4>{esc(title)}</h4>{ld}</div></div>')


def group(label: str, first: bool = False) -> None:
    """Small label that separates field groups inside one intake step."""
    st.html(f'<div class="grp{" first" if first else ""}">{esc(label)}</div>')


def application_status(status: str, badge_text: str, title: str, summary: str, step_status: list[tuple[str, int]],
                       items: list[str] | dict[str, list[str]]) -> None:
    """Banner under the application: overall decision, per-step completeness and what is outstanding."""
    tone = STATUS_TONE.get(status, NAVY)
    chips = "".join(f'<span class="stepchip" style="--tone:{GREEN if n == 0 else AMBER}"><i>{"&#10003;" if n == 0 else n}</i>'
                    f'{esc(name)}</span>' for name, n in step_status)
    if isinstance(items, dict):         # outstanding items grouped by intake step
        lst = '<div class="appst-groups">' + "".join(
            f'<div><div class="appst-g">{esc(step)}</div><ul class="appst-list one">' + "".join(f"<li>{md(t)}</li>" for t in ts)
            + "</ul></div>" for step, ts in items.items() if ts) + "</div>"
    else:
        lst = ('<ul class="appst-list">' + "".join(f"<li>{md(t)}</li>" for t in items) + "</ul>") if items else ""
    st.html(f'<div class="appst" style="--tone:{tone}"><div class="appst-h"><span class="appst-b">{esc(badge_text)}</span>'
            f'<span class="appst-t">{esc(title)}</span></div><p class="appst-s">{md(summary)}</p>'
            f'<div class="appst-steps">{chips}</div>{lst}</div>')


def risk_rating(score: int, level: str, cap: int = 100) -> None:
    """Customer AML risk rating on a low / medium / high track."""
    pos = max(1.0, min(99.0, 100.0 * score / cap))
    st.html(f'<div class="rr-track"><span class="z z3" style="width:30%;border-radius:999px 0 0 999px"></span>'
            f'<span class="z z2" style="width:30%"></span><span class="z z1" style="width:40%;border-radius:0 999px 999px 0"></span>'
            f'<span class="needle" style="left:{pos:.1f}%"></span></div>'
            f'<div class="rr-scale"><span>0 · low</span><span>30 · medium</span><span>60 · high</span><span>{cap}</span></div>')


def log_lines(lines: list[dict], open_all: bool = False) -> None:
    """Audit log. Each line is a row that opens to its practitioner instructions.

    A line is a dict with: seq, time, code, check, result (PASS/INFO/REVIEW/FAIL/OPEN), evidence,
    and the guide fields what, rule, action (what to do for this result), retain.
    """
    head = ('<div class="log-head"><span>#</span><span>Time (UTC)</span><span>Code</span><span>Check</span>'
            '<span>Result</span><span>Evidence</span></div>')
    out = []
    for ln in lines:
        tone = RESULT_TONE.get(ln["result"], SLATE)
        out.append(
            f'<details{" open" if open_all else ""}><summary><span class="log-n">{ln["seq"]:02d}</span>'
            f'<span class="log-t">{esc(ln["time"])}</span><span class="log-c">{esc(ln["code"])}</span>'
            f'<span class="log-k">{esc(ln["check"])}</span><span class="log-r" style="--tone:{tone}">{esc(ln["result"])}</span>'
            f'<span class="log-e">{md(ln["evidence"])}</span></summary>'
            f'<div class="log-body"><div class="now" style="--tone:{tone}"><b>What to do now · result {esc(ln["result"])}</b>{md(ln["action"])}</div>'
            f'<div class="log-evm"><b>Evidence</b>{md(ln["evidence"])}</div>'
            f'<div><b>What this line records</b>{md(ln["what"])}</div>'
            f'<div><b>Rule it evidences</b>{md(ln["rule"])}</div>'
            f'<div><b>Who acts</b>{md(ln["owner"])}</div>'
            f'<div><b>Keep on file</b>{md(ln["retain"])}</div></div></details>')
    st.html(f'<div class="log">{head}{"".join(out)}</div>')


def result_legend(items: list[tuple[str, str]]) -> None:
    st.html('<div class="legend">' + "".join(
        f'<span><span class="log-r" style="--tone:{RESULT_TONE.get(k, SLATE)}">{esc(k)}</span>{esc(v)}</span>' for k, v in items) + "</div>")


def eyebrow(text: str) -> None:
    st.html(f'<div class="eyebrow">{esc(text)}</div>')


ALLOC_GROUPS = [("Stocks", ("us_lc", "us_sc", "intl_dev", "em"), NAVY), ("Real estate", ("reit",), "#B5764A"),
                ("Bonds", ("ust_int", "tips", "us_agg", "us_ig", "us_hy", "ust_long"), "#3F8A70"),
                ("Cash", ("cash",), "#C7C3B8"), ("Gold & commodities", ("gold", "commod"), "#C9A55C"),
                ("Managed futures", ("mgd_fut",), "#6B4E71")]


def alloc_split(weights) -> None:
    """Stacked bar + legend of the allocation grouped into plain-English buckets."""
    parts = [(n, float(sum(weights.get(k, 0) for k in ks)), col) for n, ks, col in ALLOC_GROUPS]
    parts = [x for x in parts if x[1] >= 0.005]
    bar = "".join(f'<span style="width:{w * 100:.2f}%;background:{c}" title="{esc(n)} {w:.0%}"></span>' for n, w, c in parts)
    rows = "".join(f'<div class="split-row"><span><i style="background:{c}"></i>{esc(n)}</span><b>{w:.0%}</b></div>'
                   for n, w, c in parts)
    st.html(f'<div class="split"><div class="split-bar">{bar}</div><div class="split-rows">{rows}</div></div>')
