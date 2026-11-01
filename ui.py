"""Visual system for the robo-advisor app: palette, CSS, HTML components and the Plotly template.

Direction: a private-bank client statement. Warm ivory paper, ink-navy sidebar, a serif display
face (Newsreader) for headlines and big figures, Hanken Grotesk for reading, IBM Plex Mono for
small labels. Brass marks emphasis, spruce marks "good", claret marks "attention".
"""
from __future__ import annotations

import html
import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK, INK2, MUTED = "#18222F", "#3A4654", "#6B7480"
PAPER, CARD, HAIR = "#F6F2E9", "#FFFDF8", "#E2DACB"
SPRUCE, SPRUCE2, SAGE = "#1E4D40", "#2F6B55", "#7FA898"
BRASS, BRASS2 = "#A8823F", "#C9A55C"
CLARET, NAVY, SLATE = "#8C3B3B", "#1F3A5F", "#4C6A92"

ASSET_COLORS = {
    "cash": "#BDB5A6", "ust_int": "#7FA898", "tips": "#A8C5B5", "us_agg": "#3F7A66", "us_ig": "#5E9483",
    "us_hy": "#9DB37A", "ust_long": SPRUCE, "us_lc": NAVY, "us_sc": SLATE, "intl_dev": "#7D98BC",
    "em": "#B4C5DB", "reit": "#B5764A", "commod": "#8E6A3A", "gold": BRASS2, "mgd_fut": "#6B4E71",
}
CASE_COLORS = {"A": NAVY, "B": BRASS, "C": SPRUCE}
TONE = {"good": SPRUCE, "mid": BRASS, "bad": CLARET, "info": NAVY}

_NOISE = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E"
          "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' "
          "stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 .09 0 0 0 0 .13 0 0 0 0 .18 0 0 0 .045 0'/%3E"
          "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Hanken+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --ink:#18222F; --ink2:#3A4654; --muted:#6B7480; --paper:#F6F2E9; --card:#FFFDF8; --hair:#E2DACB; --hair2:#EEE7DA;
  --spruce:#1E4D40; --spruce2:#2F6B55; --brass:#A8823F; --brass2:#C9A55C; --claret:#8C3B3B; --navy:#1F3A5F;
  --serif:'Newsreader', Georgia, 'Times New Roman', serif;
  --sans:'Hanken Grotesk', 'Helvetica Neue', Arial, sans-serif;
  --mono:'IBM Plex Mono', ui-monospace, Menlo, monospace;
  --shadow:0 1px 0 rgba(24,34,47,.03), 0 14px 32px -24px rgba(24,34,47,.35);
}
/* ---------- canvas ---------- */
[data-testid="stAppViewContainer"]{
  background: __NOISE__, radial-gradient(1100px 520px at 88% -8%, rgba(168,130,63,.10), transparent 62%),
              radial-gradient(900px 480px at -8% 108%, rgba(30,77,64,.07), transparent 60%), var(--paper);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"]{display:none;}
[data-testid="stMainBlockContainer"]{max-width:1240px; padding-top:2.4rem; padding-bottom:5rem;}
.stMainBlockContainer p, .stMainBlockContainer li{line-height:1.6;}
h1, h2, h3, h4{font-family:var(--serif) !important; font-weight:500 !important; letter-spacing:-.01em; color:var(--ink);}
[data-testid="stCaptionContainer"]{color:var(--muted);}

/* ---------- top navigation (first-level tabs) ---------- */
.stTabs [data-baseweb="tab-list"]{gap:1.9rem; border-bottom:1px solid var(--hair); background:transparent;}
.stTabs [data-baseweb="tab"]{padding:.95rem 0 .85rem; background:transparent; height:auto;}
.stTabs [data-baseweb="tab"] p{font-family:var(--sans); font-size:.74rem; font-weight:600; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); transition:color .15s ease;}
.stTabs [data-baseweb="tab"]:hover p{color:var(--ink);}
.stTabs [data-baseweb="tab"][aria-selected="true"] p{color:var(--ink);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--brass); height:2px;}
.stTabs [data-baseweb="tab-border"]{display:none;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.6rem;}

/* ---------- second-level tabs = intake stepper ---------- */
.stTabs .stTabs [data-baseweb="tab-list"]{gap:.5rem; border-bottom:none; flex-wrap:wrap;}
.stTabs .stTabs [data-baseweb="tab"]{padding:.5rem .95rem; border:1px solid var(--hair); border-radius:999px;
  background:var(--card); transition:all .15s ease;}
.stTabs .stTabs [data-baseweb="tab"] p{font-size:.72rem; letter-spacing:.08em;}
.stTabs .stTabs [data-baseweb="tab"][aria-selected="true"]{background:var(--spruce); border-color:var(--spruce);}
.stTabs .stTabs [data-baseweb="tab"][aria-selected="true"] p{color:#F6F2E9;}
.stTabs .stTabs [data-baseweb="tab-highlight"]{display:none;}
.stTabs .stTabs [data-baseweb="tab-panel"]{padding-top:1.1rem;}

/* ---------- cards: any container keyed card_* ---------- */
[class*="st-key-card_"]{background:var(--card); border:1px solid var(--hair); border-radius:12px;
  padding:1.15rem 1.35rem 1.25rem; box-shadow:var(--shadow);}
[class*="st-key-card_"] [data-testid="stPlotlyChart"]{margin-top:-.2rem;}
[class*="st-key-dark_"]{background:linear-gradient(160deg,#1A2938 0%,#14202C 100%); border-radius:14px;
  padding:1.5rem 1.6rem; box-shadow:0 30px 60px -40px rgba(20,32,44,.9); color:#E9E3D6;}

/* ---------- widgets ---------- */
[data-testid="stExpander"] details{background:var(--card); border:1px solid var(--hair); border-radius:10px;}
[data-testid="stExpander"] summary p{font-weight:600; color:var(--ink2);}
[data-testid="stDataFrame"]{border-radius:10px; overflow:hidden;}
.stDownloadButton button, .stButton button{font-weight:600; letter-spacing:.02em;}
[data-testid="stWidgetLabel"] p{color:var(--ink2); font-weight:500;}
div[data-baseweb="slider"] [role="slider"]{box-shadow:0 0 0 4px rgba(30,77,64,.12);}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#16232F 0%,#121C26 100%);}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:.6rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{gap:.45rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]{width:100%; margin:0; padding:.6rem .75rem;
  border:1px solid #2B3C4D; border-radius:10px; background:rgba(255,255,255,.015); transition:all .15s ease;}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover{border-color:#4A5D70;}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked){
  border-color:var(--brass2); background:rgba(201,165,92,.09);}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{font-weight:600;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#9AA6B2 !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] details{background:rgba(255,255,255,.02); border-color:#2B3C4D;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p{color:#E9E3D6;}
.brand{padding:.4rem 0 1.1rem; border-bottom:1px solid #2B3C4D; margin-bottom:1rem;}
.brand-mark{font-family:var(--serif); font-size:2.05rem; line-height:1; color:#F3EDE1; letter-spacing:-.01em;}
.brand-mark i{color:var(--brass2); font-style:italic;}
.brand-sub{font-family:var(--mono); font-size:.66rem; letter-spacing:.18em; text-transform:uppercase; color:#8F9BA8; margin-top:.45rem;}
.side-h{font-family:var(--mono); font-size:.64rem; letter-spacing:.18em; text-transform:uppercase; color:#8F9BA8; margin:.4rem 0 .5rem;}
.mkt{border:1px solid #2B3C4D; border-radius:10px; padding:.75rem .85rem; background:rgba(255,255,255,.02);}
.mkt-row{display:flex; justify-content:space-between; align-items:baseline; padding:.22rem 0; font-size:.84rem; color:#C9D1D9;}
.mkt-row b{font-family:var(--mono); font-weight:500; color:#F3EDE1;}
.mkt-asof{font-size:.7rem; color:#8F9BA8; margin-top:.45rem; display:flex; align-items:center; gap:.4rem;}
.dot{width:7px; height:7px; border-radius:50%; background:#6B7480; display:inline-block;}
.dot.live{background:#6FBF8E; box-shadow:0 0 0 3px rgba(111,191,142,.18);}
.side-foot{font-size:.72rem; color:#8F9BA8; line-height:1.55; margin-top:1rem; border-top:1px solid #2B3C4D; padding-top:.9rem;}
.side-foot p{margin:0 0 .6rem;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] label p{color:#E9E3D6 !important;}
[data-testid="stPlotlyChart"]{background:transparent;}
.split{margin-top:.3rem;}
.split-bar{display:flex; height:12px; border-radius:999px; overflow:hidden; gap:2px; background:var(--hair2);}
.split-bar span{display:block; height:100%;}
.split-rows{display:grid; grid-template-columns:1fr 1fr; gap:.35rem 1.2rem; margin-top:.85rem;}
.split-row{display:flex; justify-content:space-between; font-size:.84rem; color:var(--ink2); border-bottom:1px solid var(--hair2); padding-bottom:.3rem;}
.split-row i{display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:.45rem;}
.split-row b{font-family:var(--mono); font-weight:500; color:var(--ink);}

/* ---------- masthead ---------- */
.mast{padding:.2rem 0 1.4rem; animation:rise .5s ease both;}
.eyebrow{font-family:var(--mono); font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; color:var(--brass);}
.display{font-family:var(--serif); font-weight:400; font-size:2.65rem; line-height:1.08; letter-spacing:-.02em;
  color:var(--ink); margin:.45rem 0 .55rem;}
.display em{color:var(--spruce); font-style:italic;}
.mast-sub{color:var(--ink2); font-size:1.02rem; max-width:62ch; margin:0 0 .9rem;}
.chips{display:flex; flex-wrap:wrap; gap:.45rem;}
.chip{font-size:.78rem; color:var(--ink2); background:rgba(255,253,248,.7); border:1px solid var(--hair);
  border-radius:999px; padding:.22rem .7rem;}
.chip b{font-family:var(--mono); font-weight:500; color:var(--ink);}
@keyframes rise{from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;}}

/* ---------- section headers ---------- */
.sec{margin:1.6rem 0 .8rem;}
.sec:first-child{margin-top:.2rem;}
.sec h3{font-family:var(--serif); font-weight:500; font-size:1.55rem; margin:.25rem 0 .2rem; color:var(--ink);}
.sec p{color:var(--muted); margin:0; max-width:75ch;}
.rule{height:1px; background:linear-gradient(90deg,var(--brass) 0 60px, var(--hair) 60px 100%); margin:.5rem 0 0;}

/* ---------- KPI tiles ---------- */
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:.75rem;}
.kpi{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:.9rem 1rem .85rem;
  box-shadow:var(--shadow); position:relative; overflow:hidden;}
.kpi:before{content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--tone,var(--hair));}
.kpi-l{font-family:var(--mono); font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted);}
.kpi-v{font-family:var(--serif); font-size:1.85rem; line-height:1.1; color:var(--ink); margin-top:.3rem;
  font-variant-numeric:tabular-nums lining-nums;}
.kpi-s{font-size:.8rem; color:var(--muted); margin-top:.2rem;}

/* ---------- plan statement & meter ---------- */
.stmt-top{display:flex; align-items:flex-end; gap:1rem; flex-wrap:wrap;}
.stmt-big{font-family:var(--serif); font-size:4.4rem; line-height:.9; color:var(--ink); letter-spacing:-.03em;
  font-variant-numeric:tabular-nums lining-nums;}
.stmt-big small{font-size:1.6rem; color:var(--muted); margin-left:.1rem;}
.verdict{display:inline-flex; align-items:center; gap:.45rem; font-size:.78rem; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; padding:.32rem .75rem; border-radius:999px; color:#fff; background:var(--tone); margin-bottom:.55rem;}
.stmt-text{font-size:1.02rem; color:var(--ink2); margin:.9rem 0 1.1rem; max-width:60ch;}
.meter{margin:.3rem 0 .4rem;}
.meter-track{position:relative; height:12px; border-radius:999px; overflow:visible; display:flex;}
.meter-track .z{height:100%;}
.z1{background:linear-gradient(90deg,rgba(140,59,59,.18),rgba(140,59,59,.30)); border-radius:999px 0 0 999px;}
.z2{background:rgba(168,130,63,.32);}
.z3{background:rgba(30,77,64,.38); border-radius:0 999px 999px 0;}
.needle{position:absolute; top:-6px; width:3px; height:24px; background:var(--ink); border-radius:2px; transform:translateX(-50%);
  box-shadow:0 0 0 3px var(--card);}
.meter-scale{position:relative; height:1.2rem; font-family:var(--mono); font-size:.66rem; color:var(--muted); margin-top:.35rem;}
.meter-scale span{position:absolute; transform:translateX(-50%);}
.meter-legend{display:flex; gap:1.1rem; font-size:.74rem; color:var(--muted); margin-top:.15rem; flex-wrap:wrap;}
.meter-legend i{display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:.35rem; vertical-align:-1px;}
.facts{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:.6rem 1.4rem; margin-top:1.1rem;
  padding-top:1rem; border-top:1px dashed var(--hair);}
.fact-l{font-size:.76rem; color:var(--muted);} .fact-v{font-family:var(--mono); font-size:.95rem; color:var(--ink); margin-top:.1rem;}

/* ---------- action cards ---------- */
.actions{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:.8rem;}
.act{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:1rem 1.1rem 1.05rem;
  box-shadow:var(--shadow); transition:transform .18s ease, box-shadow .18s ease;}
.act:hover{transform:translateY(-2px); box-shadow:0 1px 0 rgba(24,34,47,.03), 0 20px 36px -26px rgba(24,34,47,.45);}
.act-n{font-family:var(--mono); font-size:.7rem; color:var(--brass); letter-spacing:.1em;}
.act-t{font-family:var(--serif); font-size:1.18rem; color:var(--ink); margin:.25rem 0 .35rem; line-height:1.2;}
.act-b{font-size:.9rem; color:var(--ink2); line-height:1.55;}
.act-b strong{color:var(--ink);}

/* ---------- goal cards ---------- */
.goals{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.8rem;}
.goal{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:1rem 1.1rem; box-shadow:var(--shadow);}
.goal-h{display:flex; justify-content:space-between; align-items:center; gap:.5rem;}
.goal-n{font-family:var(--serif); font-size:1.15rem; color:var(--ink);}
.tag{font-family:var(--mono); font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; padding:.16rem .5rem;
  border-radius:4px; border:1px solid var(--hair); color:var(--muted); white-space:nowrap;}
.tag.need{color:var(--spruce); border-color:rgba(30,77,64,.35);} .tag.want{color:var(--brass); border-color:rgba(168,130,63,.4);}
.tag.wish{color:#6B4E71; border-color:rgba(107,78,113,.35);}
.goal-m{font-size:.82rem; color:var(--muted); margin:.35rem 0 .7rem;}
.bar{height:6px; background:var(--hair2); border-radius:999px; overflow:hidden;}
.bar span{display:block; height:100%; background:var(--tone,var(--spruce)); border-radius:999px;}
.goal-p{display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted); margin-top:.4rem;}
.goal-p b{font-family:var(--mono); color:var(--ink); font-weight:500;}

/* ---------- notes, pillars, checklist, tables ---------- */
.note{display:flex; gap:.8rem; background:rgba(255,253,248,.85); border:1px solid var(--hair); border-left:3px solid var(--tone,var(--brass));
  border-radius:10px; padding:.75rem 1rem; margin:.5rem 0; font-size:.92rem; color:var(--ink2);}
.note-k{font-family:var(--mono); font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; color:var(--tone,var(--brass));
  white-space:nowrap; padding-top:.2rem; min-width:88px;}
.pillars{display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:.8rem;}
.pillar{background:var(--card); border:1px solid var(--hair); border-radius:12px; padding:1.05rem 1.15rem; box-shadow:var(--shadow);}
.pillar-k{font-family:var(--mono); font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; color:var(--brass);}
.pillar-t{font-family:var(--serif); font-size:1.25rem; color:var(--ink); margin:.2rem 0 .6rem;}
.pillar-row{display:flex; justify-content:space-between; padding:.32rem 0; border-top:1px solid var(--hair2); font-size:.86rem; color:var(--ink2);}
.pillar-row b{font-family:var(--mono); font-weight:500; color:var(--ink);}
.pillar-q{font-size:.83rem; color:var(--muted); margin-top:.6rem; line-height:1.5;}
.checks{background:var(--card); border:1px solid var(--hair); border-radius:12px; box-shadow:var(--shadow); overflow:hidden;}
.check{display:grid; grid-template-columns:34px 1fr; gap:.2rem .6rem; padding:.75rem 1rem; border-top:1px solid var(--hair2);}
.check:first-child{border-top:none;}
.check-i{width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:.8rem; font-weight:700; color:#fff; background:var(--tone);}
.check-t{font-weight:600; color:var(--ink); font-size:.92rem;} .check-e{grid-column:2; font-size:.82rem; color:var(--muted);}
.tbl{width:100%; border-collapse:collapse; font-size:.86rem;}
.tbl th{font-family:var(--mono); font-size:.64rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  text-align:left; font-weight:500; padding:.5rem .6rem; border-bottom:1px solid var(--hair);}
.tbl td{padding:.55rem .6rem; border-bottom:1px solid var(--hair2); color:var(--ink2); vertical-align:top;}
.tbl td.num{font-family:var(--mono); text-align:right; color:var(--ink);}
.tbl tr:last-child td{border-bottom:none;}
.steps{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.6rem;}
@media (max-width:900px){.steps{grid-template-columns:1fr;}}
.step{background:var(--card); border:1px solid var(--hair); border-radius:10px; padding:.85rem .9rem; position:relative;}
.step-n{font-family:var(--mono); font-size:.66rem; color:var(--brass); letter-spacing:.1em;}
.step-t{font-family:var(--serif); font-size:1.05rem; color:var(--ink); margin:.15rem 0 .25rem;}
.step-b{font-size:.8rem; color:var(--muted); line-height:1.45;}
.badge{display:inline-block; font-family:var(--mono); font-size:.7rem; letter-spacing:.08em; padding:.2rem .6rem;
  border-radius:6px; color:#fff; background:var(--tone,var(--navy));}
.lede{color:var(--ink2); font-size:.98rem; max-width:72ch;}
.muted{color:var(--muted);}
.small{font-size:.82rem;}
""".replace("__NOISE__", _NOISE)


# ============================================================================ setup
def inject() -> None:
    st.html(f"<style>{CSS}</style>")
    _register_plotly()


def _register_plotly() -> None:
    if "keel" in pio.templates:
        return
    axis = dict(gridcolor="#ECE5D7", linecolor="#D9D0BE", zerolinecolor="#D9D0BE", tickcolor="#D9D0BE",
                title=dict(font=dict(size=11, color=MUTED)), tickfont=dict(size=11, color=INK2))
    pio.templates["keel"] = go.layout.Template(layout=dict(
        font=dict(family="Hanken Grotesk, Helvetica Neue, Arial, sans-serif", size=12, color=INK2),
        title=dict(font=dict(family="Newsreader, Georgia, serif", size=17, color=INK), x=0, xanchor="left", y=0.97),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        colorway=[NAVY, BRASS, SPRUCE, CLARET, SLATE, "#6B4E71", SAGE, "#B5764A"],
        xaxis=axis, yaxis=dict(axis, linecolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(color=PAPER, family="IBM Plex Mono, monospace", size=12)),
        legend=dict(font=dict(size=11, color=INK2), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, x=0),
        margin=dict(l=8, r=8, t=46, b=8),
    ))
    pio.templates.default = "plotly_white+keel"


def chart(fig: go.Figure, height: int | None = None) -> None:
    # explicit layout colours: Streamlit overrides template-level backgrounds at render time
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


# ============================================================================ components
def masthead(eyebrow: str, title_html: str, sub: str, chips: list[tuple[str, str]]) -> None:
    ch = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in chips)
    st.html(f'<div class="mast"><div class="eyebrow">{esc(eyebrow)}</div><div class="display">{title_html}</div>'
            f'<p class="mast-sub">{md(sub)}</p><div class="chips">{ch}</div></div>')


def section(title: str, lede: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    ld = f"<p>{md(lede)}</p>" if lede else ""
    st.html(f'<div class="sec">{eb}<h3>{esc(title)}</h3>{ld}<div class="rule"></div></div>')


def kpis(items: list[dict]) -> None:
    out = []
    for it in items:
        tone = TONE.get(it.get("tone", ""), it.get("tone", HAIR))
        sub = f'<div class="kpi-s">{md(it["sub"])}</div>' if it.get("sub") else ""
        out.append(f'<div class="kpi" style="--tone:{tone}"><div class="kpi-l">{esc(it["label"])}</div>'
                   f'<div class="kpi-v">{esc(it["value"])}</div>{sub}</div>')
    st.html(f'<div class="kpis">{"".join(out)}</div>')


def verdict_for(p: float) -> tuple[str, str]:
    if p >= 0.85:
        return "good", "On track"
    if p >= 0.70:
        return "mid", "Within reach"
    return "bad", "Needs attention"


def plan_statement(p: float, sentence: str, facts: list[tuple[str, str]], eyebrow: str = "Plan confidence") -> None:
    tone, label = verdict_for(p)
    pos = max(1.0, min(99.0, p * 100))
    facts_html = "".join(f'<div><div class="fact-l">{esc(k)}</div><div class="fact-v">{esc(v)}</div></div>' for k, v in facts)
    st.html(f"""
<div class="eyebrow">{esc(eyebrow)}</div>
<div class="stmt-top" style="margin-top:.6rem">
  <div class="stmt-big">{p * 100:.0f}<small>%</small></div>
  <div><span class="verdict" style="--tone:{TONE[tone]}">{label}</span></div>
</div>
<p class="stmt-text">{md(sentence)}</p>
<div class="meter">
  <div class="meter-track"><span class="z z1" style="width:70%"></span><span class="z z2" style="width:15%"></span>
  <span class="z z3" style="width:15%"></span><span class="needle" style="left:{pos:.1f}%"></span></div>
  <div class="meter-scale"><span style="left:0%">0</span><span style="left:50%">50</span><span style="left:70%">70</span>
  <span style="left:85%">85</span><span style="left:100%">100</span></div>
  <div class="meter-legend"><span><i style="background:rgba(140,59,59,.35)"></i>Needs attention &lt;70</span>
  <span><i style="background:rgba(168,130,63,.45)"></i>Within reach 70-85</span>
  <span><i style="background:rgba(30,77,64,.5)"></i>On track 85+</span></div>
</div>
<div class="facts">{facts_html}</div>""")


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
    st.html("".join(f'<div class="note" style="--tone:{TONE[tone]}"><div class="note-k">{esc(kind)}</div>'
                    f'<div>{md(t)}</div></div>' for t in items))


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
        tone = SPRUCE if ok else CLARET
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


def sidebar_brand() -> None:
    st.sidebar.html('<div class="brand"><div class="brand-mark">Keel <i>&amp;</i> Co.</div>'
                    '<div class="brand-sub">Goals-based digital advice</div></div>')


def sidebar_market(spot: float, vix: float | None, rate: float, live: bool, asof: str) -> None:
    v = f"{vix:.1f}" if vix is not None else "n/a"
    st.sidebar.html(f'<div class="side-h">Market</div><div class="mkt">'
                    f'<div class="mkt-row"><span>S&amp;P 500</span><b>{spot:,.0f}</b></div>'
                    f'<div class="mkt-row"><span>VIX</span><b>{v}</b></div>'
                    f'<div class="mkt-row"><span>3-month T-bill</span><b>{rate:.2%}</b></div>'
                    f'<div class="mkt-asof"><span class="dot {"live" if live else ""}"></span>{esc(asof)}</div></div>')


def sidebar_heading(text: str) -> None:
    st.sidebar.html(f'<div class="side-h">{esc(text)}</div>')


def sidebar_foot(*paragraphs: str) -> None:
    body = "".join(f"<p>{md(t)}</p>" for t in paragraphs)
    st.sidebar.html(f'<div class="side-foot">{body}</div>')


ALLOC_GROUPS = [("Stocks", ("us_lc", "us_sc", "intl_dev", "em"), NAVY), ("Real estate", ("reit",), "#B5764A"),
                ("Bonds", ("ust_int", "tips", "us_agg", "us_ig", "us_hy", "ust_long"), "#3F7A66"),
                ("Cash", ("cash",), "#BDB5A6"), ("Gold & commodities", ("gold", "commod"), BRASS2),
                ("Managed futures", ("mgd_fut",), "#6B4E71")]


def alloc_split(weights) -> None:
    """Stacked bar + legend of the allocation grouped into plain-English buckets."""
    parts = [(n, float(sum(weights.get(k, 0) for k in ks)), col) for n, ks, col in ALLOC_GROUPS]
    parts = [x for x in parts if x[1] >= 0.005]
    bar = "".join(f'<span style="width:{w * 100:.2f}%;background:{c}" title="{esc(n)} {w:.0%}"></span>' for n, w, c in parts)
    rows = "".join(f'<div class="split-row"><span><i style="background:{c}"></i>{esc(n)}</span><b>{w:.0%}</b></div>'
                   for n, w, c in parts)
    st.html(f'<div class="split"><div class="split-bar">{bar}</div><div class="split-rows">{rows}</div></div>')
