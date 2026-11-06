"""Sanctions name screening against OFAC's Specially Designated Nationals (SDN) list.

The list is the public file Treasury publishes (sdn.csv plus alt.csv for aliases); it is
downloaded without a key and cached by the app. A handful of clearly synthetic entries are
always appended so the potential-match workflow can be shown without typing a real
designated person's name, and so screening still works offline.

Matching is the usual first-pass "fuzzy" screen: names are normalised (case, accents,
punctuation, word order), candidates are found through a token index, and each candidate is
scored by fuzzy token overlap. A score at or above POTENTIAL_MATCH is a *potential* match
that an analyst must disposition; it is never an automatic finding that the applicant is the
listed person. Secondary identifiers (date of birth, list type) are compared and reported so
the analyst can clear false positives quickly.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
ALT_URL = "https://www.treasury.gov/ofac/downloads/alt.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"}

POTENTIAL_MATCH = 0.86      # at or above: potential match, analyst review required
STRONG_MATCH = 0.95         # at or above: treated as a strong match in the review queue
MAX_HITS = 5

# Synthetic entries (not real people or companies) used for the demonstration and offline.
DEMO_ENTRIES = [
    ("DEMO-1", "EMBARGO, Darius Kaveh", "individual", "DEMO-SDN", "DOB 14 Mar 1968; POB Tehran, Iran; synthetic test entry."),
    ("DEMO-2", "BLOCKWELL, Maximilian", "individual", "DEMO-SDN", "DOB 02 Nov 1975; nationality Russia; synthetic test entry."),
    ("DEMO-3", "LISTWORTH, Sancia Maria", "individual", "DEMO-SDN", "DOB 1981; synthetic test entry."),
    ("DEMO-4", "FROZEN HARBOR TRADING LLC", "entity", "DEMO-SDN", "Synthetic test entry."),
    ("DEMO-5", "NORTHGATE SHELL HOLDINGS LTD", "entity", "DEMO-SDN", "Synthetic test entry."),
]
_STOP = {"THE", "OF", "AND", "LLC", "LTD", "INC", "CO", "CORP", "SA", "AG", "LIMITED", "COMPANY", "DE", "AL", "BIN", "EL"}


@dataclass
class SanctionsList:
    entries: list[tuple]                 # (uid, primary name, type, program, remarks)
    names: list[tuple]                   # (entry index, normalised tokens, display name, is_alias)
    index: dict = field(default_factory=dict)   # token / 4-char prefix -> set of positions in `names`
    live: bool = False
    asof: str = ""
    source: str = ""

    @property
    def count(self) -> int:
        return len(self.entries)


@dataclass
class Hit:
    party: str               # who was screened ("Applicant", "Beneficial owner: ...")
    screened_name: str
    listed_name: str
    uid: str
    list_type: str           # individual / entity / vessel / aircraft
    program: str
    score: float
    dob_listed: str          # as published, "" if none
    dob_note: str            # "matches", "differs", "not published"
    alias: bool = False

    def describe(self) -> str:
        return (f"{self.party} '{self.screened_name}' resembles {self.listed_name} (OFAC SDN #{self.uid}, {self.program}, "
                f"{self.list_type}; name score {self.score:.0%}; date of birth {self.dob_note})")


def normalise(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return [t for t in s.split() if len(t) > 1 and t not in _STOP]


def _keys(tok: str) -> tuple[str, ...]:
    return (tok, "~" + tok[:4]) if len(tok) >= 4 else (tok,)


def build_list(sdn_rows: list[list[str]], alt_rows: list[list[str]] | None = None, live: bool = False,
               asof: str = "", source: str = "") -> SanctionsList:
    clean = lambda v: "" if v is None or str(v).strip() == "-0-" else str(v).strip()
    entries, pos = [], {}
    for r in sdn_rows:
        if len(r) < 4 or not clean(r[1]):
            continue
        pos[clean(r[0])] = len(entries)
        entries.append((clean(r[0]), clean(r[1]), clean(r[2]) or "entity", clean(r[3]), clean(r[11]) if len(r) > 11 else ""))
    names = [(i, normalise(e[1]), e[1], False) for i, e in enumerate(entries)]
    for r in alt_rows or []:
        if len(r) >= 4 and clean(r[0]) in pos and clean(r[3]):
            names.append((pos[clean(r[0])], normalise(r[3]), clean(r[3]), True))
    names = [n for n in names if n[1]]
    index: dict[str, set] = {}
    for k, (_, toks, _, _) in enumerate(names):
        for t in toks:
            for key in _keys(t):
                index.setdefault(key, set()).add(k)
    return SanctionsList(entries=entries, names=names, index=index, live=live, asof=asof, source=source)


def demo_list() -> SanctionsList:
    rows = [[u, n, t, p, "", "", "", "", "", "", "", rm] for u, n, t, p, rm in DEMO_ENTRIES]
    return build_list(rows, live=False, asof="synthetic sample", source="Built-in synthetic sample (offline)")


def load_list(live: bool = True, timeout: float = 25.0) -> SanctionsList:
    """OFAC SDN list with aliases; the synthetic sample alone if the download fails or live is off."""
    if not live:
        return demo_list()
    try:
        from concurrent.futures import ThreadPoolExecutor

        import requests

        def fetch(url: str) -> list[list[str]]:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return list(csv.reader(io.StringIO(r.content.decode("latin-1"))))

        with ThreadPoolExecutor(2) as ex:               # the two files download side by side
            f_sdn, f_alt = ex.submit(fetch, SDN_URL), ex.submit(fetch, ALT_URL)
            rows = f_sdn.result()
            try:
                alt_rows = f_alt.result()
            except Exception:
                alt_rows = []
        if len(rows) < 5000:
            raise ValueError("SDN file looks truncated")
        rows += [[u, n, t, p, "", "", "", "", "", "", "", rm] for u, n, t, p, rm in DEMO_ENTRIES]
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return build_list(rows, alt_rows, live=True, asof=f"downloaded {stamp}",
                          source="U.S. Treasury OFAC SDN list (sdn.csv, alt.csv) + synthetic test entries")
    except Exception:
        return demo_list()


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) < 4 or len(b) < 4:        # short tokens (initials, particles) must match exactly
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= 0.8 else 0.0


def name_score(q: list[str], e: list[str]) -> float:
    """Order-insensitive fuzzy token overlap in 0..1: the Dice coefficient on best token pairings,
    averaged with the better one-sided coverage so a missing middle name still matches
    ('Darius Embargo' against 'EMBARGO, Darius Kaveh')."""
    if not q or not e:
        return 0.0
    fwd = sum(max(_tok_sim(t, u) for u in e) for t in q)
    back = sum(max(_tok_sim(u, t) for t in q) for u in e)
    dice = (fwd + back) / (len(q) + len(e))
    if min(len(q), len(e)) < 2:
        return dice
    return 0.5 * dice + 0.5 * max(fwd / len(q), back / len(e))


_DOB = re.compile(r"DOB ((?:\d{2} \w{3} )?\d{4})(?: to (?:\d{2} \w{3} )?(\d{4}))?")


def _dob_compare(remarks: str, dob_iso: str) -> tuple[str, str]:
    found = _DOB.findall(remarks or "")
    if not found:
        return "", "not published"
    listed = "; ".join(a + (f" to {b}" if b else "") for a, b in found)
    if not dob_iso:
        return listed, "applicant's not provided"
    try:
        yr = int(dob_iso[:4])
    except ValueError:
        return listed, "applicant's not provided"
    for a, b in found:
        y0 = int(a[-4:])
        y1 = int(b) if b else y0
        if y0 <= yr <= y1:
            return listed, "year matches"
    return listed, "differs"


def screen(sl: SanctionsList, name: str, dob_iso: str = "", party: str = "Applicant", entity: bool = False) -> list[Hit]:
    """Potential matches for one party, best first."""
    q = normalise(name)
    if not q or (not entity and len(q) < 2):       # a single given name is not screenable
        return []
    cand: set[int] = set()
    for t in q:
        for key in _keys(t):
            cand |= sl.index.get(key, set())
    best: dict[int, tuple[float, str, bool]] = {}
    for k in cand:
        ei, toks, disp, alias = sl.names[k]
        s = name_score(q, toks)
        if s >= POTENTIAL_MATCH and s > best.get(ei, (0, "", False))[0]:
            best[ei] = (s, disp, alias)
    hits = []
    for ei, (s, disp, alias) in best.items():
        uid, primary, typ, prog, remarks = sl.entries[ei]
        listed, note = _dob_compare(remarks, dob_iso) if typ == "individual" else ("", "n/a for an entity")
        hits.append(Hit(party=party, screened_name=name.strip(), listed_name=primary if not alias else f"{primary} (a.k.a. {disp})",
                        uid=uid, list_type=typ, program=prog, score=float(s), dob_listed=listed, dob_note=note, alias=alias))
    hits.sort(key=lambda h: (-h.score, h.dob_note != "year matches"))
    return hits[:MAX_HITS]
