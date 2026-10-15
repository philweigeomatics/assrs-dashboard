"""
equity_brief.py — AI-generated qualitative content for the Equity Brief page.

Four content types, each cached in `equity_brief_cache` keyed by (ticker, section):
  - pestel       → political/economic/social/technological/environmental/legal bullets
  - porters      → 5 forces, each with a 1–5 score + summary
  - swot         → strengths/weaknesses/opportunities/threats bullets
  - competitors  → 4 A-share peers, validated against stock_basic

Cache reads/writes use data_manager.db (works for both SQLite and Supabase).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import ai_client
import data_manager


# ── DB cache ──────────────────────────────────────────────────────────────────

def ensure_equity_brief_cache_table() -> None:
    """
    Idempotent migration. SQLite only — Supabase users add the table manually
    via SQL editor (see migration block in this module's docstring).
    """
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return  # Supabase: DBNAME isn't defined in production branch

    from db_config import DBNAME
    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_brief_cache (
                ticker       TEXT NOT NULL,
                section      TEXT NOT NULL,
                payload      TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (ticker, section)
            )
        """)
        conn.commit()


def _read_cache(ticker: str, section: str) -> dict | None:
    try:
        df = data_manager.db.read_table(
            "equity_brief_cache",
            filters={"ticker": ticker, "section": section},
        )
        if df is None or df.empty:
            return None
        row = df.iloc[0].to_dict()
        return {
            "payload":      json.loads(row["payload"]),
            "generated_at": row.get("generated_at"),
        }
    except Exception:
        # Table missing, schema mismatch, Supabase API error, JSON decode error, etc.
        # Treat any failure as a cache miss — the caller will regenerate.
        return None


def _write_cache(ticker: str, section: str, payload: Any) -> bool:
    """
    Replace one section's row. Returns whether it was written.

    Best-effort for AI output, where a failed write only costs a regeneration
    later. NOT best-effort for a curated peer list, which is a person's edit
    rather than a cache — so the result is returned and that caller checks it.
    """
    try:
        data_manager.db.delete_records(
            "equity_brief_cache",
            filters={"ticker": ticker, "section": section},
        )
        data_manager.db.insert_records("equity_brief_cache", [{
            "ticker":       ticker,
            "section":      section,
            "payload":      json.dumps(payload, ensure_ascii=False),
            "generated_at": datetime.utcnow().isoformat(),
        }])
        return True
    except Exception as exc:
        print(f"[equity_brief] ⚠️ Cache write failed for {ticker}/{section}: {exc}")
        return False



# ── Prompts ───────────────────────────────────────────────────────────────────

# Overview, PESTEL and Porter's are written about the same company from the
# same two lines of input, so they are asked for together: one call, one
# reasoning pass over what the company is, three sections out. Each section's
# brief lives in _PROFILE_TASKS and the prompt is assembled from whichever
# ones are wanted — all three the first time, just one on a "Regen".

_PROFILE_HEADER = """\
You are an elite Chinese A-share equity analyst. Write the sections listed
below about ONE company. They should agree with each other: the forces you
score and the PESTEL factors you raise should be about the business the
overview describes.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- The top-level keys are exactly the section names below, nothing else.
"""

_SINGLE_HEADER = """\
You are an elite Chinese A-share equity analyst. Write the section below
about ONE company.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
"""

_PROFILE_TASKS: dict[str, tuple[str, str]] = {
    "overview": (
        """\
In 3-4 sentences, describe what this company actually does — the products
or services it sells, who its customers are, and what its main revenue
drivers are. Be concrete and factual, not promotional. Avoid filler phrases
like "is a leading provider", "is committed to", or "offers innovative
solutions". Name actual products / customer types / end markets.
"summary" is one paragraph (3-4 sentences) in English. "tagline" is one
short phrase (under 10 words) describing the company's positioning in
plain English.""",
        """{"tagline": "...", "summary": "..."}""",
    ),
    "pestel": (
        """\
A concise PESTEL analysis focused on factors that materially affect this
specific company's earnings or valuation in the next 12-24 months. Avoid
generic statements that could apply to any company. Each of the 6
dimensions has 2-3 bullet strings; each bullet is one sentence, specific to
this company's industry & market.""",
        """{"political": ["...", "..."], "economic": ["...", "..."],
   "social": ["...", "..."], "technological": ["...", "..."],
   "environmental": ["...", "..."], "legal": ["...", "..."]}""",
    ),
    "porters": (
        """\
Porter's Five Forces. Score each force 1-5 where 1 = very weak/favourable
to the company, 5 = very strong/threatening. Each summary is one sentence
specific to this company's competitive position.""",
        """{"rivalry": {"score": 1-5, "summary": "..."},
   "suppliers": {"score": 1-5, "summary": "..."},
   "buyers": {"score": 1-5, "summary": "..."},
   "substitutes": {"score": 1-5, "summary": "..."},
   "new_entrants": {"score": 1-5, "summary": "..."}}""",
    ),
}

#: The keys each section must come back with, or it is not written.
_PROFILE_KEYS: dict[str, tuple[str, ...]] = {
    "overview": ("summary",),
    "pestel": ("political", "economic", "social",
               "technological", "environmental", "legal"),
    "porters": ("rivalry", "suppliers", "buyers", "substitutes", "new_entrants"),
}

PROFILE_SECTIONS = tuple(_PROFILE_TASKS)


def _profile_prompt(sections) -> str:
    if len(sections) == 1:
        # One section is asked for in its own shape, not wrapped in
        # {"pestel": {...}}: the extra level of nesting is where the model
        # dropped a closing bracket on a live PESTEL-only regeneration.
        s = sections[0]
        return (f"{_SINGLE_HEADER}\n{_PROFILE_TASKS[s][0]}\n\n"
                f"Schema:\n{_PROFILE_TASKS[s][1]}\n")
    tasks = "\n\n".join(f"## {s}\n{_PROFILE_TASKS[s][0]}" for s in sections)
    schema = ",\n  ".join(f'"{s}": {_PROFILE_TASKS[s][1]}' for s in sections)
    return f"{_PROFILE_HEADER}\n{tasks}\n\nSchema:\n{{\n  {schema}\n}}\n"


_SWOT_PROMPT = """\
You are an elite Chinese A-share equity analyst.

Given a company AND its key financial metrics (provided), generate a SWOT
that is GROUNDED IN THE NUMBERS — reference specific metrics in your bullets
where relevant ("ROE of 22% supports…", "net debt of -1.2B 元 means…").

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- Each list has 3 bullet strings, each one sentence.

Schema:
{
  "strengths":     ["...", "...", "..."],
  "weaknesses":    ["...", "...", "..."],
  "opportunities": ["...", "...", "..."],
  "threats":       ["...", "...", "..."]
}
"""


_COMPETITORS_PROMPT = """\
You are an elite Chinese A-share equity analyst.

Given a target company and its core business segment, return EXACTLY 4 \
A-share listed competitors that operate in the SAME segment and compete \
for the SAME purchasing decisions.

UNIVERSE RULES:
- A-share only (Shanghai or Shenzhen, 6-digit tickers).
- Exclude ETFs, indices, B-shares, HK/US listings.

DIRECT COMPETITOR DEFINITION (BOTH conditions must hold):
1. Must sell products/services that can DIRECTLY REPLACE the target's \
offering in the SAME use-case or project type.
2. Must compete for the SAME contracts/tenders or purchasing decisions \
from the SAME type of customers (e.g. same ministries, operators, OEMs).

EXCLUSIONS — do NOT include:
- Suppliers to the target (different value-chain layer upstream).
- Customers of the target (different value-chain layer downstream).
- Companies mainly in a different layer: raw materials, foundry, \
contract manufacturing, pure system integrators when the target is a \
component vendor (or vice versa).
- Companies that are only 同赛道 concept peers without substitutable products.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown fences.
- Return EXACTLY 4 competitors. Each ticker MUST be exactly 6 digits.
- "why" must follow this template: \
"Both provide [product/solution] to [customer type or key customers] \
for [use-case/project], and can be chosen as substitutes in the same tenders."

Schema:
{
  "competitors": [
    {"ticker": "000001", "name": "公司名", "why": "Both provide ..."},
    {"ticker": "000002", "name": "公司名", "why": "Both provide ..."},
    {"ticker": "000003", "name": "公司名", "why": "Both provide ..."},
    {"ticker": "000004", "name": "公司名", "why": "Both provide ..."}
  ]
}
"""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_profile(ticker: str, name: str, industry: str,
                     sections=PROFILE_SECTIONS) -> dict:
    """
    One AI call for any subset of overview / PESTEL / Porter's.

    Each section that comes back with its expected keys is cached under its
    own row, so everything that reads the cache per section keeps working.
    A section the model dropped or mangled is left unwritten and named in
    `failed` — the others are still good and are kept.

    Returns {"written": {section: {payload, generated_at}}, "failed": [...]}.
    """
    sections = [s for s in PROFILE_SECTIONS if s in sections]
    if not sections:
        raise ValueError("no profile section requested")
    user_msg = f"Company: {name} ({ticker})\nIndustry: {industry}"
    # Separately these had 8,000 / 10,000 / 8,000, reasoning included.
    # JSON mode because one bare quote inside a Chinese sentence — seen live
    # on 中兴通讯's PESTEL — would otherwise sink all three sections at once.
    for attempt in (1, 2):
        try:
            raw = ai_client.call_json(_profile_prompt(sections), user_msg,
                                      max_tokens=6000 + 4000 * len(sections),
                                      timeout=60 + 30 * len(sections),
                                      json_mode=True)
            break
        except RuntimeError as exc:
            # A dropped bracket is a slip, not a verdict: asking again almost
            # always parses. Timeouts and API errors are not retried.
            if attempt == 2 or "invalid JSON" not in str(exc):
                raise
    if len(sections) == 1 and isinstance(raw, dict) and sections[0] not in raw:
        raw = {sections[0]: raw}
    now = datetime.utcnow().isoformat()
    written, failed = {}, []
    for s in sections:
        part = raw.get(s) if isinstance(raw, dict) else None
        if isinstance(part, dict) and all(k in part for k in _PROFILE_KEYS[s]):
            _write_cache(ticker, s, part)
            written[s] = {"payload": part, "generated_at": now}
        else:
            failed.append(s)
    return {"written": written, "failed": failed}


def _profile_section(ticker: str, name: str, industry: str, section: str,
                     force_refresh: bool) -> dict:
    """
    One section, generated alongside whichever of its siblings are missing.

    A first visit fills all three with one call, so the two reads after it
    are cache hits. A forced Regen redoes only the section asked for.
    """
    if not force_refresh:
        cached = _read_cache(ticker, section)
        if cached:
            return cached
        todo = [s for s in PROFILE_SECTIONS
                if s == section or _read_cache(ticker, s) is None]
    else:
        todo = [section]
    out = generate_profile(ticker, name, industry, todo)
    if section not in out["written"]:
        raise RuntimeError(f"AI 返回的 {section} 不完整，没有保存")
    return out["written"][section]


def get_company_overview(ticker, name, industry, force_refresh=False):
    """Returns {payload, generated_at}; payload = {tagline, summary}."""
    return _profile_section(ticker, name, industry, "overview", force_refresh)


def get_pestel(ticker: str, name: str, industry: str, force_refresh: bool = False) -> dict:
    """Returns {payload, generated_at}; payload is the parsed PESTEL dict."""
    return _profile_section(ticker, name, industry, "pestel", force_refresh)


def get_porters(ticker: str, name: str, industry: str, force_refresh: bool = False) -> dict:
    return _profile_section(ticker, name, industry, "porters", force_refresh)


def get_swot(ticker: str, name: str, industry: str,
             metrics_summary: str, force_refresh: bool = False) -> dict:
    """metrics_summary = short text block of 5-10 key metrics for grounding."""
    if not force_refresh:
        cached = _read_cache(ticker, "swot")
        if cached:
            return cached
    user_msg = (
        f"Company: {name} ({ticker})\n"
        f"Industry: {industry}\n\n"
        f"Key metrics:\n{metrics_summary}"
    )
    payload = ai_client.call_json(_SWOT_PROMPT, user_msg, max_tokens=10000)
    _write_cache(ticker, "swot", payload)
    return {"payload": payload, "generated_at": datetime.utcnow().isoformat()}


def get_competitors(
    ticker: str,
    name: str,
    industry: str,
    force_refresh: bool = False,
    core_products: list | None = None,
) -> dict:
    """
    Returns {payload, generated_at}. payload = {"competitors": [...]}.

    core_products: optional list of product/service strings from the supply
    chain graph — passed to the AI so it can identify substitutable competitors
    rather than same-space-but-different-product peers.

    All returned tickers are validated against stock_basic — any unknown
    ticker is silently dropped before caching. The caller can rely on
    every ticker existing in stock_basic.
    """
    if not force_refresh:
        cached = _read_cache(ticker, "competitors")
        if cached:
            return cached

    products_line = ""
    if core_products:
        products_line = "\nCore products/services: " + ", ".join(core_products[:8])

    user_msg = (
        f"Target company: {name} ({ticker})\n"
        f"Industry: {industry}"
        f"{products_line}"
    )
    raw      = ai_client.call_json(_COMPETITORS_PROMPT, user_msg, max_tokens=9000)

    # Validate each peer against stock_basic
    raw_peers = raw.get("competitors", [])
    validated, seen = [], set()
    for p in raw_peers:
        t = str(p.get("ticker", "")).strip().zfill(6)
        n = (p.get("name") or "").strip()
        why = (p.get("why") or "").strip()
        if not (len(t) == 6 and t.isdigit() and t not in seen and t != ticker):
            continue
        # Cross-check against stock_basic
        try:
            ts_code = data_manager.get_tushare_ticker(t)
            df = data_manager.db.read_table(
                "stock_basic", filters={"ts_code": ts_code}, columns="name", limit=1
            )
            if df is None or df.empty:
                continue
        except Exception:
            continue
        validated.append({"ticker": t, "name": n, "why": why})
        seen.add(t)

    payload = {"competitors": validated}
    _write_cache(ticker, "competitors", payload)
    return {"payload": payload, "generated_at": datetime.utcnow().isoformat()}


def save_competitors_curated(ticker: str, competitors: list[dict]) -> bool:
    """Admin-only: persist the curated competitor list. True if it was written."""
    return _write_cache(ticker, "competitors", {"competitors": competitors})
