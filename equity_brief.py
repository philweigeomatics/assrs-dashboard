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

import requests

import data_manager


_ENDPOINT = "https://api.deepseek.com/chat/completions"
_MODEL    = "deepseek-v4-flash"


def _api_key() -> str:
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


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
    df = data_manager.db.read_table(
        "equity_brief_cache",
        filters={"ticker": ticker, "section": section},
    )
    if df is None or df.empty:
        return None
    row = df.iloc[0].to_dict()
    try:
        return {
            "payload":      json.loads(row["payload"]),
            "generated_at": row.get("generated_at"),
        }
    except (json.JSONDecodeError, TypeError):
        return None


def _write_cache(ticker: str, section: str, payload: Any) -> None:
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


# ── DeepSeek call helper ──────────────────────────────────────────────────────

def _call_deepseek(system_prompt: str, user_msg: str, max_tokens: int = 2000) -> dict:
    """
    Calls DeepSeek and returns parsed JSON dict.
    Raises RuntimeError on transport / parse failure.
    """
    api_key = _api_key()
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens":  max_tokens,
    }
    try:
        resp = requests.post(
            _ENDPOINT, json=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type":  "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError("AI API timed out after 60 s.")
    except requests.RequestException as exc:
        raise RuntimeError(f"AI API failed: {exc}") from exc

    try:
        resp_json = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"DeepSeek response was not JSON. Status {resp.status_code}. "
            f"Body preview: {resp.text[:400]}"
        ) from exc

    # Surface any API-level error (wrong model name, quota, etc.)
    if "error" in resp_json:
        err = resp_json["error"]
        raise RuntimeError(
            f"DeepSeek API error — {err.get('type', 'unknown')}: "
            f"{err.get('message', err)}"
        )

    try:
        choice  = resp_json["choices"][0]
        message = choice["message"]
        raw     = message.get("content", "").strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected DeepSeek response shape. Full response: {resp_json}"
        ) from exc

    if not raw:
        finish_reason    = choice.get("finish_reason", "unknown")
        reasoning_tokens = (
            resp_json.get("usage", {})
                     .get("completion_tokens_details", {})
                     .get("reasoning_tokens", 0)
        )
        # Reasoning model (e.g. deepseek-v4-flash) exhausted max_tokens on
        # its thinking trace before writing any output.
        if finish_reason == "length" and reasoning_tokens > 0:
            raise RuntimeError(
                f"Model ran out of tokens before writing output "
                f"({reasoning_tokens} tokens consumed by reasoning trace). "
                f"Increase max_tokens and try again."
            )
        raise RuntimeError(
            f"DeepSeek returned empty content "
            f"(finish_reason={finish_reason!r}). Full response: {resp_json}"
        )

    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AI returned invalid JSON ({exc}).\n"
            f"Full raw response:\n{raw[:600]}"
        ) from exc


# ── Prompts ───────────────────────────────────────────────────────────────────

_OVERVIEW_PROMPT = """\
You are an elite Chinese A-share equity analyst.

In 3-4 sentences, describe what this company actually does — the products
or services it sells, who its customers are, and what its main revenue
drivers are. Be concrete and factual, not promotional.

Avoid filler phrases like "is a leading provider", "is committed to", or
"offers innovative solutions". Name actual products / customer types /
end markets.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- "summary" is one paragraph (3-4 sentences) in English.
- "tagline" is one short phrase (under 10 words) describing the company's
  positioning in plain English.

Schema:
{
  "tagline": "...",
  "summary": "..."
}
"""


_PESTEL_PROMPT = """\
You are an elite Chinese A-share equity analyst.

Given a company, generate a concise PESTEL analysis focused on factors that
materially affect this specific company's earnings or valuation in the next
12-24 months. Avoid generic statements that could apply to any company.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- Each of the 6 dimensions has 2-3 bullet strings.
- Each bullet is one sentence, specific to this company's industry & market.

Schema:
{
  "political":     ["...", "..."],
  "economic":      ["...", "..."],
  "social":        ["...", "..."],
  "technological": ["...", "..."],
  "environmental": ["...", "..."],
  "legal":         ["...", "..."]
}
"""


_PORTERS_PROMPT = """\
You are an elite Chinese A-share equity analyst.

Generate Porter's Five Forces for this company. Score each force 1-5 where
1 = very weak/favourable to the company, 5 = very strong/threatening.
Summary is one sentence specific to this company's competitive position.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.

Schema:
{
  "rivalry":      {"score": 1-5, "summary": "..."},
  "suppliers":    {"score": 1-5, "summary": "..."},
  "buyers":       {"score": 1-5, "summary": "..."},
  "substitutes":  {"score": 1-5, "summary": "..."},
  "new_entrants": {"score": 1-5, "summary": "..."}
}
"""


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

def get_company_overview(ticker, name, industry, force_refresh=False):
    """Returns {payload, generated_at}; payload = {tagline, summary}."""
    if not force_refresh:
        cached = _read_cache(ticker, "overview")
        if cached:
            return cached
    user_msg = f"Company: {name} ({ticker})\nIndustry: {industry}"
    payload  = _call_deepseek(_OVERVIEW_PROMPT, user_msg, max_tokens=1500)
    _write_cache(ticker, "overview", payload)
    return {"payload": payload, "generated_at": datetime.utcnow().isoformat()}


def get_pestel(ticker: str, name: str, industry: str, force_refresh: bool = False) -> dict:
    """Returns {payload, generated_at}; payload is the parsed PESTEL dict."""
    if not force_refresh:
        cached = _read_cache(ticker, "pestel")
        if cached:
            return cached
    user_msg = f"Company: {name} ({ticker})\nIndustry: {industry}"
    payload  = _call_deepseek(_PESTEL_PROMPT, user_msg, max_tokens=2500)
    _write_cache(ticker, "pestel", payload)
    return {"payload": payload, "generated_at": datetime.utcnow().isoformat()}


def get_porters(ticker: str, name: str, industry: str, force_refresh: bool = False) -> dict:
    if not force_refresh:
        cached = _read_cache(ticker, "porters")
        if cached:
            return cached
    user_msg = f"Company: {name} ({ticker})\nIndustry: {industry}"
    payload  = _call_deepseek(_PORTERS_PROMPT, user_msg, max_tokens=2000)
    _write_cache(ticker, "porters", payload)
    return {"payload": payload, "generated_at": datetime.utcnow().isoformat()}


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
    payload = _call_deepseek(_SWOT_PROMPT, user_msg, max_tokens=2500)
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
    raw      = _call_deepseek(_COMPETITORS_PROMPT, user_msg, max_tokens=3000)

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


def save_competitors_curated(ticker: str, competitors: list[dict]) -> None:
    """Admin-only: persist the curated competitor list back to the cache."""
    payload = {"competitors": competitors}
    _write_cache(ticker, "competitors", payload)
