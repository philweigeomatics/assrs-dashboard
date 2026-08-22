"""
game_news.py — dated macro news for the blind replay game.

The player must not learn which stock they hold, so this deliberately produces
only MACRO context: sector demand cycles, policy, commodities, rates, trade
frictions, geopolitics. Anything company-specific — the name, its products, an
earnings miss, a management change — would hand over the answer, so the prompt
forbids it and fetch_news() additionally strips the company name and ticker
from results before they are shown.

Sources
  DeepSeek — turns a stock into a handful of macro risk keywords, once per game.
  GDELT 2.0 DOC API — free, no key, and it does serve historical windows
  (verified back at least 10 months). It is shared-IP throttled, roughly one
  request per 5 seconds but in practice bursty and unpredictable: identical
  queries return articles one moment and a rate-limit notice the next. So news
  is fetched ONLY on demand, cached per (day, query), and a throttled response
  is reported as "temporarily unavailable" rather than "no news", because those
  mean very different things to someone reading a chart.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# GDELT documents ~1 request per 5s, but the limit behaves like a shared-IP
# quota with a longer memory: after a burst, even a trivial query keeps
# returning the limit notice for a while, and a throttled response still costs
# ~12s to come back. 10s of self-pacing keeps normal play under it; heavy use
# will still hit a cooldown that no client-side spacing can shorten.
_MIN_GAP_SECONDS = 10.0
_last_call = [0.0]

_RISK_PROMPT = """\
You are a Chinese A-share macro strategist.

Given one listed company, list the MACRO risk factors that drive its share
price — the forces it shares with many other companies, not anything unique to
it.

HARD RULE: the reader must NOT be able to identify the company from your
output. Never name it, its ticker, its brands, its products, its executives, or
any fact true only of it. If a factor would give it away, generalise it until
it would fit dozens of firms.

Good: "steel and coking coal input costs", "property completions", "export
tariffs on consumer electronics", "domestic semiconductor capex", "policy
support for renewables", "consumer confidence and discretionary spending".
Bad: anything naming a company, a specific plant, or a single product line.

search_terms feed a keyword news search that does EXACT matching, so long
phrases match nothing. Each must be ONE or TWO words — a noun a journalist
would actually write in a headline. "semiconductor", "tariffs", "lithium",
"property", "shipbuilding". Never a descriptive phrase like "domestic
semiconductor capital expenditure".

Return ONLY raw JSON (start { end }), no markdown:
{
  "themes": ["3-6 macro themes in English, readable by a human"],
  "cn": ["the same themes in Chinese, same order"],
  "search_terms": ["3-5 ONE-or-TWO word search keywords, no company names"]
}
"""


def derive_risk_factors(ticker: str, name: str, industry: str = "") -> dict:
    """
    One DeepSeek call turning a stock into macro-only keywords.

    Uses reasoning_effort="low": this is classification into themes, not
    multi-step derivation.
    """
    import ai_client
    try:
        data = ai_client.call_json(
            _RISK_PROMPT,
            f"Company: {name} ({ticker})\nIndustry: {industry or 'unknown'}",
            max_tokens=3000, temperature=0.3, reasoning_effort="low",
        )
        themes = [str(x) for x in (data.get("themes") or [])][:6]
        terms = [str(x) for x in (data.get("search_terms") or [])][:5]
        return {"ok": bool(themes), "themes": themes,
                "cn": [str(x) for x in (data.get("cn") or [])][:6],
                "search_terms": terms or themes[:3]}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "themes": [], "cn": [],
                "search_terms": []}


def _throttle() -> None:
    """GDELT asks for one request per 5s; keep a margin and self-pace."""
    gap = time.time() - _last_call[0]
    if gap < _MIN_GAP_SECONDS:
        time.sleep(_MIN_GAP_SECONDS - gap)
    _last_call[0] = time.time()


def fetch_news(query: str, day: str, window_days: int = 1,
               max_records: int = 8, timeout: int = 25,
               redact: "list[str] | None" = None) -> dict:
    """
    Articles GDELT saw in the window ENDING on `day` (YYYY-MM-DD).

    Looks backwards only — a replay must never surface an article published
    after the session being played.

    Returns {"ok", "articles", "throttled", "reason"}. `throttled` is kept
    distinct from an empty result on purpose: "the source is busy" and "nothing
    happened that day" would otherwise look identical to the player.
    """
    try:
        d = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return {"ok": False, "articles": [], "throttled": False,
                "reason": "bad date"}

    start = (d - timedelta(days=window_days)).strftime("%Y%m%d%H%M%S")
    end = d.strftime("%Y%m%d235959")

    try:
        _throttle()
        r = requests.get(GDELT_URL, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"},
                         params={"query": query, "format": "json",
                                 "mode": "artlist", "maxrecords": str(max_records),
                                 "startdatetime": start, "enddatetime": end})
        body = r.text or ""
        if "limit requests" in body or r.status_code == 429:
            return {"ok": False, "articles": [], "throttled": True,
                    "reason": "GDELT rate limit — try again in a few seconds"}
        arts = r.json().get("articles", []) or []
    except Exception as exc:
        return {"ok": False, "articles": [], "throttled": False,
                "reason": f"{type(exc).__name__}: {exc}"}

    bad = [b.lower() for b in (redact or []) if b and len(b) > 1]
    out = []
    for a in arts:
        title = str(a.get("title", "")).strip()
        if not title:
            continue
        # Last line of defence: a macro query can still return a piece naming
        # the company. Dropping those keeps the stock hidden.
        if any(b in title.lower() for b in bad):
            continue
        seen = str(a.get("seendate", ""))
        out.append({
            "title": title,
            "domain": a.get("domain", ""),
            "url": a.get("url", ""),
            "seendate": f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 else seen,
            "language": a.get("language", ""),
        })
    return {"ok": True, "articles": out, "throttled": False, "reason": None}


MACRO_QUERY = ('(china economy OR "chinese stocks" OR "trade war" OR tariff '
               'OR "interest rate" OR geopolitics OR war)')


# Words that carry no search signal but bloat a phrase into unmatchability.
_STOP = {"the", "a", "an", "of", "on", "in", "for", "and", "or", "to", "with",
         "domestic", "global", "chinese", "china", "policy", "risk", "sector",
         "demand", "cycle", "costs", "cost", "prices", "price"}


def _condense(term: str) -> str:
    """
    Reduce a descriptive theme to something GDELT can actually match.

    A quoted string is an EXACT phrase match against article text, so
    "export tariffs on consumer electronics" matches essentially nothing — no
    article contains that precise five-word run.

    The obvious fix, keeping the first two content words, is also wrong: it
    manufactures phrases that never occur. "steel and coking coal input costs"
    became "steel coking", which no journalist has ever written. Words are only
    kept together when they were ADJACENT in the original, which is the only
    case where the pair plausibly appears in prose. Anything longer collapses
    to its single most distinctive word, unquoted, which always matches.
    """
    words = [w for w in "".join(
        c if c.isalnum() or c.isspace() else " " for c in term).split()]
    content = [w for w in words if len(w) > 2 and w.lower() not in _STOP]
    if not content:
        return ""
    # Adjacent pair in the source → safe to keep as a phrase.
    if len(content) == 2 and words.index(content[0]) + 1 == words.index(content[1]):
        return f"{content[0]} {content[1]}"
    # Otherwise a single word: longest is a decent proxy for most specific.
    return max(content, key=len)


def build_query(search_terms: "list[str] | None", max_terms: int = 4) -> str:
    """
    Theme terms OR-ed into ONE request — never one request per theme, which
    would multiply against GDELT's rate limit for no benefit.

    Single words go in bare; a surviving two-word pair is quoted, which is
    tight enough to still match real headlines.
    """
    seen, parts = set(), []
    for raw in (search_terms or []):
        t = _condense(str(raw))
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        parts.append(f'"{t}"' if " " in t else t)
        if len(parts) >= max_terms:
            break
    if not parts:
        return MACRO_QUERY
    # Anchored to China so a bare word like "semiconductor" doesn't return
    # global noise unrelated to the market being played.
    return "(" + " OR ".join(parts) + ") (china OR chinese)"
