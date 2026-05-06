"""
Peer Discovery — Phase 1 of lead-lag analysis.

Two responsibilities (kept thin and decoupled from UI):
  - discover_peers(product)   → DeepSeek call (cached in DB) for A-share peers.
  - fetch_latest_pct_chg(...) → Single Tushare daily call for batch pct_chg.

DB interactions go through data_manager. The page code in
pages/lead_lag_analysis.py wires these together.
"""

import json
from datetime import datetime, timedelta

import requests
import pytz

import data_manager

# ── DeepSeek ──────────────────────────────────────────────────────────────────
_ENDPOINT = "https://api.deepseek.com/chat/completions"
_MODEL    = "deepseek-chat"

_SYSTEM_PROMPT = """\
You are an elite Chinese A-share equity analyst.

The user will give you ONE specific product or component (in 'English / 中文' format).
Return a list of A-share listed companies (Shanghai 6xxxxx, Shenzhen 0xxxxx/3xxxxx,
Beijing 4xxxxx/8xxxxx) whose CORE business includes manufacturing or supplying that product.

CRITICAL OUTPUT RULES:
- Return ONLY raw JSON. No markdown code fences. Start with { and end with }.
- Each ticker MUST be exactly 6 digits.
- Do NOT include ETFs, indices, or HK/US-listed companies.
- Do NOT include companies where this product is a minor side business.
- Limit to 6-12 companies, prioritising the largest / most pure-play producers.
- Each company name MUST be the official Chinese short name.

Schema:
{
  "product": "exact bilingual product name as provided",
  "peers": [
    {"ticker": "002080", "name": "中材科技"},
    ...
  ]
}
"""


def _api_key():
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


def discover_peers(display_name, force_refresh=False):
    """
    Return list of {ticker, name} for A-share peers producing `display_name`.
    Hits DeepSeek on cache miss; otherwise returns the cached DB entry.
    Raises RuntimeError on API failure.
    """
    if not display_name or not display_name.strip():
        return []

    if not force_refresh:
        cached = data_manager.get_product_peers(display_name)
        if cached and cached.get("peers"):
            return cached["peers"]

    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Product: {display_name}"},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(
            _ENDPOINT, json=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type":  "application/json"},
            timeout=40,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError("DeepSeek API timed out after 40 s.")
    except requests.RequestException as exc:
        raise RuntimeError(f"DeepSeek API failed: {exc}") from exc

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"DeepSeek returned invalid JSON ({exc}). Preview: {raw[:200]}"
        ) from exc

    raw_peers = data.get("peers", [])
    cleaned = []
    seen = set()
    for p in raw_peers:
        t = str(p.get("ticker", "")).strip().zfill(6)
        n = (p.get("name") or "").strip()
        if len(t) == 6 and t.isdigit() and t not in seen:
            cleaned.append({"ticker": t, "name": n})
            seen.add(t)

    data_manager.upsert_product_peers(display_name, cleaned)
    return cleaned


# ── Tushare batch pct_chg ─────────────────────────────────────────────────────

def fetch_latest_pct_chg(tickers):
    """
    Returns ({ticker: pct_chg_float}, trade_date_str_or_None).
    One Tushare daily() call for all stocks on the latest trading day; walks
    back up to 8 days if today/recent are non-trading.
    """
    if not tickers or not data_manager.init_tushare():
        return {}, None

    bj = datetime.now(pytz.timezone("Asia/Shanghai"))
    ts_codes = {data_manager.get_tushare_ticker(t) for t in tickers}

    for delta in range(8):
        date = (bj - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            df = data_manager.TUSHARE_API.daily(
                trade_date=date,
                fields="ts_code,pct_chg,close",
            )
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df[df["ts_code"].isin(ts_codes)].copy()
        if df.empty:
            continue
        df["ticker"] = df["ts_code"].str[:6]
        return dict(zip(df["ticker"], df["pct_chg"])), date

    return {}, None
