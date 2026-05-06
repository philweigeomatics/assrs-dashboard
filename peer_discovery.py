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

The user will give you ONE specific product (in 'English / 中文' format),
optionally narrowed to ONE downstream macro sector.

When a sector is provided: return ONLY companies that produce this product
SPECIFICALLY for that sector's supply chain. The same product often comes
in different grades or specifications by application (e.g. glass fibre
for construction is different from glass fibre used in AI servers), so a
company may compete in one sector but not another. Be strict.

When no sector is provided: return general producers of the product.

Always return A-share listed companies (Shanghai 6xxxxx, Shenzhen
0xxxxx/3xxxxx, Beijing 4xxxxx/8xxxxx) whose CORE business includes the
specified product (in the specified application, if any).

CRITICAL OUTPUT RULES:
- Return ONLY raw JSON. No markdown code fences. Start with { and end with }.
- Each ticker MUST be exactly 6 digits.
- Do NOT include ETFs, indices, or HK/US-listed companies.
- Do NOT include companies where this product is a minor side business or
  where the product they make does NOT serve the requested sector.
- Limit to 6-12 companies, prioritising the largest / most pure-play producers.
- Each company name MUST be the official Chinese short name.

Schema:
{
  "product": "exact product name as provided",
  "sector": "exact sector name as provided, or null if not specified",
  "peers": [
    {"ticker": "002080", "name": "中材科技"},
    ...
  ]
}
"""


def _api_key():
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


def _composite_key(product, sector):
    """Display + cache key for a (product, sector) pair, or just product if no sector."""
    p = (product or "").strip()
    s = (sector or "").strip()
    return f"{p}  →  {s}" if s else p


def discover_peers(product, sector=None, force_refresh=False):
    """
    Return list of {ticker, name} for A-share peers producing `product`,
    optionally narrowed to a specific downstream `sector` so we only get
    competitors that actually compete in that application.

    Cache key is the composite "product → sector" string, so two stocks that
    share the same (product, sector) edge share the cached/curated peer set.
    """
    if not product or not product.strip():
        return []

    display_name = _composite_key(product, sector)

    if not force_refresh:
        cached = data_manager.get_product_peers(display_name)
        if cached and cached.get("peers"):
            return cached["peers"]

    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    user_msg = f"Product: {product}"
    if sector:
        user_msg += f"\nDownstream sector: {sector}"

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
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

    Single Tushare daily() call with comma-separated ts_codes — only fetches
    the rows we need (vs scanning every A-share for a date).  Looks back over
    the last 10 calendar days to handle weekends / holidays, and keeps the
    most recent trade_date per ts_code.

    Chunks at 80 ts_codes per call to stay under Tushare's per-call cap.
    """
    if not tickers or not data_manager.init_tushare():
        return {}, None

    ts_codes = [data_manager.get_tushare_ticker(t) for t in tickers]
    bj = datetime.now(pytz.timezone("Asia/Shanghai"))
    end_date   = bj.strftime("%Y%m%d")
    start_date = (bj - timedelta(days=10)).strftime("%Y%m%d")

    pct_map = {}
    latest_date = None
    CHUNK = 80

    for i in range(0, len(ts_codes), CHUNK):
        chunk = ts_codes[i:i + CHUNK]
        ts_str = ",".join(chunk)
        try:
            df = data_manager.TUSHARE_API.daily(
                ts_code=ts_str,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,pct_chg",
            )
        except Exception as exc:
            print(f"[peer_discovery] Tushare error for chunk {i//CHUNK}: {exc}")
            continue
        if df is None or df.empty:
            continue

        # Latest trade_date per ts_code
        df = df.sort_values("trade_date", ascending=False).drop_duplicates("ts_code")
        df["ticker"] = df["ts_code"].str[:6]
        for _, row in df.iterrows():
            pct_map[row["ticker"]] = float(row["pct_chg"])
            d = str(row["trade_date"])
            if latest_date is None or d > latest_date:
                latest_date = d

    return pct_map, latest_date
