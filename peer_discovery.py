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
_MODEL    = "deepseek-v4-flash"

_SYSTEM_PROMPT = """\
You are an elite Chinese A-share equity analyst.

The user will give you ONE specific product or service (in 'English / 中文' format),
optionally narrowed to ONE downstream macro sector.

Your task: find A-share listed companies whose PRIMARY business is SUPPLYING
this product or service to other businesses or consumers.

"Supplying" means:
- Manufacturing or producing a physical component/material
- Providing a service (e.g. cloud computing, logistics, testing)
- Processing or refining a commodity

EXCLUDE any company that is primarily a BUYER or END-USER of this product/service
— i.e. a company that purchases this as an input to build something else entirely.
Example: for "robot joint reducer", return reducer manufacturers, NOT industrial
robot assemblers (who buy reducers as a component).

When a sector is provided: return ONLY companies that supply this product/service
SPECIFICALLY into that sector's supply chain. Be strict about application fit.

CRITICAL OUTPUT RULES:
- Return ONLY raw JSON. No markdown code fences. Start with { and end with }.
- Each ticker MUST be exactly 6 digits.
- Do NOT include ETFs, indices, or HK/US-listed companies.
- Limit to the TOP 5 companies by market share / pure-play relevance.
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


def composite_key(product, sector):
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

    display_name = composite_key(product, sector)

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


# ── Layer Stock Discovery ─────────────────────────────────────────────────────

_LAYER_STOCK_PROMPT = """\
You are an elite Chinese A-share equity analyst.

Given a specific layer in a supply chain, find the TOP 3 A-share listed companies
whose PRIMARY or MAJORITY revenue comes from SUPPLYING the products or services
listed for that layer.

Context provided:
- Sector: the overall industry theme
- Layer position in chain (1 = most upstream)
- Layer name: what this layer does
- Key items: the products or services that belong in this layer

"Supplying" means producing, manufacturing, processing, or delivering these items
to downstream customers — NOT companies that purchase these items as inputs to
build something else entirely.

Rank by: (1) revenue concentration in this layer's products, (2) market size.

CRITICAL OUTPUT RULES:
- Return ONLY raw JSON. No markdown, no fences. Start with { end with }.
- Each ticker MUST be exactly 6 digits (A-share only; no ETFs, no HK/US stocks).
- primary_product must name one specific item from the layer's key items list.
- Return exactly 3 stocks (or fewer only if fewer than 3 qualified companies exist).

Schema:
{
  "stocks": [
    {"ticker": "000001", "name": "公司名", "primary_product": "specific item they supply"}
  ]
}
"""


def discover_layer_stocks(
    sector_name: str,
    layer_name: str,
    layer_items: list,
    layer_idx: int,
    total_layers: int,
    force_refresh: bool = False,
) -> list:
    """
    Return top 3 A-share stocks for a specific supply-chain layer.

    Results are cached in product_peers under a namespaced key so the same
    (sector, layer) pair is not re-queried on every page load.

    Returns list of {ticker, name, primary_product}.
    """
    sector_clean = (sector_name or "").strip().lower()
    layer_clean  = (layer_name  or "").strip().lower()
    cache_key    = f"__layer__|{sector_clean}|{layer_clean}"

    if not force_refresh:
        cached = data_manager.get_product_peers(cache_key)
        peers = cached.get("peers") if cached else None
        # Treat invalidated (empty list) as a cache miss so Re-query works
        if peers:
            return peers

    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    position  = "upstream" if layer_idx <= (total_layers // 2) else "downstream"
    items_str = " | ".join(layer_items) if layer_items else "unspecified"

    user_msg = (
        f"Sector: {sector_name}\n"
        f"Layer {layer_idx} of {total_layers} ({position}): {layer_name}\n"
        f"Key items in this layer: {items_str}"
    )

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _LAYER_STOCK_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    try:
        resp = requests.post(
            _ENDPOINT, json=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
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

    raw_stocks = data.get("stocks", [])
    cleaned, seen = [], set()
    for s in raw_stocks:
        t  = str(s.get("ticker", "")).strip().zfill(6)
        n  = (s.get("name", "") or "").strip()
        pp = (s.get("primary_product", "") or "").strip()
        if len(t) == 6 and t.isdigit() and t not in seen:
            cleaned.append({"ticker": t, "name": n, "primary_product": pp})
            seen.add(t)

    data_manager.upsert_product_peers(cache_key, cleaned)
    return cleaned


# ── Product-Level Stock Discovery (Sector Explorer) ───────────────────────────

_PRODUCT_STOCK_PROMPT = """\
You are an elite Chinese A-share equity analyst with deep knowledge of every A-share listed company.

The user will give you:
  - A specific PRODUCT or SERVICE within a supply chain
  - The SECTOR this product is used in
  - The LAYER within that sector's supply chain this product belongs to

Your task: find A-share listed companies that currently generate SIGNIFICANT and VERIFIABLE
revenue from manufacturing or supplying this EXACT product into the named sector.

QUALIFICATION RULES — a company qualifies if ALL of the following are true:
  1. It actively manufactures or supplies the named product TODAY with paying customers
     (not "in R&D", not "announced plans", not "subsidiary exploring")
  2. Its end customers for this product are companies operating in the named sector
  Note: revenue share does NOT need to be large — a company ramping fast in an emerging
  sector is fine to include as long as it is genuinely shipping product to real customers.

DISQUALIFICATION — exclude a company if ANY of the following apply:
  - It makes a RELATED but DIFFERENT product (e.g. AI accelerator ≠ server CPU; FPGA ≠ GPU)
  - It is a buyer, integrator, or distributor of this product, not a direct manufacturer
  - It has announced plans or is in trial production but has not yet shipped to paying customers
  - It is adjacent to this market but does not directly produce the named item

SELF-CHECK — before adding each candidate, ask:
  "Has this company shipped THIS exact product to paying customers in THIS sector?"
  If the answer is uncertain or only aspirational → leave it out.

QUANTITY RULE — precision over padding:
  - Return as many as 5 or as few as 0.
  - If only 1 company truly qualifies, return just that 1.
  - If no company genuinely qualifies, return an empty stocks list.
  - NEVER pad the list with adjacent, aspirational, or loosely-related companies.

Example — "Server CPU / 服务器CPU" in "Data Center / 数据中心":
  ✓ INCLUDE: 海光信息 688041 — ships x86-compatible server CPUs to data-centre customers
  ✗ EXCLUDE: 寒武纪 688256 — AI inference accelerator, categorically different from a CPU
  ✗ EXCLUDE: 紫光国微 002049 — security chips / FPGAs, not server CPUs
  ✗ EXCLUDE: 全志科技 300458 — embedded ARM SoCs for IoT, not server-grade CPUs

CRITICAL OUTPUT RULES:
- Return ONLY raw JSON. No markdown fences. Start with { and end with }.
- Each ticker MUST be exactly 6 digits (A-share only; no ETFs, no HK/US stocks).
- primary_product must name the specific item they supply.
- "stocks" may be an empty list [] when no company genuinely qualifies.

Schema:
{
  "stocks": [
    {"ticker": "688041", "name": "海光信息", "primary_product": "x86-compatible server CPU"}
  ]
}
"""


def discover_product_stocks(
    product: str,
    layer_name: str,
    sector_name: str,
) -> list:
    """
    Return top 3-5 A-share stocks for a specific product within a sector's supply chain.

    Sector + layer context narrows results to companies supplying this product
    specifically into that sector's value chain (e.g. battery-grade vs. industrial).

    Results are NOT auto-saved to DB — the caller (admin UI) decides whether to persist.
    Cache key used only for reading: __sector_product__|{sector}|{layer}|{product}.

    Returns list of {ticker, name, primary_product}.
    """
    if not product or not product.strip():
        return []

    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    user_msg = (
        f"Sector: {sector_name}\n"
        f"Layer: {layer_name}\n"
        f"Product/Service: {product}"
    )

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _PRODUCT_STOCK_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    try:
        resp = requests.post(
            _ENDPOINT, json=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
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

    raw_stocks = data.get("stocks", [])
    cleaned, seen = [], set()
    for s in raw_stocks:
        t  = str(s.get("ticker", "")).strip().zfill(6)
        n  = (s.get("name", "") or "").strip()
        pp = (s.get("primary_product", "") or "").strip()
        if len(t) == 6 and t.isdigit() and t not in seen:
            cleaned.append({"ticker": t, "name": n, "primary_product": pp})
            seen.add(t)

    return cleaned
