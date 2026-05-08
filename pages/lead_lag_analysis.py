"""
Lead-Lag Analysis 领先滞后分析  ·  Phase 1 (v3)

Identify which (product, sector) edge of an input stock's supply chain is
co-moving with the stock today.

v3 changes
----------
  • Session-state guard so unchecking peers / clicking Generate doesn't
    bounce the page back to the empty input screen.
  • Each section in the UI is now ONE supply-chain edge (product → sector),
    not a product. Same product targeting two sectors splits cleanly:
    e.g. Glass Fiber → Construction vs Glass Fiber → Data Center get
    independent peer lists, scoring, and curation.
  • Inline supply-chain generation if the stock has no graph yet.
  • Single batched Tushare call (comma-separated ts_codes).
  • Direction-agnostic scoring (smallest |median − T_return| wins).
  • "Save curation" persists the cleaned peer list back to product_peers,
    keyed by the composite "product → sector" string.
"""

import statistics

import numpy as np
import streamlit as st

import auth_manager
import data_manager
import lead_lag_stats
import peer_discovery
import sector_themes as sector_themes_mod
import supply_chain_ui

auth_manager.require_login()

# ── Page Header ────────────────────────────────────────────────────────────────
st.title("📊 Lead-Lag Analysis | 领先滞后分析")
st.caption("Phase 1 · Dominant Edge Detection 主导环节识别")
st.markdown("---")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_company_name(ticker: str) -> str:
    """Quick lookup of Chinese name from stock_basic; ticker as fallback."""
    try:
        ts_code = data_manager.get_tushare_ticker(ticker)
        df = data_manager.db.read_table(
            "stock_basic", filters={"ts_code": ts_code}, columns="name", limit=1
        )
        if not df.empty:
            return df.iloc[0]["name"]
    except Exception:
        pass
    return ticker

@st.cache_data(ttl=300, show_spinner=False, max_entries=20)
def _cached_pct_chg(tickers_tuple):
    return peer_discovery.fetch_latest_pct_chg(list(tickers_tuple))

def _co_move_tolerance(t_ret: float) -> float:
    """Tolerance band around T's return: ±1% floor, or 50% of |T| magnitude."""
    return max(1.0, abs(t_ret) * 0.5)

@st.cache_data(ttl=3600, show_spinner=False)
def _all_stock_options_ll():
    stocks = data_manager.get_all_stock_basic()
    return [""] + [f"{s['ticker']} · {s['name']}" for s in stocks]


# ── Input ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    ticker_input = st.selectbox(
        "Stock Code or Name 股票代码或名称",
        options=_all_stock_options_ll(),
        key="ll_ticker",
        format_func=lambda x: "Type to search… (code or name)" if x == "" else x,
    )
with c2:
    st.write("")
    st.write("")
    run = st.button("🔍 Analyze", type="primary", use_container_width=True)
with c3:
    st.write("")
    st.write("")
    if st.button("✖ Reset", use_container_width=True):
        st.session_state.pop("ll_active_ticker", None)
        st.rerun()

# Extract ticker from "XXXXXX · 名称" selection
_raw = (ticker_input or "").strip()
ticker = _raw.split(" · ")[0].strip() if " · " in _raw else _raw

# Promote the typed ticker to "active" only when Analyze is pressed.  After
# that, all subsequent reruns (checkbox clicks, save, generate, regen, etc.)
# rehydrate the analysis from session state.
if run and len(ticker) == 6 and ticker.isdigit():
    st.session_state["ll_active_ticker"] = ticker

active_ticker = st.session_state.get("ll_active_ticker")
if not active_ticker:
    st.info(
        "Search by code or name and click **Analyze**. "
        "If the stock has no supply chain yet, you'll be prompted to generate one inline."
    )
    st.stop()

# From here on, work off the active ticker so reruns don't lose state.
ticker = active_ticker
company = _get_company_name(ticker)

# ── Inline Supply Chain Generation ─────────────────────────────────────────────
graph = data_manager.get_supply_chain_graph(ticker)

if not graph:
    st.warning(f"No supply-chain graph yet for **{ticker} {company}**.")
    if st.button("🧬 Generate Supply Chain", type="primary", key="ll_gen"):
        with st.spinner(f"Generating supply chain for {company} ({ticker})…"):
            try:
                graph = supply_chain_ui.generate_supply_chain_graph(ticker, company)
                if data_manager.upsert_supply_chain_graph(ticker, company, graph):
                    st.success("✅ Supply chain generated. Reloading…")
                    st.rerun()
                else:
                    st.error("Generated but failed to save to DB.")
                    st.stop()
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
                st.stop()
    st.stop()

# ── Header + Regenerate ────────────────────────────────────────────────────────
products = graph.get("products", []) or []
sectors  = graph.get("macro_sectors", []) or []
links    = graph.get("links",    []) or []

# Dedupe + drop empty links
seen, unique_links = set(), []
for l in links:
    src = (l.get("source") or "").strip()
    tgt = (l.get("target") or "").strip()
    if not src or not tgt or (src, tgt) in seen:
        continue
    seen.add((src, tgt))
    unique_links.append({"source": src, "target": tgt})

hc1, hc2 = st.columns([5, 1])
hc1.subheader(f"🔗 {company} · {ticker}")
hc1.caption(
    f"{len(products)} products · {len(sectors)} sectors · "
    f"{len(unique_links)} supply-chain edges"
)
if hc2.button("🔄 Regenerate", help="Regenerate the supply chain via AI", key="ll_regen"):
    with st.spinner("Regenerating…"):
        try:
            new_graph = supply_chain_ui.generate_supply_chain_graph(ticker, company)
            if data_manager.upsert_supply_chain_graph(ticker, company, new_graph):
                st.success("✅ Regenerated.")
                st.rerun()
        except RuntimeError as exc:
            st.error(f"❌ {exc}")

with st.expander("View supply chain graph", expanded=False):
    p1, p2 = st.columns(2)
    p1.markdown("📦 **Products:** " + " · ".join(products))
    p2.markdown("🏭 **Sectors:** "  + " · ".join(sectors))
    supply_chain_ui.render_supply_chain_graph(graph, height=460)

st.markdown("---")

if not unique_links:
    st.warning(
        "This stock's supply chain has no product → sector edges. "
        "Try regenerating the graph above."
    )
    st.stop()

# ── Discover peers per edge (cached in DB) ─────────────────────────────────────
edges = []  # (product, sector, peers, error)
with st.spinner("Discovering A-share peers per edge via AI (cached after first run)…"):
    for link in unique_links:
        product, sector = link["source"], link["target"]
        try:
            peers = peer_discovery.discover_peers(product, sector)
            peers = [p for p in peers if p["ticker"] != ticker]
            edges.append((product, sector, peers, None))
        except RuntimeError as exc:
            edges.append((product, sector, [], str(exc)))

errs = [(p, s, e) for p, s, _, e in edges if e]
if errs:
    with st.expander("⚠️ AI errors", expanded=False):
        for p, s, e in errs:
            st.write(f"**{p} → {s}**: {e}")

# ── One batched Tushare call for all unique tickers ────────────────────────────
all_tickers = sorted(
    {p["ticker"] for _, _, plist, _ in edges for p in plist} | {ticker}
)
with st.spinner(f"Fetching latest pct_chg for {len(all_tickers)} tickers…"):
    pct_map, trade_date = _cached_pct_chg(tuple(all_tickers))

if not pct_map:
    st.error("Could not fetch market data from Tushare.")
    st.stop()

input_ret = pct_map.get(ticker)
if input_ret is None:
    st.error(f"No price data for **{ticker}** in the last 10 days.")
    st.stop()

# ── T's move banner (direction-agnostic) ───────────────────────────────────────
direction = "↗ rose" if input_ret > 0.05 else ("↘ fell" if input_ret < -0.05 else "≈ flat")
sign_color = "#22c55e" if input_ret > 0 else ("#ef4444" if input_ret < 0 else "#9ca3af")
tol = _co_move_tolerance(input_ret)

st.markdown(
    f"### {ticker} {company} · "
    f"<span style='color:{sign_color}'>{direction} **{input_ret:+.2f}%**</span> "
    f"on {trade_date}",
    unsafe_allow_html=True,
)
st.caption(
    f"Looking for the supply-chain edge whose peer group co-moved with this. "
    f"Co-move tolerance: ±{tol:.1f}% from T's return."
)
st.markdown("---")

# ── Per-edge curation & scoring ────────────────────────────────────────────────
st.markdown("### 🔗 Edges & Peer Co-Movement")
st.caption(
    "One section per **product → sector** edge. Same product to different sectors "
    "= different competitors. Uncheck irrelevant peers. "
    + ("Admins can **💾 Save curation** to persist the cleaned list for all users." if auth_manager.is_admin()
       else "Admins can save the curated peer list for all users.")
)

scores = []  # (product, sector, median, abs_dist, n_co, n_active)

for product, sector, peers, err in edges:
    edge_label = f"{product}  →  {sector}"
    header = f"**{edge_label}** · {len(peers)} peers"

    with st.expander(header, expanded=True):
        if err:
            st.error(err)
            continue
        if not peers:
            st.caption("No peers found for this edge.")
            continue

        active_returns, active_peers = [], []
        cols = st.columns(2)
        for i, p in enumerate(peers):
            ret = pct_map.get(p["ticker"])
            ret_str = f"{ret:+.2f}%" if ret is not None else "—"
            label = f"`{p['ticker']}` {p['name']} · **{ret_str}**"
            key = f"ll_chk_{ticker}_{product}_{sector}_{p['ticker']}"
            with cols[i % 2]:
                checked = st.checkbox(label, value=True, key=key)
            if checked:
                active_peers.append(p)
                if ret is not None:
                    active_returns.append(ret)

        # Curation actions — keys include product+sector so each edge is independent
        _is_admin = auth_manager.is_admin()
        action_cols = st.columns([1.5, 1.7, 4]) if _is_admin else st.columns([1.7, 4])
        save_col   = action_cols[0] if _is_admin else None
        refresh_col = action_cols[1] if _is_admin else action_cols[0]

        if _is_admin and save_col.button(
            "💾 Save curation",
            key=f"ll_save_{ticker}_{product}_{sector}",
            help="Admin only — persist the cleaned peer list (shared across all users for this edge).",
        ):
            composite_name = peer_discovery.composite_key(product, sector)
            if data_manager.upsert_product_peers(
                composite_name, active_peers, source_method="user_curated"
            ):
                st.success(f"✅ Saved {len(active_peers)} peers for {edge_label}.")
                st.rerun()
            else:
                st.error("Save failed.")

        if refresh_col.button(
            "🔄 Refresh from AI",
            key=f"ll_refresh_{ticker}_{product}_{sector}",
        ):
            with st.spinner("Re-querying AI…"):
                try:
                    peer_discovery.discover_peers(product, sector, force_refresh=True)
                    st.success("✅ Peers refreshed.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")

        if active_returns:
            med = statistics.median(active_returns)
            abs_dist = abs(med - input_ret)
            n_co = sum(1 for r in active_returns if abs(r - input_ret) <= tol)
            scores.append((product, sector, med, abs_dist, n_co, len(active_returns)))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Median peer", f"{med:+.2f}%")
            mc2.metric("Δ from T",   f"{abs_dist:.2f}%")
            mc3.metric(f"Co-move (±{tol:.1f}%)", f"{n_co}/{len(active_returns)}")
        else:
            st.caption("No active peers with market data.")

st.markdown("---")

# ── Dominant Edge ──────────────────────────────────────────────────────────────
if not scores:
    st.warning("No peer scoring data.")
    st.stop()

# Sort: smallest |median − T_ret| first; tie-break by more co-movers
scores.sort(key=lambda s: (s[3], -s[4]))
top = scores[0]
top_p, top_s, top_med, top_dist, top_co, top_n = top

st.subheader("🎯 Dominant Edge 主导环节")
st.success(
    f"**{top_p}  →  {top_s}**\n\n"
    f"Median peer: **{top_med:+.2f}%** · Δ from T: **{top_dist:.2f}%** · "
    f"{top_co}/{top_n} peers within ±{tol:.1f}% of T"
)
st.caption(
    f"This (product, sector) edge has the cleanest co-movement with **{ticker}** today — "
    f"likely the story driving the move."
)

with st.expander("📊 Full ranking", expanded=False):
    import pandas as pd
    df_rank = pd.DataFrame(
        [(f"{p}  →  {s}", f"{m:+.2f}%", f"{d:.2f}%", f"{co}/{n}")
         for p, s, m, d, co, n in scores],
        columns=["Edge (Product → Sector)", "Median", "Δ from T",
                 f"Co-move (±{tol:.1f}%)"],
    )
    st.dataframe(df_rank, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Phase 2 · Sector Supply Chain Walk-Down ────────────────────────────────────
st.subheader("🔍 Phase 2 · Sector Supply Chain 产业链走查")
st.caption(
    "Select the sector whose supply chain you want to walk. "
    "Defaults to the top-ranked edge identified above."
)

# Dropdown — all unique sectors from the stock's supply chain
all_sectors = sorted({link["target"] for link in unique_links})
default_idx = all_sectors.index(top_s) if top_s in all_sectors else 0
chosen_sector = st.selectbox(
    "Sector to explore 选择产业链",
    all_sectors,
    index=default_idx,
    key="ll_phase2_sector",
)

st.markdown("---")

# Cache the match result in session state so node-click reruns don't re-call DeepSeek
all_themes = data_manager.get_all_sector_themes()
match_cache_key = f"ll_p2_match_{ticker}_{chosen_sector}_{len(all_themes)}"
if match_cache_key not in st.session_state:
    with st.spinner(f"Matching '{chosen_sector}' to stored sector themes…"):
        st.session_state[match_cache_key] = sector_themes_mod.match_sector_theme(
            chosen_sector, all_themes
        )
matched = st.session_state[match_cache_key]

if matched:
    st.success(f"✅ Matched to saved theme: **{matched['formal_name']}**")

    # Cache the full theme (layers) too
    theme_cache_key = f"ll_p2_theme_{matched['id']}"
    if theme_cache_key not in st.session_state:
        st.session_state[theme_cache_key] = data_manager.get_sector_theme_by_id(matched["id"])
    theme_full = st.session_state[theme_cache_key]

    if theme_full:
        sector_themes_mod.render_sector_layers(
            theme_full,
            key_prefix=f"ll_p2_{ticker}_{matched['id']}",
        )
    else:
        st.error("Could not load theme details from the database.")

else:
    st.warning(f"No stored theme matches **'{chosen_sector}'** yet.")

    if auth_manager.is_admin():
        st.caption(
            "As an admin you can generate and save a new theme for this sector. "
            "It will then be available in the Sector Explorer as well."
        )
        if st.button(
            f"🧬 Generate & Save Theme for '{chosen_sector}'",
            type="primary",
            key="ll_p2_gen_theme",
        ):
            with st.spinner(f"Generating supply chain for '{chosen_sector}'…"):
                try:
                    data = sector_themes_mod.generate_sector_theme(chosen_sector)
                    user_info = auth_manager.get_current_user()
                    ok = data_manager.add_sector_theme(
                        raw_input=chosen_sector,
                        formal_name=data["name"],
                        layers_data=data,
                        created_by=user_info["username"],
                    )
                    if ok:
                        # Bust the match cache so the next rerun picks up the new theme
                        st.session_state.pop(match_cache_key, None)
                        st.success(f"✅ Saved: **{data['name']}**. Reloading…")
                        st.rerun()
                    else:
                        st.error("Generated but failed to save to DB.")
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")
    else:
        st.info(
            "No supply chain theme has been saved for this sector yet. "
            "Ask an admin to generate one from the Sector Explorer page."
        )

# ── Gate: Phase 3 + 4 require a valid matched theme ───────────────────────────
if not matched:
    st.stop()
theme_full = st.session_state.get(f"ll_p2_theme_{matched['id']}")
if not theme_full:
    st.stop()

p2_layers = sorted(theme_full.get("layers") or [], key=lambda l: l.get("layer_index", 0))
if not p2_layers:
    st.stop()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 · Key Stocks per Layer
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏭 Phase 3 · Key Stocks per Layer 各层核心标的")
st.caption(
    "AI identifies the top 3 A-share companies per supply-chain layer, "
    "validated against the stock_basic database. 🔗 = supply chain graph already saved."
)

p3_key = f"ll_p3_stocks_{ticker}_{matched['id']}"

c_run, c_refresh = st.columns([2, 1])
if c_run.button("🔍 Identify Key Stocks per Layer", type="primary",
                key="ll_p3_run", use_container_width=True,
                disabled=p3_key in st.session_state):
    st.session_state[p3_key] = {}   # empty dict = run discovery below
    st.rerun()

if c_refresh.button("🔄 Re-query AI", key="ll_p3_refresh",
                    use_container_width=True):
    # Clear the page-level cache and also bust the product_peers DB cache for
    # each layer by writing an empty entry; discovery block will re-call DeepSeek.
    st.session_state.pop(p3_key, None)
    for l in p2_layers:
        layer_key = f"__layer__|{chosen_sector.strip().lower()}|{l.get('layer_name','').strip().lower()}"
        data_manager.upsert_product_peers(layer_key, [], source_method="invalidated")
    st.rerun()

# ── Discovery + display ────────────────────────────────────────────────────────
if p3_key not in st.session_state:
    st.info("Click **Identify Key Stocks per Layer** to run AI discovery.")
    st.stop()

# Fetch stocks for every layer (cached in product_peers after first run)
if not st.session_state[p3_key]:
    graphs_in_db = data_manager.get_all_supply_chain_tickers()
    layer_results = {}

    prog = st.progress(0, text="Querying AI for each layer…")
    for i, layer in enumerate(p2_layers):
        prog.progress((i + 1) / len(p2_layers),
                      text=f"Layer {i+1}/{len(p2_layers)}: {layer.get('layer_name','')}")
        try:
            raw_stocks = peer_discovery.discover_layer_stocks(
                chosen_sector,
                layer.get("layer_name", ""),
                layer.get("items", []),
                layer.get("layer_index", i + 1),
                len(p2_layers),
            )
        except RuntimeError as exc:
            st.error(f"❌ {exc}")
            raw_stocks = []

        # Validate against stock_basic
        all_t = [s["ticker"] for s in raw_stocks]
        validation = data_manager.validate_tickers_against_stock_basic(all_t)

        enriched = []
        for s in raw_stocks:
            t_val = validation.get(s["ticker"], {})
            enriched.append({
                "ticker":        s["ticker"],
                "name":          t_val.get("official_name") or s["name"],
                "primary_product": s.get("primary_product", ""),
                "valid":         t_val.get("valid", False),
                "has_graph":     s["ticker"] in graphs_in_db,
                "layer_name":    layer.get("layer_name", ""),
                "layer_idx":     layer.get("layer_index", i + 1),
            })

        layer_results[layer.get("layer_name", f"Layer {i+1}")] = enriched

    prog.empty()
    st.session_state[p3_key] = layer_results

layer_results = st.session_state[p3_key]

# ── Render one expander per layer ──────────────────────────────────────────────
for layer_name, stocks in layer_results.items():
    with st.expander(f"**{layer_name}** · {len(stocks)} stocks", expanded=True):
        if not stocks:
            st.caption("No stocks found.")
            continue
        cols = st.columns(len(stocks)) if len(stocks) > 0 else [st]
        for col, s in zip(cols, stocks):
            with col:
                badges = ""
                if s["has_graph"]:
                    badges += " 🔗"
                valid_badge = "✅" if s["valid"] else "⚠️"
                st.markdown(
                    f"**{s['ticker']}** {valid_badge}{badges}  \n"
                    f"{s['name']}  \n"
                    f"*{s['primary_product']}*"
                )
                chk_key = f"ll_p3_chk_{ticker}_{s['layer_idx']}_{s['ticker']}"
                st.checkbox("Include in analysis", value=s["valid"],
                            key=chk_key)

st.markdown("---")

# ── Collect checked stocks for Phase 4 ────────────────────────────────────────
p4_peers = []
for layer_name, stocks in layer_results.items():
    for s in stocks:
        chk_key = f"ll_p3_chk_{ticker}_{s['layer_idx']}_{s['ticker']}"
        if st.session_state.get(chk_key, s["valid"]) and s["ticker"] != ticker:
            p4_peers.append({
                "ticker":     s["ticker"],
                "name":       s["name"],
                "layer_name": s["layer_name"],
                "layer_idx":  s["layer_idx"],
            })

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 · Lead-Lag Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Phase 4 · Lead-Lag Statistical Analysis 领先滞后统计分析")

with st.expander("ℹ️ How to read the results", expanded=False):
    st.markdown("""
**Three statistical layers per stock pair (T stock vs each identified stock S):**

| Layer | Method | Interpretation |
|---|---|---|
| **Granger causality** | F-test: do T's past returns predict S's future returns (and vice versa)? | p < 0.05 → significant lead-lag relationship |
| **Cross-correlation** | Pearson correlation at each lag offset (-N to +N days) | Positive lag = T leads S; negative lag = S leads T |
| **Cointegration + half-life** | Engle-Granger test on price levels; OU half-life of the spread | Short half-life (< 10d) → exploitable mean-reversion |

**Beta adjustment:** Beta measures how much T moves per 1% move in S (estimated via OLS).
A beta of 1.2 means if S rises 8%, T's "expected" move is 9.6% — any difference beyond that is the signal.

**Signal strength:** 🔥 Strong = Granger p < 0.01 + cointegrated + half-life < 15d ·
⚡ Moderate = Granger p < 0.05 + cointegrated or |corr| > 0.4 ·
〰 Weak = Granger p < 0.10 · ✗ None = no statistical evidence.

**Multiple-testing guard:** All Granger signals also require |peak cross-correlation| > 0.15.
With many peer stocks tested simultaneously, a bare p < 0.05 threshold produces ~1 spurious
"significant" result per 20 pairs by chance. The correlation floor ensures the p-value reflects
real co-movement rather than sampling noise.
""")

import pandas as pd

lb_col, lag_col, _ = st.columns([2, 2, 4])
lookback = lb_col.radio("Lookback 回溯天数", [90, 120, 180], index=1,
                        horizontal=True, key="ll_p4_lookback")
max_lag  = lag_col.radio("Max lag 最大滞后", [3, 5, 10], index=1,
                         horizontal=True, key="ll_p4_maxlag")

p4_results_key = f"ll_p4_results_{ticker}_{matched['id']}_{lookback}_{max_lag}"

if not p4_peers:
    st.warning("No stocks selected in Phase 3. Check at least one stock above.")
    st.stop()

if st.button("📊 Run Lead-Lag Analysis", type="primary", key="ll_p4_run"):
    st.session_state[p4_results_key] = None   # None = compute on next rerun
    st.rerun()

if p4_results_key not in st.session_state:
    st.info("Configure lookback / max-lag above, then click **Run Lead-Lag Analysis**.")
    st.stop()

# ── Fetch + compute (runs once; stored in session state) ──────────────────────
if st.session_state[p4_results_key] is None:
    all_tickers_needed = list({ticker} | {p["ticker"] for p in p4_peers})
    n = len(all_tickers_needed)

    prog4 = st.progress(0, text=f"Fetching {lookback}d qfq data for {n} stocks…")
    # We can't easily stream progress inside fetch_qfq_returns, so show indeterminate
    prog4.progress(0.1, text=f"Fetching qfq data for {n} tickers (this may take ~{n*2}s)…")

    try:
        returns_df, prices_df = lead_lag_stats.fetch_qfq_returns(
            all_tickers_needed, lookback_days=lookback
        )
    except Exception as exc:
        st.error(f"❌ Data fetch failed: {exc}")
        st.stop()

    prog4.progress(0.6, text="Running Granger causality, cross-correlation, cointegration…")

    if returns_df.empty or ticker not in returns_df.columns:
        st.error(f"Could not fetch price data for T stock {ticker}. Check Tushare quota.")
        st.stop()

    n_obs = returns_df[ticker].dropna().__len__()
    if n_obs < 60:
        st.warning(
            f"Only {n_obs} trading days of data available (60 minimum recommended). "
            "Results may be unreliable."
        )

    try:
        results_df = lead_lag_stats.compute_lead_lag(
            ticker, p4_peers, returns_df, prices_df, max_lag=max_lag
        )
    except RuntimeError as exc:
        st.error(f"❌ {exc}")
        st.stop()

    prog4.empty()
    st.session_state[p4_results_key] = {
        "results": results_df,
        "returns": returns_df,
        "prices":  prices_df,
    }

stored = st.session_state.get(p4_results_key)
if not stored:
    st.stop()

results_df = stored["results"]
returns_df = stored["returns"]

if results_df.empty:
    st.warning("No results — none of the selected stocks had sufficient price data.")
    st.stop()

# ── Summary table ──────────────────────────────────────────────────────────────
st.markdown("#### 📋 Results Summary")

def _fmt_p(v):
    if pd.isna(v):
        return "—"
    return f"{v:.3f}" if v >= 0.001 else "< 0.001"

def _fmt_f(v, dec=2):
    return f"{v:.{dec}f}" if not pd.isna(v) else "—"

display_rows = []
for _, row in results_df.iterrows():
    hl = _fmt_f(row["half_life"], 1) + "d" if not pd.isna(row["half_life"]) else "—"
    display_rows.append({
        "Signal":          row["signal"],
        "Ticker":          row["ticker"],
        "Name":            row["name"],
        "Layer":           row["layer"],
        "Beta (T/S)":      _fmt_f(row["beta"], 2),
        "Relationship":    row["relationship"],
        "p T→S":           _fmt_p(row["p_T_leads_S"]),
        "Lag T→S":         f"{row['lag_T_leads_S']}d" if row["p_T_leads_S"] < 0.1 else "—",
        "p S→T":           _fmt_p(row["p_S_leads_T"]),
        "Lag S→T":         f"{row['lag_S_leads_T']}d" if row["p_S_leads_T"] < 0.1 else "—",
        "Cointegrated":    "✅" if row["cointegrated"] else "✗",
        "Half-life":       hl,
        "n obs":           int(row["n_obs"]),
    })

display_df = pd.DataFrame(display_rows)
# Sort: Strong first, then Moderate, then by p-value
signal_order = {"🔥 Strong": 0, "⚡ Moderate": 1, "〰 Weak": 2, "✗ None": 3}
display_df["_sort"] = display_df["Signal"].map(signal_order)
display_df = display_df.sort_values("_sort").drop(columns=["_sort"])

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── Cross-correlation heatmap ──────────────────────────────────────────────────
st.markdown("#### 🌡️ Cross-Correlation Heatmap")
st.caption(
    "Columns = lag offset · **Positive lag (T+Nd→S)**: T leads S by N days · "
    "**Negative lag (S+Nd→T)**: S leads T by N days · "
    "Green = positive correlation, Red = negative."
)

heatmap_df = lead_lag_stats.build_heatmap_df(results_df, max_lag=max_lag)
if not heatmap_df.empty:
    styled = (
        heatmap_df
        .style
        .background_gradient(cmap="RdYlGn", vmin=-0.6, vmax=0.6, axis=None)
        .format("{:.2f}", na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)

# ── Strong signal callouts ─────────────────────────────────────────────────────
strong   = results_df[results_df["signal"].str.startswith("🔥")]
moderate = results_df[results_df["signal"].str.startswith("⚡")]

if not strong.empty or not moderate.empty:
    st.markdown("#### 🎯 Highlighted Signals")
    for _, row in pd.concat([strong, moderate]).iterrows():
        hl_str    = f" · half-life **{row['half_life']:.1f}d**" if not pd.isna(row["half_life"]) else ""
        coint_str = " · cointegrated ✅" if row["cointegrated"] else ""
        st.info(
            f"{row['signal']} &nbsp; **{row['ticker']} {row['name']}** "
            f"({row['layer']})  \n"
            f"**{row['relationship']}** · beta {_fmt_f(row['beta'])} "
            f"· best corr {_fmt_f(row['peak_corr'])} at lag {row['peak_lag']}d"
            f"{coint_str}{hl_str}"
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 · Relationship Charts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### 📈 Relationship Charts")
st.caption("Select a stock to visualise its relationship with T across three chart types.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

if not _PLOTLY_OK:
    st.warning("Install plotly (`pip install plotly`) to enable relationship charts.")
    st.stop()

# Stock selector — prefer strongest signals at top
_signal_rank = {"🔥 Strong": 0, "⚡ Moderate": 1, "〰 Weak": 2, "✗ None": 3}
chart_options = (
    results_df
    .assign(_r=results_df["signal"].map(_signal_rank))
    .sort_values("_r")
    [["ticker", "name", "signal"]]
    .apply(lambda r: f"{r['signal']} {r['ticker']} {r['name']}", axis=1)
    .tolist()
)
chart_choice = st.selectbox(
    "Stock to chart 选择标的",
    chart_options,
    key="ll_chart_choice",
)
# Ticker is always 6 digits; extract reliably rather than relying on split position
import re as _re
_m = _re.search(r'\b(\d{6})\b', chart_choice)
chosen_s_ticker = _m.group(1) if _m else None
if not chosen_s_ticker or chosen_s_ticker not in results_df["ticker"].values:
    st.warning("Could not identify the selected stock. Please re-run the analysis.")
    st.stop()
chosen_row = results_df[results_df["ticker"] == chosen_s_ticker].iloc[0]

# ── Shared data prep ──────────────────────────────────────────────────────────
import math as _math

# Use the Granger lag (which direction is significant) rather than the raw
# cross-correlation peak, which can be 0 even when Granger found a 4-day lead.
_p_tls = chosen_row["p_T_leads_S"]
_p_slt = chosen_row["p_S_leads_T"]
if not _math.isnan(_p_tls) and _p_tls < 0.05:
    display_lag = int(chosen_row["lag_T_leads_S"])   # +N: T leads S by N days
elif not _math.isnan(_p_slt) and _p_slt < 0.05:
    display_lag = -int(chosen_row["lag_S_leads_T"])  # -N: S leads T by N days
else:
    display_lag = int(chosen_row["peak_lag"])         # fallback to cross-corr

beta_val  = chosen_row["beta"]
xcorrs    = chosen_row["_xcorrs"]
lags      = sorted(xcorrs.keys())
corr_vals = [xcorrs[k] for k in lags]

# Prices/returns for this pair
p_t_full = stored["prices"].get(ticker)
p_s_full = stored["prices"].get(chosen_s_ticker)
r_t_full = stored["returns"].get(ticker)
r_s_full = stored["returns"].get(chosen_s_ticker)

# ── Debug expander ────────────────────────────────────────────────────────────
with st.expander("🔍 Raw data (debug)", expanded=False):
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown(f"**T · {ticker} prices** ({len(p_t_full) if p_t_full is not None else 0} rows)")
        if p_t_full is not None:
            st.dataframe(
                p_t_full.rename("close").reset_index().rename(columns={"index": "date"}),
                use_container_width=True, height=280,
            )
        else:
            st.warning("No price data for T.")
        st.markdown(f"**T · {ticker} daily returns**")
        if r_t_full is not None:
            st.dataframe(
                r_t_full.rename("pct_return").reset_index().rename(columns={"index": "date"}),
                use_container_width=True, height=280,
            )
    with dc2:
        st.markdown(f"**S · {chosen_s_ticker} prices** ({len(p_s_full) if p_s_full is not None else 0} rows)")
        if p_s_full is not None:
            st.dataframe(
                p_s_full.rename("close").reset_index().rename(columns={"index": "date"}),
                use_container_width=True, height=280,
            )
        else:
            st.warning("No price data for S.")
        st.markdown(f"**S · {chosen_s_ticker} daily returns**")
        if r_s_full is not None:
            st.dataframe(
                r_s_full.rename("pct_return").reset_index().rename(columns={"index": "date"}),
                use_container_width=True, height=280,
            )

import plotly.graph_objects as go

direction_lbl = (
    f"T leads S by {display_lag}d" if display_lag > 0
    else f"S leads T by {-display_lag}d" if display_lag < 0
    else "no significant lag"
)

# Shared light-theme layout defaults (no xaxis/yaxis — each chart sets its own)
_LAYOUT = dict(
    height=400,
    margin=dict(l=50, r=20, t=45, b=60),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#ffffff",
    font=dict(color="#1e293b", size=12),
    legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
)
_AXIS = dict(gridcolor="#e2e8f0", linecolor="#cbd5e1")

tab_price, tab_corr, tab_scatter = st.tabs(
    ["📉 Normalized Price", "📊 Cross-Correlation", "🔵 Return Scatter"]
)

# ── Tab 1: Normalized price ───────────────────────────────────────────────────
with tab_price:
    apply_shift = st.checkbox(
        f"Apply Granger lag shift ({display_lag:+d}d · {direction_lbl}) — shift S to align with T",
        value=(display_lag != 0),
        key="ll_chart_shift",
    )

    if p_t_full is not None and p_s_full is not None:
        pt, ps = p_t_full.align(p_s_full, join="inner")
        pt, ps = pt.dropna().align(ps.dropna(), join="inner")

        pt_norm = pt / pt.iloc[0] * 100
        ps_norm = ps / ps.iloc[0] * 100

        if apply_shift and display_lag != 0:
            ps_norm = ps_norm.shift(-display_lag)

        s_label = f"S · {chosen_s_ticker} {chosen_row['name']}"
        if apply_shift and display_lag != 0:
            s_label += f" (shifted {display_lag:+d}d)"

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=pt_norm.index, y=pt_norm.values,
            name=f"T · {ticker} {company}",
            line=dict(color="#2563eb", width=2),
        ))
        ps_plot = ps_norm.dropna()
        fig1.add_trace(go.Scatter(
            x=ps_plot.index, y=ps_plot.values,
            name=s_label,
            line=dict(color="#ea580c", width=2,
                      dash="dot" if (apply_shift and display_lag != 0) else "solid"),
        ))
        fig1.update_layout(
            **_LAYOUT,
            title=dict(text=f"Normalized Price (base 100) · Granger lag: {display_lag:+d}d",
                       font=dict(size=13)),
            xaxis_title="Date",
            yaxis_title="Indexed Price (100 = start)",
            xaxis=_AXIS,
            yaxis=_AXIS,
        )
        st.plotly_chart(fig1, use_container_width=True)
        if apply_shift and display_lag != 0:
            st.caption(
                f"S shifted {abs(display_lag)}d ({direction_lbl}). "
                "Closer overlap after the shift = stronger evidence the relationship is real."
            )
    else:
        st.warning("Price data unavailable for one or both stocks.")

# ── Tab 2: Cross-correlation bar chart ───────────────────────────────────────
with tab_corr:
    col_labels = []
    for k in lags:
        if k < 0:
            col_labels.append(f"S+{-k}d→T")
        elif k == 0:
            col_labels.append("±0d")
        else:
            col_labels.append(f"T+{k}d→S")

    bar_colors = ["#16a34a" if v > 0 else "#dc2626" for v in corr_vals]
    # Highlight the Granger lag bar in amber
    if display_lag in lags:
        bar_colors[lags.index(display_lag)] = "#d97706"

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=col_labels, y=corr_vals,
        marker_color=bar_colors,
        text=[f"{v:.2f}" if not _math.isnan(v) else "" for v in corr_vals],
        textposition="outside",
        textfont=dict(size=11),
    ))
    for lvl, lbl in [(0.4, "0.4"), (-0.4, "-0.4"), (0.2, "0.2"), (-0.2, "-0.2")]:
        fig2.add_hline(y=lvl, line_dash="dot", line_color="#94a3b8", line_width=1,
                       annotation_text=lbl, annotation_position="right",
                       annotation_font=dict(color="#64748b", size=10))
    fig2.add_hline(y=0, line_color="#64748b", line_width=1)
    fig2.update_layout(
        **_LAYOUT,
        title=dict(text=f"Cross-Correlation at Each Lag · Granger lag {display_lag:+d}d (🟠)",
                   font=dict(size=13)),
        xaxis_title="Lag  (S+Nd→T = S leads T  ·  T+Nd→S = T leads S)",
        xaxis=_AXIS,
        yaxis=dict(range=[-1, 1], gridcolor="#e2e8f0", linecolor="#cbd5e1"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "🟠 Amber = Granger lag · 🟢 Green = positive correlation · "
        "🔴 Red = negative correlation · Dotted lines at ±0.2 and ±0.4."
    )

# ── Tab 3: Return scatter with OLS regression ────────────────────────────────
with tab_scatter:
    if r_t_full is not None and r_s_full is not None:
        s_shifted = r_s_full.shift(-display_lag)
        combined  = pd.concat([r_t_full, s_shifted], axis=1).dropna()
        combined.columns = ["T_ret", "S_ret"]
        x_vals = combined["S_ret"].values * 100
        y_vals = combined["T_ret"].values * 100

        x_line = y_line = None
        if len(combined) >= 10 and not _math.isnan(beta_val):
            x_line    = np.array([x_vals.min(), x_vals.max()])
            intercept = y_vals.mean() - beta_val * x_vals.mean()
            y_line    = beta_val * x_line + intercept

        lag_note = f" (S shifted {display_lag:+d}d)" if display_lag != 0 else ""
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(color="#2563eb", size=5, opacity=0.55),
            name="Daily returns",
            hovertemplate="S: %{x:.2f}%<br>T: %{y:.2f}%<extra></extra>",
        ))
        if x_line is not None:
            fig3.add_trace(go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color="#d97706", width=2, dash="dash"),
                name=f"OLS fit (β = {beta_val:.2f})",
            ))
        fig3.add_hline(y=0, line_color="#94a3b8", line_width=1)
        fig3.add_vline(x=0, line_color="#94a3b8", line_width=1)
        fig3.update_layout(
            **_LAYOUT,
            title=dict(text=f"T vs S returns{lag_note} · β = {_fmt_f(beta_val)}",
                       font=dict(size=13)),
            xaxis_title=f"S · {chosen_s_ticker} {chosen_row['name']} return (%){lag_note}",
            yaxis_title=f"T · {ticker} {company} return (%)",
            xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
            yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zeroline=False),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            f"Each dot = one trading day · S shifted {display_lag:+d}d · "
            f"Amber line = OLS fit · β = {_fmt_f(beta_val)}: "
            f"for every 1% S moves, T expected to move {_fmt_f(beta_val)}%."
        )
    else:
        st.warning("Return data unavailable for one or both stocks.")
