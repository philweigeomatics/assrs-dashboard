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

import streamlit as st

import auth_manager
import data_manager
import peer_discovery
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


# ── Input ──────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    ticker_input = st.text_input(
        "Stock Code 股票代码",
        placeholder="e.g., 002080",
        key="ll_ticker",
        help="Enter the 6-digit code of any A-share stock that moved today.",
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

ticker = (ticker_input or "").strip()

# Promote the typed ticker to "active" only when Analyze is pressed.  After
# that, all subsequent reruns (checkbox clicks, save, generate, regen, etc.)
# rehydrate the analysis from session state.
if run and len(ticker) == 6 and ticker.isdigit():
    st.session_state["ll_active_ticker"] = ticker

active_ticker = st.session_state.get("ll_active_ticker")
if not active_ticker:
    st.info(
        "Enter a 6-digit ticker and click **Analyze**. "
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
if hc2.button("🔄 Regenerate", help="Regenerate the supply chain via DeepSeek", key="ll_regen"):
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
with st.spinner("Discovering A-share peers per edge via DeepSeek (cached after first run)…"):
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
    with st.expander("⚠️ DeepSeek errors", expanded=False):
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
    "= different competitors. Uncheck irrelevant peers, then **💾 Save curation** "
    "to persist the cleaned list (shared across any stock with this same edge)."
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
        ar1, ar2, _ = st.columns([1.5, 1.7, 4])
        if ar1.button(
            "💾 Save curation",
            key=f"ll_save_{ticker}_{product}_{sector}",
            help="Persist the cleaned list. Shared with any stock that has this same edge.",
        ):
            composite_name = peer_discovery._composite_key(product, sector)
            if data_manager.upsert_product_peers(
                composite_name, active_peers, source_method="user_curated"
            ):
                st.success(f"✅ Saved {len(active_peers)} peers for {edge_label}.")
                st.rerun()
            else:
                st.error("Save failed.")
        if ar2.button(
            "🔄 Refresh from DeepSeek",
            key=f"ll_refresh_{ticker}_{product}_{sector}",
        ):
            with st.spinner("Re-querying DeepSeek…"):
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
    f"likely the story driving the move. Phase 2 will fuzzy-match **{top_s}** to a "
    f"saved sector theme and walk the chain."
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
