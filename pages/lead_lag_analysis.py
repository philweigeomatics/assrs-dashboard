"""
Lead-Lag Analysis 领先滞后分析  ·  Phase 1 (v2)

Identify which of an input stock's products is co-moving with the stock today.
Direction-agnostic — works for both up moves and down moves; what matters is
whether peers moved with T, not whether they all rose.

v2 changes
----------
  • Inline supply-chain generation if the stock has no graph yet.
  • Single batched Tushare call (comma-separated ts_codes).
  • Co-movement metric: smallest |median_peer − T_return| wins.
  • "Save curation" persists the user's checked-peer list back to product_peers
    (so the next analysis only sees the cleaned list).
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
st.caption("Phase 1 · Dominant Theme Detection 主导主题识别")
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
c1, c2 = st.columns([3, 1])
with c1:
    ticker_input = st.text_input(
        "Stock Code 股票代码",
        placeholder="e.g., 002080",
        key="ll_ticker",
        help="Enter the 6-digit code of any A-share stock that moved today (up or down).",
    )
with c2:
    st.write("")
    st.write("")
    run = st.button("🔍 Analyze", type="primary", use_container_width=True)

ticker = (ticker_input or "").strip()
if not run:
    st.info("Enter a ticker and click **Analyze**. If the stock has no supply chain yet, you'll be prompted to generate one inline.")
    st.stop()

if not (len(ticker) == 6 and ticker.isdigit()):
    st.error("❌ Please enter a 6-digit ticker.")
    st.stop()

company = _get_company_name(ticker)

# ── Inline Supply Chain Generation (no need to leave the page) ─────────────────
graph = data_manager.get_supply_chain_graph(ticker)

if not graph:
    st.warning(f"No supply-chain graph yet for **{ticker} {company}**.")
    if st.button("🧬 Generate Supply Chain", type="primary"):
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

# ── Show Supply Chain (collapsible) ────────────────────────────────────────────
products = graph.get("products", []) or []
sectors  = graph.get("macro_sectors", []) or []
links    = graph.get("links",    []) or []

hc1, hc2 = st.columns([5, 1])
hc1.subheader(f"🔗 {company} · {ticker}")
hc1.caption(f"{len(products)} products · {len(sectors)} sectors · {len(links)} links")
if hc2.button("🔄 Regenerate", help="Regenerate the supply chain via DeepSeek"):
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

if not products:
    st.warning("This stock's supply chain has no products listed.")
    st.stop()

# ── Discover peers (cached in DB on first call) ────────────────────────────────
peers_by_product = {}
errors = []
with st.spinner("Discovering A-share peers via DeepSeek (cached after first run)…"):
    for product in products:
        try:
            peers = peer_discovery.discover_peers(product)
            peers_by_product[product] = [p for p in peers if p["ticker"] != ticker]
        except RuntimeError as exc:
            errors.append(f"{product}: {exc}")
            peers_by_product[product] = []

if errors:
    with st.expander("⚠️ DeepSeek errors", expanded=False):
        for e in errors:
            st.write(e)

# ── One batched Tushare call for all unique tickers ────────────────────────────
all_tickers = sorted(
    {p["ticker"] for plist in peers_by_product.values() for p in plist} | {ticker}
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

# ── T's move (direction-agnostic banner) ───────────────────────────────────────
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
    f"Looking for products whose peer group co-moved with this. "
    f"Co-move tolerance: ±{tol:.1f}% from T's return."
)
st.markdown("---")

# ── Per-product scoring & curation ─────────────────────────────────────────────
st.markdown("### 📦 Products & Peer Co-Movement")
st.caption("Uncheck irrelevant peers. Click **💾 Save curation** to persist the cleaned list.")

scores = []  # (product, median, abs_dist, n_co, n_active, linked)

for product, peers in peers_by_product.items():
    linked = [l.get("target") for l in links if l.get("source") == product]
    header = f"**{product}** · {len(peers)} peers"
    if linked:
        header += "  →  " + ", ".join(linked)

    with st.expander(header, expanded=True):
        if not peers:
            st.caption("No peers found for this product.")
            continue

        active_returns, active_peers = [], []
        cols = st.columns(2)
        for i, p in enumerate(peers):
            ret = pct_map.get(p["ticker"])
            ret_str = f"{ret:+.2f}%" if ret is not None else "—"
            label = f"`{p['ticker']}` {p['name']} · **{ret_str}**"
            key = f"ll_chk_{ticker}_{product}_{p['ticker']}"
            with cols[i % 2]:
                checked = st.checkbox(label, value=True, key=key)
            if checked:
                active_peers.append(p)
                if ret is not None:
                    active_returns.append(ret)

        # Curation actions
        ar1, ar2, _ = st.columns([1.5, 1.7, 4])
        if ar1.button("💾 Save curation", key=f"ll_save_{ticker}_{product}"):
            if data_manager.upsert_product_peers(
                product, active_peers, source_method="user_curated"
            ):
                st.success(f"✅ Saved {len(active_peers)} curated peers.")
                st.rerun()
            else:
                st.error("Save failed.")
        if ar2.button("🔄 Refresh from DeepSeek", key=f"ll_refresh_{ticker}_{product}"):
            with st.spinner("Re-querying DeepSeek…"):
                try:
                    peer_discovery.discover_peers(product, force_refresh=True)
                    st.success("✅ Peers refreshed.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"❌ {exc}")

        if active_returns:
            med = statistics.median(active_returns)
            abs_dist = abs(med - input_ret)
            n_co = sum(1 for r in active_returns if abs(r - input_ret) <= tol)
            scores.append((product, med, abs_dist, n_co, len(active_returns), linked))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Median peer",         f"{med:+.2f}%")
            mc2.metric("Δ from T",            f"{abs_dist:.2f}%")
            mc3.metric(f"Co-move (±{tol:.1f}%)", f"{n_co}/{len(active_returns)}")
        else:
            st.caption("No active peers with market data.")

st.markdown("---")

# ── Dominant Product (closest co-movement, direction-agnostic) ─────────────────
if not scores:
    st.warning("No peer scoring data.")
    st.stop()

# Sort: smallest |median - T_ret| first; tie-break by more co-movers
scores.sort(key=lambda s: (s[2], -s[3]))
top = scores[0]
top_product, top_med, top_dist, top_co, top_n, top_linked = top

st.subheader("🎯 Dominant Product 主导产品")
st.success(
    f"**{top_product}**\n\n"
    f"Median peer: **{top_med:+.2f}%** · Δ from T: **{top_dist:.2f}%** · "
    f"{top_co}/{top_n} peers within ±{tol:.1f}% of T"
)

if top_linked:
    st.markdown("**Linked macro sectors (from supply chain):**")
    for s in top_linked:
        st.markdown(f"- 🏭 **{s}**")
    st.caption("Phase 2 will fuzzy-match these to saved sector themes and walk the chain.")

with st.expander("📊 Full ranking", expanded=False):
    import pandas as pd
    df_rank = pd.DataFrame(
        [(p, f"{m:+.2f}%", f"{d:.2f}%", f"{co}/{n}", ", ".join(L))
         for p, m, d, co, n, L in scores],
        columns=["Product", "Median", "Δ from T", f"Co-move (±{tol:.1f}%)", "Linked sectors"],
    )
    st.dataframe(df_rank, use_container_width=True, hide_index=True)
