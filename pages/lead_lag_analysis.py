"""
Lead-Lag Analysis 领先滞后分析

Phase 1 — Dominant Theme Detection:
  Given an input ticker that's moved today, identify which of its products is
  driving the move by checking which peer group co-moved.

  User curates the peer list (uncheck stocks that aren't relevant).
  We score each product by median peer return and flag the dominant one.

Phases 2+ (sector match, chain walk, lag math) build on this.
"""

import statistics

import streamlit as st

import auth_manager
import data_manager
import peer_discovery

auth_manager.require_login()

# ── Page Header ────────────────────────────────────────────────────────────────
st.title("📊 Lead-Lag Analysis | 领先滞后分析")
st.caption("Phase 1 · Dominant Theme Detection 主导主题识别")
st.markdown("---")

# ── Input ──────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    ticker_input = st.text_input(
        "Stock Code 股票代码",
        placeholder="e.g., 002080",
        key="ll_ticker",
        help="Enter the 6-digit code of a stock that moved today.",
    )
with c2:
    st.write("")
    st.write("")
    run = st.button("🔍 Analyze", type="primary", use_container_width=True)

ticker = (ticker_input or "").strip()
if not run:
    st.info(
        "Enter a ticker and click **Analyze**.\n\n"
        "The tool needs an existing supply-chain graph for that stock — "
        "if the analysis fails to load, generate one first via the **Watchlist** page."
    )
    st.stop()

if not (len(ticker) == 6 and ticker.isdigit()):
    st.error("❌ Please enter a 6-digit ticker.")
    st.stop()

# ── Cache wrapper for the Tushare batch call (5 min TTL) ───────────────────────
@st.cache_data(ttl=300, show_spinner=False, max_entries=20)
def _cached_pct_chg(tickers_tuple):
    return peer_discovery.fetch_latest_pct_chg(list(tickers_tuple))


# ── Load supply-chain graph ────────────────────────────────────────────────────
graph = data_manager.get_supply_chain_graph(ticker)
if not graph:
    st.error(
        f"No supply-chain graph found for **{ticker}**. "
        "Generate one on the **Watchlist** page first."
    )
    st.stop()

products = graph.get("products", []) or []
links    = graph.get("links",    []) or []
company  = graph.get("company_name", ticker)

if not products:
    st.warning("This stock's supply chain has no products listed.")
    st.stop()

st.subheader(f"🔗 {company} · {ticker}")

# ── Discover peers per product (cached in DB) ──────────────────────────────────
peers_by_product = {}
errors = []

with st.spinner("Discovering A-share peers via DeepSeek (first run only)…"):
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

# ── One Tushare call for all unique tickers ────────────────────────────────────
all_tickers = sorted(
    {p["ticker"] for plist in peers_by_product.values() for p in plist} | {ticker}
)

with st.spinner("Fetching latest pct_chg from Tushare…"):
    pct_map, trade_date = _cached_pct_chg(tuple(all_tickers))

if not pct_map:
    st.error("Could not fetch market data from Tushare. Try again later.")
    st.stop()

input_ret = pct_map.get(ticker)
mc1, mc2 = st.columns([1, 3])
mc1.metric(
    f"{ticker} on {trade_date}",
    f"{input_ret:+.2f}%" if input_ret is not None else "N/A",
)
mc2.caption(f"📅 Trade date: **{trade_date}** · {len(all_tickers)} tickers fetched in one API call.")

st.markdown("---")

# ── Per-product curated peer list + scoring ────────────────────────────────────
st.markdown("### 📦 Products & Peer Co-Movement")
st.caption(
    "Uncheck any peer you don't want included in the score. "
    "Median return of the active peers determines the dominant product."
)

scores = []  # [(product, median, n_strong, n_active, linked_sectors)]

for product, peers in peers_by_product.items():
    linked = [l.get("target") for l in links if l.get("source") == product]

    with st.expander(
        f"**{product}** · {len(peers)} peers"
        + (f"  →  {', '.join(linked)}" if linked else ""),
        expanded=True,
    ):
        if not peers:
            st.caption("No peers found for this product.")
            continue

        active_returns = []

        # Two-column layout for peer checkboxes when there are many
        n = len(peers)
        cols = st.columns(2)
        for i, p in enumerate(peers):
            ret = pct_map.get(p["ticker"])
            ret_str = f"{ret:+.2f}%" if ret is not None else "—"
            label = f"`{p['ticker']}` {p['name']}  · **{ret_str}**"
            key = f"ll_chk_{ticker}_{product}_{p['ticker']}"
            with cols[i % 2]:
                include = st.checkbox(label, value=True, key=key)
            if include and ret is not None:
                active_returns.append(ret)

        if active_returns:
            med = statistics.median(active_returns)
            n_strong = sum(1 for r in active_returns if r > 2)
            scores.append((product, med, n_strong, len(active_returns), linked))

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Median peer return", f"{med:+.2f}%")
            mc2.metric("Peers up >2%",       f"{n_strong}/{len(active_returns)}")
            mc3.metric("Active peers",       f"{len(active_returns)}/{n}")
        else:
            st.caption("No active peers with market data.")

st.markdown("---")

# ── Dominant product ───────────────────────────────────────────────────────────
if not scores:
    st.warning("No peer scoring data. Try a different ticker, or check the DeepSeek errors above.")
    st.stop()

scores.sort(key=lambda x: x[1], reverse=True)
top_product, top_med, top_strong, top_n, top_linked = scores[0]

st.subheader("🎯 Dominant Product 主导产品")
st.success(
    f"**{top_product}** drove the move on {trade_date}.\n\n"
    f"Median peer return: **{top_med:+.2f}%** · "
    f"{top_strong}/{top_n} active peers up >2%"
)

if top_linked:
    st.markdown("**Linked macro sectors (from the stock's supply chain):**")
    for s in top_linked:
        st.markdown(f"- 🏭 **{s}**")
    st.caption("Phase 2 will fuzzy-match these to the saved sector themes and walk the chain.")
else:
    st.caption("No macro sector links recorded for this product in the supply chain graph.")

# ── Full scoring table (collapsed) ─────────────────────────────────────────────
with st.expander("📊 Full ranking", expanded=False):
    import pandas as pd
    df = pd.DataFrame(
        [(p, f"{m:+.2f}%", f"{s}/{n}", ", ".join(L)) for p, m, s, n, L in scores],
        columns=["Product", "Median", "Up >2%", "Linked sectors"],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
