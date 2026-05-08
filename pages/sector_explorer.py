"""
Macro Sector Explorer 行业产业链浏览器
Display chronological supply-chain layers for industry themes.

  - Admins: an st.expander control panel for adding / deleting themes.
  - Viewers: pick a saved theme and click any node to query related stocks.
"""

import hashlib as _hashlib

import streamlit as st

import auth_manager
import data_manager
import peer_discovery
import sector_themes

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

# ── Auth ───────────────────────────────────────────────────────────────────────
auth_manager.require_login()
user      = auth_manager.get_current_user()
is_admin  = auth_manager.is_admin()

# ── Page Header ────────────────────────────────────────────────────────────────
st.title("🌐 Macro Sector Explorer | 行业产业链浏览器")
st.caption("Click on any node to find related stocks · 点击节点查找相关股票")
st.markdown("---")

# ── Cached lookup of all themes (cheap; one DB hit per ttl window) ─────────────
@st.cache_data(ttl=60, max_entries=1, show_spinner=False)
def _all_themes():
    return data_manager.get_all_sector_themes()

def _bust_cache():
    _all_themes.clear()


# ── Admin Panel ────────────────────────────────────────────────────────────────
if is_admin:
    with st.expander("🛠️ Admin Controls 管理员控制台", expanded=False):
        tab_add, tab_del = st.tabs(["➕ Add Theme 添加", "🗑️ Delete Theme 删除"])

        # ── Add ────────────────────────────────────────────────────────────────
        with tab_add:
            raw = st.text_input(
                "Industry theme 行业主题",
                placeholder="e.g. 'ev', 'data centre', 'liquid cooling'",
                key="sector_admin_raw",
                help="Anything goes — DeepSeek will formalise it.",
            )
            if st.button("🧬 Generate & Save", type="primary"):
                if not raw.strip():
                    st.warning("⚠️ Please type a theme first.")
                else:
                    existing = data_manager.get_sector_theme_by_raw_input(raw)
                    if existing:
                        st.warning(
                            f"This theme already exists: **{existing['formal_name']}** "
                            f"(raw: `{existing['raw_input']}`). "
                            f"Delete it first if you want to regenerate."
                        )
                    else:
                        with st.spinner(f"Generating supply chain for '{raw.strip()}'…"):
                            try:
                                data = sector_themes.generate_sector_theme(raw)
                                ok = data_manager.add_sector_theme(
                                    raw_input    = raw,
                                    formal_name  = data["name"],
                                    layers_data  = data,
                                    created_by   = user["username"],
                                )
                                if ok:
                                    _bust_cache()
                                    st.success(f"✅ Saved: **{data['name']}**")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save to database.")
                            except RuntimeError as exc:
                                st.error(f"❌ {exc}")

        # ── Delete ─────────────────────────────────────────────────────────────
        with tab_del:
            themes = _all_themes()
            if not themes:
                st.info("No themes to delete.")
            else:
                opts = {
                    f"{t['formal_name']}  (raw: {t['raw_input']})": t['id']
                    for t in themes
                }
                target_label = st.selectbox(
                    "Select theme to delete",
                    list(opts.keys()),
                    key="sector_admin_del_choice",
                )
                if st.button("🗑️ Delete", type="secondary"):
                    if data_manager.delete_sector_theme(opts[target_label]):
                        _bust_cache()
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error("Delete failed.")

    st.markdown("---")


# ── Viewer ─────────────────────────────────────────────────────────────────────
themes = _all_themes()
if not themes:
    st.info("📭 No sector themes yet. " + (
        "Use the admin panel above to add one." if is_admin
        else "Ask an admin to add one."
    ))
    st.stop()

# Theme selector — preserve choice across reruns
opts = {t['formal_name']: t['id'] for t in themes}
chosen_label = st.selectbox(
    "Choose a theme 选择主题",
    list(opts.keys()),
    key="sector_explorer_theme",
)
chosen_id = opts[chosen_label]

# Reset selected product + discovery state when the theme changes
if st.session_state.get("_sector_explorer_last_theme") != chosen_id:
    st.session_state.pop("sector_selected_item", None)
    st.session_state.pop("sector_selected_layer", None)
    st.session_state["_sector_explorer_last_theme"] = chosen_id

# ── Load theme ─────────────────────────────────────────────────────────────────
theme = data_manager.get_sector_theme_by_id(chosen_id)
if not theme:
    st.error("Could not load theme.")
    st.stop()

layers = theme.get("layers") or []
if not layers:
    st.warning("This theme has no layers stored. An admin should regenerate it.")
    st.stop()

layers = sorted(layers, key=lambda l: l.get("layer_index", 0))

# ── Render Layers as horizontal columns ────────────────────────────────────────
st.subheader(f"🔗 {theme['formal_name']}")
st.caption("Upstream → Downstream  ·  上游 → 下游")

cols = st.columns(len(layers), gap="small")
for layer_pos, (col, layer) in enumerate(zip(cols, layers)):
    with col:
        idx        = layer.get("layer_index", layer_pos + 1)
        layer_name = layer.get("layer_name", f"Layer {idx}")
        items      = layer.get("items", [])

        st.markdown(
            f"<div style='text-align:center;color:#9ca3af;font-size:11px;font-weight:600;'>"
            f"LAYER {idx}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center;color:#e5e7eb;font-size:13px;font-weight:700;"
            f"margin-bottom:6px;line-height:1.2;'>{layer_name}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        for item_idx, item in enumerate(items):
            highlighted = st.session_state.get("sector_selected_item") == item
            btn_key = f"sec_{chosen_id}_{idx}_{item_idx}"
            if st.button(
                item,
                key=btn_key,
                use_container_width=True,
                type=("primary" if highlighted else "secondary"),
            ):
                # Clear any previous discovery results when a new product is selected
                prev = st.session_state.get("sector_selected_item")
                if prev != item:
                    prev_layer = st.session_state.get("sector_selected_layer", "")
                    prev_pk = _hashlib.md5(
                        f"{chosen_id}|{prev_layer}|{prev}".encode()
                    ).hexdigest()[:12]
                    st.session_state.pop(f"se_results_{prev_pk}", None)
                    st.session_state.pop(f"se_triggered_{prev_pk}", None)
                st.session_state["sector_selected_item"] = item
                st.session_state["sector_selected_layer"] = layer_name
                st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Product Stock Discovery Panel
# ══════════════════════════════════════════════════════════════════════════════
selected       = st.session_state.get("sector_selected_item")
selected_layer = st.session_state.get("sector_selected_layer", "")
sector_name    = theme["formal_name"]

if not selected:
    st.caption("💡 Click any product node above to find related A-share stocks.")
    st.stop()

# Stable short key for session-state entries tied to this (theme, layer, product) triple
_pk            = _hashlib.md5(f"{chosen_id}|{selected_layer}|{selected}".encode()).hexdigest()[:12]
se_results_key = f"se_results_{_pk}"
se_trigger_key = f"se_triggered_{_pk}"

# Cache key in product_peers DB (includes all three dimensions)
_db_key = (
    f"__sector_product__"
    f"|{sector_name.strip().lower()}"
    f"|{selected_layer.strip().lower()}"
    f"|{selected.strip().lower()}"
)

# ── Section header + Clear ─────────────────────────────────────────────────────
hdr_col, clr_col = st.columns([7, 1])
hdr_col.markdown(
    f"### 🔍 {selected}\n"
    f"<span style='color:#9ca3af;font-size:12px;'>"
    f"{sector_name}  ·  {selected_layer}</span>",
    unsafe_allow_html=True,
)
if clr_col.button("✖ Clear", key="se_clear_sel", use_container_width=True):
    st.session_state.pop("sector_selected_item", None)
    st.session_state.pop("sector_selected_layer", None)
    st.session_state.pop(se_results_key, None)
    st.session_state.pop(se_trigger_key, None)
    st.rerun()

# ── Check DB for previously saved results ─────────────────────────────────────
db_record = data_manager.get_product_peers(_db_key)
db_peers  = (db_record.get("peers") or []) if db_record else []
db_method = (db_record.get("source_method") or "") if db_record else ""

# Hydrate session from DB if we have saved data and no fresh query was triggered
if db_peers and se_trigger_key not in st.session_state and se_results_key not in st.session_state:
    st.session_state[se_results_key] = db_peers

# ── Action buttons ─────────────────────────────────────────────────────────────
btn_find_disabled = (
    se_trigger_key in st.session_state and se_results_key not in st.session_state
)
b1, b2, _ = st.columns([2, 2, 4])

if b1.button(
    "🔍 Find Stocks",
    type="primary",
    key="se_find_btn",
    disabled=btn_find_disabled,
    help="Ask DeepSeek to identify top A-share stocks for this product in this sector.",
):
    st.session_state[se_trigger_key] = True
    st.session_state.pop(se_results_key, None)
    st.rerun()

if is_admin and b2.button(
    "🔄 Re-query DeepSeek",
    key="se_requery_btn",
    help="Bust the DB cache and run a fresh DeepSeek query.",
):
    data_manager.upsert_product_peers(_db_key, [], source_method="invalidated")
    st.session_state.pop(se_results_key, None)
    st.session_state[se_trigger_key] = True
    st.rerun()

# Show provenance of currently displayed data
if db_peers and se_trigger_key not in st.session_state:
    if db_method == "admin_curated":
        st.caption("✅ Admin-curated list")
    else:
        st.caption("🤖 DeepSeek results (not yet curated by admin)")

# ── Run DeepSeek if triggered and results not yet in session ──────────────────
if st.session_state.get(se_trigger_key) and se_results_key not in st.session_state:
    with st.spinner(f"Querying DeepSeek for '{selected}' in {sector_name}…"):
        try:
            raw_stocks = peer_discovery.discover_product_stocks(
                product=selected,
                layer_name=selected_layer,
                sector_name=sector_name,
            )
            st.session_state[se_results_key] = raw_stocks
        except RuntimeError as exc:
            st.error(f"❌ {exc}")
            st.session_state.pop(se_trigger_key, None)
            st.stop()

if se_results_key not in st.session_state:
    st.stop()

raw_results = st.session_state[se_results_key]
if not raw_results:
    st.warning("No stocks returned. Try Re-query or check if the product name is specific enough.")
    st.stop()

# ── Validate tickers against stock_basic ──────────────────────────────────────
validation = data_manager.validate_tickers_against_stock_basic(
    [s["ticker"] for s in raw_results]
)

# ── Render stock cards in a responsive grid ───────────────────────────────────
admin_include: dict[str, bool] = {}   # ticker → checkbox state (admin only)

n      = len(raw_results)
n_cols = min(n, 3)   # up to 3 cards per row
cols   = st.columns(n_cols, gap="small")

for i, s in enumerate(raw_results):
    t      = s["ticker"]
    v      = validation.get(t, {})
    valid  = v.get("valid", False)
    name   = v.get("official_name") or s.get("name", "")
    pp     = s.get("primary_product", "")
    badge  = "✅" if valid else "⚠️"
    badge_title = "" if valid else " · not in stock_basic"

    with cols[i % n_cols]:
        with st.container(border=True):
            # Ticker + validity badge on one line
            st.markdown(
                f"<div style='font-size:18px;font-weight:700;line-height:1.2;'>"
                f"{t} <span style='font-size:14px;'>{badge}</span>"
                f"<span style='color:#ef4444;font-size:11px;'>{badge_title}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Company name
            st.markdown(
                f"<div style='font-size:14px;font-weight:600;margin-top:2px;'>{name}</div>",
                unsafe_allow_html=True,
            )
            # Primary product in muted text
            st.markdown(
                f"<div style='color:#6b7280;font-size:12px;margin-top:2px;"
                f"margin-bottom:8px;'>{pp}</div>",
                unsafe_allow_html=True,
            )
            if is_admin:
                # Checkbox label hidden visually but present for accessibility
                admin_include[t] = st.checkbox(
                    "Include in save",
                    value=valid,
                    disabled=not valid,
                    key=f"se_chk_{_pk}_{t}",
                    label_visibility="collapsed",
                    help="Uncheck to exclude this stock from the saved list."
                    if valid else "Not in stock_basic — cannot be saved.",
                )

# ── Admin save panel ──────────────────────────────────────────────────────────
if is_admin:
    st.markdown("---")

    # Build the list to save: only tickers the admin checked AND that passed stock_basic
    to_save = [
        {
            "ticker":          s["ticker"],
            "name":            validation.get(s["ticker"], {}).get("official_name") or s.get("name", ""),
            "primary_product": s.get("primary_product", ""),
        }
        for s in raw_results
        if admin_include.get(s["ticker"], False)
        and validation.get(s["ticker"], {}).get("valid", False)
    ]

    if st.button(
        f"💾 Save {len(to_save)} checked stocks",
        type="primary",
        key="se_save_btn",
        disabled=len(to_save) == 0,
    ):
        ok = data_manager.upsert_product_peers(
            _db_key, to_save, source_method="admin_curated"
        )
        if ok:
            st.session_state[se_results_key] = to_save
            st.session_state.pop(se_trigger_key, None)
            st.success(f"✅ Saved {len(to_save)} stocks for **{selected}**.")
            st.rerun()
        else:
            st.error("Save failed — check DB connection.")
    st.caption(
        "Uncheck any card above to exclude it. "
        "⚠️ tickers not in stock_basic cannot be saved regardless."
    )
else:
    # Non-admin: if data came from DB (admin-curated) no note needed;
    # if it's a fresh session-only query, let them know
    if se_trigger_key in st.session_state and db_method != "admin_curated":
        st.caption(
            "ℹ️ These results are session-only. "
            "Ask an admin to curate and save the list for everyone."
        )

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 · Product Revenue Breakdown  主营业务收入分析
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Phase 2 · Product Revenue Breakdown 主营业务收入分析")
st.caption(
    "Select a stock to see how its revenue is split across product lines "
    "(Tushare `fina_mainbz`, type=P · 按产品分类)."
)

# Only offer stocks that passed stock_basic validation (they have real data)
valid_for_p2 = [
    s for s in raw_results
    if validation.get(s["ticker"], {}).get("valid", False)
]

if not valid_for_p2:
    st.info("No valid stocks available for breakdown — run Phase 1 first.")
    st.stop()

# Stable key scoped to this (theme, layer, product) triple
p2_select_key = f"se_p2_pick_{_pk}"

p2_options = [""] + [
    f"{s['ticker']} · {validation.get(s['ticker'], {}).get('official_name') or s.get('name', '')}"
    for s in valid_for_p2
]

def _on_p2_pick():
    pass  # selection is read directly from session state on next rerun

st.selectbox(
    "Select stock for breakdown 选择标的",
    p2_options,
    key=p2_select_key,
    format_func=lambda x: "Choose a stock…" if x == "" else x,
    on_change=_on_p2_pick,
)

p2_pick = (st.session_state.get(p2_select_key) or "").strip()
if not p2_pick:
    st.caption("Select a stock above to expand its product revenue breakdown.")
    st.stop()

p2_ticker = p2_pick.split(" · ")[0].strip()
p2_name   = p2_pick.split(" · ", 1)[1].strip() if " · " in p2_pick else p2_ticker

# ── Fetch fina_mainbz (cached 24 h — quarterly filings don't change intraday) ─
@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_mainbz(ticker: str):
    return data_manager.fetch_fina_mainbz(ticker, bz_type="P")

with st.spinner(f"Fetching product breakdown for {p2_name} ({p2_ticker})…"):
    bz_df = _fetch_mainbz(p2_ticker)

if bz_df is None or bz_df.empty:
    st.warning(
        f"No product revenue data available for **{p2_ticker}**. "
        "The company may not report by product line, or data is unavailable."
    )
    st.stop()

# ── Period selector ────────────────────────────────────────────────────────────
periods      = sorted(bz_df["end_date"].dropna().unique(), reverse=True)
period_labels = []
for p in periods[:6]:           # up to 6 most recent periods
    ps = str(p)
    label = f"{ps[:4]}-{ps[4:6]}-{ps[6:]}" if len(ps) == 8 else ps
    period_labels.append(label)

chosen_label = st.radio(
    "Reporting period 报告期",
    period_labels,
    horizontal=True,
    key=f"se_p2_period_{_pk}",
)
# Map label back to raw end_date value
chosen_period = periods[period_labels.index(chosen_label)]

period_df = bz_df[bz_df["end_date"] == chosen_period].copy()
if period_df.empty:
    st.warning("No data for selected period.")
    st.stop()

# ── Derived columns ────────────────────────────────────────────────────────────
total_sales = period_df["bz_sales"].sum()
if total_sales > 0:
    period_df["share_pct"] = (period_df["bz_sales"] / total_sales * 100).round(1)
else:
    period_df["share_pct"] = 0.0
period_df["revenue_yi"] = (period_df["bz_sales"] / 1e8).round(2)   # 亿 RMB
period_df["profit_yi"]  = (period_df["bz_profit"] / 1e8).round(2)
period_df = period_df.sort_values("bz_sales", ascending=False).reset_index(drop=True)

# ── Layout: chart left, table right ───────────────────────────────────────────
ch1, ch2 = st.columns([3, 2], gap="medium")

with ch1:
    if _PLOTLY_OK:
        # Horizontal bar sorted ascending so largest is at top visually
        df_plot = period_df.sort_values("bz_sales", ascending=True)
        bar_text = [
            f"{v:.1f}亿 ({p:.1f}%)"
            for v, p in zip(df_plot["revenue_yi"], df_plot["share_pct"])
        ]
        fig = go.Figure(go.Bar(
            x=df_plot["revenue_yi"],
            y=df_plot["bz_item"],
            orientation="h",
            text=bar_text,
            textposition="outside",
            marker_color="#2563eb",
            marker_line_color="#1d4ed8",
            marker_line_width=0.5,
        ))
        fig.update_layout(
            height=max(320, len(period_df) * 52),
            margin=dict(l=10, r=90, t=45, b=40),
            xaxis=dict(
                title="Revenue 营收 (亿 RMB)",
                gridcolor="#e2e8f0", linecolor="#cbd5e1",
            ),
            yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1"),
            title=dict(
                text=f"{p2_name} — Product Revenue · {chosen_label}",
                font=dict(size=13),
            ),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            font=dict(color="#1e293b", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(
            period_df.set_index("bz_item")["revenue_yi"],
            use_container_width=True,
        )

with ch2:
    curr = str(period_df["curr_type"].iloc[0]) if "curr_type" in period_df.columns else "CNY"
    st.markdown(
        f"**{p2_name}** · {chosen_label}  \n"
        f"Total revenue: **{total_sales/1e8:.2f} 亿 {curr}**"
    )
    display_df = (
        period_df[["bz_item", "revenue_yi", "share_pct", "profit_yi"]]
        .rename(columns={
            "bz_item":     "Product / Segment",
            "revenue_yi":  "Revenue (亿)",
            "share_pct":   "Share %",
            "profit_yi":   "Profit (亿)",
        })
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
