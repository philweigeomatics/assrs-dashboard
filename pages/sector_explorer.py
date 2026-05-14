"""
Macro Sector Explorer 行业产业链浏览器
Display chronological supply-chain layers for industry themes.

  - Admins: an st.expander control panel for adding / deleting themes.
  - Viewers: pick a saved theme and click any node to query related stocks.

Two analysis modes (tabs):
  - Single Product: click a product node → AI finds A-share suppliers
  - Layer Sankey: pick a layer → M2M company-product Sankey diagram
"""

import hashlib as _hashlib

import pandas as _pd
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
user     = auth_manager.get_current_user()
is_admin = auth_manager.is_admin()

# ── Page Header ────────────────────────────────────────────────────────────────
st.title("🌐 Macro Sector Explorer | 行业产业链浏览器")
st.caption("Click on any node to find related stocks · 点击节点查找相关股票")
st.markdown("---")

# ── Sankey colour palette ──────────────────────────────────────────────────────
_PRODUCT_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#f97316", "#ec4899",
]
_COMPANY_COLOR = "#64748b"

def _hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Phase 2 data fetcher (module-level for cache decorator) ───────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_mainbz(ticker: str):
    return data_manager.fetch_fina_mainbz(ticker, bz_type="P")


def _render_phase2(valid_for_p2: list, validation: dict, pk_suffix: str) -> None:
    """Render Phase 2 product revenue breakdown section (reusable for both tabs)."""
    st.markdown("### 📊 Phase 2 · Product Revenue Breakdown 主营业务收入分析")
    st.caption(
        "Select a stock · choose a base period · compare same period across years "
        "(Tushare `fina_mainbz`, type=P · 按产品分类)."
    )

    if not valid_for_p2:
        st.info("No valid stocks available for breakdown — run Phase 1 first.")
        return

    p2_select_key = f"se_p2_pick_{pk_suffix}"
    p2_options    = [""] + [
        f"{s['ticker']} · {validation.get(s['ticker'], {}).get('official_name') or s.get('name', '')}"
        for s in valid_for_p2
    ]

    p2_sel_col, p2_nav_col = st.columns([3, 1])
    with p2_sel_col:
        st.selectbox(
            "Select stock for breakdown 选择标的",
            p2_options,
            key=p2_select_key,
            format_func=lambda x: "Choose a stock…" if x == "" else x,
        )
    with p2_nav_col:
        st.write(""); st.write("")
        _p2_cur = (st.session_state.get(p2_select_key) or "").strip()
        if _p2_cur and st.button(
            "📈 Single Stock Analysis",
            key=f"se_p2_goto_ssa_{pk_suffix}",
            use_container_width=True,
            help="Open this stock in Single Stock Analysis.",
        ):
            _p2_t = _p2_cur.split(" · ")[0].strip()
            st.session_state["active_ticker"]  = _p2_t
            st.session_state["ssa_stock_pick"] = _p2_cur
            st.switch_page("pages/2_Single_Stock_Analysis_个股分析.py")

    p2_pick = (st.session_state.get(p2_select_key) or "").strip()
    if not p2_pick:
        st.caption("Select a stock above to expand its product revenue breakdown.")
        return

    p2_ticker = p2_pick.split(" · ")[0].strip()
    p2_name   = p2_pick.split(" · ", 1)[1].strip() if " · " in p2_pick else p2_ticker

    with st.spinner(f"Fetching product breakdown for {p2_name} ({p2_ticker})…"):
        bz_df = _fetch_mainbz(p2_ticker)

    if bz_df is None or bz_df.empty:
        st.warning(
            f"No product revenue data available for **{p2_ticker}**. "
            "The company may not report by product line, or Tushare data is unavailable."
        )
        return

    all_periods    = sorted(bz_df["end_date"].dropna().unique(), reverse=True)

    def _fmt_period(p):
        ps = str(p)
        return f"{ps[:4]}-{ps[4:6]}-{ps[6:]}" if len(ps) == 8 else ps

    base_period_opts = all_periods[:4]
    chosen_base = st.radio(
        "Base period 基准报告期",
        base_period_opts,
        horizontal=True,
        key=f"se_p2_period_{pk_suffix}",
        format_func=_fmt_period,
    )

    month_day   = str(chosen_base)[4:]
    yoy_periods = sorted(
        [p for p in all_periods if str(p).endswith(month_day)],
        reverse=True,
    )[:3]
    if not yoy_periods:
        yoy_periods = [chosen_base]

    yoy_labels = [_fmt_period(p) for p in yoy_periods]

    yoy_frames = {}
    for period in yoy_periods:
        df_p = bz_df[bz_df["end_date"] == period].copy()
        if df_p.empty:
            continue
        total = df_p["bz_sales"].sum()
        df_p["revenue_yi"] = (df_p["bz_sales"] / 1e8).round(2)
        df_p["profit_yi"]  = (df_p["bz_profit"] / 1e8).round(2)
        df_p["share_pct"]  = ((df_p["bz_sales"] / total * 100).round(1) if total > 0 else 0.0)
        yoy_frames[period] = df_p

    if not yoy_frames:
        st.warning("No data found for the selected period.")
        return

    base_df      = yoy_frames.get(yoy_periods[0], next(iter(yoy_frames.values())))
    products_asc = base_df.sort_values("bz_sales", ascending=True)["bz_item"].tolist()

    YEAR_COLORS = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]

    ch1, ch2 = st.columns([3, 2], gap="medium")

    with ch1:
        if _PLOTLY_OK:
            fig = go.Figure()
            for idx, (period, df_p) in enumerate(
                sorted(yoy_frames.items(), reverse=False)
            ):
                revenue_by_product = df_p.set_index("bz_item")["revenue_yi"]
                x_vals = [float(revenue_by_product.get(p, 0)) for p in products_asc]
                label  = _fmt_period(period)
                color  = YEAR_COLORS[len(yoy_periods) - 1 - idx % len(YEAR_COLORS)]
                fig.add_trace(go.Bar(
                    x=x_vals,
                    y=products_asc,
                    orientation="h",
                    name=label,
                    marker_color=color,
                    opacity=0.9,
                    text=[f"{v:.1f}" if v else "" for v in x_vals],
                    textposition="outside",
                    textfont=dict(size=10),
                ))
            fig.update_layout(
                barmode="group",
                height=max(340, len(products_asc) * 55 + 80),
                margin=dict(l=10, r=80, t=50, b=40),
                xaxis=dict(
                    title="Revenue 营收 (亿 RMB)",
                    gridcolor="#e2e8f0", linecolor="#cbd5e1",
                ),
                yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1"),
                title=dict(
                    text=f"{p2_name} — Product Revenue YoY · {' vs '.join(yoy_labels)}",
                    font=dict(size=13),
                ),
                plot_bgcolor="#f8fafc",
                paper_bgcolor="#ffffff",
                font=dict(color="#1e293b", size=12),
                legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(
                base_df.set_index("bz_item")["revenue_yi"],
                use_container_width=True,
            )

    with ch2:
        curr = str(base_df["curr_type"].iloc[0]) if "curr_type" in base_df.columns else "CNY"
        st.markdown(f"**{p2_name}** · Revenue (亿 {curr})")

        pivot_rows = {}
        for period, df_p in yoy_frames.items():
            label = _fmt_period(period)
            for _, row in df_p.iterrows():
                item = row["bz_item"]
                if item not in pivot_rows:
                    pivot_rows[item] = {}
                pivot_rows[item][label] = row["revenue_yi"]

        pivot_df  = _pd.DataFrame(pivot_rows).T
        col_order = [_fmt_period(p) for p in yoy_periods if _fmt_period(p) in pivot_df.columns]
        pivot_df  = pivot_df[col_order].fillna(0.0)

        if len(col_order) >= 2:
            newest, prev = col_order[0], col_order[1]
            pivot_df["YoY Δ%"] = (
                ((pivot_df[newest] - pivot_df[prev]) / pivot_df[prev].replace(0, float("nan"))) * 100
            ).round(1)

        if col_order:
            pivot_df = pivot_df.sort_values(col_order[0], ascending=False)

        st.dataframe(pivot_df, use_container_width=True)


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
                help="Anything goes — AI will formalise it.",
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
                                    raw_input   = raw,
                                    formal_name = data["name"],
                                    layers_data = data,
                                    created_by  = user["username"],
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
            themes_del = _all_themes()
            if not themes_del:
                st.info("No themes to delete.")
            else:
                opts_del = {
                    f"{t['formal_name']}  (raw: {t['raw_input']})": t['id']
                    for t in themes_del
                }
                target_label = st.selectbox(
                    "Select theme to delete",
                    list(opts_del.keys()),
                    key="sector_admin_del_choice",
                )
                if st.button("🗑️ Delete", type="secondary"):
                    if data_manager.delete_sector_theme(opts_del[target_label]):
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
st.caption(
    "Upstream → Downstream  ·  上游 → 下游  ·  "
    "Click a product for **Single Product** analysis, or use **Layer Sankey** tab for M2M view."
)

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
                prev = st.session_state.get("sector_selected_item")
                if prev != item:
                    prev_layer = st.session_state.get("sector_selected_layer", "")
                    prev_pk = _hashlib.md5(
                        f"{chosen_id}|{prev_layer}|{prev}".encode()
                    ).hexdigest()[:12]
                    st.session_state.pop(f"se_results_{prev_pk}", None)
                    st.session_state.pop(f"se_triggered_{prev_pk}", None)
                st.session_state["sector_selected_item"]  = item
                st.session_state["sector_selected_layer"] = layer_name
                st.rerun()

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS — either/or: Single Product  |  Layer Sankey
# ══════════════════════════════════════════════════════════════════════════════
_tab1, _tab2 = st.tabs(["🔍 Single Product  单品分析", "🔬 Layer Sankey  板块图谱"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PRODUCT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════
with _tab1:
    selected       = st.session_state.get("sector_selected_item")
    selected_layer = st.session_state.get("sector_selected_layer", "")
    sector_name    = theme["formal_name"]

    if not selected:
        st.caption("💡 Click any product node above to explore related A-share stocks.")
    else:
        _pk            = _hashlib.md5(f"{chosen_id}|{selected_layer}|{selected}".encode()).hexdigest()[:12]
        se_results_key = f"se_results_{_pk}"
        se_trigger_key = f"se_triggered_{_pk}"

        _db_key = (
            f"__sector_product__"
            f"|{sector_name.strip().lower()}"
            f"|{selected_layer.strip().lower()}"
            f"|{selected.strip().lower()}"
        )

        # ── Section header ─────────────────────────────────────────────────────
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

        # ── Hydrate from DB ────────────────────────────────────────────────────
        db_record = data_manager.get_product_peers(_db_key)
        db_peers  = (db_record.get("peers") or []) if db_record else []
        db_method = (db_record.get("source_method") or "") if db_record else ""

        if db_peers and se_trigger_key not in st.session_state and se_results_key not in st.session_state:
            st.session_state[se_results_key] = db_peers

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_find_disabled = (
            se_trigger_key in st.session_state and se_results_key not in st.session_state
        )
        b1, b2, _ = st.columns([2, 2, 4])

        if b1.button(
            "🔍 Find Stocks",
            type="primary",
            key="se_find_btn",
            disabled=btn_find_disabled,
            help="Ask AI to identify top A-share stocks for this product in this sector.",
        ):
            st.session_state[se_trigger_key] = True
            st.session_state.pop(se_results_key, None)
            st.rerun()

        if is_admin and b2.button(
            "🔄 Re-query AI",
            key="se_requery_btn",
            help="Bust the DB cache and run a fresh AI query.",
        ):
            data_manager.upsert_product_peers(_db_key, [], source_method="invalidated")
            st.session_state.pop(se_results_key, None)
            st.session_state[se_trigger_key] = True
            st.rerun()

        if db_peers and se_trigger_key not in st.session_state:
            if db_method == "admin_curated":
                st.caption("✅ Admin-curated list")
            else:
                st.caption("🤖 AI results (not yet curated by admin)")

        # ── Run AI ─────────────────────────────────────────────────────────────
        if st.session_state.get(se_trigger_key) and se_results_key not in st.session_state:
            with st.spinner(f"Querying AI for '{selected}' in {sector_name}…"):
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

        if se_results_key in st.session_state:
            raw_results = st.session_state[se_results_key]

            if not raw_results:
                st.warning("No stocks returned. Try Re-query or check if the product name is specific enough.")
            else:
                # ── Validate tickers ───────────────────────────────────────────
                validation = data_manager.validate_tickers_against_stock_basic(
                    [s["ticker"] for s in raw_results]
                )

                # ── Stock cards ────────────────────────────────────────────────
                stock_include: dict[str, bool] = {}
                n_cols = min(len(raw_results), 3)
                s_cols = st.columns(n_cols, gap="small")

                for i, s in enumerate(raw_results):
                    t           = s["ticker"]
                    v           = validation.get(t, {})
                    valid       = v.get("valid", False)
                    name        = v.get("official_name") or s.get("name", "")
                    pp          = s.get("primary_product", "")
                    badge       = "✅" if valid else "⚠️"
                    badge_title = "" if valid else " · not in stock_basic"

                    with s_cols[i % n_cols]:
                        with st.container(border=True):
                            st.markdown(
                                f"<div style='font-size:18px;font-weight:700;line-height:1.2;'>"
                                f"{t} <span style='font-size:14px;'>{badge}</span>"
                                f"<span style='color:#ef4444;font-size:11px;'>{badge_title}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='font-size:14px;font-weight:600;margin-top:2px;'>{name}</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='color:#6b7280;font-size:12px;margin-top:2px;"
                                f"margin-bottom:8px;'>{pp}</div>",
                                unsafe_allow_html=True,
                            )
                            stock_include[t] = st.checkbox(
                                "Include",
                                value=valid,
                                disabled=not valid,
                                key=f"se_chk_{_pk}_{t}",
                                label_visibility="collapsed",
                                help=(
                                    "Uncheck to exclude from pair analysis and save."
                                    if valid else "Not in stock_basic — cannot be used."
                                ),
                            )

                # ── Action bar ─────────────────────────────────────────────────
                to_act = [
                    {
                        "ticker":          s["ticker"],
                        "name":            validation.get(s["ticker"], {}).get("official_name") or s.get("name", ""),
                        "primary_product": s.get("primary_product", ""),
                    }
                    for s in raw_results
                    if stock_include.get(s["ticker"], False)
                    and validation.get(s["ticker"], {}).get("valid", False)
                ]
                act_tickers = [s["ticker"] for s in to_act]
                n_act       = len(act_tickers)

                st.markdown("---")

                if is_admin:
                    save_col, pt_col, cap_col = st.columns([2, 2, 3])
                else:
                    save_col = None
                    pt_col, cap_col = st.columns([2, 5])

                if is_admin:
                    with save_col:
                        if st.button(
                            f"💾 Save {n_act} to peers",
                            type="primary",
                            key="se_save_btn",
                            disabled=n_act == 0,
                            use_container_width=True,
                        ):
                            ok = data_manager.upsert_product_peers(
                                _db_key, to_act, source_method="admin_curated"
                            )
                            if ok:
                                st.session_state[se_results_key] = to_act
                                st.session_state.pop(se_trigger_key, None)
                                st.success(f"✅ Saved {n_act} stocks for **{selected}**.")
                                st.rerun()
                            else:
                                st.error("Save failed — check DB connection.")

                with pt_col:
                    if st.button(
                        f"🔗 Analyse {n_act} in Pair Trader",
                        key="se_to_pt_btn",
                        disabled=n_act < 2,
                        use_container_width=True,
                        help=(
                            "Need at least 2 checked ✅ stocks for pair analysis."
                            if n_act < 2 else
                            f"Send {n_act} checked stocks to Pair Trader."
                        ),
                    ):
                        st.session_state["pt_preload_tickers"]  = "\n".join(act_tickers)
                        st.session_state["pt_from_sector"]      = True
                        st.session_state["pt_from_sector_name"] = f"{selected}  ·  {sector_name}"
                        st.switch_page("pages/pair_trader.py")

                with cap_col:
                    if n_act == 0:
                        st.caption("Check at least one valid stock above to enable actions.")
                    elif n_act == 1:
                        st.caption(f"**{act_tickers[0]}** selected · check ≥ 2 stocks to enable pair trade.")
                    else:
                        st.caption(f"**{', '.join(act_tickers)}** — {n_act} stocks selected.")
                    if not is_admin and se_trigger_key in st.session_state and db_method != "admin_curated":
                        st.caption("ℹ️ Session-only results. Ask an admin to save this list to the database.")

                # ── Phase 2 ────────────────────────────────────────────────────
                st.markdown("---")
                valid_for_p2 = [
                    s for s in raw_results
                    if validation.get(s["ticker"], {}).get("valid", False)
                ]
                _render_phase2(valid_for_p2, validation, _pk)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LAYER SANKEY  (M2M company-product relationships)
# ══════════════════════════════════════════════════════════════════════════════
with _tab2:
    if not _PLOTLY_OK:
        st.error("Plotly is not installed. Run `pip install plotly` to enable Sankey diagrams.")
    else:
        st.subheader("🔬 Layer Sankey — M2M Company Relationships")
        st.caption(
            "Select a supply-chain layer, then load the Sankey to see which A-share companies "
            "supply each product. Admin users can curate the M2M table and save to the peer database."
        )

        # ── Layer selector ─────────────────────────────────────────────────────
        layer_name_map = {
            l.get("layer_name", f"Layer {l.get('layer_index', i + 1)}"): l
            for i, l in enumerate(layers)
        }
        sel_tab2_layer = st.selectbox(
            "Select supply-chain layer  选择产业链层",
            list(layer_name_map.keys()),
            key=f"se_tab2_layer_{chosen_id}",
        )
        sel_layer      = layer_name_map[sel_tab2_layer]
        sel_layer_idx  = sel_layer.get("layer_index", 0)
        sel_layer_items = sel_layer.get("items", [])

        st.caption(f"**{theme['formal_name']}  ·  Layer {sel_layer_idx}** · {len(sel_layer_items)} products")

        # ── Data keys (keyed per layer so each layer caches independently) ─────
        _sk_data_key    = f"se_m2m_data_{chosen_id}_{sel_layer_idx}"
        _sk_trigger_key = f"se_m2m_trigger_{chosen_id}_{sel_layer_idx}"

        # ── Controls row ───────────────────────────────────────────────────────
        _sk_ctl_left, _sk_ctl_right = st.columns([3, 1])
        with _sk_ctl_left:
            if _sk_data_key not in st.session_state and not st.session_state.get(_sk_trigger_key):
                if st.button(
                    "🔬 Load Sankey",
                    type="primary",
                    key=f"sk_load_{chosen_id}_{sel_layer_idx}",
                    help=f"Query AI for all {len(sel_layer_items)} products in this layer.",
                ):
                    st.session_state[_sk_trigger_key] = True
                    st.rerun()
        if is_admin:
            with _sk_ctl_right:
                if st.button(
                    "🔄 Re-query",
                    key=f"sk_requery_{chosen_id}_{sel_layer_idx}",
                    help="Run a fresh AI query, overwriting cached results.",
                ):
                    st.session_state.pop(_sk_data_key, None)
                    st.session_state[_sk_trigger_key] = True
                    st.rerun()

        # ── AI query ───────────────────────────────────────────────────────────
        if st.session_state.get(_sk_trigger_key):
            with st.spinner(f"Querying AI for all products in '{sel_tab2_layer}'… (may take ~30 s)"):
                try:
                    _m2m_result = peer_discovery.discover_layer_m2m(
                        sector_name  = theme["formal_name"],
                        layer_name   = sel_tab2_layer,
                        layer_items  = sel_layer_items,
                    )
                    st.session_state[_sk_data_key] = _m2m_result
                    st.session_state.pop(_sk_trigger_key, None)
                    st.rerun()
                except Exception as _exc:
                    st.error(f"AI query failed: {_exc}")
                    st.session_state.pop(_sk_trigger_key, None)

        _m2m = st.session_state.get(_sk_data_key)

        if _m2m is None:
            st.info("Click **Load Sankey** above to query AI for M2M company relationships.")
        elif not _m2m:
            st.info("No results returned — try Re-query.")
        else:
            # ── Build company + product data ───────────────────────────────────
            _all_cos: dict[str, str] = {}
            for _stocks in _m2m.values():
                for _s in _stocks:
                    _all_cos[_s["ticker"]] = _s.get("name", "")

            _products   = list(_m2m.keys())
            _col_keys   = list(_all_cos.keys())
            _col_labels = {t: f"{t} · {_all_cos[t]}" for t in _col_keys}

            # ── Boolean matrix ─────────────────────────────────────────────────
            _matrix_rows = {}
            for _prod in _products:
                _prod_tickers = {_s["ticker"] for _s in _m2m[_prod]}
                _matrix_rows[_prod] = {_col_labels[t]: (t in _prod_tickers) for t in _col_keys}
            _df = _pd.DataFrame(_matrix_rows).T

            st.markdown("**M2M Relationships** — uncheck to remove a link from the Sankey and peer saves")
            if is_admin:
                _edited = st.data_editor(
                    _df,
                    use_container_width=True,
                    key=f"sk_table_{chosen_id}_{sel_layer_idx}",
                )
            else:
                st.dataframe(_df, use_container_width=True)
                _edited = _df

            # ── Admin: save peers per product ──────────────────────────────────
            if is_admin:
                if st.button("💾 Save peers per product", key=f"sk_save_{chosen_id}_{sel_layer_idx}", type="primary"):
                    _saved_n = 0
                    for _prod in _products:
                        _db_key_prod = (
                            f"__sector_product__"
                            f"|{theme['formal_name'].strip().lower()}"
                            f"|{sel_tab2_layer.strip().lower()}"
                            f"|{_prod.strip().lower()}"
                        )
                        _checked = [
                            t for t in _col_keys
                            if _edited.loc[_prod, _col_labels[t]]
                        ]
                        _to_save = [
                            {"ticker": t, "name": _all_cos[t], "primary_product": _prod}
                            for t in _checked
                        ]
                        data_manager.upsert_product_peers(
                            _db_key_prod, _to_save, source_method="admin_curated"
                        )
                        _saved_n += len(_to_save)
                    st.success(f"✅ Saved {_saved_n} product-company links across {len(_products)} products.")

            st.markdown("---")

            # ── Sankey diagram ─────────────────────────────────────────────────
            _n_prod = len(_products)
            _n_cos  = len(_col_keys)

            # Per-product colors; companies use neutral gray
            _prod_color_map = {
                p: _PRODUCT_COLORS[i % len(_PRODUCT_COLORS)]
                for i, p in enumerate(_products)
            }
            _node_colors = (
                [_prod_color_map[p] for p in _products] +
                [_COMPANY_COLOR] * _n_cos
            )
            _node_labels = _products + [_all_cos[t] for t in _col_keys]

            _sources, _targets, _vals, _link_colors, _link_labels = [], [], [], [], []
            for _pi, _prod in enumerate(_products):
                _pc = _prod_color_map[_prod]
                _link_rgba = _hex_to_rgba(_pc, 0.35)
                for _ci, _t in enumerate(_col_keys):
                    if _edited.loc[_prod, _col_labels[_t]]:
                        _sources.append(_pi)
                        _targets.append(_n_prod + _ci)
                        _vals.append(1)
                        _link_colors.append(_link_rgba)
                        _link_labels.append(f"{_prod} → {_all_cos[_t]}")

            if _sources:
                _sk_height = max(500, _n_prod * 80 + _n_cos * 30 + 150)
                _sk_fig = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=_node_labels,
                        color=_node_colors,
                        pad=25,
                        thickness=25,
                        line=dict(color="#1e293b", width=0.5),
                    ),
                    link=dict(
                        source=_sources,
                        target=_targets,
                        value=_vals,
                        label=_link_labels,
                        color=_link_colors,
                    ),
                ))
                _sk_fig.update_layout(
                    height=_sk_height,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="#ffffff",
                    font=dict(size=14, color="#1e293b"),
                )
                st.plotly_chart(_sk_fig, use_container_width=True)
            else:
                st.info("No checked relationships to display in the Sankey.")

            st.markdown("---")

            # ── Action bar ─────────────────────────────────────────────────────
            _unique_cos = sorted({
                t for t in _col_keys
                if any(_edited.loc[_prod, _col_labels[t]] for _prod in _products)
            })
            _co_opts = [f"{t} · {_all_cos[t]}" for t in _unique_cos]

            st.markdown("**Select companies for actions**")
            _sk_selected = st.multiselect(
                "Companies",
                options=_co_opts,
                default=_co_opts,
                key=f"sk_multisel_{chosen_id}_{sel_layer_idx}",
                label_visibility="collapsed",
            )
            _sk_tickers = [x.split(" · ")[0].strip() for x in _sk_selected]
            _n_sk = len(_sk_tickers)

            _sk_act_cols = st.columns([2, 2, 3]) if is_admin else st.columns([2, 5])
            with _sk_act_cols[0]:
                if st.button(
                    f"🔗 Pair Trader ({_n_sk})",
                    key=f"sk_to_pt_{chosen_id}_{sel_layer_idx}",
                    disabled=_n_sk < 2,
                    use_container_width=True,
                ):
                    st.session_state["pt_preload_tickers"]  = "\n".join(_sk_tickers)
                    st.session_state["pt_from_sector"]      = True
                    st.session_state["pt_from_sector_name"] = f"{sel_tab2_layer}  ·  {theme['formal_name']}"
                    st.switch_page("pages/pair_trader.py")

            if is_admin:
                with _sk_act_cols[1]:
                    _sector_input = st.text_input(
                        "New sector name",
                        placeholder="e.g. 液冷散热",
                        key=f"sk_sector_name_{chosen_id}_{sel_layer_idx}",
                        label_visibility="collapsed",
                    )
                    if st.button(
                        f"⚙️ Create Sector ({_n_sk})",
                        key=f"sk_to_sm_{chosen_id}_{sel_layer_idx}",
                        disabled=(_n_sk < 2 or not _sector_input.strip()),
                        use_container_width=True,
                        help="Pre-load these companies into the New Sector tab of Sector Management.",
                    ):
                        st.session_state["_new_sector_stocks"]   = _sk_tickers
                        st.session_state["_preseed_sector_name"] = _sector_input.strip()
                        st.switch_page("pages/Admin_Sector_Management.py")

            # ── Phase 2 for Sankey companies ───────────────────────────────────
            st.markdown("---")
            # Build valid_for_p2 from all checked Sankey companies
            _sk_valid_tickers = sorted({
                t for t in _col_keys
                if any(_edited.loc[_prod, _col_labels[t]] for _prod in _products)
            })
            if _sk_valid_tickers:
                _sk_validation = data_manager.validate_tickers_against_stock_basic(_sk_valid_tickers)
                _valid_for_p2_t2 = [
                    {
                        "ticker": t,
                        "name":   _sk_validation.get(t, {}).get("official_name") or _all_cos.get(t, ""),
                        "primary_product": "",
                    }
                    for t in _sk_valid_tickers
                    if _sk_validation.get(t, {}).get("valid", False)
                ]
                _pk_t2 = f"t2_{chosen_id}_{sel_layer_idx}"
                _render_phase2(_valid_for_p2_t2, _sk_validation, _pk_t2)
