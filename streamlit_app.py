"""
ASSRS V2 Enhanced - Advanced Stock Rotation & Selection System
A-share Market Analysis Platform
"""

import streamlit as st
import auth_manager

# set_page_config MUST be the very first st.* call — before any component.
st.set_page_config(
    page_title="ASSRS Dashboard",
    page_icon="📈",
    layout="wide"
)

# Ensure the app_sessions table exists (no-op on Supabase; auto-creates on SQLite)
auth_manager.ensure_sessions_table()

# Restore login from browser cookie before any routing decision.
auth_manager.restore_session_from_cookie()

# Flush a deferred cookie write from the previous login render. The
# CookieManager iframe runs JS to call document.cookie — that JS needs the
# component to stay mounted long enough to execute. By writing here (on the
# render right AFTER login's st.rerun), there's no immediate rerun chaser
# unmounting the iframe before it fires.
_pending_token = st.session_state.pop("_pending_cookie_write", None)
if _pending_token:
    auth_manager.write_session_cookie(_pending_token)

# ── Page registry ────────────────────────────────────────────────────
# Build all pages up front (before the auth check) so st.navigation can
# match the current URL against a registered page even during the
# cookie-restore loading state. This preserves the URL through the rerun
# so a new tab opened to /Interaction_Lab actually lands there instead of
# defaulting to the first page.
login_page = st.Page("pages/Login.py", title="Login 登录", icon="🔐")

pages = {
    "📊 Sector 板块": [
        st.Page("pages/sector_dashboard.py", title="Dashboard"),
        st.Page("pages/sector_interaction_lab.py", title="Interaction Lab 互动"),
        st.Page("pages/sector_performance_rotation.py", title="Rotation 轮动"),
    ],
    "📈 Stock 股票": [
        st.Page("pages/Equity_Brief.py", title="Equity Report 个股研报", url_path="equity-brief"),
        st.Page("pages/2_Single_Stock_Analysis_个股分析.py", title="Technical Analysis 技术分析", url_path="single-stock-analysis"),
        st.Page("pages/sector_stock_selector.py", title="Stock Selector 选股器"),
        st.Page("pages/watchlist.py", title="Watchlist 观察名单"),
        st.Page("pages/watchlist_earnings_calendar.py", title="Earnings Calendar 财报日历"),
        st.Page("pages/sector_explorer.py", title="Sector Explorer 产业链"),
    ],
    "💼 Portfolio 组合": [
        st.Page("pages/Portfolio_Optimization.py", title="Optimization 组合优化"),
        st.Page("pages/Fund_Manager.py", title="Portfolio Management 组合管理"),
        st.Page("pages/ai_supply_chain3.py", title="AI Supply Chain AI供应链"),
    ],
    "🌊 Strategy 策略": [
        st.Page("pages/wave_trader.py", title="Wave Trader 波段交易"),
        st.Page("pages/pair_trader.py", title="Pair Trader 配对交易", url_path="pair-trader"),
        st.Page("pages/lead_lag_analysis.py", title="Lead-Lag 领先滞后", url_path="lead-lag"),
        st.Page("pages/sentiment_mean_reversion.py", title="Mean Reversion 反转候选", url_path="mean-reversion"),
        st.Page("pages/t_trading_scanner.py", title="T-Trading Scanner 做T候选", url_path="t-trading"),
    ],
    "🔔 Alerts 提示": [
        st.Page("pages/4_Todays_Alerts_今日提醒.py", title="Today's Alerts 今日提醒"),
    ],
    "🌍 Macro 宏观": [
        st.Page("pages/macro_commodities.py", title="Macro & Commodities 宏观与大宗", url_path="macro-commodities"),
    ],
    "📖 About 关于": [
        st.Page("pages/about.py", title="声明"),
    ],
}

# Admin-only section — only injected when the current user is an admin
if auth_manager.is_admin():
    pages["⚙️ Admin"] = [
        st.Page("pages/Admin_Invite_Users.py", title="Invite Users 邀请用户", icon="👥"),
        st.Page("pages/Admin_Sector_Management.py", title="Sector Management 板块管理", icon="⚙️", url_path="admin-sector-management"),
    ]


if not auth_manager.is_logged_in():
    # On the very first render of a new tab, the CookieManager iframe is
    # still loading. To preserve the requested URL through the rerun, we
    # register the full pages dict so st.navigation can match the URL —
    # but we hide the menu and skip pg.run() so nothing actually renders
    # except a loading message.
    #
    # When the iframe responds, Streamlit reruns; on that rerun
    # is_logged_in() is True, st.navigation runs normally, and the page
    # corresponding to the preserved URL is rendered.
    if not st.session_state.get("_cookie_check_attempted"):
        st.session_state["_cookie_check_attempted"] = True
        st.navigation(pages, position="hidden")  # URL → page mapping only
        st.info("🔄 Restoring session…")
        st.stop()

    # CookieManager has responded with no valid cookie — really need login
    pg = st.navigation([login_page])
    pg.run()
    st.stop()

# Logged in — clear the check flag so logout → re-login still gets a clean
# restore attempt on any new tabs.
st.session_state.pop("_cookie_check_attempted", None)


user = auth_manager.get_current_user()

# ── Sidebar: user info + logout ──────────────────────────────────
with st.sidebar:
    st.markdown("---")
    role_badge = "👑 Admin" if auth_manager.is_admin() else "👤 User"
    st.caption(f"{role_badge} | **{user['username']}**")
    if st.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()

# Create navigation and run the selected page
pg = st.navigation(pages)
pg.run()
