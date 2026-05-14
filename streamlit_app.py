"""
ASSRS V2 Enhanced - Advanced Stock Rotation & Selection System
A-share Market Analysis Platform
"""

import streamlit as st
import auth_manager


st.set_page_config(
    page_title="ASSRS Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Always define login page ─────────────────────────────────────────
login_page = st.Page("pages/Login.py", title="Login 登录", icon="🔐")


if not auth_manager.is_logged_in():
    # Only login page available
    pg = st.navigation([login_page])
    pg.run()
    st.stop()


user = auth_manager.get_current_user();

# ── Sidebar: user info + logout ──────────────────────────────────
with st.sidebar:
    st.markdown("---")
    role_badge = "👑 Admin" if auth_manager.is_admin() else "👤 User"
    st.caption(f"{role_badge} | **{user['username']}**")
    if st.button("🚪 Logout", use_container_width=True):
        auth_manager.logout()
        st.rerun()

    # ── Authenticated pages ──────────────────────────────────────────


pages = {
    "📊 Sector 板块": [
        st.Page("pages/sector_dashboard.py", title="Dashboard"),
        st.Page("pages/sector_interaction_lab.py", title="Interaction Lab 互动"),
        st.Page("pages/sector_performance_rotation.py", title="Rotation 轮动"),
    ],
    "📈 Stock 股票": [
        st.Page("pages/Equity_Brief.py", title="Equity Report 个股研报"),
        st.Page("pages/2_Single_Stock_Analysis_个股分析.py", title="Technical Analysis 技术分析"),
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
        st.Page("pages/pair_trader.py", title="Pair Trader 配对交易"),
        st.Page("pages/lead_lag_analysis.py", title="Lead-Lag 领先滞后"),
    ],
    "🔔 Alerts 提示": [
        st.Page("pages/4_Todays_Alerts_今日提醒.py", title="Today's Alerts 今日提醒"),
    ],
    "📖 About 关于": [
        st.Page("pages/about.py", title="声明"),
    ],
}

# Admin-only section — only injected when the current user is an admin
if auth_manager.is_admin():
    pages["⚙️ Admin"] = [
        st.Page("pages/Admin_Invite_Users.py", title="Invite Users 邀请用户", icon="👥"),
        st.Page("pages/Admin_Sector_Management.py", title="Sector Management 板块管理", icon="⚙️"),
    ]

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()



