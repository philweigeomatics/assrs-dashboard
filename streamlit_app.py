"""
ASSRS V2 Enhanced - Advanced Stock Rotation & Selection System
A-share Market Analysis Platform
"""

import streamlit as st


st.set_page_config(
    page_title="ASSRS Dashboard",
    page_icon="📈",
    layout="wide"
)

# Define page structure with collapsible sections
pages = {
    "📊 Sector 板块": [
        st.Page("pages/sector_dashboard.py", title="Dashboard"),
        st.Page("pages/sector_interaction_lab.py", title="Interaction Lab 互动"),
        st.Page("pages/sector_performance_rotation.py", title="Rotation 轮动"),
    ],
    "📈 Stock 股票": [
        st.Page("pages/2_Single_Stock_Analysis_个股分析.py", title="Stock Analysis 个股分析"),
        st.Page("pages/sector_stock_selector.py", title="Stock Selector 选股器"),
        st.Page("pages/watchlist.py", title="Watchlist 观察名单"),
    ],
    "💼 Portfolio 组合": [
        st.Page("pages/3_Portfolio_Optimization_组合优化.py", title="Optimization 组合优化"),
    ],
    "🔔 Alerts 提示": [
        st.Page("pages/4_Todays_Alerts_今日提醒.py", title="Today's Alerts 今日提醒"),
    ],

    "📖 About 关于": [
        st.Page("pages/about.py", title="声明"),
    ]
}

# Create navigation
pg = st.navigation(pages)

# Run the selected page
pg.run()



