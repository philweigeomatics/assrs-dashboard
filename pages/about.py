# ==========================================
# HEADER
# ==========================================
import streamlit as st

st.markdown("### Advanced Stock Rotation & Selection System | 高级股票轮动与选股系统")

st.markdown("---")

# ==========================================
# ABOUT / 关于
# ==========================================

col_en, col_cn = st.columns(2)

with col_en:
    st.markdown("## 📖 About")
    st.markdown("""
    **ASSRS** is an **analysis platform** designed to assist traders and investors in researching stock sectors, 
individual equities, and portfolio strategies. This tool provides data visualization and analytical insights 
to support your investment research process. This platform is designed for quantitative traders and investors who want data-driven insights into 
    China's stock market dynamics, with a focus on sector momentum and individual stock selection.
    """)

with col_cn:
    st.markdown("## 📖 关于")
    st.markdown("""
    **ASSRS**是一个A股市场股票和板块的分析平台。
    该系统结合板块轮动信号、技术分析和投资组合优化，识别高概率交易机会。
    
    本平台专为量化交易者和投资者设计，提供数据驱动的中国股市洞察，
    重点关注板块动量和个股选择。
    """)

st.markdown("---")

# ==========================================
# FEATURES / 功能特性
# ==========================================

st.markdown("## ✨ Key Features | 核心功能")

# Feature 1: Sector Analysis
st.markdown("### 1️⃣ Sector Rotation Analysis | 板块轮动分析")

col1_en, col1_cn = st.columns(2)

with col1_en:
    st.markdown("""
    **📊 Real-time sector scoring and signals**
    - Machine learning regime detection
    - Market breadth analysis
    - Sector correlation and rotation metrics
    - Interactive drill-down charts
    - Actionable BUY/SELL/HOLD signals with position sizing
    
    **Use Case:** Identify which sectors are leading or lagging the market, 
    and allocate capital accordingly.
    """)

with col1_cn:
    st.markdown("""
    **📊 实时板块评分与信号**
    - 机器学习市场状态检测
    - 市场广度分析
    - 板块相关性与轮动指标
    - 交互式下钻图表
    - 可操作的买入/卖出/持有信号及仓位建议
    
    **应用场景：** 识别领涨或落后板块，
    相应配置资金。
    """)

st.markdown("---")

# Feature 2: Single Stock Analysis
st.markdown("### 2️⃣ Single Stock Analysis | 个股分析")

col2_en, col2_cn = st.columns(2)

with col2_en:
    st.markdown("""
    **📈 Advanced technical analysis with 3-phase trading system**
    - **Phase 1 - Accumulation:** OBV divergence detection
    - **Phase 2 - Squeeze:** Bollinger Band contraction
    - **Phase 3 - Golden Launch:** Breakout confirmation with ADX
    - **Trading Block Theory:** Volume-based support/resistance zones
    - Statistical forecasting (Linear, ARIMA, Holt-Winters)
    - Multi-panel charts with MACD, RSI, ADX, OBV
    
    **Use Case:** Deep-dive into individual stocks to time entries and exits 
    based on volume accumulation and price action.
    """)

with col2_cn:
    st.markdown("""
    **📈 三阶段交易系统的高级技术分析**
    - **阶段1 - 吸筹：** OBV背离检测
    - **阶段2 - 收窄：** 布林带收缩
    - **阶段3 - 黄金启动：** ADX确认突破
    - **交易箱体理论：** 基于成交量的支撑/阻力区域
    - 统计预测（线性、ARIMA、Holt-Winters）
    - 多面板图表（MACD、RSI、ADX、OBV）
    
    **应用场景：** 深入分析个股，基于成交量吸筹和价格走势
    把握买卖时机。
    """)

st.markdown("---")

# Feature 3: Portfolio Optimization
st.markdown("### 3️⃣ Portfolio Optimization | 投资组合优化")

col3_en, col3_cn = st.columns(2)

with col3_en:
    st.markdown("""
    **💼 Modern Portfolio Theory (MPT) implementation**
    - Mean-variance optimization
    - Efficient frontier calculation
    - Maximum Sharpe ratio portfolio
    - Customizable constraints (max allocation per stock)
    - Risk-return analysis
    - Correlation heatmap
    - Support for all A-share exchanges (SH/SZ/BJ)
    
    **Use Case:** Build diversified portfolios that maximize risk-adjusted returns 
    based on historical data and your risk preferences.
    """)

with col3_cn:
    st.markdown("""
    **💼 现代投资组合理论（MPT）实现**
    - 均值-方差优化
    - 有效前沿计算
    - 最大夏普比率组合
    - 可定制约束（单股最大配置比例）
    - 风险收益分析
    - 相关性热力图
    - 支持所有A股交易所（沪/深/北）
    
    **应用场景：** 构建多元化投资组合，基于历史数据和风险偏好
    最大化风险调整收益。
    """)

st.markdown("---")

# ==========================================
# DISCLAIMER / 免责声明
# ==========================================

st.markdown("## ⚠️ Disclaimer | 免责声明")

col_dis_en, col_dis_cn = st.columns(2)

with col_dis_en:
    st.warning("""
    **IMPORTANT LEGAL NOTICE**
    
    This software is provided for **informational and educational purposes only**. 
    It is NOT financial advice, and should NOT be considered as a recommendation to buy, 
    sell, or hold any securities.
    
    - **No Warranty:** The information is provided "as is" without warranty of any kind.
    - **Risk Warning:** Trading stocks involves substantial risk of loss. Past performance 
      does not guarantee future results.
    - **Your Responsibility:** You are solely responsible for your investment decisions. 
      Always conduct your own research and consult with qualified financial advisors.
    - **No Liability:** The creator accepts no liability for any financial losses incurred 
      from using this software.
    
    By using this platform, you acknowledge and accept these terms.
    """)

with col_dis_cn:
    st.warning("""
    **重要法律声明**
    
    本软件仅用于**信息和教育目的**。
    它不是财务建议，不应被视为买入、卖出或持有任何证券的推荐。
    
    - **无担保：** 信息按"原样"提供，不提供任何形式的担保。
    - **风险警示：** 股票交易涉及重大损失风险。过往表现不代表未来结果。
    - **您的责任：** 您对自己的投资决策负全部责任。
      请务必进行独立研究并咨询合格的财务顾问。
    - **免责：** 创建者对使用本软件造成的任何财务损失不承担责任。
    
    使用本平台即表示您承认并接受这些条款。
    """)

st.markdown("---")

# ==========================================
# NAVIGATION / 导航
# ==========================================

st.markdown("## 🚀 Get Started | 开始使用")

st.info("""
👈 **Use the sidebar to navigate between pages**  
请使用左侧边栏在不同页面间导航

- **📊 Sector Analysis** - View sector rotation signals and market overview  
  **板块分析** - 查看板块轮动信号和市场概览

- **📈 Single Stock Analysis** - Analyze individual stocks with technical indicators  
  **个股分析** - 使用技术指标分析个股

- **💼 Portfolio Optimization** - Build optimized portfolios using MPT  
  **投资组合优化** - 使用现代投资组合理论构建优化组合
""")

st.markdown("---")

# ==========================================
# CREDITS / 作者信息
# ==========================================

st.markdown("## 👨‍💻 About the Author | 关于作者")

col_credit1, col_credit2, col_credit3 = st.columns([1, 2, 1])

with col_credit2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <h3>Phil Wei | 魏先生</h3>
        <p style="font-size: 18px;">
            📧 <a href="mailto:phil.wei@outlook.com">phil.wei@outlook.com</a>
        </p>
        <p style="color: #6c757d;">
            Quantitative Trader & Developer<br>
            量化交易者与开发者
        </p>
        <p style="font-size: 14px; color: #6c757d; margin-top: 20px;">
            Built with ❤️ using Python, Streamlit, and Tushare<br>
            使用 Python、Streamlit 和 Tushare 构建
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">
    ASSRS V2 Enhanced © 2026 Phil Wei. All rights reserved.<br>
    For educational and research purposes only. Not financial advice.<br><br>
    高级股票轮动与选股系统增强版 © 2026 魏先生。保留所有权利。<br>
    仅供教育和研究目的。不构成财务建议。
</div>
""", unsafe_allow_html=True)
