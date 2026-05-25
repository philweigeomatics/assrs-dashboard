# ==========================================
# HEADER
# ==========================================
import streamlit as st

st.markdown("### Advanced Stock Rotation & Selection System")
st.markdown("### 高级股票轮动与选股系统")
st.caption(
    "An A-share research platform that fuses sector regime models, AI-narrative "
    "equity briefs, and rule-based trading strategies into one workspace."
)

st.markdown("---")

# ==========================================
# ABOUT / 关于
# ==========================================

col_en, col_cn = st.columns(2)

with col_en:
    st.markdown("## 📖 About")
    st.markdown("""
**ASSRS** is a quant-research platform for the Chinese A-share market. It pulls
live OHLCV, fundamentals, margin balances and 龙虎榜 (top-list) data from
Tushare, runs a market-regime model on aggregated sector PPIs, generates
AI-narrative equity briefs via DeepSeek, and ships ready-to-use strategy
scanners over your personal watchlist.

It's designed for traders who want **data-driven evidence** behind every
decision — sector rotation, individual entry/exit timing, peer comparison,
portfolio construction, and same-day做T trading.

The platform is **invitation-only**. Logins persist for 30 days via a
server-validated cookie, so you can open any page in a new tab and pick up
exactly where you left off.
    """)

with col_cn:
    st.markdown("## 📖 关于")
    st.markdown("""
**ASSRS** 是一个面向中国A股市场的量化研究平台。它通过 Tushare 实时获取
行情、基本面、融资融券和龙虎榜数据，对板块 PPI 运行市场状态模型，
通过 DeepSeek 生成 AI 个股研报，并提供基于自选股的策略筛选工具。

平台面向希望以**数据驱动**做决策的交易者 —— 涵盖板块轮动、个股
买卖时机、可比公司分析、组合构建以及当日做T 策略。

平台**仅限受邀用户使用**。登录状态通过服务端校验的 cookie 保持 30 天，
新标签页打开任意页面均可继续使用。
    """)

st.markdown("---")

# ==========================================
# WHAT'S INSIDE / 平台功能概览
# ==========================================

st.markdown("## 🗂 What's inside · 平台功能概览")
st.caption(
    "Six navigation groups in the sidebar. Each is summarised below. "
    "侧边栏分六大模块，下方为各模块说明。"
)

# ── Sector ───────────────────────────────────────────────────────────────────
st.markdown("### 📊 Sector 板块")

s_en, s_cn = st.columns(2)
with s_en:
    st.markdown("""
- **Dashboard** — Per-sector PPI charts with ML-classified market regimes,
  market-breadth proxy, CSI 300 volatility cone, plus a clickable 龙虎榜
  (institutional-flow) table that jumps straight into Lead-Lag analysis
  for the named stock.
- **Interaction Lab** — Cross-sector correlation explorer with adjustable
  lookback windows.
- **Performance Rotation** — Rolling sector-return heatmaps showing which
  themes lead and which lag, day by day.
    """)
with s_cn:
    st.markdown("""
- **板块仪表盘** —— 各行业 PPI 走势 + 机器学习市场状态分类、市场广度、
  沪深300 波动率锥形图，以及可点击的龙虎榜表格（点击个股直接进入领先
  滞后分析）。
- **互动实验室** —— 跨板块相关性分析，可调回溯窗口。
- **轮动分析** —— 滚动收益热力图，展示哪些行业主题领涨、哪些落后。
    """)

# ── Stock ────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Stock 股票")

st_en, st_cn = st.columns(2)
with st_en:
    st.markdown("""
- **Equity Report** — One-page AI-generated brief per stock: company
  overview, PESTEL, SWOT, Porter's Five Forces, validated peer set with
  cross-checked fundamentals, supply-chain knowledge graph, revenue
  segments, and earnings-forecast / express cards. Peer names are
  clickable anchors — click any to regenerate the brief.
- **Technical Analysis** — 7-panel chart (price / volume / MACD / RSI /
  ADX / P/E / Z-score oscillator) with direction-gated trend markers
  and Entry/Exit candidate triangles on the price pane. Adjustable
  comparison stock overlay (same-% or new-price-scale axis).
- **Stock Selector** — Filter stocks within a sector by price, volume,
  fundamental metrics.
- **Watchlist** — Personal watchlist with live PE/PB/turnover; companion
  **Earnings Calendar** view for upcoming filings.
- **Sector Explorer** — Industry-chain Sankey diagrams showing layer
  positions of stocks and supply-chain relationships.
    """)
with st_cn:
    st.markdown("""
- **个股研报** —— AI 一页式深度研报：公司概览、PESTEL、SWOT、波特五力、
  经数据库交叉验证的可比公司表、供应链知识图谱、收入构成、业绩预告/快报
  卡片。可比公司名称可点击，一键生成对应公司的研报。
- **技术分析** —— 7 面板技术图表（价格/成交量/MACD/RSI/ADX/市盈率/Z 分数
  振荡器），价格面板叠加方向门控的趋势标记和买入/卖出候选三角。可选叠加
  对照股票（同比例 / 独立 Y 轴）。
- **选股器** —— 在指定板块内按价格、成交量、基本面指标筛选。
- **自选股** —— 实时显示 PE/PB/换手率；配套**财报日历**视图。
- **产业链浏览** —— 行业链 Sankey 图，展示各股票在产业链中的层级位置和
  上下游关系。
    """)

# ── Portfolio ────────────────────────────────────────────────────────────────
st.markdown("### 💼 Portfolio 组合")

p_en, p_cn = st.columns(2)
with p_en:
    st.markdown("""
- **Optimization** — Mean-variance portfolio construction (MPT): efficient
  frontier, max-Sharpe portfolio, customisable single-name caps,
  correlation heatmap. Supports SH / SZ / BJ.
- **Portfolio Management** — Save fund mandates with locked weights;
  nightly NAV roll-forward computes daily returns, AUM, P&L, allocation
  drift. Manual force-rollup button if the cron didn't run.
- **AI Supply Chain** — Build a custom supply-chain network for any A-share
  company via DeepSeek; explore upstream / downstream linkages
  interactively.
    """)
with p_cn:
    st.markdown("""
- **组合优化** —— 基于现代投资组合理论的均值-方差优化：有效前沿、最大
  夏普组合、单股仓位上限可调、相关性热力图。支持沪/深/北。
- **组合管理** —— 保存基金任务书及锁定权重；每晚 NAV 滚动计算日收益、
  规模、盈亏、配置漂移。若 cron 未运行可手动触发。
- **AI 供应链** —— 通过 DeepSeek 为任意 A 股公司生成定制化的供应链
  网络图，交互探索上下游关联。
    """)

# ── Strategy ─────────────────────────────────────────────────────────────────
st.markdown("### 🌊 Strategy 策略")

strat_en, strat_cn = st.columns(2)
with strat_en:
    st.markdown("""
- **Wave Trader** — Multi-timeframe swing-trade decision support.
- **Pair Trader** — Walk-forward statistical pair trading with OOS backtest.
- **Lead-Lag Analysis** — 4-phase pipeline: pick a sector → AI-expand into
  3–5 supply-chain layers → discover up to 10 key stocks per layer →
  run Granger / cross-correlation / cointegration tests on each peer
  vs. the target stock. Dedupes vertically-integrated names automatically.
- **Mean Reversion 反转候选** — Scan your watchlist for retail-panic
  oversold snapback candidates. 5 rules (Z-score, consecutive down days,
  volume exhaustion, RSI, sector divergence) + automatic ST / *ST
  rejection.
- **T-Trading Scanner 做T候选** — Rank your watchlist by structural
  suitability for intraday做T (5-component composite score: intraday
  range, turnover, mean-reversion bias, ADX regime, range position).
  Click any row for a per-stock plan with trade-zone visualisation.
  Results persist in the database with an age badge.
    """)
with strat_cn:
    st.markdown("""
- **波段交易** —— 多时间周期波段交易决策辅助。
- **配对交易** —— Walk-forward 统计套利配对交易，含样本外回测。
- **领先滞后分析** —— 4 阶段流程：选定板块 → AI 展开 3–5 层供应链 →
  每层发掘最多 10 只关键个股 → 对每只个股与目标股票运行格兰杰因果、
  互相关、协整检验。自动去重纵向一体化公司。
- **反转候选** —— 扫描自选股，发现因散户恐慌被错杀、有反弹机会的标的。
  5 条规则（Z 分数、连跌天数、量能枯竭、RSI、板块背离）+ 自动剔除
  ST/*ST。
- **做T候选** —— 按结构性"做T适合度"对自选股排序（综合评分包含日内
  振幅、换手率、均值回归倾向、ADX 状态、相对区间位置 5 项）。点击任意
  行展开单股做T方案与交易区间可视化。结果持久化保存并显示数据新旧。
    """)

# ── Alerts ───────────────────────────────────────────────────────────────────
st.markdown("### 🔔 Alerts 提示")

a_en, a_cn = st.columns(2)
with a_en:
    st.markdown("""
- **Today's Alerts** — Daily summary of signal triggers across your
  watchlist: technical breakouts, oversold reversals, fundamental
  releases.
    """)
with a_cn:
    st.markdown("""
- **今日提醒** —— 自选股的每日信号汇总：技术突破、超卖反转、基本面
  发布等触发事件。
    """)

st.markdown("---")

# ==========================================
# UNDER THE HOOD / 技术架构
# ==========================================

st.markdown("## 🛠 Under the hood · 技术架构")

tech_en, tech_cn = st.columns(2)
with tech_en:
    st.markdown("""
- **Data**: Tushare API — A-share OHLCV (qfq-adjusted), `daily_basic`
  fundamentals, margin balances (融资融券), `stk_limit` limit-up/down
  prices, `top_list` (龙虎榜).
- **AI narrative**: DeepSeek reasoning model for company overviews,
  PESTEL, SWOT, Porter's, competitor discovery, supply-chain graph
  generation, sector theme expansion, layer-stock identification.
- **Nightly pipeline** (cron, 20:00 Beijing): sector-PPI aggregation,
  market-breadth calculation, institutional-portfolio NAV roll-forward.
  Self-correcting date logic so manual reruns at any time of day target
  the most recent close.
- **Auth**: Invitation-only, 30-day persistent sessions via
  server-validated browser cookies.
- **Convention**: A-share colour scheme — red = up, green = down (the
  opposite of Western finance).
    """)
with tech_cn:
    st.markdown("""
- **数据**：Tushare API —— A 股行情（前复权）、`daily_basic` 基本面、
  融资融券余额、`stk_limit` 涨跌停价、`top_list` 龙虎榜。
- **AI 叙述**：DeepSeek 推理模型用于公司概览、PESTEL、SWOT、波特五力、
  可比公司发掘、供应链图谱生成、行业主题展开、层级核心标的识别。
- **每日更新**（cron，北京时间 20:00）：板块 PPI 聚合、市场广度计算、
  机构组合 NAV 滚动。日期逻辑自我校正，任意时段手动触发都能定位到
  最近一个交易日。
- **认证**：仅限受邀注册，30 天持久会话，通过服务端校验的浏览器
  cookie 维持。
- **颜色约定**：遵循 A 股惯例 —— 红涨绿跌（与西方金融市场相反）。
    """)

st.markdown("---")

# ==========================================
# GET STARTED / 快速上手
# ==========================================

st.markdown("## 🚀 Get started · 快速上手")

gs_en, gs_cn = st.columns(2)
with gs_en:
    st.markdown("""
**Suggested first-visit flow:**

1. Open **📊 Sector → Dashboard** to see the current market regime and
   today's 龙虎榜 institutional flow.
2. Open **📈 Stock → Equity Report**, type a 6-digit ticker or company
   name, generate the brief.
3. Open **📈 Stock → Watchlist** and add the stocks you want to track.
4. Run a strategy scan over your watchlist:
   **🌊 Strategy → T-Trading Scanner** or **Mean Reversion**.
5. For any candidate, click into the per-stock plan to see the
   structural diagnostic and trade-zone visualisation.
    """)
with gs_cn:
    st.markdown("""
**建议首次使用流程：**

1. 打开 **📊 板块 → 仪表盘**，查看当前市场状态和今日龙虎榜机构资金
   动向。
2. 打开 **📈 股票 → 个股研报**，输入 6 位代码或公司名，生成研报。
3. 打开 **📈 股票 → 自选股**，添加需要追踪的标的。
4. 在自选股上运行策略扫描：
   **🌊 策略 → 做T候选** 或 **反转候选**。
5. 对感兴趣的候选股，点击进入单股方案，查看结构性诊断和交易区间
   可视化。
    """)

st.markdown("---")

# ==========================================
# DISCLAIMER / 免责声明
# ==========================================

st.markdown("## ⚠️ Disclaimer · 免责声明")

col_dis_en, col_dis_cn = st.columns(2)

with col_dis_en:
    st.warning("""
**IMPORTANT LEGAL NOTICE**

This software is provided for **informational and educational purposes only**.
It is NOT financial advice, and should NOT be considered as a recommendation to
buy, sell, or hold any securities.

- **No warranty** — Information provided "as is" without warranty of any kind.
- **Risk warning** — Trading stocks involves substantial risk of loss. Past
  performance does not guarantee future results.
- **Your responsibility** — You are solely responsible for your investment
  decisions. Always conduct your own research and consult qualified financial
  advisors.
- **No liability** — The creator accepts no liability for any financial losses
  incurred from using this software.

By using this platform, you acknowledge and accept these terms.
    """)

with col_dis_cn:
    st.warning("""
**重要法律声明**

本软件仅用于**信息和教育目的**。
不构成财务建议，也不应被视为买入、卖出或持有任何证券的推荐。

- **无担保** —— 信息按"原样"提供，不提供任何形式的担保。
- **风险警示** —— 股票交易涉及重大损失风险。过往表现不代表未来结果。
- **您的责任** —— 您对自己的投资决策负全部责任。请务必进行独立研究
  并咨询合格的财务顾问。
- **免责** —— 创建者对使用本软件造成的任何财务损失不承担责任。

使用本平台即表示您承认并接受这些条款。
    """)

st.markdown("---")

# ==========================================
# CREDITS / 作者信息
# ==========================================

st.markdown("## 👨‍💻 About the author · 关于作者")

col_credit1, col_credit2, col_credit3 = st.columns([1, 2, 1])

with col_credit2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa;
                border-radius: 10px;">
        <h3>Phil Wei · 魏先生</h3>
        <p style="font-size: 18px;">
            📧 <a href="mailto:phil.wei@outlook.com">phil.wei@outlook.com</a>
        </p>
        <p style="color: #6c757d;">
            Quantitative Trader &amp; Developer<br>
            量化交易者与开发者
        </p>
        <p style="font-size: 14px; color: #6c757d; margin-top: 20px;">
            Built with Python · Streamlit · Tushare · DeepSeek · Supabase<br>
            使用 Python · Streamlit · Tushare · DeepSeek · Supabase 构建
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
    仅供教育和研究目的，不构成财务建议。
</div>
""", unsafe_allow_html=True)
