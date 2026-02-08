# Add this after the language selector


INTERACTION_LAB = {
    "English": {
        "title": "ℹ️ How does this work?",
        "content": """
### 📊 State-Based Prediction Logic

This system predicts **tomorrow's probability of outperformance** based on **today's market state**.

---

#### 🎯 Step 1: Define States

Each sector can be in one of **9 possible states** based on two dimensions:

**Excess Return Z-Score** (relative performance vs. market):
- **L (Low)**: Underperforming (z-score < -0.5)
- **M (Medium)**: In-line (-0.5 ≤ z ≤ 0.5)  
- **H (High)**: Outperforming (z > 0.5)

**Volume Z-Score** (trading activity):
- **L (Low)**: Below-average volume (z < -0.5)
- **M (Medium)**: Normal volume (-0.5 ≤ z ≤ 0.5)
- **H (High)**: Above-average volume (z > 0.5)

**9 States:** `L|L`, `L|M`, `L|H`, `M|L`, `M|M`, `M|H`, `H|L`, `H|M`, `H|H`

---

#### 🧮 Step 2: Calculate Z-Scores

Z-scores normalize performance to identify **unusual** conditions:

- Excess Return Z = (Today's Excess Return - 20-day Mean) / 20-day Std Dev
- Volume Z = (Today's Volume - 20-day Mean) / 20-day Std Dev


**Example:**
- Tech sector: +2.0% (Market: +0.5%) → Excess Return = +1.5%
- If 20-day mean excess = +0.3%, std = 0.5%
- Z-score = (1.5% - 0.3%) / 0.5% = **+2.4** → **High (H)**
- Volume today is 2x normal → Z-score = **+2.0** → **High (H)**
- **Current State: H|H** (outperforming with high volume)

---

#### 📈 Step 3: Learn from History

For each state, we track:
- How many times it occurred
- How often the next day outperformed
- Average next-day excess return

**Historical Data (60-day lookback):**

| State | Occurrences | Next-Day Wins | Win Rate | Avg Return |
|-------|------------|---------------|----------|------------|
| H\|H  | 18         | 12            | 66.7%    | +0.52%     |
| L\|H  | 15         | 9             | 60.0%    | +0.35%     |
| M\|M  | 22         | 11            | 50.0%    | +0.02%     |

---

#### 💡 Step 4: Make Predictions

**If today's state is H|H:**
- **P(Next-Day Outperform) = 66.7%** (12 wins / 18 occurrences)
- **E(Next-Day Excess Return) = +0.52%**
- **Sample Size: 18** (reliability indicator)

---

#### 🎲 What Each State Means

| State | Typical Behavior | Strategy |
|-------|-----------------|----------|
| **H\|H** | Strong momentum + volume confirmation | **Buy** - continuation likely |
| **H\|L** | Outperforming but weak volume | **Caution** - reversal risk |
| **L\|H** | Underperforming with high volume | **Reversal** - capitulation? |
| **L\|L** | Weak performance, ignored | **Avoid** - no interest |
| **M\|M** | Neutral, in-line | **Neutral** - coin flip |

---

#### ⚠️ Important Notes

- **Not absolute direction**: Predicts **relative** outperformance vs. market
- **Sample size matters**: Ignore states with <10 samples (unreliable)
- **Historical patterns**: Past performance ≠ future results
- **66% win rate**: Still means 1 in 3 times you're wrong!

---

#### 🔬 Why This Works

1. **Mean Reversion**: Extreme states often revert to average
2. **Momentum**: H|H states can continue if volume confirms
3. **Volume Confirmation**: High volume = institutional conviction (more reliable)
4. **Market Microstructure**: L|H (selling exhaustion) may bounce
"""
    },
    "中文": {
        "title": "ℹ️ 这个怎么运作的？",
        "content": """
### 📊 基于状态的预测逻辑

该系统根据**今天的市场状态**预测**明天的跑赢概率**。

---

#### 🎯 步骤1：定义状态

每个板块可以处于基于两个维度的**9种可能状态**之一：

**超额收益率Z值**（相对市场的表现）：
- **L（低）**：跑输市场（z值 < -0.5）
- **M（中）**：与市场持平（-0.5 ≤ z ≤ 0.5）
- **H（高）**：跑赢市场（z > 0.5）

**成交量Z值**（交易活跃度）：
- **L（低）**：低于平均成交量（z < -0.5）
- **M（中）**：正常成交量（-0.5 ≤ z ≤ 0.5）
- **H（高）**：高于平均成交量（z > 0.5）

**9种状态：** `L|L`, `L|M`, `L|H`, `M|L`, `M|M`, `M|H`, `H|L`, `H|M`, `H|H`

---

#### 🧮 步骤2：计算Z值

Z值标准化表现以识别**异常**状况：

超额收益率Z = (今日超额收益率 - 20日均值) / 20日标准差
成交量Z = (今日成交量 - 20日均值) / 20日标准差


**示例：**
- 科技板块：+2.0%（市场：+0.5%）→ 超额收益率 = +1.5%
- 如果20日均值超额 = +0.3%，标准差 = 0.5%
- Z值 = (1.5% - 0.3%) / 0.5% = **+2.4** → **高（H）**
- 今日成交量是正常的2倍 → Z值 = **+2.0** → **高（H）**
- **当前状态：H|H**（高超额收益率 + 高成交量）

---

#### 📈 步骤3：从历史中学习

对于每种状态，我们追踪：
- 发生次数
- 第二天跑赢的次数
- 第二天的平均超额收益率

**历史数据（60日回看）：**

| 状态 | 出现次数 | 次日跑赢次数 | 胜率 | 平均收益 |
|------|---------|------------|------|---------|
| H\|H | 18      | 12         | 66.7% | +0.52% |
| L\|H | 15      | 9          | 60.0% | +0.35% |
| M\|M | 22      | 11         | 50.0% | +0.02% |

---

#### 💡 步骤4：做出预测

**如果今天的状态是H|H：**
- **P(次日跑赢) = 66.7%**（12次胜利 / 18次出现）
- **E(次日超额收益率) = +0.52%**
- **样本数：18**（可靠性指标）

---

#### 🎲 每种状态的含义

| 状态 | 典型行为 | 策略 |
|------|---------|------|
| **H\|H** | 强势动能 + 成交量确认 | **买入** - 可能延续 |
| **H\|L** | 跑赢但成交量弱 | **谨慎** - 反转风险 |
| **L\|H** | 跑输但成交量高 | **反转** - 恐慌抛售？ |
| **L\|L** | 表现弱，无人关注 | **规避** - 缺乏兴趣 |
| **M\|M** | 中性，持平 | **中性** - 随机 |

---

#### ⚠️ 重要提示

- **非绝对方向**：预测的是**相对**市场的跑赢概率
- **样本量很重要**：忽略样本数<10的状态（不可靠）
- **历史模式**：过去表现 ≠ 未来结果
- **66%胜率**：仍意味着3次中有1次会错！

---

#### 🔬 为什么这有效

1. **均值回归**：极端状态通常会回归平均水平
2. **动量**：如果成交量确认，H|H状态可能延续
3. **成交量确认**：高成交量 = 机构信念（更可靠）
4. **市场微观结构**：L|H（抛售枯竭）可能反弹
"""
    }
}
