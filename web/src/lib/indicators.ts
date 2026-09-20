/**
 * What every number on the 配对交易 page means, in one place.
 *
 * The columns are named after statistical tests, and a column header that
 * says "协整 p" to someone who has not read a time-series textbook is a
 * number with a colour on it and nothing else. Worse, a reader who cannot
 * tell what Hurst measures cannot tell when the screen is warning them.
 *
 * One source, two renderings: the short line becomes the column's tooltip,
 * and the whole entry becomes a card in the 指标怎么读 panel. Writing the
 * tooltip separately is how the two end up disagreeing.
 *
 * Thresholds are NOT written here. They come from the `gates` the API ships
 * out of the module that applies them (pair_trade.GATES), because an
 * explanation that restates a threshold defined elsewhere is one edit away
 * from confidently describing a rule that is no longer the rule.
 */

import type { PairGates } from "./types";

export type Indicator = {
  /** Column header / field label, exactly as the table writes it. */
  label: string;
  /** One line, used as the column tooltip. */
  short: string;
  /** What the number actually measures. */
  what: string;
  /** The threshold, when there is one. */
  pass?: string;
  /** How to read it once you have it. */
  reads: string;
  /**
   * Which stretch of data this number is computed over.
   *
   * Three different windows are in play and the page never said so, which
   * invites the reasonable assumption that everything uses the z window. The
   * four gates use the WHOLE out-of-sample history; only z uses the rolling
   * 60; β uses a rolling 252.
   */
  window?: string;
  /** What it does not tell you. The most useful line of the three. */
  caveat?: string;
};

const nd = (v: number, d = 2) => v.toFixed(d);

/**
 * Used only when the response has no `gates` — i.e. the deployed API is
 * older than this page, which happens for a few minutes whenever the
 * frontend rolls out before the API does.
 *
 * Reading a field off `undefined` is a TypeError, and a TypeError during
 * render unmounts the tree: the whole page goes white because the help text
 * could not name a threshold. Slightly stale numbers are the better failure.
 *
 * test_pair_trade.py keeps these equal to pair_trade.GATES, so they cannot
 * quietly drift into being wrong for the normal case.
 */
export const DEFAULT_GATES: PairGates = {
  coint_p: 0.1,
  adf_p: 0.1,
  hurst_max: 0.45,
  hl_min: 5,
  hl_max: 30,
  entry_z: 2,
  watch_z: 1.5,
  good_score: 7,
  max_score: 11,
  z_window: 60,
  ols_window: 252,
};

/** The pair table's columns, in the order the table shows them. */
export function pairIndicators(gates?: PairGates): Indicator[] {
  const g = gates ?? DEFAULT_GATES;
  return [
    {
      label: "评分",
      short: `把下面几项压成一个数，满分约 ${g.max_score}；${g.good_score} 分以上值得看`,
      what: `下面几项检验的加总，满分约 ${g.max_score} 分。`,
      pass: `≥ ${nd(g.good_score, 1)} 值得细看`,
      reads: "只是一个排序用的数，方便把几十个组合排出先后。",
      caveat: "它不是概率，也不是预期收益。高分只说明几项检验都通过了。",
    },
    {
      label: "协整 p",
      short: `Engle-Granger 协整检验 p 值，< ${nd(g.coint_p)} 通过`,
      what: "两只股票之间有没有长期稳定的价格关系。",
      window: "整段样本外历史",
      pass: `< ${nd(g.coint_p)}`,
      reads: "注意它问的不是「是不是一起涨跌」，而是「走散之后会不会被拉回来」。"
        + "p 越小，「它们只是恰好都在往上漂」这个解释越站不住。",
      caveat: "它说的是这段历史里成立，不是这段关系会继续存在。基本面变了，统计量不会提前知道。",
    },
    {
      label: "ADF p",
      short: `价差本身的平稳性检验 p 值，< ${nd(g.adf_p)} 通过`,
      what: "价差这条线自己会不会回归 —— 拉开之后回不回得来。",
      pass: `< ${nd(g.adf_p)}`,
      reads: "和协整检验是两个问题：协整问「两者有没有关系」，ADF 问「价差会不会回来」。"
        + "两个都通过，才算一对可以交易的组合。",
      window: "整段样本外历史",
      caveat: "算在样本外的价差上（对冲比率没看过当天），所以不是拟合出来的好看结果。",
    },
    {
      label: "Hurst",
      short: `< ${nd(g.hurst_max)} 为均值回归，0.5 是随机游走，高于 0.55 是趋势`,
      what: "价差是回归型的还是趋势型的。0.5 相当于抛硬币。",
      window: "整段样本外历史",
      pass: `< ${nd(g.hurst_max)}`,
      reads: `低于 ${nd(g.hurst_max)}：拉开了会回来。高于 0.55：拉开了更可能继续拉开 —— `
        + "这时候「偏离两个标准差」不是入场理由，是反过来的理由。这一栏没通过，"
        + "其余几项通过也要留神。",
      caveat: "短样本上这个估计量偏高，所以 0.5 附近不必太当真；0.8 就要当真了。",
    },
    {
      label: "半衰期",
      short: `价差消掉一半偏离要几个交易日，${g.hl_min}–${g.hl_max} 天最可交易`,
      what: "价差回到一半所需的交易日数（Ornstein-Uhlenbeck 回归）。",
      window: "整段样本外历史",
      pass: `${g.hl_min}–${g.hl_max} 天`,
      reads: `低于 ${g.hl_min} 天，行情还没等你下单就走完了；超过 ${g.hl_max} 天，`
        + "一笔仓位要压住一个季度等它回来。显示「不收敛」表示回归速度估不出来。",
      caveat: "这是历史上的平均速度，不是这一次的承诺。",
    },
    {
      label: "相关性",
      short: "两只股票每日收益率的相关系数 —— 是配对的前提，不是配对的理由",
      what: "两只股票日收益率的相关系数。",
      reads: "高相关是能配对的前提。但同涨同跌不等于价差会回归 —— 那是协整和 ADF 回答的问题。",
      caveat: "整个大盘一起动的时候，任何两只 A 股都相关。跨行业的高相关，"
        + "通常说的是大盘，不是这两只股票之间的关系。",
    },
    {
      label: "Z",
      short: `当前价差偏离最近 ${g.z_window} 天均值几个标准差，|Z| ≥ ${nd(g.entry_z, 1)} 触发`,
      what: `价差偏离它自己最近 ${g.z_window} 天均值的标准差倍数。`,
      window: `最近 ${g.z_window} 天（滚动）`,
      pass: `|Z| ≥ ${nd(g.entry_z, 1)} 入场，`
        + `|Z| ≥ ${nd(g.watch_z, 1)} 接近信号，回到 0 出场`,
      reads: "负数表示 A 相对 B 便宜（买 A、减持 B），正数反过来。"
        + "它衡量的是两者之间的差距，和两只股票本身是涨是跌无关。",
      caveat: `基准是滚动的。价差自己没回来、而最近 ${g.z_window} 天的均值挪上去追上了它，`
        + "Z 一样会回到 0。所以 Z 归零不等于价差收敛 —— 下面那张两条腿的走势图才看得出是哪一种。",
    },
    {
      label: "对冲比率 β",
      short: `每 1 份 A 对应多少份 B，由前 ${g.ols_window} 天滚动回归估计并前推一天`,
      what: `价差 = log(A) − β·log(B)。β 由之前 ${g.ols_window} 天的滚动回归估计。`,
      window: `之前 ${g.ols_window} 天（滚动，前推一天）`,
      reads: "前推一天是关键：当天用的 β 只看过当天之前的数据，"
        + "所以下面所有统计量和历史交易都是样本外的，而不是对过去的描述。",
      caveat: "β 每天都在变。表头显示的是最新一天的值，不一定是某笔历史交易当时用的那个。",
    },
    {
      label: "历史",
      short: "样本外历史交易的胜率与平均盈亏",
      what: "按同一套规则在历史上走一遍的结果：胜率 · 平均盈亏。",
      reads: "因为 β 是前推的，这些交易是当时真的能发出的信号，不是事后才看得见的。",
      caveat: "A 股不能做空，盈亏只算买入的那条腿。减持另一条腿是仓位调整，不是空头，"
        + "所以这里没有配对交易本该有的对冲。笔数通常只有几笔，胜率当不得统计量。",
    },
  ];
}

/** 从自选股中搜索 — the discovery funnel's own columns. */
export const DISCOVER_INDICATORS: Indicator[] = [
  {
    label: "相关（粗筛）",
    short: "先按相关性把组合数缩小到能逐个做检验的量。这不是显著性检验。",
    what: "候选阶段的筛子：把 N² 个组合缩到能逐个跑统计检验的数量。",
    reads: "调低它会让更多组合进入检验，不会让结论更宽松 —— 后面的检验照做。",
    caveat: "它本身不是证据。相关性高只说明值得一测。",
  },
  {
    label: "p 前",
    short: "前半程的 p 值 —— 组合就是靠它挑出来的，所以它不是证据",
    what: "把历史对半切开，前半程算出来的 p 值。",
    reads: "组合是靠这个数挑出来的。挑出来的东西在挑选它的那份数据上显著，是必然的。",
    caveat: "只看这一栏，等于用同一份数据既选又证。它在这里是为了让你看见筛选发生过。",
  },
  {
    label: "p 后",
    short: "后半程的 p 值 —— 这半段没参与挑选，所以它才是证据",
    what: "后半程算出来的 p 值。这半段数据没有参与挑选。",
    reads: "前半程挑、后半程验。只有这一栏说明关系在选它的数据之外也还在。",
    caveat: "单独一个 p 后还不够 —— 检验了几千个组合，光靠运气也会有一批显著的。看 q。",
  },
  {
    label: "q",
    short: "p 后经 Benjamini-Hochberg 多重检验校正后的值 —— 结论看这一栏",
    what: "后半程 p 值经 Benjamini-Hochberg 校正后的 q 值。",
    reads: "检验三千个组合、每个用 5% 的门槛，平均会有一百多个纯属巧合地「显著」。"
      + "q 把这件事算了进去，所以它才是结论。",
    caveat: "校正之后一个都不剩是常见结果，也是诚实的结果 —— 那说明这批自选股里"
      + "确实没有站得住的配对，不是搜索失败。",
  },
  {
    label: "一致",
    short: "两半必须指向同一只领先。方向相反的不算通过 —— 那是噪声，不是弱关系。",
    what: "前后两半对「谁领先谁」的判断是否一致。这是通过的必要条件之一。",
    reads: "✗ 表示两半给出相反的方向，这种组合一律不算通过。"
      + "纯噪声有一半的概率两次都显著还刚好方向一致，所以「纯靠运气」的预期也按一半算。",
    caveat: "在真实自选股上量过：清掉多重检验之后仍有接近一半的组合方向相反 —— "
      + "而纯噪声本来就有一半会相反。不把这一条算进通过条件，这张表基本就是在报巧合。",
  },
];
