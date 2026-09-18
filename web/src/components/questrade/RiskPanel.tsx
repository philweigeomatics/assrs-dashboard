/**
 * 风险分析 — the book against an index, both measured in the base currency.
 *
 * Two things this panel refuses to let you misread.
 *
 * It is a REPLAY. Today's weights applied backwards over three years: what the
 * book you hold now would have done, not what your account actually did, which
 * would need the trade history. Stated on the panel, not only in a docstring.
 *
 * And weight is not risk. The right-hand table puts each holding's share of
 * the portfolio next to its share of the portfolio's VOLATILITY — the marginal
 * contribution w·cov(r, r_p)/σ_p, which sums exactly to 100%. A 4% position in
 * something wild routinely carries more risk than a 20% position in a utility,
 * and a column of individual volatilities will never show that.
 *
 * Everything is in the base currency, benchmark included. A CAD book regressed
 * on a USD S&P 500 produces a beta that is part equity exposure and part
 * exchange rate, and no label on it would make that number mean anything.
 */

import type { QtRisk, QtScope, Num } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const NA = false;   // North America: green up, red down

function tone(v: Num, goodIsHigh = true): string {
  if (v == null) return "text-flat";
  const good = goodIsHigh ? v > 0 : v < 0;
  return good === NA ? "text-up" : "text-down";
}

const SCOPES: { id: QtScope; label: string }[] = [
  { id: "all", label: "全部" },
  { id: "stock", label: "仅股票" },
  { id: "etf", label: "仅 ETF" },
];

export function RiskPanel({ risk, benchmarks, benchmark, onBenchmark,
                            scope, onScope }: {
  risk: QtRisk; benchmarks: { id: string; name: string }[];
  benchmark: string; onBenchmark: (b: string) => void;
  scope: QtScope; onScope: (s: QtScope) => void;
}) {
  const p = risk;
  const b = risk.benchmark_stats;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="label">
          {p.from} → {p.to} · {p.sessions} 个交易日 · 全部以 {p.base} 计价
        </span>
        <div className="ml-auto flex items-center gap-2">
          <div className="flex rounded-lg bg-sunken p-0.5"
            title="ETF 会把组合的 beta 拉向 1。想看自己选股带来的风险，就只看股票那一段。">
            {SCOPES.map((o) => (
              <button key={o.id} onClick={() => onScope(o.id)}
                className={`px-2 h-7 rounded-md text-[12.5px] font-medium ${
                  scope === o.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
                {o.label}
              </button>
            ))}
          </div>
        <div className="flex rounded-lg bg-sunken p-0.5">
          {benchmarks.map((o) => (
            <button key={o.id} onClick={() => onBenchmark(o.id)}
              className={`px-2.5 h-7 rounded-md text-[12.5px] font-medium transition-colors ${
                benchmark === o.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
              {o.name}
            </button>
          ))}
        </div>
        </div>
      </div>

      <p className="text-[12px] text-brand-ink leading-snug rounded-lg bg-sunken px-2.5 py-2">
        ⚠ {p.basis}。用当前持仓的权重回放三年历史，回答的是「我现在这个组合过去会怎样」，
        不是账户的真实业绩（那需要成交记录）。
        {p.scope !== "all" && (
          <b> 当前只看{p.scope_label}，占整个组合的 {fixed(p.sleeve_pct, 1)}%。</b>
        )}
        {p.covered_pct < 99.5 && (
          <b> 另外，只有 {fixed(p.covered_pct, 1)}% 的市值能取得行情，
            以下所有指标只代表这一部分。</b>
        )}
      </p>

      {p.excluded?.length > 0 && (
        <p className="text-[11.5px] text-ink-mute leading-snug">
          未纳入统计：{p.excluded.map((e) => `${e.symbol}（${e.reason}）`).join("、")}
        </p>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-3">
        <div className="flex flex-col gap-2">
          <Compare label="年化收益" mine={p.ann_return_pct} theirs={b.ann_return_pct}
            suffix="%" good hint="几何年化，含分红（行情已复权）。" />
          <Compare label="年化波动" mine={p.ann_vol_pct} theirs={b.ann_vol_pct}
            suffix="%" good={false} hint="日收益标准差 × √252。越低越稳。" />
          <Compare label="最大回撤" mine={p.max_drawdown_pct} theirs={b.max_drawdown_pct}
            suffix="%" good hint="期间从高点到低点的最大跌幅。" />
          <Compare label="夏普" mine={p.sharpe} theirs={b.sharpe}
            good hint="年化收益 ÷ 年化波动，无风险利率按 0 计。" />
          <Compare label="索提诺" mine={p.sortino} theirs={b.sortino}
            good hint="只用下行波动做分母 —— 涨得猛不算风险。" />
          <Compare label="单日 VaR 95%" mine={p.var95_pct} theirs={b.var95_pct}
            suffix="%" good hint="历史分位数，不是正态近似：厚尾不会被抹平。约每 20 个交易日会有一天比它更差。" />
        </div>

        <div className="flex flex-col gap-2">
          <Stat label="Beta" value={fixed(p.beta, 2)}
            hint={`对 ${p.benchmark_name} 的敏感度。1.0 = 与指数同步；1.3 = 指数涨跌 1% 时预期涨跌 1.3%。`} />
          <Stat label="Alpha（年化）" value={`${signed(p.alpha_pct, 2)}%`}
            cls={tone(p.alpha_pct)}
            hint="扣掉 beta 解释的部分之后剩下的收益。样本期内的统计量，不是预测。" />
          <Stat label="R²" value={fixed(p.r2, 2)}
            hint="收益被指数解释的比例。接近 1 说明这个组合基本就是指数。" />
          <Stat label="跟踪误差" value={`${fixed(p.tracking_error_pct, 2)}%`}
            hint="组合与指数收益之差的年化波动。0 表示完全复制指数。" />
          <Stat label="上行 / 下行捕获"
            value={`${fixed(p.up_capture_pct, 0)}% / ${fixed(p.down_capture_pct, 0)}%`}
            hint="指数上涨日你吃到多少、下跌日你跟跌多少。理想是上高下低。" />
          <Stat label="有效持仓数"
            value={`${fixed(p.concentration.effective_n, 1)} / ${p.concentration.positions}`}
            hint={`1/Σw²。${p.concentration.positions} 只里最大一只占 ${fixed(p.concentration.top1_pct, 1)}%，前五占 ${fixed(p.concentration.top5_pct, 1)}%。`} />
        </div>
      </div>

      <div>
        <div className="flex items-baseline gap-2 mb-1">
          <span className="text-[13px] font-medium">风险来自哪里</span>
          <span className="label">权重 ≠ 风险：右侧是各持仓对组合波动的边际贡献，合计 100%</span>
        </div>
        <div className="overflow-auto rounded-lg border border-line max-h-[420px]">
          <table className="w-full border-collapse text-[12px]">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-line text-ink-mute">
                <th className="text-left font-medium px-2 py-1.5">代码</th>
                <th className="text-left font-medium px-2 py-1.5">名称</th>
                <th className="text-right font-medium px-2 py-1.5">权重</th>
                <th className="text-right font-medium px-2 py-1.5">年化波动</th>
                <th className="text-right font-medium px-2 py-1.5">Beta</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="与基准的相关系数">相关</th>
                <th className="text-right font-medium px-2 py-1.5">年化收益</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="边际风险贡献 w·cov(r, r_p)/σ_p，合计 100%">风险占比</th>
              </tr>
            </thead>
            <tbody>
              {p.holdings.map((h) => {
                const heavy = (h.risk_pct ?? 0) > h.weight_pct * 1.25;
                return (
                  <tr key={h.symbol} className="border-b border-line/60 hover:bg-sunken">
                    <td className="px-2 py-1 font-mono">{h.symbol}</td>
                    <td className="px-2 py-1 max-w-[200px] truncate" title={h.name}>{h.name}</td>
                    <td className="px-2 py-1 text-right tnum">{fixed(h.weight_pct, 1)}%</td>
                    <td className="px-2 py-1 text-right tnum">{fixed(h.ann_vol_pct, 1)}%</td>
                    <td className="px-2 py-1 text-right tnum">{fixed(h.beta, 2)}</td>
                    <td className="px-2 py-1 text-right tnum">{fixed(h.corr_bench, 2)}</td>
                    <td className={`px-2 py-1 text-right tnum ${tone(h.ann_return_pct)}`}>
                      {signed(h.ann_return_pct, 1)}%
                    </td>
                    <td className={`px-2 py-1 text-right tnum font-medium ${
                      heavy ? "text-brand-ink" : ""}`}
                      title={heavy ? "风险占比明显高于权重 —— 这只股票在替整个组合承担波动" : ""}>
                      {fixed(h.risk_pct, 1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Compare({ label, mine, theirs, suffix = "", good, hint }: {
  label: string; mine: Num; theirs: Num; suffix?: string;
  good: boolean; hint: string;
}) {
  const diff = mine != null && theirs != null ? mine - theirs : null;
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-1.5 flex items-baseline gap-2" title={hint}>
      <span className="text-[12.5px] flex-1 min-w-0 truncate">{label}</span>
      <span className="tnum text-[14px] font-semibold w-20 text-right">
        {fixed(mine, 2)}{suffix}
      </span>
      <span className="tnum text-[12px] text-ink-mute w-20 text-right"
        title="基准同期">{fixed(theirs, 2)}{suffix}</span>
      <span className={`tnum text-[11.5px] w-16 text-right ${
        diff == null ? "text-flat" : tone(good ? diff : -diff)}`}>
        {diff == null ? "—" : signed(diff, 2)}
      </span>
    </div>
  );
}

function Stat({ label, value, hint, cls = "" }: {
  label: string; value: string; hint: string; cls?: string;
}) {
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-1.5 flex items-baseline gap-2" title={hint}>
      <span className="text-[12.5px] flex-1 min-w-0 truncate">{label}</span>
      <span className={`tnum text-[14px] font-semibold ${cls}`}>{value}</span>
    </div>
  );
}
