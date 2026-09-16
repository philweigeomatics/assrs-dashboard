/**
 * 筹码分布 — where the float's cost basis sits, drawn sideways so price runs
 * down the panel and lines up with how you read the chart beside it.
 *
 * Bars are coloured by the Chinese convention: red below the current price is
 * 获利盘 (holders in profit), green above is 套牢盘 (trapped). The dashed rule
 * is the current price, the dotted one the weighted average cost.
 *
 * Plain SVG rather than a chart library: it is 200-odd horizontal bars with
 * two reference lines and no interaction beyond a tooltip, and the chart
 * bundle is already the biggest thing on the page.
 */

import type { Chips } from "../lib/types";
import { fixed } from "../lib/format";

const W = 312;
const H = 250;
const PAD_R = 46; // room for the price labels on the right

export function ChipPanel({ chips, price }: { chips: Chips | null; price: number }) {
  if (!chips || chips.prices.length === 0) {
    return (
      <section className="card p-3">
        <h2 className="text-[14px] font-semibold mb-1">🧮 筹码分布</h2>
        <p className="label">缺少换手率数据，无法计算。</p>
      </section>
    );
  }

  const lo = chips.prices[0]!;
  const hi = chips.prices[chips.prices.length - 1]!;
  const span = hi - lo || 1;
  const maxW = Math.max(...chips.weights);
  const y = (p: number) => H - ((p - lo) / span) * H;
  const barW = (w: number) => (w / maxW) * (W - PAD_R);
  const rowH = Math.max(1, H / chips.prices.length);

  const Line = ({ p, color, dash, label }: { p: number; color: string; dash: string; label: string }) => {
    if (p < lo || p > hi) return null;
    const yy = y(p);
    return (
      <g>
        <line x1={0} x2={W - PAD_R} y1={yy} y2={yy} stroke={color} strokeWidth={1} strokeDasharray={dash} />
        <text x={W - PAD_R + 3} y={yy + 3.5} fontSize={9.5} fill={color} className="tnum">
          {label}
        </text>
      </g>
    );
  };

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[14px] font-semibold">🧮 筹码分布</h2>
        <span className="label" title={`${chips.sessions} 个交易日 · 衰减系数 ${chips.decay}`}>
          {chips.setup_label ?? ""} {chips.setup_score != null ? chips.setup_score.toFixed(2) : ""}
        </span>
      </div>

      {!chips.converged && (
        <p className="text-[11.5px] text-brand-ink leading-snug">
          ⚠️ 换手不足（累计 {fixed(chips.cum_turnover_pct, 0)}%），初始假设仍占{" "}
          {fixed(chips.seed_remaining, 1)}%，数字仅供参考。
        </p>
      )}

      <div className="grid grid-cols-3 gap-2">
        <Metric k="获利盘" v={`${fixed(chips.winner_rate, 1)}%`}
          cls={(chips.winner_rate ?? 0) >= 50 ? "text-up" : "text-down"} />
        <Metric k="平均成本" v={fixed(chips.weight_avg)} />
        <Metric k="集中度" v={fixed(chips.concentration, 3)} />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img"
        aria-label="筹码分布直方图">
        {chips.prices.map((p, i) => {
          const w = chips.weights[i] ?? 0;
          return (
            <rect key={i} x={0} y={y(p) - rowH} width={barW(w)} height={Math.max(rowH, 1)}
              fill={p < price ? "rgba(220,38,38,0.55)" : "rgba(22,163,74,0.55)"}>
              <title>{`¥${p.toFixed(2)} · ${w.toFixed(2)}%`}</title>
            </rect>
          );
        })}
        <Line p={price} color="#111827" dash="4 3" label={price.toFixed(2)} />
        {chips.weight_avg != null && (
          <Line p={chips.weight_avg} color="#2563eb" dash="2 3" label="均价" />
        )}
      </svg>

      <div className="flex items-center justify-between text-[11px] text-ink-mute">
        <span>🔴 获利 {fixed(chips.winner_rate, 0)}%</span>
        <span>套牢 {fixed(chips.trapped_rate, 0)}% 🟢</span>
      </div>

      <div className="grid grid-cols-5 gap-1 text-center">
        {([["5%", chips.cost_5pct], ["15%", chips.cost_15pct], ["50%", chips.cost_50pct],
           ["85%", chips.cost_85pct], ["95%", chips.cost_95pct]] as const).map(([k, v]) => (
          <div key={k} className="flex flex-col leading-tight">
            <span className="text-[10.5px] text-ink-mute">{k}</span>
            <span className="font-mono tnum text-[11.5px]">{fixed(v)}</span>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-ink-mute leading-snug">
        主峰 ¥{fixed(chips.peak_price)}（{chips.n_peaks} 个峰
        {chips.peaks[0]?.share != null ? `，最大一峰持有 ${fixed(chips.peaks[0].share, 0)}%` : ""}）
        ·{" "}
        {chips.peak_price != null && price >= chips.peak_price
          ? "主峰在现价下方，构成支撑"
          : "主峰在现价上方，是压力"}
      </p>
    </section>
  );
}

function Metric({ k, v, cls = "" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="flex flex-col leading-tight">
      <span className="label">{k}</span>
      <span className={`font-mono tnum text-[15px] font-semibold ${cls}`}>{v}</span>
    </div>
  );
}
