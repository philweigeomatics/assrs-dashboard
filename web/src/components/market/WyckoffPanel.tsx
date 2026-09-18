/**
 * 统计威科夫阶段 — the market's regime, defined arithmetically.
 *
 * Candles with the 120-day Donchian channel over them and the phase shaded
 * behind them, so the question "when did this change, and what was the price
 * doing when it did" is answered by looking rather than by reading a label.
 *
 * Drawn as plain SVG rather than through lightweight-charts: the panel needs
 * background bands spanning date ranges, which is the one thing that library
 * makes awkward, and 180 bars of static history need none of what it is good
 * at. Bars are laid out by index, so a holiday is a missing bar rather than a
 * gap — the same choice the phase maths makes (see wyckoff.py).
 */

import { useMemo } from "react";
import type { Wyckoff } from "../../lib/types";
import { fixed, signed } from "../../lib/format";
import { useSize } from "../../lib/useSize";

const PAD = { t: 8, r: 52, b: 20, l: 8 };

export function WyckoffPanel({ data, index, onIndex, indices }: {
  data: Wyckoff; index: string; onIndex: (i: string) => void;
  indices: Record<string, string>;
}) {
  const meta = data.phases[data.phase]!;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="px-2 h-6 rounded-md text-[12.5px] font-semibold flex items-center"
          style={{ background: `${meta.color}1f`, color: meta.color }}>
          {meta.label} · {meta.en}
        </span>
        <span className="label">
          自 {data.since} 起 · 已 {data.days_in_phase} 个交易日
          {data.confirm > 1 && ` · 连续 ${data.confirm} 日成立方才切换`}
        </span>
        <div className="ml-auto flex rounded-lg bg-sunken p-0.5">
          {Object.entries(indices).map(([code, label]) => (
            <button key={code} onClick={() => onIndex(code)}
              className={`px-2 h-6 rounded-md text-[12px] font-medium transition-colors ${
                index === code ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <p className="text-[12.5px] text-ink-dim leading-snug">{meta.means}</p>

      <div className="grid grid-cols-3 gap-2">
        <Stat label={`区间位置 (${data.lookback}日)`} value={`${fixed(data.position_pct, 1)}%`}
          hint="收盘价在120日最高/最低区间中的位置。100% = 贴着区间顶部。" />
        <Stat label="成交量 Z 值" value={fixed(data.volume_z, 2)}
          hint="相对60日均量的标准差数。>2 为异常放量。" />
        <Stat label="波动率 / 基准"
          value={`${fixed(data.volatility, 2)} / ${fixed(data.vol_baseline, 2)}`}
          hint="20日收益率标准差，与其自身120日均值比较。高于基准 = 换手加剧。" />
      </div>

      <Chart data={data} />
      <Edge data={data} />
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-1.5" title={hint}>
      <div className="label truncate">{label}</div>
      <div className="text-[15px] font-semibold tnum">{value}</div>
    </div>
  );
}

function Chart({ data }: { data: Wyckoff }) {
  const [ref, size] = useSize<HTMLDivElement>();
  const w = Math.max(size.w, 280);
  const h = 320;

  const geom = useMemo(() => {
    const lows = data.bars.map((b) => b.l).concat(data.channel.low);
    const highs = data.bars.map((b) => b.h).concat(data.channel.high);
    const lo = Math.min(...lows);
    const hi = Math.max(...highs);
    const pad = (hi - lo) * 0.04 || 1;
    return { lo: lo - pad, hi: hi + pad };
  }, [data]);

  const n = data.bars.length;
  const plotW = w - PAD.l - PAD.r;
  const plotH = h - PAD.t - PAD.b;
  const step = plotW / Math.max(n, 1);
  const x = (i: number) => PAD.l + (i + 0.5) * step;
  const y = (v: number) => PAD.t + (1 - (v - geom.lo) / (geom.hi - geom.lo)) * plotH;
  const body = Math.max(step * 0.62, 1);

  const line = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  // Evenly spaced, ending on the last bar. Ticks on phase boundaries read as
  // a better idea than they are: after confirmation there are few of them and
  // they land at arbitrary spacings, so the axis stops working as a ruler.
  const every = Math.max(Math.floor(n / 6), 1);
  const ticks = Array.from({ length: n }, (_, i) => i)
    .filter((i) => (n - 1 - i) % every === 0);

  return (
    <div ref={ref} className="min-w-0">
      {size.w > 0 && (
        <svg width={w} height={h} className="block select-none" role="img"
          aria-label={`${data.name} 的K线与威科夫阶段`}>
          {data.spans.map((s, i) => (
            <rect key={i} x={PAD.l + s.from * step} y={PAD.t}
              width={(s.to - s.from + 1) * step} height={plotH}
              fill={data.phases[s.phase]!.color} opacity={0.1}>
              <title>{`${data.phases[s.phase]!.label} · ${data.dates[s.from]} → ${data.dates[s.to]}`}</title>
            </rect>
          ))}

          <path d={line(data.channel.high)} fill="none" stroke="var(--color-ink-mute)"
            strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />
          <path d={line(data.channel.low)} fill="none" stroke="var(--color-ink-mute)"
            strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />

          {data.bars.map((b, i) => {
            // A-share convention: red when the bar closed above its open.
            const up = b.c >= b.o;
            const color = up ? "var(--color-up)" : "var(--color-down)";
            const top = y(Math.max(b.o, b.c));
            const bottom = y(Math.min(b.o, b.c));
            return (
              <g key={i}>
                <line x1={x(i)} y1={y(b.h)} x2={x(i)} y2={y(b.l)}
                  stroke={color} strokeWidth={1} />
                <rect x={x(i) - body / 2} y={top} width={body}
                  height={Math.max(bottom - top, 0.8)} fill={color} />
              </g>
            );
          })}

          {/* Price axis on the right, where the last bar is. */}
          {[geom.hi, (geom.hi + geom.lo) / 2, geom.lo].map((v) => (
            <text key={v} x={w - PAD.r + 5} y={y(v) + 3.5} fontSize={10.5}
              fill="var(--color-ink-mute)" className="tnum">{Math.round(v)}</text>
          ))}
          {ticks.map((i) => (
            <text key={i} x={x(i)} y={h - 6} fontSize={10} textAnchor="middle"
              fill="var(--color-ink-mute)" className="tnum">
              {data.dates[i]!.slice(2, 7)}
            </text>
          ))}
        </svg>
      )}
    </div>
  );
}

function Edge({ data }: { data: Wyckoff }) {
  return (
    <div className="rounded-lg border border-line p-2.5 flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">阶段验证</span>
        <span className="label">{data.name} 自身历史 · 未来 {data.edge.horizon} 个交易日</span>
      </div>
      {data.edge.rows.map((r) => (
        <div key={r.phase} className="flex items-baseline gap-2 text-[11.5px]">
          <span className="flex-1 truncate" style={{ color: data.phases[r.phase]!.color }}>
            {data.phases[r.phase]!.label}
            {r.phase === data.phase && <b className="text-ink"> ←当前</b>}
          </span>
          <span className={`tnum w-14 text-right ${r.thin ? "text-brand-ink" : "text-ink-mute"}`}
            title={r.thin ? `少于 ${data.edge.meaningful_at} 次，基本是噪声` : ""}>
            n={r.n}
          </span>
          <span className="tnum w-12 text-right text-ink-mute">{fixed(r.win_pct, 0)}%</span>
          <span className={`tnum w-16 text-right font-medium ${
            (r.edge_pp ?? 0) > 0 ? "text-up" : (r.edge_pp ?? 0) < 0 ? "text-down" : "text-flat"}`}>
            {r.edge_pp == null ? "—" : `${signed(r.edge_pp, 2)}pp`}
          </span>
        </div>
      ))}
      <p className="text-[11px] text-ink-mute leading-snug">
        每个阶段之后 {data.edge.horizon} 个交易日的平均涨跌，减去全样本均值
        （{signed(data.edge.baseline_pct, 2)}%）。只有显著为正/为负才说明这个标签本身带信息。
        窗口互相重叠，n 远大于独立观测数；样本不足 {data.edge.meaningful_at} 次的行已标黄。
      </p>
    </div>
  );
}
