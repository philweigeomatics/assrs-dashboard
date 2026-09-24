/**
 * 📊 宏观 — inflation, growth, liquidity and rates.
 *
 * Each card is a level, the change from the prior period, and two years of
 * shape. The shape is the point: a PMI of 49.8 means one thing after three
 * months at 51 and something else after three months at 48, and a card that
 * shows only the latest reading cannot tell you which.
 *
 * Where a series has a level that divides two regimes — 50 on a PMI, zero on
 * a YoY inflation print — it is drawn on the sparkline, so "below the line"
 * is something you see rather than something you have to know.
 *
 * Red up, green down, as everywhere else in this app.
 */

import type { MacroBoard, MacroCard } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const VW = 120;
const VH = 32;

export function MacroPanel({ data }: { data: MacroBoard }) {
  return (
    <div className="flex flex-col gap-3">
      {data.groups.map((g) => (
        <div key={g} className="flex flex-col gap-1.5">
          <h4 className="text-[12.5px] font-semibold text-ink-dim">{g}</h4>
          <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3
            xl:grid-cols-4">
            {data.cards.filter((c) => c.group === g).map((c) => (
              <Card key={c.label} c={c} />
            ))}
          </div>
        </div>
      ))}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="label">
          数据来自 Tushare，{data.as_of} 读取。红涨绿跌，变化是与上一期相比。
        </span>
        {data.missing.length > 0 && (
          <span className="text-[11.5px] text-brand-ink">
            这几项没取到：{data.missing.join("、")}
          </span>
        )}
      </div>
    </div>
  );
}

function Card({ c }: { c: MacroCard }) {
  const up = (c.change ?? 0) > 0;
  const flat = c.change == null || Math.abs(c.change) < 0.005;
  const tone = flat ? "text-ink-mute" : up ? "text-up" : "text-down";

  // Only meaningful where the series has two regimes to be in.
  const below = c.threshold != null && c.value < c.threshold;
  const regime = c.threshold == null ? null
    : below ? { text: c.threshold === 50 ? "收缩" : "负值", cls: "text-down" }
            : { text: c.threshold === 50 ? "扩张" : "正值", cls: "text-up" };

  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 flex flex-col gap-1"
      title={`${c.note}\n最新 ${c.period}${
        c.prev == null ? "" : ` · 上期 ${c.prev}${c.unit}`}`}>
      <div className="flex items-baseline gap-1.5">
        <span className="label truncate">{c.label}</span>
        {regime && (
          <span className={`text-[10.5px] font-medium ${regime.cls}`}>
            {regime.text}
          </span>
        )}
        <span className="ml-auto label font-mono tnum text-[10.5px]">
          {c.period}
        </span>
      </div>

      <div className="flex items-end gap-2">
        <div className="flex items-baseline gap-1.5 min-w-0">
          <span className="text-[17px] font-semibold tnum">
            {fixed(c.value, 2)}
          </span>
          <span className="label">{c.unit}</span>
        </div>
        <span className={`text-[12px] tnum font-medium ${tone}`}>
          {flat ? "持平" : signed(c.change, 2, c.unit)}
        </span>
        <Spark c={c} />
      </div>
    </div>
  );
}

/** Two years of shape, with the regime line drawn where there is one. */
function Spark({ c }: { c: MacroCard }) {
  const v = c.history;
  if (v.length < 2) return <span className="ml-auto" />;

  const first = v[0] ?? 0;
  const last = v[v.length - 1] ?? 0;
  const pts = c.threshold == null ? v : [...v, c.threshold];
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (v.length - 1)) * VW;
  const y = (n: number) => VH - 3 - ((n - lo) / span) * (VH - 6);

  const rising = last >= first;
  const stroke = rising ? "var(--color-up)" : "var(--color-down)";
  const d = v.map((n, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(n).toFixed(1)}`)
    .join("");

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
      className="ml-auto w-[120px] h-[32px] shrink-0 block" role="img"
      aria-label={`${c.label} 近 ${v.length} 期走势`}>
      {c.threshold != null && (
        <line x1={0} x2={VW} y1={y(c.threshold)} y2={y(c.threshold)}
          stroke="var(--color-line-bright)" strokeDasharray="3 3"
          vectorEffect="non-scaling-stroke" />
      )}
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.5}
        vectorEffect="non-scaling-stroke" />
      <circle cx={x(v.length - 1)} cy={y(last)} r={2} fill={stroke} />
    </svg>
  );
}
