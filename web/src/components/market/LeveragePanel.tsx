/**
 * 市场杠杆 — how much of the money in the market is borrowed.
 *
 * Margin debt is the cleanest available read on risk appetite: it is money
 * people had to ask permission for. Rising = appetite (red, on the A-share
 * convention); falling = de-leveraging.
 *
 * The headline is a BALANCE — a stock. The A-share panel also shows the daily
 * FLOWS behind it, because they are what actually moves it:
 *
 *     融资余额(今) = 融资余额(昨) + 融资买入额 − 融资偿还额
 *
 * A flat balance can hide enormous gross churn, and a balance that fell
 * because repayments spiked is active de-leveraging rather than drift. The
 * total alone cannot tell those apart.
 */

import type { Leverage } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

export function LeveragePanel({ data }: { data: Leverage[] }) {
  const cn = data.find((d) => d.market === "CN");

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {data.map((d) => <Card key={d.market} d={d} />)}
      </div>
      {cn?.ok && cn.detail.length > 0 && <Flows d={cn} />}
    </div>
  );
}

function Card({ d }: { d: Leverage }) {
  if (!d.ok || d.latest == null) {
    return (
      <div className="rounded-lg bg-sunken p-2.5">
        <div className="label">{d.label}</div>
        <div className="text-[15px] font-semibold text-ink-mute">—</div>
        <p className="text-[11px] text-ink-mute leading-snug mt-1">
          {d.error || "暂时不可用"}
        </p>
      </div>
    );
  }
  const delta = d.prev == null ? null : d.latest - d.prev;
  // Rising leverage is red: the A-share convention, applied to the thing
  // itself rather than to a price.
  const tone = delta == null || delta === 0 ? "text-flat"
    : delta > 0 ? "text-up" : "text-down";

  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="label truncate">{d.label}</span>
        <span className="label ml-auto shrink-0">{d.asof}</span>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-[18px] font-semibold tnum">
          {d.latest.toLocaleString(undefined, { maximumFractionDigits: 1 })}
        </span>
        <span className="label">{d.unit}</span>
        {delta != null && (
          <span className={`text-[12.5px] tnum font-medium ${tone}`}>
            {signed(delta, 1)}
          </span>
        )}
      </div>
      <Spark d={d} />
      <p className="text-[10.5px] text-ink-mute leading-snug">{d.note}</p>
    </div>
  );
}

function Spark({ d }: { d: Leverage }) {
  const values = d.series.map((p) => p.value).filter((v): v is number => v != null);
  if (values.length < 2) return null;

  const w = 240;
  const h = 34;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i / (values.length - 1)) * w},${
      h - ((v - lo) / span) * h}`)
    .join("");
  const rising = values[values.length - 1]! >= values[0]!;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
      className="w-full h-[34px]" role="img"
      aria-label={`${d.label} 走势，${d.series[0]!.period} 至 ${d.asof}`}>
      <path d={path} fill="none" strokeWidth={1.5}
        stroke={rising ? "var(--color-up)" : "var(--color-down)"}
        vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Flows({ d }: { d: Leverage }) {
  const last = d.detail[d.detail.length - 1]!;
  const prev = d.detail.length > 1 ? d.detail[d.detail.length - 2]! : null;
  const bars = d.detail.slice(-60);
  const nets = bars.map((b) => b.net_fin ?? 0);
  const peak = Math.max(...nets.map(Math.abs), 1);

  const deLevering = (last.rzche ?? 0) > (last.rzmre ?? 0);

  return (
    <div className="rounded-lg border border-line p-2.5 flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <Metric label="融资余额 · 多头杠杆" value={last.rzye}
          delta={prev ? (last.rzye ?? 0) - (prev.rzye ?? 0) : null}
          hint="投资者借钱买股的余额，占两融的 95% 以上。" />
        <Metric label="融券余额 · 空头" value={last.rqye}
          delta={prev ? (last.rqye ?? 0) - (prev.rqye ?? 0) : null}
          hint="借券做空的余额，通常远小于融资余额。" />
        <Metric label="当日净融资" value={last.net_fin} delta={null}
          hint="融资买入额 − 融资偿还额。>0 净加杠杆，<0 净偿还。" />
      </div>

      <svg viewBox={`0 0 ${bars.length * 4} 60`} preserveAspectRatio="none"
        className="w-full h-[60px]" role="img" aria-label="近60个交易日的净融资流入">
        <line x1={0} y1={30} x2={bars.length * 4} y2={30}
          stroke="var(--color-line)" strokeWidth={0.5} vectorEffect="non-scaling-stroke" />
        {nets.map((v, i) => {
          const hgt = (Math.abs(v) / peak) * 28;
          return (
            <rect key={i} x={i * 4 + 0.5} width={3}
              y={v >= 0 ? 30 - hgt : 30} height={Math.max(hgt, 0.5)}
              fill={v >= 0 ? "var(--color-up)" : "var(--color-down)"} />
          );
        })}
      </svg>

      <p className="text-[11px] text-ink-mute leading-snug">
        当日融资买入 <b>{fixed(last.rzmre, 1)} 亿</b>，融资偿还 <b>{fixed(last.rzche, 1)} 亿</b>。
        {deLevering
          ? "偿还大于买入 → 资金在去杠杆 / 获利了结。"
          : "买入大于偿还 → 资金在加杠杆 / 加仓。"}
        余额是存量、买入与偿还是当日流量 —— 余额走平也可能对应巨大的换手。
        柱状图为近 60 个交易日的净融资流入（红 = 加杠杆，绿 = 去杠杆）。
      </p>
    </div>
  );
}

function Metric({ label, value, delta, hint }: {
  label: string; value: number | null | undefined; delta: number | null; hint: string;
}) {
  return (
    <div title={hint} className="min-w-[120px]">
      <div className="label truncate">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-[15px] font-semibold tnum">{fixed(value, 1)}</span>
        <span className="label">亿元</span>
        {delta != null && (
          <span className={`text-[11.5px] tnum ${
            delta > 0 ? "text-up" : delta < 0 ? "text-down" : "text-flat"}`}>
            {signed(delta, 1)}
          </span>
        )}
      </div>
    </div>
  );
}
