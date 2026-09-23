/**
 * 全球指数 — the strip above the heatmap.
 *
 * The one thing this has to get right is that these markets are not on the
 * same clock. At ten in the morning in Shanghai the S&P's number is from
 * last night and Tokyo may be on holiday, so every card shows the date its
 * number belongs to whenever that is not the newest in the strip, and says
 * how many sessions behind it is. Without that, "欧美 down while we are up"
 * reads as a fact about this morning when it is a fact about last night.
 *
 * A card whose bar is dated today in its own timezone is marked 今日 — not
 * 盘中, because that is also true ten minutes after the close and this does
 * not model opening hours. The tooltip carries the rest: if the session is
 * still running the move will change before it settles.
 *
 * 红涨绿跌 throughout, including on the Western indices — the whole
 * dashboard reads that way, and mixing conventions inside one strip would be
 * worse than applying the local one to foreign markets.
 */

import type { WorldIndex, WorldIndices } from "../../lib/types";
import { moveClass, signed } from "../../lib/format";

/** Sparkline viewBox. The svg stretches to the card; these set the shape. */
const SW = 100;
const SH = 22;

export function IndexStrip({ data }: { data: WorldIndices }) {
  const anyBehind = data.groups.some((g) =>
    g.indices.some((i) => (i.behind_days ?? 0) > 0));

  return (
    <div className="flex flex-col gap-2">
      {data.groups.map((g) => (
        <div key={g.name} className="flex flex-col gap-1">
          <div className="flex items-baseline gap-2">
            <span className="text-[12px] text-ink-mute">{g.name}</span>
            {g.missing > 0 && (
              <span className="label text-brand-ink">{g.missing} 个读取失败</span>
            )}
          </div>
          <div className="grid gap-1.5 grid-cols-2 sm:grid-cols-3
            lg:grid-cols-4 xl:grid-cols-6">
            {g.indices.map((i) => <Card key={i.code} i={i} newest={data.as_of} />)}
            {g.indices.length === 0 && (
              <span className="self-center label">暂无数据</span>
            )}
          </div>
        </div>
      ))}

      <p className="label leading-snug">
        各市场收盘时间不同，每张卡片标的是它自己那一根的日期。
        {anyBehind && " 标了日期、颜色偏淡的是更早一个交易日的数字，不是今天的。"}
        {" 标「今日」的是该市场今天的数据，还在交易时段的话收盘前会变。"}
        {data.as_of && ` 最新一根：${data.as_of}`}
        {` · 取数 ${data.fetched_at}`}
      </p>
    </div>
  );
}

function Card({ i, newest }: { i: WorldIndex; newest: string | null }) {
  const behind = i.behind_days ?? 0;
  const stale = behind > 0;
  return (
    <div title={`${i.name}（${i.code}）· ${i.date} · ${i.close}`
      + (stale ? `，比最新的 ${newest} 落后 ${behind} 个交易日` : "")
      + (i.today ? " · 这是该市场今天的数据；若还在交易时段，涨跌是盘中的，收盘前还会变"
                 : "")}
      className={`card px-2.5 py-2 flex flex-col gap-1 ${
        stale ? "opacity-60" : ""}`}>
      <div className="flex items-baseline gap-1.5">
        <span className="text-[12px] truncate">{i.name}</span>
        {i.today && (
          /* Today's bar in that market's timezone — which is also true just
             after its close, so this says 今日 and not 盘中. */
          <span className="text-[10px] text-brand-ink shrink-0">今日</span>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold tnum">
          {i.close.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </span>
        <span className={`ml-auto text-[12.5px] font-semibold tnum ${
          moveClass(i.change_pct)}`}>
          {signed(i.change_pct, 2, "%")}
        </span>
      </div>
      <div className="flex items-end gap-1.5">
        <div className="flex-1 min-w-0"><Spark values={i.spark} up={i.change_pct >= 0} /></div>
        {stale && (
          <span className="text-[10px] text-ink-mute tnum shrink-0">
            {i.date.slice(5)}
          </span>
        )}
      </div>
    </div>
  );
}

function Spark({ values, up }: { values: number[]; up: boolean }) {
  if (values.length < 2) return <div style={{ height: SH }} />;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const d = values.map((v, k) =>
    `${k === 0 ? "M" : "L"}${((k / (values.length - 1)) * SW).toFixed(1)} ${
      (SH - ((v - lo) / span) * (SH - 2) - 1).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${SW} ${SH}`} preserveAspectRatio="none"
      className="block w-full" style={{ height: SH }} aria-hidden="true">
      <path d={d} fill="none" strokeWidth={1.25} vectorEffect="non-scaling-stroke"
        stroke={up ? "var(--color-up)" : "var(--color-down)"} />
    </svg>
  );
}
