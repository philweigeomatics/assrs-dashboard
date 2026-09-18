/**
 * 板块趋势历史 — where each sector index sits against its own 20-day mean.
 *
 * Called "market breadth" everywhere in this repo, and it is not. Each cell
 * comes from main.calculate_ppi_breadth_proxy, which takes the SECTOR INDEX's
 * percentage distance from its MA20 and maps ±5% linearly onto 0–100, clipped.
 * It counts no constituents, so it cannot tell you a sector rose on two names
 * — and it saturates, which is why so many cells read exactly 0 or 100. The
 * caption says so rather than repeating the old page's claim.
 *
 * The Streamlit page paged through this ten days at a time, which made the one
 * thing the grid is good for — watching a sector go from green to red over
 * three weeks — impossible without clicking back and forth and holding the
 * previous page in your head. Here the whole window is one scrollable grid,
 * sectors ranked by today, newest column on the right.
 *
 * Colour is a three-stop scale rather than the page's red/green cutoff at 50%,
 * because 49 and 51 are the same market and the old styling drew them as
 * opposites.
 */

import type { Breadth } from "../../lib/types";
import type { Num } from "../../lib/types";

const CELL = 34;
const LABEL = 96;

function cellColor(v: Num): string {
  if (v == null) return "var(--color-panel-alt)";
  // 0 → green, 0.5 → neutral, 1 → red. A-share convention: red is strength.
  const t = Math.min(Math.max(v, 0), 1);
  const k = Math.abs(t - 0.5) * 2;
  const [r, g, b] = t >= 0.5 ? [215, 0, 21] : [31, 122, 53];
  const mix = (c: number) => Math.round(248 + (c - 248) * (0.1 + 0.9 * k));
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

function ink(v: Num): string {
  return v != null && Math.abs(v - 0.5) > 0.32 ? "#fff" : "var(--color-ink-dim)";
}

export function BreadthGrid({ data }: { data: Breadth }) {
  const dates = data.dates;

  return (
    <div className="flex flex-col gap-2">
      <p className="label leading-snug">
        每格 = 该<b>板块指数</b>相对自身 20 日均线的位置：0 表示低于均线 5% 以上，
        100 表示高于均线 5% 以上，中间线性映射。
        <b className="text-up">红</b> = 在均线之上，
        <b className="text-down">绿</b> = 在均线之下。
        今日 <b>{data.hot}/{data.total}</b> 个板块在均线之上。
        注意：这<b>不是</b>成分股占比 —— 它按指数计算，会在 ±5% 处饱和，
        所以大量格子正好是 0 或 100。
      </p>

      <div className="overflow-auto max-h-[460px] rounded-lg border border-line">
        <table className="border-collapse text-[11px]" style={{ minWidth: LABEL + dates.length * CELL }}>
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-20 bg-panel px-2 py-1 text-left
                             font-medium text-ink-mute border-b border-line"
                style={{ width: LABEL }}>板块</th>
              {dates.map((d, i) => (
                <th key={d} className="sticky top-0 z-10 bg-panel py-1 font-normal
                                       text-ink-mute border-b border-line tnum"
                  style={{ width: CELL }}>
                  {/* Every fifth date only: 60 rotated labels is a hedge, not an axis. */}
                  {i % 5 === 0 || i === dates.length - 1 ? d.slice(5) : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.sectors.map((s) => (
              <tr key={s.name}>
                <td className="sticky left-0 z-10 bg-panel px-2 py-0.5 whitespace-nowrap
                               border-r border-line">{s.name}</td>
                {s.values.map((v, i) => (
                  <td key={i} className="text-center tnum p-0"
                    style={{ background: cellColor(v), color: ink(v), height: 20 }}
                    title={`${s.name} · ${dates[i]} · ${v == null ? "无数据" : `${Math.round(v * 100)}%`}`}>
                    {v == null ? "" : Math.round(v * 100)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
