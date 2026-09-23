/**
 * The correlation matrix, as a grid readable at a glance.
 *
 * Deep colour everywhere means the basket is one bet wearing several names,
 * which is the single most useful thing this page can tell someone who
 * believes they have diversified. The average off-diagonal is spelled out
 * underneath so that judgement does not depend on reading shades.
 */

import type { PortfolioBuild } from "../../lib/types";
import { fixed } from "../../lib/format";

/** Cyan positive, violet negative — this is sign, not direction of a move. */
function cell(v: number | null): string {
  if (v == null) return "transparent";
  const a = 0.08 + Math.min(1, Math.abs(v)) * 0.72;
  return v >= 0 ? `rgba(6,182,212,${a})` : `rgba(168,85,247,${a})`;
}

export function CorrMatrix({ c, holdings }: {
  c: PortfolioBuild["correlation"];
  holdings: PortfolioBuild["holdings"];
}) {
  const name = Object.fromEntries(holdings.map((h) => [h.t, h.n]));
  const short = (t: string) => (name[t] || t).slice(0, 4);

  const off = c.rows.flatMap((row, i) =>
    row.filter((_, j) => j !== i).map((v) => v ?? 0));
  const avg = off.length ? off.reduce((a, b) => a + b, 0) / off.length : 0;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="overflow-x-auto">
        <table className="border-collapse text-[11px]">
          <thead>
            <tr>
              <th />
              {c.labels.map((t) => (
                <th key={t}
                  className="px-1 pb-1 font-normal text-ink-mute whitespace-nowrap">
                  {short(t)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {c.rows.map((row, i) => (
              <tr key={c.labels[i]}>
                <td className="pr-1.5 text-right text-ink-mute whitespace-nowrap">
                  {short(c.labels[i] ?? "")}
                </td>
                {row.map((v, j) => (
                  <td key={j} className="p-0">
                    <div
                      title={`${short(c.labels[i] ?? "")} / ${short(c.labels[j] ?? "")}`
                        + ` · ${fixed(v, 2)}`}
                      style={{ background: cell(v) }}
                      className="w-9 h-7 flex items-center justify-center tnum">
                      {i === j ? "" : fixed(v, 1).replace("0.", ".")}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        非对角线平均 <b className="tnum">{fixed(avg, 2)}</b>。
        {avg > 0.6
          ? " 这组股票基本同涨同跌，分散化的空间很小。"
          : avg > 0.35
            ? " 相关性偏高，但还有一些分散空间。"
            : " 相关性不高，分散是真的。"}
      </p>
    </div>
  );
}
