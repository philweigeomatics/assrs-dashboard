/**
 * The bar under the crosshair — or the latest bar at rest.
 *
 * 涨跌 is against the PREVIOUS CLOSE (what a Chinese trader means by 涨跌幅),
 * and 振幅 is (high − low) / previous close, matching the Streamlit hover.
 */

import type { Analysis } from "../../lib/types";
import { compact, moveClass } from "../../lib/format";

export function Readout({ data, index }: { data: Analysis; index: number }) {
  const o = data.ohlcv.o[index];
  const h = data.ohlcv.h[index];
  const l = data.ohlcv.l[index];
  const c = data.ohlcv.c[index];
  const v = data.ohlcv.v[index];
  if (o == null || h == null || l == null || c == null) return null;
  const prev = index > 0 ? data.ohlcv.c[index - 1] ?? null : null;
  const chg = prev ? (c / prev - 1) * 100 : null;
  const amp = prev ? ((h - l) / prev) * 100 : null;

  const F = ({ k, val, cls = "text-ink-dim" }: { k: string; val: string; cls?: string }) => (
    <span className="flex items-baseline gap-1 shrink-0">
      <span className="text-[11px] text-ink-mute">{k}</span>
      <span className={`font-mono tnum text-[12.5px] ${cls}`}>{val}</span>
    </span>
  );

  return (
    <div className="float-ground w-fit max-w-[calc(100%-90px)] m-1.5 mb-0 px-2 py-1 flex flex-wrap items-center gap-x-3 gap-y-0.5">
      <span className="font-mono tnum text-[12.5px] font-semibold">{data.dates[index]}</span>
      <F k="开" val={o.toFixed(2)} />
      <F k="高" val={h.toFixed(2)} cls="text-up" />
      <F k="低" val={l.toFixed(2)} cls="text-down" />
      <F k="收" val={c.toFixed(2)} cls={`${moveClass(chg)} font-semibold`} />
      <F k="涨跌" val={chg == null ? "—" : `${chg > 0 ? "+" : ""}${chg.toFixed(2)}%`} cls={moveClass(chg)} />
      <F k="振幅" val={amp == null ? "—" : `${amp.toFixed(2)}%`} />
      <F k="量" val={v == null ? "—" : compact(v)} />
    </div>
  );
}
