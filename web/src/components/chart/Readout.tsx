/**
 * The bar under the crosshair — or the latest bar at rest.
 *
 * 涨跌 is against the PREVIOUS CLOSE (what a Chinese trader means by 涨跌幅),
 * and 振幅 is (high − low) / previous close, matching the Streamlit hover.
 */

import type { Analysis, CompareResult, SimResult } from "../../lib/types";
import { compact, moveClass } from "../../lib/format";

export function Readout({
  data,
  index,
  ghost = null,
  compare = null,
}: {
  data: Analysis;
  index: number;
  ghost?: SimResult | null;
  compare?: CompareResult | null;
}) {
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
    // `pointer-events-none` is load-bearing, exactly as in Legend: this panel
    // floats over the top-left of the price chart, which is precisely where
    // the cursor goes. Without it the readout swallows the mouse and the
    // crosshair never moves — the numbers sit frozen on the last bar while
    // you hover straight at them.
    <div className="float-ground w-fit max-w-[calc(100%-90px)] m-1.5 mb-0 px-2 py-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 pointer-events-none">
      <span className="font-mono tnum text-[12.5px] font-semibold">{data.dates[index]}</span>
      <F k="开" val={o.toFixed(2)} />
      {/* High is the "gain" colour and low the "loss" colour, which is red/green
          in Shanghai and green/red in New York — the same inversion as the
          candles, not two fixed classes. */}
      <F k="高" val={h.toFixed(2)} cls={data.up_is_red ? "text-up" : "text-down"} />
      <F k="低" val={l.toFixed(2)} cls={data.up_is_red ? "text-down" : "text-up"} />
      <F k="收" val={c.toFixed(2)} cls={`${moveClass(chg, data.up_is_red)} font-semibold`} />
      <F k="涨跌" val={chg == null ? "—" : `${chg > 0 ? "+" : ""}${chg.toFixed(2)}%`} cls={moveClass(chg, data.up_is_red)} />
      <F k="振幅" val={amp == null ? "—" : `${amp.toFixed(2)}%`} />
      <F k="量" val={v == null ? "—" : compact(v)} />
      {compare && (
        <span className="flex items-baseline gap-1 shrink-0 text-[#7c3aed]">
          <span className="text-[11px]">{compare.name}</span>
          <span className="font-mono tnum text-[12.5px]">
            {compare.price[index] == null ? "—" : compare.price[index]!.toFixed(2)}
          </span>
        </span>
      )}
      {ghost && ghost.ohlcv.c != null && (
        <span className="flex items-baseline gap-1 shrink-0 text-brand-ink">
          <span className="text-[11px]">👻 {ghost.date}</span>
          <span className="font-mono tnum text-[12.5px]">{ghost.ohlcv.c.toFixed(2)}</span>
        </span>
      )}
    </div>
  );
}
