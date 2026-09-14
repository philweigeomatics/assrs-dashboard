/**
 * Per-pane legend, floating top-left over the canvas.
 *
 * Each entry reads its value at the crosshair (or the latest bar at rest) and
 * is a toggle: click to hide or show that line or marker set. Hidden entries
 * stay in the legend, struck through, so there is always a way back.
 *
 * `pointer-events-none` on the wrapper is load-bearing: lightweight-charts
 * owns pointer events on the canvas, and an overlay that accepted them would
 * swallow every drag that started under it. Only the buttons opt back in.
 */

import type { PaneSpec } from "../../lib/panes";

const GLYPH: Record<string, string> = {
  arrowUp: "▲",
  arrowDown: "▼",
  circle: "●",
  square: "■",
};

export function Legend({
  pane,
  index,
  hidden,
  onToggle,
}: {
  pane: PaneSpec;
  index: number;
  hidden: string[];
  onToggle: (id: string) => void;
}) {
  const off = new Set(hidden);
  return (
    <div className="float-ground w-fit max-w-[calc(100%-90px)] m-1.5 px-2 py-1 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 pointer-events-none">
      <span className="text-[12px] font-semibold shrink-0">{pane.title}</span>
      {pane.lines.map((l) => {
        const id = `${pane.id}:${l.key}`;
        const v = l.values[index];
        const isOff = off.has(id);
        return (
          <button
            key={id}
            type="button"
            onClick={() => onToggle(id)}
            title={isOff ? "点击显示" : "点击隐藏"}
            className={`pointer-events-auto flex items-center gap-1 text-[11.5px] font-mono tnum rounded px-1 -mx-1 hover:bg-elevated ${
              isOff ? "opacity-40 line-through" : ""
            }`}
          >
            <span aria-hidden className="w-2 h-2 shrink-0 rounded-[2px]" style={{ background: l.color }} />
            <span style={{ color: l.color }}>{l.label}</span>
            <span className="text-ink-dim">
              {v == null || !Number.isFinite(v) ? "—" : v.toFixed(l.decimals ?? 2)}
            </span>
          </button>
        );
      })}
      {pane.markers.map((m) => {
        const id = `${pane.id}:${m.key}`;
        const isOff = off.has(id);
        const firesHere = m.idx.includes(index);
        return (
          <button
            key={id}
            type="button"
            onClick={() => onToggle(id)}
            title={`${m.label}：${m.idx.length} 次${isOff ? "（已隐藏）" : ""}`}
            className={`pointer-events-auto flex items-center gap-0.5 text-[11px] rounded px-1 -mx-0.5 hover:bg-elevated ${
              isOff ? "opacity-40 line-through" : ""
            } ${firesHere && !isOff ? "ring-1 ring-offset-0" : ""}`}
            style={firesHere && !isOff ? { boxShadow: `0 0 0 1px ${m.color}` } : undefined}
          >
            <span style={{ color: m.color }}>{GLYPH[m.shape]}</span>
            <span className="text-ink-mute">{m.label}</span>
          </button>
        );
      })}
    </div>
  );
}
