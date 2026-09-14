/**
 * A chart pane with its legend overlaid and a drag handle BELOW it.
 *
 * The handle is a normal-flow row, not a strip laid over the canvas: over
 * the canvas, lightweight-charts' own pointer handlers win and a press pans
 * the chart instead of resizing it. Height is tracked locally while dragging
 * and reported once on release — persisting every pointermove would be a
 * write per pixel. From BlindlyTrade's PriceChart.
 */

import { useEffect, useState, type ReactNode, type Ref } from "react";

export function ResizablePane({
  height,
  minHeight,
  onResize,
  hostRef,
  overlay,
}: {
  height: number;
  minHeight: number;
  onResize: (h: number) => void;
  hostRef: Ref<HTMLDivElement>;
  overlay: ReactNode;
}) {
  const [live, setLive] = useState(height);
  const [dragging, setDragging] = useState(false);
  useEffect(() => setLive(height), [height]);

  const clamp = (start: number, dy: number) => Math.max(minHeight, Math.min(1000, start + dy));

  function startDrag(e: React.PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const y0 = e.clientY;
    const h0 = live;
    setDragging(true);
    const prevSelect = document.body.style.userSelect;
    document.body.style.userSelect = "none";
    const move = (ev: PointerEvent) => setLive(clamp(h0, ev.clientY - y0));
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      document.body.style.userSelect = prevSelect;
      setDragging(false);
      onResize(clamp(h0, ev.clientY - y0));
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  }

  function onKey(e: React.KeyboardEvent) {
    const step = e.shiftKey ? 40 : 10;
    const d = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
    if (!d) return;
    e.preventDefault();
    const next = clamp(live, d);
    setLive(next);
    onResize(next);
  }

  return (
    <div className="card relative overflow-hidden">
      <div ref={hostRef} style={{ height: live }} />
      <div className="absolute inset-x-0 top-0 z-10 pointer-events-none">{overlay}</div>
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="拖动调整高度"
        aria-valuenow={Math.round(live)}
        tabIndex={0}
        title="拖动调整高度"
        onPointerDown={startDrag}
        onKeyDown={onKey}
        className={`group h-2.5 flex items-center justify-center cursor-ns-resize touch-none border-t transition-colors ${
          dragging ? "bg-cyan/15 border-cyan" : "border-line hover:bg-cyan/10"
        }`}
      >
        <span className={`h-0.5 w-8 rounded-full ${dragging ? "bg-cyan" : "bg-line-bright group-hover:bg-cyan"}`} />
      </div>
    </div>
  );
}
