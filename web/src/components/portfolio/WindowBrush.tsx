/**
 * Ten years of an index, with a window you drag along it.
 *
 * Drag the middle to move the window, drag an edge to resize it, click a
 * shaded leg to snap to it. Numbers only change when you let go: every release
 * prices each stock again, so a drag must not.
 */

import { useMemo, useRef, useState } from "react";
import type { PortfolioWindowIndex, WindowRegime } from "../../lib/types";
import { signed } from "../../lib/format";
import {
  MIN_WINDOW, lastSessions, moveWindow, resizeEnd, resizeStart, type Range,
} from "../../lib/windowRange";

const VW = 1000;
const VH = 170;
const HANDLE_PX = 10;

const PRESETS = [
  { label: "60天", sessions: 60 },
  { label: "120天", sessions: 120 },
  { label: "1年", sessions: 242 },
  { label: "3年", sessions: 726 },
  { label: "5年", sessions: 1210 },
  { label: "全部", sessions: Infinity },
];

type Drag = { mode: "start" | "end" | "move" | "none"; x: number; k: number; base: Range; moved: boolean };

export function legTitle(r: WindowRegime) {
  const word = r.kind === "up" ? "上行" : "下行";
  return `${r.from} → ${r.to} · ${word} ${signed(r.change_pct, 1, "%")} · ${r.sessions} 个交易日`
    + (r.open ? " · 还没走完" : ` · ${r.confirmed} 才确认`)
    + (r.sessions < MIN_WINDOW ? ` · 不足 ${MIN_WINDOW} 天，不能选作窗口` : "");
}

export function WindowBrush({ data, range, onCommit }: {
  data: PortfolioWindowIndex;
  range: Range;
  onCommit: (r: Range) => void;
}) {
  const { dates, closes, regimes, label } = data;
  const n = dates.length;
  const [live, setLive] = useState<Range | null>(null);
  const shown = live ?? range;
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);

  const legs = useMemo(() => {
    const at = new Map(dates.map((d, i) => [d, i]));
    return regimes.map((r) => ({ r, i: at.get(r.from) ?? 0, j: at.get(r.to) ?? n - 1 }));
  }, [dates, regimes, n]);

  const path = useMemo(() => {
    const logs = closes.map(Math.log);
    const lo = Math.min(...logs);
    const hi = Math.max(...logs);
    return logs.map((v, i) => {
      const y = VH - 6 - ((v - lo) / (hi - lo || 1)) * (VH - 12);
      return `${i ? "L" : "M"}${((i / (n - 1)) * VW).toFixed(1)} ${y.toFixed(1)}`;
    }).join("");
  }, [closes, n]);

  const years = useMemo(() => {
    const out: { y: string; i: number }[] = [];
    dates.forEach((d, i) => {
      const y = d.slice(0, 4);
      if (i > 0 && y !== dates[i - 1]!.slice(0, 4)) out.push({ y, i });
    });
    return out;
  }, [dates]);

  const x = (i: number) => (i / (n - 1)) * VW;
  const pct = (i: number) => `${(i / (n - 1)) * 100}%`;

  const locate = (clientX: number) => {
    const rect = box.current!.getBoundingClientRect();
    const px = clientX - rect.left;
    const k = Math.max(0, Math.min(n - 1, Math.round((px / rect.width) * (n - 1))));
    const [i, j] = shown;
    const left = (i / (n - 1)) * rect.width;
    const right = (j / (n - 1)) * rect.width;
    // A window drawn narrow still has a middle to grab.
    const reach = Math.min(HANDLE_PX, (right - left) / 3);
    const mode: Drag["mode"] = Math.abs(px - left) <= reach ? "start"
      : Math.abs(px - right) <= reach ? "end"
      : px > left && px < right ? "move" : "none";
    return { k, mode };
  };

  const onDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const { k, mode } = locate(e.clientX);
    drag.current = { mode, x: e.clientX, k, base: shown, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    const { k, mode } = locate(e.clientX);
    if (!d) {
      e.currentTarget.style.cursor = mode === "start" || mode === "end" ? "ew-resize"
        : mode === "move" ? "grab" : "pointer";
      return;
    }
    if (Math.abs(e.clientX - d.x) > 3) d.moved = true;
    if (!d.moved || d.mode === "none") return;
    setLive(d.mode === "move" ? moveWindow(d.base, k - d.k, n)
      : d.mode === "start" ? resizeStart(d.base, k)
      : resizeEnd(d.base, k, n));
  };

  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.moved) {
      if (live) onCommit(live);
      setLive(null);
      return;
    }
    const leg = legs.find((l) => d.k >= l.i && d.k <= l.j);
    if (leg && leg.j - leg.i >= MIN_WINDOW) onCommit([leg.i, leg.j]);
  };

  const [i, j] = shown;
  const move = (closes[j]! / closes[i]! - 1) * 100;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-[13px] font-medium tnum">
          {dates[i]} → {dates[j]}
        </span>
        <span className="label tnum">{j - i} 个交易日</span>
        <span className={`text-[12.5px] tnum font-medium ${move >= 0 ? "text-up" : "text-down"}`}>
          {label} {signed(move, 1, "%")}
        </span>
        <div className="ml-auto flex items-center gap-0.5">
          {PRESETS.map((p) => (
            <button key={p.label} onClick={() => onCommit(lastSessions(n, p.sessions))}
              className="h-6 px-2 rounded-md text-[12px] text-ink-mute hover:text-ink hover:bg-elevated">
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <div ref={box} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}
        onPointerCancel={() => { drag.current = null; setLive(null); }}
        className="relative select-none" style={{ touchAction: "none" }}>
        <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="none"
          className="w-full h-[170px] block" role="img"
          aria-label={`${label} 十年走势与选中的时间窗口`}>
          {legs.map(({ r, i: a, j: b }) => (
            <rect key={r.from} x={x(a)} y={0} width={Math.max(0, x(b) - x(a))} height={VH}
              style={{ fill: `color-mix(in srgb, var(--color-${r.kind === "up" ? "up" : "down"}) ${r.open ? 7 : 13}%, transparent)` }}>
              <title>{legTitle(r)}</title>
            </rect>
          ))}
          <path d={path} fill="none" stroke="var(--color-ink)" strokeWidth={1.3}
            vectorEffect="non-scaling-stroke" />
          <rect x={0} y={0} width={x(i)} height={VH} pointerEvents="none"
            style={{ fill: "var(--color-panel)", opacity: 0.68 }} />
          <rect x={x(j)} y={0} width={VW - x(j)} height={VH} pointerEvents="none"
            style={{ fill: "var(--color-panel)", opacity: 0.68 }} />
        </svg>
        {[i, j].map((p, s) => (
          <div key={s} className="absolute top-0 bottom-0 pointer-events-none"
            style={{ left: pct(p), width: 0 }}>
            <div className="absolute inset-y-0 -left-px w-0.5 bg-cyan" />
            <div className="absolute top-1/2 -translate-y-1/2 -left-[5px] w-2.5 h-7 rounded-sm bg-cyan" />
          </div>
        ))}
      </div>

      <div className="relative h-3.5 label tnum">
        {years.map((t, s) => (
          <span key={t.y} className={`absolute -translate-x-1/2 ${s % 2 ? "hidden sm:inline" : ""}`}
            style={{ left: pct(t.i) }}>
            {t.y}
          </span>
        ))}
      </div>
    </div>
  );
}
