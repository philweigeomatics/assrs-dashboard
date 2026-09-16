/**
 * The chart's toolbar: zoom, drawing tools, and the comparison stock.
 *
 * "清除绘图" removes only what YOU drew. Anything the analysis computed —
 * 箱体, regime shading, markers — is untouched, and redrawing the analysis
 * never wipes a measurement. They are separate stores for that reason.
 */

import { useState } from "react";
import type { Tool } from "./chart/ChartStack";
import type { CompareResult, StockRef } from "../lib/types";
import { suggest } from "../lib/search";

const BTN = "h-7 px-2 rounded-md text-[12.5px] border transition-colors";
const OFF = "border-line bg-panel hover:bg-elevated";
const ON = "border-cyan bg-cyan text-white";

export function ChartTools({
  tool, setTool, onReset, drawingCount, onClearDrawings,
  stocks, compare, compareMode, onCompare, onCompareMode,
}: {
  tool: Tool;
  setTool: (t: Tool) => void;
  onReset: () => void;
  drawingCount: number;
  onClearDrawings: () => void;
  stocks: StockRef[];
  compare: CompareResult | null;
  compareMode: "pct" | "price";
  onCompare: (t: string | null) => void;
  onCompareMode: (m: "pct" | "price") => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const hits = q.trim() ? suggest(q, stocks, [], 6) : [];

  const T = ({ id, label, title }: { id: Tool; label: string; title: string }) => (
    <button title={title} onClick={() => setTool(tool === id ? "none" : id)}
      className={`${BTN} ${tool === id ? ON : OFF}`}>{label}</button>
  );

  return (
    <div className="card px-2 py-1.5 flex flex-wrap items-center gap-1.5">
      <T id="zoom" label="🔍 框选放大" title="在价格图上拖出一段，放大到该区间" />
      <T id="measure" label="📏 测量" title="拖出一个矩形，显示涨跌幅与跨越的K线数" />
      <T id="hline" label="➖ 水平线" title="点击价格图画一条水平参考线" />
      <button onClick={onReset} className={`${BTN} ${OFF}`} title="回到默认显示区间">↺ 重置视图</button>
      <button onClick={onClearDrawings} disabled={drawingCount === 0}
        className={`${BTN} ${OFF} disabled:opacity-40`}
        title="只清除你画的东西，不影响箱体等计算结果">
        🧹 清除绘图{drawingCount ? ` (${drawingCount})` : ""}
      </button>

      <div className="ml-auto flex items-center gap-1.5">
        {compare ? (
          <>
            <span className="text-[12px] text-[#7c3aed] font-medium">
              {compare.name} {compare.change_pct != null ? `${compare.change_pct > 0 ? "+" : ""}${compare.change_pct}%` : ""}
            </span>
            <div className="flex rounded-md overflow-hidden border border-line">
              <button onClick={() => onCompareMode("pct")}
                title="两只股票共用同一纵轴，对比股按起点重定基准——线之间的差就是相对涨跌"
                className={`px-2 h-7 text-[12px] ${compareMode === "pct" ? "bg-cyan text-white" : "bg-panel"}`}>
                同一%刻度
              </button>
              <button onClick={() => onCompareMode("price")}
                title="对比股使用左侧独立价格轴，显示真实价格"
                className={`px-2 h-7 text-[12px] ${compareMode === "price" ? "bg-cyan text-white" : "bg-panel"}`}>
                独立价格轴
              </button>
            </div>
            <button onClick={() => onCompare(null)} className={`${BTN} ${OFF}`}>✕</button>
          </>
        ) : (
          <div className="relative">
            <input
              value={q} placeholder="对比另一只股票…"
              onChange={(e) => { setQ(e.target.value); setOpen(true); }}
              onFocus={() => setOpen(true)}
              onBlur={() => window.setTimeout(() => setOpen(false), 150)}
              className="h-7 w-44 px-2 rounded-md bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40"
            />
            {open && hits.length > 0 && (
              <div className="card absolute right-0 mt-1 z-30 w-56 py-1">
                {hits.map((s) => (
                  <button key={s.t} onMouseDown={(e) => e.preventDefault()}
                    onClick={() => { onCompare(s.t); setQ(""); setOpen(false); }}
                    className="w-full flex gap-2 px-2 py-1 text-left hover:bg-elevated">
                    <span className="font-mono tnum text-[12px] text-ink-dim">{s.t}</span>
                    <span className="text-[12.5px] truncate">{s.n}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
