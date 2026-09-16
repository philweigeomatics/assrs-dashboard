/**
 * What-If: type tomorrow's bar, see it drawn.
 *
 * Every edit re-simulates and the ghost redraws — no toggle, because in a
 * single-page app there is nothing to toggle: the ghost exists while there is
 * a simulation and vanishes when you clear it. 清除 removes only the ghost;
 * the analysis, the boxes and your drawings stay.
 *
 * O/H/L re-seed whenever Δ% changes, so the candle always matches the close
 * being simulated; editing them afterwards sticks. Supplying them makes ADX
 * and ±DI exact rather than estimated from the close alone.
 *
 * 尾盘推演 reads whichever bar is on screen: the ghost when one is drawn, the
 * last real session when it is not.
 */

import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Analysis, SimResult, WhatIfAi } from "../lib/types";
import { fixed, moveClass } from "../lib/format";

export function WhatIfPanel({
  data,
  ghost,
  onGhost,
}: {
  data: Analysis;
  ghost: SimResult | null;
  onGhost: (g: SimResult | null) => void;
}) {
  const last = data.dates.length - 1;
  const close0 = data.ohlcv.c[last] ?? 0;
  const vol0 = data.ohlcv.v[last] ?? 0;

  const [pct, setPct] = useState(0);
  const [open, setOpen] = useState(close0);
  const [high, setHigh] = useState(close0 * 1.005);
  const [low, setLow] = useState(close0 * 0.995);
  const [volume, setVolume] = useState(vol0);
  const [ai, setAi] = useState<WhatIfAi | null>(null);
  const seeded = useRef<number | null>(null);

  // A new stock is a new bar: reset everything, including the ghost.
  useEffect(() => {
    setPct(0);
    setOpen(close0);
    setHigh(close0 * 1.005);
    setLow(close0 * 0.995);
    setVolume(vol0);
    setAi(null);
    seeded.current = null;
    onGhost(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.ticker]);

  // Re-seed O/H/L on a Δ% change — at that point it is a different bar.
  const target = close0 * (1 + pct / 100);
  useEffect(() => {
    if (seeded.current === pct) return;
    seeded.current = pct;
    setOpen(close0);
    setHigh(Math.max(close0, target) * 1.005);
    setLow(Math.min(close0, target) * 0.995);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pct, close0]);

  const sim = useMutation({
    mutationFn: () =>
      api.simulate(data.ticker, {
        pct,
        volume: Math.max(volume, 1),
        open,
        high: Math.max(high, open, target),
        low: Math.min(low, open, target),
      }),
    onSuccess: (r) => onGhost(r),
  });

  const aiCall = useMutation({
    mutationFn: () =>
      api.whatifAi(data.ticker, ghost
        ? { mode: "ghost", pct, volume: Math.max(volume, 1), open,
            high: Math.max(high, open, target), low: Math.min(low, open, target) }
        : { mode: "actual" }),
    onSuccess: setAi,
  });

  // Debounced: typing a price should not fire a request per keystroke.
  const timer = useRef<number | null>(null);
  const simulateSoon = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => sim.mutate(), 250);
  };
  useEffect(() => () => {
    if (timer.current) window.clearTimeout(timer.current);
  }, []);

  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h2 className="text-[14px] font-semibold">🎮 What-If · 明日推演</h2>
        {ghost && (
          <button onClick={() => { onGhost(null); setAi(null); }}
            className="text-[12px] text-cyan">清除幻影</button>
        )}
      </div>

      <div className="grid grid-cols-5 gap-1.5">
        <NumField label="Δ%" value={pct} set={setPct} after={simulateSoon} step={0.1} />
        <NumField label="开" value={open} set={setOpen} after={simulateSoon} />
        <NumField label="高" value={high} set={setHigh} after={simulateSoon} />
        <NumField label="低" value={low} set={setLow} after={simulateSoon} />
        <NumField label="量" value={volume} set={setVolume} after={simulateSoon} step={100} />
      </div>

      <div className="flex items-center gap-2">
        <button onClick={() => sim.mutate()} disabled={sim.isPending}
          className="h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
          {sim.isPending ? "计算中…" : ghost ? "更新幻影" : "画出幻影"}
        </button>
        <span className="text-[11.5px] text-ink-mute">
          昨收 {fixed(close0)} → <span className={moveClass(pct)}>{fixed(target)}</span>
        </span>
      </div>

      {sim.isError && <p className="text-[12px] text-up">{(sim.error as Error).message}</p>}

      {ghost && (
        <div className="border-t border-line pt-2 grid grid-cols-3 gap-x-2 gap-y-1">
          {([["MACD", ghost.series.MACD], ["信号", ghost.series.MACD_Signal],
             ["RSI", ghost.series.RSI], ["ADX", ghost.series.ADX],
             ["+DI", ghost.series.DI_Plus], ["−DI", ghost.series.DI_Minus],
             ["MA5", ghost.series.MA5], ["MA20", ghost.series.MA20],
             ["价格Z", ghost.series.Price_Z]] as const).map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-1">
              <span className="text-[11px] text-ink-mute">{k}</span>
              <span className="font-mono tnum text-[12px]">{fixed(v, 2)}</span>
            </div>
          ))}
          {!ghost.ohl_supplied && (
            <p className="col-span-3 text-[11px] text-brand-ink">O/H/L 为估算值</p>
          )}
        </div>
      )}

      <div className="border-t border-line pt-2">
        <button onClick={() => aiCall.mutate()} disabled={aiCall.isPending}
          className="w-full h-8 rounded-lg bg-violet text-white text-[13px] font-semibold disabled:opacity-60">
          {aiCall.isPending ? "推演中…（20–60 秒）" : "🤖 尾盘推演"}
        </button>
        <p className="mt-1 text-[11px] text-ink-mute">
          读取{ghost ? "上面这根幻影K线" : "最后一个真实交易日"}。技术推演，非投资建议。
        </p>
        {aiCall.isError && (
          <p className="text-[12px] text-up mt-1">{(aiCall.error as Error).message}</p>
        )}
      </div>

      {ai && <AiRead ai={ai} />}
    </section>
  );
}

/**
 * Defined at module level, NOT inside WhatIfPanel. A component declared inside
 * another is a new type on every render, so React unmounts and remounts it —
 * which drops focus after every keystroke and leaves stale DOM values behind.
 */
function NumField({ label, value, set, after, step = 0.01 }: {
  label: string;
  value: number;
  set: (v: number) => void;
  after: () => void;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[11px] text-ink-mute">{label}</span>
      <input
        type="number" step={step} value={Number.isFinite(value) ? value : 0}
        onChange={(e) => {
          set(parseFloat(e.target.value));
          after();
        }}
        className="bg-sunken rounded-md h-7 px-1.5 text-[12.5px] font-mono tnum outline-none focus:ring-2 focus:ring-cyan/40"
      />
    </label>
  );
}

function AiRead({ ai }: { ai: WhatIfAi }) {
  const r = ai.read;
  const call = r.stance?.call ?? "";
  const tone = /买入|加仓/.test(call) ? "text-up" : /减仓|清仓/.test(call) ? "text-down" : "text-ink";
  return (
    <div className="border-t border-line pt-2 flex flex-col gap-1.5 text-[12px] leading-snug">
      <div className="flex items-baseline gap-2">
        <span className={`font-semibold ${tone}`}>{call || "—"}</span>
        <span className="text-[11px] text-ink-mute">信心 {r.stance?.conviction ?? "—"}</span>
        <span className="text-[11px] text-ink-mute ml-auto">
          {ai.mode === "ghost" ? "幻影" : "真实"} {ai.bar_date}
        </span>
      </div>
      {r.headline && <p className="font-medium">{r.headline}</p>}
      {ai.crossings.length > 0 && (
        <details>
          <summary className="cursor-pointer text-ink-mute">
            程序算出的状态变化 · {ai.crossings.length} 项
          </summary>
          <ul className="mt-1 space-y-0.5">
            {ai.crossings.map((c, i) => (
              <li key={i}>{c.dir === "up" ? "🔺" : "🔻"} {c.what} — {c.detail}</li>
            ))}
          </ul>
        </details>
      )}
      {r.bar_read && <p className="text-ink-dim">{r.bar_read}</p>}
      {r.levels?.confirm && <p>✅ {r.levels.confirm}</p>}
      {r.levels?.invalidate && <p>❌ {r.levels.invalidate}</p>}
      {r.next_session_plan && <p className="text-ink-dim">📋 {r.next_session_plan}</p>}
      {r.caveats?.length ? (
        <details>
          <summary className="cursor-pointer text-ink-mute">风险提示</summary>
          <ul className="mt-1 space-y-0.5">{r.caveats.map((c, i) => <li key={i}>· {c}</li>)}</ul>
        </details>
      ) : null}
    </div>
  );
}
