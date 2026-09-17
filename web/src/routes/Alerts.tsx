/**
 * 今日提醒 — everything that fired on your watchlist last night, filterable.
 *
 * The Streamlit version was one long table you had to read top to bottom. The
 * question it could not answer is the one you actually have: "which of my
 * stocks are sitting on box support AND in a sector I like?" So the page is a
 * filter over the feed instead of a rendering of it —
 *
 *   * by signal, grouped by indicator family, because nineteen loose
 *     checkboxes is not a filter;
 *   * by sector, including 未归类 so watchlist stocks in no sector index are
 *     still reachable;
 *   * by 筹码 shape, which is a distribution the nightly job already scored;
 *   * by direction.
 *
 * Facet counts are computed over the WHOLE feed, not the filtered subset, so a
 * count never drops to zero in a way that hides the option that would bring
 * results back. Selecting several signals is OR within a group and AND across
 * groups — "any RSI signal, and also on a box edge".
 *
 * Picking a stock opens everything known about it today: each signal with the
 * numbers behind it, the box it is sitting in, its chip structure, and a link
 * through to the full analysis.
 *
 * Nothing here is computed on request. scan_watchlists.py did all of it in
 * GitHub Actions at 20:00 Beijing, which is the entire reason the page is
 * instant and did not need the free tier's CPU.
 */

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { AlertFeed, AlertSignal, AlertStock } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { AlertNotes } from "../components/AlertNotes";
import { fixed, moveClass, signed } from "../lib/format";

export function Alerts() {
  useEffect(() => { document.title = "ASSRS · 今日提醒"; }, []);
  const q = useQuery({ queryKey: ["alerts"], queryFn: () => api.alerts(), staleTime: 10 * 60_000 });

  const [signals, setSignals] = useState<Set<string>>(new Set());
  const [sectors, setSectors] = useState<Set<string>>(new Set());
  const [shapes, setShapes] = useState<Set<string>>(new Set());
  const [bias, setBias] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  const stocks = useMemo(
    () => (q.data ? filter(q.data.stocks, { signals, sectors, shapes, bias }) : []),
    [q.data, signals, sectors, shapes, bias]);

  const current = stocks.find((s) => s.t === picked) ?? stocks[0] ?? null;
  const active = signals.size + sectors.size + shapes.size + (bias ? 1 : 0);

  function clearAll() {
    setSignals(new Set());
    setSectors(new Set());
    setShapes(new Set());
    setBias(null);
  }

  return (
    <div className="min-h-screen">
      <NavBar>
        {q.data && (
          <span className="label truncate">
            数据日期 {q.data.scan_date}
            {q.data.stale && <b className="text-up"> · 已过期，夜间任务可能未运行</b>}
          </span>
        )}
      </NavBar>

      <main className="max-w-[1800px] mx-auto px-3 py-3">
        {q.isPending && <div className="card p-10 text-center label">正在读取昨夜扫描结果…</div>}
        {q.isError && (
          <div className="card p-6 text-center">
            <p className="text-up text-[13px]">{(q.error as Error).message}</p>
            <button onClick={() => q.refetch()} className="mt-2 text-cyan text-[13px]">重试</button>
          </div>
        )}

        {q.data && (
          <div className="grid gap-3 items-start lg:grid-cols-[240px_minmax(0,1fr)_minmax(0,380px)]">
            <Filters d={q.data} signals={signals} setSignals={setSignals}
              sectors={sectors} setSectors={setSectors}
              shapes={shapes} setShapes={setShapes}
              bias={bias} setBias={setBias} active={active} onClear={clearAll} />

            <List stocks={stocks} total={q.data.stocks.length}
              picked={current?.t ?? null} onPick={setPicked} />

            <Detail stock={current} scanDate={q.data.scan_date} />
          </div>
        )}
      </main>
    </div>
  );
}

type Sel = { signals: Set<string>; sectors: Set<string>; shapes: Set<string>; bias: string | null };

/** OR inside each facet, AND across facets. */
function filter(all: AlertStock[], sel: Sel): AlertStock[] {
  return all.filter((s) => {
    if (sel.bias && s.bias !== sel.bias) return false;
    if (sel.signals.size && !s.signals.some((x) => sel.signals.has(x.id))) return false;
    if (sel.shapes.size && !(s.chip_shape && sel.shapes.has(s.chip_shape))) return false;
    if (sel.sectors.size) {
      const mine = s.sectors.length ? s.sectors : ["未归类"];
      if (!mine.some((x) => sel.sectors.has(x))) return false;
    }
    return true;
  });
}

function toggle(set: Set<string>, id: string): Set<string> {
  const next = new Set(set);
  next.has(id) ? next.delete(id) : next.add(id);
  return next;
}

function Filters({ d, signals, setSignals, sectors, setSectors, shapes, setShapes,
                   bias, setBias, active, onClear }: {
  d: AlertFeed;
  signals: Set<string>; setSignals: (s: Set<string>) => void;
  sectors: Set<string>; setSectors: (s: Set<string>) => void;
  shapes: Set<string>; setShapes: (s: Set<string>) => void;
  bias: string | null; setBias: (b: string | null) => void;
  active: number; onClear: () => void;
}) {
  const byGroup = d.groups
    .map((g) => ({ group: g, items: d.facets.signals.filter((s) => s.group === g) }))
    .filter((g) => g.items.length > 0);

  return (
    <aside className="card p-3 flex flex-col gap-3 lg:sticky lg:top-[3.75rem] lg:max-h-[calc(100vh-4.5rem)] lg:overflow-y-auto">
      <div className="flex items-baseline justify-between">
        <h2 className="text-[14px] font-semibold">筛选</h2>
        {active > 0 && (
          <button onClick={onClear} className="text-[12px] text-cyan">清除 ({active})</button>
        )}
      </div>

      <div className="flex gap-1">
        {d.facets.bias.map((b) => (
          <button key={b.id} onClick={() => setBias(bias === b.id ? null : b.id)}
            className={`flex-1 h-7 rounded-md text-[12px] border transition-colors ${
              bias === b.id ? "border-cyan bg-cyan text-white" : "border-line bg-panel hover:bg-elevated"
            }`}>
            {b.id.slice(0, 2)} {b.count}
          </button>
        ))}
      </div>

      {byGroup.map(({ group, items }) => (
        <Section key={group} title={group}>
          {items.map((s) => (
            <Chip key={s.id} on={signals.has(s.id)} onClick={() => setSignals(toggle(signals, s.id))}
              dir={s.dir} label={s.cn} count={s.count} />
          ))}
        </Section>
      ))}

      {d.facets.shapes.length > 0 && (
        <Section title="筹码结构">
          {d.facets.shapes.map((s) => (
            <Chip key={s.name} on={shapes.has(s.name)} onClick={() => setShapes(toggle(shapes, s.name))}
              label={s.name} count={s.count} />
          ))}
        </Section>
      )}

      <Section title="板块">
        {d.facets.sectors.map((s) => (
          <Chip key={s.name} on={sectors.has(s.name)} onClick={() => setSectors(toggle(sectors, s.name))}
            label={s.name} count={s.count} />
        ))}
      </Section>
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{title}</span>
      <div className="flex flex-wrap gap-1">{children}</div>
    </div>
  );
}

function Chip({ on, onClick, label, count, dir }: {
  on: boolean; onClick: () => void; label: string; count: number; dir?: "bull" | "bear";
}) {
  const mark = dir === "bull" ? "▲" : dir === "bear" ? "▼" : "";
  return (
    <button onClick={onClick}
      className={`h-6 px-1.5 rounded-md text-[11.5px] border transition-colors ${
        on ? "border-cyan bg-cyan text-white" : "border-line bg-panel hover:bg-elevated"
      }`}>
      {mark && <span className={on ? "" : dir === "bull" ? "text-up" : "text-down"}>{mark}</span>}
      {label} <span className={on ? "opacity-80" : "text-ink-mute"}>{count}</span>
    </button>
  );
}

function List({ stocks, total, picked, onPick }: {
  stocks: AlertStock[]; total: number; picked: string | null; onPick: (t: string) => void;
}) {
  return (
    <div className="card p-2 flex flex-col gap-1">
      <div className="px-1 flex items-baseline justify-between">
        <span className="text-[13px] font-semibold">
          {stocks.length} 只{stocks.length !== total && <span className="label"> / 共 {total}</span>}
        </span>
        <span className="label">点击查看全部提醒</span>
      </div>

      {stocks.length === 0 && (
        <p className="py-10 text-center label">没有符合条件的股票 — 放宽筛选试试</p>
      )}

      {stocks.map((s) => (
        <button key={s.t} onClick={() => onPick(s.t)}
          className={`text-left rounded-lg px-2 py-1.5 border transition-colors ${
            picked === s.t ? "border-cyan bg-elevated" : "border-transparent hover:bg-elevated"
          }`}>
          <div className="flex items-baseline gap-2">
            <span className="text-[13px] font-medium truncate">{s.n}</span>
            <span className="font-mono tnum text-[11.5px] text-ink-mute">{s.t}</span>
            <span className="font-mono tnum text-[12.5px] ml-auto">¥{fixed(s.price)}</span>
            <span className="text-[11.5px]">{s.bias.slice(0, 2)}</span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-1">
            {s.signals.map((x, i) => (
              <span key={i} className={`text-[11px] ${x.dir === "bull" ? "text-up" : "text-down"}`}>
                {x.dir === "bull" ? "▲" : "▼"}{x.cn}
              </span>
            ))}
            {s.chip_shape && <span className="text-[11px] text-ink-mute">· {s.chip_shape}</span>}
            {s.sectors.length > 0 && (
              <span className="text-[11px] text-ink-mute ml-auto truncate">{s.sectors.join(" / ")}</span>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}

function Detail({ stock, scanDate }: { stock: AlertStock | null; scanDate: string }) {
  if (!stock) {
    return <div className="card p-6 text-center label">选择一只股票查看今天的全部提醒</div>;
  }
  const c = stock.chips;
  return (
    <div className="card p-3 flex flex-col gap-3 lg:sticky lg:top-[3.75rem] lg:max-h-[calc(100vh-4.5rem)] lg:overflow-y-auto">
      <div>
        <div className="flex items-baseline gap-2">
          <h2 className="text-[15px] font-semibold">{stock.n}</h2>
          <span className="font-mono tnum text-[12px] text-ink-mute">{stock.t}</span>
          <Link to={`/?t=${stock.t}`} className="ml-auto text-[12.5px] text-cyan">打开个股分析 →</Link>
        </div>
        <div className="mt-1 grid grid-cols-4 gap-2">
          <Stat k="价格" v={`¥${fixed(stock.price)}`} />
          <Stat k="RSI" v={fixed(stock.rsi, 1)}
            cls={(stock.rsi ?? 50) >= 70 ? "text-up" : (stock.rsi ?? 50) <= 30 ? "text-down" : ""} />
          <Stat k="ADX" v={fixed(stock.adx, 1)} />
          <Stat k="MACD" v={fixed(stock.macd, 3)} cls={moveClass(stock.macd)} />
        </div>
        {stock.sectors.length > 0 && (
          <p className="mt-1 label">板块：{stock.sectors.join(" · ")}</p>
        )}
      </div>

      <div className="border-t border-line pt-2 flex flex-col gap-2">
        <span className="label">今日信号 · {stock.signals.length} 项</span>
        {stock.signals.map((s, i) => <SignalCard key={i} s={s} price={stock.price ?? 0} />)}
      </div>

      <AlertNotes ticker={stock.t} name={stock.n} scanDate={scanDate} />

      {c && (
        <div className="border-t border-line pt-2 flex flex-col gap-1">
          <div className="flex items-baseline justify-between">
            <span className="label">筹码结构</span>
            <span className="text-[12px]">{c.setup_label ?? "—"} {fixed(c.setup_score, 2)}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[12px]">
            <Line k="获利盘" v={`${fixed((c.winner_rate ?? 0) * 100, 1)}%`} />
            <Line k="集中度" v={fixed(c.concentration, 3)} />
            <Line k="主峰价" v={`¥${fixed(c.peak_price)}`} />
            <Line k="距主峰" v={signed(c.pct_from_peak, 1, "%")} />
          </div>
          {c.converged === false && (
            <p className="text-[11px] text-brand-ink">换手不足，筹码数字仅供参考</p>
          )}
        </div>
      )}
    </div>
  );
}

function SignalCard({ s, price }: { s: AlertSignal; price: number }) {
  const tone = s.dir === "bull" ? "text-up" : "text-down";
  const d = s.detail;
  return (
    <div className="rounded-lg bg-sunken px-2 py-1.5">
      <div className="flex items-baseline gap-2">
        <span className={`text-[12.5px] font-medium ${tone}`}>
          {s.dir === "bull" ? "▲" : "▼"} {s.cn}
        </span>
        <span className="label ml-auto">{s.group}</span>
      </div>
      {d && (
        <>
          <div className="mt-1 relative h-5 rounded bg-panel border border-line">
            {/* Where price sits inside the box — the whole point of the alert. */}
            <div className="absolute inset-y-0 w-[2px] bg-cyan"
              style={{ left: `${Math.max(0, Math.min(100, d.position_pct))}%` }} />
            <div className="absolute inset-0 flex items-center justify-between px-1 text-[10.5px] font-mono tnum text-ink-mute">
              <span>{d.bot}</span><span>{d.top}</span>
            </div>
          </div>
          <p className="mt-0.5 text-[11px] text-ink-dim">
            高度 {d.height_pct}% · 位置 {d.position_pct}% · 触碰 上{d.touches_top}/下{d.touches_bot} ·
            质量 {d.quality}
            {price > 0 && d.top > price && (
              <span className="text-up"> · 到上沿还有 {(((d.top / price) - 1) * 100).toFixed(1)}%</span>
            )}
          </p>
        </>
      )}
    </div>
  );
}

function Stat({ k, v, cls = "" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10.5px] text-ink-mute">{k}</span>
      <span className={`font-mono tnum text-[13px] ${cls}`}>{v}</span>
    </div>
  );
}

function Line({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-ink-dim">{k}</span>
      <span className="font-mono tnum">{v}</span>
    </div>
  );
}
