/**
 * 板块接力 · 背离 — build a chain from your 板块, upstream to downstream, and
 * ask it two questions:
 *
 *   接力  when one leg has a big day (big for it, market removed), does
 *         another move more than its norm over the next 1/3/5 days — the
 *         same way, or the opposite way?
 *   背离  when an upstream leg rolls over while a downstream leg is still at
 *         its highs, what did the downstream leg do next — against the same
 *         strength without the upstream weakness?
 *
 * The chain is built by dragging stocks (or a whole 板块) from the left into
 * lanes; clicking a stock adds it to the selected lane, for touch screens.
 * See chain_rotation.py for the statistics and the checks against chance.
 */

import { useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { ChainCell, ChainResult } from "../lib/types";
import { fixed, signed } from "../lib/format";
import { usePersistentState } from "../lib/usePersistentState";
import { NEG, POS, corrFill } from "../lib/seesaw";
import {
  LEG_COLOURS, MAX_LEGS, MAX_PER_LEG, addLeg, addToLeg, emptyChain, moveLeg, problem,
  removeLeg, removeTicker, renameLeg, toRequest, type Chain,
} from "../lib/chain";

const LOOKBACKS = [{ v: 252, label: "1年" }, { v: 504, label: "2年" }, { v: 756, label: "3年" }];
const KS = [1.5, 2, 2.5];
const DRAG = "application/x-assrs-stocks";

type Saved = { chains: Chain[]; active: number };
type Req = { legs: { name: string; tickers: string[] }[]; lookback: number; k: number };

export function ChainRotation() {
  const [saved, setSaved] = usePersistentState<Saved>("assrs.chain.v1", { chains: [emptyChain()], active: 0 });
  const [lookback, setLookback] = usePersistentState<number>("assrs.chain.lookback", 504);
  const [k, setK] = usePersistentState<number>("assrs.chain.k", 2);
  const [req, setReq] = useState<Req | null>(null);

  const idx = Math.min(saved.active, saved.chains.length - 1);
  const chain = saved.chains[idx] ?? emptyChain();
  const setChain = (c: Chain) =>
    setSaved({ ...saved, chains: saved.chains.map((x, i) => (i === idx ? c : x)) });

  const members = useQuery({ queryKey: ["sector-members"], queryFn: api.sectorMembers, staleTime: 3600_000 });
  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const names = useMemo(() => new Map((stocks.data ?? []).map((s) => [s.t, s.n])), [stocks.data]);
  const nameOf = (t: string) => names.get(t) ?? t;

  // The result lives in the query cache keyed by the request, so leaving the
  // tab and coming back does not throw away a 15-second analysis.
  const result = useQuery({
    queryKey: ["chain", JSON.stringify(req)],
    queryFn: () => api.chain(req!),
    enabled: req != null,
    staleTime: 30 * 60_000,
    retry: false,
  });

  const issue = problem(chain);

  return (
    <div className="flex flex-col gap-3">
      <section className="card p-3 flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-[14px] font-semibold">🔁 板块接力 · 背离</h2>
          <span className="label">仅 A 股 · 从左边把股票或整个板块拖进环节，上游在左、下游在右</span>
        </div>
        <ChainPicker saved={saved} setSaved={setSaved} />
        <div className="grid gap-3 lg:grid-cols-[260px_minmax(0,1fr)] items-start">
          <SectorList members={members.data} loading={members.isPending} nameOf={nameOf}
            onAdd={(tickers, sector) => {
              const target = chain.legs.find((l) => l.id === selectedLeg(chain)) ?? chain.legs[0];
              if (target) setChain(addToLeg(chain, target.id, tickers, sector));
            }} />
          <Lanes chain={chain} setChain={setChain} nameOf={nameOf} />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Seg value={lookback} set={setLookback} options={LOOKBACKS} />
          <Seg value={k} set={setK} options={KS.map((v) => ({ v, label: `大涨跌 ≥${v}σ` }))} />
          <button
            onClick={() => setReq({ legs: toRequest(chain), lookback, k })}
            disabled={!!issue || result.isFetching}
            className="h-8 px-4 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-50">
            {result.isFetching ? "分析中…（约 10–20 秒）" : "分析这条链"}
          </button>
          <span className="label">{issue ?? `${toRequest(chain).length} 个环节，${toRequest(chain).reduce((n, l) => n + l.tickers.length, 0)} 只股票`}</span>
        </div>
      </section>

      {result.isError && (
        <div className="card p-4 text-[12.5px] text-up">
          {(result.error as Error).message}
          <button onClick={() => result.refetch()} className="ml-2 text-cyan">重试</button>
        </div>
      )}
      {result.data && <Results d={result.data} nameOf={nameOf} />}
    </div>
  );
}

// ── building the chain ───────────────────────────────────────────────────────
const SELECTED_KEY = "assrs.chain.selectedLeg";
function selectedLeg(chain: Chain): string | undefined {
  try {
    const id = localStorage.getItem(SELECTED_KEY);
    return chain.legs.some((l) => l.id === id) ? id ?? undefined : chain.legs[0]?.id;
  } catch {
    return chain.legs[0]?.id;
  }
}

function ChainPicker({ saved, setSaved }: { saved: Saved; setSaved: (s: Saved) => void }) {
  const idx = Math.min(saved.active, saved.chains.length - 1);
  const chain = saved.chains[idx]!;
  return (
    <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
      <span className="label">产业链</span>
      <select value={idx} onChange={(e) => setSaved({ ...saved, active: Number(e.target.value) })}
        className="h-7 px-1.5 rounded-md bg-sunken outline-none">
        {saved.chains.map((c, i) => <option key={i} value={i}>{c.name || `链 ${i + 1}`}</option>)}
      </select>
      <input value={chain.name} maxLength={24}
        onChange={(e) => setSaved({ ...saved, chains: saved.chains.map((c, i) => i === idx ? { ...c, name: e.target.value } : c) })}
        className="h-7 w-40 px-2 rounded-md bg-sunken outline-none focus:ring-2 focus:ring-cyan/40" />
      <button onClick={() => setSaved({ chains: [...saved.chains, emptyChain(`链 ${saved.chains.length + 1}`)], active: saved.chains.length })}
        className="h-7 px-2 rounded-md bg-sunken hover:bg-elevated">＋ 新建</button>
      {saved.chains.length > 1 && (
        <button onClick={() => {
          if (!globalThis.confirm(`删除「${chain.name}」？`)) return;
          const chains = saved.chains.filter((_, i) => i !== idx);
          setSaved({ chains, active: Math.max(0, idx - 1) });
        }} className="h-7 px-2 rounded-md text-ink-mute hover:text-up">删除</button>
      )}
      <span className="label">保存在本浏览器</span>
    </div>
  );
}

function SectorList({ members, loading, nameOf, onAdd }: {
  members: Record<string, string[]> | undefined; loading: boolean;
  nameOf: (t: string) => string; onAdd: (tickers: string[], sector: string) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const list = Object.entries(members ?? {}).filter(([s, ts]) =>
    !q || s.includes(q) || ts.some((t) => t.includes(q) || nameOf(t).includes(q)));
  const drag = (tickers: string[], sector: string) => (e: React.DragEvent) => {
    e.dataTransfer.setData(DRAG, JSON.stringify({ tickers, sector }));
    e.dataTransfer.effectAllowed = "copyMove";
  };
  return (
    <div className="rounded-lg bg-sunken p-2 flex flex-col gap-1.5 max-h-[420px] overflow-y-auto">
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="搜索板块或股票"
        className="h-7 px-2 rounded-md bg-panel text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40" />
      {loading && <span className="label p-2">读取板块…</span>}
      {list.map(([sector, tickers]) => {
        const shown = open === sector || !!q;
        return (
          <div key={sector} className="rounded-md bg-panel">
            <div draggable onDragStart={drag(tickers, sector)}
              className="flex items-center gap-1.5 px-2 h-8 cursor-grab active:cursor-grabbing"
              title="拖到右边的环节里，整个板块一起加入">
              <button onClick={() => setOpen(open === sector ? null : sector)} className="text-ink-mute w-4">
                {shown ? "▾" : "▸"}
              </button>
              <span className="text-[12.5px] font-medium truncate">{sector}</span>
              <span className="label">{tickers.length}</span>
              <button onClick={() => onAdd(tickers, sector)} title="整个板块加入选中的环节"
                className="ml-auto text-[11.5px] text-cyan hover:underline">全部加入</button>
            </div>
            {shown && (
              <div className="flex flex-wrap gap-1 px-2 pb-2">
                {tickers.filter((t) => !q || sector.includes(q) || t.includes(q) || nameOf(t).includes(q)).map((t) => (
                  <button key={t} draggable onDragStart={drag([t], sector)} onClick={() => onAdd([t], sector)}
                    title={`${t} · 拖进环节，或点一下加入选中的环节`}
                    className="h-6 px-1.5 rounded bg-sunken hover:bg-elevated text-[12px] cursor-grab">
                    {nameOf(t)}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Lanes({ chain, setChain, nameOf }: {
  chain: Chain; setChain: (c: Chain) => void; nameOf: (t: string) => string;
}) {
  const [sel, setSel] = useState<string | undefined>(() => selectedLeg(chain));
  const [over, setOver] = useState<string | null>(null);
  const pick = (id: string) => {
    setSel(id);
    try { localStorage.setItem(SELECTED_KEY, id); } catch { /* private mode */ }
  };
  return (
    <div className="flex gap-2 overflow-x-auto pb-1 items-stretch">
      {chain.legs.map((leg, i) => {
        const active = (sel ?? chain.legs[0]?.id) === leg.id;
        return (
          <div key={leg.id} className="flex items-stretch">
            <div onClick={() => pick(leg.id)}
              onDragOver={(e) => { if (e.dataTransfer.types.includes(DRAG)) { e.preventDefault(); setOver(leg.id); } }}
              onDragLeave={() => setOver(null)}
              onDrop={(e) => {
                e.preventDefault();
                setOver(null);
                try {
                  const { tickers, sector } = JSON.parse(e.dataTransfer.getData(DRAG));
                  setChain(addToLeg(chain, leg.id, tickers, sector));
                  pick(leg.id);
                } catch { /* not ours */ }
              }}
              className={`w-[190px] shrink-0 rounded-lg border-2 p-2 flex flex-col gap-1.5 min-h-[180px] ${
                over === leg.id ? "border-cyan bg-cyan/5" : active ? "border-ink/30" : "border-dashed border-line"}`}>
              <div className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: LEG_COLOURS[i % LEG_COLOURS.length] }} />
                <input value={leg.name} placeholder={`环节 ${i + 1}`} maxLength={20}
                  onChange={(e) => setChain(renameLeg(chain, leg.id, e.target.value))}
                  onClick={(e) => e.stopPropagation()}
                  className="min-w-0 flex-1 h-6 px-1 rounded bg-transparent text-[12.5px] font-semibold outline-none focus:bg-sunken" />
                <button onClick={(e) => { e.stopPropagation(); setChain(moveLeg(chain, leg.id, -1)); }}
                  disabled={i === 0} className="text-ink-mute disabled:opacity-30 px-0.5" title="往上游移">‹</button>
                <button onClick={(e) => { e.stopPropagation(); setChain(moveLeg(chain, leg.id, 1)); }}
                  disabled={i === chain.legs.length - 1} className="text-ink-mute disabled:opacity-30 px-0.5" title="往下游移">›</button>
                <button onClick={(e) => { e.stopPropagation(); setChain(removeLeg(chain, leg.id)); }}
                  className="text-ink-mute hover:text-up px-0.5" title="删除这个环节">✕</button>
              </div>
              <span className="label">{i === 0 ? "最上游" : i === chain.legs.length - 1 ? "最下游" : `第 ${i + 1} 环`} · {leg.tickers.length}/{MAX_PER_LEG}</span>
              <div className="flex flex-wrap gap-1 content-start">
                {leg.tickers.map((t) => (
                  <span key={t} draggable
                    onDragStart={(e) => e.dataTransfer.setData(DRAG, JSON.stringify({ tickers: [t] }))}
                    className="h-6 pl-1.5 pr-0.5 rounded bg-sunken text-[12px] flex items-center gap-0.5 cursor-grab"
                    title={t}>
                    {nameOf(t)}
                    <button onClick={(e) => { e.stopPropagation(); setChain(removeTicker(chain, leg.id, t)); }}
                      className="text-ink-mute hover:text-up px-0.5">✕</button>
                  </span>
                ))}
                {!leg.tickers.length && <span className="label py-4 text-center w-full">拖到这里</span>}
              </div>
            </div>
            {i < chain.legs.length - 1 && <span className="self-center px-1 text-ink-mute">→</span>}
          </div>
        );
      })}
      {chain.legs.length < MAX_LEGS && (
        <button onClick={() => setChain(addLeg(chain))}
          className="w-[90px] shrink-0 rounded-lg border-2 border-dashed border-line text-ink-mute hover:text-ink text-[12.5px]">
          ＋ 环节
        </button>
      )}
    </div>
  );
}

function Seg<T extends string | number>({ value, set, options }: {
  value: T; set: (v: T) => void; options: { v: T; label: string }[];
}) {
  return (
    <div className="flex rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={String(o.v)} onClick={() => set(o.v)}
          className={`h-6 px-2.5 rounded text-[12px] ${value === o.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ── results ──────────────────────────────────────────────────────────────────
function Results({ d, nameOf }: { d: ChainResult; nameOf: (t: string) => string }) {
  const [view, setView] = usePersistentState<"relay" | "div">("assrs.chain.view", "relay");
  return (
    <section className="card p-3 flex flex-col gap-3">
      {d.live.length > 0 && (
        <div className="rounded-lg bg-brand/10 p-2.5 flex flex-col gap-1">
          {d.live.map((x, i) => (
            <span key={i} className="text-[12.5px] text-brand-ink">
              <b>{x.date} {x.leg}</b> {x.direction === "up" ? "大涨" : "大跌"} {signed(x.z, 1, "σ")}（扣除大盘 {signed(x.pct, 1, "%")}）
              {x.followers.length
                ? " · 历史上之后：" + x.followers.slice(0, 3).map((f) =>
                  `${f.follower} ${f.horizon} 天内${(f.excess ?? 0) > 0 ? "同向" : "反向"} ${fixed(Math.abs(f.excess ?? 0), 2)}σ（${f.n} 次）`).join("；")
                : " · 历史上链上没有超出随机的跟随"}
            </span>
          ))}
        </div>
      )}
      <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1.5">
        <span className="text-[13px] font-semibold">结论 <span className="label font-normal">{d.dates[0]} → {d.dates[d.dates.length - 1]}</span></span>
        <ul className="flex flex-col gap-1 list-disc pl-4">
          {d.conclusions.map((c, i) => <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>)}
        </ul>
        {d.missing.length > 0 && <span className="label">没有行情、未计入：{d.missing.map(nameOf).join("、")}</span>}
      </div>
      <Seg value={view} set={setView} options={[{ v: "relay" as const, label: "① 接力" }, { v: "div" as const, label: "② 背离" }]} />
      {view === "relay" ? <Relay d={d} /> : <Divergence d={d} />}
    </section>
  );
}

function Relay({ d }: { d: ChainResult }) {
  const [H, setH] = usePersistentState<string>("assrs.chain.h", "3");
  const cells = d.relay[H] ?? [];
  const legs = d.legs.map((l) => l.name);
  const [pick, setPick] = useState<[string, string] | null>(null);
  const best = useMemo(() => [...cells].filter((c) => c.q != null).sort((a, b) => (a.q ?? 1) - (b.q ?? 1))[0], [cells]);
  const sel = cells.find((c) => pick && c.leader === pick[0] && c.follower === pick[1]) ?? best;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-semibold">谁大涨大跌之后，谁跟</span>
        <Seg value={H} set={setH} options={["1", "3", "5"].map((v) => ({ v, label: `之后 ${v} 天` }))} />
      </div>
      <div className="grid gap-3 xl:grid-cols-[minmax(0,auto)_minmax(0,1fr)] items-start">
        <div className="overflow-x-auto">
          <table className="text-[12px] tnum border-separate border-spacing-[3px]">
            <thead>
              <tr>
                <th className="label font-normal text-left pr-1">先动 ↓ / 之后 →</th>
                {legs.map((f) => <th key={f} className="label font-normal px-1 max-w-[80px] truncate" title={f}>{f}</th>)}
              </tr>
            </thead>
            <tbody>
              {legs.map((L) => (
                <tr key={L}>
                  <td className="label pr-1 whitespace-nowrap max-w-[110px] truncate" title={L}>{L}</td>
                  {legs.map((F) => {
                    if (F === L) return <td key={F} className="w-[72px] h-11 rounded bg-sunken" />;
                    const c = cells.find((x) => x.leader === L && x.follower === F);
                    const strong = c?.q != null && c.q <= 0.1;
                    const maybe = c?.q != null && c.q <= 0.2;
                    const on = sel && sel.leader === L && sel.follower === F;
                    return (
                      <td key={F} onClick={() => setPick([L, F])}
                        title={c ? `${L} 大涨大跌后 ${H} 天，${F}：${signed(c.excess, 2, "σ")}（${c.n} 次，跟随率 ${Math.round((c.follow_rate ?? 0) * 100)}%，q=${fixed(c.q, 3)}）` : ""}
                        className={`w-[72px] h-11 rounded text-center cursor-pointer leading-tight ${on ? "ring-2 ring-ink" : ""}`}
                        style={{ background: maybe ? corrFill(Math.max(-1, Math.min(1, c?.excess ?? 0))) : "var(--color-panel-alt)",
                                 color: maybe ? (Math.abs(c?.excess ?? 0) > 0.55 ? "#fff" : "var(--color-ink)") : "var(--color-ink-mute)" }}>
                        <div className="font-semibold">{signed(c?.excess, 2)}{strong ? "★" : ""}</div>
                        <div className="text-[10.5px] opacity-80">{c?.n ?? 0} 次</div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex flex-wrap items-center gap-3 label mt-1">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm" style={{ background: POS }} />同向跟随</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm" style={{ background: NEG }} />反向（此消彼长）</span>
            <span>灰 = 与随机日无异 · ★ = 校正后仍显著</span>
          </div>
        </div>
        {sel && <CellDetail c={sel} H={Number(H)} />}
      </div>
      <p className="label leading-snug">
        「大涨大跌」= 该环节扣除大盘后的当日涨跌超过自身过去 60 日波动的 {d.k} 倍；连续几天的只算第一天。
        「之后」= 跟随方之后 {H} 天扣除大盘的累计涨跌，以它自己的日波动为单位，减去它平常任意 {H} 天的表现，按领先方的方向计正负。
        显著性：与同样数量的随机日比较 2,000 次，并按页面上所有格子一起做多重检验校正。
      </p>
    </div>
  );
}

function CellDetail({ c, H }: { c: ChainCell; H: number }) {
  const W = 320, Ht = 140;
  const ev = c.curve ?? [];
  const base = c.base_curve ?? [];
  const pts = [0, ...ev.map((v) => v ?? 0)];
  const bpts = [0, ...base.map((v) => v ?? 0)];
  const ext = Math.max(0.5, ...pts.map(Math.abs), ...bpts.map(Math.abs)) * 1.15;
  const x = (i: number) => 20 + (i / H) * (W - 30);
  const y = (v: number) => Ht / 2 - (v / ext) * (Ht / 2 - 10);
  const line = (vs: number[]) => vs.map((v, i) => `${i ? "L" : "M"}${x(i)} ${y(v)}`).join("");
  const colour = (c.excess ?? 0) >= 0 ? POS : NEG;
  return (
    <div className="flex flex-col gap-2 min-w-0">
      <div className="flex flex-wrap items-baseline gap-2 text-[13px]">
        <span className="font-semibold">{c.leader} → {c.follower}</span>
        <b className="tnum" style={{ color: colour }}>{signed(c.excess, 2, "σ")}</b>
        <span className="label">{c.n} 次 · 跟随率 {Math.round((c.follow_rate ?? 0) * 100)}% · q={fixed(c.q, 3)}</span>
      </div>
      <div className="flex flex-wrap gap-3 text-[12px]">
        <span>大涨后 <b className="tnum">{signed(c.up, 2, "σ")}</b> <span className="label">({c.n_up})</span></span>
        <span>大跌后 <b className="tnum">{signed(c.down, 2, "σ")}</b> <span className="label">({c.n_down})</span></span>
        <span>前半段 <b className="tnum">{signed(c.first, 2)}</b> / 后半段 <b className="tnum">{signed(c.second, 2)}</b>
          <span className="label"> {c.stable ? "· 两段一致" : "· 两段不一致"}</span></span>
      </div>
      <svg viewBox={`0 0 ${W} ${Ht}`} className="w-full max-w-[360px] h-auto" role="img"
        aria-label={`${c.leader} 大涨大跌后 ${c.follower} 的平均路径`}>
        <line x1={20} x2={W - 10} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" />
        <path d={line(bpts)} fill="none" stroke="var(--color-ink-mute)" strokeDasharray="4 4" strokeWidth={1.4} />
        <path d={line(pts)} fill="none" stroke={colour} strokeWidth={2.2} />
        {pts.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r={2.5} fill={colour} />)}
        {Array.from({ length: H + 1 }, (_, i) => (
          <text key={i} x={x(i)} y={Ht - 2} textAnchor="middle" fontSize={10} fill="var(--color-ink-mute)">D{i ? `+${i}` : "0"}</text>
        ))}
        <text x={22} y={12} fontSize={10} fill="var(--color-ink-mute)">按领先方方向累计（σ）· 虚线 = 平常</text>
      </svg>
      <div className="max-h-[220px] overflow-y-auto">
        <table className="w-full text-[12px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal">日期</th>
              <th className="font-normal text-right">{c.leader}</th>
              <th className="font-normal text-right">{c.follower} {H} 天</th>
              <th className="font-normal text-right">跟随</th>
            </tr>
          </thead>
          <tbody>
            {[...c.events].reverse().map((e) => (
              <tr key={e.date} className="border-t border-line">
                <td className="py-0.5 font-mono">{e.date}</td>
                <td className="py-0.5 text-right">{signed(e.leader_z, 1, "σ")} <span className="label">{signed(e.leader_pct, 1, "%")}</span></td>
                <td className="py-0.5 text-right">{signed(e.end, 2, "σ")}</td>
                <td className="py-0.5 text-right" style={{ color: e.followed ? POS : NEG }}>{e.followed ? "✓" : "✗"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Divergence({ d }: { d: ChainResult }) {
  const pairs = d.divergence;
  const first = pairs.find((p) => p.now.diverging) ?? pairs.find((p) => p.episodes.length) ?? pairs[0];
  const [pk, setPk] = useState<string | null>(null);
  const sel = pairs.find((p) => `${p.up}→${p.down}` === pk) ?? first;
  if (!sel) return <p className="label">至少需要两个环节。</p>;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-1.5">
        {pairs.map((p) => {
          const key = `${p.up}→${p.down}`;
          const on = sel === p;
          return (
            <button key={key} onClick={() => setPk(key)}
              className={`h-7 px-2 rounded-md text-[12px] border ${on ? "border-ink bg-panel font-semibold" : "border-line bg-sunken"}`}>
              {p.up} → {p.down}
              {p.now.diverging
                ? <span className="ml-1 text-brand-ink">· 进行中 {p.now.days} 天</span>
                : <span className="ml-1 label">· {p.episodes.length} 次</span>}
            </button>
          );
        })}
      </div>
      <LegChart d={d} mask={sel.mask} up={sel.up} down={sel.down} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-[12px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal">背离开始</th><th className="font-normal text-right">持续</th>
              <th className="font-normal text-right">{sel.down} 几天后见顶</th>
              <th className="font-normal text-right">到顶再涨</th>
              <th className="font-normal text-right">见顶后回落</th>
              <th className="font-normal text-right">开始后 20 天</th>
            </tr>
          </thead>
          <tbody>
            {[...sel.episodes].reverse().map((e) => (
              <tr key={e.start} className="border-t border-line">
                <td className="py-1 font-mono">{e.start}{e.open && <span className="label font-sans"> 进行中</span>}</td>
                <td className="py-1 text-right">{e.days} 天</td>
                <td className="py-1 text-right">{e.days_to_peak ?? "—"}{e.complete === false && e.days_to_peak != null ? "+" : ""}</td>
                <td className="py-1 text-right">{signed(e.peak_gain_pct, 1, "%")}</td>
                <td className="py-1 text-right">{signed(e.fall_after_peak_pct, 1, "%")}</td>
                <td className="py-1 text-right font-semibold">{signed(e.fwd20_pct, 1, "%")}</td>
              </tr>
            ))}
            {!sel.episodes.length && <tr><td colSpan={6} className="py-2 label">这段历史里没有出现过这对背离。</td></tr>}
            <tr className="border-t-2 border-line">
              <td className="py-1 label" colSpan={5}>
                对照：{sel.down} 同样在高位、但 {sel.up} 没走弱的日子（{sel.control.n_days} 天）之后 20 天平均
              </td>
              <td className="py-1 text-right font-semibold">{signed(sel.control.fwd20_mean, 1, "%")}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        上游「走弱」= 低于自身 20 日高点 5% 以上，且低于 10 天前；下游「高位」= 在自身 60 日高点 3% 以内。都只用当天以前的数据判断，
        所以历史上的每一次都是当时就能看到的。橙色底 = 所选这对正在背离的日子。
      </p>
    </div>
  );
}

function LegChart({ d, mask, up, down }: { d: ChainResult; mask: boolean[]; up: string; down: string }) {
  const W = 1000, H = 240, BH = 60;
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const names = d.legs.map((l) => l.name);
  const n = d.dates.length;
  const all = [...names.flatMap((nm) => d.index[nm] ?? []), ...d.bench].filter((v): v is number => v != null);
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const pad = (hi - lo) * 0.06 || 1;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => H - ((v - (lo - pad)) / (hi + pad - (lo - pad))) * H;
  const path = (vs: (number | null)[]) => {
    let s = "", pen = false;
    vs.forEach((v, i) => { if (v == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; pen = true; });
    return s;
  };
  const bands: [number, number][] = [];
  mask.forEach((m, i) => {
    if (!m) return;
    const last = bands[bands.length - 1];
    if (last && last[1] === i - 1) last[1] = i; else bands.push([i, i]);
  });
  const by = (v: number) => BH - (v / 100) * BH;
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        {names.map((nm, i) => (
          <span key={nm} className={`flex items-center gap-1 ${nm === up || nm === down ? "font-semibold" : ""}`}>
            <span className="w-4 h-[3px] rounded-full" style={{ background: LEG_COLOURS[i % LEG_COLOURS.length] }} />{nm}
            <span className="tnum">{signed((d.index[nm] ?? [])[n - 1], 1, "%")}</span>
          </span>
        ))}
        <span className="flex items-center gap-1 text-ink-mute"><span className="w-4 border-t-2 border-dashed border-ink-mute" />沪深300 {signed(d.bench[n - 1], 1, "%")}</span>
      </div>
      <div className="pr-16">
      <div ref={box} className="relative h-[240px]"
        onMouseMove={(e) => {
          const r = box.current?.getBoundingClientRect();
          if (r) setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))));
        }} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img"
          aria-label="各环节累计涨跌，橙色为背离期">
          {bands.map(([a, b]) => (
            <rect key={a} x={x(a)} width={Math.max(2, x(b) - x(a))} y={0} height={H} fill={NEG} opacity={0.12} />
          ))}
          <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
          <path d={path(d.bench)} fill="none" stroke="var(--color-ink-mute)" strokeDasharray="5 4" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
          {names.map((nm, i) => (
            <path key={nm} d={path(d.index[nm] ?? [])} fill="none" stroke={LEG_COLOURS[i % LEG_COLOURS.length]}
              strokeWidth={nm === up || nm === down ? 2.4 : 1.2} opacity={nm === up || nm === down ? 1 : 0.55}
              vectorEffect="non-scaling-stroke" />
          ))}
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
        </svg>
        {/* Direct labels at the line ends: identity never rests on colour alone. */}
        {endLabels(names.map((nm, i) => ({ nm, i, v: (d.index[nm] ?? [])[n - 1] ?? null }))
          .filter((l): l is { nm: string; i: number; v: number } => l.v != null)
          .map((l) => ({ ...l, px: (y(l.v) / H) * 240 }))).map((l) => (
          <span key={l.nm} className="absolute left-full pl-1 text-[10.5px] leading-[12px] whitespace-nowrap max-w-[64px] truncate"
            style={{ top: l.px, transform: "translateY(-50%)", color: LEG_COLOURS[l.i % LEG_COLOURS.length] }}>{l.nm}</span>
        ))}
        {hover != null && (
          <div className="absolute top-1 card px-2 py-1 text-[12px] pointer-events-none z-10"
            style={hover > n * 0.6 ? { right: `${100 - (hover / Math.max(1, n - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, n - 1)) * 100 + 1}%` }}>
            <div className="font-mono">{d.dates[hover]}{mask[hover] ? " · 背离" : ""}</div>
            {names.map((nm, i) => (
              <div key={nm} className="flex gap-2"><span style={{ color: LEG_COLOURS[i % LEG_COLOURS.length] }}>{nm}</span>
                <span className="ml-auto tnum">{signed((d.index[nm] ?? [])[hover], 1, "%")}</span></div>
            ))}
            <div className="text-ink-mute">宽度 {fixed(d.breadth[hover], 0)}%</div>
          </div>
        )}
      </div>
      </div>
      <div className="flex items-center gap-2 pr-16">
        <span className="label w-[86px] shrink-0">20日线上方占比</span>
        <svg viewBox={`0 0 ${W} ${BH}`} preserveAspectRatio="none" className="flex-1 h-[60px]" role="img" aria-label="全链宽度">
          <line x1={0} x2={W} y1={by(50)} y2={by(50)} stroke="var(--color-line)" strokeDasharray="3 4" vectorEffect="non-scaling-stroke" />
          <path d={d.breadth.map((v, i) => v == null ? "" : `${i && d.breadth[i - 1] != null ? "L" : "M"}${x(i).toFixed(1)} ${by(v).toFixed(1)}`).join("")}
            fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.3} vectorEffect="non-scaling-stroke" />
        </svg>
      </div>
      <div className="flex justify-between label font-mono tnum"><span>{d.dates[0]}</span><span>{d.dates[n - 1]}</span></div>
    </div>
  );
}

/**
 * Line-end labels nudged apart: legs often finish within a few percent of
 * each other, and two names printed on one spot identify neither.
 */
function endLabels<T extends { px: number }>(labels: T[], gap = 14, height = 240): T[] {
  const out = [...labels].sort((a, b) => a.px - b.px).map((l) => ({ ...l }));
  for (let i = 1; i < out.length; i++) {
    if (out[i]!.px < out[i - 1]!.px + gap) out[i]!.px = out[i - 1]!.px + gap;
  }
  const over = out.length ? out[out.length - 1]!.px - (height - 6) : 0;
  if (over > 0) out.forEach((l) => { l.px -= over; });
  return out;
}
