/**
 * 领先滞后 — which of these stocks moves first, and whether that is tradeable.
 *
 * Three different questions per pair, and they are worth keeping apart:
 *
 *   Granger     does the other stock's PAST help predict this one's future,
 *               beyond its own past? Direction.
 *   Lag grid    at which lag is the co-movement strongest? Shape — and the
 *               sanity check on the above, because a "significant" lead with
 *               a correlation of 0.04 is a number, not a relationship.
 *   协整 + 半衰期 do the prices actually travel together, and how long does a
 *               gap take to close? Whether it can be traded at all.
 *
 * A pair routinely passes the first and fails the last. A real one-day lead
 * worth half a percent, closing over eleven days, is not a trade after costs.
 *
 * The number this screen exists to show honestly is q, not p. Testing ten
 * peers in both directions is twenty tests, and twenty tests at a 5% threshold
 * throw up one pass from noise on average — enough to put a spurious
 * "relationship" on screen most runs. So the raw count is shown NEXT TO how
 * many to expect from noise, and rows that do not survive the correction are
 * dimmed rather than hidden: seeing that eleven of twelve failed is the
 * finding.
 *
 * The peer set can be picked by hand, or found: 从自选股中搜索 below sweeps the
 * whole watchlist. That sweep is NOT the same screen run wider — searching
 * 3,160 pairs at a 5% threshold returns hundreds of "relationships" from
 * noise — so it screens on one half of the history and confirms on the other.
 * See Discover, and pair_scan.py for why that shape.
 */

import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { LeadLagResult, LeadLagRow, Num, StockRef } from "../lib/types";
import { useSymbolSearch } from "../lib/useSymbolSearch";
import { usePersistentState } from "../lib/usePersistentState";
import { DiscoverInputs, DiscoverResults, useDiscover } from "./Discover";
import { LeadLagHistory } from "./LeadLagHistory";
import { fixed } from "../lib/format";

const MAX_PEERS = 15;
const LOOKBACKS = [90, 180, 252, 504];

export function LeadLag() {
  const [subject, setSubject] = usePersistentState<StockRef | null>("assrs.ll.t", null);
  const [peers, setPeers] = usePersistentState<StockRef[]>("assrs.ll.peers", []);
  const [lookback, setLookback] = usePersistentState<number>("assrs.ll.days", 180);
  const [maxLag, setMaxLag] = usePersistentState<number>("assrs.ll.lag", 5);

  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks,
    staleTime: 6 * 3600_000 });
  // The search's state lives here so its controls can sit beside this
  // panel's while its results sit below both.
  const discover = useDiscover("lead-lag");
  const run = useMutation({
    mutationFn: () => api.leadLag({
      ticker: subject!.t, peers: peers.map((p) => p.t),
      lookback_days: lookback, max_lag: maxLag,
    }),
  });

  const ready = Boolean(subject) && peers.length > 0;

  return (
    <>
      {/* Inputs on one row, outputs underneath. Two half-width panels rather
          than two full-width ones stacked: the controls are short and the
          results are what needs the room. */}
      <div className="grid gap-3 lg:grid-cols-2 items-start">
      <section className="card p-3 flex flex-col gap-2.5">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="text-[14px] font-semibold">🕰️ 领先滞后</h2>
          <span className="label">仅 A 股 · 一只主股，对比 1–{MAX_PEERS} 只</span>
          <div className="ml-auto flex items-center gap-2">
            <label className="label flex items-center gap-1" title="用多少个交易日的数据做检验">
              回看
              <select value={lookback} onChange={(e) => setLookback(Number(e.target.value))}
                className="h-7 px-1 rounded-md bg-sunken text-[12.5px] tnum outline-none">
                {LOOKBACKS.map((d) => <option key={d} value={d}>{d} 天</option>)}
              </select>
            </label>
            <label className="label flex items-center gap-1" title="最多检验到几天的滞后">
              最大滞后
              <input type="number" value={maxLag} min={1} max={10}
                onChange={(e) => setMaxLag(Number(e.target.value))}
                className="w-14 h-7 px-1.5 rounded-md bg-sunken text-[12.5px] font-mono tnum outline-none" />
            </label>
          </div>
        </div>

        <Picker label="主股 T" single stocks={stocks.data ?? []}
          picked={subject ? [subject] : []} exclude={peers.map((p) => p.t)}
          onAdd={(s) => setSubject(s)} onRemove={() => setSubject(null)} />

        <Picker label={`对比股 S · ${peers.length}/${MAX_PEERS}`} stocks={stocks.data ?? []}
          picked={peers} exclude={subject ? [subject.t] : []}
          max={MAX_PEERS}
          onAdd={(s) => setPeers([...peers, s])}
          onRemove={(t) => setPeers(peers.filter((p) => p.t !== t))} />

        <div className="flex items-center gap-2">
          <button onClick={() => run.mutate()} disabled={!ready || run.isPending}
            className="h-8 px-4 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
            {run.isPending ? "检验中…" : "运行检验"}
          </button>
          <span className="label">
            {ready
              ? `将进行 ${peers.length * 2} 次格兰杰检验（双向），并做多重检验校正`
              : "先选一只主股和至少一只对比股"}
          </span>
        </div>
        {run.isError && (
          <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
        )}
      </section>

        <DiscoverInputs d={discover} />
      </div>

      <DiscoverResults d={discover} onUse={(a, b, [na, nb]) => {
        setSubject({ t: a, n: na });
        setPeers([{ t: b, n: nb }]);
        globalThis.scrollTo({ top: 0, behavior: "smooth" });
      }} />

      {run.data && (
        <section className="card p-3 flex flex-col gap-3">
          <Results data={run.data} />
          {/* The verdict above is a summary of a history; this is the
              history. Same box, because they are two readings of one test. */}
          {subject && peers[0] && (
            <LeadLagHistory a={subject.t} b={peers[0].t}
              nameA={subject.n} nameB={peers[0].n} />
          )}
        </section>
      )}
    </>
  );
}

function Picker({ label, stocks, picked, exclude, onAdd, onRemove, single, max }: {
  label: string; stocks: StockRef[]; picked: StockRef[]; exclude: string[];
  onAdd: (s: StockRef) => void; onRemove: (t: string) => void;
  single?: boolean; max?: number;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const { items } = useSymbolSearch({ query: q, stocks, markets: ["CN"], limit: 8 });

  const taken = new Set([...picked.map((p) => p.t), ...exclude]);
  const hits = items.filter((s) => !taken.has(s.t));
  const full = single ? picked.length >= 1 : max != null && picked.length >= max;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="label w-[110px] shrink-0">{label}</span>
      {picked.map((s) => (
        <span key={s.t}
          className="h-7 pl-2 pr-1 rounded-md bg-sunken flex items-center gap-1.5 text-[12.5px]">
          <span className="truncate max-w-[130px]">{s.n}</span>
          <span className="font-mono tnum text-[11px] text-ink-mute">{s.t}</span>
          <button onClick={() => onRemove(s.t)} aria-label={`移除 ${s.n}`}
            className="text-ink-mute hover:text-ink px-1">✕</button>
        </span>
      ))}
      {!full && (
        <div className="relative">
          <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            onBlur={() => globalThis.setTimeout(() => setOpen(false), 150)}
            placeholder={picked.length ? "再加一只…" : "输入代码或名称…"}
            className="h-7 w-48 px-2 rounded-md bg-sunken text-[12.5px] outline-none focus:ring-2 focus:ring-cyan/40" />
          {open && hits.length > 0 && (
            <div className="card absolute left-0 mt-1 z-30 w-64 py-1">
              {hits.map((s) => (
                <button key={s.t} onMouseDown={(e) => e.preventDefault()}
                  onClick={() => { onAdd({ t: s.t, n: s.n }); setQ(""); setOpen(false); }}
                  className="w-full flex items-baseline gap-2 px-2 py-1 text-left hover:bg-elevated">
                  <span className="font-mono tnum text-[12px] text-ink-dim shrink-0">{s.t}</span>
                  <span className="text-[12.5px] truncate">{s.n}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Rendered inside the results card, so no card of its own. */
function Results({ data }: { data: LeadLagResult }) {
  const t = data.tests;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-[13.5px] font-semibold">
          {data.name} <span className="font-mono text-ink-mute">{data.ticker}</span>
        </h3>
        <span className="label">
          {data.from} → {data.to} · {data.sessions} 个交易日 · 最大滞后 {data.max_lag} 天
        </span>
      </div>

      {/* The honest headline. A raw count of "significant" pairs is not a
          finding until you know how many to expect from nothing at all. */}
      <p className="text-[12px] leading-snug rounded-lg bg-sunken px-2.5 py-2">
        共 <b>{t.n}</b> 次检验（每只对比股双向各一次）。
        按 p&lt;{t.alpha} 有 <b>{t.raw_hits}</b> 个通过 —— 而<b>纯噪声下预计就有
        约 {fixed(t.expected_false, 1)} 个会通过</b>。
        经 {t.method} 多重检验校正后，真正站得住的有 <b className={
          t.survivors ? "text-up" : "text-ink-mute"}>{t.survivors}</b> 个。
        下表按 q 值排序，未通过校正的行已置灰 —— 看到十一个里只有一个站得住，本身就是结论。
      </p>

      <div className="overflow-auto rounded-lg border border-line">
        <table className="w-full border-collapse text-[12px]">
          <thead className="bg-panel">
            <tr className="border-b border-line text-ink-mute">
              <th className="text-left font-medium px-2 py-1.5">代码</th>
              <th className="text-left font-medium px-2 py-1.5">名称</th>
              <th className="text-left font-medium px-2 py-1.5">关系</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="格兰杰检验的原始 p 值，未校正">p</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="多重检验校正后的 q 值。只有 q<0.05 才值得看。">q</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="T 相对 S 的回归斜率">β</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="最强的滞后相关系数，以及出现在第几天">峰值相关</th>
              <th className="text-center font-medium px-2 py-1.5"
                title="价格是否长期同行（协整）">协整</th>
              <th className="text-right font-medium px-2 py-1.5"
                title="价差回到均值所需天数。太长就不是交易机会。">半衰期</th>
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => <Row key={r.ticker} r={r} />)}
          </tbody>
        </table>
      </div>

      <Heatmap data={data} />

      {data.missing.length > 0 && (
        <p className="text-[11.5px] text-brand-ink">
          ⚠ 没有可用行情，未纳入检验：
          {data.missing.map((m) => `${m.ticker} ${m.name}`).join("、")}
        </p>
      )}
    </div>
  );
}

function Row({ r }: { r: LeadLagRow }) {
  const dim = r.survives_fdr ? "" : "opacity-45";
  const lead = /T leads/.test(r.relationship) ? "T 领先"
    : /S leads/.test(r.relationship) ? "S 领先"
      : r.relationship === "Bidirectional" ? "双向" : "无关系";
  const days = /T leads/.test(r.relationship) ? r.lag_t_leads_s
    : /S leads/.test(r.relationship) ? r.lag_s_leads_t : null;

  return (
    <tr className={`border-b border-line/60 hover:bg-sunken ${dim}`}>
      <td className="px-2 py-1">
        <Link to={`/?t=${r.ticker}`} className="text-cyan font-mono font-semibold">
          {r.ticker}
        </Link>
      </td>
      <td className="px-2 py-1 max-w-[150px] truncate" title={r.name}>{r.name}</td>
      <td className="px-2 py-1 whitespace-nowrap">
        {lead}{days ? ` ${days} 天` : ""}
        {r.survives_fdr && <span className="text-up ml-1" title="通过多重检验校正">✓</span>}
      </td>
      <td className="px-2 py-1 text-right tnum text-ink-mute">
        {fixed(Math.min(r.p_t_leads_s ?? 1, r.p_s_leads_t ?? 1), 3)}
      </td>
      <td className={`px-2 py-1 text-right tnum font-medium ${
        r.survives_fdr ? "text-up" : ""}`}>
        {fixed(r.q_best, 3)}
      </td>
      <td className="px-2 py-1 text-right tnum">{fixed(r.beta, 2)}</td>
      <td className="px-2 py-1 text-right tnum">
        {fixed(r.peak_corr, 2)}
        <span className="text-ink-mute text-[11px]"> @{r.peak_lag > 0 ? "+" : ""}{r.peak_lag}d</span>
      </td>
      <td className="px-2 py-1 text-center">{r.cointegrated ? "✓" : "—"}</td>
      <td className="px-2 py-1 text-right tnum">
        {r.half_life == null ? "—" : `${fixed(r.half_life, 1)} 天`}
      </td>
    </tr>
  );
}

/**
 * The lag grid. Red where the other stock leads, blue where this one does —
 * and the columns are written out ("S 领先 2 天") rather than signed, because
 * a signed lag makes every reader re-derive which way round it goes.
 */
function Heatmap({ data }: { data: LeadLagResult }) {
  const peak = Math.max(
    0.2, ...data.rows.flatMap((r) => r.xcorr.map((v) => Math.abs(v ?? 0))));

  const shade = (v: Num) => {
    if (v == null) return "var(--color-panel-alt)";
    const k = Math.min(Math.abs(v) / peak, 1);
    const [r, g, b] = v >= 0 ? [215, 0, 21] : [0, 98, 204];
    const mix = (c: number) => Math.round(248 + (c - 248) * (0.08 + 0.92 * k));
    return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
  };

  return (
    <div className="flex flex-col gap-1">
      <span className="label">
        滞后相关热力图 · 红 = 正相关，蓝 = 负相关，颜色深浅按本次最大绝对值缩放。
        中间一列是同日相关；左侧是对比股先动，右侧是主股先动。
      </span>
      <div className="overflow-auto rounded-lg border border-line">
        <table className="border-collapse text-[11px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-panel px-2 py-1 text-left font-medium
                             text-ink-mute border-b border-line">对比股</th>
              {data.lag_labels.map((l, i) => (
                <th key={l} className={`px-1 py-1 font-normal text-ink-mute whitespace-nowrap
                  border-b border-line ${data.lags[i] === 0 ? "font-semibold text-ink" : ""}`}>
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((r) => (
              <tr key={r.ticker} className={r.survives_fdr ? "" : "opacity-45"}>
                <td className="sticky left-0 bg-panel px-2 py-0.5 whitespace-nowrap
                               border-r border-line">
                  <span className="font-mono">{r.ticker}</span>{" "}
                  <span className="text-ink-mute">{r.name}</span>
                </td>
                {r.xcorr.map((v, i) => (
                  <td key={i} className="text-center tnum px-1"
                    style={{ background: shade(v), minWidth: 52, height: 20 }}
                    title={`${data.lag_labels[i]} · ${fixed(v, 3)}`}>
                    {v == null ? "" : fixed(v, 2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
