/**
 * 波动与反弹 — if the market is near a bottom, do the bigger swingers win
 * back more, and what do they cost on the way down?
 *
 * Four answers, narrowest last: which side each stock's swings are on (up-day
 * vs down-day multiples), what a market move of your choosing would roughly
 * do to each, and what actually happened after every real drawdown in six
 * years — measured from the exact low (hindsight) or from a confirmed bounce.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { Swing } from "../lib/types";
import { fixed, signed } from "../lib/format";
import { usePersistentState } from "../lib/usePersistentState";
import { NEG, POS } from "../lib/seesaw";
import { scenario } from "../lib/swing";

type Entry = "trough" | "confirm";

export function SwingPanel({ symbols, window, ready }: { symbols: string[]; window: string; ready: boolean }) {
  const [thr, setThr] = usePersistentState<number>("assrs.swing.thr", 8);
  const [entry, setEntry] = usePersistentState<Entry>("assrs.swing.entry", "trough");
  const q = useQuery({
    queryKey: ["swing", symbols.join(","), window, thr, entry],
    queryFn: () => api.swing(symbols, window, thr, entry),
    enabled: ready && symbols.length >= 2,
    staleTime: 20 * 60_000,
  });

  return (
    <section className="card p-3 flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-[14px] font-semibold">🎯 波动与反弹</h2>
        <span className="label">波动大的，是涨得多还是跌得多？大盘回撤后谁收复得快？</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Seg value={thr} set={setThr}
            options={[5, 8, 10, 15].map((v) => ({ v, label: `回撤≥${v}%` }))} />
          <Seg value={entry} set={setEntry}
            options={[
              { v: "trough" as Entry, label: "最低点买入", title: "事后才知道哪天是最低点 —— 这是最好的情况" },
              { v: "confirm" as Entry, label: "确认反弹后买入", title: "大盘离低点反弹 3% 的那天买入 —— 当时就能做到" },
            ]} />
        </div>
      </div>
      {q.isPending && <p className="label py-6 text-center">{ready ? "正在读取六年历史并识别回撤…" : "等待排名数据…"}</p>}
      {q.isError && (
        <p className="text-[12.5px] text-up">
          {(q.error as Error).message}
          <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
        </p>
      )}
      {q.data && <Body d={q.data} />}
    </section>
  );
}

function Seg<T extends string | number>({ value, set, options }: {
  value: T; set: (v: T) => void; options: { v: T; label: string; title?: string }[];
}) {
  return (
    <div className="flex rounded-md bg-sunken p-0.5">
      {options.map((o) => (
        <button key={String(o.v)} title={o.title} onClick={() => set(o.v)}
          className={`h-6 px-2.5 rounded text-[12px] ${
            value === o.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Body({ d }: { d: Swing }) {
  const name = (s: string) => d.labels[s] ?? s;
  return (
    <>
      <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] font-semibold">结论</span>
          {d.now && (
            <span className={`px-2 py-0.5 rounded-full text-[11.5px] font-semibold ${
              d.now.in_drawdown ? "bg-brand/15 text-brand-ink" : "bg-panel text-ink-mute"}`}>
              {d.benchmark} 距一年高点 {signed(d.now.dd_from_high_pct, 1, "%")}
              {d.now.in_drawdown ? " · 回撤中" : ""}
            </span>
          )}
          <span className="label">波动统计 {d.from} → {d.to} · 回撤记录自 {d.history_from}</span>
        </div>
        <ul className="flex flex-col gap-1 list-disc pl-4">
          {d.conclusions.map((c, i) => <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>)}
        </ul>
      </div>

      <div className="grid gap-3 xl:grid-cols-[300px_minmax(0,1fr)] items-start">
        <Asymmetry d={d} name={name} />
        <SwingTable d={d} name={name} />
      </div>
      <Scenario d={d} name={name} />
      <Episodes d={d} name={name} />

      <p className="label leading-snug">
        上涨日 / 下跌日倍数 = 大盘上涨（下跌）那些天，个股平均涨跌 ÷ 大盘平均涨跌，与上方排名表的上/下行捕获相同。
        回撤按摆动高低点识别：从高点跌够阈值才算一次，从低点反弹够阈值才确定低点，所以长期熊市里的每一波下跌都单独计算。
        个股的「跌」是大盘高点到大盘低点这段时间的涨跌。历史记录，非预测；底部只有事后才能确认。
      </p>
    </>
  );
}

// ── which side the swings are on ─────────────────────────────────────────────
function Asymmetry({ d, name }: { d: Swing; name: (s: string) => string }) {
  const pts = d.symbols
    .map((s) => ({ s, x: d.stats[s]?.beta_down, y: d.stats[s]?.beta_up }))
    .filter((p): p is { s: string; x: number; y: number } => p.x != null && p.y != null);
  if (!pts.length) return <p className="label">指数数据不可用，算不出涨跌倍数。</p>;
  const S = 280;
  const PAD = 26;
  const hi = Math.max(1.2, ...pts.flatMap((p) => [p.x, p.y])) * 1.08;
  const lo = Math.min(0, ...pts.flatMap((p) => [p.x, p.y]));
  const m = (v: number) => PAD + ((v - lo) / (hi - lo)) * (S - 2 * PAD);
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] font-semibold">涨跌不对称</span>
      <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-[300px] h-auto block" role="img"
        aria-label="上涨日倍数对下跌日倍数">
        <polygon points={`${m(lo)},${S - m(lo)} ${m(lo)},${S - m(hi)} ${m(hi)},${S - m(hi)}`} fill={POS} opacity={0.06} />
        <polygon points={`${m(lo)},${S - m(lo)} ${m(hi)},${S - m(lo)} ${m(hi)},${S - m(hi)}`} fill={NEG} opacity={0.06} />
        <line x1={m(lo)} y1={S - m(lo)} x2={m(hi)} y2={S - m(hi)} stroke="var(--color-line-bright)" strokeDasharray="4 4" />
        {[1].filter((t) => t > lo && t < hi).map((t) => (
          <g key={t}>
            <line x1={m(t)} x2={m(t)} y1={PAD} y2={S - PAD} stroke="var(--color-line)" />
            <line x1={PAD} x2={S - PAD} y1={S - m(t)} y2={S - m(t)} stroke="var(--color-line)" />
            <text x={m(t)} y={S - 8} textAnchor="middle" fontSize={10} fill="var(--color-ink-mute)">大盘 1×</text>
          </g>
        ))}
        <text x={PAD + 4} y={PAD + 10} fontSize={10} fill={POS}>涨多跌少 ↖</text>
        <text x={S - PAD - 4} y={S - PAD - 6} textAnchor="end" fontSize={10} fill={NEG}>↘ 跌多涨少</text>
        <text x={S / 2} y={S - 1} textAnchor="middle" fontSize={10} fill="var(--color-ink-mute)">下跌日倍数 →</text>
        <text x={8} y={S / 2} fontSize={10} fill="var(--color-ink-mute)" transform={`rotate(-90 8 ${S / 2})`} textAnchor="middle">上涨日倍数 →</text>
        {placeLabels(pts.map((p) => ({ s: p.s, cx: m(p.x), cy: S - m(p.y), text: name(p.s) })), S).map((p) => (
          <g key={p.s}>
            {p.moved && <line x1={p.cx} y1={p.cy} x2={p.lx - 2} y2={p.ly - 4} stroke="var(--color-ink-mute)" strokeWidth={0.75} />}
            <circle cx={p.cx} cy={p.cy} r={4.5} fill="var(--color-ink)" stroke="var(--color-panel)" strokeWidth={2} />
            <text x={p.lx} y={p.ly} fontSize={11} fill="var(--color-ink)">{p.text}</text>
          </g>
        ))}
      </svg>
      <span className="label">对角线 = 涨跌跟随一样多。线上方的，大盘涨时放大得比跌时多。</span>
    </div>
  );
}

/**
 * Direct labels that do not sit on each other: placed top to bottom, each
 * pushed down until it clears the ones already placed. Stocks in one basket
 * often share a multiple to the first decimal, so collisions are the norm.
 */
function placeLabels(pts: { s: string; cx: number; cy: number; text: string }[], S: number) {
  const width = (t: string) => [...t].reduce((w, ch) => w + (ch.charCodeAt(0) > 255 ? 11 : 6.5), 0);
  const placed: { s: string; cx: number; cy: number; text: string; lx: number; ly: number; moved: boolean }[] = [];
  for (const p of [...pts].sort((a, b) => a.cy - b.cy)) {
    const w = width(p.text);
    const lx = Math.min(p.cx + 7, S - w - 2);
    let ly = p.cy + 4;
    const clash = () => placed.some((q) =>
      Math.abs(q.ly - ly) < 14 && lx < q.lx + width(q.text) + 2 && q.lx < lx + w + 2);
    while (clash()) ly += 14;
    placed.push({ ...p, lx, ly, moved: ly !== p.cy + 4 });
  }
  return placed;
}

function SwingTable({ d, name }: { d: Swing; name: (s: string) => string }) {
  const rows = [...d.symbols].sort((a, b) => (d.stats[b]?.vol_annual_pct ?? 0) - (d.stats[a]?.vol_annual_pct ?? 0));
  const maxVol = Math.max(1, ...rows.map((s) => d.stats[s]?.vol_annual_pct ?? 0));
  const mk = d.stats.__market__;
  const asymColour = (v: number | null | undefined) => v == null ? undefined : v > 0.15 ? POS : v < -0.15 ? NEG : undefined;
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="text-[13px] font-semibold">波动对比 <span className="label font-normal">按年化波动排序</span></span>
      <div className="overflow-x-auto">
        <table className="w-full text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal pr-2">股票</th>
              <th className="font-normal pr-2 min-w-[150px]">年化波动</th>
              <th className="font-normal pr-2 text-right" title="每天平均涨跌幅（不分方向）">日均振幅</th>
              <th className="font-normal pr-2 text-right">β</th>
              <th className="font-normal pr-2 text-right" title="大盘上涨的日子，个股平均涨幅是大盘的几倍">上涨日倍数</th>
              <th className="font-normal pr-2 text-right" title="大盘下跌的日子，个股平均跌幅是大盘的几倍">下跌日倍数</th>
              <th className="font-normal text-right" title="上涨日倍数 − 下跌日倍数。正 = 波动偏向上涨一侧">不对称</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => {
              const st = d.stats[s]!;
              return (
                <tr key={s} className="border-t border-line">
                  <td className="py-1 pr-2 whitespace-nowrap">{name(s)}</td>
                  <td className="py-1 pr-2">
                    <div className="flex items-center gap-1.5">
                      <div className="h-2 rounded-full bg-ink-dim" style={{ width: `${((st.vol_annual_pct ?? 0) / maxVol) * 100}px` }} />
                      <span>{fixed(st.vol_annual_pct, 0)}%</span>
                    </div>
                  </td>
                  <td className="py-1 pr-2 text-right">{fixed(st.avg_abs_daily_pct, 2)}%</td>
                  <td className="py-1 pr-2 text-right">{fixed(st.beta, 2)}</td>
                  <td className="py-1 pr-2 text-right">{fixed(st.beta_up, 2)}×</td>
                  <td className="py-1 pr-2 text-right">{fixed(st.beta_down, 2)}×</td>
                  <td className="py-1 text-right font-semibold" style={{ color: asymColour(st.asymmetry) }}>{signed(st.asymmetry, 2)}</td>
                </tr>
              );
            })}
            {mk && (
              <tr className="border-t-2 border-line text-ink-mute">
                <td className="py-1 pr-2">{d.benchmark}</td>
                <td className="py-1 pr-2">
                  <div className="flex items-center gap-1.5">
                    <div className="h-2 rounded-full bg-line-bright" style={{ width: `${((mk.vol_annual_pct ?? 0) / maxVol) * 100}px` }} />
                    <span>{fixed(mk.vol_annual_pct, 0)}%</span>
                  </div>
                </td>
                <td className="py-1 pr-2 text-right">{fixed(mk.avg_abs_daily_pct, 2)}%</td>
                <td className="py-1 pr-2 text-right">1.00</td>
                <td className="py-1 pr-2 text-right">1.00×</td>
                <td className="py-1 pr-2 text-right">1.00×</td>
                <td />
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── what a market move would do ──────────────────────────────────────────────
function Scenario({ d, name }: { d: Swing; name: (s: string) => string }) {
  const [move, setMove] = useState(10);
  const [days, setDays] = useState(20);
  const rows = scenario(d, move, days);
  if (!rows.length) return null;
  const ext = Math.max(5, ...rows.map((r) => Math.abs(r.mid) + r.band));
  const pos = (v: number) => 50 + (v / ext) * 50;
  const rise = d.up_is_red ? "var(--color-up)" : "var(--color-down)";
  const fall = d.up_is_red ? "var(--color-down)" : "var(--color-up)";
  return (
    <div className="rounded-lg bg-sunken p-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[13px] font-semibold">如果 {d.benchmark} 接下来</span>
        <input type="range" min={-20} max={30} step={1} value={move}
          onChange={(e) => setMove(Number(e.target.value))} className="w-48 accent-[var(--color-cyan)]" />
        <b className="font-mono tnum text-[13px] w-12" style={{ color: move >= 0 ? rise : fall }}>{signed(move, 0, "%")}</b>
        <span className="label">用时</span>
        <Seg value={days} set={setDays} options={[{ v: 20, label: "约 1 个月" }, { v: 60, label: "约 3 个月" }]} />
      </div>
      <div className="flex flex-col gap-1">
        {rows.map((r) => (
          <div key={r.s} className="grid grid-cols-[90px_minmax(0,1fr)_120px] items-center gap-2 text-[12.5px]">
            <span className="truncate">{r.s === "__market__" ? d.benchmark : name(r.s)}</span>
            <div className="relative h-4">
              <div className="absolute top-0 bottom-0 w-px bg-line-bright" style={{ left: "50%" }} />
              <div className="absolute top-[7px] h-[2px] bg-ink-mute/40"
                style={{ left: `${pos(r.mid - r.band)}%`, width: `${pos(r.mid + r.band) - pos(r.mid - r.band)}%` }} />
              <div className="absolute top-[3px] h-[10px] rounded-sm"
                style={{ left: `${Math.min(50, pos(r.mid))}%`, width: `${Math.abs(pos(r.mid) - 50)}%`,
                         background: r.mid >= 0 ? rise : fall, opacity: r.s === "__market__" ? 0.45 : 0.85 }} />
            </div>
            <span className="tnum text-right">
              <b>{signed(r.mid, 1, "%")}</b>
              {r.band > 0 && <span className="label"> ±{r.band.toFixed(0)}</span>}
            </span>
          </div>
        ))}
      </div>
      <span className="label leading-snug">
        中心 = {move >= 0 ? "上涨日" : "下跌日"}倍数 × 大盘涨跌；灰线 = 个股自身（与大盘无关）波动在这段时间里的一个标准差，
        大约三分之二的情况落在里面。线性近似，用来比较谁放大得多，不是预测。
      </span>
    </div>
  );
}

// ── what actually happened ───────────────────────────────────────────────────
function Episodes({ d, name }: { d: Swing; name: (s: string) => string }) {
  const k = d.entry === "confirm" ? "reb60_confirm" : "reb60";
  const k20 = d.entry === "confirm" ? "reb20_confirm" : "reb20";
  const eps = [...d.episodes].reverse();
  const tone = (v: number | null | undefined) =>
    v == null ? undefined : { color: v > 0 === d.up_is_red ? "var(--color-up)" : "var(--color-down)" };
  if (!eps.length) {
    return <p className="label">{d.history_from} 以来 {d.benchmark} 没有 ≥{d.threshold}% 的回撤。</p>;
  }
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[13px] font-semibold">
        历次回撤 <span className="label font-normal">
          {d.benchmark} 跌 ≥{d.threshold}% · 每格：下跌段 / {d.entry === "confirm" ? "确认反弹后" : "最低点起"} 60 个交易日 · 粗体 = 跑赢大盘
        </span>
      </span>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-[12px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal pr-2">高点 → 低点</th>
              <th className="font-normal pr-2 text-right">{d.benchmark}</th>
              {d.symbols.map((s) => <th key={s} className="font-normal pr-2 text-right whitespace-nowrap">{name(s)}</th>)}
            </tr>
          </thead>
          <tbody>
            {eps.map((e) => (
              <tr key={e.peak} className="border-t border-line align-top">
                <td className="py-1 pr-2 font-mono whitespace-nowrap">
                  {e.peak} → {e.trough}
                  {e.open && <span className="label font-sans"> 进行中</span>}
                  {d.entry === "confirm" && e.confirm && <div className="label font-sans">确认 {e.confirm}</div>}
                </td>
                <td className="py-1 pr-2 text-right text-ink-mute">
                  <div style={tone(e.market.fall)}>{signed(e.market.fall, 1, "%")}</div>
                  <div style={tone(e.market[k])} title={`20 日 ${signed(e.market[k20], 1, "%")}`}>{signed(e.market[k], 1, "%")}</div>
                </td>
                {d.symbols.map((s) => {
                  const m = e.stocks[s];
                  const beat = m?.[k] != null && e.market[k] != null && m[k]! > e.market[k]!;
                  return (
                    <td key={s} className="py-1 pr-2 text-right">
                      <div style={tone(m?.fall)}>{signed(m?.fall, 1, "%")}</div>
                      <div style={tone(m?.[k])} className={beat ? "font-bold" : ""}
                        title={`20 日 ${signed(m?.[k20], 1, "%")}`}>{signed(m?.[k], 1, "%")}</div>
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr className="border-t-2 border-line">
              <td className="py-1 pr-2 label">平均 · 跑赢次数</td>
              <td />
              {d.symbols.map((s) => {
                const sm = d.summary[s];
                return (
                  <td key={s} className="py-1 pr-2 text-right">
                    {sm ? (
                      <>
                        <div style={tone(sm.avg_fall)}>{signed(sm.avg_fall, 1, "%")}</div>
                        <div style={tone(sm.avg_rebound)} className="font-semibold">{signed(sm.avg_rebound, 1, "%")}</div>
                        <div className="label">{sm.beat}/{sm.n} 次跑赢</div>
                        {sm.ratio != null && <div className="label" title="平均反弹 ÷ 平均下跌">反弹/下跌 {sm.ratio.toFixed(2)}</div>}
                      </>
                    ) : <span className="label">—</span>}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
