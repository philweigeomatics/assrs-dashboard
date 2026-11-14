/**
 * 价值区间 — the price ranges a stock kept coming back to over five years.
 *
 * Read top to bottom: the summary (where 70% of the time went, where today's
 * price sits), the price line with each zone as a band and the time-at-price
 * profile beside it, the zones as a table, and the stays of the selected zone.
 *
 * Closed until opened: it is its own five-year read (from listing when the
 * stock is younger), separate from the chart's 3-year, regime-anchored frame.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { PriceZone, PriceZones, ZoneReport } from "../lib/types";
import { usePersistentState } from "../lib/usePersistentState";
import { useSize } from "../lib/useSize";

// Ranked colours: #1 is always indigo. None of them is red or green — those
// mean up and down on this page, and a zone is neither.
const ZONE_COLOURS = ["#6366f1", "#f59e0b", "#a855f7", "#0ea5e9", "#ec4899"];
const colourOf = (z: PriceZone) => ZONE_COLOURS[(z.rank - 1) % ZONE_COLOURS.length] ?? "#6366f1";
const NUMERALS = ["①", "②", "③", "④", "⑤"];
const tag = (z: PriceZone) => NUMERALS[z.rank - 1] ?? `#${z.rank}`;

const pct = (v: number, nd = 0) => `${(v * 100).toFixed(nd)}%`;
// One decimal under 10%: 4.9% growth must not read "+5%" beside "below 5%".
const signedPct = (v: number | null | undefined) =>
  (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(Math.abs(v) < 0.1 ? 1 : 0)}%`);

/**
 * Which multiple the zones are read in. PE is what the market paid for the
 * earnings; PEG is what it paid per point of the growth it could see — on
 * the last year's growth, or the three-year rate one good or bad year does
 * not swing.
 */
type Basis = "pe" | "peg1" | "peg3";
const BASES: { v: Basis; label: string; hint: string }[] = [
  { v: "pe", label: "PE", hint: "市盈率（TTM）" },
  { v: "peg1", label: "PEG · 1年", hint: "市盈率 ÷ 近一年利润增速（TTM 同比）" },
  { v: "peg3", label: "PEG · 3年", hint: "市盈率 ÷ 三年利润复合增速，不受单独一年的好坏左右" },
];
const growthOf = (b: Basis) => (b === "peg3" ? "g3" : "g1") as "g1" | "g3";
const fmtMult = (b: Basis, v: number | null) => (v == null ? "—" : b === "pe" ? `${v.toFixed(1)}×` : v.toFixed(2));
/** "1年" or "3年" — which growth a PEG basis reads. */
const span = (b: Basis) => (b === "peg3" ? "三年复合" : "近一年");
const yi = (v: number | null | undefined) => (v == null ? "—" : `${(v / 1e8).toFixed(2)}亿`);
const pct1 = (v: number | null | undefined) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);
/** The report in force on `date`: the latest one counted, effective on or before it. */
function reportOn(reports: ZoneReport[], date: string): ZoneReport | null {
  let best: ZoneReport | null = null;
  for (const r of reports) if (r.used && r.effective <= date && (!best || r.effective > best.effective)) best = r;
  return best;
}
const quarter = (e: string | null) => (e ? `${e.slice(0, 4)}${{ "0331": "Q1", "0630": "H1", "0930": "Q3", "1231": "年报" }[e.slice(4)] ?? e.slice(4)}` : "");

export function PriceZonesPanel({ ticker }: { ticker: string }) {
  const [open, setOpen] = usePersistentState("assrs.zones.open", false);
  const q = useQuery({
    queryKey: ["zones", ticker],
    queryFn: () => api.zones(ticker),
    enabled: open,
    staleTime: 3600_000,
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
  });

  return (
    <section className="card">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-elevated rounded-[10px]">
        <span className="text-[14px] font-semibold shrink-0">🧭 价值区间</span>
        <span className="label truncate">近5年最常停留的价格带 · 什么时候在那里 · 当时的估值</span>
        <span className="ml-auto label shrink-0">{open ? "收起 ▲" : "展开 ▼"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 flex flex-col gap-3 border-t border-line">
          {q.isPending && (
            <div className="py-10 flex flex-col items-center gap-2 text-ink-mute">
              <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
              <span className="label">读取近5年日线…</span>
            </div>
          )}
          {q.isError && (
            <p className="text-[12.5px] text-up py-4">
              {(q.error as Error).message}
              <button onClick={() => q.refetch()} className="ml-2 text-cyan">重试</button>
            </p>
          )}
          {q.data && <Body d={q.data} />}
        </div>
      )}
    </section>
  );
}

function Body({ d }: { d: PriceZones }) {
  const sym = d.currency_symbol;
  const money = (v: number | null | undefined) =>
    v == null ? "—" : `${sym}${v >= 1000 ? v.toFixed(0) : v.toFixed(2)}`;

  const initial = d.now.zone ?? d.zones.find((z) => z.rank === 1)?.id ?? null;
  const [sel, setSel] = useState<number | null>(initial);
  const selected = d.zones.find((z) => z.id === sel) ?? null;
  const byId = (id: number | null) => d.zones.find((z) => z.id === id) ?? null;

  const [basisPref, setBasis] = usePersistentState<Basis>("assrs.zones.basis", "pe");
  const basis: Basis = basisPref !== "pe" && !d.has_growth ? "pe" : basisPref;

  const inZone = byId(d.now.zone);
  const above = byId(d.now.above);
  const below = byId(d.now.below);
  const flat = d.settled < 0.15;

  return (
    <>
      <div className="pt-2 text-[13px] leading-relaxed flex flex-col gap-1">
        <p>
          <span className="label">
            {d.start} → {d.end}，{d.sessions.toLocaleString()} 个交易日
            {d.listed_late && `（${d.start} 才上市，从上市起算；前 ${d.ipo_skipped} 个交易日是新股定价期，不计入）`}
            ，前复权价
          </span>
        </p>
        <p>
          {pct(d.value_area.share)} 的时间在 <b>{money(d.value_area.lo)} – {money(d.value_area.hi)}</b> 之间，
          停留最多的价格是 <b>{money(d.value_area.poc)}</b>。
          {" "}现价 <b>{money(d.now.close)}</b>
          {inZone && <>，就在区间 <ZoneTag z={inZone} /> 里。</>}
          {!inZone && above && <>，比区间 <ZoneTag z={above} /> 的上沿高 {pct(d.now.close / above.hi - 1)}
            {below ? <>，离区间 <ZoneTag z={below} /> 的下沿还差 {pct(1 - d.now.close / below.lo)}。</> : "。"}</>}
          {!inZone && !above && below && <>，比区间 <ZoneTag z={below} /> 的下沿低 {pct(1 - d.now.close / below.lo)}。</>}
          {!inZone && !above && !below && "。"}
        </p>
        {d.has_pe && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <div className="flex rounded-md bg-sunken p-0.5" role="tablist" aria-label="估值口径">
              {BASES.map((b) => {
                const off = b.v !== "pe" && !d.has_growth;
                return (
                  <button key={b.v} disabled={off} onClick={() => setBasis(b.v)}
                    title={off ? "读不到这只股票的历史利润，算不了增速" : b.hint}
                    className={`h-6 px-2.5 rounded text-[12px] whitespace-nowrap disabled:opacity-40 ${
                      basis === b.v ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>{b.label}</button>
                );
              })}
            </div>
            <span className="label">
              现在 PE {fmtMult("pe", d.now.pe)}
              {d.has_growth && <>
                {" "}· 利润增速 近一年 {signedPct(d.now.g1)} / 三年复合 {signedPct(d.now.g3)}
                {d.now.growth_period && `（截至 ${quarter(d.now.growth_period)}）`}
                {" "}· PEG {fmtMult("peg1", d.now.peg1)} / {fmtMult("peg3", d.now.peg3)}
              </>}
            </span>
          </div>
        )}
        {d.has_pe && d.has_growth && basis !== "pe" && <Formula d={d} basis={basis} />}
        {flat && (
          <p className="text-brand-ink">
            这只股票这几年基本在单边走，没在哪个价位久待（停留最多的区间也只占 {pct(d.settled)} 的时间）——
            这种情况下“常待的价格”说明不了多少。
          </p>
        )}
      </div>

      <Chart d={d} selected={selected} money={money} />
      {d.valuation && <ValuationChart d={d} selected={selected} basis={basis} money={money} />}

      {d.zones.length > 0 ? (
        <ZoneTable d={d} sel={sel} setSel={setSel} money={money} basis={basis} />
      ) : (
        <p className="label">没有找到明显的停留区间。</p>
      )}

      {selected && <Stays z={selected} d={d} money={money} basis={basis} />}

      {d.reports.length > 0 && <Reports d={d} />}

      <details className="text-[12px] text-ink-mute">
        <summary className="cursor-pointer hover:text-ink">怎么读、能不能当“真实价值”</summary>
        <div className="mt-2 flex flex-col gap-1.5 leading-relaxed">
          <p>
            <b>怎么算的：</b>每个交易日按当天最高到最低价，平均摊到它碰过的每个价格格子里（对数刻度，¥10 和 ¥1000 的股票一样分格），
            叠起来就是右边的“时间分布”。分布里明显鼓起的峰就是一个区间，边界取峰高一半的位置。
            收盘价落在区间里的连续日子算一段“停留”，中间出去不超过 5 个交易日不算离开，少于 10 个交易日的算路过。
          </p>
          <p>
            <b>是不是真实价值：</b>它是市场<b>在那段时间</b>反复认可的价格，是共识，不是内在价值。
            公司盈利变了，同一个价格就是另一种估值——所以表里给了每个区间当时的市盈率中位数，
            以及“按现在的盈利折算”：如果市场今天还付当时那个市盈率，股价该是多少。盈利涨了很多的股票，旧区间按今天的盈利会高出一大截，
            那个折算价往往比旧价格更接近“市场认可的价值”。
          </p>
          <p>
            <b>PEG：</b>市盈率 ÷ 利润增速（百分数）。同样 30 倍市盈率，增速 5% 的公司很贵，增速 40% 的不算贵——PEG 把增长算进去。
            增速是当时<b>已经公布</b>的：每份财报从公告的第二天起才算数，用首次公布的数字，不用事后修订的。
            近一年 = 滚动 12 个月归母净利润同比；三年 = 滚动 12 个月利润的三年复合增速，单独一年的大起大落影响小。
            增速为负或低于 5% 时 PEG 不适用（1% 的增速会把 10 倍市盈率变成 PEG 10，没有意义），这些日子不计入区间的 PEG。
            “隐含增速”反过来读：今天的市盈率 ÷ 当时的 PEG = 如果市场还按那时给增长定价，现价在押利润每年涨多少。
            拿它和实际增速比——押得比实际高，按那时的标准现在偏贵；押得比实际低，偏便宜。
            它不需要今天的增速，所以利润在下滑、现在的 PEG 不适用时也照样能读。
            只有 A 股有：美股的免费数据只有最近五个季度的利润。
          </p>
          <p>
            <b>注意：</b>区间是事后统计，不保证价格会回去；近一年停留得多的区间更能代表现在的共识。
            横跨 2024-09-24 前后的区间，前后是两个不同的市场环境。
          </p>
        </div>
      </details>
    </>
  );
}

function ZoneTag({ z }: { z: PriceZone }) {
  return <b style={{ color: colourOf(z) }}>{tag(z)}</b>;
}

// ── chart ──────────────────────────────────────────────────────────────────
const H = 340;
const PAD = { l: 52, r: 10, t: 10, b: 24 };

function Chart({ d, selected, money }: {
  d: PriceZones; selected: PriceZone | null; money: (v: number | null) => string;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(size.w, 300);
  const histW = Math.min(170, Math.round(W * 0.24));
  const x0 = PAD.l, x1 = W - histW - PAD.r - 8;
  const hx0 = x1 + 8, hx1 = W - PAD.r;
  const y0 = PAD.t, y1 = H - PAD.b;

  const geo = useMemo(() => {
    const prices = d.close.filter((v): v is number => v != null);
    const lows = [...prices, ...d.zones.map((z) => z.lo)];
    const highs = [...prices, ...d.zones.map((z) => z.hi)];
    const lmin = Math.log(Math.min(...lows)) - 0.03;
    const lmax = Math.log(Math.max(...highs)) + 0.03;
    const n = d.close.length;
    const X = (i: number) => x0 + (n <= 1 ? 0 : (i / (n - 1)) * (x1 - x0));
    const Y = (p: number) => y1 - ((Math.log(p) - lmin) / (lmax - lmin)) * (y1 - y0);
    const path = (from: number, to: number) => {
      let s = "";
      for (let i = from; i <= to; i++) {
        const v = d.close[i];
        if (v == null) continue;
        s += `${s ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`;
      }
      return s;
    };
    const ticks = Array.from({ length: 5 }, (_, k) => Math.exp(lmin + ((k + 0.5) / 5) * (lmax - lmin)));
    const years: { i: number; y: string }[] = [];
    d.dates.forEach((t, i) => {
      const y = t.slice(0, 4);
      if (i === 0 || y !== d.dates[i - 1]!.slice(0, 4)) years.push({ i, y });
    });
    const idx = new Map(d.dates.map((t, i) => [t, i]));
    return { X, Y, path, ticks, years, idx, n };
  }, [d, x0, x1, y0, y1]);

  const { X, Y, path, ticks, years, idx, n } = geo;
  const tmax = Math.max(...d.profile.time.map((v) => v ?? 0), 1e-9);
  const vmax = Math.max(...d.profile.volume.map((v) => v ?? 0), 1e-9);
  const zoneAt = (p: number) => d.zones.find((z) => p >= z.lo && p <= z.hi) ?? null;
  const skip = d.ipo_skipped;

  const volPath = d.profile.price.map((p, i) => {
    const v = d.profile.volume[i] ?? 0;
    return p == null ? "" : `${i ? "L" : "M"}${(hx0 + (v / vmax) * (hx1 - hx0)).toFixed(1)},${Y(p).toFixed(1)}`;
  }).join("");

  const h = hover != null ? { i: hover, t: d.dates[hover], v: d.close[hover] } : null;

  return (
    <div ref={ref} className="w-full relative select-none">
      {size.w > 0 && (
        <svg width={W} height={H} className="block"
          onMouseLeave={() => setHover(null)}
          onMouseMove={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const mx = e.clientX - r.left;
            if (mx < x0 || mx > x1) return setHover(null);
            setHover(Math.round(((mx - x0) / (x1 - x0)) * (n - 1)));
          }}>
          {/* zone bands */}
          {d.zones.map((z) => (
            <g key={z.id}>
              <rect x={x0} width={hx1 - x0} y={Y(z.hi)} height={Math.max(1, Y(z.lo) - Y(z.hi))}
                fill={colourOf(z)} opacity={selected?.id === z.id ? 0.2 : 0.09} />
              <text x={x0 + 4} y={Y(z.hi) + 12} fontSize={11} fill={colourOf(z)} fontWeight={600}
                stroke="var(--color-panel)" strokeWidth={3} paintOrder="stroke">
                {tag(z)} {money(z.lo)}–{money(z.hi)}
              </text>
            </g>
          ))}
          {/* the selected zone's stays, as time spans */}
          {selected?.periods.map((p) => {
            const a = idx.get(p.from), b = idx.get(p.to);
            if (a == null || b == null) return null;
            return <rect key={p.from} x={X(a)} width={Math.max(1, X(b) - X(a))} y={y0} height={y1 - y0}
              fill={colourOf(selected)} opacity={0.08} />;
          })}
          {/* axes */}
          {ticks.map((p) => (
            <g key={p}>
              <line x1={x0} x2={x1} y1={Y(p)} y2={Y(p)} stroke="var(--color-line)" strokeDasharray="2 4" />
              <text x={x0 - 6} y={Y(p) + 4} fontSize={11} textAnchor="end" fill="var(--color-ink-mute)">
                {money(p)}
              </text>
            </g>
          ))}
          {/* A year that starts within a few weeks of the window's first bar
              would print on top of the next one. */}
          {years.filter(({ i }, k) => k === years.length - 1 || X(years[k + 1]!.i) - X(i) > 34).map(({ i, y }) => (
            <g key={y}>
              <line x1={X(i)} x2={X(i)} y1={y1} y2={y1 + 4} stroke="var(--color-line-bright)" />
              <text x={X(i) + 2} y={H - 6} fontSize={11} fill="var(--color-ink-mute)">{y}</text>
            </g>
          ))}
          {/* price: the IPO weeks faint, the rest solid, the selected stays in colour */}
          {skip > 0 && <path d={path(0, skip)} fill="none" stroke="var(--color-ink-mute)"
            strokeWidth={1.2} strokeDasharray="3 3" opacity={0.6} />}
          <path d={path(skip, n - 1)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.2} />
          {selected?.periods.map((p) => {
            const a = idx.get(p.from), b = idx.get(p.to);
            return a == null || b == null ? null :
              <path key={p.from} d={path(a, b)} fill="none" stroke={colourOf(selected)} strokeWidth={2} />;
          })}
          {/* today */}
          <line x1={x0} x2={hx1} y1={Y(d.now.close)} y2={Y(d.now.close)}
            stroke="var(--color-ink)" strokeDasharray="1 3" />
          <circle cx={X(n - 1)} cy={Y(d.now.close)} r={3.5} fill="var(--color-ink)" />

          {/* time at price */}
          <line x1={hx0} x2={hx0} y1={y0} y2={y1} stroke="var(--color-line-bright)" />
          {d.profile.price.map((p, i) => {
            if (p == null) return null;
            const t = d.profile.time[i] ?? 0;
            const up = d.profile.price[i + 1], dn = d.profile.price[i - 1];
            const top = up != null ? Y(Math.sqrt(p * up)) : Y(p) - 2;
            const bot = dn != null ? Y(Math.sqrt(p * dn)) : Y(p) + 2;
            const z = zoneAt(p);
            return <rect key={i} x={hx0} y={top} width={(t / tmax) * (hx1 - hx0)}
              height={Math.max(0.8, bot - top - 0.4)}
              fill={z ? colourOf(z) : "var(--color-ink-mute)"} opacity={z ? 0.75 : 0.35} />;
          })}
          <path d={volPath} fill="none" stroke="var(--color-ink)" strokeWidth={1} opacity={0.45} />
          {/* value area bracket */}
          <line x1={hx1 - 1} x2={hx1 - 1} y1={Y(d.value_area.hi)} y2={Y(d.value_area.lo)}
            stroke="var(--color-ink)" strokeWidth={2} opacity={0.5} />
          <text x={hx0 + 2} y={y0 + 10} fontSize={10.5} fill="var(--color-ink-mute)">{histW >= 150 ? "时间分布 · 线=成交量" : "时间分布"}</text>

          {h && h.v != null && (
            <g pointerEvents="none">
              <line x1={X(h.i)} x2={X(h.i)} y1={y0} y2={y1} stroke="var(--color-ink-mute)" />
              <circle cx={X(h.i)} cy={Y(h.v)} r={3} fill="var(--color-ink)" />
            </g>
          )}
        </svg>
      )}
      {h && h.v != null && (
        <div className="absolute top-1 pointer-events-none card px-2 py-1 text-[12px] shadow-sm"
          style={{ left: Math.min(X(h.i) + 8, x1 - 130) }}>
          {h.t} · <b>{money(h.v)}</b>
          {(() => { const z = zoneAt(h.v); return z ? <> · 区间 <ZoneTag z={z} /></> : null; })()}
        </div>
      )}
    </div>
  );
}

// ── table ──────────────────────────────────────────────────────────────────
function ZoneTable({ d, sel, setSel, money, basis }: {
  d: PriceZones; sel: number | null; setSel: (v: number | null) => void;
  money: (v: number | null) => string; basis: Basis;
}) {
  const peg = basis !== "pe";
  const g = growthOf(basis);
  return (
    <div className="overflow-x-auto -mx-1">
      <table className="w-full text-[12.5px] tabular-nums">
        <thead>
          <tr className="label text-left whitespace-nowrap">
            <th className="px-1 py-1 font-normal">区间</th>
            <th className="px-1 py-1 font-normal text-right" title="收盘价在区间内的交易日">停留</th>
            <th className="px-1 py-1 font-normal text-right" title="最近 242 个交易日里在区间内的比例">近一年</th>
            <th className="px-1 py-1 font-normal text-right" title="停留了几段 · 最长一段多少个交易日">几段 · 最长</th>
            <th className="px-1 py-1 font-normal text-right">最近一次</th>
            {d.has_pe && !peg && <>
              <th className="px-1 py-1 font-normal text-right" title="在区间内时的市盈率(TTM)中位数">当时PE</th>
              <th className="px-1 py-1 font-normal text-right"
                title={`按现在的盈利（PE ${d.now.pe ?? "—"}×），付当时的市盈率对应的股价`}>按今天盈利</th>
            </>}
            {d.has_pe && peg && <>
              <th className="px-1 py-1 font-normal text-right" title="区间内每个交易日当时已公布的利润增速，取中位数。不是某一天的数——点一行，下面列出每段停留各自的数">当时增速<span className="text-ink-dim">·中位</span></th>
              <th className="px-1 py-1 font-normal text-right" title="区间内每个交易日的 PEG（当天市盈率 ÷ 当天已知增速），取中位数；增速低于 5% 的日子不计">当时PEG<span className="text-ink-dim">·中位</span></th>
              <th className="px-1 py-1 font-normal text-right"
                title={`今天的价格、今天的盈利和${span(basis)}增速（${signedPct(d.now[g])}）算出的 PEG；增速低于 5% 时不适用`}>现在PEG</th>
              <th className="px-1 py-1 font-normal text-right"
                title={`如果今天还按当时的 PEG 定价，今天 ${fmtMult("pe", d.now.pe)} 的市盈率等于在押多少年增速：市盈率 ÷ 当时PEG。对比实际${span(basis)} ${signedPct(d.now[g])}`}>
                隐含增速 <span className="text-ink-dim">vs 实际 {signedPct(d.now[g])}</span>
              </th>
            </>}
          </tr>
        </thead>
        <tbody>
          {d.zones.map((z) => (
            <tr key={z.id} onClick={() => setSel(sel === z.id ? null : z.id)}
              className={`cursor-pointer border-t border-line ${sel === z.id ? "bg-elevated" : "hover:bg-panel-alt"}`}>
              <td className="px-1 py-1.5 whitespace-nowrap">
                <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 align-[-1px]" style={{ background: colourOf(z) }} />
                <b style={{ color: colourOf(z) }}>{tag(z)}</b> {money(z.lo)} – {money(z.hi)}
                {d.now.zone === z.id && <span className="ml-1.5 label">← 现价</span>}
              </td>
              <td className="px-1 py-1.5 text-right whitespace-nowrap">{z.days} 天 · <b>{pct(z.share)}</b></td>
              <td className="px-1 py-1.5 text-right">{z.recent_days ? pct(z.recent_share) : "—"}</td>
              <td className="px-1 py-1.5 text-right whitespace-nowrap">{z.visits} 段 · {z.longest} 天</td>
              <td className="px-1 py-1.5 text-right whitespace-nowrap">{z.last ?? "—"}</td>
              {d.has_pe && !peg && <>
                <td className="px-1 py-1.5 text-right">{fmtMult("pe", z.pe)}</td>
                <td className="px-1 py-1.5 text-right whitespace-nowrap">
                  {z.pe_price != null ? <b>{money(z.pe_price)}</b> : "—"}
                </td>
              </>}
              {d.has_pe && peg && <>
                <td className="px-1 py-1.5 text-right">{signedPct(z[g])}</td>
                <td className="px-1 py-1.5 text-right">{fmtMult(basis, z[basis])}</td>
                <td className="px-1 py-1.5 text-right whitespace-nowrap">
                  {d.now[basis] != null ? fmtMult(basis, d.now[basis]) : <span className="label">不适用</span>}
                  {d.now[basis] != null && z[basis] != null && (
                    <span className="label ml-1">{d.now[basis]! >= z[basis]! ? "↑" : "↓"}{Math.abs(d.now[basis]! / z[basis]! * 100 - 100).toFixed(0)}%</span>
                  )}
                </td>
                <td className="px-1 py-1.5 text-right whitespace-nowrap">
                  <Implied v={z[`${basis}_implied` as "peg1_implied" | "peg3_implied"]} actual={d.now[g]} />
                </td>
              </>}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="label mt-1 px-1">
        点一行看它的每一段停留；图上会把那几段时间标出来。按价格从高到低排，圈号是停留时间的名次。
        {d.has_pe && d.now.pe == null && " 现在亏损（没有市盈率），所以没法按今天的盈利折算。"}
        {peg && <>
          {" "}隐含增速 = 今天的市盈率 ÷ 当时的 PEG：如果市场今天还按那时的标准给增长定价，现价等于在押利润每年涨多少。
          比实际增速高，说明按那时的标准现价已经押了更快的增长；比实际低，说明押得比实际保守。
          {d.now.pe != null && d.now[basis] == null && ` 现在${span(basis)}利润增速 ${signedPct(d.now[g])}，低于 5%，现在的 PEG 不适用，但隐含增速照样能读。`}
        </>}
      </p>
    </div>
  );
}

/** The growth today's PE implies at a zone's PEG, set against the growth there actually is. */
function Implied({ v, actual }: { v: number | null; actual: number | null }) {
  if (v == null) return <>—</>;
  const gap = actual != null ? v - actual : null;
  return (
    <span title={gap != null ? `比实际增速${gap >= 0 ? "高" : "低"} ${Math.abs(gap * 100).toFixed(0)} 个百分点` : undefined}>
      <b>{signedPct(v)}</b>
      {gap != null && (
        <span className={`ml-1 text-[11.5px] ${Math.abs(gap) < 0.03 ? "text-ink-mute" : gap > 0 ? "text-brand-ink" : "text-cyan"}`}>
          {Math.abs(gap) < 0.03 ? "≈ 实际" : gap > 0 ? "押得比实际高" : "押得比实际低"}
        </span>
      )}
    </span>
  );
}

function Stays({ z, d, money, basis }: {
  z: PriceZone; d: PriceZones; money: (v: number | null) => string; basis: Basis;
}) {
  const peg = basis !== "pe";
  const g = growthOf(basis);
  const then = peg ? z[basis] : z.pe;
  const now = peg ? d.now[basis] : d.now.pe;
  const price = peg ? z[`${basis}_price` as "peg1_price" | "peg3_price"] : z.pe_price;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[13px]">
        区间 <ZoneTag z={z} /> {money(z.lo)} – {money(z.hi)} 的 {z.visits} 段停留
        {!peg && then != null && now != null && price != null && (
          <span className="label">
            {" "}· 当时市场付 {fmtMult("pe", then)} 市盈率，今天是 {fmtMult("pe", now)}；
            按今天的盈利，同样的估值对应 {money(price)}
          </span>
        )}
        {peg && then != null && (
          <span className="label">
            {" "}· 当时{span(basis)}利润增速 {signedPct(z[g])}，市场付 PEG {fmtMult(basis, then)}。
            今天市盈率 {fmtMult("pe", d.now.pe)}、{span(basis)}增速 {signedPct(d.now[g])}，
            PEG {now != null ? fmtMult(basis, now) : "不适用"}。
            {z[`${basis}_implied` as "peg1_implied" | "peg3_implied"] != null && <>
              按当时的 PEG，今天的市盈率等于在押利润每年增长 {signedPct(z[`${basis}_implied` as "peg1_implied" | "peg3_implied"])}
              {price != null && <>；按今天的增速付当时的 PEG，股价是 {money(price)}</>}。
            </>}
          </span>
        )}
      </div>
      {z.periods.length === 0 ? (
        <p className="label">没有超过 10 个交易日的连续停留——是多次短暂经过累积起来的。</p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {z.periods.map((p) => (
            <span key={p.from} className="rounded-md bg-sunken px-2 py-1 text-[12px] tabular-nums">
              {p.from} → {p.to} · {p.sessions} 天 · 均价 {money(p.avg)}
              {basis === "pe" && p.pe != null && <> · PE {fmtMult("pe", p.pe)}</>}
              {basis !== "pe" && (p[growthOf(basis)] != null || p[basis] != null) && (
                <> · 增速 {signedPct(p[growthOf(basis)])} · PEG {p[basis] != null ? fmtMult(basis, p[basis]!) : "不适用"}</>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * The arithmetic behind today's PEG, with the numbers in it — and, because a
 * report quotes 累计同比 (this year so far against last year so far), how
 * that differs from the trailing-twelve-month growth used here.
 */
function Formula({ d, basis }: { d: PriceZones; basis: Basis }) {
  const r = d.reports.find((x) => x.used) ?? null;
  const g = d.now[growthOf(basis)];
  const pe = d.now.pe;
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-2 text-[12.5px] leading-relaxed">
      <div>
        <b>PEG = 市盈率 ÷ 增速（百分数）</b>：
        {pe != null && g != null
          ? <> {fmtMult("pe", pe)} ÷ {(g * 100).toFixed(1)} = <b>{d.now[basis] != null ? fmtMult(basis, d.now[basis]) : `${(pe / (g * 100)).toFixed(2)}（增速低于 5%，不适用）`}</b></>
          : " —"}
        {r && <span className="label"> · 增速来自 {quarter(r.end_date)} 报告（{r.ann_date} 公布，{r.effective} 起算）</span>}
      </div>
      {r && basis === "peg1" && r.ttm != null && r.ttm_prev != null && (
        <div className="label">
          近一年 = 滚动 12 个月归母净利润 {yi(r.ttm)} ÷ 一年前的滚动 12 个月 {yi(r.ttm_prev)} − 1 = {pct1(r.g1)}
        </div>
      )}
      {r && basis === "peg3" && r.ttm != null && r.base3_ttm != null && (
        <div className="label">
          三年复合 = ( 滚动 12 个月 {yi(r.ttm)} ÷ 三年前同期（{quarter(r.base3_end)}）的滚动 12 个月 {yi(r.base3_ttm)} )^(1/3) − 1 = {pct1(r.g3)}
        </div>
      )}
      {r && r.ytd_yoy != null && !r.end_date.endsWith("1231") && (
        <div className="label">
          财报里常说的“累计同比 {pct1(r.ytd_yoy)}”是今年到 {quarter(r.end_date)} 对去年同期（{yi(r.profit)}），只比今年这几个月；
          这里用滚动 12 个月，四个季度对四个季度，和滚动市盈率（PE TTM）对得上。每一步的数见下方「财报明细」。
        </div>
      )}
    </div>
  );
}

const VH = 96;

/**
 * Each day's growth (as known that day) and PEG — or PE — on the same time
 * axis as the price chart, so "when" has an answer: every zone median is a
 * median of these lines over the days the price sat in the zone.
 */
function ValuationChart({ d, selected, basis, money }: {
  d: PriceZones; selected: PriceZone | null; basis: Basis; money: (v: number | null) => string;
}) {
  const [ref, size] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const v = d.valuation!;
  const W = Math.max(size.w, 300);
  const histW = Math.min(170, Math.round(W * 0.24));
  const x0 = PAD.l, x1 = W - histW - PAD.r - 8;
  const n = d.dates.length;
  const X = (i: number) => x0 + (n <= 1 ? 0 : (i / (n - 1)) * (x1 - x0));
  const idx = useMemo(() => new Map(d.dates.map((t, i) => [t, i])), [d.dates]);

  const peg = basis !== "pe";
  const g = growthOf(basis);
  const panes = peg
    ? [{ key: g, label: `增速 · ${span(basis)}`, arr: v[g], fmt: (x: number) => pct1(x), lo: -0.5, hi: 1.5, ref: [0, 0.05] },
       { key: basis, label: `PEG · ${span(basis)}`, arr: v[basis], fmt: (x: number) => x.toFixed(2), lo: 0, hi: 4, ref: [1] }]
    : [{ key: "pe", label: "市盈率 TTM", arr: v.pe, fmt: (x: number) => `${x.toFixed(1)}×`, lo: 0, hi: 200, ref: [] as number[] }];

  const H2 = panes.length * (VH + 8) + 22;
  const h = hover;
  const rep = h != null ? reportOn(d.reports, d.dates[h]!) : null;

  return (
    <div ref={ref} className="w-full relative select-none">
      <div className="flex flex-wrap gap-x-3 text-[12px] tabular-nums min-h-[18px] pl-[52px]">
        {h != null ? <>
          <span className="font-semibold">{d.dates[h]}</span>
          <span>收盘 {money(d.close[h] ?? null)}</span>
          <span>PE {fmtMult("pe", v.pe[h] ?? null)}</span>
          {peg && <span>增速 {pct1(v[g][h])}</span>}
          {peg && <span>PEG {v[basis][h] != null ? fmtMult(basis, v[basis][h]!) : "不适用"}</span>}
          {peg && rep && <span className="label">用 {quarter(rep.end_date)} 报告（{rep.ann_date} 公布）</span>}
        </> : <span className="label">移到图上看某一天：当天的市盈率、当时已公布的增速、PEG，和用的是哪份报告</span>}
      </div>
      {size.w > 0 && (
        <svg width={W} height={H2} className="block"
          onMouseLeave={() => setHover(null)}
          onMouseMove={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const mx = e.clientX - r.left;
            if (mx < x0 || mx > x1) return setHover(null);
            setHover(Math.round(((mx - x0) / (x1 - x0)) * (n - 1)));
          }}>
          {panes.map((p, k) => {
            const top = 4 + k * (VH + 8), bot = top + VH;
            const vals = p.arr.filter((x): x is number => x != null);
            // Clamped: a write-down year's −90% or a recovery's +300% would
            // flatten every other day into a line on the floor.
            const lo = Math.max(p.lo, Math.min(...vals, ...p.ref, 0));
            const hi = Math.min(p.hi, Math.max(...vals, ...p.ref) * 1.05);
            const Y = (x: number) => bot - ((Math.min(hi, Math.max(lo, x)) - lo) / (hi - lo || 1)) * (bot - top);
            let path = "";
            p.arr.forEach((x, i) => {
              if (x == null) { path += " "; return; }
              const cmd = path === "" || path.endsWith(" ") ? "M" : "L";
              path += `${cmd}${X(i).toFixed(1)},${Y(x).toFixed(1)}`;
            });
            const now = [...p.arr].reverse().find((x) => x != null) ?? null;
            return (
              <g key={p.key}>
                <rect x={x0} y={top} width={x1 - x0} height={VH} fill="var(--color-sunken)" opacity={0.5} />
                {selected?.periods.map((s) => {
                  const a = idx.get(s.from), b = idx.get(s.to);
                  return a == null || b == null ? null :
                    <rect key={s.from} x={X(a)} width={Math.max(1, X(b) - X(a))} y={top} height={VH}
                      fill={colourOf(selected)} opacity={0.14} />;
                })}
                {p.ref.map((r) => (
                  <g key={r}>
                    <line x1={x0} x2={x1} y1={Y(r)} y2={Y(r)} stroke="var(--color-ink-mute)"
                      strokeDasharray={r === 0 ? undefined : "3 3"} opacity={0.6} />
                    {/* 0% and the 5% floor sit a few pixels apart: only 0 gets an axis label. */}
                    {r !== 0.05 && <text x={x0 - 6} y={Y(r) + 4} fontSize={10.5} textAnchor="end" fill="var(--color-ink-mute)">{r === 0 && peg && p.key !== basis ? "0" : p.fmt(r)}</text>}
                  </g>
                ))}
                <text x={x0 - 6} y={top + 10} fontSize={10.5} textAnchor="end" fill="var(--color-ink-mute)">{p.fmt(hi)}</text>
                <path d={path.replace(/ /g, "")} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.3} />
                <text x={x1 + 10} y={top + 14} fontSize={11.5} fill="var(--color-ink)" fontWeight={600}>{p.label}</text>
                <text x={x1 + 10} y={top + 30} fontSize={11.5} fill="var(--color-ink-mute)">
                  现在 {now != null ? p.fmt(now) : "不适用"}
                </text>
                {h != null && <line x1={X(h)} x2={X(h)} y1={top} y2={bot} stroke="var(--color-ink-mute)" />}
              </g>
            );
          })}
          {peg && <>
            <text x={x1 + 10} y={4 + 46} fontSize={10.5} fill="var(--color-ink-mute)">虚线 = 5% 门槛</text>
            <text x={x1 + 10} y={4 + 2 * (VH + 8) + 8} fontSize={10.5} fill="var(--color-ink-mute)">
              增速低于 5% 时 PEG 断开
            </text>
          </>}
        </svg>
      )}
    </div>
  );
}

/** Every report the growth is built from, with each step of the arithmetic, newest first. */
function Reports({ d }: { d: PriceZones }) {
  const [open, setOpen] = usePersistentState("assrs.zones.reports", true);
  return (
    <div className="flex flex-col gap-1">
      <button onClick={() => setOpen(!open)} className="self-start text-[13px] font-semibold hover:text-cyan">
        📄 财报明细：增速每一步怎么算的 <span className="label font-normal">（{d.reports.length} 份报告，归母净利润，首次公布的数字）{open ? " ▲" : " ▼"}</span>
      </button>
      {open && (
        <div className="overflow-x-auto max-h-80 overflow-y-auto -mx-1 border border-line rounded-lg">
          <table className="w-full text-[12px] tabular-nums">
            <thead className="sticky top-0 bg-panel">
              <tr className="label text-right whitespace-nowrap">
                <th className="px-1.5 py-1 font-normal text-left">报告期</th>
                <th className="px-1.5 py-1 font-normal text-left" title="公布日的第二天起，这份报告的增速才算数">公布 → 起算</th>
                <th className="px-1.5 py-1 font-normal" title="年初到报告期末的累计归母净利润">累计利润</th>
                <th className="px-1.5 py-1 font-normal" title="累计利润 ÷ 去年同期累计利润 − 1：财报和行情软件常说的“同比”">累计同比</th>
                <th className="px-1.5 py-1 font-normal" title="滚动 12 个月 = 今年累计 + 去年全年 − 去年同期累计">滚动12个月</th>
                <th className="px-1.5 py-1 font-normal">一年前滚动</th>
                <th className="px-1.5 py-1 font-normal" title="滚动12个月 ÷ 一年前滚动 − 1">近一年</th>
                <th className="px-1.5 py-1 font-normal">三年前同期滚动</th>
                <th className="px-1.5 py-1 font-normal" title="(滚动12个月 ÷ 三年前同期滚动)^(1/3) − 1">三年复合</th>
              </tr>
            </thead>
            <tbody>
              {d.reports.map((r) => (
                <tr key={r.end_date} className={`border-t border-line text-right whitespace-nowrap ${r.used ? "" : "text-ink-mute"}`}>
                  <td className="px-1.5 py-1 text-left font-semibold">{quarter(r.end_date)}</td>
                  <td className="px-1.5 py-1 text-left">
                    {r.ann_date} → {r.effective}
                    {r.replaced_by && <span className="label" title="同一天还公布了更新的报告期，那天起用它"> · 同日公布 {quarter(r.replaced_by)}，用后者</span>}
                  </td>
                  <td className="px-1.5 py-1">{yi(r.profit)}</td>
                  <td className="px-1.5 py-1">{pct1(r.ytd_yoy)}</td>
                  <td className="px-1.5 py-1">{yi(r.ttm)}</td>
                  <td className="px-1.5 py-1">{yi(r.ttm_prev)}</td>
                  <td className="px-1.5 py-1 font-semibold">{pct1(r.g1)}</td>
                  <td className="px-1.5 py-1">{r.base3_end ? <>{yi(r.base3_ttm)} <span className="label">{quarter(r.base3_end)}</span></> : "—"}</td>
                  <td className="px-1.5 py-1 font-semibold">{pct1(r.g3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="label">
        数据来自 Tushare 利润表（合并报表，归母净利润），每个报告期用第一次公布的数字，事后修订不算。
        一次性的大额减值会让某一年的利润很低，紧接着的增速就会很高；它三年后还会成为“三年前同期”，把三年复合拉高——看这张表就能看出来。
      </p>
    </div>
  );
}
