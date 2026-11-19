/**
 * ⚖️ 估值 — A-shares: what the market paid for this stock's growth on its
 * normal days, applied to today. The arithmetic is in fair_value.py; this
 * shows the verdict first, then every number it rests on, then the days that
 * were left out and why.
 */

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { FairValue, FvCall, FvMethodKey } from "../lib/types";
import { usePersistentState } from "../lib/usePersistentState";

const METHOD: Record<FvMethodKey, { label: string; growth: string }> = {
  pe: { label: "市盈率", growth: "" },
  peg1: { label: "PEG · 近一年增速", growth: "近一年" },
  peg3: { label: "PEG · 三年复合增速", growth: "三年复合" },
};

// Not red/green: in this app those mean up and down, and a valuation call is
// neither a price move nor its direction.
const CALL: Record<FvCall, { label: string; cls: string }> = {
  under: { label: "低估", cls: "bg-cyan/15 text-ink ring-1 ring-cyan/50" },
  fair: { label: "合理", cls: "bg-sunken text-ink ring-1 ring-line-bright" },
  over: { label: "高估", cls: "bg-brand-ink/10 text-brand-ink ring-1 ring-brand-ink/40" },
  mixed: { label: "两种读数矛盾", cls: "bg-violet/10 text-ink ring-1 ring-violet/40" },
};

const money = (v: number | null | undefined) =>
  v == null ? "—" : `¥${v >= 1000 ? v.toFixed(0) : v.toFixed(2)}`;
const pct = (v: number | null | undefined, nd = 1) =>
  v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(nd)}%`;
const g = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const mult = (k: FvMethodKey, v: number | null | undefined) =>
  v == null ? "—" : k === "pe" ? `${v.toFixed(1)}×` : v.toFixed(2);
const quarter = (e: string | null) =>
  e ? `${e.slice(0, 4)} ${{ "0331": "一季报", "0630": "中报", "0930": "三季报", "1231": "年报" }[e.slice(4)] ?? e.slice(4)}` : "";

export function FairValuePanel({ ticker }: { ticker: string }) {
  const [open, setOpen] = usePersistentState("assrs.fv.open", false);
  const q = useQuery({
    queryKey: ["fair-value", ticker],
    queryFn: () => api.fairValue(ticker),
    enabled: open,
    staleTime: 3600_000,
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
  });

  return (
    <section className="card">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-elevated rounded-[10px]">
        <span className="text-[14px] font-semibold shrink-0">⚖️ 估值</span>
        <span className="label truncate">
          去掉大盘过热和过冷的时期，市场正常时为它的增长付多少钱 · 今天的价格合不合理
        </span>
        <span className="ml-auto label shrink-0">{open ? "收起 ▲" : "展开 ▼"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 flex flex-col gap-3 border-t border-line">
          {q.isPending && (
            <div className="py-10 flex flex-col items-center gap-2 text-ink-mute">
              <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
              <span className="label">读取十年市盈率、财报和沪深300…</span>
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

function Body({ d }: { d: FairValue }) {
  const v = d.verdict;
  const prim = d.primary ? d.methods[d.primary] : null;
  const im = d.implied;
  return (
    <>
      {/* The answer first. */}
      <div className="pt-2 flex flex-col gap-2">
        {v && prim?.fair ? (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <span className={`px-2.5 h-8 rounded-lg flex items-center text-[15px] font-semibold ${CALL[v.call].cls}`}>
              {CALL[v.call].label}
            </span>
            <span className="text-[13.5px]">
              现价 <b className="tnum">{money(d.now.price)}</b>，
              按 <b>{METHOD[v.method].label}</b> 的合理区间{" "}
              <b className="tnum">{money(prim.fair.lo)} – {money(prim.fair.hi)}</b>
              <span className="text-ink-mute">（中值 {money(prim.fair.mid)}，现价{v.gap_mid >= 0 ? "高" : "低"} {(Math.abs(v.gap_mid) * 100).toFixed(1)}%）</span>
            </span>
          </div>
        ) : (
          <p className="text-[13.5px]">
            <b>给不出合理价。</b>
            {d.now.pe == null ? "现在亏损，没有市盈率；" : ""}
            {Object.values(d.methods).map((m) => m.why).filter(Boolean)[0] ?? ""}
          </p>
        )}

        {im && (
          <p className="text-[13px] leading-relaxed">
            反过来读：按正常时期的 {METHOD[im.method].label}
            （中位 {mult(im.method, d.methods[im.method].band?.mid)}），今天 <b className="tnum">{d.now.pe?.toFixed(1)}×</b> 的市盈率
            隐含利润每年增长 <b className="tnum">{g(im.mid)}</b>
            <span className="text-ink-mute">（{g(im.lo)} – {g(im.hi)}）</span>；
            实际{METHOD[im.method].growth}增速 <b className="tnum">{g(im.actual)}</b>
            {im.call === "over" && " —— 价格押的增长比公司交出来的多。"}
            {im.call === "under" && " —— 价格押的增长比公司交出来的少。"}
            {im.call === "fair" && " —— 在价格押的范围里。"}
          </p>
        )}

        <Warnings d={d} />
      </div>

      {d.scenarios.length > 0 && <Scenarios d={d} />}
      <Methods d={d} />
      <Regimes d={d} />

      <p className="label leading-snug">
        「合理」是市场在大盘不过热也不过冷的日子里，为这家公司这样的增长通常付的价钱，不是内在价值；
        公司本身变了，它自己的历史也会误导。增速按财报公告后第二天起算，即当时市场能看到的数字。
        不构成投资建议。
      </p>
    </>
  );
}

function Warnings({ d }: { d: FairValue }) {
  const pe = d.methods.pe;
  const items: string[] = [];
  if (d.verdict?.call === "mixed" || (d.primary === "pe" && pe.slowed)) {
    items.push(`正常时期市场为约 ${g(pe.growth_then)} 的增速付了 ${mult("pe", pe.band?.mid)} 的市盈率；` +
      `现在增速只有 ${g(d.now.g3 ?? d.now.g1)}，按当时的市盈率算的合理价会偏高。`);
  }
  if (d.cyclical.flag) {
    items.push(`利润是周期性的（十年里增速正负翻转 ${d.cyclical.flips} 次` +
      `${d.cyclical.losses ? `，有 ${d.cyclical.losses} 期亏损` : ""}）：周期股的市盈率在利润顶点最低，` +
      "按 PEG 看起来便宜的时候往往正是周期高点。");
  }
  if (d.confidence === "low") {
    items.push("正常时期可用的交易日不到一年，区间是从很少的样本里估的。");
  }
  if (!items.length) return null;
  return (
    <ul className="flex flex-col gap-1">
      {items.map((t) => (
        <li key={t} className="text-[12.5px] leading-snug rounded-lg bg-brand-ink/5 px-2.5 py-1.5">⚠️ {t}</li>
      ))}
    </ul>
  );
}

function Scenarios({ d }: { d: FairValue }) {
  const basis = d.scenario_basis;
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-[13px] font-semibold">
        不同增速假设下的合理价
        <span className="label font-normal ml-2">
          都按正常时期的 {basis ? METHOD[basis].label : "PEG"}（{mult(basis ?? "peg3", basis ? d.methods[basis].band?.lo : null)} – {mult(basis ?? "peg3", basis ? d.methods[basis].band?.hi : null)}）
        </span>
      </h4>
      <div className="overflow-x-auto">
        <table className="w-full text-[12.5px] border-collapse min-w-[460px]">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">假设</th>
              <th className="text-right font-normal pb-1 px-2">增速</th>
              <th className="text-right font-normal pb-1 px-2">合理价（低 – 中 – 高）</th>
              <th className="text-right font-normal pb-1 pl-2">中值相对现价</th>
            </tr>
          </thead>
          <tbody>
            {d.scenarios.map((s) => (
              <tr key={s.key} className="border-t border-line">
                <td className="py-1.5 pr-2">
                  {s.label}
                  {s.period && <span className="label ml-1">{quarter(s.period)} 累计同比</span>}
                </td>
                <td className="py-1.5 px-2 text-right tnum">{g(s.growth)}</td>
                <td className="py-1.5 px-2 text-right tnum">
                  {s.mid == null ? <span className="label">增速低于 5%，PEG 不适用</span>
                    : <>{money(s.lo)} – <b>{money(s.mid)}</b> – {money(s.hi)}</>}
                </td>
                <td className="py-1.5 pl-2 text-right tnum">{pct(s.upside)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Methods({ d }: { d: FairValue }) {
  return (
    <details>
      <summary className="text-[13px] font-semibold cursor-pointer">
        三种算法的明细 <span className="label font-normal">正常时期的 25% / 50% / 75% 分位</span>
      </summary>
      <div className="overflow-x-auto mt-1.5">
        <table className="w-full text-[12.5px] border-collapse min-w-[560px]">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">算法</th>
              <th className="text-right font-normal pb-1 px-2">正常时期区间</th>
              <th className="text-right font-normal pb-1 px-2">当时的增速</th>
              <th className="text-right font-normal pb-1 px-2">天数</th>
              <th className="text-right font-normal pb-1 px-2">今天的合理价</th>
              <th className="text-left font-normal pb-1 pl-2">说明</th>
            </tr>
          </thead>
          <tbody>
            {(["peg3", "peg1", "pe"] as FvMethodKey[]).map((k) => {
              const m = d.methods[k];
              return (
                <tr key={k} className={`border-t border-line ${d.primary === k ? "bg-cyan/5" : ""}`}>
                  <td className="py-1.5 pr-2 whitespace-nowrap">
                    {METHOD[k].label}{d.primary === k && <span className="label ml-1">· 采用</span>}
                  </td>
                  <td className="py-1.5 px-2 text-right tnum whitespace-nowrap">
                    {m.band ? `${mult(k, m.band.lo)} / ${mult(k, m.band.mid)} / ${mult(k, m.band.hi)}` : "—"}
                  </td>
                  <td className="py-1.5 px-2 text-right tnum">{g(m.growth_then)}</td>
                  <td className="py-1.5 px-2 text-right tnum">{m.band?.n ?? "—"}</td>
                  <td className="py-1.5 px-2 text-right tnum whitespace-nowrap">
                    {m.fair ? `${money(m.fair.lo)} – ${money(m.fair.hi)}` : "—"}
                  </td>
                  <td className="py-1.5 pl-2 text-ink-mute">{m.why ?? (k === "pe" && m.slowed ? "增速已明显放慢，偏高" : "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug mt-1.5">
        今天：市盈率 {d.now.pe?.toFixed(2) ?? "亏损"}，近一年增速 {g(d.now.g1)}，三年复合 {g(d.now.g3)}
        {d.now.period && `（截至 ${quarter(d.now.period)}）`}。
        合理价 = 现价 × 合理市盈率 ÷ 今天的市盈率；PEG 的合理市盈率 = PEG × 今天的增速（百分数）。
        增速低于 5% 时 PEG 只是除法，不算估值。
      </p>
    </details>
  );
}

/**
 * The market's PE with the cut-offs, and the stock's PEG on the same dates,
 * the left-out days pale. So which days the band came from is visible, not
 * just stated.
 */
function Regimes({ d }: { d: FairValue }) {
  const mk = d.market;
  const basis: FvMethodKey = d.primary === "peg1" ? "peg1" : d.primary === "pe" ? "pe" : "peg3";
  const series = basis === "pe" ? d.chart.pe : d.chart[basis];
  const band = d.methods[basis].band;
  const W = 720, H = 120, P = { l: 40, r: 8, t: 8, b: 18 };
  const n = d.chart.dates.length;
  const x = (i: number) => P.l + (i / Math.max(1, n - 1)) * (W - P.l - P.r);

  const paths = useMemo(() => {
    const scale = (vals: (number | null)[], clip?: number) => {
      const ok = vals.filter((v): v is number => v != null && (clip == null || v <= clip));
      if (!ok.length) return null;
      const lo = Math.min(...ok), hi = Math.max(...ok);
      const pad = (hi - lo) * 0.08 || 1;
      return { lo: lo - pad, hi: hi + pad };
    };
    // A PEG of 30 from one quarter of near-zero growth would flatten the rest.
    const sPeg = scale(series, band ? band.hi * 4 : undefined);
    const sMkt = scale(d.chart.market_pe);
    const y = (s: { lo: number; hi: number }, v: number) =>
      P.t + (1 - (v - s.lo) / (s.hi - s.lo)) * (H - P.t - P.b);
    const line = (vals: (number | null)[], s: { lo: number; hi: number }, keep: (i: number) => boolean) => {
      let out = "", pen = false;
      vals.forEach((v, i) => {
        if (v == null || !keep(i) || v > s.hi || v < s.lo) { pen = false; return; }
        out += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(s, v).toFixed(1)}`;
        pen = true;
      });
      return out;
    };
    return { sPeg, sMkt, y, line };
  }, [d, series, band]);

  const years = useMemo(() => {
    const out: { i: number; y: string }[] = [];
    d.chart.dates.forEach((t, i) => {
      if (i === 0 || t.slice(0, 4) !== (d.chart.dates[i - 1] ?? "").slice(0, 4)) out.push({ i, y: t.slice(0, 4) });
    });
    // A window starting late in a year would print that year on top of the next.
    if (out.length > 1 && (out[1]?.i ?? 0) - (out[0]?.i ?? 0) < n / 20) out.shift();
    return out;
  }, [d.chart.dates]);

  const spans = d.market.excluded.map((r) => {
    const a = d.chart.dates.findIndex((t) => t >= r.from);
    let b = d.chart.dates.findIndex((t) => t > r.to);
    b = b < 0 ? n - 1 : Math.max(a, b - 1);
    return a < 0 ? null : { ...r, a, b };
  }).filter((s): s is NonNullable<typeof s> => s != null);

  const Shade = () => (
    <>
      {spans.map((s) => (
        <rect key={`${s.from}${s.kind}`} x={x(s.a)} y={P.t} width={Math.max(1, x(s.b) - x(s.a))}
          height={H - P.t - P.b}
          fill={s.kind === "high" ? "var(--color-brand-ink)" : "var(--color-cyan)"} opacity={0.12} />
      ))}
      {years.map((t) => (
        <text key={t.y} x={x(t.i)} y={H - 4} fontSize={10} fill="var(--color-ink-mute)">{t.y}</text>
      ))}
    </>
  );

  const { sPeg, sMkt, y, line } = paths;
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-[13px] font-semibold">哪些日子算「正常」</h4>
      <p className="text-[12.5px] leading-relaxed">
        以 {mk.label} 的市盈率（TTM）代表大盘，{mk.from} → {mk.to}：去掉最高 {(mk.cut * 100).toFixed(0)}%
        （高于 <b className="tnum">{mk.hi.toFixed(2)}×</b>）和最低 {(mk.cut * 100).toFixed(0)}%（低于{" "}
        <b className="tnum">{mk.lo.toFixed(2)}×</b>）的时期，短于 5 天的越线不算、两段之间不到 10 天的回落连成一段。
        这只股票 {d.sessions.toLocaleString()} 个交易日里有 <b>{d.normal_sessions.toLocaleString()}</b> 天算正常。
        {mk.label}现在 <b className="tnum">{mk.now.toFixed(2)}×</b>，处于十年第 {(mk.now_pct * 100).toFixed(0)} 百分位
        —— {mk.now_state === 0 ? "正常" : mk.now_state > 0 ? "过热（被排除的那一档）" : "过冷（被排除的那一档）"}。
      </p>

      {sMkt && (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
          aria-label={`${mk.label} 市盈率与正常区间`}>
          <Shade />
          {[mk.lo, mk.hi].map((v) => (
            <g key={v}>
              <line x1={P.l} x2={W - P.r} y1={y(sMkt, v)} y2={y(sMkt, v)}
                stroke="var(--color-ink-mute)" strokeDasharray="3 3" strokeWidth={1} />
              <text x={P.l - 4} y={y(sMkt, v) + 3} fontSize={10} textAnchor="end" fill="var(--color-ink-mute)">
                {v.toFixed(1)}
              </text>
            </g>
          ))}
          <path d={line(d.chart.market_pe, sMkt, () => true)} fill="none"
            stroke="var(--color-ink-dim)" strokeWidth={1.4} />
          <text x={P.l + 4} y={P.t + 10} fontSize={10.5} fill="var(--color-ink-dim)">{mk.label} 市盈率</text>
        </svg>
      )}

      {sPeg && (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
          aria-label={`这只股票的 ${METHOD[basis].label}，正常时期加深`}>
          <Shade />
          {band && [band.lo, band.mid, band.hi].map((v, k) => v <= sPeg.hi && v >= sPeg.lo && (
            <g key={k}>
              <line x1={P.l} x2={W - P.r} y1={y(sPeg, v)} y2={y(sPeg, v)}
                stroke="var(--color-cyan)" strokeDasharray={k === 1 ? undefined : "3 3"} strokeWidth={1} />
              {/* The quartiles labelled, the median not: they sit too close to
                  carry three labels, and the legend says which line it is. */}
              {k !== 1 && (
                <text x={P.l - 4} y={y(sPeg, v) + 3} fontSize={10} textAnchor="end" fill="var(--color-cyan)">
                  {mult(basis, v)}
                </text>
              )}
            </g>
          ))}
          <path d={line(series, sPeg, () => true)} fill="none"
            stroke="var(--color-line-bright)" strokeWidth={1.2} />
          <path d={line(series, sPeg, (i) => d.chart.normal[i] === true)} fill="none"
            stroke="var(--color-ink)" strokeWidth={1.5} />
          <text x={P.l + 4} y={P.t + 10} fontSize={10.5} fill="var(--color-ink-dim)">
            这只股票的 {METHOD[basis].label}（深色 = 正常时期，虚线 = 25% / 75%，实线 = 中位）
          </text>
        </svg>
      )}

      <details>
        <summary className="label cursor-pointer">
          被排除的时期（{spans.length} 段：
          <span style={{ color: "var(--color-brand-ink)" }}>■</span> 过热
          <span className="ml-1" style={{ color: "var(--color-cyan)" }}>■</span> 过冷）
        </summary>
        <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[12px] tnum">
          {mk.excluded.map((r) => (
            <li key={`${r.from}${r.kind}`}>
              {r.kind === "high" ? "过热" : "过冷"} {r.from} → {r.to}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}
