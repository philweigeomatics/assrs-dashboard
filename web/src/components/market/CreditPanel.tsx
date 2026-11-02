/**
 * 信用与债务 — is the debt behind the AI build-out turning into credit
 * stress? US first: six fixed rules (credit_stress.py), each ✓/✗ with its
 * reading, a level that is only how many hold, and the record of past
 * entries. China below as credit-cycle context.
 *
 * Levels are amber, never red/green: stress is a risk, not a direction.
 * US prices use the US convention (green up, red down); China's growth
 * figures are neutral lines.
 */

import { useState } from "react";
import type { CreditBoard, CreditLevel, Weekly } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const LEVEL: Record<CreditLevel, string> = {
  calm: "bg-sunken text-ink-dim",
  weak: "bg-brand/15 text-brand-ink",
  stress: "bg-brand text-white",
};

export function CreditPanel({ d }: { d: CreditBoard }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className={`px-2.5 py-1 rounded-lg text-[14px] font-semibold ${LEVEL[d.level]}`}>美国信用：{d.level_label}</span>
        <span className="label">{d.hold}/{d.of} 条成立 · 自 {d.level_since} · 截至 {d.as_of}</span>
        {d.us_debt && (
          <span className="label ml-auto tnum">美国非金融企业债务 ${fixed(d.us_debt.now, 1)} 万亿 · 同比 {signed(d.us_debt.yoy, 1, "%")}（{d.us_debt.asof}）</span>
        )}
      </div>

      <div className="grid gap-1.5 grid-cols-1 md:grid-cols-2">
        {d.rules.map((r) => (
          <div key={r.key} className="flex gap-2 rounded-lg bg-sunken px-2.5 py-1.5 text-[12.5px]" title={r.rule}>
            <span className={`w-4 shrink-0 text-center font-semibold ${r.ok ? "text-brand-ink" : "text-ink-mute"}`}>
              {r.ok == null ? "—" : r.ok ? "✓" : "✗"}
            </span>
            <div className="min-w-0">
              <div className={r.ok ? "font-medium" : "text-ink-dim"}>{r.label}</div>
              <div className="label tnum">{r.value ?? "—"}{r.ok && r.since ? ` · 自 ${r.since}` : ""}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid gap-2 grid-cols-1 md:grid-cols-3">
        {Object.entries(d.spreads).map(([k, s]) => (
          <Mini key={k} title={s.label} unit="%" w={s} nd={2}
            sub={`6 个月前 ${fixed(s.m6, 2)}% · 3 年区间 ${fixed(s.lo, 2)}–${fixed(s.hi, 2)}%`} />
        ))}
      </div>
      <Mini title="Baa − 10 年期国债（1986 年以来）" unit="%" w={d.baa} nd={2} tall
        lines={[d.baa.bands["20"], d.baa.bands["80"]]}
        sub={`分位 ${fixed((d.baa.rank ?? 0) * 100, 0)}% · 虚线 = 20 / 80 分位（${fixed(d.baa.bands["20"], 2)}% / ${fixed(d.baa.bands["80"], 2)}%）`} />

      <div className="grid gap-2 grid-cols-2 md:grid-cols-5">
        {d.market.map((m) => (
          <div key={m.key} className="rounded-lg bg-sunken p-2" title={m.name}>
            <div className="flex items-baseline gap-1">
              <span className="text-[12.5px] font-semibold">{m.key}</span>
              <span className="label truncate">{m.name}</span>
            </div>
            <div className="flex items-baseline gap-2 tnum text-[12.5px]">
              <span className={moveUS(m.off_high)}>距一年高点 {signed(m.off_high, 1, "%")}</span>
              <span className={`label ${moveUS(m.m3)}`}>3 月 {signed(m.m3, 1, "%")}</span>
            </div>
            <Spark w={m} />
          </div>
        ))}
      </div>

      <div className="grid gap-2 grid-cols-1 md:grid-cols-2">
        {d.nfci && <Mini title="芝加哥联储金融条件指数 NFCI（> 0 = 比平均紧）" unit="" w={d.nfci} nd={2} lines={[0]} />}
        {d.sloos && <Mini title="收紧企业贷款标准的银行净占比（季度）" unit="%" w={d.sloos} nd={1} lines={[0, 20]} />}
      </div>

      <Record d={d} />

      <section className="flex flex-col gap-1.5">
        <span className="text-[13px] font-semibold">中国信用周期（背景，不参与评级）</span>
        <div className="grid gap-2 grid-cols-1 md:grid-cols-4">
          {d.credit_gdp.CN && <Mini title="中国私人部门信贷 / GDP（BIS）" unit="%" w={d.credit_gdp.CN} nd={1} />}
          {d.credit_gdp.US && <Mini title="美国私人部门信贷 / GDP（BIS）" unit="%" w={d.credit_gdp.US} nd={1} />}
          {d.china.tsf_yoy && <Mini title="社融存量同比" unit="%" w={d.china.tsf_yoy} nd={1} />}
          {d.china.m1_yoy && <Mini title="M1 同比" unit="%" w={d.china.m1_yoy} nd={1}
            sub={d.china.m2_yoy ? `M2 同比 ${fixed(d.china.m2_yoy.v[d.china.m2_yoy.v.length - 1], 1)}%` : undefined} />}
        </div>
      </section>

      <p className="label leading-snug">
        规则只用历史足够长的序列，所以每个读数都有历史记录；每个数据按它实际公布的日子进入（贷款官员调查约晚 5 周、
        BIS 信贷/GDP 约晚半年），分位只用当天之前的数据——历史记录里没有后见之明。ICE 高收益 / 投资级 / BBB 利差更贴切，
        但 FRED 现在只提供最近 3 年，所以只作读数和走势，不参与规则。BIZD 是上市 BDC，只是私募信贷看得见的一端。
        中国公司债利差不在当前 Tushare 权限内，中国部分只作背景。数据：FRED（圣路易斯联储）、Yahoo、Tushare，3 小时更新。
      </p>
    </div>
  );
}

function moveUS(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v) || v === 0) return "text-flat";
  return v > 0 ? "text-down" : "text-up";      // US convention: green up, red down
}

function path(v: (number | null)[], x: (i: number) => number, y: (n: number) => number) {
  let s = "", pen = false;
  v.forEach((n, i) => { if (n == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(n).toFixed(1)}`; pen = true; });
  return s;
}

function Mini({ title, unit, w, nd, sub, lines = [], tall = false }: {
  title: string; unit: string; w: Weekly; nd: number; sub?: string; lines?: (number | null)[]; tall?: boolean;
}) {
  const W = 300, H = tall ? 110 : 80;
  const vals = [...w.v, ...lines].filter((x): x is number => x != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const n = w.v.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const y = (v: number) => H - 4 - ((v - lo) / (hi - lo || 1)) * (H - 8);
  const last = [...w.v].reverse().find((v) => v != null) ?? null;
  const [hover, setHover] = useState<number | null>(null);
  const h = hover ?? n - 1;
  return (
    <div className="rounded-lg bg-sunken p-2 min-w-0">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium truncate">{title}</span>
        <span className="ml-auto text-[13px] font-semibold tnum">{fixed(hover != null ? w.v[h] : last, nd)}{unit}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }} preserveAspectRatio="none"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))));
        }}
        onMouseLeave={() => setHover(null)}>
        {lines.filter((l): l is number => l != null).map((l, i) => (
          <line key={i} x1={0} x2={W} y1={y(l)} y2={y(l)} stroke="var(--color-line-bright)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        ))}
        <path d={path(w.v, x, y)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="label truncate">{hover != null ? w.dates[h] : sub ?? `${w.dates[0] ?? ""} 起`}</div>
    </div>
  );
}

function Spark({ w }: { w: Weekly }) {
  const W = 200, H = 34;
  const v = w.v.filter((x): x is number => x != null);
  if (v.length < 2) return null;
  const lo = Math.min(...v), hi = Math.max(...v);
  const x = (i: number) => (i / (w.v.length - 1)) * W;
  const y = (n: number) => H - 2 - ((n - lo) / (hi - lo || 1)) * (H - 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[34px]" preserveAspectRatio="none">
      <path d={path(w.v, x, y)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.1} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Record({ d }: { d: CreditBoard }) {
  const [open, setOpen] = useState(false);
  const r = d.record;
  const row = (label: string, x: { median63: number | null; up63: number | null; dd63: number | null }, n?: number) => (
    <tr key={label} className="border-t border-line">
      <td className="py-1 pr-2">{label}</td>
      <td className="py-1 pr-2 text-right">{n ?? "—"}</td>
      <td className="py-1 pr-2 text-right">{signed(x.median63, 1, "%")}</td>
      <td className="py-1 pr-2 text-right">{x.up63 != null ? `${fixed(x.up63 * 100, 0)}%` : "—"}</td>
      <td className="py-1 text-right">{signed(x.dd63, 1, "%")}</td>
    </tr>
  );
  return (
    <section className="flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold">历史上进入这一级别之后</span>
        <span className="label">标普 500 此后 3 个月；同一级别相隔 20 个交易日以上才算再次进入</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal">级别</th><th className="font-normal text-right">次数</th>
              <th className="font-normal text-right">3 个月中位</th><th className="font-normal text-right">上涨占比</th>
              <th className="font-normal text-right">期间最大回撤中位</th>
            </tr>
          </thead>
          <tbody>
            {row("走弱或紧张", r.weak, r.weak.n)}
            {row("紧张", r.stress, r.stress.n)}
            {row("任意一天（对照）", r.base)}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        紧张往往出现在下跌的中后段：此后 3 个月的回撤明显更深，但收益并不更差——它是风险读数，不是卖出信号。
      </p>
      <button onClick={() => setOpen(!open)} className="text-[12px] text-cyan self-start">{open ? "收起" : "看每一次的日期"}</button>
      {open && (
        <div className="grid gap-2 grid-cols-1 md:grid-cols-2 text-[12px] tnum">
          {(["stress", "weak"] as const).map((k) => (
            <div key={k} className="rounded-lg bg-sunken p-2">
              <div className="font-medium mb-1">{k === "stress" ? "紧张" : "走弱或紧张"}</div>
              {r[k].list.map((x) => (
                <div key={x.date} className="flex gap-2">
                  <span className="label">{x.date}</span>
                  <span className="ml-auto">{x.r63 == null ? "进行中" : `${signed(x.r63, 1, "%")} · 回撤 ${signed(x.dd63, 1, "%")}`}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
