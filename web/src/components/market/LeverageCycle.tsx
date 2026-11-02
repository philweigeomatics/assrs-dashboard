/**
 * 杠杆周期 — where the A-share leverage cycle stands: 杠杆出清 → 底部区域 →
 * 右侧确认, from four layers of fixed rules (leverage_cycle.py). Not a buy or
 * sell call: every ✓ is a rule with a number, and each stage's past entries
 * and what the market did after them sit beside it to be judged.
 *
 * Stages are drawn in amber / blue / ink, never red and green — under 红涨绿跌
 * a red stage would read as "rising". Money flows keep the market convention:
 * inflow red, outflow green.
 */

import { useMemo, useState } from "react";
import type { CycleStage, LeverageCycle as Cycle } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const STAGE: Record<CycleStage, { label: string; color: string; chip: string }> = {
  none: { label: "未进入出清", color: "transparent", chip: "bg-sunken text-ink-mute" },
  clearing: { label: "杠杆出清", color: "var(--color-brand)", chip: "bg-brand/15 text-brand-ink" },
  bottom: { label: "底部区域", color: "var(--color-cyan)", chip: "bg-cyan/15 text-cyan" },
  right: { label: "右侧确认", color: "var(--color-ink-dim)", chip: "bg-ink-dim text-white" },
};
const CODE: Record<string, CycleStage> = { "-": "none", c: "clearing", b: "bottom", r: "right" };
const STEPS = ["clearing", "bottom", "right"] as const;
const RANGES = [{ k: 250, label: "1 年" }, { k: 750, label: "3 年" }, { k: 0, label: "2014 起" }] as const;

export function LeverageCycle({ d }: { d: Cycle }) {
  return (
    <div className="flex flex-col gap-3">
      <Verdict d={d} />
      <div className="grid gap-2 grid-cols-1 lg:grid-cols-2">
        {d.layers.map((l) => <Layer key={l.key} l={l} />)}
      </div>
      <Charts d={d} />
      <Record d={d} />
      <p className="label leading-snug">
        规则是固定的数字门槛，不是对过去走势的拟合；下方「历史上进入这一阶段之后」是对规则的检验，不是买卖指令。
        融资余额是负债，不随股价涨跌：只有偿还和平仓会让它下降，所以指数新低而它不新低才有意义。
        融资余额/流通市值在暴跌时会因为市值缩水而上升，所以 ① 同时看余额本身。
        比例和波动按 2014 年以来每一天当时已知的数据排分位（此前两融仍在试点）。
        指数为中证全指，两融为沪深两市，ETF 为 15 只主要宽基（沪深300、上证50、中证500/1000/2000、创业板、科创50、中证A500）。
        数据：Tushare，每个交易日 20:00 后更新。
      </p>
    </div>
  );
}

function Verdict({ d }: { d: Cycle }) {
  const at = (STEPS as readonly CycleStage[]).indexOf(d.stage);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <span className={`px-2.5 py-1 rounded-lg text-[14px] font-semibold ${STAGE[d.stage].chip}`}>{d.stage_label}</span>
      <span className="label">自 {d.stage_since} · 截至 {d.as_of}</span>
      <div className="flex items-center gap-1 text-[12.5px]" aria-label="阶段顺序">
        {STEPS.map((s, i) => (
          <span key={s} className="flex items-center gap-1">
            {i > 0 && <span className="label">→</span>}
            <span className={`px-1.5 py-0.5 rounded ${i === at ? STAGE[s].chip + " font-semibold" : i < at ? "text-ink-dim" : "text-ink-mute"}`}>
              {STAGE[s].label}
            </span>
          </span>
        ))}
      </div>
      <span className="label ml-auto tnum">
        融资余额 {fixed(d.latest.balance, 0)} 亿 · 占流通市值 {fixed(d.latest.ratio, 2)}% · 融资买入占成交 {fixed(d.latest.buy_share, 2)}%
      </span>
    </div>
  );
}

function Layer({ l }: { l: Cycle["layers"][number] }) {
  const mark = ["①", "②", "③", "④"][Number(l.key) - 1];
  return (
    <section className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold">{mark} {l.title}</span>
        <span className="label tnum">{l.hold}/{l.of}</span>
      </div>
      {l.items.map((x) => (
        <div key={x.key} className="flex gap-2 text-[12.5px] leading-snug" title={x.rule}>
          <span className={`w-4 shrink-0 text-center font-semibold ${x.ok ? "text-brand-ink" : "text-ink-mute"}`}>
            {x.ok == null ? "—" : x.ok ? "✓" : "✗"}
          </span>
          <div className="min-w-0">
            <div className={x.ok ? "font-medium" : "text-ink-dim"}>{x.label}</div>
            <div className="label tnum break-words">
              {x.value ?? "—"}{x.ok && x.since ? ` · 自 ${x.since}` : ""}
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}

// ── charts ───────────────────────────────────────────────────────────────────
function Charts({ d }: { d: Cycle }) {
  const [range, setRange] = useState<number>(750);
  const s = d.series;
  const from = range ? Math.max(0, s.dates.length - range) : 0;
  const cut = <T,>(a: T[]) => a.slice(from);
  const dates = cut(s.dates);
  const stage = s.stage.slice(from);
  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-[13px] font-semibold">中证全指 vs 融资余额</span>
        <span className="label">底部色带 = 当天的阶段</span>
        <div className="ml-auto flex rounded-md bg-sunken p-0.5">
          {RANGES.map((r) => (
            <button key={r.k} onClick={() => setRange(r.k)}
              className={`h-6 px-2 rounded text-[12px] ${range === r.k ? "bg-panel shadow-sm font-semibold" : "text-ink-mute"}`}>{r.label}</button>
          ))}
        </div>
      </div>
      <Dual dates={dates} a={cut(s.index)} b={cut(s.balance)} stage={stage} />
      <div className="grid gap-2 grid-cols-1 md:grid-cols-3">
        <Strip title="融资余额 / 流通市值" unit="%" dates={dates} v={cut(s.ratio)} bands={d.bands.ratio} nd={2} />
        <Strip title="融资买入 / 成交额（5 日均）" unit="%" dates={dates} v={cut(s.buy_share)} bands={d.bands.buy_share} nd={2} />
        <Flows dates={dates} v={cut(s.etf)} />
      </div>
    </section>
  );
}

function scale(vs: (number | null)[], lo: number, hi: number, log = false) {
  const t = (v: number) => (log ? Math.log(v) : v);
  const ok = vs.filter((v): v is number => v != null && (!log || v > 0));
  const a = Math.min(...ok), b = Math.max(...ok);
  return (v: number) => hi - ((t(v) - t(a)) / (t(b) - t(a) || 1)) * (hi - lo);
}

function path(vs: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let out = "", pen = false;
  vs.forEach((v, i) => { if (v == null) { pen = false; return; } out += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; pen = true; });
  return out;
}

function Dual({ dates, a, b, stage }: { dates: string[]; a: (number | null)[]; b: (number | null)[]; stage: string }) {
  const W = 1000, H = 230, BAND = 10;
  const n = dates.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const ya = scale(a, 8, H - BAND - 8, true);
  const yb = scale(b, 8, H - BAND - 8);
  const [hover, setHover] = useState<number | null>(null);
  const runs = useMemo(() => {
    const out: { s: CycleStage; i: number; j: number }[] = [];
    for (let i = 0; i < stage.length; i++) {
      const c = CODE[stage[i] ?? "-"] ?? "none";
      const last = out[out.length - 1];
      if (last && last.s === c) last.j = i; else out.push({ s: c, i, j: i });
    }
    return out.filter((r) => r.s !== "none");
  }, [stage]);
  const h = hover ?? n - 1;
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[230px]" preserveAspectRatio="none"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))));
        }}
        onMouseLeave={() => setHover(null)}>
        {runs.map((r) => (
          <rect key={r.i} x={x(r.i)} width={Math.max(1.5, x(r.j) - x(r.i) + W / n)} y={H - BAND} height={BAND}
            fill={STAGE[r.s].color} opacity={0.85} />
        ))}
        <path d={path(b, x, yb)} fill="none" stroke="var(--color-brand)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
        <path d={path(a, x, ya)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
        {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />}
      </svg>
      <div className="flex flex-wrap gap-x-3 text-[12px] tnum">
        <span className="label">{dates[h]}</span>
        <span><span className="inline-block w-3 h-0.5 align-middle mr-1 bg-ink-dim" />中证全指 {fixed(a[h], 0)}</span>
        <span><span className="inline-block w-3 h-0.5 align-middle mr-1 bg-brand" />融资余额 {fixed(b[h], 0)} 亿</span>
        <span className="label">{STAGE[CODE[stage[h] ?? "-"] ?? "none"].label}</span>
        <span className="label ml-auto">
          {STEPS.map((s) => (
            <span key={s} className="ml-2"><span className="inline-block w-2.5 h-2.5 rounded-sm align-middle mr-1" style={{ background: STAGE[s].color }} />{STAGE[s].label}</span>
          ))}
        </span>
      </div>
    </div>
  );
}

function Strip({ title, unit, dates, v, bands, nd }: {
  title: string; unit: string; dates: string[]; v: (number | null)[];
  bands: Record<string, number | null>; nd: number;
}) {
  const W = 300, H = 90;
  const n = v.length;
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const lines = [bands["20"], bands["80"]].filter((b): b is number => b != null);
  const y = scale([...v, ...lines], 6, H - 6);
  const last = [...v].reverse().find((t) => t != null) ?? null;
  return (
    <div className="rounded-lg bg-sunken p-2">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">{title}</span>
        <span className="ml-auto text-[13px] font-semibold tnum">{fixed(last, nd)}{unit}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[90px]" preserveAspectRatio="none">
        {lines.map((b, i) => (
          <line key={i} x1={0} x2={W} y1={y(b)} y2={y(b)} stroke="var(--color-line-bright)" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
        ))}
        <path d={path(v, x, y)} fill="none" stroke="var(--color-ink-dim)" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="label">虚线 = 2014 年以来 20 / 80 分位（{fixed(bands["20"], nd)}{unit} / {fixed(bands["80"], nd)}{unit}）· 起 {dates[0]}</div>
    </div>
  );
}

function Flows({ dates, v }: { dates: string[]; v: (number | null)[] }) {
  // Weekly sums: a daily bar over three years is too thin to read.
  const weeks = useMemo(() => {
    const out: { d: string; v: number }[] = [];
    let acc = 0, k = 0, d0 = "";
    v.forEach((x, i) => {
      if (k === 0) d0 = dates[i] ?? "";
      acc += x ?? 0; k++;
      if (k === 5 || i === v.length - 1) { out.push({ d: d0, v: acc }); acc = 0; k = 0; }
    });
    return out;
  }, [dates, v]);
  const W = 300, H = 90, mid = H / 2;
  const max = Math.max(1, ...weeks.map((w) => Math.abs(w.v)));
  const bw = W / Math.max(1, weeks.length);
  const last20 = v.slice(-20).reduce<number>((a, b) => a + (b ?? 0), 0);
  return (
    <div className="rounded-lg bg-sunken p-2">
      <div className="flex items-baseline gap-2">
        <span className="text-[12.5px] font-medium">宽基 ETF 净申购（周）</span>
        <span className={`ml-auto text-[13px] font-semibold tnum ${last20 >= 0 ? "text-up" : "text-down"}`}>20 日 {signed(last20, 0)} 亿</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[90px]" preserveAspectRatio="none">
        <line x1={0} x2={W} y1={mid} y2={mid} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
        {weeks.map((w, i) => {
          const hgt = (Math.abs(w.v) / max) * (mid - 2);
          return <rect key={i} x={i * bw} width={Math.max(0.8, bw - 0.6)} y={w.v >= 0 ? mid - hgt : mid} height={hgt}
            fill={w.v >= 0 ? "var(--color-up)" : "var(--color-down)"} opacity={0.8}><title>{`${w.d} 起一周 ${signed(w.v, 0)} 亿`}</title></rect>;
        })}
      </svg>
      <div className="label">红 = 净申购（流入），绿 = 净赎回 · 起 {dates[0]}</div>
    </div>
  );
}

// ── the record ───────────────────────────────────────────────────────────────
function Record({ d }: { d: Cycle }) {
  const [open, setOpen] = useState(false);
  const e = d.episodes;
  const row = (label: string, x: { median63: number | null; up63: number | null; median126: number | null }, n?: number, cls = "") => (
    <tr key={label} className={`border-t border-line ${cls}`}>
      <td className="py-1 pr-2 whitespace-nowrap">{label}</td>
      <td className="py-1 pr-2 text-right">{n ?? "—"}</td>
      <td className="py-1 pr-2 text-right">{signed(x.median63, 1, "%")}</td>
      <td className="py-1 pr-2 text-right">{x.up63 != null ? `${fixed(x.up63 * 100, 0)}%` : "—"}</td>
      <td className="py-1 text-right">{signed(x.median126, 1, "%")}</td>
    </tr>
  );
  return (
    <section className="flex flex-col gap-1">
      <div className="flex items-baseline gap-2">
        <span className="text-[13px] font-semibold">历史上进入这一阶段之后</span>
        <span className="label">2014 年以来；同一阶段相隔 20 个交易日以上才算再次进入；中证全指此后的涨跌</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="font-normal">阶段</th><th className="font-normal text-right">次数</th>
              <th className="font-normal text-right">3 个月中位</th><th className="font-normal text-right">3 个月上涨占比</th>
              <th className="font-normal text-right">6 个月中位</th>
            </tr>
          </thead>
          <tbody>
            {STEPS.map((s) => row(STAGE[s].label, e[s], e[s].n, s === d.stage ? "font-semibold" : ""))}
            {row("任意一天（对照）", e.base, undefined, "text-ink-dim")}
          </tbody>
        </table>
      </div>
      <p className="label leading-snug">
        阶段按顺序成立：底部区域要求此前 120 个交易日内出现过杠杆出清，右侧确认要求此前出现过底部区域。
        这样的右侧确认少而准；代价是 2024 年以来的几次大底（2024-02、9·24、2025-04）都只走到「杠杆出清」——
        它们是资金承接驱动的 V 形底：杠杆浓度没有进入低位，融资余额和指数同时创新低（强平），没有底背离，
        但 ④ 承接和 ③ 的「由降转升」在低点之后很快成立。遇到这种情况，单看 ④ 比看阶段更早。
      </p>
      <button onClick={() => setOpen(!open)} className="text-[12px] text-cyan self-start">{open ? "收起" : "看每一次的日期"}</button>
      {open && (
        <div className="grid gap-2 grid-cols-1 md:grid-cols-3 text-[12px] tnum">
          {STEPS.map((s) => (
            <div key={s} className="rounded-lg bg-sunken p-2">
              <div className="font-medium mb-1">{STAGE[s].label}</div>
              {e[s].list.map((x) => (
                <div key={x.date} className="flex gap-2">
                  <span className="label">{x.date}</span>
                  <span className="ml-auto">{x.r63 == null ? "进行中" : `${signed(x.r63, 1, "%")} / ${signed(x.r126, 1, "%")}`}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
