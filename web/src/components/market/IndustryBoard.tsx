/**
 * 行业与主题 — where each industry stands, read through its ETF or index.
 *
 * Two markets on one board: US industry ETFs (Yahoo) and the A-share theme
 * indices the big theme ETFs track (Tushare), read with the same arithmetic.
 * The watcher's own 自建指数 for that market sit on top as 我的指数, with the
 * same readings — but outside the breadth count, which describes the market.
 *
 * One stock against its 200-day says something about the stock; the
 * industry's ETF against its 200-day says whether the whole group is in a
 * structural uptrend, stretched, or broken. Read top to bottom: the breadth
 * summary and what it means, then every ETF with its level, structure,
 * stretch for ITSELF, and — at the extremes — what followed in the past.
 * Click a row for five years of it.
 *
 * Colours follow the market: A-shares red up, US green up. The stretch
 * badges are amber and blue on purpose — "very extended" is a risk note,
 * not a direction.
 */

import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import type { BoardMarket, EtfRow, MineBoard, UsEtfBoard, UsEtfDetail } from "../../lib/types";
import { fixed, moveClass, signed } from "../../lib/format";
import { usePersistentState } from "../../lib/usePersistentState";

const MINE = "我的指数";

// Colours follow the market. Read from context so every small chart in a
// row agrees with the board it is on.
const Tone = createContext<{ red: boolean }>({ red: false });
function useTone() {
  const { red } = useContext(Tone);
  return { red, UP: red ? "var(--color-up)" : "var(--color-down)", DOWN: red ? "var(--color-down)" : "var(--color-up)" };
}

/** An ETF is known by its ticker; an index code like H30184.CSI is not. */
const isTicker = (t: string) => /^[A-Z]{1,5}$/.test(t);

type SortKey = "group" | "d200" | "stretch" | "3m" | "rs" | "high";
const SORTS: { v: SortKey; label: string }[] = [
  { v: "group", label: "按类别" }, { v: "d200", label: "偏离200日" }, { v: "stretch", label: "拉伸分位" },
  { v: "3m", label: "3个月涨跌" }, { v: "rs", label: "相对基准" }, { v: "high", label: "距52周高" },
];

const STATE_CLASS: Record<string, string> = {
  多头: "bg-down/15 text-down", 回调: "bg-brand/15 text-brand-ink",
  反弹: "bg-cyan/15 text-cyan", 空头: "bg-up/15 text-up",
};
const STATE_HINT: Record<string, string> = {
  多头: "在 50 日和 200 日均线上方", 回调: "仍在 200 日线上方，但跌破了 50 日线",
  反弹: "站上 50 日线，但仍在 200 日线下方", 空头: "在 50 日和 200 日均线下方",
};

export function IndustryBoard({ d, mine, mineLoading, mineError, market }: {
  d: UsEtfBoard; mine?: MineBoard; mineLoading?: boolean; mineError?: string | null; market: BoardMarket;
}) {
  const [picked, setGroup] = usePersistentState<string>(`assrs.board.group.${market}`, "全部");
  const [sort, setSort] = usePersistentState<SortKey>("assrs.etf.sort", "group");
  const [open, setOpen] = useState<string | null>(null);
  // The table scrolls sideways on narrow screens; the opened detail is pinned
  // to the visible width so its charts are never half off-screen.
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const set = () => el.style.setProperty("--etf-w", `${el.clientWidth}px`);
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const own = mine?.rows ?? [];
  const groups = own.length ? [MINE, ...d.groups] : d.groups;
  const group = groups.includes(picked) ? picked : "全部";
  const rows = useMemo(() => {
    const r = [...own, ...d.rows].filter((x) => group === "全部" || x.group === group);
    const key: Record<SortKey, (x: EtfRow) => number> = {
      group: (x) => groups.indexOf(x.group) * 1000 - (x.d200 ?? 0) / 1000,
      d200: (x) => -(x.d200 ?? -999), stretch: (x) => -(x.stretch_rank ?? -1),
      "3m": (x) => -(x.ret["3m"] ?? -999), rs: (x) => -(x.rs_3m ?? -999), high: (x) => -(x.from_high ?? -999),
    };
    return [...r].sort((a, b) => key[sort](a) - key[sort](b));
  }, [d, own, groups, group, sort]);

  const b = d.breadth;
  const noun = market === "US" ? "行业 ETF" : "行业与主题指数";
  return (
    <Tone.Provider value={{ red: market === "CN" }}>
    <div className="flex flex-col gap-3">
      <div className="grid gap-2 grid-cols-2 lg:grid-cols-5">
        <Tile k="200 日线上方" v={`${b.above200}/${b.n}`} share={b.above200 / b.n} hint="长期趋势向上的行业" />
        <Tile k="50 日线上方" v={`${b.above50}/${b.n}`} share={b.above50 / b.n} hint="短期仍在走强的行业" />
        <Tile k="50/200 金叉" v={`${b.golden}/${b.n}`} share={b.golden / b.n} hint="50 日线在 200 日线上方" />
        <Tile k="距 52 周高点 3% 内" v={`${b.near_high}/${b.n}`} share={b.near_high / b.n} hint="正在创新高附近" />
        <Tile k="偏离 200 日线（中位）" v={signed(b.median_d200, 1, "%")} hint={`全部${noun}的中位数`} />
      </div>
      <div className="rounded-lg bg-sunken p-3 flex flex-col gap-1">
        <span className="text-[13px] font-semibold">结论
          <span className="label font-normal ml-2">数据截至 {d.as_of}{d.session === "live" ? " · 盘中（延迟 15 分钟）" : " · 收盘"}</span></span>
        <ul className="flex flex-col gap-1 list-disc pl-4">
          {d.conclusions.map((c, i) => <li key={i} className="text-[12.5px] leading-snug text-ink-dim">{c}</li>)}
        </ul>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap rounded-md bg-sunken p-0.5">
          {["全部", ...groups].map((g) => (
            <button key={g} onClick={() => setGroup(g)}
              className={`h-6 px-2.5 rounded text-[12px] ${group === g ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>{g}</button>
          ))}
        </div>
        <label className="label flex items-center gap-1 ml-auto">排序
          <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}
            className="h-7 px-1.5 rounded-md bg-sunken text-[12px] outline-none">
            {SORTS.map((s) => <option key={s.v} value={s.v}>{s.label}</option>)}
          </select>
        </label>
      </div>

      <div ref={scroller} className="overflow-x-auto">
        <table className="w-full min-w-[1080px] text-[12.5px] tnum">
          <thead>
            <tr className="label text-left">
              <th className="sticky left-0 z-10 bg-panel font-normal py-1">{market === "US" ? "ETF" : "指数"}</th>
              <th className="font-normal text-right pr-2">价格</th>
              <th className="font-normal">一年走势 · 虚线 200 日线</th>
              <th className="font-normal" title="价格离 200 日均线多远；徽章 = 这个偏离在它自己 10 年里的分位">偏离 200 日线</th>
              <th className="font-normal text-right pr-2">偏离 50 日</th>
              <th className="font-normal">趋势</th>
              <th className="font-normal" title="在过去 52 周最低—最高之间的位置">52 周区间</th>
              <th className="font-normal text-right pr-2" title="20 日年化波动；括号 = 在它自己近 3 年里的分位">波动</th>
              <th className="font-normal text-right pr-2">3 个月</th>
              <th className="font-normal text-right pr-2" title={`3 个月相对 ${d.bench_name ?? "基准"} 的超额`}>相对 {d.bench_name ?? "基准"}</th>
              <th className="font-normal" title="只在拉伸到自身 80 分位以上或 20 分位以下时给出：过去同样拉伸之后 3 个月的表现">历史同等拉伸后</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <EtfRowView key={r.ticker} r={r} market={market} open={open === r.ticker}
                groupBreak={sort === "group" && (i === 0 || rows[i - 1]!.group !== r.group)}
                onToggle={() => setOpen(open === r.ticker ? null : r.ticker)} />
            ))}
          </tbody>
        </table>
      </div>
      <MineNote mine={mine} loading={mineLoading} error={mineError} market={market} />
      <p className="label leading-snug">
        {market === "US"
          ? "价格为 Yahoo 复权数据（含分红拆股）。"
          : "中证 / 国证指数日线来自 Tushare（价格指数，不含分红），收盘后更新；这些是主流主题 ETF 跟踪的指数。"}
        拉伸分位 = 今天离 200 日线的距离，在它自己最多 10 年历史中的分位：
        +15% 对半导体平常、对公用事业就是极端，所以只和自己比；不足 2 年历史的不计分位。
        起点较近、会定期调仓的自建指数，用同样规则往前回溯最多 10 年来读这些数（标「回溯自」）。
        「历史同等拉伸后」按段计算（连续拉伸的日子算一段），统计每段之后 63 个交易日的涨跌；段数很少时只是参照。
        {market === "US" ? "美股配色：绿涨红跌。" : "A 股配色：红涨绿跌。"}
        「我的指数」不计入上方的广度统计。
      </p>
    </div>
    </Tone.Provider>
  );
}

function MineNote({ mine, loading, error, market }: {
  mine?: MineBoard; loading?: boolean; error?: string | null; market: BoardMarket;
}) {
  if (error) return <p className="label">我的指数暂不可用：{error}</p>;
  // Each index is read over up to ten years of its members' prices — the first load takes a while.
  if (loading) return <p className="label animate-pulse">正在计算我的{market === "US" ? "美股" : " A 股"}指数（首次读取十年行情，约半分钟）…</p>;
  if (!mine) return null;
  const parts: string[] = [];
  if (mine.young.length) parts.push(`不足 220 个交易日、暂不上榜：${mine.young.map((y) => `${y.name}（${y.days} 天）`).join("、")}`
    + (mine.young.some((y) => y.hold) ? "（买入持有不会自动往前回溯 —— 起点不同就是另一个组合；把起始日提前即可）" : ""));
  if (mine.failed.length) parts.push(`计算失败：${mine.failed.map((f) => `${f.name}（${f.error}）`).join("、")}`);
  if (!mine.rows.length && !mine.young.length && !mine.failed.length) {
    return (
      <p className="label">
        在 <Link to="/portfolio?tab=index" className="text-cyan">💼 组合 → 自建组合&指数</Link> 建立你自己的{market === "US" ? "美股" : "A 股"}指数，
        它会以「我的指数」出现在这张表的最上面，用同样的方式读。
      </p>
    );
  }
  return parts.length ? <p className="label leading-snug">我的指数：{parts.join("；")}。</p> : null;
}

function Tile({ k, v, share, hint }: { k: string; v: string; share?: number; hint: string }) {
  const { UP, DOWN } = useTone();
  return (
    <div className="rounded-lg bg-sunken px-3 py-2" title={hint}>
      <div className="label">{k}</div>
      <div className="text-[18px] font-semibold tnum">{v}</div>
      {share != null && (
        <div className="h-1.5 rounded-full bg-panel overflow-hidden mt-1">
          <div className="h-full" style={{ width: `${share * 100}%`, background: share >= 0.5 ? UP : DOWN }} />
        </div>
      )}
    </div>
  );
}

function StretchBadge({ rank }: { rank: number | null }) {
  if (rank == null) return null;
  const cls = rank >= 90 ? "bg-brand/20 text-brand-ink font-semibold" : rank <= 10 ? "bg-cyan/15 text-cyan font-semibold"
    : "bg-sunken text-ink-mute";
  const word = rank >= 90 ? "过热" : rank <= 10 ? "超跌" : "";
  return <span className={`px-1.5 rounded text-[11px] whitespace-nowrap ${cls}`} title="今天的偏离在它自己历史中的分位">{rank.toFixed(0)}分位{word && ` ${word}`}</span>;
}

function EtfRowView({ r, market, open, onToggle, groupBreak }: {
  r: EtfRow; market: BoardMarket; open: boolean; onToggle: () => void; groupBreak: boolean;
}) {
  const { red } = useTone();
  const a = r.after;
  return (
    <>
      {groupBreak && (
        <tr><td colSpan={11} className="pt-3 pb-1 text-[12px] font-semibold text-ink-dim"><div className="sticky left-0 w-fit">{r.group}</div></td></tr>
      )}
      <tr onClick={onToggle} className={`border-t border-line cursor-pointer ${open ? "bg-cyan/5" : "hover:bg-elevated"}`}>
        <td className="sticky left-0 z-10 bg-panel py-1.5 pr-2 whitespace-nowrap">
          {r.index_id != null ? (
            <><span className="font-semibold">{r.name}</span> <span className="label">自建 · {r.method}{r.extended_from ? ` · 回溯自 ${r.extended_from.slice(0, 4)}` : ""}</span></>
          ) : isTicker(r.ticker) ? (
            <><span className="font-mono font-semibold">{r.ticker}</span> <span className="text-ink-dim">{r.name}</span></>
          ) : (
            <><span className="font-semibold">{r.name}</span> <span className="font-mono text-[11px] text-ink-mute">{r.ticker}</span></>
          )}
        </td>
        <td className="py-1.5 pr-2 text-right whitespace-nowrap">
          <div>{fixed(r.price, 2)}</div>
          <div className={`text-[11px] ${moveClass(r.ret["1d"], red)}`}>{signed(r.ret["1d"], 1, "%")}</div>
        </td>
        <td className="py-1.5 pr-2"><Spark r={r} /></td>
        <td className="py-1.5 pr-2">
          <div className="flex items-center gap-1.5">
            <Diverge v={r.d200} max={30} />
            <span className={`w-12 text-right ${moveClass(r.d200, red)}`}>{signed(r.d200, 1, "%")}</span>
            <StretchBadge rank={r.stretch_rank} />
          </div>
        </td>
        <td className={`py-1.5 pr-2 text-right ${moveClass(r.d50, red)}`}>{signed(r.d50, 1, "%")}</td>
        <td className="py-1.5 pr-2 whitespace-nowrap">
          <span title={STATE_HINT[r.state]} className={`px-1.5 py-0.5 rounded-full text-[11.5px] font-semibold ${STATE_CLASS[r.state] ?? ""}`}>{r.state}</span>
          <span className="label ml-1" title="50 日线与 200 日线的交叉">
            {r.golden ? "金叉" : "死叉"}{r.cross_days != null ? ` ${r.cross_days}天` : ""}
            {r.slope200 != null && <> · 200日线{r.slope200 >= 0 ? "↗" : "↘"}</>}
          </span>
        </td>
        <td className="py-1.5 pr-2">
          <div className="flex items-center gap-1.5">
            <span className="relative w-16 h-1.5 rounded-full bg-sunken">
              {r.range_pos != null && <span className="absolute top-[-2px] w-1.5 h-2.5 rounded-sm bg-ink"
                style={{ left: `calc(${r.range_pos * 100}% - 3px)` }} />}
            </span>
            <span className={`text-[11.5px] ${moveClass(r.from_high, red)}`} title="距 52 周高点">{signed(r.from_high, 1, "%")}</span>
          </div>
        </td>
        <td className="py-1.5 pr-2 text-right whitespace-nowrap">
          {fixed(r.vol20, 0)}%<span className={`label ${(r.vol_rank ?? 0) >= 80 ? "text-brand-ink" : ""}`}> ({fixed(r.vol_rank, 0)})</span>
        </td>
        <td className={`py-1.5 pr-2 text-right ${moveClass(r.ret["3m"], red)}`}>{signed(r.ret["3m"], 1, "%")}</td>
        <td className={`py-1.5 pr-2 text-right ${moveClass(r.rs_3m, red)}`}>
          {signed(r.rs_3m, 1, "%")}
          {r.rs_vs && <div className="label text-[10.5px]" title="自建指数和它自己选的对比指数比">vs {r.rs_vs}</div>}
        </td>
        <td className="py-1.5 whitespace-nowrap text-[11.5px]">
          {a ? (
            <span title={`${a.episodes} 段：${a.dates.join("、")}`}>
              3月中位 <b className={moveClass(a.median, red)}>{signed(a.median, 1, "%")}</b>
              <span className="label"> · {a.episodes}次 · 平常 {signed(a.base_median, 1, "%")}</span>
            </span>
          ) : (r.stretch_rank != null && (r.stretch_rank >= 97 || r.stretch_rank <= 3))
            ? <span className="text-brand-ink" title="历史上不足 3 段可比">无先例</span>
            : r.stretch_rank == null ? <span className="label" title="历史不足 2 年，不和自己比">历史不足</span>
            : <span className="label">—</span>}
        </td>
      </tr>
      {open && (
        <tr><td colSpan={11} className="pb-3">
          <div className="sticky left-0" style={{ width: "var(--etf-w, 100%)" }}><Detail r={r} market={market} /></div>
        </td></tr>
      )}
    </>
  );
}

function Diverge({ v, max }: { v: number | null; max: number }) {
  const { UP, DOWN } = useTone();
  const w = v == null ? 0 : Math.min(1, Math.abs(v) / max) * 50;
  return (
    <span className="relative w-20 h-2 rounded-sm bg-sunken inline-block">
      <span className="absolute top-0 bottom-0 w-px bg-line-bright" style={{ left: "50%" }} />
      {v != null && <span className="absolute top-0 bottom-0 rounded-sm"
        style={{ left: v >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: v >= 0 ? UP : DOWN, opacity: 0.8 }} />}
    </span>
  );
}

function Spark({ r }: { r: EtfRow }) {
  const { UP, DOWN } = useTone();
  const W = 140, H = 30;
  const vals = [...r.spark, ...r.spark_ma200].filter((v): v is number => v != null);
  if (!vals.length) return null;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const n = r.spark.length;
  const x = (i: number) => (n < 2 ? 0 : (i / (n - 1)) * W);
  const y = (v: number) => H - 2 - ((v - lo) / (hi - lo || 1)) * (H - 4);
  const path = (vs: (number | null)[]) => vs.map((v, i) => (v == null ? "" : `${i && vs[i - 1] != null ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`)).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-[140px] h-[30px] block" role="img" aria-label={`${r.ticker} 一年走势`}>
      <path d={path(r.spark_ma200)} fill="none" stroke="var(--color-ink-mute)" strokeDasharray="3 2" strokeWidth={1} />
      <path d={path(r.spark)} fill="none" stroke={(r.d200 ?? 0) >= 0 ? UP : DOWN} strokeWidth={1.4} />
    </svg>
  );
}

function Detail({ r, market }: { r: EtfRow; market: BoardMarket }) {
  const q = useQuery({
    queryKey: ["board-detail", market, r.ticker],
    queryFn: () => (r.index_id != null ? api.indexStretch(r.index_id)
      : market === "CN" ? api.cnIndexDetail(r.ticker) : api.usEtfDetail(r.ticker)),
    staleTime: 30 * 60_000,
  });
  if (q.isPending) return <p className="label py-6 text-center">读取 {r.name} 的走势…</p>;
  if (q.isError) return <p className="text-[12.5px] text-up">{(q.error as Error).message}</p>;
  const d = q.data!;
  return (
    <div className="flex flex-col gap-1 pt-2">
    {d.extended_from && (
      <p className="label leading-snug">
        {d.extended_from} 至 {d.start} 是用同样规则、按现在的创始成分往前回溯的假设走势（有事后选股的偏差），
        只为给 200 日线和拉伸分位足够的历史；{d.start} 之后的第一次调仓起，与指数本身逐日一致。
      </p>
    )}
    <div className="grid gap-3 lg:grid-cols-2">
      <PriceChart d={d} />
      <StretchChart d={d} r={r} />
    </div>
    </div>
  );
}

function useHover(n: number) {
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const onMove = (e: React.MouseEvent) => {
    const rc = box.current?.getBoundingClientRect();
    if (rc) setHover(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - rc.left) / rc.width) * (n - 1)))));
  };
  return { box, hover, onMove, clear: () => setHover(null) };
}

function linePath(vs: (number | null)[], x: (i: number) => number, y: (v: number) => number) {
  let s = "", pen = false;
  vs.forEach((v, i) => { if (v == null) { pen = false; return; } s += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`; pen = true; });
  return s;
}

function PriceChart({ d }: { d: UsEtfDetail }) {
  const W = 1000, H = 200, n = d.dates.length;
  const { box, hover, onMove, clear } = useHover(n);
  const vals = [...d.close, ...d.ma200].filter((v): v is number => v != null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const y = (v: number) => H - ((v - lo) / (hi - lo || 1)) * H;
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-center gap-3 text-[12px]">
        <span className="font-semibold">{isTicker(d.ticker) ? `${d.ticker} ${d.name}` : d.name} · {fixed(n / 252, 1)} 年</span>
        <span className="flex items-center gap-1"><span className="w-4 h-[2px] bg-ink" />价格</span>
        <span className="flex items-center gap-1"><span className="w-4 h-[2px]" style={{ background: "#0062cc" }} />50 日线</span>
        <span className="flex items-center gap-1"><span className="w-4 border-t-2 border-dashed" style={{ borderColor: "#b45309" }} />200 日线</span>
      </div>
      <div ref={box} onMouseMove={onMove} onMouseLeave={clear} className="relative h-[200px]">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img" aria-label={`${d.ticker} 五年价格与均线`}>
          <path d={linePath(d.ma200, x, y)} fill="none" stroke="#b45309" strokeDasharray="6 4" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
          <path d={linePath(d.ma50, x, y)} fill="none" stroke="#0062cc" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
          <path d={linePath(d.close, x, y)} fill="none" stroke="var(--color-ink)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
        </svg>
        {hover != null && (
          <div className="absolute top-1 card px-2 py-1 text-[12px] pointer-events-none z-10"
            style={hover > n * 0.6 ? { right: `${100 - (hover / Math.max(1, n - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, n - 1)) * 100 + 1}%` }}>
            <div className="font-mono">{d.dates[hover]}</div>
            <div>价格 {fixed(d.close[hover], 2)}</div>
            <div className="text-ink-mute">50日 {fixed(d.ma50[hover], 2)} · 200日 {fixed(d.ma200[hover], 2)}</div>
          </div>
        )}
      </div>
      <div className="flex justify-between label font-mono tnum"><span>{d.dates[0]}</span><span>{d.dates[n - 1]}</span></div>
    </div>
  );
}

function StretchChart({ d, r }: { d: UsEtfDetail; r: EtfRow }) {
  const { UP, DOWN } = useTone();
  const W = 1000, H = 200, n = d.dates.length;
  const { box, hover, onMove, clear } = useHover(n);
  const vals = d.stretch.filter((v): v is number => v != null);
  const lo = Math.min(0, ...vals, d.bands["10"] ?? 0), hi = Math.max(0, ...vals, d.bands["90"] ?? 0);
  const x = (i: number) => (i / Math.max(1, n - 1)) * W;
  const y = (v: number) => H - ((v - lo) / (hi - lo || 1)) * H;
  // Bands from under two years of history are one regime's range, not a norm.
  const ranked = r.stretch_rank != null;
  const band = (k: "10" | "50" | "90", label: string, colour: string) => {
    const v = ranked ? d.bands[k] : null;
    return v == null ? null : (
      <g key={k}>
        <line x1={0} x2={W} y1={y(v)} y2={y(v)} stroke={colour} strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
        <title>{label} {v}%</title>
      </g>
    );
  };
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        <span className="font-semibold">偏离 200 日线</span>
        <span className="label">{ranked && <>虚线 = 它自己历史的 10 / 50 / 90 分位（{fixed(d.bands["10"], 0)}% / {fixed(d.bands["50"], 0)}% / {fixed(d.bands["90"], 0)}%）· </>}今天 {signed(r.d200, 1, "%")}{r.stretch_rank != null ? `，${fixed(r.stretch_rank, 0)} 分位` : "（历史不足 2 年，不计分位）"}</span>
      </div>
      <div ref={box} onMouseMove={onMove} onMouseLeave={clear} className="relative h-[200px]">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full" role="img" aria-label={`${d.ticker} 偏离 200 日线`}>
          <rect x={0} y={y(hi)} width={W} height={y(Math.max(0, lo)) - y(hi)} fill={UP} opacity={0.04} />
          <rect x={0} y={y(0)} width={W} height={H - y(0)} fill={DOWN} opacity={0.04} />
          <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--color-line-bright)" vectorEffect="non-scaling-stroke" />
          {band("90", "90 分位", "#b45309")}{band("50", "中位", "var(--color-ink-mute)")}{band("10", "10 分位", "#0062cc")}
          <path d={linePath(d.stretch, x, y)} fill="none" stroke="var(--color-ink)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
          {hover != null && <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-ink-mute)" vectorEffect="non-scaling-stroke" />}
        </svg>
        {hover != null && (
          <div className="absolute top-1 card px-2 py-1 text-[12px] pointer-events-none z-10"
            style={hover > n * 0.6 ? { right: `${100 - (hover / Math.max(1, n - 1)) * 100 + 1}%` } : { left: `${(hover / Math.max(1, n - 1)) * 100 + 1}%` }}>
            <div className="font-mono">{d.dates[hover]}</div>
            <div>偏离 {signed(d.stretch[hover], 1, "%")}</div>
          </div>
        )}
      </div>
      <div className="flex justify-between label font-mono tnum"><span>{d.dates[0]}</span><span>{d.dates[n - 1]}</span></div>
    </div>
  );
}
