/**
 * 市场看板 — the whole market before any single stock.
 *
 * Four tabs, one question each, under a 一览 strip that is on every tab:
 *
 *   今日         what happened today?          全球指数 · 市场风险 · 热力图 · 龙虎榜
 *   行业与轮动    where is each sector?         行业与主题 · 轮动图 · 威科夫
 *   资金与信用    how much of it is borrowed?   杠杆周期 · 信用与债务 · 宏观
 *   商品         what are commodities saying?  期限结构
 *
 * The 一览 strip carries the three verdicts worth seeing whatever tab is
 * open — today's risk flags, the leverage-cycle stage, US credit — and each
 * jumps to its tab. Only the open tab's panels are fetched.
 *
 * The heatmap and the trend grid share one slot because they answer the same
 * question at two horizons — today's move and the last three months of it.
 *
 * Every panel is its own query. A Tushare hiccup that kills 龙虎榜 must not
 * take the rotation map with it, so each one fails inside its own card and the
 * rest of the page stays usable.
 */

import { useEffect, useState, type ReactNode } from "react";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { api } from "../lib/api";
import { NavBar } from "../components/NavBar";
import { Heatmap } from "../components/market/Heatmap";
import { BreadthGrid } from "../components/market/BreadthGrid";
import { RotationMap } from "../components/market/RotationMap";
import { WyckoffPanel } from "../components/market/WyckoffPanel";
import { LeveragePanel } from "../components/market/LeveragePanel";
import { LeverageCycle } from "../components/market/LeverageCycle";
import { CreditPanel } from "../components/market/CreditPanel";
import { TopList } from "../components/market/TopList";
import { IndexStrip } from "../components/market/IndexStrip";
import { RiskStrip } from "../components/market/RiskStrip";
import { MacroPanel } from "../components/market/MacroPanel";
import { CommodityPanel } from "../components/market/CommodityPanel";
import { IndustryBoard } from "../components/market/IndustryBoard";
import { usePersistentState } from "../lib/usePersistentState";
import type { BoardMarket, CreditBoard, LeverageCycle as Cycle, RiskBoard } from "../lib/types";
import { useSize } from "../lib/useSize";
import { signed } from "../lib/format";

const INDICES: Record<string, string> = {
  "000300.SH": "沪深300",
  "000905.SH": "中证500",
  "399006.SZ": "创业板指",
  "000001.SH": "上证",
};

type Tab = "today" | "sectors" | "money" | "commod";
const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "today", label: "今日", hint: "全球指数 · 市场风险 · 热力图 · 龙虎榜" },
  { id: "sectors", label: "行业与轮动", hint: "行业与主题 · 轮动图 · 威科夫阶段" },
  { id: "money", label: "资金与信用", hint: "杠杆周期 · 信用与债务 · 宏观" },
  { id: "commod", label: "商品", hint: "期货期限结构" },
];

function tabFromUrl(): Tab | null {
  const q = new URLSearchParams(window.location.search).get("tab");
  return TABS.some((t) => t.id === q) ? (q as Tab) : null;
}

export function Dashboard() {
  useEffect(() => { document.title = "ASSRS · 市场看板"; }, []);

  const [saved, setSaved] = usePersistentState<Tab>("assrs.mkt.tab", "today");
  const [tab, setTabState] = useState<Tab>(() => tabFromUrl() ?? saved);
  const setTab = (t: Tab) => {
    setTabState(t);
    setSaved(t);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", t);
    window.history.replaceState(null, "", url);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const on = (t: Tab) => tab === t;

  const [view, setView] = usePersistentState<"heatmap" | "breadth">("assrs.mkt.view", "heatmap");
  const [freq, setFreq] = usePersistentState<"w" | "d">("assrs.mkt.rotfreq", "w");
  const [index, setIndex] = usePersistentState<string>("assrs.mkt.index", "000300.SH");
  const [commodity, setCommodity] = usePersistentState<string>("assrs.mkt.comm", "CU");
  const [liquidOnly, setLiquidOnly] = usePersistentState<boolean>(
    "assrs.mkt.commliquid", true);

  // The three 一览 verdicts load on every tab; everything else only with its tab.
  const risk = useQuery({ queryKey: ["mkt", "risk"], queryFn: api.risk, staleTime: 20 * 60_000 });
  const cycle = useQuery({ queryKey: ["mkt", "leverage-cycle"], queryFn: api.leverageCycle,
    staleTime: 30 * 60_000 });
  const credit = useQuery({ queryKey: ["mkt", "credit"], queryFn: api.credit, staleTime: 60 * 60_000 });

  // On its own short clock: half these markets are open while someone looks.
  const indices = useQuery({ queryKey: ["mkt", "indices"], queryFn: api.indices,
    staleTime: 5 * 60_000, refetchInterval: 5 * 60_000, enabled: on("today") });
  const heatmap = useQuery({ queryKey: ["mkt", "heatmap"], queryFn: api.heatmap,
    staleTime: 20 * 60_000, enabled: on("today") && view === "heatmap" });
  const breadth = useQuery({ queryKey: ["mkt", "breadth"], queryFn: () => api.breadth(60),
    staleTime: 60 * 60_000, enabled: on("today") && view === "breadth" });
  const toplist = useQuery({ queryKey: ["mkt", "toplist"], queryFn: api.topList,
    staleTime: 2 * 3600_000, enabled: on("today") });

  // 行业与主题: ten years of 45 US ETFs is one ~12s Yahoo download; the A-share
  // board is one Tushare call per index. Both held 30 minutes server-side and
  // shared by every user. Only the market on screen is fetched.
  const [boardMkt, setBoardMkt] = usePersistentState<BoardMarket>("assrs.board.market", "US");
  const usEtf = useQuery({ queryKey: ["mkt", "us-etf"], queryFn: api.usEtf,
    staleTime: 30 * 60_000, enabled: on("sectors") && boardMkt === "US" });
  const cnIdx = useQuery({ queryKey: ["mkt", "cn-index"], queryFn: api.cnIndex,
    staleTime: 30 * 60_000, enabled: on("sectors") && boardMkt === "CN" });
  // The user's own indexes in that market, read from their stored records.
  const mine = useQuery({ queryKey: ["idx", "board", boardMkt], queryFn: () => api.indexBoard(boardMkt),
    staleTime: 15 * 60_000, retry: false, enabled: on("sectors") });
  const board = boardMkt === "US" ? usEtf : cnIdx;
  const rotation = useQuery({ queryKey: ["mkt", "rotation", freq],
    queryFn: () => api.rotation(freq), staleTime: 60 * 60_000, enabled: on("sectors") });
  const wyckoff = useQuery({ queryKey: ["mkt", "wyckoff", index],
    queryFn: () => api.wyckoff(index), staleTime: 30 * 60_000, enabled: on("sectors") });

  const leverage = useQuery({ queryKey: ["mkt", "leverage"], queryFn: api.leverage,
    staleTime: 20 * 60_000, enabled: on("money") });
  // Monthly and quarterly prints, and settlements struck once a day —
  // nothing here changes while the page is open.
  const macro = useQuery({ queryKey: ["mkt", "macro"], queryFn: api.macro,
    staleTime: 3600_000, enabled: on("money") });
  const commodities = useQuery({
    queryKey: ["mkt", "commodities", commodity, liquidOnly],
    queryFn: () => api.commodities(commodity, liquidOnly),
    staleTime: 3600_000, enabled: on("commod") });

  return (
    <div className="min-h-screen">
      <NavBar />
      <div className="sticky top-14 z-30 bg-canvas/90 backdrop-blur border-b border-line">
        <div className="max-w-[1800px] mx-auto px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-2">
          <nav className="flex rounded-lg bg-sunken p-0.5 overflow-x-auto max-w-full" aria-label="看板分区">
            {TABS.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)} title={t.hint}
                className={`h-8 px-3 rounded-md text-[13px] whitespace-nowrap ${
                  tab === t.id ? "bg-panel shadow-sm font-semibold text-ink" : "text-ink-mute hover:text-ink"}`}>
                {t.label}
              </button>
            ))}
          </nav>
          <Glance risk={risk.data} cycle={cycle.data} credit={credit.data} onGo={setTab} />
        </div>
      </div>
      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        {on("today") && <>
          <Panel title="🌏 全球指数"
            subtitle="A 股、亚太、欧美主要指数的最新涨跌。各市场收盘时间不同，卡片标的是它自己那根的日期。"
            q={indices}>
            {indices.data && <IndexStrip data={indices.data} />}
          </Panel>

          <Panel title="🚦 市场风险"
            subtitle="流动性、杠杆、情绪、估值与波动，各自按新行情以来的分位分级。琥珀色是风险提示，不是涨跌。"
            q={risk}>
            {risk.data && <RiskStrip data={risk.data} />}
          </Panel>

          <Panel
            title={view === "heatmap" ? "🗺️ 市场热力图" : "📊 板块趋势历史"}
            subtitle={view === "heatmap"
              ? "方块大小 = 流通市值，颜色 = 当日涨跌（红涨绿跌）。点击任一个股打开分析。"
              : "各板块指数相对自身 20 日均线的位置，越往右越新。"}
            right={
              <Segmented value={view} onChange={setView} options={[
                { id: "heatmap", label: "热力图" },
                { id: "breadth", label: "趋势历史" },
              ]} />
            }
            q={view === "heatmap" ? heatmap : breadth}>
            {view === "heatmap" && heatmap.data && <HeatmapSlot data={heatmap.data} />}
            {view === "breadth" && breadth.data && <BreadthGrid data={breadth.data} />}
          </Panel>

          <Panel title="🐉 龙虎榜"
            subtitle="触发交易所异动披露的个股，以及当日席位净买卖金额。"
            q={toplist}>
            {toplist.data && <TopList data={toplist.data} />}
          </Panel>
        </>}

        {on("sectors") && <>
          <Panel title="🌐 行业与主题"
            subtitle={`${boardMkt === "US" ? "每个美国行业 / 主题 ETF" : "每个 A 股主题与行业指数"}离 50 日、200 日均线多远，这个偏离在它自己 10 年里算不算极端，以及过去同样拉伸之后发生了什么。你的自建指数排在最上面。${boardMkt === "US" ? "绿涨红跌" : "红涨绿跌"}。点任一行看走势。`}
            right={
              <div className="flex rounded-md bg-sunken p-0.5">
                {([["US", "🇺🇸 美股 ETF"], ["CN", "🇨🇳 A 股主题"]] as const).map(([m, label]) => (
                  <button key={m} onClick={() => setBoardMkt(m)}
                    className={`h-7 px-2.5 rounded text-[12.5px] ${boardMkt === m ? "bg-panel shadow-sm font-semibold" : "text-ink-mute hover:text-ink"}`}>{label}</button>
                ))}
              </div>
            }
            q={board}>
            {board.data && <IndustryBoard key={boardMkt} d={board.data} market={boardMkt}
              mine={mine.data} mineLoading={mine.isPending}
              mineError={mine.isError ? (mine.error as Error).message : null} />}
          </Panel>

          <Panel title="🔄 板块轮动 · 相对轮动图"
            subtitle="板块相对大盘的强弱（横轴）与这份强弱本身的变化（纵轴）。按顺时针轮动：改善 → 领先 → 走弱 → 落后。"
            q={rotation}>
            {rotation.data && <RotationMap data={rotation.data} freq={freq} onFreq={setFreq} />}
          </Panel>

          <Panel title="📈 统计威科夫阶段"
            subtitle="用 120 日区间位置、波动率与成交量 Z 值定义市场阶段，不依赖均线，也不依赖眼力。"
            q={wyckoff}>
            {wyckoff.data && (
              <WyckoffPanel data={wyckoff.data} index={index} onIndex={setIndex}
                indices={INDICES} />
            )}
          </Panel>
        </>}

        {on("money") && <>
          <Panel title="💳 杠杆周期 · 出清 → 底部 → 右侧"
            subtitle="A 股两融从去杠杆到见底再到右侧确认，按四层固定规则逐条判断：存量、浓度、背离、承接。给出可验证的信号和每个阶段的历史表现，不是买卖指令。"
            q={cycle}>
            {cycle.data && <LeverageCycle d={cycle.data} />}
            {leverage.data && (
              <details className="mt-1">
                <summary className="label cursor-pointer">美股保证金借款（FINRA，月度）</summary>
                <div className="mt-2"><LeveragePanel data={leverage.data.filter((x) => x.market === "US")} /></div>
              </details>
            )}
          </Panel>

          <Panel title="🏦 信用与债务"
            subtitle="AI 资本开支背后的债务有没有变成信用紧张？美国为主：六条固定规则，各自成立与否和历史记录；中国信用周期作背景。"
            q={credit}>
            {credit.data && <CreditPanel d={credit.data} />}
          </Panel>

          <Panel title="📊 宏观"
            subtitle="通胀、增长、流动性与利率。中国数据是月度或季度的，SHIBOR 和美债是日度的；变化是与上一期相比。"
            q={macro}>
            {macro.data && <MacroPanel data={macro.data} />}
          </Panel>
        </>}

        {on("commod") && (
          <Panel title="🛢 大宗商品 · 期限结构"
            subtitle="一个期货价格本身说明不了什么，近月和远月的高低才说明问题。点任一品种看它的远期曲线。"
            q={commodities}>
            {commodities.data && (
              <CommodityPanel data={commodities.data} onCode={setCommodity}
                liquid={liquidOnly} onLiquid={setLiquidOnly} />
            )}
          </Panel>
        )}
      </main>
    </div>
  );
}

/**
 * 一览: the three verdicts worth seeing from any tab, each a chip that jumps
 * to where it is explained. Amber and ink only — these are risk readings, not
 * price moves, so never red or green.
 */
function Glance({ risk, cycle, credit, onGo }: {
  risk?: RiskBoard; cycle?: Cycle; credit?: CreditBoard; onGo: (t: Tab) => void;
}) {
  const chip = (label: string, value: string | null, warn: number, go: Tab, title: string) => (
    <button key={label} onClick={() => onGo(go)} title={title}
      className={`h-8 px-2.5 rounded-lg text-[12.5px] whitespace-nowrap flex items-center gap-1.5 ${
        warn >= 2 ? "bg-brand text-white" : warn === 1 ? "bg-brand/15 text-brand-ink" : "bg-sunken text-ink-dim"}`}>
      <span className="opacity-80">{label}</span>
      <b className="font-semibold">{value ?? "…"}</b>
    </button>
  );
  const flags = risk?.flags ?? 0;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="label mr-0.5">一览</span>
      {chip("🚦 市场风险", risk ? `${flags} 项提示` : null, flags >= 3 ? 2 : flags > 0 ? 1 : 0,
        "today", "今日 → 市场风险：流动性、杠杆、情绪、估值、波动中有几项在提示风险")}
      {chip("💳 杠杆周期", cycle?.stage_label ?? null, cycle && cycle.stage !== "none" && cycle.stage !== "right" ? 1 : 0,
        "money", "资金与信用 → 杠杆周期：A 股两融处在出清、底部、右侧确认的哪一步")}
      {chip("🏦 美国信用", credit ? `${credit.level_label} ${credit.hold}/${credit.of}` : null,
        credit ? { calm: 0, weak: 1, stress: 2 }[credit.level] : 0,
        "money", "资金与信用 → 信用与债务：六条信用紧张规则成立了几条")}
    </div>
  );
}

/** The heatmap needs real pixels; this is the box that measures them. */
function HeatmapSlot({ data }: { data: Parameters<typeof Heatmap>[0]["data"] }) {
  const [ref, size] = useSize<HTMLDivElement>();
  return (
    <div className="flex flex-col gap-1.5">
      <p className="label">
        {data.trade_date} · 全市场市值加权
        <b className={data.pct >= 0 ? "text-up" : "text-down"}> {signed(data.pct, 2)}%</b>
        {!data.has_moves && (
          <b className="text-brand-ink"> · 当日涨跌数据读取失败，方块颜色暂不可用</b>
        )}
      </p>
      <div ref={ref} className="w-full" style={{ height: 560 }}>
        {size.w > 0 && <Heatmap data={data} width={size.w} height={560} />}
      </div>
    </div>
  );
}

function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { id: T; label: string }[];
}) {
  return (
    <div className="flex rounded-lg bg-sunken p-0.5">
      {options.map((o) => (
        <button key={o.id} onClick={() => onChange(o.id)}
          className={`px-2.5 h-7 rounded-md text-[12.5px] font-medium transition-colors ${
            value === o.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * One card, one query, one failure.
 *
 * The retry button matters more here than elsewhere: most of these panels fail
 * for a reason that fixes itself (Tushare rate limit, FINRA timeout), and the
 * alternative is reloading the whole page and refetching the five panels that
 * were fine.
 */
function Panel({ title, subtitle, right, q, children }: {
  title: string; subtitle?: string; right?: ReactNode;
  q: UseQueryResult<unknown, unknown>; children: ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <section className="card p-3 flex flex-col gap-2">
      <div className="flex items-start gap-2">
        <div className="min-w-0">
          <button onClick={() => setOpen(!open)}
            className="text-[14.5px] font-semibold flex items-center gap-1.5">
            <span className="text-ink-mute text-[11px] w-3">{open ? "▾" : "▸"}</span>
            {title}
          </button>
          {subtitle && open && (
            <p className="label leading-snug mt-0.5 pl-[18px]">{subtitle}</p>
          )}
        </div>
        {open && <div className="ml-auto shrink-0">{right}</div>}
      </div>

      {open && (
        <>
          {q.isPending && <div className="py-10 text-center label">加载中…</div>}
          {q.isError && (
            <div className="py-6 text-center flex flex-col gap-1.5">
              <p className="text-[12.5px] text-up">{(q.error as Error).message}</p>
              <button onClick={() => q.refetch()} className="text-cyan text-[13px]">重试</button>
            </div>
          )}
          {!q.isPending && !q.isError && children}
        </>
      )}
    </section>
  );
}
