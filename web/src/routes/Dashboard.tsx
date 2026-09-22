/**
 * 市场看板 — the whole market before any single stock.
 *
 * Six questions, in the order you would actually ask them on opening the app:
 *
 *   how did the world close?         全球指数
 *   where did money go today?        heatmap  ⇄  sector trend history
 *   where is it moving next?         相对轮动图
 *   what regime is the index in?     统计威科夫
 *   how much of it is borrowed?      两融 / margin debt
 *   where was the abnormal flow?     龙虎榜
 *
 * The heatmap and the trend grid share one slot because they answer the same
 * question at two horizons — today's move and the last three months of it —
 * and showing both at once makes the page scroll past the point of being a
 * dashboard.
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
import { TopList } from "../components/market/TopList";
import { IndexStrip } from "../components/market/IndexStrip";
import { usePersistentState } from "../lib/usePersistentState";
import { useSize } from "../lib/useSize";
import { signed } from "../lib/format";

const INDICES: Record<string, string> = {
  "000300.SH": "沪深300",
  "000905.SH": "中证500",
  "399006.SZ": "创业板指",
  "000001.SH": "上证",
};

export function Dashboard() {
  useEffect(() => { document.title = "ASSRS · 市场看板"; }, []);

  const [view, setView] = usePersistentState<"heatmap" | "breadth">("assrs.mkt.view", "heatmap");
  const [freq, setFreq] = usePersistentState<"w" | "d">("assrs.mkt.rotfreq", "w");
  const [index, setIndex] = usePersistentState<string>("assrs.mkt.index", "000300.SH");

  // Above everything else and on its own short clock: half these markets
  // are open while someone is looking at the page.
  const indices = useQuery({ queryKey: ["mkt", "indices"], queryFn: api.indices,
    staleTime: 5 * 60_000, refetchInterval: 5 * 60_000 });
  const heatmap = useQuery({ queryKey: ["mkt", "heatmap"], queryFn: api.heatmap,
    staleTime: 20 * 60_000, enabled: view === "heatmap" });
  const breadth = useQuery({ queryKey: ["mkt", "breadth"], queryFn: () => api.breadth(60),
    staleTime: 60 * 60_000, enabled: view === "breadth" });
  const rotation = useQuery({ queryKey: ["mkt", "rotation", freq],
    queryFn: () => api.rotation(freq), staleTime: 60 * 60_000 });
  const wyckoff = useQuery({ queryKey: ["mkt", "wyckoff", index],
    queryFn: () => api.wyckoff(index), staleTime: 30 * 60_000 });
  const leverage = useQuery({ queryKey: ["mkt", "leverage"], queryFn: api.leverage,
    staleTime: 20 * 60_000 });
  const toplist = useQuery({ queryKey: ["mkt", "toplist"], queryFn: api.topList,
    staleTime: 2 * 3600_000 });

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        <Panel title="🌏 全球指数"
          subtitle="A 股、亚太、欧美主要指数的最新涨跌。各市场收盘时间不同，卡片标的是它自己那根的日期。"
          q={indices}>
          {indices.data && <IndexStrip data={indices.data} />}
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

        <Panel title="💳 市场杠杆"
          subtitle="客户保证金借款余额 —— 最直接的风险偏好指标。红 = 加杠杆，绿 = 去杠杆。"
          q={leverage}>
          {leverage.data && <LeveragePanel data={leverage.data} />}
        </Panel>

        <Panel title="🐉 龙虎榜"
          subtitle="触发交易所异动披露的个股，以及当日席位净买卖金额。"
          q={toplist}>
          {toplist.data && <TopList data={toplist.data} />}
        </Panel>
      </main>
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
