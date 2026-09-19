/**
 * 策略 — the watchlist screens, ported from the Streamlit app.
 *
 * Both screens answer "which of the stocks I already hold suit this trade
 * today", which is a different question from 今日提醒's "what fired last
 * night". They are also expensive: two or three Tushare calls per holding,
 * walked serially, so scanning is a button rather than something opening the
 * page triggers. The last result is shown with its age until you rescan.
 *
 * 做T shares its stored rows with the Streamlit page — scan in either and both
 * see it — which is why that tab can show a result before you ever press scan.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { Num, StrategyResult, StrategyRow } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { PairTrade } from "../components/PairTrade";
import { LeadLag } from "../components/LeadLag";
import { fixed, signed } from "../lib/format";

type TabId = "t-trading" | "mean-reversion" | "pair-trade" | "lead-lag";

const TABS: { id: TabId; label: string; blurb: string }[] = [
  {
    id: "t-trading",
    label: "⚡ 做T候选",
    blurb: "T+1 下做日内回转：卖掉手里的票再低位买回，仓位不变、差价落袋。"
      + "需要日内有波动、能进出、且收盘回到区间中部——单边趋势和封板都做不了。",
  },
  {
    id: "pair-trade",
    label: "🔗 配对交易",
    blurb: "两只通常同涨同跌的票，在价差被拉开时买便宜的那条腿。"
      + "对冲比率逐日滚动估计并前推一天，所以价差与历史交易都是样本外的。",
  },
  {
    id: "lead-lag",
    label: "🕰️ 领先滞后",
    blurb: "哪只票先动、哪只跟着动，以及这个时间差能不能拿来交易。"
      + "格兰杰检验给方向，滞后相关给形状，协整与半衰期决定价差会不会收敛——"
      + "三个问题分开看，因为一对股票常常过得了第一关、过不了第三关。",
  },
  {
    id: "mean-reversion",
    label: "🔄 反转候选",
    blurb: "被情绪砸下去、而不是被消息砸下去的票。价格异常下挫、连跌、缩量、"
      + "RSI 极低，而板块并没有跟着跌——孤立的恐慌才有反弹的统计基础。",
  },
];

//: Nothing left to port. Kept as the place to name anything that is not yet
//: here, so the tab bar goes on telling the truth about what exists.
const COMING: { label: string; why: string }[] = [];

export function Strategies() {
  useEffect(() => { document.title = "ASSRS · 策略"; }, []);
  const [tab, setTab] = useState<TabId>("t-trading");

  return (
    <div className="min-h-screen">
      <NavBar />
      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        <nav className="card p-2 flex flex-wrap items-center gap-1.5">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`h-8 px-3 rounded-lg text-[13px] font-medium transition-colors ${
                tab === t.id ? "bg-cyan text-white" : "hover:bg-elevated"
              }`}>
              {t.label}
            </button>
          ))}
          <span className="ml-auto flex items-center gap-1.5">
            {COMING.map((c) => (
              <span key={c.label} title={`${c.why} — 尚未迁移`}
                className="h-8 px-2.5 rounded-lg text-[12.5px] text-ink-mute border border-dashed border-line flex items-center">
                {c.label} · 待迁移
              </span>
            ))}
          </span>
        </nav>

        {tab === "pair-trade" ? <PairTrade />
          : tab === "lead-lag" ? <LeadLag />
            : <Screen key={tab} id={tab} />}
      </main>
    </div>
  );
}

function Screen({ id }: { id: "t-trading" | "mean-reversion" }) {
  const meta = TABS.find((t) => t.id === id)!;
  const qc = useQueryClient();

  const q = useQuery({
    queryKey: ["strategy", id],
    queryFn: () => api.strategy(id),
    // 404 means "never scanned", which is a normal state, not a failure to retry.
    retry: (n, err) => !(err instanceof ApiError && err.status === 404) && n < 1,
    staleTime: 10 * 60_000,
  });

  const scan = useMutation({
    mutationFn: () => api.strategyScan(id),
    onSuccess: (data) => qc.setQueryData(["strategy", id], data),
  });

  const never = q.isError && (q.error as ApiError)?.status === 404;
  const data = scan.data ?? q.data;

  return (
    <>
      <section className="card p-3 flex flex-col gap-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="text-[14px] font-semibold">{meta.label}</h2>
          {data?.scanned_at && (
            <span className="label">
              上次扫描 {new Date(data.scanned_at).toLocaleString("zh-CN")}
              {data.age_hours != null && ` · ${fixed(data.age_hours, 0)} 小时前`}
              {data.stale && <b className="text-up"> · 已过期，建议重新扫描</b>}
            </span>
          )}
          <button onClick={() => scan.mutate()} disabled={scan.isPending}
            className="ml-auto h-8 px-3 rounded-lg bg-cyan text-white text-[13px] font-semibold disabled:opacity-60">
            {scan.isPending ? "扫描中…（每只股票需查询行情，请稍候）" : "🔍 扫描自选股"}
          </button>
        </div>
        <p className="label leading-snug max-w-[70ch]">{meta.blurb}</p>
        {data && <Counts d={data} id={id} />}
      </section>

      {never && !data && (
        <div className="card p-10 text-center label">
          还没有扫描过 — 点击右上角「扫描自选股」
        </div>
      )}
      {q.isError && !never && !data && (
        <div className="card p-6 text-center text-up text-[13px]">
          {(q.error as Error).message}
        </div>
      )}
      {scan.isError && (
        <div className="card p-4 text-center text-up text-[13px]">
          扫描失败：{(scan.error as Error).message}
        </div>
      )}
      {scan.isPending && (
        <div className="card p-8 flex flex-col items-center gap-2 text-ink-mute">
          <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
          <div className="label">逐只查询中，自选股越多越慢</div>
        </div>
      )}

      {data && (id === "t-trading" ? <TTable rows={data.rows} /> : <MRTable rows={data.rows} />)}
    </>
  );
}

const VERDICT_ORDER: Record<string, string[]> = {
  "t-trading": ["strong", "ok", "not_now", "skip", "no_data"],
  "mean-reversion": ["strong", "watch", "not_now", "skip", "no_data"],
};
const VERDICT_LABEL: Record<string, string> = {
  strong: "🟢 强", ok: "🟡 可以", watch: "🟡 观察",
  not_now: "⚪ 暂不", skip: "⛔ 排除", no_data: "⚠️ 无数据",
};

function Counts({ d, id }: { d: StrategyResult; id: "t-trading" | "mean-reversion" }) {
  const order = VERDICT_ORDER[id]!;
  const keys = Object.keys(d.counts).sort(
    (a, b) => (order.indexOf(a) + 99) % 99 - (order.indexOf(b) + 99) % 99);
  return (
    <div className="flex flex-wrap items-center gap-2 text-[12.5px]">
      <span className="label">共 {d.count} 只</span>
      {keys.map((k) => (
        <span key={k} className="rounded-md bg-sunken px-2 py-0.5">
          {VERDICT_LABEL[k] ?? k} {d.counts[k]}
        </span>
      ))}
    </div>
  );
}

function Shell({ head, children }: { head: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card p-3 overflow-x-auto">
      <table className="w-full text-[12.5px] border-collapse">
        <thead><tr className="text-ink-mute">{head}</tr></thead>
        <tbody>{children}</tbody>
      </table>
    </section>
  );
}

const TH = "text-right font-normal pb-1 px-2 whitespace-nowrap";

function Name({ r }: { r: StrategyRow }) {
  return (
    <td className="py-1 pr-3">
      <Link to={`/?t=${encodeURIComponent(r.ticker)}`} className="hover:text-cyan">
        <span className="truncate">{r.name}</span>{" "}
        <span className="font-mono tnum text-[11px] text-ink-mute">{r.ticker}</span>
      </Link>
    </td>
  );
}

function Cell({ v, nd = 2, suffix = "" }: { v: Num | undefined; nd?: number; suffix?: string }) {
  return (
    <td className="py-1 px-2 text-right font-mono tnum text-ink-dim whitespace-nowrap">
      {v == null ? "—" : `${fixed(v, nd)}${suffix}`}
    </td>
  );
}

function TTable({ rows }: { rows: StrategyRow[] }) {
  return (
    <Shell head={<>
      <th className="text-left font-normal pb-1 pr-3">股票</th>
      <th className={TH}>T评分</th>
      <th className="text-left font-normal pb-1 px-2">判定</th>
      <th className={TH} title="20日平均 (高−低)/开盘">日内波幅</th>
      <th className={TH} title="20日平均换手率">换手</th>
      <th className={TH} title="|收−开|/(高−低)，越低越像震荡">回归倾向</th>
      <th className={TH}>ADX</th>
      <th className={TH} title="离60日高低点的距离，1=区间中部">区间位置</th>
      <th className="text-left font-normal pb-1 pl-2">说明</th>
    </>}>
      {rows.map((r) => (
        <tr key={r.ticker} className="border-t border-line">
          <Name r={r} />
          <td className="py-1 px-2 text-right font-mono tnum font-semibold">
            {r.score == null ? "—" : fixed(r.score, 1)}
          </td>
          <td className="py-1 px-2 whitespace-nowrap">{r.verdict_cn}</td>
          <Cell v={r.range_pct} suffix="%" />
          <Cell v={r.turnover_pct} suffix="%" />
          <Cell v={r.meanrev_bias} nd={3} />
          <Cell v={r.adx} nd={1} />
          <Cell v={r.range_pos} />
          <td className="py-1 pl-2 text-ink-mute">{r.why || ""}</td>
        </tr>
      ))}
    </Shell>
  );
}

const RULES: { k: string; label: string; hint: string }[] = [
  { k: "z", label: "Z", hint: "今日跌幅相对自身20日的异常程度 ≤ −2.5" },
  { k: "down", label: "连跌", hint: "连续下跌 ≥ 4 天" },
  { k: "vol", label: "缩量", hint: "近期下跌日的成交量低于之前的下跌日——卖盘在枯竭" },
  { k: "rsi", label: "RSI", hint: "RSI(14) < 25" },
  { k: "sector", label: "弱于板块", hint: "比所属板块弱 5 个百分点以上；无板块数据记为半分" },
];

function MRTable({ rows }: { rows: StrategyRow[] }) {
  const mark = (v: boolean | null | undefined) =>
    v === true ? <span className="text-up">✓</span>
      : v === null || v === undefined ? <span className="text-ink-mute">—</span>
        : <span className="text-ink-mute">✗</span>;

  return (
    <Shell head={<>
      <th className="text-left font-normal pb-1 pr-3">股票</th>
      <th className="text-left font-normal pb-1 px-2">判定</th>
      {RULES.map((r) => (
        <th key={r.k} className="font-normal pb-1 px-2 whitespace-nowrap" title={r.hint}>
          {r.label}
        </th>
      ))}
      <th className={TH}>Z</th>
      <th className={TH}>RSI</th>
      <th className={TH} title="连续下跌天数">连跌</th>
      <th className={TH} title="近5日跌停次数">跌停</th>
      <th className={TH} title="5日涨跌相对所属板块">弱于板块</th>
      <th className="text-left font-normal pb-1 pl-2">说明</th>
    </>}>
      {rows.map((r) => (
        <tr key={r.ticker} className="border-t border-line">
          <Name r={r} />
          <td className="py-1 px-2 whitespace-nowrap">{r.verdict_cn}</td>
          {RULES.map((rule) => (
            <td key={rule.k} className="py-1 px-2 text-center">{mark(r.rules?.[rule.k])}</td>
          ))}
          <Cell v={r.z} />
          <Cell v={r.rsi} nd={1} />
          <Cell v={r.down_days} nd={0} />
          <Cell v={r.limit_down_streak} nd={0} />
          <td className="py-1 px-2 text-right font-mono tnum text-ink-dim whitespace-nowrap">
            {r.vs_sector_pp == null ? "—" : signed(r.vs_sector_pp, 1, "pp")}
          </td>
          <td className="py-1 pl-2 text-ink-mute">{r.why || ""}</td>
        </tr>
      ))}
    </Shell>
  );
}
