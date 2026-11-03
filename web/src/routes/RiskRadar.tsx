/**
 * 🧭 风险雷达 — one stock's systematic and idiosyncratic risks, each read from
 * both sides: how much 利空 pressure the evidence carries, how much 利好.
 *
 * The GET is cheap and never calls the AI: it scores whatever evidence is
 * stored. 刷新 is the expensive path — every feed, the tagger, the bull/bear
 * cases — and is a button, so a page view is never silently billed.
 *
 * Colours keep the app's convention everywhere: red is up (利好), green is
 * down (利空), including on US stocks.
 */

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { RiskEvidence, RiskRadar as Board, RiskRow, StockRef } from "../lib/types";
import { NavBar } from "../components/NavBar";
import { StockPicker } from "../components/StockPicker";
import { KIND_LABEL, citations, leanClass, sparkPoints, tierClass } from "../lib/risk";

const TABS = [
  { id: "all", label: "全部" },
  { id: "macro", label: "系统性" },
  { id: "micro", label: "个股" },
] as const;
type Tab = typeof TABS[number]["id"];

const GRID = "grid grid-cols-[minmax(0,2.1fr)_minmax(0,0.6fr)_minmax(0,0.8fr)_minmax(0,2.2fr)_minmax(0,0.7fr)_minmax(0,0.9fr)] gap-3 items-center";

function ago(iso: string | null): string {
  if (!iso) return "从未刷新";
  const d = new Date(iso);
  return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

export function RiskRadar() {
  useEffect(() => { document.title = "ASSRS · 风险雷达"; }, []);
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const ticker = params.get("t");
  const [tab, setTab] = useState<Tab>("all");
  const [sel, setSel] = useState<string | null>(null);

  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const history = useQuery({ queryKey: ["history"], queryFn: api.history, staleTime: 60_000 });
  const board = useQuery({
    queryKey: ["risk", ticker],
    queryFn: () => api.riskRadar(ticker!),
    enabled: Boolean(ticker),
    staleTime: 10 * 60_000,
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
  });
  // Two requests, shown as two stages: collecting and filing, then the cases.
  // The board after the first is drawn at once, so the scores are visible
  // while the AI writes. Without stored evidence (no tables) there is nothing
  // for the second to read, so it is skipped.
  const [stage, setStage] = useState<"collect" | "synth" | null>(null);
  const refresh = useMutation({
    mutationFn: async () => {
      setStage("collect");
      const t = ticker!;
      const first = await api.riskRefresh(t);
      qc.setQueryData(["risk", t], first);
      if (!first.persisted) return first;
      setStage("synth");
      return api.riskSynthesize(t);
    },
    onSuccess: (b) => qc.setQueryData(["risk", b.symbol], b),
    onSettled: () => setStage(null),
  });
  const record = useMutation({
    mutationFn: (t: string) => api.addHistory(t),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["history"] }),
  });

  useEffect(() => {
    if (!ticker && history.data?.[0]) setParams({ t: history.data[0].t }, { replace: true });
  }, [ticker, history.data, setParams]);

  const current: StockRef | null = useMemo(() => {
    if (!ticker) return null;
    const hit = stocks.data?.find((s) => s.t === ticker) ?? history.data?.find((h) => h.t === ticker);
    return hit ? { t: hit.t, n: hit.n } : { t: ticker, n: ticker };
  }, [ticker, stocks.data, history.data]);

  function pick(s: StockRef) {
    setParams({ t: s.t });
    record.mutate(s.t);
    setSel(null);
    refresh.reset();
  }

  // A refresh result is written into the query cache, so it shows even when
  // the GET itself failed.
  const data = board.data ?? (qc.getQueryData(["risk", ticker]) as Board | undefined);
  const selected = data?.risks.find((r) => r.id === sel) ?? (data ? topRisk(data) : undefined);

  return (
    <div className="min-h-screen">
      <NavBar>
        <StockPicker stocks={stocks.data ?? []} history={history.data ?? []}
          current={current} onPick={pick} />
      </NavBar>
      <main className="max-w-[1500px] mx-auto px-3 py-3 flex flex-col gap-3">
        {!ticker && <div className="card p-10 text-center text-ink-mute">在上方选择一只股票</div>}
        {ticker && board.isLoading && <div className="card p-10 text-center label">读取中…</div>}
        {board.error && !data && (
          // Most often the tables do not exist yet. A refresh still works
          // without them — it returns the board, it just cannot keep it.
          <div className="card p-6 flex flex-wrap items-center gap-3">
            <span className="text-[14px] text-up">{(board.error as Error).message}</span>
            <button onClick={() => refresh.mutate()} disabled={stage !== null}
              className="ml-auto h-9 px-4 rounded-lg bg-cyan text-white text-[14px] font-medium disabled:opacity-50">
              {stage ? "生成中…" : "仍然生成（不保存）"}
            </button>
            {refresh.error && <span className="w-full text-[13px] text-up">{(refresh.error as Error).message}</span>}
          </div>
        )}
        {data && (
          <>
            <Header data={data} stage={stage} error={refresh.error as Error | null}
              onRefresh={() => refresh.mutate()} />
            <Summary data={data} />
            <div className="flex flex-wrap gap-3 items-start">
              <section className="card flex-[999_1_640px] min-w-0">
                <div className="flex flex-wrap items-center justify-between gap-2 p-3">
                  <h2 className="text-[16px] font-semibold">风险清单</h2>
                  <div className="flex gap-1">
                    {TABS.map((t) => (
                      <button key={t.id} onClick={() => setTab(t.id)} aria-pressed={tab === t.id}
                        className={`h-8 px-3 rounded-lg text-[13px] font-medium ${
                          tab === t.id ? "bg-ink text-white" : "bg-elevated text-ink-dim hover:text-ink"}`}>
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <div className="min-w-[820px]">
                    <div className={`${GRID} px-3 py-2 bg-panel-alt border-t border-line label`}>
                      <span>风险</span><span>敞口</span><span>等级</span>
                      <span className="flex justify-between">
                        <span className="text-down">← 利空压力</span>
                        <span className="text-up">利好动能 →</span>
                      </span>
                      <span>倾向</span><span>证据 · 7日Δ利空</span>
                    </div>
                    {(["macro", "micro"] as const).filter((k) => tab === "all" || tab === k).map((k) => (
                      <div key={k}>
                        <div className="px-3 py-1.5 bg-panel-alt border-t border-line text-[12px] font-semibold text-ink-dim">
                          {k === "macro" ? "系统性 · 宏观（与大量公司共有）" : "个股 · 微观（只属于这家公司）"}
                        </div>
                        {data.risks.filter((r) => r.kind === k).map((r) => (
                          <Row key={r.id} r={r} on={selected?.id === r.id} onPick={() => setSel(r.id)} />
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
                <p className="px-3 py-2 border-t border-line label">
                  红 = 利好，绿 = 利空。等级只看利空一侧；窗口内少于 {data.method.min_evidence} 条证据显示「证据不足」，
                  不显示「低」。V1 分数为绝对刻度，积累数月证据后改为相对自身历史分位。
                </p>
              </section>
              {selected && <Detail r={selected} />}
            </div>
            <Sources data={data} />
          </>
        )}
      </main>
    </div>
  );
}

function topRisk(b: Board): RiskRow | undefined {
  return b.risks.find((r) => r.id === b.summary.top?.id) ?? b.risks[0];
}

function Header({ data, stage, error, onRefresh }: {
  data: Board; stage: "collect" | "synth" | null; error: Error | null; onRefresh: () => void;
}) {
  const busy = stage !== null;
  return (
    <section className="card p-3 flex flex-wrap items-center gap-3">
      <div className="flex flex-col">
        <span className="label">🧭 风险雷达</span>
        <span className="text-[18px] font-semibold">{data.name} <span className="text-ink-mute text-[14px] font-normal">{data.symbol}</span></span>
      </div>
      <span className="label">上次刷新：{ago(data.refreshed_at)} · 论点生成：{ago(data.synthesized_at)}</span>
      <div className="ml-auto flex items-center gap-3">
        {stage === "collect" && <span className="label">① 抓取各数据源并归类新证据，约 1–2 分钟…</span>}
        {stage === "synth" && <span className="label">② 分数已更新，正在生成利空/利好论点…</span>}
        {error && <span className="text-[13px] text-up">{error.message}</span>}
        <button onClick={onRefresh} disabled={busy}
          className="h-9 px-4 rounded-lg bg-cyan text-white text-[14px] font-medium disabled:opacity-50">
          {busy ? "刷新中…" : data.refreshed_at ? "刷新" : "首次生成"}
        </button>
      </div>
      {!data.persisted && (
        <p className="w-full text-[13px] text-brand-ink">
          本次结果未能写入数据库（请先运行 supabase/migrations/20261103_risk_radar.sql），刷新后的证据不会保留。
        </p>
      )}
    </section>
  );
}

function Tile({ label, children, foot }: { label: string; children: React.ReactNode; foot: string }) {
  return (
    <div className="card p-3 flex flex-col gap-1">
      <span className="label">{label}</span>
      <div className="flex items-baseline gap-2">{children}</div>
      <span className="label">{foot}</span>
    </div>
  );
}

function Summary({ data }: { data: Board }) {
  const s = data.summary;
  return (
    <section className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-3">
      <Tile label="综合利空压力" foot="敞口加权，只计证据充足的风险">
        <span className="text-[26px] font-bold text-down">{s.down_tier}</span>
        {s.down !== null && <span className="tnum text-down">{s.down} / 100</span>}
      </Tile>
      <Tile label="综合利好动能" foot="同一批证据中的正面部分">
        <span className="text-[26px] font-bold text-up">{s.up_tier}</span>
        {s.up !== null && <span className="tnum text-up">{s.up} / 100</span>}
      </Tile>
      <Tile label="当前最大风险"
        foot={s.top ? `利空 ${s.top.down} · 敞口${s.top.exposure} · 7日 ${s.top.d7_down >= 0 ? "+" : ""}${s.top.d7_down}` : "证据不足以判断"}>
        <span className="text-[20px] font-semibold">{s.top?.name ?? "—"}</span>
      </Tile>
      <Tile label="计入的证据" foot={`事实 ${s.facts} · 观点 ${s.opinions}（观点权重 ${data.method.opinion_weight}）· 库存 ${s.stored} 条，无关 ${s.irrelevant}`}>
        <span className="text-[26px] font-bold tnum">{s.n}</span>
        <span className="label">条 · 近 {data.method.window_days} 天</span>
      </Tile>
    </section>
  );
}

function Row({ r, on, onPick }: { r: RiskRow; on: boolean; onPick: () => void }) {
  return (
    <button onClick={onPick} aria-pressed={on}
      className={`${GRID} w-full text-left px-3 py-2.5 border-t border-line min-h-[56px] ${on ? "bg-[#eaf1fb]" : "hover:bg-panel-alt"}`}>
      <span className="flex flex-col min-w-0">
        <span className="text-[14px] font-medium">{r.name}</span>
        <span className="label truncate">{r.en}</span>
      </span>
      <span className="text-[13px]" title={r.exposure_note}>
        {r.exposure}<span className="label">{r.exposure_source === "measured" ? " 实测" : r.exposure_source === "ai" ? " AI" : ""}</span>
      </span>
      <span><span className={`inline-block px-2 py-0.5 rounded-full text-[12px] font-medium ${tierClass(r.tier)}`}>{r.tier}</span></span>
      <span className="flex items-center gap-2">
        <span className="w-6 text-right tnum text-[12px] text-down">{r.down}</span>
        <span className="flex-1 flex h-2.5">
          <span className="flex-1 flex justify-end bg-sunken rounded-l">
            <span className="block bg-down rounded-l" style={{ width: `${r.down}%` }} />
          </span>
          <span className="block w-0.5 bg-ink" />
          <span className="flex-1 flex bg-sunken rounded-r">
            <span className="block bg-up rounded-r" style={{ width: `${r.up}%` }} />
          </span>
        </span>
        <span className="w-6 tnum text-[12px] text-up">{r.up}</span>
      </span>
      <span className={`text-[13px] font-medium ${leanClass(r.lean)}`}>{r.lean}</span>
      <span className="tnum text-[12px] text-ink-dim">
        {r.n} 条 · <span className={r.d7_down > 0 ? "text-down" : r.d7_down < 0 ? "text-up" : ""}>
          {r.d7_down > 0 ? "+" : ""}{r.d7_down}</span>
      </span>
    </button>
  );
}

function Cited({ text, evidence }: { text: string | null; evidence: RiskEvidence[] }) {
  return (
    <>
      {citations(text, evidence).map((s, i) =>
        "cite" in s
          ? <sup key={i} className="text-cyan font-medium">[{s.cite}]</sup>
          : <span key={i}>{s.text}</span>)}
    </>
  );
}

function Detail({ r }: { r: RiskRow }) {
  return (
    <aside className="card flex-[1_1_420px] min-w-0 p-4 flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="label">{r.kind === "macro" ? "系统性 · 宏观" : "个股 · 微观"}</span>
          <span className={`px-2 py-0.5 rounded-full text-[12px] font-medium ${tierClass(r.tier)}`}>风险等级 {r.tier}</span>
          <span className={`text-[12px] font-medium ${leanClass(r.lean)}`}>{r.lean}</span>
        </div>
        <h2 className="text-[20px] font-semibold">{r.name}</h2>
        <span className="label">{r.en}</span>
      </div>

      <div className="rounded-lg bg-panel-alt p-3 flex flex-col gap-1">
        <span className="label">这项风险对本股有多重要（敞口）· {
          r.exposure_source === "measured" ? "价格回归实测" : r.exposure_source === "ai" ? "AI 按业务判断" : "未评估"}</span>
        <span className="text-[14px]"><b>{r.exposure}</b> · {r.exposure_note}</span>
      </div>

      <div className="flex flex-col gap-1">
        <div className="flex justify-between label">
          <span>近 30 日压力</span>
          <span><span className="text-down">— 利空</span> · <span className="text-up">— 利好</span></span>
        </div>
        <svg viewBox="0 0 400 80" className="w-full h-20" role="img" aria-label="近30日利空与利好压力">
          <polyline points={sparkPoints(r.trend.down, 400, 80)} fill="none" stroke="var(--color-down)" strokeWidth="2" />
          <polyline points={sparkPoints(r.trend.up, 400, 80)} fill="none" stroke="var(--color-up)" strokeWidth="2" />
        </svg>
      </div>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
        <div className="rounded-lg border border-[#c9e2d1] p-3 flex flex-col gap-1">
          <span className="text-[12px] font-semibold text-down">利空论点 · {r.down}</span>
          <p className="m-0 text-[14px] leading-relaxed">{r.bear ? <Cited text={r.bear} evidence={r.evidence} /> : <span className="label">刷新后生成</span>}</p>
        </div>
        <div className="rounded-lg border border-[#f2c9c9] p-3 flex flex-col gap-1">
          <span className="text-[12px] font-semibold text-up">利好论点 · {r.up}</span>
          <p className="m-0 text-[14px] leading-relaxed">{r.bull ? <Cited text={r.bull} evidence={r.evidence} /> : <span className="label">刷新后生成</span>}</p>
        </div>
      </div>

      <div className="flex flex-col">
        <h3 className="text-[14px] font-semibold mb-1">证据（按当前权重排序）</h3>
        {r.evidence.length === 0 && <span className="label">窗口内暂无证据</span>}
        {r.evidence.map((e, i) => (
          <div key={e.eid} className="py-2 border-t border-sunken flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
              <span className="text-cyan font-medium">[{i + 1}]</span>
              <span className="tnum text-ink-dim">{e.d}</span>
              <span className="text-ink-dim">{KIND_LABEL[e.kind]} · {e.source}</span>
              <span className={`px-1.5 rounded-full font-medium ${e.dir === "down" ? "bg-[#e3f1e7] text-down" : "bg-[#fbe5e6] text-up"}`}>
                {e.dir === "down" ? "利空" : "利好"}</span>
              <span className={`px-1.5 rounded-full ${e.fact ? "bg-sunken text-ink-dim" : "bg-[#fff1dc] text-brand-ink"}`}>
                {e.fact ? "事实" : "观点"}</span>
              <span className="text-ink-mute">强度 {e.strength} · {e.tagged_by === "rule" ? "规则" : "AI"}归类 · 权重 {e.weight.toFixed(2)}</span>
            </div>
            <span className="text-[14px] leading-snug">
              {e.url ? <a href={e.url} target="_blank" rel="noreferrer" className="hover:underline">{e.title}</a> : e.title}
            </span>
            {e.note && <span className="label">{e.note}</span>}
          </div>
        ))}
      </div>

      {r.watch.length > 0 && (
        <div>
          <h3 className="text-[14px] font-semibold mb-1">什么会改变判断</h3>
          <ul className="m-0 pl-5 text-[14px] leading-relaxed">
            {r.watch.map((w) => <li key={w}>{w}</li>)}
          </ul>
        </div>
      )}
      <span className="label border-t border-sunken pt-2">
        论点由 DeepSeek 仅依据上列证据生成，每句须引用证据编号。
      </span>
    </aside>
  );
}

function Sources({ data }: { data: Board }) {
  if (!data.sources.length) return null;
  return (
    <section className="card p-3 flex flex-col gap-2">
      <h2 className="text-[14px] font-semibold">数据源状态（上次刷新）</h2>
      <div className="flex flex-wrap gap-2">
        {data.sources.map((s) => (
          <span key={s.source} title={s.reason ?? ""}
            className={`px-2 py-1 rounded-lg text-[12px] ${s.ok ? "bg-sunken text-ink-dim" : "bg-[#fff1dc] text-brand-ink"}`}>
            {s.ok ? "✓" : "✕"} {s.label} {s.ok ? `· ${s.n}` : "· 不可用"}
          </span>
        ))}
      </div>
      <p className="m-0 label">「不可用」表示该源本次抓取失败，不代表没有新闻。多数快讯源只提供最新条目，证据随每次刷新累积。</p>
    </section>
  );
}
