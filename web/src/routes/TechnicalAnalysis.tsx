import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { supabase } from "../lib/supabase";
import { useAuth } from "../auth/AuthProvider";
import type { CompareResult, SimResult, StockRef } from "../lib/types";
import { StockPicker } from "../components/StockPicker";
import { InfoHeader } from "../components/InfoHeader";
import { ChipPanel } from "../components/ChipPanel";
import { ChartStack, type Tool } from "../components/chart/ChartStack";
import { usePersistentState } from "../lib/usePersistentState";
import type { Drawing } from "../components/chart/drawings";
import { ChartTools } from "../components/ChartTools";
import { WhatIfPanel } from "../components/WhatIfPanel";
import { SectorPanel } from "../components/SectorPanel";
import { CompareStats } from "../components/CompareStats";

export function TechnicalAnalysis() {
  const { dev } = useAuth();
  const qc = useQueryClient();
  const [params, setParams] = useSearchParams();
  const ticker = params.get("t");

  const stocks = useQuery({ queryKey: ["stocks"], queryFn: api.stocks, staleTime: 6 * 3600_000 });
  const history = useQuery({ queryKey: ["history"], queryFn: api.history, staleTime: 60_000 });
  const analysis = useQuery({
    queryKey: ["analysis", ticker],
    queryFn: () => api.analysis(ticker!),
    enabled: Boolean(ticker),
    staleTime: 10 * 60_000,
    retry: (n, err) => !(err instanceof ApiError && err.status < 500) && n < 1,
  });

  const [ghost, setGhost] = useState<SimResult | null>(null);
  const [compare, setCompare] = useState<CompareResult | null>(null);
  const [compareMode, setCompareMode] = useState<"pct" | "price">("pct");
  const [tool, setTool] = useState<Tool>("none");
  const [resetSignal, setResetSignal] = useState(0);
  const [drawings, setDrawings] = usePersistentState<Drawing[]>(
    `assrs.draw.${ticker ?? "none"}`, []);

  const compareM = useMutation({
    mutationFn: (other: string) => api.compare(ticker!, other),
    onSuccess: setCompare,
  });

  const record = useMutation({
    mutationFn: (t: string) => api.addHistory(t),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["history"] }),
  });

  // Opening the page with no ?t= picks up where you left off.
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
    // A ghost and a comparison belong to the stock they were made against.
    setGhost(null);
    setCompare(null);
    setTool("none");
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 bg-canvas/90 backdrop-blur border-b border-line">
        <div className="max-w-[1800px] mx-auto px-3 h-14 flex items-center gap-4">
          <span className="font-semibold text-[15px] shrink-0">📈 个股分析</span>
          <StockPicker
            stocks={stocks.data ?? []}
            history={history.data ?? []}
            current={current}
            onPick={pick}
          />
          <div className="ml-auto flex items-center gap-3 shrink-0">
            {dev && <span className="text-[12px] text-brand-ink">本地开发模式</span>}
            {!dev && (
              <button onClick={() => supabase.auth.signOut()} className="text-[13px] text-ink-mute hover:text-ink">
                退出
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-3 py-3 flex flex-col gap-3">
        {!ticker && (
          <div className="card p-10 text-center text-ink-mute">在上方输入股票代码或名称开始分析</div>
        )}

        {ticker && analysis.isPending && (
          <div className="card p-10 flex flex-col items-center gap-2 text-ink-mute">
            <div className="h-6 w-6 rounded-full border-2 border-line border-t-cyan animate-spin" />
            <div>正在分析 {current?.n ?? ticker}…</div>
            <div className="label">首次分析约需 10–25 秒（计算隐马尔可夫波动状态）；之后 20 分钟内秒开</div>
          </div>
        )}

        {ticker && analysis.isError && (
          <div className="card p-6 text-up">
            分析失败：{(analysis.error as Error).message}
            <button onClick={() => analysis.refetch()} className="ml-3 text-cyan">重试</button>
          </div>
        )}

        {analysis.data && (
          // Chart left, everything you read ALONGSIDE it right: identity and
          // daily_basic, the signal summary, then 筹码分布 under them. The
          // sidebar sticks while the chart stack scrolls, so the price and
          // the cost distribution stay on screen next to whichever pane you
          // are looking at. Below `lg` it stacks, chart first.
          <div className="grid gap-3 items-start lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="min-w-0 flex flex-col gap-2">
              <ChartTools
                tool={tool}
                setTool={setTool}
                onReset={() => setResetSignal((n) => n + 1)}
                drawingCount={drawings.length}
                onClearDrawings={() => setDrawings([])}
                stocks={stocks.data ?? []}
                compare={compare}
                compareMode={compareMode}
                onCompare={(t) => (t ? compareM.mutate(t) : setCompare(null))}
                onCompareMode={setCompareMode}
              />
              <ChartStack
                data={analysis.data}
                ghost={ghost}
                compare={compare}
                compareMode={compareMode}
                tool={tool}
                onToolDone={() => setTool("none")}
                resetSignal={resetSignal}
                drawings={drawings}
                setDrawings={setDrawings}
              />
            </div>
            <aside className="flex flex-col gap-3 lg:sticky lg:top-[3.75rem] lg:max-h-[calc(100vh-4.5rem)] lg:overflow-y-auto">
              <InfoHeader data={analysis.data} />
              <WhatIfPanel data={analysis.data} ghost={ghost} onGhost={setGhost} />
              <ChipPanel chips={analysis.data.chips} price={analysis.data.header.close ?? 0} />
            </aside>
          </div>
        )}

        {/* Only while a comparison is on screen: it is the answer to a
            question you asked by picking a second stock, not a permanent
            fixture of the page. */}
        {analysis.data && compare && (
          <CompareStats ticker={analysis.data.ticker} other={compare.ticker} />
        )}

        {/* Full width, below the chart: the correlation bars and the rotation
            strip both read across a year, and squeezing them into the 340px
            sidebar would make the strip unreadable. */}
        {analysis.data && <SectorPanel ticker={analysis.data.ticker} />}
      </main>
    </div>
  );
}
