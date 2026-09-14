import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import { supabase } from "../lib/supabase";
import { useAuth } from "../auth/AuthProvider";
import type { StockRef } from "../lib/types";
import { StockPicker } from "../components/StockPicker";
import { InfoHeader } from "../components/InfoHeader";
import { ChartStack } from "../components/chart/ChartStack";

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
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 bg-canvas/90 backdrop-blur border-b border-line">
        <div className="max-w-[1500px] mx-auto px-4 h-14 flex items-center gap-4">
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

      <main className="max-w-[1500px] mx-auto px-4 py-3 flex flex-col gap-3">
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
          <>
            <InfoHeader data={analysis.data} />
            <ChartStack data={analysis.data} />
          </>
        )}
      </main>
    </div>
  );
}
