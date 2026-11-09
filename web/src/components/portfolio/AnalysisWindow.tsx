/**
 * 🗓️ 时间窗口 — pick the stretch of history, then read the stocks inside it.
 *
 * The window is what the optimiser fits to, so it lives above the panel: the
 * page needs the two dates whether or not this card has finished loading.
 * Each market keeps its own index and window: their calendars differ, and an
 * A-share regime means nothing to a US book.
 */

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { StockRef } from "../../lib/types";
import { usePersistentState } from "../../lib/usePersistentState";
import { resolveRange, toSaved, type Range, type SavedWindow } from "../../lib/windowRange";
import { WindowBrush } from "./WindowBrush";
import { WindowStats } from "./WindowStats";
import type { Market } from "./Optimiser";

const INDICES: Record<Market, { code: string; label: string }[]> = {
  CN: [
    { code: "000300.SH", label: "沪深300" },
    { code: "000001.SH", label: "上证指数" },
    { code: "399001.SZ", label: "深证成指" },
    { code: "399006.SZ", label: "创业板指" },
    { code: "000688.SH", label: "科创50" },
  ],
  US: [
    { code: "^GSPC", label: "标普500" },
    { code: "^IXIC", label: "纳斯达克综合" },
    { code: "^NDX", label: "纳斯达克100" },
    { code: "^DJI", label: "道琼斯" },
    { code: "^RUT", label: "罗素2000" },
  ],
};

/** A-shares keep the keys they were first saved under. */
const KEY: Record<Market, string> = { CN: "assrs.pf.win", US: "assrs.pf.us.win" };

const THRESHOLDS = [15, 20, 25, 30];

export function useAnalysisWindow(market: Market, enabled: boolean) {
  const [code, setCode] = usePersistentState<string>(
    `${KEY[market]}.index`, INDICES[market][0]!.code);
  const [pct, setPct] = usePersistentState<number>(`${KEY[market]}.pct`, 20);
  const [saved, setSaved] = usePersistentState<SavedWindow | null>(KEY[market], null);

  const index = useQuery({
    queryKey: ["portfolio", "window", code, pct],
    queryFn: () => api.portfolioWindowIndex(code, pct),
    enabled,
    staleTime: 30 * 60_000,
    retry: 1,
    placeholderData: keepPreviousData,
  });

  const dates = index.data?.dates ?? null;
  const range: Range | null = dates ? resolveRange(dates, saved) : null;
  return {
    market, code, setCode, pct, setPct, index, range,
    from: dates && range ? dates[range[0]] : null,
    to: dates && range ? dates[range[1]] : null,
    commit: (r: Range) => { if (dates) setSaved(toSaved(dates, r)); },
  };
}

export type AnalysisWindowState = ReturnType<typeof useAnalysisWindow>;

export function AnalysisWindow({ w, picks, rf }: {
  w: AnalysisWindowState;
  picks: StockRef[];
  rf: number;
}) {
  const { index, range, from, to } = w;
  const symbols = picks.map((p) => p.t);

  const stats = useQuery({
    queryKey: ["portfolio", "windowStats", symbols, from, to, w.code, w.pct, rf],
    queryFn: () => api.portfolioWindowStats({
      symbols, start: from!, end: to!, index: w.code, regime_pct: w.pct, rf_pct: rf,
    }),
    enabled: symbols.length > 0 && !!from && !!to,
    staleTime: 10 * 60_000,
    retry: false,
    placeholderData: keepPreviousData,
  });

  const snap = (leg: { from: string; to: string }) => {
    const dates = index.data?.dates;
    if (!dates) return;
    const i = dates.indexOf(leg.from);
    const j = dates.indexOf(leg.to);
    if (i >= 0 && j >= 0) w.commit([i, j]);
  };

  return (
    <section className="card p-3 flex flex-col gap-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <h3 className="text-[13.5px] font-semibold">🗓️ 时间窗口</h3>
        <div className="flex items-center gap-0.5 flex-wrap">
          {INDICES[w.market].map((i) => (
            <button key={i.code} onClick={() => w.setCode(i.code)}
              className={`h-7 px-2.5 rounded-md text-[12.5px] font-medium transition-colors ${
                w.code === i.code ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"}`}>
              {i.label}
            </button>
          ))}
        </div>
        <label className="label flex items-center gap-1.5 ml-auto"
          title="价格从高点或低点反转超过这个幅度，才算一次行情转换">
          拐点幅度
          <select value={w.pct} onChange={(e) => w.setPct(Number(e.target.value))}
            className="h-7 px-1.5 rounded-md bg-sunken text-[12.5px] tnum outline-none">
            {THRESHOLDS.map((t) => <option key={t} value={t}>±{t}%</option>)}
          </select>
        </label>
      </div>

      {index.isError && (
        <p className="text-[12.5px] text-up">
          指数行情读取失败：{(index.error as ApiError).message}。下面的优化先按「回看」天数计算。
        </p>
      )}
      {index.isPending && <p className="label py-10 text-center">读取十年指数行情…</p>}

      {index.data && range && (
        <>
          <WindowBrush data={index.data} range={range} onCommit={w.commit} />
          <p className="label leading-snug">
            拖动窗口中间平移，拖两边的把手放大缩小，点一段阴影直接选中它（最短 60 个交易日）。
            <span className="text-up">红色</span>是上行段，<span className="text-down">绿色</span>是下行段，
            都是事后划分的：拐点要等价格反转够幅度才知道，不能当实时信号，最后一段也还没走完。
          </p>
        </>
      )}

      {symbols.length === 0 ? (
        <p className="label">选了股票之后，这里会列出每只股票在这个窗口和各段行情里的表现。</p>
      ) : (
        <div className={`transition-opacity ${stats.isFetching ? "opacity-50" : ""}`}>
          {stats.isError && (
            <p className="text-[12.5px] text-up">{(stats.error as ApiError).message}</p>
          )}
          {stats.isPending && !!from && !stats.isError && (
            <p className="label py-4 text-center">读取个股行情…</p>
          )}
          {stats.data && from && to && (
            <WindowStats s={stats.data} picks={picks} selected={{ from, to }} onSnap={snap} />
          )}
        </div>
      )}
    </section>
  );
}
