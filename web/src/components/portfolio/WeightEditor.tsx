/**
 * ⚖️ 调整权重 — argue with the optimiser.
 *
 * The optimiser's answer is the weights that were best on one window of
 * history, which is not the same as the weights you want to hold. This lets
 * you move them and see the cost immediately: the amber dot on the frontier
 * is where your allocation lands, and the distance from the curve is what
 * the opinion costs in risk.
 *
 * The total is reported, never silently normalised. Weights summing to 94%
 * is a thing you need to see, not something to quietly fix — and saving is
 * blocked until it is exactly 100%, the same rule the Streamlit page had.
 */

import { useEffect, useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { PortfolioBuild, WeighResult } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

const TOLERANCE = 0.1;

export function WeightEditor({ d, onResult, onSave }: {
  d: PortfolioBuild;
  onResult: (w: WeighResult | null) => void;
  onSave: (holdings: { t: string; n: string; weight_pct: number }[]) => void;
}) {
  const start = useMemo(() => {
    const out: Record<string, number> = {};
    for (const h of d.holdings) out[h.t] = h.weight_pct;
    return out;
  }, [d.holdings]);

  const [w, setW] = useState<Record<string, number>>(start);
  // A fresh optimisation replaces the draft — the old numbers belonged to
  // a different answer and silently keeping them would be misleading.
  useEffect(() => { setW(start); onResult(null); }, [start]);

  const total = Object.values(w).reduce((a, b) => a + b, 0);
  const balanced = Math.abs(total - 100) <= TOLERANCE;
  const dirty = d.holdings.some((h) => Math.abs((w[h.t] ?? 0) - h.weight_pct) > 0.001);

  const run = useMutation({
    mutationFn: () => api.portfolioWeigh({
      symbols: d.holdings.map((h) => h.t), weights: w,
      max_weight_pct: d.max_weight_pct, lookback: d.lookback,
      duration: d.duration, rf_pct: d.rf_pct,
    }),
    onSuccess: onResult,
  });

  const set = (t: string, v: number) =>
    setW((prev) => ({ ...prev, [t]: Math.max(0, Math.min(100, v)) }));

  const spread = () => {
    const n = d.holdings.length;
    const each = Math.round((100 / n) * 100) / 100;
    const out: Record<string, number> = {};
    d.holdings.forEach((h, i) => {
      out[h.t] = i === n - 1
        ? Math.round((100 - each * (n - 1)) * 100) / 100
        : each;
    });
    setW(out);
  };

  const res = run.data;

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-1.5">
        {d.holdings.map((h) => {
          const v = w[h.t] ?? 0;
          const moved = v - h.weight_pct;
          return (
            <div key={h.t} className="flex items-center gap-2 text-[12.5px]">
              <span className="w-20 sm:w-24 shrink-0 truncate" title={h.n}>
                {h.n}
              </span>
              <span className="hidden md:block w-16 shrink-0 font-mono tnum
                text-[11px] text-ink-mute">
                {h.t}
              </span>
              <input type="range" min={0} max={100} step={0.5} value={v}
                onChange={(e) => set(h.t, Number(e.target.value))}
                aria-label={`${h.n} 权重`}
                className="flex-1 min-w-[48px] accent-[var(--color-cyan)]" />
              <input type="number" min={0} max={100} step={0.5} value={v}
                onChange={(e) => set(h.t, Number(e.target.value))}
                className="w-14 sm:w-16 h-7 px-1.5 rounded-md bg-sunken text-[12px]
                  tnum text-right outline-none shrink-0" />
              <span className={`hidden sm:block w-14 text-right tnum text-[11.5px] ${
                Math.abs(moved) < 0.05 ? "text-ink-mute"
                  : moved > 0 ? "text-up" : "text-down"}`}
                title="相对优化结果的变化">
                {Math.abs(moved) < 0.05 ? "—" : signed(moved, 1, "pp")}
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className={`text-[12.5px] tnum font-medium ${
          balanced ? "text-up" : "text-brand-ink"}`}>
          合计 {fixed(total, 2)}%
          {!balanced && <span className="font-normal"> · 必须正好 100%</span>}
        </span>
        <button onClick={spread}
          className="h-7 px-2 rounded-md bg-sunken text-[12px]">等权</button>
        <button onClick={() => setW(start)} disabled={!dirty}
          className="h-7 px-2 rounded-md bg-sunken text-[12px] disabled:opacity-50">
          还原
        </button>
        <button onClick={() => run.mutate()} disabled={run.isPending || total <= 0}
          className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px]
            font-semibold disabled:opacity-60">
          {run.isPending ? "计算中…" : "看看落在哪"}
        </button>
        <button
          onClick={() => onSave(d.holdings
            .filter((h) => (w[h.t] ?? 0) > 0)
            .map((h) => ({ t: h.t, n: h.n, weight_pct: w[h.t] ?? 0 })))}
          disabled={!balanced}
          title={balanced ? "存成组合" : "权重合计必须正好 100%"}
          className="h-8 px-3 rounded-lg bg-sunken text-[12.5px] font-medium
            disabled:opacity-50">
          💾 存为组合
        </button>
      </div>

      {run.isError && (
        <p className="text-[12.5px] text-up">{(run.error as ApiError).message}</p>
      )}

      {res && <Landed d={d} res={res} />}
    </div>
  );
}

function Landed({ d, res }: { d: PortfolioBuild; res: WeighResult }) {
  const gap = res.ann_vol_pct - d.opt.ann_vol_pct;
  const retGap = res.ann_return_pct - d.opt.ann_return_pct;
  return (
    <div className="rounded-lg bg-sunken p-2.5 flex flex-col gap-1.5">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <Stat label="年化收益" v={signed(res.ann_return_pct, 2, "%")}
          note={signed(retGap, 2, "pp")} />
        <Stat label="年化波动" v={`${fixed(res.ann_vol_pct, 2)}%`}
          note={signed(gap, 2, "pp")} />
        <Stat label="夏普" v={fixed(res.sharpe, 2)}
          note={d.opt.sharpe ? `优化 ${fixed(d.opt.sharpe, 2)}` : undefined} />
        <Stat label="有效仓位数" v={fixed(res.risk.enb, 2)} />
        {!res.balanced && (
          <span className="text-[12px] text-brand-ink">
            合计 {fixed(res.sum_pct, 2)}% —— 下面的数字按这个权重算，没有归一
          </span>
        )}
      </div>
      <p className="label leading-snug">
        橙点就是这组权重在前沿图上的位置。它离曲线越远，说明同样的收益本可以用更小的波动拿到
        —— 这段距离就是你的判断相对历史最优解的代价。
      </p>
    </div>
  );
}

function Stat({ label, v, note }: { label: string; v: string; note?: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="label">{label}</span>
      <span className="text-[13px] tnum font-medium">{v}</span>
      {note && <span className="text-[11px] text-ink-mute tnum">({note})</span>}
    </span>
  );
}
