/**
 * 配置优化 — what the weights could be, and the evidence that it matters.
 *
 * The table nobody usually shows is the one at the bottom. Fed historical
 * average returns, mean-variance optimisation is an estimation-error
 * maximiser: it piles into whatever got luckiest in the sample and then
 * underperforms equal weighting out of sample. So the panel is arranged around
 * the methods that need only a covariance matrix (minimum variance, risk
 * parity), flags the one that needs return forecasts, and ends with a
 * walk-forward: weights fitted on the first half of the history, scored on the
 * second, against equal weighting and against the book you already hold.
 *
 * If the "optimal" allocation loses to what you own on data it never saw, this
 * page says so rather than quietly showing you a prettier in-sample number.
 *
 * The per-holding rows lead with the DELTA, because "sell 6% of this, buy 4%
 * of that" is the actionable form and a column of target percentages is not.
 */

import type { QtAllocStats, QtOptimise } from "../../lib/types";
import { fixed, signed } from "../../lib/format";

export function OptimisePanel({ data, method, onMethod, cap, onCap }: {
  data: QtOptimise; method: string; onMethod: (m: string) => void;
  cap: number; onCap: (c: number) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg bg-sunken p-0.5">
          {data.methods.map((m) => (
            <button key={m.id} onClick={() => onMethod(m.id)} title={m.means}
              className={`px-2.5 h-7 rounded-md text-[12.5px] font-medium ${
                method === m.id ? "bg-panel text-ink shadow-sm" : "text-ink-mute"}`}>
              {m.label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-[12px] text-ink-mute"
          title="任一标的的权重上限。没有上限时，优化器经常给出两只股票的组合 —— 那是对样本的描述，不是建议。">
          单只上限
          <input type="range" min={5} max={100} step={5} value={Math.round(cap * 100)}
            onChange={(e) => onCap(Number(e.target.value) / 100)}
            className="w-24 accent-[var(--color-cyan)]" />
          <span className="tnum w-9">{fixed(data.cap_pct, 0)}%</span>
        </label>
        <span className="label ml-auto">
          {data.scope_label} · {data.from} → {data.to} · {data.sessions} 个交易日
        </span>
      </div>

      <p className="text-[12px] leading-snug rounded-lg bg-sunken px-2.5 py-2">
        <b>{data.label}</b>：{data.means}
        {data.overfit_risk && (
          <b className="text-up"> 这一项依赖历史收益的外推，最容易过拟合 —— 请以下方样本外检验为准。</b>
        )}
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-3">
        <div className="flex flex-col gap-2">
          <Row label="年化波动" now={data.current.ann_vol_pct}
            next={data.target.ann_vol_pct} lowerIsBetter suffix="%" />
          <Row label="年化收益" now={data.current.ann_return_pct}
            next={data.target.ann_return_pct} suffix="%" />
          <Row label="夏普" now={data.current.sharpe} next={data.target.sharpe} />
          <Row label="最大回撤" now={data.current.max_drawdown_pct}
            next={data.target.max_drawdown_pct} suffix="%" />
          <p className="text-[11.5px] text-ink-mute leading-snug">
            ⚠ {data.basis} 换手 <b>{fixed(data.turnover_pct, 1)}%</b>
            （需要买卖的市值占组合的比例，未计佣金与税）。
          </p>
        </div>

        <div className="overflow-auto rounded-lg border border-line max-h-[340px]">
          <table className="w-full border-collapse text-[12px]">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-line text-ink-mute">
                <th className="text-left font-medium px-2 py-1.5">代码</th>
                <th className="text-right font-medium px-2 py-1.5">当前</th>
                <th className="text-right font-medium px-2 py-1.5">建议</th>
                <th className="text-right font-medium px-2 py-1.5">调整</th>
                <th className="text-right font-medium px-2 py-1.5"
                  title="调整后该标的对组合波动的贡献">风险占比</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={r.symbol} className="border-b border-line/60 hover:bg-sunken">
                  <td className="px-2 py-1 font-mono" title={r.name}>{r.symbol}</td>
                  <td className="px-2 py-1 text-right tnum text-ink-mute">
                    {fixed(r.current_pct, 1)}%
                  </td>
                  <td className="px-2 py-1 text-right tnum font-medium">
                    {r.target_pct === 0 ? "清仓" : `${fixed(r.target_pct, 1)}%`}
                  </td>
                  <td className={`px-2 py-1 text-right tnum ${
                    r.delta_pct > 0.05 ? "text-down"
                      : r.delta_pct < -0.05 ? "text-up" : "text-flat"}`}>
                    {signed(r.delta_pct, 1)}%
                  </td>
                  <td className="px-2 py-1 text-right tnum text-ink-mute">
                    {fixed(r.risk_pct, 1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <WalkForward data={data} />

      {data.excluded.length > 0 && (
        <p className="text-[11.5px] text-brand-ink leading-snug">
          ⚠ 未纳入优化：{data.excluded.map((e) => `${e.symbol}（${e.reason}）`).join("、")}
        </p>
      )}
    </div>
  );
}

function WalkForward({ data }: { data: QtOptimise }) {
  const wf = data.walk_forward;
  if (!wf) {
    return (
      <p className="text-[12px] text-brand-ink">
        ⚠ 无法做样本外检验：{data.walk_forward_error || "历史长度不足"}。
        没有它，上面的配置只是对历史的拟合。
      </p>
    );
  }

  const best = wf.rows.reduce((a, b) =>
    a.out_of_sample.ann_vol_pct <= b.out_of_sample.ann_vol_pct ? a : b);
  const current = wf.rows.find((r) => r.method === "current");
  const beatsCurrent = current
    ? wf.rows.filter((r) => r.method !== "current"
        && r.out_of_sample.ann_vol_pct < current.out_of_sample.ann_vol_pct)
    : [];

  return (
    <div className="rounded-lg border border-line p-2.5 flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-[13px] font-medium">样本外检验</span>
        <span className="label">
          用 {wf.train.from} → {wf.train.to}（{wf.train.sessions} 日）算权重，
          在 {wf.test.from} → {wf.test.to}（{wf.test.sessions} 日）上打分
        </span>
      </div>

      <div className="overflow-auto">
        <table className="w-full border-collapse text-[12px]">
          <thead>
            <tr className="border-b border-line text-ink-mute">
              <th className="text-left font-medium px-2 py-1">方法</th>
              <th className="text-right font-medium px-2 py-1"
                title="拟合区间内的表现 —— 优化器总是赢，因为它就是为此挑选的">样本内波动</th>
              <th className="text-right font-medium px-2 py-1">样本外波动</th>
              <th className="text-right font-medium px-2 py-1">样本外收益</th>
              <th className="text-right font-medium px-2 py-1">样本外夏普</th>
              <th className="text-right font-medium px-2 py-1">样本外回撤</th>
            </tr>
          </thead>
          <tbody>
            {wf.rows.map((r) => (
              <tr key={r.method} className={`border-b border-line/60 ${
                r.method === "current" ? "bg-sunken" : ""}`}>
                <td className="px-2 py-1 font-medium">
                  {r.label}
                  {r === best && <span className="text-cyan text-[11px]"> ← 最稳</span>}
                </td>
                <Cell v={r.in_sample.ann_vol_pct} suffix="%" muted />
                <Cell v={r.out_of_sample.ann_vol_pct} suffix="%" strong />
                <Cell v={r.out_of_sample.ann_return_pct} suffix="%" />
                <Cell v={r.out_of_sample.sharpe} />
                <Cell v={r.out_of_sample.max_drawdown_pct} suffix="%" />
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-[11.5px] text-ink-mute leading-snug">
        {beatsCurrent.length === 0
          ? "在这段没被用来拟合的数据上，没有任何一种优化的波动低于你当前的持仓 —— "
            + "也就是说，现在换仓没有证据支持。"
          : `在这段没被用来拟合的数据上，${beatsCurrent.map((r) => r.label).join("、")}`
            + " 的波动低于当前持仓。"}
        {" "}只有一次切分，样本很小；波动比收益稳定得多，所以右侧的收益列不应该当作依据。
      </p>
    </div>
  );
}

function Cell({ v, suffix = "", muted, strong }: {
  v: number | null; suffix?: string; muted?: boolean; strong?: boolean;
}) {
  return (
    <td className={`px-2 py-1 text-right tnum ${
      muted ? "text-ink-mute" : ""} ${strong ? "font-medium" : ""}`}>
      {fixed(v, 2)}{suffix}
    </td>
  );
}

function Row({ label, now, next, suffix = "", lowerIsBetter = false }: {
  label: string; now: number | null; next: number | null;
  suffix?: string; lowerIsBetter?: boolean;
}) {
  const diff = now != null && next != null ? next - now : null;
  const good = diff == null ? null : lowerIsBetter ? diff < 0 : diff > 0;
  return (
    <div className="rounded-lg bg-sunken px-2.5 py-1.5 flex items-baseline gap-2">
      <span className="text-[12.5px] flex-1 min-w-0 truncate">{label}</span>
      <span className="tnum text-[12px] text-ink-mute w-16 text-right"
        title="当前持仓">{fixed(now, 2)}{suffix}</span>
      <span className="text-ink-mute text-[11px]">→</span>
      <span className="tnum text-[14px] font-semibold w-16 text-right">
        {fixed(next, 2)}{suffix}
      </span>
      <span className={`tnum text-[11.5px] w-14 text-right ${
        good == null ? "text-flat" : good ? "text-down" : "text-up"}`}>
        {diff == null ? "—" : signed(diff, 2)}
      </span>
    </div>
  );
}

export type { QtAllocStats };
