/**
 * The picked stocks over the window, and over every leg of the index.
 *
 * The first table is the reference for setting weights. The second is the
 * regime question directly: which of these names led when the index rose, and
 * which held up when it fell. Red is up and green is down.
 */

import { useState } from "react";
import type {
  PortfolioWindowStats, StockRef, WindowRegime, WindowStockStats,
} from "../../lib/types";
import { fixed, signed } from "../../lib/format";
import { MIN_WINDOW } from "../../lib/windowRange";
import { legTitle } from "./WindowBrush";

type Num = number | null;

const updown = (v: Num) => (v == null ? "" : v >= 0 ? "text-up" : "text-down");

const COLS: {
  key: keyof WindowStockStats; label: string; hint: string;
  show: (v: Num) => string; tone?: boolean;
}[] = [
  { key: "change_pct", label: "区间涨跌", tone: true, show: (v) => signed(v, 1, "%"),
    hint: "窗口第一天到最后一天的累计涨跌（前复权，含分红）。" },
  { key: "ann_return_pct", label: "年化收益", tone: true, show: (v) => signed(v, 1, "%"),
    hint: "把区间收益折算成一年。窗口越短，这个数被放大得越厉害。" },
  { key: "ann_vol_pct", label: "年化波动", show: (v) => (v == null ? "—" : `${fixed(v, 1)}%`),
    hint: "日收益的标准差折算成一年。" },
  { key: "sharpe", label: "夏普", show: (v) => (v == null ? "—" : fixed(v, 2)),
    hint: "（年化收益 − 无风险利率）÷ 年化波动。" },
  { key: "max_drawdown_pct", label: "最大回撤", show: (v) => (v == null ? "—" : `${fixed(v, 1)}%`),
    hint: "窗口内从高点到低点最深的一次下跌。" },
  { key: "beta", label: "β", show: (v) => (v == null ? "—" : fixed(v, 2)),
    hint: "指数涨跌 1% 时，这只股票平均涨跌多少。" },
];

export function WindowStats({ s, picks, selected, onSnap }: {
  s: PortfolioWindowStats;
  picks: StockRef[];
  selected: { from: string; to: string };
  onSnap: (r: WindowRegime) => void;
}) {
  const name = (t: string) => picks.find((p) => p.t === t)?.n ?? t;
  const [sort, setSort] = useState<{ key: keyof WindowStockStats; desc: boolean } | null>(null);
  const rows = sort
    ? [...s.rows].sort((a, b) => {
        const x = a[sort.key] as Num;
        const y = b[sort.key] as Num;
        if (x == null) return 1;
        if (y == null) return -1;
        return sort.desc ? y - x : x - y;
      })
    : s.rows;
  const clickSort = (key: keyof WindowStockStats) =>
    setSort((c) => (c?.key === key ? { key, desc: !c.desc } : { key, desc: true }));

  const partial = s.rows.filter((r) => r.partial || r.days === 0);

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto">
        <table className="w-full text-[12.5px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-3 whitespace-nowrap">本窗口</th>
              {COLS.map((c) => (
                <th key={c.key} title={c.hint}
                  className="text-right font-normal pb-1 px-2 whitespace-nowrap">
                  <button onClick={() => clickSort(c.key)} className="hover:text-ink">
                    {c.label}{sort?.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <StatRow label={s.index.label} r={s.index} muted />
            {rows.map((r) => (
              <StatRow key={r.t} label={name(r.t)} code={r.t} r={r} />
            ))}
          </tbody>
        </table>
      </div>

      {partial.map((r) => (
        <p key={r.t} className="text-[12px] text-brand-ink leading-snug">
          {name(r.t)}{r.days === 0
            ? " 在这个窗口里没有行情。"
            : ` 从 ${r.from} 才有行情，这一行只算它交易过的 ${r.days} 天，和其他股票不是同一段。`}
        </p>
      ))}
      {s.missing.length > 0 && (
        <p className="text-[12px] text-up">读不到行情：{s.missing.join("、")}</p>
      )}
      {s.sessions < 120 && (
        <p className="label leading-snug">窗口较短，年化数字会被放大，先看区间涨跌。</p>
      )}

      <RegimeGrid s={s} name={name} selected={selected} onSnap={onSnap} />
    </div>
  );
}

function StatRow({ label, code, r, muted }: {
  label: string; code?: string; r: WindowStockStats; muted?: boolean;
}) {
  return (
    <tr className={`border-t border-line ${muted ? "text-ink-dim" : ""}`}>
      <td className="py-1 pr-3 whitespace-nowrap">
        <span className={muted ? "" : "font-medium"}>{label}</span>
        {code && <span className="hidden sm:inline ml-1.5 font-mono tnum text-[11px] text-ink-mute">{code}</span>}
        {r.partial && <span className="ml-1 text-brand-ink" title={`从 ${r.from} 才有行情`}>⚠</span>}
      </td>
      {COLS.map((c) => {
        const v = r[c.key] as Num;
        return (
          <td key={c.key} className={`py-1 px-2 text-right tnum ${c.tone ? updown(v) : ""}`}>
            {v == null ? "—" : c.show(v)}
          </td>
        );
      })}
    </tr>
  );
}

function RegimeGrid({ s, name, selected, onSnap }: {
  s: PortfolioWindowStats;
  name: (t: string) => string;
  selected: { from: string; to: string };
  onSnap: (r: WindowRegime) => void;
}) {
  const stocks = s.rows.map((r) => r.t);
  // Scaled per column: one +300% leg should not wash out every other leg.
  const scale = s.regimes.map((_, c) => Math.max(
    1, ...stocks.map((t) => Math.abs(s.cells[t]?.[c] ?? 0))));

  return (
    <div className="flex flex-col gap-1">
      <span className="text-[12.5px] font-medium">各段行情里的涨跌（最新的在左）</span>
      <div className="overflow-x-auto">
        <table className="text-[12px] border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 pr-3" style={{ background: "var(--color-panel)" }} />
              {s.regimes.map((r) => {
                const pickable = r.sessions >= MIN_WINDOW;
                const on = r.from === selected.from && r.to === selected.to;
                return (
                  <th key={r.from} className="font-normal pb-1 px-0.5">
                    <button onClick={() => pickable && onSnap(r)} disabled={!pickable}
                      title={legTitle(r)}
                      className={`w-[74px] rounded-md py-1 tnum leading-tight text-[11px] ${
                        on ? "ring-2 ring-cyan" : ""} ${
                        pickable ? "hover:bg-elevated" : "cursor-default opacity-70"}`}>
                      <span className="block text-ink-mute">{r.from.slice(2)}</span>
                      <span className="block text-ink-mute">{r.open ? "至今" : r.to.slice(2)}</span>
                      <span className={`block font-medium ${r.kind === "up" ? "text-up" : "text-down"}`}>
                        {r.kind === "up" ? "上行" : "下行"}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-line text-ink-dim">
              <td className="sticky left-0 z-10 py-1 pr-3 whitespace-nowrap"
                style={{ background: "var(--color-panel)" }}>{s.index.label}</td>
              {s.regimes.map((r) => (
                <td key={r.from} className="py-1 px-0.5 text-center tnum font-medium">
                  {signed(r.change_pct, 1, "%")}
                </td>
              ))}
            </tr>
            {stocks.map((t) => (
              <tr key={t} className="border-t border-line">
                <td className="sticky left-0 z-10 py-1 pr-3 whitespace-nowrap font-medium"
                  style={{ background: "var(--color-panel)" }}>{name(t)}</td>
                {s.regimes.map((r, c) => {
                  const v = s.cells[t]?.[c] ?? null;
                  const strength = v == null ? 0 : Math.round((Math.abs(v) / scale[c]!) * 38);
                  return (
                    <td key={r.from} className="p-0.5 text-center tnum">
                      <div className="rounded-sm py-0.5"
                        title={v == null
                          ? "这段行情里还没上市"
                          : `比${s.index.label} ${signed(v - r.change_pct, 1, "pp")}`}
                        style={v == null ? undefined : {
                          background: `color-mix(in srgb, var(--color-${v >= 0 ? "up" : "down"}) ${strength}%, transparent)`,
                        }}>
                        {v == null ? "—" : signed(v, 0, "%")}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
