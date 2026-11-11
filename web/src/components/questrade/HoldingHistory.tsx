/**
 * 交易记录 — opened under one holding's row, for resizing that position.
 *
 * Two questions: "what has this name made me, all in?" — the total return
 * at the top, dividends and closed trades included — and "which of the
 * shares I hold were bought above today's price?", which the table answers
 * fill by fill.
 *
 * Dividends, fees and the rest are listed too — a filter away — but get no
 * price comparison: there is no price on a dividend to compare.
 *
 * Shares the cache key with 🔎 按代码查找, so opening a name here that was
 * already looked up there costs nothing.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";
import type { QtHolding, QtSymbolActivity } from "../../lib/types";
import { fixed, moveClass, signed } from "../../lib/format";
import { lotOf, summarise, totalReturn, type ActivityRow } from "../../lib/lots";

/** North America: green up, red down — same as the holdings table. */
const NA = false;
const SPANS = [3, 5, 10] as const;
const ALL = "*";

function cash(v: number | null | undefined, ccy: string, nd = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const sym = ccy === "CAD" ? "C$" : ccy === "USD" ? "$" : "";
  const s = Math.abs(v).toLocaleString(undefined, {
    minimumFractionDigits: nd, maximumFractionDigits: nd });
  return `${v < 0 ? "-" : ""}${sym}${s}`;
}

export function HoldingHistory({ h }: { h: QtHolding }) {
  const [years, setYears] = useState<number>(3);
  // Buys and sells first: that is what the panel is opened for.
  const [picked, setType] = useState<string>("Trades");

  const q = useQuery({
    queryKey: ["qt", "activity", h.symbol, years],
    queryFn: () => api.qtActivity(h.symbol, years),
    staleTime: 6 * 3600_000,
  });
  // A name that was only ever transferred in has no trades; don't leave the
  // filter pointing at a type with nothing in it.
  const type = q.data && picked !== ALL
    && !q.data.summary.by_type.some((b) => b.type === picked) ? ALL : picked;

  return (
    <div className="flex flex-col gap-2.5 px-3 py-2.5 bg-sunken/60 border-l-2 border-cyan">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[12.5px] font-semibold">
          <span className="font-mono">{h.symbol}</span> 交易记录
        </span>
        {q.data && (
          <TypeFilter d={q.data} value={type} onChange={setType} />
        )}
        <div className="ml-auto flex rounded-lg bg-panel p-0.5" title="往回查多少年">
          {SPANS.map((y) => (
            <button key={y} onClick={() => setYears(y)}
              className={`px-2 h-6 rounded-md text-[11.5px] ${
                years === y ? "bg-sunken text-ink font-medium" : "text-ink-mute"}`}>
              近 {y} 年
            </button>
          ))}
        </div>
      </div>

      {q.isPending && (
        <p className="label py-4 text-center">
          读取中…{years > 3 ? "（每多一年多一次查询，首次较慢）" : ""}
        </p>
      )}
      {q.isError && (
        <p className="text-[12px] text-up">{(q.error as ApiError).message}</p>
      )}
      {q.data && <Body d={q.data} h={h} type={type} years={years} onYears={setYears} />}
    </div>
  );
}

function TypeFilter({ d, value, onChange }: {
  d: QtSymbolActivity; value: string; onChange: (t: string) => void;
}) {
  const opts = [
    { type: ALL, label: "全部", count: d.rows.length },
    ...d.summary.by_type.map((b) => ({ type: b.type, label: b.type_label, count: b.count })),
  ];
  return (
    <div className="flex flex-wrap gap-1">
      {opts.map((o) => (
        <button key={o.type} onClick={() => onChange(o.type)}
          className={`h-6 px-2 rounded-md text-[11.5px] transition-colors ${
            value === o.type ? "bg-cyan text-white"
              : "bg-panel text-ink-dim hover:text-ink"}`}>
          {o.label}
          <span className={`ml-1 tnum ${value === o.type ? "text-white/70" : "text-ink-mute"}`}>
            {o.count}
          </span>
        </button>
      ))}
    </div>
  );
}

function Body({ d, h, type, years, onYears }: {
  d: QtSymbolActivity; h: QtHolding; type: string; years: number;
  onYears: (y: number) => void;
}) {
  const rows = useMemo(
    () => (type === ALL ? d.rows : d.rows.filter((r) => r.type === type)),
    [d.rows, type]);
  const s = useMemo(() => summarise(d.rows, h.price, h.currency), [d.rows, h]);
  const showTrades = type === ALL || type === "Trades";

  if (d.rows.length === 0) {
    return (
      <p className="label py-3">
        近 {years} 年没有 {h.symbol} 的记录 —— 可能是更早买入，或从别处转入。试试更长的年份。
      </p>
    );
  }

  return (
    <>
      <Total d={d} h={h} years={years} onYears={onYears} />

      {showTrades && s.buys + s.sells > 0 && (
        <>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12px]">
            <Stat label="买入" v={`${s.buys} 笔 · ${fixed(s.bought, s.bought % 1 ? 4 : 0)} 股`}
              note={s.avg_buy == null ? undefined : `均价 ${cash(s.avg_buy, h.currency)}`} />
            {s.sells > 0 && (
              <Stat label="卖出" v={`${s.sells} 笔 · ${fixed(s.sold, s.sold % 1 ? 4 : 0)} 股`}
                note={s.avg_sell == null ? undefined : `均价 ${cash(s.avg_sell, h.currency)}`} />
            )}
            <Stat label="现价" v={cash(h.price, h.currency)}
              note={s.avg_vs_now_pct == null ? undefined
                : `较买入均价 ${signed(s.avg_vs_now_pct, 1)}%`}
              tone={moveClass(s.avg_vs_now_pct, NA)} />
            <Stat label="持仓均价" v={cash(h.avg_cost, h.currency)}
              note="Questrade 成本" />
            {s.buys > 0 && s.buy_pnl != null && (
              <span className="flex items-baseline gap-1.5">
                <span className="label">买入</span>
                <span className="text-down tnum">{s.below} 笔低于现价</span>
                <span className="text-ink-mute">·</span>
                <span className="text-up tnum">{s.above} 笔高于现价</span>
              </span>
            )}
          </div>
        </>
      )}

      <Table rows={rows} h={h} />

      {showTrades && s.buys + s.sells > 0 && (
        <p className="label leading-snug">
          「相对现价」与「按现价」只算买卖：买入 = 数量 ×（现价 − 成交价），
          即这一笔现在的浮动盈亏；卖出 = 数量 ×（成交价 − 现价），正数表示当时卖得比现在高。
          按成交价计算，不含佣金，也不是调整后成本基础（ACB）。
        </p>
      )}
    </>
  );
}

function Table({ rows, h }: { rows: ActivityRow[]; h: QtHolding }) {
  if (rows.length === 0) return <p className="label py-2">这一类没有记录。</p>;
  // Dividends and fees have no fill price: drop the comparison columns
  // rather than show three columns of dashes.
  const trades = rows.some((r) => lotOf(r, h.price, h.currency));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px] border-collapse">
        <thead>
          <tr className="text-ink-mute">
            <th className="text-left font-normal pb-1 pr-2">日期</th>
            <th className="text-left font-normal pb-1 px-2">账户</th>
            <th className="text-left font-normal pb-1 px-2">类型</th>
            <th className="text-right font-normal pb-1 px-2">数量</th>
            {trades && (
              <>
                <th className="text-right font-normal pb-1 px-2">成交价</th>
                <th className="text-right font-normal pb-1 px-2"
                  title="现价相对这一笔成交价">相对现价</th>
                <th className="text-right font-normal pb-1 px-2"
                  title="买入：这一笔现在的浮动盈亏。卖出：当时卖出比现在卖多得（正）或少得（负）。">
                  按现价
                </th>
              </>
            )}
            <th className="text-right font-normal pb-1 px-2">净额</th>
            <th className="text-left font-normal pb-1 pl-2">说明</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const lot = lotOf(r, h.price, h.currency);
            return (
              <tr key={`${r.date}-${r.account_id}-${i}`} className="border-t border-line">
                <td className="py-1 pr-2 font-mono tnum whitespace-nowrap">{r.date}</td>
                <td className="py-1 px-2 whitespace-nowrap">{r.account}</td>
                <td className="py-1 px-2 whitespace-nowrap">
                  {lot ? (
                    <span className={`inline-block px-1.5 h-[17px] leading-[17px] rounded
                      text-[10.5px] font-medium ${lot.side === "buy"
                        ? "bg-cyan/10 text-cyan" : "bg-brand/15 text-brand-ink"}`}>
                      {lot.side === "buy" ? "买入" : "卖出"}
                    </span>
                  ) : (
                    <>
                      {r.type_label}
                      {r.action && <span className="text-ink-mute"> {r.action}</span>}
                    </>
                  )}
                </td>
                <td className="py-1 px-2 text-right tnum">
                  {r.quantity ? fixed(Math.abs(r.quantity), r.quantity % 1 ? 4 : 0) : "—"}
                </td>
                {trades && (
                  <>
                    <td className="py-1 px-2 text-right tnum">
                      {r.price ? cash(r.price, r.currency) : "—"}
                    </td>
                    {/* Coloured by how the fill looks from today, not by the
                        sign of the move: a sell the price fell after is good. */}
                    <td className={`py-1 px-2 text-right tnum ${
                      lot ? moveClass(lot.pnl, NA) : "text-ink-mute"}`}>
                      {lot?.vs_now_pct != null ? `${signed(lot.vs_now_pct, 1)}%` : "—"}
                    </td>
                    <td className={`py-1 px-2 text-right tnum font-medium ${
                      lot ? moveClass(lot.pnl, NA) : "text-ink-mute"}`}>
                      {lot?.pnl != null ? cash(lot.pnl, h.currency, 0) : "—"}
                    </td>
                  </>
                )}
                <td className="py-1 px-2 text-right tnum whitespace-nowrap">
                  {r.net == null ? "—" : `${r.currency} ${r.net.toLocaleString(undefined,
                    { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
                </td>
                <td className="py-1 pl-2 text-ink-dim max-w-[240px] truncate"
                  title={r.description}>
                  {r.description}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * 总回报 — the whole of what this ticker has made, in one number.
 *
 * Today's market value plus every cash flow on record: buys (with their
 * commission) out, sells and dividends in, fees and withholding out. Then
 * the same number taken apart into what is still open, what the sells
 * locked in, dividends, and the rest — so the parts add up to the total.
 */
function Total({ d, h, years, onYears }: {
  d: QtSymbolActivity; h: QtHolding; years: number; onYears: (y: number) => void;
}) {
  const t = useMemo(() => totalReturn(d.rows, h), [d.rows, h]);
  // Always shown when records are missing: that gap is the thing to say.
  if (t.complete && t.invested <= 0 && t.dividends === 0) return null;
  const traded = t.invested > 0 || t.value > 0;
  const ccy = h.currency;
  const gap = t.held - t.covered;
  const longer = SPANS.find((y) => y > years);

  return (
    <div className="flex flex-col gap-1.5 rounded-lg bg-panel px-3 py-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="text-[12.5px] font-semibold">
          {t.complete ? "总回报" : "有记录部分的回报"}
        </span>
        <span className={`text-[18px] font-semibold tnum ${moveClass(t.total, NA)}`}>
          {cash(t.total, ccy, 0)}
        </span>
        {t.pct != null && (
          <span className={`text-[13px] font-medium tnum ${moveClass(t.total, NA)}`}>
            {signed(t.pct, 1)}%
          </span>
        )}
        {t.invested > 0 && (
          <span className="label tnum"
            title="累计买入花掉的现金（含佣金）。卖了又买的部分会重复计入，所以回报率偏保守。">
            相对累计买入 {cash(t.invested, ccy, 0)}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-[12px]">
        <span className="label">=</span>
        {t.unrealised != null && t.realised != null ? (
          <>
            <Part label="浮动盈亏（持有中）" v={t.unrealised} ccy={ccy}
              title="现在持有的股份，按 Questrade 的持仓成本算的浮动盈亏" />
            <Plus />
            <Part label="已实现（卖出）" v={t.realised} ccy={ccy}
              title="卖出已经锁定的盈亏，含买卖佣金" />
            <Plus />
          </>
        ) : traded ? (
          <>
            <Part label="买卖（含现市值）" v={t.total - t.dividends - t.other}
              ccy={ccy} title="有记录的股份现市值 + 全部买卖现金流（含佣金）" />
            <Plus />
          </>
        ) : null}
        <Part label="股息" v={t.dividends} ccy={ccy} title="到账股息，已扣预扣税" />
        {Math.abs(t.other) >= 0.005 && (
          <>
            <Plus />
            <Part label="费用及其他" v={t.other} ccy={ccy}
              title="费用、返还和其他带这个代码的现金记录" />
          </>
        )}
      </div>

      {!t.complete && (
        <p className="text-[11.5px] text-ink-dim">
          近 {years} 年的记录只覆盖 {fixed(t.covered, t.covered % 1 ? 4 : 0)} 股，
          目前持有 {fixed(t.held, t.held % 1 ? 4 : 0)} 股。
          {gap > 0 && <>另外 {fixed(gap, gap % 1 ? 4 : 0)} 股是更早买入或转入的，成本不在记录里，所以没算进去。</>}
          {longer && (
            <button onClick={() => onYears(longer)} className="ml-1 text-cyan">
              查近 {longer} 年 →
            </button>
          )}
        </p>
      )}
      {t.skipped > 0 && (
        <p className="label">另有 {t.skipped} 条非 {ccy} 记录没有计入（不同货币不相加）。</p>
      )}
    </div>
  );
}

function Part({ label, v, ccy, title }: {
  label: string; v: number; ccy: string; title?: string;
}) {
  return (
    <span className="flex items-baseline gap-1" title={title}>
      <span className="label">{label}</span>
      <span className={`tnum font-medium ${moveClass(v, NA)}`}>
        {v > 0 ? "+" : ""}{cash(v, ccy, 0)}
      </span>
    </span>
  );
}

function Plus() {
  return <span className="text-ink-mute">+</span>;
}

function Stat({ label, v, note, tone }: {
  label: string; v: string; note?: string; tone?: string;
}) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="label">{label}</span>
      <span className="tnum font-medium">{v}</span>
      {note && <span className={`text-[11px] tnum ${tone ?? "text-ink-mute"}`}>{note}</span>}
    </span>
  );
}
