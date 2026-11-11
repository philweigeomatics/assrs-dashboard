/**
 * 交易记录 — opened under one holding's row, for resizing that position.
 *
 * The question is "which of the shares I hold were bought above today's
 * price, and by how much?", so the buys and sells are laid against the
 * current price twice: on a price ladder (where each fill sits relative to
 * now and to the position's average cost) and in the table, fill by fill.
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
import { lotOf, summarise, type ActivityRow } from "../../lib/lots";

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
      {q.data && <Body d={q.data} h={h} type={type} years={years} />}
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

function Body({ d, h, type, years }: {
  d: QtSymbolActivity; h: QtHolding; type: string; years: number;
}) {
  const rows = useMemo(
    () => (type === ALL ? d.rows : d.rows.filter((r) => r.type === type)),
    [d.rows, type]);
  const s = useMemo(() => summarise(d.rows, h.price, h.currency), [d.rows, h]);
  const showTrades = type === ALL || type === "Trades";

  // Fewer shares on record than are held means the rest came in before the
  // window (or by transfer), and the averages above leave them out.
  const net = s.bought - s.sold;
  const missing = h.quantity - net;

  if (d.rows.length === 0) {
    return (
      <p className="label py-3">
        近 {years} 年没有 {h.symbol} 的记录 —— 可能是更早买入，或从别处转入。试试更长的年份。
      </p>
    );
  }

  return (
    <>
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
              <span className="flex items-baseline gap-1.5"
                title="每笔买入 ×（现价 − 成交价）之和，不扣除之后卖出的部分，也不含佣金。">
                <span className="label">买入按现价</span>
                <span className="text-down tnum">{s.below} 笔低于现价</span>
                <span className="text-ink-mute">·</span>
                <span className="text-up tnum">{s.above} 笔高于现价</span>
                <span className={`tnum font-medium ${moveClass(s.buy_pnl, NA)}`}>
                  {cash(s.buy_pnl, h.currency, 0)}
                </span>
              </span>
            )}
          </div>
          <Ladder rows={d.rows} h={h} />
        </>
      )}

      {showTrades && missing > 1e-6 && (
        <p className="text-[11.5px] text-ink-dim">
          记录里净买入 {fixed(net, net % 1 ? 4 : 0)} 股，目前持有
          {" "}{fixed(h.quantity, h.quantity % 1 ? 4 : 0)} 股 ——
          另外 {fixed(missing, missing % 1 ? 4 : 0)} 股是更早买入或转入的，不在近 {years} 年的记录里。
        </p>
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
 * Every fill on one price axis, with today's price and the position's
 * average cost drawn across it. Buys sit above the axis, sells below; size
 * is the share count. Green is a fill that looks right from today (a buy
 * below the current price, a sell above it), red the opposite.
 */
function Ladder({ rows, h }: { rows: ActivityRow[]; h: QtHolding }) {
  const lots = rows
    .map((r) => ({ r, lot: lotOf(r, h.price, h.currency) }))
    .filter((x): x is { r: ActivityRow; lot: NonNullable<ReturnType<typeof lotOf>> } =>
      !!x.lot && x.r.currency === h.currency);
  if (lots.length === 0 || h.price == null) return null;

  const W = 640, H = 78, PAD = 18, MID = 44;
  const prices = [...lots.map((x) => x.lot.price), h.price,
    ...(h.avg_cost != null ? [h.avg_cost] : [])];
  let lo = Math.min(...prices), hi = Math.max(...prices);
  const span = hi - lo || hi * 0.1 || 1;
  lo -= span * 0.06; hi += span * 0.06;
  const x = (p: number) => PAD + (p - lo) / (hi - lo) * (W - PAD * 2);
  const maxQ = Math.max(...lots.map((l) => l.lot.quantity));
  const rad = (q: number) => 3 + Math.sqrt(q / maxQ) * 6;

  const good = "var(--color-down)", bad = "var(--color-up)";
  const now = h.price;
  return (
    <div className="flex flex-col gap-0.5">
    <span className="label text-[11px]">
      轴线上方为买入、下方为卖出（虚线圈），圈越大股数越多；绿色 = 从今天看是对的
      （买在现价以下 / 卖在现价以上），红色相反。悬停可看每笔详情。
    </span>
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="max-w-[760px] h-auto"
      role="img" aria-label={`${h.symbol} 每笔成交价与现价`}>
      <line x1={PAD} x2={W - PAD} y1={MID} y2={MID} stroke="var(--color-line)" />
      <text x={PAD} y={H - 3} fontSize="10" fill="var(--color-ink-mute)">
        {cash(lo, h.currency)}
      </text>
      <text x={W - PAD} y={H - 3} fontSize="10" textAnchor="end" fill="var(--color-ink-mute)">
        {cash(hi, h.currency)}
      </text>

      {h.avg_cost != null && (
        <g>
          <line x1={x(h.avg_cost)} x2={x(h.avg_cost)} y1={14} y2={H - 14}
            stroke="var(--color-cyan)" strokeDasharray="3 3" />
          <text x={x(h.avg_cost)} y={H - 3} fontSize="10" textAnchor="middle"
            fill="var(--color-cyan)">均价 {fixed(h.avg_cost, 2)}</text>
        </g>
      )}
      <line x1={x(now)} x2={x(now)} y1={10} y2={H - 14}
        stroke="var(--color-ink)" strokeWidth="1.5" />
      <text x={x(now)} y={9} fontSize="10" textAnchor="middle" fontWeight="600"
        fill="var(--color-ink)">现价 {fixed(now, 2)}</text>

      {lots.map(({ r, lot }, i) => {
        const cx = x(lot.price);
        const ok = lot.side === "buy" ? lot.price <= now : lot.price >= now;
        const rr = rad(lot.quantity);
        const y = lot.side === "buy" ? MID - 3 - rr : MID + 3 + rr;
        return (
          <circle key={i} cx={cx} cy={y} r={rr}
            fill={ok ? good : bad} fillOpacity={0.35}
            stroke={ok ? good : bad} strokeWidth={lot.side === "sell" ? 1.5 : 1}
            strokeDasharray={lot.side === "sell" ? "2 1.5" : undefined}>
            <title>
              {`${r.date} ${lot.side === "buy" ? "买入" : "卖出"} `
                + `${fixed(lot.quantity, lot.quantity % 1 ? 4 : 0)} 股 @ `
                + `${cash(lot.price, r.currency)} · ${r.account}`
                + (lot.vs_now_pct == null ? "" : `\n现价相对 ${signed(lot.vs_now_pct, 1)}%`)}
            </title>
          </circle>
        );
      })}
    </svg>
    </div>
  );
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
