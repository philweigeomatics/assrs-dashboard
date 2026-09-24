/**
 * 🛢 大宗商品 — front-month prices, and one forward curve in full.
 *
 * A futures price on its own says nothing. "Copper is 111,330" is a number;
 * "copper is 111,330 and the near contract is richer than the far one" says
 * spot is tight. So every tile on the board carries its market state, and
 * picking one opens the curve that state was read off.
 *
 * The curve is drawn against real maturities rather than evenly spaced
 * contract names: the gap between the front two months is what the roll
 * yield is annualised over, and an even spacing hides it.
 *
 * Bubble size is open interest. Trust the slope between big bubbles; a thin
 * contract near expiry can fake a steep segment on its own.
 */

import type { CommodityBoard } from "../../lib/types";
import { signed } from "../../lib/format";

const VW = 760;
const VH = 250;
const PAD_L = 62;
const PAD_B = 34;
const PAD_T = 14;

const STATE: Record<string, { label: string; cls: string; dot: string }> = {
  backwardation: { label: "近强远弱 Backwardation",
                   cls: "text-up border-up/40 bg-up/5", dot: "var(--color-up)" },
  contango: { label: "近弱远强 Contango",
              cls: "text-down border-down/40 bg-down/5", dot: "var(--color-down)" },
  flat: { label: "曲线平坦 Flat",
          cls: "text-ink-mute border-line bg-sunken", dot: "var(--color-ink-mute)" },
};

export function CommodityPanel({ data, onCode, liquid, onLiquid }: {
  data: CommodityBoard;
  onCode: (c: string) => void;
  liquid: boolean;
  onLiquid: (v: boolean) => void;
}) {
  const groups: string[] = [];
  for (const b of data.board) if (!groups.includes(b.group)) groups.push(b.group);
  const picked = data.detail;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2">
        {groups.map((g) => (
          <div key={g} className="flex flex-col gap-1.5">
            <h4 className="text-[12.5px] font-semibold text-ink-dim">{g}</h4>
            <div className="grid gap-2 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4
              xl:grid-cols-6">
              {data.board.filter((b) => b.group === g).map((b) => {
                const on = picked?.code === b.code;
                const st = b.state ? STATE[b.state] : null;
                return (
                  <button key={b.code} onClick={() => onCode(b.code)}
                    title={`${b.en}${b.symbol ? ` · ${b.symbol}` : ""}`}
                    className={`rounded-lg px-2.5 py-2 flex flex-col gap-0.5 text-left
                      transition-colors border ${
                        on ? "bg-elevated border-cyan/50" : "bg-sunken border-transparent"
                      } hover:border-cyan/30`}>
                    <span className="flex items-center gap-1.5">
                      <span className="text-[12.5px] font-medium truncate">
                        {b.label}
                      </span>
                      {st && (
                        <span className="w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ background: st.dot }} />
                      )}
                    </span>
                    <span className="text-[15px] font-semibold tnum">
                      {b.price == null ? "—" : b.price.toLocaleString(undefined,
                        { maximumFractionDigits: 2 })}
                    </span>
                    <span className="label truncate">{b.unit}</span>
                    {b.roll_ann_pct != null && (
                      <span className={`text-[11px] tnum ${
                        b.roll_ann_pct > 0 ? "text-up"
                          : b.roll_ann_pct < 0 ? "text-down" : "text-ink-mute"}`}
                        title="年化展期收益：把近月换成次近月，一年下来是赚是赔">
                        展期 {signed(b.roll_ann_pct, 1, "%")}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {picked ? (
        <Detail d={picked} liquid={liquid} onLiquid={onLiquid} />
      ) : (
        <p className="label py-4">这个品种今天没有返回可用合约。</p>
      )}

      <p className="label leading-snug">
        结算价来自 Tushare，交易日 {data.trade_date}。曲线向下（近月更贵）是现货紧张，
        向上（远月更贵）是仓储与资金成本被计入。
      </p>
    </div>
  );
}

function Detail({ d, liquid, onLiquid }: {
  d: NonNullable<CommodityBoard["detail"]>;
  liquid: boolean; onLiquid: (v: boolean) => void;
}) {
  const t = d.term;
  const st = t ? STATE[t.state] : null;

  return (
    <div className="flex flex-col gap-2.5 rounded-lg bg-sunken p-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-[13.5px] font-semibold">
          {d.label} <span className="text-ink-mute font-normal">{d.en}</span>
        </h4>
        {st && (
          <span className={`rounded-md px-2 py-0.5 text-[12px] font-medium border ${st.cls}`}>
            {st.label}
          </span>
        )}
        <label className="ml-auto label flex items-center gap-1.5 cursor-pointer"
          title="中国商品的流动性集中在 1/5/9 月合约，其余月份的结算价是挂出来的，不是成交出来的。">
          <input type="checkbox" checked={liquid}
            onChange={(e) => onLiquid(e.target.checked)}
            className="accent-[var(--color-cyan)]" />
          只看活跃月份
        </label>
      </div>

      {t && (
        <>
          <p className="text-[12.5px] leading-snug">{t.note}</p>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <Stat label={`近月 ${t.front_symbol}`}
              v={t.front_price.toLocaleString()} />
            <Stat label={`次近 ${t.next_symbol}`}
              v={t.next_price.toLocaleString()} />
            <Stat label="跨期价差" v={signed(t.spread, 2, "")}
              tone={t.spread > 0 ? "up" : t.spread < 0 ? "down" : undefined}
              hint={`近月减次近，单位 ${t.unit}`} />
            <Stat label="年化展期收益" v={signed(t.roll_ann_pct, 2, "%")}
              tone={t.roll_ann_pct > 0 ? "up" : t.roll_ann_pct < 0 ? "down" : undefined}
              strong
              hint={`把近月换成次近月的年化损益，两者相隔 ${t.days_between} 天`} />
          </div>
        </>
      )}

      <Curve d={d} />

      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="text-ink-mute">
              <th className="text-left font-normal pb-1 pr-2">合约</th>
              <th className="text-left font-normal pb-1 px-2">最后交易日</th>
              <th className="text-right font-normal pb-1 px-2">结算价</th>
              <th className="text-right font-normal pb-1 px-2">对近月价差</th>
              <th className="text-right font-normal pb-1 pl-2">持仓量</th>
            </tr>
          </thead>
          <tbody>
            {d.curve.map((r, i) => (
              <tr key={r.symbol} className="border-t border-line">
                <td className="py-1 pr-2 font-mono tnum">{r.symbol}</td>
                <td className="py-1 px-2 font-mono tnum text-ink-mute">
                  {r.maturity}
                </td>
                <td className="py-1 px-2 text-right tnum font-medium">
                  {r.price.toLocaleString()}
                </td>
                <td className={`py-1 px-2 text-right tnum ${
                  i === 0 ? "text-ink-mute"
                    : r.spread > 0 ? "text-up" : "text-down"}`}>
                  {i === 0 ? "近月" : `${signed(r.spread, 2, "")} (${
                    signed(r.spread_pct, 2, "%")})`}
                </td>
                <td className="py-1 pl-2 text-right tnum text-ink-dim">
                  {r.oi.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="label">
        {d.liquid_only && d.shown < d.listed
          ? `${d.listed} 个挂牌合约里，${d.shown} 个持仓量达到主力合约的 10% 以上。`
          : `全部 ${d.listed} 个挂牌合约。`}
        {" "}气泡大小是持仓量 —— 相信大气泡之间的斜率，别理会小的。
      </p>
    </div>
  );
}

/** The curve, spaced by real maturity rather than by contract index. */
function Curve({ d }: { d: NonNullable<CommodityBoard["detail"]> }) {
  const pts = d.curve;
  if (pts.length < 2) {
    return <p className="label py-4">合约不足两个，画不出曲线。</p>;
  }

  const days = pts.map((p) => {
    const [y, m, dd] = p.maturity.split("-").map(Number);
    return Date.UTC(y ?? 2000, (m ?? 1) - 1, dd ?? 1) / 86_400_000;
  });
  const t0 = days[0] ?? 0;
  const tSpan = ((days[days.length - 1] ?? t0) - t0) || 1;
  const prices = pts.map((p) => p.price);
  const first = prices[0] ?? 0;
  const last = prices[prices.length - 1] ?? 0;
  const lo = Math.min(...prices);
  const hi = Math.max(...prices);
  const pad = (hi - lo) * 0.18 || Math.abs(hi) * 0.01 || 1;
  const top = hi + pad;
  const bottom = lo - pad;

  const x = (i: number) =>
    PAD_L + (((days[i] ?? t0) - t0) / tSpan) * (VW - PAD_L - 18);
  const y = (v: number) =>
    VH - PAD_B - ((v - bottom) / (top - bottom)) * (VH - PAD_B - PAD_T);

  const maxOi = Math.max(...pts.map((p) => p.oi), 0);
  const r = (oi: number) => (maxOi > 0 ? 4 + 11 * (oi / maxOi) : 6);

  // Down-sloping = near richer = backwardation = red, on this app's convention.
  const rising = last >= first;
  const stroke = rising ? "var(--color-down)" : "var(--color-up)";
  const line = pts.map((p, i) =>
    `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p.price).toFixed(1)}`).join("");

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full block"
      style={{ maxHeight: VH * 1.2 }} role="img"
      aria-label={`${d.label} 远期曲线，气泡大小为持仓量`}>
      {[top, (top + bottom) / 2, bottom].map((v) => (
        <g key={v}>
          <line x1={PAD_L} x2={VW - 6} y1={y(v)} y2={y(v)}
            stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
          <text x={PAD_L - 6} y={y(v) + 3} textAnchor="end" fill="currentColor"
            className="text-ink-mute tnum" style={{ fontSize: 9.5 }}>
            {Math.round(v).toLocaleString()}
          </text>
        </g>
      ))}

      <path d={line} fill="none" stroke={stroke} strokeWidth={2}
        vectorEffect="non-scaling-stroke" />

      {pts.map((p, i) => (
        <circle key={p.symbol} cx={x(i)} cy={y(p.price)} r={r(p.oi)}
          fill={stroke} fillOpacity={0.35} stroke={stroke} strokeWidth={1.5}
          vectorEffect="non-scaling-stroke">
          <title>{`${p.symbol} · 结算 ${p.price.toLocaleString()} ${d.unit}`
            + ` · 持仓 ${p.oi.toLocaleString()} · 到期 ${p.maturity}`}</title>
        </circle>
      ))}

      {pts.map((p, i) => (
        <text key={p.symbol} x={x(i)} y={VH - PAD_B + 14} textAnchor="middle"
          fill="currentColor" className="text-ink-mute tnum"
          style={{ fontSize: 9 }}>
          {p.symbol.replace(/^[A-Za-z]+/, "")}
        </text>
      ))}
      <text x={PAD_L} y={VH - 6} fill="currentColor" className="text-ink-mute"
        style={{ fontSize: 9.5 }}>近月 → 远月（按实际到期日间隔）</text>
    </svg>
  );
}

function Stat({ label, v, hint, tone, strong }: {
  label: string; v: string; hint?: string;
  tone?: "up" | "down"; strong?: boolean;
}) {
  return (
    <span className="flex items-baseline gap-1.5" title={hint}>
      <span className="label">{label}</span>
      <span className={`tnum ${strong ? "text-[14px] font-semibold"
        : "text-[13px] font-medium"} ${
        tone === "up" ? "text-up" : tone === "down" ? "text-down" : ""}`}>
        {v}
      </span>
    </span>
  );
}
