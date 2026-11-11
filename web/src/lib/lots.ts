/**
 * Each fill measured against today's price.
 *
 * The question this answers is a sizing one — "of the shares I bought, which
 * were bought above where it trades now, and by how much?" — so every number
 * is plain price arithmetic on one fill. It is NOT a realised gain, not an
 * adjusted cost base, and commissions are left out: those belong to the
 * ledger and the tax return, not to "should I add or trim this".
 *
 * Only buys and sells get a comparison. A dividend has no price to compare,
 * and pretending it does would put a number where there is no question.
 */

import type { Num, QtSymbolActivity } from "./types";

export type ActivityRow = QtSymbolActivity["rows"][number];

export type Lot = {
  side: "buy" | "sell";
  /** Always positive; `side` carries the direction. */
  quantity: number;
  price: number;
  /** Today's price relative to the fill: +20 means it trades 20% above it. */
  vs_now_pct: Num;
  /**
   * Buy: what that lot is up or down at today's price.
   * Sell: what selling then made versus selling now — positive means the
   * shares went for more than they fetch today.
   */
  pnl: Num;
};

/** A row is a fill when it is a trade with a quantity and a price. */
export function lotOf(r: ActivityRow, now: Num, ccy: string): Lot | null {
  if (r.type !== "Trades") return null;
  const q = r.quantity ?? 0;
  const p = r.price ?? 0;
  if (!q || !p) return null;
  const side = q > 0 ? "buy" : "sell";
  const quantity = Math.abs(q);
  // A fill in another currency (a .U listing bought with USD, say) has no
  // honest comparison with a price quoted in this one.
  const comparable = now != null && Number.isFinite(now) && r.currency === ccy;
  if (!comparable) return { side, quantity, price: p, vs_now_pct: null, pnl: null };
  const diff = side === "buy" ? now - p : p - now;
  return {
    side, quantity, price: p,
    vs_now_pct: (now / p - 1) * 100,
    pnl: diff * quantity,
  };
}

export type LotSummary = {
  buys: number;
  /** Buys whose price is below today's — the ones in the money. */
  below: number;
  above: number;
  bought: number;
  avg_buy: Num;
  avg_vs_now_pct: Num;
  /** Every buy marked to today's price, summed. Ignores what was sold since. */
  buy_pnl: Num;
  sells: number;
  sold: number;
  avg_sell: Num;
};

export function summarise(rows: ActivityRow[], now: Num, ccy: string): LotSummary {
  let buys = 0, below = 0, above = 0, bought = 0, cost = 0, pnl = 0, priced = 0;
  let sells = 0, sold = 0, proceeds = 0;
  for (const r of rows) {
    const lot = lotOf(r, now, ccy);
    if (!lot || r.currency !== ccy) continue;
    if (lot.side === "buy") {
      buys += 1;
      bought += lot.quantity;
      cost += lot.quantity * lot.price;
      if (lot.pnl != null) {
        priced += 1;
        pnl += lot.pnl;
        if (lot.price < now!) below += 1;
        else if (lot.price > now!) above += 1;
      }
    } else {
      sells += 1;
      sold += lot.quantity;
      proceeds += lot.quantity * lot.price;
    }
  }
  const avg_buy = bought ? cost / bought : null;
  return {
    buys, below, above, bought, avg_buy,
    avg_vs_now_pct: avg_buy && now != null && Number.isFinite(now)
      ? (now / avg_buy - 1) * 100 : null,
    buy_pnl: priced ? pnl : null,
    sells, sold,
    avg_sell: sold ? proceeds / sold : null,
  };
}
