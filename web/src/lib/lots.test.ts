import { describe, expect, it } from "vitest";
import { lotOf, summarise, type ActivityRow } from "./lots";

const row = (over: Partial<ActivityRow>): ActivityRow => ({
  date: "2026-01-02", settled: "2026-01-03", type: "Trades", type_label: "买卖",
  action: "Buy", symbol: "VFV.TO", description: "", quantity: 10, price: 100,
  gross: null, commission: null, net: null, currency: "CAD", internal: false,
  account: "TFSA", account_id: "1", registered: true, ...over,
});

describe("lotOf", () => {
  it("marks a buy to today's price", () => {
    const lot = lotOf(row({ quantity: 10, price: 100 }), 120, "CAD")!;
    expect(lot.side).toBe("buy");
    expect(lot.vs_now_pct).toBeCloseTo(20);
    expect(lot.pnl).toBeCloseTo(200);
  });

  it("scores a sell by what it fetched against today", () => {
    // Sold at 100, now 80: selling then was worth 20 a share more.
    const lot = lotOf(row({ quantity: -5, price: 100, action: "Sell" }), 80, "CAD")!;
    expect(lot.side).toBe("sell");
    expect(lot.quantity).toBe(5);
    expect(lot.pnl).toBeCloseTo(100);
    expect(lot.vs_now_pct).toBeCloseTo(-20);
  });

  it("gives no comparison for a dividend", () => {
    expect(lotOf(row({ type: "Dividends", quantity: 0, price: 0, net: 12 }), 120, "CAD"))
      .toBeNull();
  });

  it("refuses to compare across currencies", () => {
    const lot = lotOf(row({ currency: "USD" }), 120, "CAD")!;
    expect(lot.pnl).toBeNull();
    expect(lot.vs_now_pct).toBeNull();
  });
});

describe("summarise", () => {
  it("counts buys above and below today's price and averages them", () => {
    const s = summarise([
      row({ quantity: 10, price: 90 }),
      row({ quantity: 10, price: 110 }),
      row({ quantity: 20, price: 95 }),
      row({ quantity: -5, price: 130, action: "Sell" }),
      row({ type: "Dividends", quantity: 0, price: 0, net: 4 }),
    ], 100, "CAD");
    expect(s.buys).toBe(3);
    expect(s.below).toBe(2);
    expect(s.above).toBe(1);
    expect(s.bought).toBe(40);
    expect(s.avg_buy).toBeCloseTo(97.5);
    expect(s.avg_vs_now_pct).toBeCloseTo((100 / 97.5 - 1) * 100);
    expect(s.buy_pnl).toBeCloseTo(100 + -100 + 100);
    expect(s.sells).toBe(1);
    expect(s.avg_sell).toBeCloseTo(130);
  });

  it("has no averages when nothing was traded", () => {
    const s = summarise([row({ type: "Dividends", quantity: 0, price: 0 })], 100, "CAD");
    expect(s.avg_buy).toBeNull();
    expect(s.buy_pnl).toBeNull();
  });
});
