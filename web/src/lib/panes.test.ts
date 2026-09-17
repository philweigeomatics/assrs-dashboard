/**
 * buildPanes: no pane may be built with nothing in it.
 *
 * This guards a bug whose symptom was nowhere near its cause. The panes sync
 * their visible logical range to each other, and a pane whose every point is
 * whitespace has no data range at all — so one empty pane dragged every other
 * pane to a degenerate range. The chart opened on a single bar near the start
 * of history, the crosshair reported the same date at every x position, and
 * neither 框选放大 nor 重置视图 could escape it, because the empty pane kept
 * pulling the others back.
 *
 * It only showed up on US stocks, where PE_TTM is null for every bar, while
 * the moneyflow pane next to it was already gated on has_moneyflow.
 */

import { describe, expect, it } from "vitest";
import { buildPanes } from "./panes";
import type { Analysis, Num, SeriesKey } from "./types";

const KEYS: SeriesKey[] = [
  "MA5", "MA10", "MA20", "MA50", "MA60", "MA200", "EMA5",
  "BB_Upper", "BB_Lower", "Vol_Scaled_OBV", "OBV_Mom",
  "MACD", "MACD_Signal", "MACD_Hist", "RSI", "RSI_P10", "RSI_P90",
  "ADX", "ADX_LOWESS", "ADX_BB_Upper", "ADX_BB_Lower", "DI_Plus", "DI_Minus",
  "Price_Z", "Volume_Z", "PE_TTM", "MF_Daily", "MF_Rolling",
];

const N = 120;

function analysis(over: Partial<Analysis> = {}, nulls: SeriesKey[] = []): Analysis {
  const dates = Array.from({ length: N }, (_, i) =>
    new Date(Date.UTC(2025, 0, 1 + i)).toISOString().slice(0, 10));
  const ramp = (): Num[] => Array.from({ length: N }, (_, i) => i + 1);
  const empty = (): Num[] => Array.from({ length: N }, () => null);

  const series = Object.fromEntries(
    KEYS.map((k) => [k, nulls.includes(k) ? empty() : ramp()]),
  ) as Record<SeriesKey, Num[]>;

  const noMarks = { price: {}, macd: {}, rsi: {}, adx: {}, z: {} } as Analysis["markers"];
  for (const k of ["accumulation", "squeeze", "squeeze_bull", "squeeze_bear",
    "downtrend_reversal", "uptrend_reversal", "exit_macd",
    "screaming_buy", "screaming_sell"]) (noMarks.price as never as Record<string, number[]>)[k] = [];
  for (const k of ["trigger", "peaking", "bearish_cross"]) (noMarks.macd as never as Record<string, number[]>)[k] = [];
  for (const k of ["bottoming", "peaking"]) (noMarks.rsi as never as Record<string, number[]>)[k] = [];
  for (const k of ["di_screaming_buy", "di_screaming_sell", "bottoming",
    "reversing_up", "peaking", "reversing_down"]) (noMarks.adx as never as Record<string, number[]>)[k] = [];
  for (const k of ["oversold", "overbought"]) (noMarks.z as never as Record<string, number[]>)[k] = [];

  return {
    ticker: "US:TEST", name: "Test", market: "US", currency: "USD",
    currency_symbol: "$", up_is_red: false, benchmark_name: "S&P 500", sector: null,
    header: {
      date: dates[N - 1]!, close: 10, prev_close: 9, change_pct: 1,
      total_mv_yi: null, circ_mv_yi: null, market_cap: 1e12,
      pe_ttm: null, pb: null, turnover_rate: null,
    },
    signals: {
      squeeze: false, accumulation: false, bull: [], bear: [], box: null,
      regime: "Normal", adx: 20, adx_pattern: "",
    },
    boxes: [], dates,
    ohlcv: { o: ramp(), h: ramp(), l: ramp(), c: ramp(), v: ramp() },
    series,
    markers: noMarks,
    adx_ribbon: [],
    bands: { regime: [], macd_uptrend: [], macd_downtrend: [] },
    chips: null, has_moneyflow: false, initial_visible: 60,
    ...over,
  };
}

/** Every value a pane would draw, across all of its line series. */
function paneValues(pane: { lines: { values: Num[] }[] }): Num[] {
  return pane.lines.flatMap((l) => l.values);
}

describe("buildPanes", () => {
  it("never builds a pane with no data in any of its series", () => {
    // The real shape of a US stock: no PE history, no moneyflow.
    const d = analysis({}, ["PE_TTM", "MF_Daily", "MF_Rolling"]);
    const { main, subs } = buildPanes(d);

    for (const pane of [main, ...subs]) {
      const hasSomething = paneValues(pane).some((v) => v != null && Number.isFinite(v));
      expect(hasSomething, `pane "${pane.id}" has no data and must not be built`).toBe(true);
    }
  });

  it("drops the P/E pane when PE_TTM is empty", () => {
    const withPe = buildPanes(analysis());
    const without = buildPanes(analysis({}, ["PE_TTM"]));

    expect(withPe.subs.map((p) => p.id)).toContain("pe");
    expect(without.subs.map((p) => p.id)).not.toContain("pe");
  });

  it("keeps the P/E pane for a stock that has one", () => {
    const { subs } = buildPanes(analysis({ market: "CN", up_is_red: true }));
    const pe = subs.find((p) => p.id === "pe");
    expect(pe).toBeDefined();
    expect(paneValues(pe!).some((v) => v != null)).toBe(true);
  });

  it("gates the moneyflow pane on has_moneyflow, as it already did", () => {
    expect(buildPanes(analysis({ has_moneyflow: false })).subs.map((p) => p.id))
      .not.toContain("mf");
    expect(buildPanes(analysis({ has_moneyflow: true })).subs.map((p) => p.id))
      .toContain("mf");
  });
});

describe("marker colours follow the market", () => {
  const colourOf = (d: Analysis, key: string) => {
    const { main } = buildPanes(d);
    return main.markers.find((m) => m.key === key)?.color;
  };

  it("paints a bullish marker red for A-shares and green for North America", () => {
    const cn = analysis({ market: "CN", up_is_red: true });
    const us = analysis({ market: "US", up_is_red: false });

    // 强买 is a buy signal: red in Shanghai, green in New York.
    expect(colourOf(cn, "m_sbuy")).not.toBe(colourOf(us, "m_sbuy"));
    expect(colourOf(cn, "m_sbuy")).toBe(colourOf(us, "m_ssell"));
    expect(colourOf(cn, "m_ssell")).toBe(colourOf(us, "m_sbuy"));
  });

  it("keeps bullish and bearish markers distinguishable in both markets", () => {
    for (const upIsRed of [true, false]) {
      const d = analysis({ up_is_red: upIsRed });
      expect(colourOf(d, "m_sbuy")).not.toBe(colourOf(d, "m_ssell"));
    }
  });
});
