/**
 * What each chart pane draws, built from the API payload.
 *
 * Same panels and the same data as the Streamlit chart (ta_payload.py is
 * parity-tested against it); only the presentation changes. One difference
 * worth knowing: Plotly had one y-axis per panel, so the Streamlit chart
 * squashed OBV and OBV动能 onto the volume axis with a min-max rescale. Here
 * each gets its own overlay price scale and plots its real values, so the
 * legend reads the true number.
 *
 * Colour rule: markets are Chinese-convention (bullish red, bearish green).
 * Line rule, from the user: moving averages are SOLID; dashed is reserved for
 * the EMA.
 */

import type { Analysis, Num, Segment, SeriesKey } from "./types";

export type LineStyleName = "solid" | "dashed" | "dotted";

export type LineSpec = {
  key: string;
  label: string;
  values: Num[];
  color: string;
  width?: 1 | 2 | 3 | 4;
  style?: LineStyleName;
  kind?: "line" | "histogram";
  /** Histogram colouring: by the value's sign, or by the candle body. */
  signBy?: "value" | "body";
  signColors?: [string, string];
  /** Overlay scale id; omitted = the pane's right axis. */
  scaleId?: string;
  scaleMargins?: { top: number; bottom: number };
  decimals?: number;
  /** Hidden until the user turns it on from the legend. */
  defaultHidden?: boolean;
};

export type MarkerSpec = {
  key: string;
  label: string;
  idx: number[];
  position: "aboveBar" | "belowBar";
  shape: "arrowUp" | "arrowDown" | "circle" | "square";
  color: string;
  text?: string;
  /** Line this marker set rides on; main pane markers ride on the candles. */
  anchor?: string;
  defaultHidden?: boolean;
};

export type GuideSpec = { value: number; color?: string; scaleId?: string };

export type BandSpec = { segments: Segment[]; color: string; label?: string };

export type PaneSpec = {
  id: string;
  title: string;
  height: number;
  lines: LineSpec[];
  markers: MarkerSpec[];
  guides: GuideSpec[];
  bands: BandSpec[];
  ribbon?: { i: number; color: string; state: string }[];
};

// Marker colours by MEANING, resolved per market: a bullish marker is red in
// Shanghai and green in New York, the same inversion the candles follow.
const RED = "#dc2626";
const GREEN = "#16a34a";

const bull = (d: Analysis) => (d.up_is_red ? RED : GREEN);
const bear = (d: Analysis) => (d.up_is_red ? GREEN : RED);

// Deeper shades, where a marker needs to read apart from its neighbours.
const RED_DARK = "#b91c1c";
const GREEN_DARK = "#15803d";
const bullDark = (d: Analysis) => (d.up_is_red ? RED_DARK : GREEN_DARK);
const bearDark = (d: Analysis) => (d.up_is_red ? GREEN_DARK : RED_DARK);

// Translucent [bullish, bearish] pair for sign-coloured bars and bands, in
// that same inversion — these take their colours as literals, so written out
// by hand they stay Shanghai-coloured on a New York chart while the candles
// flip, which is exactly what the volume and MACD panes used to do.
const fill = (d: Analysis, a: number): [string, string] => {
  const red = `rgba(220,38,38,${a})`;
  const green = `rgba(22,163,74,${a})`;
  return d.up_is_red ? [red, green] : [green, red];
};

/** True when at least one bar of this series carries a value. */
function hasData(d: Analysis, key: SeriesKey): boolean {
  const v = d.series[key];
  return Array.isArray(v) && v.some((x) => x != null && Number.isFinite(x));
}

const s = (d: Analysis, k: SeriesKey) => d.series[k];

export function buildPanes(d: Analysis): { main: PaneSpec; subs: PaneSpec[] } {
  const main: PaneSpec = {
    id: "main",
    title: "K线",
    height: 440,
    lines: [
      { key: "MA5", label: "MA5", values: s(d, "MA5"), color: "#f59e0b", decimals: 2 },
      { key: "MA10", label: "MA10", values: s(d, "MA10"), color: "#2563eb", decimals: 2 },
      { key: "MA20", label: "MA20", values: s(d, "MA20"), color: "#7c3aed", decimals: 2 },
      { key: "MA60", label: "MA60", values: s(d, "MA60"), color: "#0d9488", decimals: 2 },
      { key: "MA50", label: "MA50", values: s(d, "MA50"), color: "#64748b", decimals: 2, defaultHidden: true },
      { key: "MA200", label: "MA200", values: s(d, "MA200"), color: "#111827", decimals: 2, defaultHidden: true },
      { key: "EMA5", label: "EMA5", values: s(d, "EMA5"), color: "#db2777", style: "dashed", decimals: 2 },
      { key: "BB_Upper", label: "BB上", values: s(d, "BB_Upper"), color: "#94a3b8", decimals: 2 },
      { key: "BB_Lower", label: "BB下", values: s(d, "BB_Lower"), color: "#94a3b8", decimals: 2 },
    ],
    markers: [
      { key: "m_acc", label: "吸筹", idx: d.markers.price.accumulation, position: "belowBar", shape: "circle", color: "#ca8a04" },
      { key: "m_sqz", label: "挤压", idx: d.markers.price.squeeze, position: "aboveBar", shape: "square", color: "#64748b", defaultHidden: true },
      { key: "m_sqz_bull", label: "挤压突破", idx: d.markers.price.squeeze_bull, position: "belowBar", shape: "arrowUp", color: bull(d), text: "突破" },
      { key: "m_sqz_bear", label: "挤压下破", idx: d.markers.price.squeeze_bear, position: "aboveBar", shape: "arrowDown", color: bear(d), text: "下破" },
      { key: "m_dt_rev", label: "下跌反转", idx: d.markers.price.downtrend_reversal, position: "belowBar", shape: "arrowUp", color: bullDark(d) },
      { key: "m_ut_rev", label: "上涨反转", idx: d.markers.price.uptrend_reversal, position: "aboveBar", shape: "arrowDown", color: bearDark(d) },
      { key: "m_exit", label: "MACD离场", idx: d.markers.price.exit_macd, position: "aboveBar", shape: "arrowDown", color: "#ea580c" },
      { key: "m_sbuy", label: "强买", idx: d.markers.price.screaming_buy, position: "belowBar", shape: "arrowUp", color: bull(d), text: "★买" },
      { key: "m_ssell", label: "强卖", idx: d.markers.price.screaming_sell, position: "aboveBar", shape: "arrowDown", color: bear(d), text: "★卖" },
    ],
    guides: [],
    bands: d.bands.regime.map((r) => ({ segments: [r], color: r.color })),
  };

  const subs: PaneSpec[] = [
    {
      id: "volume", title: "成交量 / OBV", height: 110,
      lines: [
        { key: "Volume", label: "量", values: d.ohlcv.v, color: "#94a3b8", kind: "histogram",
          signBy: "body", signColors: fill(d, 0.6), decimals: 0 },
        { key: "Vol_Scaled_OBV", label: "OBV", values: s(d, "Vol_Scaled_OBV"), color: "#f59e0b",
          width: 2, scaleId: "obv", scaleMargins: { top: 0.1, bottom: 0.1 }, decimals: 2 },
        { key: "OBV_Mom", label: "OBV动能20", values: s(d, "OBV_Mom"), color: "#2563eb",
          scaleId: "obvmom", scaleMargins: { top: 0.1, bottom: 0.1 }, decimals: 2 },
      ],
      markers: [], bands: [],
      guides: [{ value: 0, color: "rgba(37,99,235,0.45)", scaleId: "obvmom" }],
    },
    {
      id: "macd", title: "MACD", height: 140,
      lines: [
        { key: "MACD_Hist", label: "柱×2.5", values: s(d, "MACD_Hist"), color: "#94a3b8", kind: "histogram",
          signBy: "value", signColors: fill(d, 0.55), decimals: 3 },
        { key: "MACD", label: "MACD", values: s(d, "MACD"), color: "#2563eb", decimals: 3 },
        { key: "MACD_Signal", label: "信号", values: s(d, "MACD_Signal"), color: "#f59e0b", decimals: 3 },
      ],
      markers: [
        { key: "mm_trig", label: "触发", idx: d.markers.macd.trigger, position: "belowBar", shape: "arrowUp", color: bull(d), anchor: "MACD" },
        { key: "mm_peak", label: "见顶", idx: d.markers.macd.peaking, position: "aboveBar", shape: "arrowDown", color: bear(d), anchor: "MACD" },
        { key: "mm_bear", label: "死叉", idx: d.markers.macd.bearish_cross, position: "aboveBar", shape: "circle", color: bear(d), anchor: "MACD" },
      ],
      guides: [{ value: 0 }],
      bands: [
        { segments: d.bands.macd_uptrend, color: fill(d, 0.08)[0], label: "大级别上涨" },
        { segments: d.bands.macd_downtrend, color: fill(d, 0.08)[1], label: "大级别下跌" },
      ],
    },
    {
      id: "rsi", title: "RSI(14)", height: 110,
      lines: [
        { key: "RSI", label: "RSI", values: s(d, "RSI"), color: "#7c3aed", width: 2, decimals: 1 },
        { key: "RSI_P90", label: "P90", values: s(d, "RSI_P90"), color: "#ef4444", style: "dotted", decimals: 1 },
        { key: "RSI_P10", label: "P10", values: s(d, "RSI_P10"), color: "#2563eb", style: "dotted", decimals: 1 },
      ],
      markers: [
        { key: "mr_bot", label: "RSI底", idx: d.markers.rsi.bottoming, position: "belowBar", shape: "arrowUp", color: bull(d), anchor: "RSI" },
        { key: "mr_top", label: "RSI顶", idx: d.markers.rsi.peaking, position: "aboveBar", shape: "arrowDown", color: bear(d), anchor: "RSI" },
      ],
      guides: [{ value: 70, color: "#fca5a5" }, { value: 30, color: "#93c5fd" }],
      bands: [],
    },
    {
      id: "adx", title: "ADX / DMI", height: 140,
      lines: [
        { key: "ADX", label: "ADX", values: s(d, "ADX"), color: "#111827", decimals: 1 },
        { key: "ADX_LOWESS", label: "平滑", values: s(d, "ADX_LOWESS"), color: "#f59e0b", width: 2, decimals: 1 },
        { key: "ADX_BB_Upper", label: "带上", values: s(d, "ADX_BB_Upper"), color: "#cbd5e1", decimals: 1, defaultHidden: true },
        { key: "ADX_BB_Lower", label: "带下", values: s(d, "ADX_BB_Lower"), color: "#cbd5e1", decimals: 1, defaultHidden: true },
        { key: "DI_Plus", label: "+DI", values: s(d, "DI_Plus"), color: bull(d), decimals: 1 },
        { key: "DI_Minus", label: "−DI", values: s(d, "DI_Minus"), color: bear(d), decimals: 1 },
      ],
      markers: [
        { key: "ma_sbuy", label: "DI强买", idx: d.markers.adx.di_screaming_buy, position: "belowBar", shape: "arrowUp", color: bull(d), text: "★", anchor: "ADX" },
        { key: "ma_ssell", label: "DI强卖", idx: d.markers.adx.di_screaming_sell, position: "aboveBar", shape: "arrowDown", color: bear(d), text: "★", anchor: "ADX" },
        { key: "ma_bot", label: "筑底", idx: d.markers.adx.bottoming, position: "belowBar", shape: "circle", color: bull(d), anchor: "ADX" },
        { key: "ma_rup", label: "转强", idx: d.markers.adx.reversing_up, position: "belowBar", shape: "arrowUp", color: "#f97316", anchor: "ADX" },
        { key: "ma_peak", label: "见顶", idx: d.markers.adx.peaking, position: "aboveBar", shape: "circle", color: bear(d), anchor: "ADX" },
        { key: "ma_rdn", label: "转弱", idx: d.markers.adx.reversing_down, position: "aboveBar", shape: "arrowDown", color: bear(d), anchor: "ADX" },
      ],
      guides: [{ value: 25, color: "#e5e7eb" }],
      bands: [],
      ribbon: d.adx_ribbon,
    },
    {
      id: "z", title: "Z-Score (20日)", height: 100,
      lines: [
        // Bars first so the price-Z line draws over them. Coloured by the
        // candle body, like the volume pane: the sign of 量Z only says heavy
        // or thin, so red/green here has to mean the day's direction.
        { key: "Volume_Z", label: "量Z", values: s(d, "Volume_Z"), color: "#94a3b8", kind: "histogram",
          signBy: "body", signColors: fill(d, 0.45), decimals: 2 },
        { key: "Price_Z", label: "价格Z", values: s(d, "Price_Z"), color: "#7c3aed", width: 2, decimals: 2 },
      ],
      markers: [
        { key: "mz_os", label: "超卖≤−2.5", idx: d.markers.z.oversold, position: "belowBar", shape: "arrowUp", color: bull(d), anchor: "Price_Z" },
        { key: "mz_ob", label: "超买≥+2", idx: d.markers.z.overbought, position: "aboveBar", shape: "arrowDown", color: bear(d), anchor: "Price_Z" },
      ],
      guides: [{ value: 2, color: "#fca5a5" }, { value: 0 }, { value: -2.5, color: "#93c5fd" }],
      bands: [],
    },
  ];

  // A pane whose every point is whitespace has NO data range, and the panes
  // sync their visible range to each other — so one empty pane drags all of
  // them to a degenerate range, the chart opens on a single bar near the start
  // of history and no amount of zooming or resetting escapes it. That is what
  // the P/E pane did on US stocks, where PE_TTM is null for every bar.
  //
  // Gated on having data rather than on the market, so a Chinese stock with no
  // PE history (a loss-maker) is covered by the same rule.
  if (hasData(d, "PE_TTM")) {
    subs.push({
      id: "pe", title: "P/E (TTM)", height: 80,
      lines: [{ key: "PE_TTM", label: "PE", values: s(d, "PE_TTM"), color: "#334155", decimals: 2 }],
      markers: [], guides: [], bands: [],
    });
  }

  if (d.has_moneyflow) {
    subs.push({
      id: "mf", title: "主力净流入 (万元)", height: 110,
      lines: [
        { key: "MF_Daily", label: "每日", values: s(d, "MF_Daily"), color: "#94a3b8", kind: "histogram",
          signBy: "value", signColors: fill(d, 0.7), decimals: 0 },
        { key: "MF_Rolling", label: "近20日", values: s(d, "MF_Rolling"), color: "#2563eb", width: 2,
          scaleId: "mfroll", scaleMargins: { top: 0.1, bottom: 0.1 }, decimals: 0 },
      ],
      markers: [], bands: [],
      guides: [{ value: 0, color: "rgba(37,99,235,0.4)", scaleId: "mfroll" }],
    });
  }

  return { main, subs };
}

/** Everything hidden by default, as `${pane}:${key}` ids. */
export function defaultHidden(main: PaneSpec, subs: PaneSpec[]): string[] {
  const out: string[] = [];
  for (const p of [main, ...subs]) {
    for (const l of p.lines) if (l.defaultHidden) out.push(`${p.id}:${l.key}`);
    for (const m of p.markers) if (m.defaultHidden) out.push(`${p.id}:${m.key}`);
  }
  return out;
}
