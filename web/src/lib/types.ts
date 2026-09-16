/** Shapes returned by the FastAPI service (api/main.py, ta_payload.py). */

export type StockRef = { t: string; n: string };
export type HistoryRef = StockRef & { at?: string | null };

export type Num = number | null;

export type Header = {
  date: string;
  close: Num;
  prev_close: Num;
  change_pct: Num;
  total_mv_yi: Num;
  circ_mv_yi: Num;
  pe_ttm: Num;
  pb: Num;
  turnover_rate: Num;
};

export type ActiveBox = {
  kind: "BOX" | "RISING_CHANNEL" | "FALLING_CHANNEL";
  status: string;
  status_cn: string;
  top?: Num;
  bot?: Num;
  height_pct?: Num;
  position?: Num;
  touches_top?: number;
  touches_bot?: number;
  sessions?: number;
  quality?: Num;
  drift_pct?: Num;
  r2?: Num;
};

export type Signals = {
  squeeze: boolean;
  accumulation: boolean;
  bull: string[];
  bear: string[];
  box: ActiveBox | null;
  regime: string;
  adx: Num;
  adx_pattern: string;
};

export type ChartBox = {
  from: number;
  to: number;
  kind: string;
  status_cn: string;
  is_active: boolean;
  top?: Num;
  bot?: Num;
  zone?: Num;
  touches_top?: number;
  touches_bot?: number;
  height_pct?: Num;
  sessions?: number;
  quality?: Num;
  drift_pct?: Num;
  r2?: Num;
};

export type Segment = { from: number; to: number };
export type RegimeSegment = Segment & { regime: string; color: string };

export type SeriesKey =
  | "MA5" | "MA10" | "MA20" | "MA50" | "MA60" | "MA200" | "EMA5"
  | "BB_Upper" | "BB_Lower"
  | "Vol_Scaled_OBV" | "OBV_Mom"
  | "MACD" | "MACD_Signal" | "MACD_Hist"
  | "RSI" | "RSI_P10" | "RSI_P90"
  | "ADX" | "ADX_LOWESS" | "ADX_BB_Upper" | "ADX_BB_Lower" | "DI_Plus" | "DI_Minus"
  | "Price_Z" | "Volume_Z"
  | "PE_TTM"
  | "MF_Daily" | "MF_Rolling";

export type Chips = {
  /** Price of each histogram bin, low → high. */
  prices: number[];
  /** Percent of the float held at that price. Same length as `prices`. */
  weights: number[];
  winner_rate: Num;
  trapped_rate: Num;
  weight_avg: Num;
  concentration: Num;
  cost_5pct: Num;
  cost_15pct: Num;
  cost_50pct: Num;
  cost_85pct: Num;
  cost_95pct: Num;
  peak_price: Num;
  n_peaks: number;
  peaks: { price: Num; share: Num }[];
  setup_score: Num;
  setup_label: string | null;
  converged: boolean;
  seed_remaining: Num;
  cum_turnover_pct: Num;
  sessions: number;
  decay: number;
};

export type Analysis = {
  ticker: string;
  name: string;
  header: Header;
  signals: Signals;
  boxes: ChartBox[];
  dates: string[];
  ohlcv: { o: Num[]; h: Num[]; l: Num[]; c: Num[]; v: Num[] };
  series: Record<SeriesKey, Num[]>;
  markers: {
    price: Record<
      | "accumulation" | "squeeze" | "squeeze_bull" | "squeeze_bear"
      | "downtrend_reversal" | "uptrend_reversal" | "exit_macd"
      | "screaming_buy" | "screaming_sell",
      number[]
    >;
    macd: Record<"trigger" | "peaking" | "bearish_cross", number[]>;
    rsi: Record<"bottoming" | "peaking", number[]>;
    adx: Record<
      | "di_screaming_buy" | "di_screaming_sell"
      | "bottoming" | "reversing_up" | "peaking" | "reversing_down",
      number[]
    >;
    z: Record<"oversold" | "overbought", number[]>;
  };
  adx_ribbon: { i: number; state: string; color: string }[];
  bands: {
    regime: RegimeSegment[];
    macd_uptrend: Segment[];
    macd_downtrend: Segment[];
  };
  chips: Chips | null;
  has_moneyflow: boolean;
  initial_visible: number;
};

/** POST /simulate/{ticker} — the hypothetical next bar and its indicators. */
export type SimResult = {
  date: string;
  ohlcv: { o: Num; h: Num; l: Num; c: Num; v: Num };
  ohl_supplied: boolean;
  series: Partial<Record<SeriesKey, Num>>;
  signals: Record<string, boolean | string>;
  adx_pattern: string | null;
  conditions_met: number;
};

/** GET /compare/{ticker}?with= — a second stock, in both scalings. */
export type CompareResult = {
  ticker: string;
  name: string;
  price: Num[];
  rebased: Num[];
  change_pct: Num;
};

/** POST /whatif-ai/{ticker} — 尾盘推演. */
export type WhatIfAi = {
  mode: "ghost" | "actual";
  bar_date: string;
  crossings: { what: string; dir: "up" | "down"; detail: string }[];
  read: {
    headline?: string;
    bar_read?: string;
    key_changes?: string[];
    bull_case?: string;
    bear_case?: string;
    stance?: { call?: string; conviction?: string; why?: string; if_holding?: string; if_flat?: string };
    levels?: { confirm?: string; invalidate?: string; note?: string };
    next_session_plan?: string;
    what_would_change_my_mind?: string[];
    caveats?: string[];
  };
};
