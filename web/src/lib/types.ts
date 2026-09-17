/** Shapes returned by the FastAPI service (api/main.py, ta_payload.py). */

export type StockRef = { t: string; n: string; ex?: string };
export type HistoryRef = StockRef & { at?: string | null };

export type Num = number | null;

export type Header = {
  date: string;
  close: Num;
  prev_close: Num;
  change_pct: Num;
  total_mv_yi: Num;
  circ_mv_yi: Num;
  /** Raw, in the instrument's own currency. Format by market, not by 亿. */
  market_cap: Num;
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

export type MarketCode = "CN" | "US" | "CA";

export type Analysis = {
  ticker: string;
  name: string;
  market: MarketCode;
  currency: string;
  currency_symbol: string;
  /** Red means UP in Shanghai and DOWN in New York. Never assume. */
  up_is_red: boolean;
  benchmark_name: string;
  sector: string | null;
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

/** GET /sectors/{ticker}?window= — 板块相关性 + 板块轮动. */
export type SectorRow = {
  name: string;
  color: string;
  /** 沪深300, drawn apart from the sectors: it is the market, not a theme. */
  benchmark: boolean;
  /** The stock is a constituent of this sector index. */
  member: boolean;
  /** One of the five highest-affinity sectors — drawn in colour. */
  top: boolean;
  /** Correlation in the most recent window. */
  r: Num;
  /** Mean correlation across the window shown. */
  mean_r: Num;
  series: Num[];
};

export type SectorAnalysis = {
  window: number;
  dates: string[];
  sectors: SectorRow[];
  /** Leading sector per bar, run-length encoded. */
  dominant: { from: number; to: number; sector: string }[];
  summary: {
    top: string;
    top_r: Num;
    top_is_member: boolean;
    /** Heavy constituent tracking its own index — r near 1 is arithmetic. */
    self_index: boolean;
    trend: "strengthening" | "weakening" | "stable";
    r5: Num;
    r20: Num;
    rotations: number;
    verdict: "high" | "moderate" | "low";
    leaders: { sector: string; days: number; pct: number }[];
    n_leaders: number;
    sessions: number;
  };
};

/** GET /compare-stats/{ticker}?with=&window= — 量化对比. */
export type Valuation = {
  pe_start: Num; pe_end: Num;
  pb_start: Num; pb_end: Num;
  mv_yi: Num;
  turnover_avg_pct: Num;
  /** Change in the multiple, and in implied trailing EPS. These COMPOUND. */
  rerating_pct: Num;
  earnings_pct: Num;
  priced_from: string;
};

export type StockProfile = {
  label: string;
  total_return_pct: Num;
  cagr_pct: Num;
  vol_annual_pct: Num;
  sharpe: Num;
  max_drawdown_pct: Num;
  best_day_pct: Num;
  worst_day_pct: Num;
  positive_days_pct: Num;
  bars: number;
  beta?: Num;
  alpha_annual_pct?: Num;
  r2?: Num;
  up_capture_pct?: Num;
  down_capture_pct?: Num;
  valuation: Valuation | null;
};

export type PairStats = {
  window: string;
  bars: number;
  from: string;
  to: string;
  benchmark: { label: string; total_return_pct: Num } | null;
  a: StockProfile;
  b: StockProfile;
  pair: {
    correlation: Num;
    beta_a_on_b: Num;
    r2: Num;
    return_gap_pct: Num;
    tracking_error_pct: Num;
    information_ratio: Num;
    ratio: Num[];
    dates: string[];
    monthly: { month: string; rel_pct: Num }[];
  };
  attribution: {
    market_return_pct: Num;
    gap_pct: Num;
    /** A's growth divided by B's. beta_factor × alpha_factor equals this. */
    gap_ratio: Num;
    beta_factor_pct: Num;
    alpha_factor_pct: Num;
    /** Zero by construction — kept as proof the split is exact. */
    residual_pct: Num;
    beta_a: Num; beta_b: Num;
    alpha_a_pct: Num; alpha_b_pct: Num;
    years: Num;
  } | null;
};

/** GET /alerts — the nightly watchlist scan, reshaped for filtering. */
export type AlertSignal = {
  id: string;
  cn: string;
  en: string;
  group: string;
  dir: "bull" | "bear";
  /** Present on 箱体 signals: the numbers from the bracketed tag. */
  detail: {
    bot: number; top: number; position_pct: number;
    touches_top: number; touches_bot: number;
    quality: number; height_pct: number;
  } | null;
};

export type AlertChips = {
  setup_score: Num;
  setup_label: string | null;
  winner_rate: Num;
  concentration: Num;
  pct_from_peak: Num;
  n_peaks: number | null;
  peak_price: Num;
  converged: boolean | null;
};

export type AlertStock = {
  t: string; n: string;
  bias: string;
  price: Num; rsi: Num; adx: Num; macd: Num; volume: Num;
  signal_count: number;
  signals: AlertSignal[];
  sectors: string[];
  chips: AlertChips | null;
  chip_shape: string | null;
};

export type AlertFeed = {
  scan_date: string;
  age_days: number | null;
  stale: boolean;
  stocks: AlertStock[];
  facets: {
    signals: { id: string; cn: string; group: string; dir: "bull" | "bear"; count: number }[];
    sectors: { name: string; count: number }[];
    bias: { id: string; count: number }[];
    shapes: { name: string; count: number }[];
  };
  groups: string[];
};
