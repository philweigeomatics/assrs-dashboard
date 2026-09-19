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
  /** Served from the cache rather than freshly generated. */
  cached?: boolean;
  /** When the cached read was made. Absent on a fresh one. */
  generated_at?: string | null;
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
  market: MarketCode | null;
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

/** GET /basket?symbols=&window= — 多股对比. */
export type BasketStock = StockProfile & {
  symbol: string;
  /** Mean correlation to the OTHER members, never to itself. */
  corr_to_peers: Num;
  /** Growth relative to the equal-weight basket: 1.2 = 20% ahead of it. */
  vs_basket: Num;
  rank_return: number;
  rank_alpha?: number;
};

export type BasketVerdict = {
  symbol: string | null;
  kind: "warn" | "weak" | "ok" | "mixed";
  text: string;
};

export type BasketStats = {
  window: string;
  market: MarketCode | null;
  bars: number;
  from: string;
  to: string;
  benchmark: { label: string; total_return_pct: Num } | null;
  basket: {
    total_return_pct: Num;
    avg_correlation: Num;
    /** False when "they move together" is not true of this group. */
    cohesive: boolean;
    spread_pct: Num;
  };
  stocks: BasketStock[];
  symbols: string[];
  correlation: Num[][];
  verdicts: BasketVerdict[];
};

/** GET/POST /strategies/{name} — the watchlist screens. */
export type StrategyRow = {
  ticker: string;
  name: string;
  verdict?: string;
  verdict_cn: string;
  rank?: number;
  why?: string;
  /** 做T */
  score?: Num;
  range_pct?: Num;
  turnover_pct?: Num;
  meanrev_bias?: Num;
  adx?: Num;
  range_pos?: Num;
  limit_event?: boolean;
  parts?: Record<string, Num>;
  /** 反转 */
  rules?: Record<string, boolean | null>;
  passed?: number;
  z?: Num;
  rsi?: Num;
  down_days?: Num;
  vol_exhausted?: boolean;
  limit_down_streak?: number;
  vs_sector_pp?: Num;
  ret_5d_pct?: Num;
};

export type StrategyResult = {
  scanned_at: string | null;
  age_hours: Num;
  stale: boolean;
  count: number;
  counts: Record<string, number>;
  rows: StrategyRow[];
};

/** POST /strategies/pair-trade — 配对交易. */
/** How the two legs actually came back together — see pair_trade.leg_pattern. */
export type LegPattern = "BOTH_UP" | "BOTH_DOWN" | "A_UP_B_DOWN" | "B_UP_A_DOWN" | "FLAT";

export type PairTrade = {
  entry: string; exit: string;
  entry_z: number; exit_z: number;
  direction: "BUY_A" | "BUY_B";
  open: boolean;
  buy_code: string; buy_name: string;
  entry_price: Num; exit_price: Num; pnl_pct: Num;
  /** Each leg over the window. pnl_pct is whichever of these we bought. */
  a_ret_pct: Num; b_ret_pct: Num;
  pattern: LegPattern | null;
};

export type PairResult = {
  code_a: string; code_b: string; name_a: string; name_b: string;
  eg_p: number; adf_p: number; hurst: number; corr: number;
  half_life: number; beta_now: number; z_now: number; score: number;
  coint_ok: boolean; adf_ok: boolean; hurst_ok: boolean; hl_ok: boolean;
  signal: "BUY_A" | "BUY_B" | "WATCH" | "NEUTRAL";
  signal_cn: string; buy: string; reduce: string;
  buy_name: string; reduce_name: string;
  /** Shared x-axis: dates, the z-score, and both legs' closes are aligned. */
  dates: string[]; z_series: Num[]; px_a: number[]; px_b: number[];
  trades: PairTrade[];
  closed: number; win_rate: Num; avg_pnl_pct: Num;
};

/**
 * The thresholds the engine actually applied, shipped from the module that
 * applies them (pair_trade.GATES) so the on-screen help cannot drift from
 * the rules it describes.
 */
export type PairGates = {
  coint_p: number; adf_p: number;
  hurst_max: number; hl_min: number; hl_max: number;
  entry_z: number; watch_z: number;
  good_score: number; max_score: number;
  z_window: number; ols_window: number;
};

export type PairTradeResult = {
  from: string; to: string; bars: number;
  z_window: number; ols_window: number;
  gates: PairGates;
  codes: StockRef[];
  pairs: PairResult[];
  skipped: { code_a: string; code_b: string;
             name_a: string; name_b: string; why: string }[];
};

/** GET /equity/{ticker} — 个股研报. */
export type EquityPeriod = {
  period: string;
  revenue?: Num; operate_profit?: Num; net_profit?: Num;
  roe?: Num; roa?: Num; gross_margin?: Num; net_margin?: Num;
  debt_to_assets?: Num; current_ratio?: Num;
  revenue_yoy?: Num; profit_yoy?: Num; eps_yoy?: Num; ocf_to_revenue?: Num;
};

export type SegmentItem = {
  item: string; sales: Num; profit: Num; cost: Num;
  share_pct: Num; margin_pct: Num;
};

export type PeerRow = {
  ticker: string; name: string; is_target: boolean; why: string;
  pe_ttm: Num; pb: Num; mv_yi: Num; roe: Num;
  gross_margin: Num; net_margin: Num;
  revenue_yoy: Num; profit_yoy: Num; debt_to_assets: Num;
  period: string | null;
  ranks: Record<string, number>;
};

export type EquityBrief = {
  ticker: string; name: string; industry: string;
  fundamentals: { periods: EquityPeriod[] };
  filings: {
    forecast: Record<string, string | number | null>[];
    express: Record<string, string | number | null>[];
  };
  segments: {
    product: { period: string | null; items: SegmentItem[]; periods: string[] };
    region: { period: string | null; items: SegmentItem[]; periods: string[] };
  };
  /** products → macro sectors. Not nodes/edges. */
  supply_chain: {
    company_name?: string;
    products?: string[];
    macro_sectors?: string[];
    links?: { source: string; target: string }[];
  } | null;
  ai: {
    overview: Record<string, unknown> | null;
    porters: Record<string, unknown> | null;
    pestel: Record<string, unknown> | null;
    competitors: { competitors?: { ticker: string; name: string; why: string }[] } | null;
  };
  generated_at: Record<string, string | null>;
  missing: string[];
  peers: { rows: PeerRow[]; n: number } | null;
};

/** GET/POST /alerts/notes — 明日预判. */
export type NoteClaim = {
  kind: string;
  label?: string;
  value?: number;
  value2?: number;
  lookback?: number;
  /** Filled in at resolution. null = nothing could decide it. */
  hit?: boolean | null;
  actual?: string;
  /** How often this claim would have been true anyway, in percent. */
  baseline?: number | null;
};

export type AlertNote = {
  id: number;
  ticker: string;
  scan_date: string;
  note: string;
  predictions: NoteClaim[];
  created_at: string;
  resolved_date: string | null;
  outcome: {
    bar: { date: string; open: number; high: number; low: number;
           close: number; volume: number; prev_close: number | null };
    claims: NoteClaim[];
    hits: number;
    decided: number;
    score_pct: Num;
    resolved?: string;
  } | null;
};

export type NoteScorecard = {
  rows: { kind: string; label: string; n: number; hits: number;
          rate_pct: number; baseline_pct: Num; edge_pp: Num }[];
  total: number;
  hits: number;
  rate_pct: Num;
  /** Below this many resolved claims the rates are noise. */
  meaningful_at: number;
};

// ── 市场看板 ────────────────────────────────────────────────────────────────

export type HeatStock = { t: string; n: string; mcap: number; pct: number };
export type HeatSector = { name: string; mcap: number; pct: number; stocks: HeatStock[] };
export type Heatmap = {
  trade_date: string;
  mcap: number;
  pct: number;
  /** False when the move fetch failed — every box grey for a reason. */
  has_moves: boolean;
  sectors: HeatSector[];
};

/**
 * The market_breadth table. NOT a member count, whatever the name says: each
 * cell is the sector index's own distance from its MA20, mapped from ±5% onto
 * 0–1 and clipped. See sector_rotation.load_trend.
 */
export type Breadth = {
  dates: string[];
  sectors: { name: string; latest: Num; values: Num[] }[];
  /** Sectors currently above their own 20-day mean. */
  hot: number;
  total: number;
};

export type Leverage = {
  market: string;
  label: string;
  ok: boolean;
  unit: string;
  freq: string;
  latest: Num;
  prev: Num;
  asof: string | null;
  note: string | null;
  error: string | null;
  series: { period: string; value: Num }[];
  detail: { period: string; rzye?: Num; rqye?: Num;
            rzmre?: Num; rzche?: Num; net_fin?: Num }[];
};

export type TopList = {
  trade_date: string;
  rows: { t: string; n: string; close: Num; pct: Num;
          net: Num; net_rate: Num; reason: string }[];
};

export type WyckoffPhase = "accumulation" | "markup" | "distribution" | "markdown" | "transition";

export type Wyckoff = {
  name: string;
  phase: WyckoffPhase;
  phases: Record<string, { label: string; en: string; color: string; means: string }>;
  asof: string;
  since: string;
  days_in_phase: number;
  position_pct: number;
  volume_z: Num;
  volatility: number;
  vol_baseline: number;
  lookback: number;
  /** Bars a new phase must hold before it is reported. */
  confirm: number;
  dates: string[];
  bars: { o: number; h: number; l: number; c: number }[];
  channel: { high: number[]; low: number[] };
  spans: { from: number; to: number; phase: WyckoffPhase }[];
  edge: {
    horizon: number;
    baseline_pct: number;
    meaningful_at: number;
    overlapping: boolean;
    rows: { phase: WyckoffPhase; n: number; mean_pct: Num;
            edge_pp: Num; win_pct: Num; thin: boolean }[];
  };
};

export type Quadrant = "improving" | "leading" | "weakening" | "lagging";
export type RotationTrend = { now_pct: number; delta_pp: number };
export type RotationPoint = { date: string; ratio: number; mom: number };

export type RotationSector = {
  name: string;
  ratio: number;
  mom: number;
  quadrant: Quadrant;
  /** Sitting on the crossing point — the label is a coin flip. */
  neutral: boolean;
  distance: number;
  /** Compass bearing of the last leg, clockwise from north. */
  heading: Num;
  tail: RotationPoint[];
  /** Absolute trend: where the sector index sits against its own MA20. */
  trend: RotationTrend | null;
};

export type RotationCall = {
  name: string;
  ratio: number;
  mom: number;
  heading: Num;
  trend: RotationTrend | null;
  /** Whether the absolute trend moved the way the relative signal claims. */
  rising: boolean | null;
};

export type Rotation = {
  freq: "w" | "d";
  benchmark: string;
  asof: string;
  bars: number;
  dates: string[];
  params: { rs_window: number; mom_window: number; tail: number; horizon: number };
  quadrants: Record<Quadrant, { label: string; en: string; color: string; means: string }>;
  sectors: RotationSector[];
  calls: { into: RotationCall[]; outof: RotationCall[];
           leading: string[]; lagging: string[] };
  edge: {
    horizon: number;
    baseline_pct: number;
    rows: { quadrant: Quadrant; n: number; mean_pct: Num;
            edge_pp: Num; win_pct: Num }[];
  };
};

// ── MyQuestrade ─────────────────────────────────────────────────────────────

export type QtStatus = {
  connected: boolean;
  api_server?: string | null;
  connected_at?: string | null;
  access_expires_at?: string | null;
  /** Why the last refresh failed, if it did. Shown verbatim. */
  reason?: string | null;
  accounts?: QtAccountRef[];
};

export type QtAccountRef = {
  id: string; type: string; label: string;
  status: string; primary: boolean; client_type: string;
};

export type QtAccount = QtAccountRef & {
  positions: number;
  market_value_base: number;
  cash_base: number;
  per_currency: { currency: string; cash: number;
                  market_value: Num; total_equity: Num }[];
};

export type QtLot = {
  id: string; label: string; type: string;
  quantity: number;
  /** Per account, because ACB is per account — and a TFSA has none for tax. */
  avg_cost: Num;
  market_value: Num;
  market_value_base: Num;
  open_pnl: Num;
};

export type QtHolding = {
  symbol: string; name: string; kind: string; currency: string; exchange: string;
  /** Yahoo's spelling, or null when it could not be mapped. */
  yahoo: string | null;
  /** "stock" | "etf" | "other" — what the client filters and groups on. */
  group: string;
  quantity: number;
  avg_cost: Num;
  price: Num;
  market_value: number;
  market_value_base: number;
  cost: number;
  open_pnl: number;
  open_pnl_pct: Num;
  weight_pct: Num;
  /** Held in more than one account. */
  split: boolean;
  accounts: QtLot[];
};

export type QtBook = {
  as_of: string;
  base: string;
  delayed: boolean;
  fx: { rates: Record<string, number>; source: string };
  accounts: QtAccount[];
  holdings: QtHolding[];
  totals: {
    market_value: number; cash: number; equity: number; cost: number;
    open_pnl: number; open_pnl_pct: Num; positions: number; accounts: number;
    /** Residual rows too small to show, dropped but counted. */
    dust: number;
  };
  mix: { currency: { name: string; value: number; pct: Num }[];
         kind: { name: string; value: number; pct: Num }[] };
  warnings: string[];
};

export type QtStats = {
  ann_return_pct: number;
  ann_vol_pct: number;
  downside_vol_pct: Num;
  sharpe: Num;
  sortino: Num;
  max_drawdown_pct: number;
  beta: Num;
  alpha_pct: Num;
  r2: Num;
  tracking_error_pct: Num;
  info_ratio: Num;
  var95_pct: number;
  worst_day_pct: number;
  up_capture_pct: Num;
  down_capture_pct: Num;
};

export type QtRisk = QtStats & {
  base: string;
  as_of: string;
  scope: QtScope;
  scope_label: string;
  /** This sleeve's share of the whole book — context covered_pct cannot give. */
  sleeve_pct: number;
  /** Holdings left out of the statistics, and why. */
  excluded: { symbol: string; reason: string }[];
  benchmark: string;
  benchmark_name: string;
  benchmark_stats: QtStats;
  sessions: number;
  from: string;
  to: string;
  /** Share of the book these statistics actually speak for. */
  covered_pct: number;
  /** Says out loud that this is today's weights replayed, not a track record. */
  basis: string;
  holdings: {
    symbol: string; name: string; weight_pct: number; ann_vol_pct: number;
    beta: Num; corr_bench: number; risk_pct: Num; ann_return_pct: number;
  }[];
  concentration: {
    positions: number; effective_n: Num; hhi: number;
    top1_pct: number; top5_pct: number;
  };
  totals: QtBook["totals"];
  warnings: string[];
};

export type QtScope = "all" | "stock" | "etf";

export type QtExposure = {
  base: string;
  as_of: string;
  total: number;
  rows: {
    sector: string; label: string; value: number; pct: number;
    from_stocks_pct: number; from_etfs_pct: number;
    holdings: { symbol: string; name: string }[];
  }[];
  /** Stocks only — an ETF publishes sector weights, never industry weights. */
  industries: { name: string; value: number; pct: Num }[];
  industry_basis: number;
  split: { stock_pct: number; etf_pct: number };
  unknown_pct: number;
  concentration: {
    sectors: number; top_sector: string | null;
    top_pct: Num; top3_pct: number;
  };
};

export type QtAllocStats = {
  ann_return_pct: number; ann_vol_pct: number;
  sharpe: Num; max_drawdown_pct: number;
};

export type QtOptimise = {
  base: string; scope: QtScope; scope_label: string; as_of: string;
  method: string; label: string; means: string;
  /** True for max-Sharpe: it needs return forecasts and overfits. */
  overfit_risk: boolean;
  cap_pct: number; min_weight_pct: number;
  sessions: number; from: string; to: string;
  basis: string;
  turnover_pct: number;
  current: QtAllocStats;
  target: QtAllocStats;
  rows: {
    symbol: string; name: string;
    current_pct: number; target_pct: number; delta_pct: number; risk_pct: number;
  }[];
  methods: { id: string; label: string; en: string;
             needs_returns: boolean; means: string }[];
  excluded: { symbol: string; reason: string }[];
  walk_forward: {
    train: { from: string; to: string; sessions: number };
    test: { from: string; to: string; sessions: number };
    rows: { method: string; label: string;
            in_sample: QtAllocStats; out_of_sample: QtAllocStats }[];
  } | null;
  walk_forward_error?: string;
};

/** A watchlist row. `market` is "CN" | "US" | "CA", or "??" for an unparseable one. */
export type WatchRef = StockRef & { market: string; at?: string | null };

export type ChainGraphPayload = {
  ticker: string;
  /** False until somebody has generated it — the normal state of a new stock. */
  generated: boolean;
  company_name?: string;
  products?: string[];
  macro_sectors?: string[];
  links?: { source: string; target: string }[];
};

export type LeadLagRow = {
  ticker: string; name: string; n_obs: number;
  beta: Num;
  relationship: string;
  signal: string;
  peak_corr: Num; peak_lag: number;
  p_t_leads_s: Num; lag_t_leads_s: number;
  p_s_leads_t: Num; lag_s_leads_t: number;
  /** Benjamini-Hochberg adjusted. The raw p is not the finding. */
  q_t_leads_s: Num; q_s_leads_t: Num; q_best: Num;
  survives_fdr: boolean;
  cointegrated: boolean;
  half_life: Num;
  /** One correlation per lag, aligned with `lags` / `lag_labels`. */
  xcorr: Num[];
};

export type LeadLagResult = {
  ticker: string; name: string;
  lookback_days: number; max_lag: number;
  sessions: number; from: string; to: string;
  lags: number[];
  lag_labels: string[];
  rows: LeadLagRow[];
  tests: {
    n: number; alpha: number; q: number;
    /** How many passed the raw threshold… */
    raw_hits: number;
    /** …against how many to expect from noise alone. */
    expected_false: number;
    survivors: number;
    method: string;
  };
  missing: { ticker: string; name: string }[];
};

export type DiscoverRow = {
  a: string; b: string;
  name_a: string; name_b: string;
  sector_a: string; sector_b: string;
  corr: number;
  /** In-sample, on the first half. */
  p_train: Num;
  /** Out-of-sample, on the half that played no part in choosing the pair. */
  p_test: Num;
  q: Num;
  survives: boolean;
  n_test: number;
  /** lead-lag only */
  leads?: "a" | "b"; lag?: number; same_direction?: boolean;
  /** pair-trade only */
  beta?: Num; half_life?: Num; tradeable?: boolean;
};

export type DiscoverResult = {
  kind: "pair-trade" | "lead-lag";
  /** Served from storage rather than re-run. */
  cached?: boolean;
  generated_at?: string | null;
  /** The published session this search was built on. */
  session?: string | null;
  requested: number;
  within_sector: boolean;
  lookback_days: number;
  sessions: number;
  train: { from: string; to: string; sessions: number };
  test: { from: string; to: string; sessions: number };
  /** The whole point: four survivors mean nothing without the 3,160 tested. */
  funnel: {
    pairs_possible: number; pairs_correlated: number; shortlisted: number;
    min_corr: number; universe: number; screened: number; retested: number;
    /** Cleared the holdout on the RAW threshold — the counterpart to
     *  expected_by_chance, which is also uncorrected. */
    retest_hits: number;
    expected_by_chance: number; survivors: number; alpha: number;
  };
  rows: DiscoverRow[];
};
