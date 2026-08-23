"""
paper_trading.py — engine for the blind A-share replay game.

The point of the game is training judgement against real price action with the
stock's identity withheld, so the player reacts to the chart rather than to
what they already believe about the company.

NO LOOKAHEAD — the property everything else depends on
------------------------------------------------------
Indicators are computed here rather than reused from analysis_engine, because
that module smooths ADX with scipy's savgol_filter, which is a CENTERED window:
it reads bars after the one it is labelling. Charting that in a replay would
quietly show the player part of tomorrow, and a training tool that leaks the
future teaches the wrong reflex. Every indicator below is strictly backward
looking, and each day's frame is computed on data up to and including that day
only.

A-share rules modelled
----------------------
  T+1        — shares bought today cannot be sold until the next session.
  Lot size   — buys in multiples of 100 (1手). A residual odd lot may be sold
               in full, which is what the exchange actually permits.
  Costs      — commission 0.025% (¥5 minimum), stamp duty 0.05% on SELL only,
               transfer fee 0.001% both sides. Ignoring these makes frequent
               trading look far better than it is, which is exactly the habit
               a training tool should not build.

Price limits are not enforced: the replay uses prices that actually printed, so
every fill is a price the market really traded at.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOT = 100
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
STAMP_DUTY_SELL = 0.0005
TRANSFER_FEE = 0.00001

# Never touch a bar before the 2024-09-24 regime change. A window straddling it
# would be decided by the policy break rather than by anything readable on the
# chart, which is the opposite of what this trains. Clamping the whole series —
# warmup included — keeps every moving average and percentile band built purely
# from post-regime data. ~485 sessions exist after the anchor, which is enough
# for a 180-session game plus warmup with ~165 possible start positions.
import analysis_engine as ae
from analysis_engine import REGIME_ANCHOR

WARMUP_BARS = 80           # MA60 needs 60; 80 leaves it settled before day one
MIN_HISTORY_BARS = 60      # ≈3 months visible before the first playable day
DEFAULT_PLAY_BARS = 120


# ── Indicators (all strictly causal) ──────────────────────────────────────────

def _trailing_slope(s: pd.Series, window: int = 5) -> pd.Series:
    """
    Slope from a TRAILING linear fit.

    The Technical Analysis page derives ADX slope from a Savitzky-Golay
    smooth, which is centered and therefore reads bars after the one it
    labels. Fitting only the trailing window gives the same shape of
    information without seeing the future.
    """
    def _f(a):
        if np.isnan(a).any():
            return np.nan
        return np.polyfit(np.arange(len(a)), a, 1)[0]
    return s.rolling(window).apply(_f, raw=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Technical Analysis page's indicator set, rebuilt strictly causally.

    Column names match analysis_engine where they overlap (RSI_14,
    Volume_ZScore, BB_Width_Percentile...) so accumulation_signals.detect()
    consumes this frame unchanged.
    """
    d = df.sort_index().copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    # ── Moving averages / bands ──
    for w in (5, 10, 20, 60):
        d[f"MA{w}"] = c.rolling(w).mean()
    d["EMA5"] = c.ewm(span=5, adjust=False).mean()
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_Upper"], d["BB_Lower"] = mid + 2 * sd, mid - 2 * sd
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / c
    # Trailing percentile — the squeeze read, without a forward-looking rank.
    d["BB_Width_Percentile"] = d["BB_Width"].rolling(120, min_periods=20).rank(pct=True)

    # ── MACD, with turn markers ──
    ef, es = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ef - es
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]
    _hg = d["MACD_Hist"]
    # A turn is only knowable one bar AFTER it happens — the histogram must
    # have fallen and then risen. Marking the low itself would require knowing
    # the next bar, which is precisely the leak being avoided.
    _gap = d["MACD"] - d["MACD_Signal"]
    d["MACD_Bottoming"] = (_hg > _hg.shift(1)) & (_hg.shift(1) <= _hg.shift(2)) & (d["MACD"] < 0)
    d["MACD_Peaking"] = (_hg < _hg.shift(1)) & (_hg.shift(1) >= _hg.shift(2)) & (d["MACD"] > 0)
    d["MACD_ClassicCrossover"] = (_gap > 0) & (_gap.shift(1) <= 0)
    d["MACD_BearishCrossover"] = (_gap < 0) & (_gap.shift(1) >= 0)
    # Still below the signal line but closing on it for two bars — the setup
    # that precedes a crossover, which is the point of flagging it separately.
    d["MACD_Approaching"] = (_gap < 0) & (_gap > _gap.shift(1)) & \
                            (_gap.shift(1) > _gap.shift(2))
    d["MACD_MomentumBuilding"] = (_hg > 0) & (_hg > _hg.shift(1)) & \
                                 (_hg.shift(1) > _hg.shift(2))

    def _macd_label(r):
        if r["MACD_ClassicCrossover"]:
            return "Crossover"
        if r["MACD_Approaching"]:
            return "Approaching"
        if r["MACD_Bottoming"]:
            return "Bottoming"
        if r["MACD_MomentumBuilding"]:
            return "Momentum"
        return ""
    d["MACD_Scenario"] = d[["MACD_ClassicCrossover", "MACD_Approaching",
                            "MACD_Bottoming", "MACD_MomentumBuilding"]].apply(
        _macd_label, axis=1)

    # ── RSI with dynamic bands ──
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    d["RSI"] = d["RSI_14"]                      # alias for the chart
    # Percentile bands beat fixed 30/70: what counts as oversold differs by name.
    d["RSI_P10"] = d["RSI_14"].rolling(120, min_periods=30).quantile(0.10)
    d["RSI_P90"] = d["RSI_14"].rolling(120, min_periods=30).quantile(0.90)
    d["RSI_Bottoming"] = (d["RSI_14"] > d["RSI_14"].shift(1)) & \
                         (d["RSI_14"].shift(1) <= d["RSI_P10"].shift(1))
    d["RSI_Peaking"] = (d["RSI_14"] < d["RSI_14"].shift(1)) & \
                       (d["RSI_14"].shift(1) >= d["RSI_P90"].shift(1))

    # ── ADX / DI (Wilder — causal by construction) ──
    up, dn = h.diff(), -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ATR"] = atr
    d["DI_Plus"] = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    d["DI_Minus"] = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (d["DI_Plus"] - d["DI_Minus"]).abs() / (d["DI_Plus"] + d["DI_Minus"]).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # The Technical Analysis page smooths ADX with savgol_filter, which is
    # centered. analysis_engine already ships an EWM fallback for when scipy is
    # unavailable, and that fallback is causal — so using it deliberately gives
    # the same downstream machinery without reading future bars. Every ADX
    # derivative below therefore matches the TA page's definitions exactly.
    _adxf = d["ADX"].bfill().fillna(0)
    d["ADX_LOWESS"] = _adxf.ewm(span=ae.ADX_EWM_SPAN, adjust=False).mean()
    d["ADX_BB_Middle"] = d["ADX_LOWESS"].rolling(ae.ADX_BB_WINDOW).mean()
    d["ADX_BB_Std"] = d["ADX_LOWESS"].rolling(ae.ADX_BB_WINDOW).std()
    d["ADX_BB_Upper"] = d["ADX_BB_Middle"] + ae.ADX_BB_STD * d["ADX_BB_Std"]
    d["ADX_BB_Lower"] = d["ADX_BB_Middle"] - ae.ADX_BB_STD * d["ADX_BB_Std"]
    d["ADX_Slope"] = _trailing_slope(d["ADX_LOWESS"], ae.ADX_SLOPE_WINDOW)
    d["ADX_Acceleration"] = d["ADX_Slope"].diff()
    # Reuse the real classifier rather than re-deriving it: same nine lifecycle
    # states, same thresholds, so the panel reads identically to the TA page.
    d = ae.apply_adx_patterns(d)
    d["DI_Screaming_Buy"] = (d["DI_Plus"] > d["DI_Minus"]) & \
        (d["DI_Plus"].shift(1) <= d["DI_Minus"].shift(1)) & (d["ADX"] > ae.ADX_WEAK)
    d["DI_Screaming_Sell"] = (d["DI_Minus"] > d["DI_Plus"]) & \
        (d["DI_Minus"].shift(1) <= d["DI_Plus"].shift(1)) & (d["ADX"] > ae.ADX_WEAK)

    # ── Volume / OBV / z-scores ──
    d["OBV"] = (np.sign(c.diff().fillna(0)) * v).cumsum()
    d["Vol_MA20"] = v.rolling(20).mean()
    # OBV动能 — the SLOPE of OBV, not its level. Cumulative OBV's level is
    # arbitrary (it depends where the series starts); its 20-day change divided
    # by average volume is zero-centred and reads directly as current
    # accumulation force in units of "× average daily volume".
    d["OBV_Momentum"] = (d["OBV"] - d["OBV"].shift(20)) / \
                        v.rolling(20, min_periods=1).mean().replace(0, np.nan)
    d["Volume_ZScore"] = ((v - v.rolling(100, min_periods=20).mean())
                          / v.rolling(100, min_periods=20).std().replace(0, np.nan))
    d["Price_ZScore"] = ((c - c.rolling(100, min_periods=20).mean())
                         / c.rolling(100, min_periods=20).std().replace(0, np.nan))
    return d


# ── Stock selection ───────────────────────────────────────────────────────────

def eligible_universe(data_manager, listed_before: str = "20210101") -> list:
    """
    Candidates worth dealing, read from stock_basic.

    stock_basic holds ~5,500 codes but the app only keeps per-ticker PRICE
    tables for the couple of hundred in its sector map, so most picks have to
    be fetched live. Filtering here is what stops that being slow: a candidate
    rejected on metadata costs nothing, while one rejected after a live fetch
    costs a round trip.

    Excluded, and why:
      listed too recently — cannot have warmup + history + play behind it.
      ST / *ST / 退 — ±5% limits and delisting mechanics, different game.
      北交所 (4xx/8xx/920xxx) — ±30% limits, also a different game.
    """
    try:
        df = data_manager.db.read_table(
            "stock_basic", columns="symbol,name,industry,list_date")
    except Exception:
        # Fall back to the name-only helper rather than failing outright.
        try:
            return [{"ticker": s["ticker"], "name": s["name"], "industry": ""}
                    for s in data_manager.get_all_stock_basic()]
        except Exception:
            return []
    if df is None or df.empty:
        return []

    out = []
    for _, r in df.iterrows():
        t = str(r.get("symbol") or "").strip()
        nm = str(r.get("name") or "").strip()
        if len(t) != 6 or not nm:
            continue
        if t[0] not in ("0", "3", "6"):          # main / ChiNext / STAR only
            continue
        if "ST" in nm.upper() or "退" in nm:
            continue
        ld = str(r.get("list_date") or "")
        if len(ld) == 8 and ld > listed_before:
            continue
        out.append({"ticker": t, "name": nm,
                    "industry": str(r.get("industry") or "")})
    return out


def _load_prices(data_manager, ticker: str, need: int):
    """
    DB first, live second.

    A per-ticker table is effectively instant; the live call is a round trip.
    Only ~188 tickers are maintained locally, so most deals still go live —
    but the ones that don't are free.
    """
    try:
        df = data_manager.get_single_stock_data_from_db(ticker)
        if df is not None and not df.empty and len(df) >= need:
            return df, "db"
    except Exception:
        pass
    try:
        # 3 years, not 5: warmup + history + a 180-session game needs ~370
        # bars, so 5 years was fetching roughly double what the game can use.
        df = data_manager.get_single_stock_data_live(ticker, lookback_years=3)
        return df, "live"
    except Exception:
        return None, "live"


def pick_random_stock(data_manager, play_bars: int = DEFAULT_PLAY_BARS,
                      hist_bars: int = MIN_HISTORY_BARS,
                      attempts: int = 12, seed: "int | None" = None,
                      prefer_db: bool = False) -> dict:
    """
    Choose a random A-share with enough history to run a full game.

    Returns {"ok", "ticker", "name", "industry", "data", "source"} — `data`
    already carries indicators. The caller must keep ticker/name out of the UI
    until the player reveals it.

    `prefer_db=True` restricts to tickers already cached locally: instant, but
    only the ~188 maintained names, which a regular player would start to
    recognise. Off by default because variety is the point of the game.
    """
    rng = random.Random(seed)
    need = WARMUP_BARS + int(hist_bars) + play_bars

    universe = eligible_universe(data_manager)
    if not universe:
        return {"ok": False, "reason": "no eligible stocks in stock_basic"}

    if prefer_db:
        # The locally-maintained set is the sector map — ONE call. Probing
        # db.table_exists() per ticker instead means thousands of sequential
        # Supabase round trips, which hangs for minutes.
        try:
            local = {t for v in data_manager.get_sector_stock_map().values()
                     for t in v}
            cached = [u for u in universe if u["ticker"] in local]
            if cached:
                universe = cached
        except Exception:
            pass

    rng.shuffle(universe)
    for pick in universe[:attempts]:
        t = pick["ticker"]
        raw, source = _load_prices(data_manager, t, need)
        if raw is None or raw.empty:
            continue
        raw = raw.sort_index()
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in raw.columns for c in cols):
            continue
        raw = raw.dropna(subset=cols)
        # Clamp BEFORE computing indicators, so no moving average or percentile
        # band is contaminated by pre-regime bars either.
        raw = raw[raw.index >= pd.Timestamp(REGIME_ANCHOR)]
        if len(raw) < need:
            continue
        return {"ok": True, "ticker": t, "name": pick["name"],
                "industry": pick.get("industry", ""), "source": source,
                "data": compute_indicators(raw)}

    return {"ok": False,
            "reason": f"no stock with {need}+ sessions found in {attempts} tries"}


def choose_start(n_bars: int, play_bars: int, hist_bars: int = MIN_HISTORY_BARS,
                 seed: "int | None" = None) -> int:
    """
    Index of the first playable day: random, but leaving warmup + the chosen
    visible history behind it and a full game ahead of it.
    """
    rng = random.Random(seed)
    lo = WARMUP_BARS + int(hist_bars)
    hi = n_bars - play_bars - 1
    return lo if hi <= lo else rng.randint(lo, hi)


# ── Costs ─────────────────────────────────────────────────────────────────────

def trade_costs(value: float, side: str) -> dict:
    """Commission + transfer fee, plus stamp duty on sells. Always positive."""
    commission = max(value * COMMISSION_RATE, COMMISSION_MIN)
    transfer = value * TRANSFER_FEE
    stamp = value * STAMP_DUTY_SELL if side == "sell" else 0.0
    total = commission + transfer + stamp
    return {"commission": commission, "transfer": transfer,
            "stamp": stamp, "total": total}


# ── Game state ────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    ticker: str
    name: str
    start_idx: int
    cur_idx: int
    play_bars: int
    hist_bars: int          # sessions of chart visible BEFORE session 1
    cash: float
    initial_cash: float
    # lots: [{"shares": int, "price": float, "date": "YYYY-MM-DD"}] — kept as
    # separate lots because T+1 eligibility is per purchase date, not per name.
    lots: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    undo: list = field(default_factory=list)
    revealed: bool = False

    # -- derived ----------------------------------------------------------
    @property
    def shares(self) -> int:
        return int(sum(l["shares"] for l in self.lots))

    def sellable(self, today: str) -> int:
        """T+1: only lots bought strictly before today can be sold."""
        return int(sum(l["shares"] for l in self.lots if l["date"] < today))

    def avg_cost(self) -> float:
        s = self.shares
        if not s:
            return 0.0
        return sum(l["shares"] * l["price"] for l in self.lots) / s

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price

    def day_number(self) -> int:
        return self.cur_idx - self.start_idx + 1

    def finished(self) -> bool:
        return self.day_number() >= self.play_bars


def new_game(pick: dict, cash: float = 100_000.0,
             play_bars: int = DEFAULT_PLAY_BARS,
             hist_bars: int = MIN_HISTORY_BARS,
             seed: "int | None" = None) -> GameState:
    df = pick["data"]
    start = choose_start(len(df), play_bars, hist_bars=hist_bars, seed=seed)
    return GameState(ticker=pick["ticker"], name=pick["name"],
                     start_idx=start, cur_idx=start, play_bars=play_bars,
                     hist_bars=int(hist_bars),
                     cash=float(cash), initial_cash=float(cash))


def _snapshot(g: GameState) -> dict:
    """Everything an action can change — enough to step all the way back."""
    return {"cur_idx": g.cur_idx, "cash": g.cash,
            "lots": [dict(l) for l in g.lots],
            "trades": [dict(t) for t in g.trades]}


def _push_undo(g: GameState) -> None:
    g.undo.append(_snapshot(g))


def revert(g: GameState) -> bool:
    """Step back one action. Returns False when already at the start."""
    if not g.undo:
        return False
    s = g.undo.pop()
    g.cur_idx, g.cash = s["cur_idx"], s["cash"]
    g.lots, g.trades = s["lots"], s["trades"]
    return True


# ── Actions ───────────────────────────────────────────────────────────────────

def advance(g: GameState) -> bool:
    """Pass — move to the next session. False if the game is over."""
    if g.finished():
        return False
    _push_undo(g)
    g.cur_idx += 1
    return True


def buy(g: GameState, shares: int, price: float, date: str,
        advance_after: bool = False) -> dict:
    """
    Buy at the close. Multiples of 100 only; must be affordable with costs.

    `advance_after` moves to the next session as part of the SAME undo step.
    That matters: the UI always trades and then advances, and when buy and
    advance each pushed their own snapshot, one revert only undid the advance
    — landing back on the purchase date with the shares still held, so
    repeating the trade doubled the position. One user action must be one
    undo entry.

    Validation happens before the snapshot, so a rejected order leaves no
    undo step behind.
    """
    if shares <= 0 or shares % LOT:
        return {"ok": False, "reason": f"买入必须是 {LOT} 的整数倍 · multiples of {LOT}"}
    value = shares * price
    costs = trade_costs(value, "buy")
    if value + costs["total"] > g.cash + 1e-9:
        return {"ok": False,
                "reason": f"资金不足 · need ¥{value + costs['total']:,.2f}, have ¥{g.cash:,.2f}"}

    _push_undo(g)
    g.cash -= value + costs["total"]
    g.lots.append({"shares": shares, "price": price, "date": date})
    g.trades.append({"date": date, "side": "buy", "shares": shares, "price": price,
                     "value": value, "fees": costs["total"], "day": g.day_number()})
    if advance_after and not g.finished():
        g.cur_idx += 1          # no second snapshot — same undo step
    return {"ok": True, "fees": costs["total"]}


def sell(g: GameState, shares: int, price: float, date: str,
         advance_after: bool = False) -> dict:
    """
    Sell at the close, T+1 enforced.

    Lots are consumed oldest first. A sell need not be a round lot only when it
    clears the entire remaining position — the exchange allows an odd residual
    to be closed but not created.
    """
    avail = g.sellable(date)
    if avail <= 0:
        return {"ok": False,
                "reason": "无可卖股份 · nothing sellable today (T+1: bought today "
                          "cannot be sold until tomorrow)"}
    if shares <= 0 or shares > avail:
        return {"ok": False, "reason": f"最多可卖 {avail} 股 · at most {avail}"}
    if shares % LOT and shares != g.shares:
        return {"ok": False,
                "reason": f"必须是 {LOT} 的整数倍，或一次清仓 · multiples of {LOT}, "
                          "or sell the whole remaining position"}

    _push_undo(g)
    value = shares * price
    costs = trade_costs(value, "sell")
    g.cash += value - costs["total"]

    left = shares
    for lot in sorted(g.lots, key=lambda x: x["date"]):
        if left <= 0 or lot["date"] >= date:
            continue
        take = min(lot["shares"], left)
        lot["shares"] -= take
        left -= take
    g.lots = [l for l in g.lots if l["shares"] > 0]

    g.trades.append({"date": date, "side": "sell", "shares": shares, "price": price,
                     "value": value, "fees": costs["total"], "day": g.day_number()})
    if advance_after and not g.finished():
        g.cur_idx += 1          # same undo step as the sale — see buy()
    return {"ok": True, "fees": costs["total"]}


# ── Views ─────────────────────────────────────────────────────────────────────

def visible_frame(g: GameState, df: pd.DataFrame) -> pd.DataFrame:
    """
    Everything the player is allowed to see: history up to and including today,
    never beyond. This slice is the anti-lookahead guarantee.
    """
    lo = max(0, g.start_idx - int(getattr(g, "hist_bars", MIN_HISTORY_BARS)))
    return df.iloc[lo:g.cur_idx + 1]


def today_row(g: GameState, df: pd.DataFrame) -> pd.Series:
    return df.iloc[g.cur_idx]


def today_str(g: GameState, df: pd.DataFrame) -> str:
    return pd.Timestamp(df.index[g.cur_idx]).strftime("%Y-%m-%d")


def performance(g: GameState, df: pd.DataFrame) -> dict:
    """Equity vs starting cash, and vs simply holding from day one."""
    price = float(today_row(g, df)["Close"])
    eq = g.equity(price)
    first = float(df.iloc[g.start_idx]["Close"])
    return {
        "price": price,
        "equity": eq,
        "cash": g.cash,
        "shares": g.shares,
        "avg_cost": g.avg_cost(),
        "pnl": eq - g.initial_cash,
        "pnl_pct": (eq / g.initial_cash - 1) * 100 if g.initial_cash else 0.0,
        "buy_hold_pct": (price / first - 1) * 100 if first else 0.0,
        "fees_paid": sum(t["fees"] for t in g.trades),
        "n_trades": len(g.trades),
        "day": g.day_number(),
        "play_bars": g.play_bars,
    }
