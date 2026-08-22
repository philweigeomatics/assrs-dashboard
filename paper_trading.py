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

WARMUP_BARS = 130          # burn-in so MA60/BB are valid on the first shown bar
MIN_HISTORY_BARS = 60      # ≈3 months visible before the first playable day
DEFAULT_PLAY_BARS = 120


# ── Indicators (all strictly causal) ──────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-looking indicator set, matching the Technical Analysis page in
    definition but computed independently so nothing centered can creep in.
    """
    d = df.sort_index().copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    for w in (5, 10, 20, 60):
        d[f"MA{w}"] = c.rolling(w).mean()
    d["EMA5"] = c.ewm(span=5, adjust=False).mean()

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_Upper"], d["BB_Lower"] = mid + 2 * sd, mid - 2 * sd

    ef, es = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ef - es
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    d["OBV"] = (np.sign(c.diff().fillna(0)) * v).cumsum()
    d["Vol_MA20"] = v.rolling(20).mean()
    return d


# ── Stock selection ───────────────────────────────────────────────────────────

def pick_random_stock(data_manager, play_bars: int = DEFAULT_PLAY_BARS,
                      attempts: int = 12, seed: "int | None" = None) -> dict:
    """
    Choose a random A-share with enough history to run a full game.

    Returns {"ok", "ticker", "name", "industry", "data"} — `data` already
    carries indicators. The caller must keep ticker/name away from the UI until
    the player reveals it.
    """
    rng = random.Random(seed)
    need = WARMUP_BARS + MIN_HISTORY_BARS + play_bars

    try:
        universe = data_manager.get_all_stock_basic()
    except Exception as exc:
        return {"ok": False, "reason": f"stock list unavailable: {exc}"}
    if not universe:
        return {"ok": False, "reason": "stock_basic is empty"}

    tried = set()
    for _ in range(attempts):
        pick = rng.choice(universe)
        t = str(pick.get("ticker", "")).strip()
        if not t or t in tried:
            continue
        tried.add(t)
        try:
            raw = data_manager.get_single_stock_data_live(t, lookback_years=5)
        except Exception:
            continue
        if raw is None or raw.empty or len(raw) < need:
            continue
        raw = raw.sort_index()
        if raw[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
            raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
            if len(raw) < need:
                continue
        return {"ok": True, "ticker": t, "name": pick.get("name", t),
                "industry": pick.get("industry", ""), "data": compute_indicators(raw)}

    return {"ok": False,
            "reason": f"no stock with {need}+ sessions found in {attempts} tries"}


def choose_start(n_bars: int, play_bars: int, seed: "int | None" = None) -> int:
    """
    Index of the first playable day: random, but leaving warmup + visible
    history behind it and a full game ahead of it.
    """
    rng = random.Random(seed)
    lo = WARMUP_BARS + MIN_HISTORY_BARS
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
             seed: "int | None" = None) -> GameState:
    df = pick["data"]
    start = choose_start(len(df), play_bars, seed=seed)
    return GameState(ticker=pick["ticker"], name=pick["name"],
                     start_idx=start, cur_idx=start, play_bars=play_bars,
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


def buy(g: GameState, shares: int, price: float, date: str) -> dict:
    """Buy at the close. Multiples of 100 only; must be affordable with costs."""
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
    return {"ok": True, "fees": costs["total"]}


def sell(g: GameState, shares: int, price: float, date: str) -> dict:
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
    return {"ok": True, "fees": costs["total"]}


# ── Views ─────────────────────────────────────────────────────────────────────

def visible_frame(g: GameState, df: pd.DataFrame) -> pd.DataFrame:
    """
    Everything the player is allowed to see: history up to and including today,
    never beyond. This slice is the anti-lookahead guarantee.
    """
    lo = max(0, g.start_idx - MIN_HISTORY_BARS)
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
