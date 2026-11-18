"""Strategies produce per-symbol signal frames; they never place orders or size positions.

Every strategy is *stateless*: signals depend only on price history, never on
what the account holds. That keeps backtests and the live bot identical:
a signal on bar t is executed at the open of bar t+1 in both.

Signal frame columns (indexed like the input bars):
    entry     bool   open a position at the next bar's open
    exit      bool   close an existing position at the next bar's open
    stop_dist float  protective stop distance in price units (NaN = no stop)
    score     float  higher = preferred when there are more entries than free slots
    weight    float  target fraction of equity (NaN = size by risk only)
"""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from .indicators import atr, rsi, sma


class Strategy:
    name = "base"
    timeframe = "1Day"   # "1Day" or "5Min"
    flat_eod = False     # close everything at the end of each session
    warmup_bars = 0      # history needed before the first valid signal
    description = ""

    def __init__(self, **params):
        self.params = params

    def universe(self, cfg: dict) -> list[str]:
        return list(self.params.get("universe") or cfg["universe"])

    def signals(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        raise NotImplementedError


def _frame(index, entry, exit_, stop_dist=np.nan, score=0.0, weight=np.nan) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["entry"] = pd.Series(entry, index=index).fillna(False).astype(bool)
    out["exit"] = pd.Series(exit_, index=index).fillna(False).astype(bool)
    out["stop_dist"] = stop_dist
    out["score"] = score
    out["weight"] = weight
    return out


class TrendPullback(Strategy):
    """Swing: buy short-term dips inside a long-term uptrend, hold days.

    Entry: close > SMA(200) and RSI(2) < 10.
    Exit:  close > SMA(5) (bounce done) or close < SMA(200) (trend broken).
    Stop:  2.5 x ATR(14) below entry.
    """

    name = "trend_pullback"
    description = "Swing: buy oversold dips in uptrends, exit on the bounce (holds ~2-10 days)"

    def __init__(self, sma_trend=200, rsi_len=2, rsi_entry=10, sma_exit=5, atr_len=14, atr_stop=2.5, **kw):
        super().__init__(sma_trend=sma_trend, rsi_len=rsi_len, rsi_entry=rsi_entry,
                         sma_exit=sma_exit, atr_len=atr_len, atr_stop=atr_stop, **kw)
        self.warmup_bars = sma_trend + 5

    def signals(self, data):
        p = self.params
        out = {}
        for sym, df in data.items():
            c = df["close"]
            trend = sma(c, p["sma_trend"])
            r = rsi(c, p["rsi_len"])
            entry = (c > trend) & (r < p["rsi_entry"])
            exit_ = (c > sma(c, p["sma_exit"])) | (c < trend)
            out[sym] = _frame(df.index, entry, exit_, stop_dist=p["atr_stop"] * atr(df, p["atr_len"]), score=-r)
        return out


class MomentumRotation(Strategy):
    """Slow: once a month, hold the top-N ETFs by 6-month return that are above
    their 200-day average; anything that drops out is sold. If nothing
    qualifies, the account sits in cash. Very few trades, low effort.
    """

    name = "momentum_rotation"
    description = "Slow: monthly rotation into the strongest trending ETFs, cash otherwise"

    def __init__(self, lookback=126, top_n=3, sma_trend=200, **kw):
        super().__init__(lookback=lookback, top_n=top_n, sma_trend=sma_trend, **kw)
        self.warmup_bars = max(lookback, sma_trend) + 5

    def signals(self, data):
        p = self.params
        closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).sort_index()
        closes_ff = closes.ffill()
        mom = closes_ff / closes_ff.shift(p["lookback"]) - 1
        trend_ok = closes_ff > sma(closes_ff, p["sma_trend"])
        ranks = mom.where(trend_ok & mom.notna()).rank(axis=1, ascending=False, method="first")
        in_top = ranks <= p["top_n"]
        # Rebalance on the first trading day of each month (uses only past dates,
        # so it works identically in live trading); executes the next morning.
        month = pd.Series(closes.index.year * 12 + closes.index.month, index=closes.index)
        first_of_month = month.ne(month.shift())
        first_of_month.iloc[0] = False
        out = {}
        for sym, df in data.items():
            rb = first_of_month.reindex(df.index).fillna(False).astype(bool)
            top = in_top[sym].reindex(df.index).fillna(False).astype(bool)
            out[sym] = _frame(df.index, rb & top, rb & ~top,
                              score=mom[sym].reindex(df.index), weight=1.0 / p["top_n"])
        return out


class OpeningRangeBreakout(Strategy):
    """Intraday: buy when price breaks above the first 30 minutes' high (and is
    above VWAP), stop at the opening-range low, always flat by the close.
    Needs 5-minute bars.
    """

    name = "opening_range_breakout"
    timeframe = "5Min"
    flat_eod = True
    description = "Intraday: buy the breakout of the first 30 minutes' range, flat by the close"

    def __init__(self, range_minutes=30, last_entry="14:00", **kw):
        super().__init__(range_minutes=range_minutes, last_entry=last_entry, **kw)
        self.warmup_bars = 0

    def signals(self, data):
        p = self.params
        hh, mm = (int(x) for x in str(p["last_entry"]).split(":"))
        cutoff = time(hh, mm)
        out = {}
        for sym, df in data.items():
            idx = df.index
            day = pd.Series(idx.date, index=idx)
            mins = pd.Series(idx.hour * 60 + idx.minute - (9 * 60 + 30), index=idx)
            in_range = mins < p["range_minutes"]
            or_high = df["high"].where(in_range).groupby(day).transform("max")
            or_low = df["low"].where(in_range).groupby(day).transform("min")
            typical = (df["high"] + df["low"] + df["close"]) / 3
            vol = df["volume"].replace(0, np.nan).fillna(1.0)
            vwap = (typical * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
            before_cut = pd.Series([t < cutoff for t in idx.time], index=idx)
            brk = (df["close"] > or_high) & (mins >= p["range_minutes"]) & before_cut & (df["close"] > vwap)
            first = brk & (brk.astype(int).groupby(day).cumsum() == 1)
            out[sym] = _frame(idx, first, False, stop_dist=(df["close"] - or_low).clip(lower=1e-6),
                              score=(df["close"] - or_high) / or_high)
        return out


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (TrendPullback, MomentumRotation, OpeningRangeBreakout)
}


def make_strategy(name: str, cfg: dict) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name](**(cfg.get("strategies", {}).get(name) or {}))
