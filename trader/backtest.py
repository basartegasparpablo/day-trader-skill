"""Bar-by-bar portfolio backtester that enforces the same risk rules as the live bot.

Timing (no look-ahead): signals computed at the close of bar t are executed at
the open of bar t+1. Stops are checked against each bar's low and fill at the
stop price, or at the open if the bar gapped through it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk import daily_loss_breached, drawdown_breached, position_size
from .strategies import Strategy


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    qty: int
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    reason: str = ""

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty if self.exit_price is not None else 0.0

    @property
    def ret(self) -> float:
        return self.exit_price / self.entry_price - 1 if self.exit_price is not None else 0.0


@dataclass
class Result:
    strategy: str
    equity: pd.Series
    trades: list[Trade]
    exposure: float
    halted_at: pd.Timestamp | None
    benchmark: pd.Series | None = None
    stats: dict = field(default_factory=dict)


def run_backtest(data: dict[str, pd.DataFrame], strategy: Strategy, cfg: dict,
                 start=None, end=None, capital: float | None = None,
                 benchmark: pd.Series | None = None) -> Result:
    risk = cfg["risk"]
    slip = float(cfg["costs"]["slippage_bps"]) / 1e4
    fee = float(cfg["costs"]["commission_per_order"])
    capital = float(capital or cfg["backtest"]["capital"])
    max_pos = int(risk["max_positions"])

    sig = strategy.signals(data)  # computed on full history so indicators are warmed up
    syms = sorted(data)
    idx = pd.DatetimeIndex(sorted(set().union(*(df.index for df in data.values()))))

    def panel(col, src):
        return pd.DataFrame({s: src[s][col] for s in syms}).reindex(idx)

    O, H, L, C = (panel(c, data) for c in ("open", "high", "low", "close"))
    Cff = C.ffill()
    E = panel("entry", sig).fillna(False).astype(bool)
    X = panel("exit", sig).fillna(False).astype(bool)
    SD, SC, W = panel("stop_dist", sig), panel("score", sig), panel("weight", sig)

    lo = int(np.searchsorted(idx, _ts(start, idx))) if start is not None else 0
    lo = max(lo, min(strategy.warmup_bars, len(idx) - 1))
    hi = int(np.searchsorted(idx, _ts(end, idx), side="right")) if end is not None else len(idx)
    sess = pd.Series(idx.normalize() if idx.tz is None else idx.tz_localize(None).normalize(), index=idx)
    last_of_session = sess.ne(sess.shift(-1)).to_numpy()

    cash = capital
    pos: dict[str, dict] = {}
    trades: list[Trade] = []
    pending_exit: set[str] = set()
    pending_entry: list[tuple[str, float, float]] = []
    equity = capital
    peak = capital
    day_start_equity = capital
    halted_at = None
    curve, invested = [], []

    def sell(sym, price, t, reason):
        nonlocal cash
        p = pos.pop(sym)
        cash += p["qty"] * price - fee
        tr = p["trade"]
        tr.exit_time, tr.exit_price, tr.reason = t, price, reason

    def liquidate_at_close(i, t, reason):
        for s in list(pos):
            sell(s, Cff.iat[i, syms.index(s)] * (1 - slip), t, reason)

    for i in range(lo, hi):
        t = idx[i]
        if i == lo or sess.iat[i] != sess.iat[i - 1]:
            day_start_equity = equity
            day_halted = False

        # 1) execute yesterday's decisions at today's open
        for s in sorted(pending_exit):
            j = syms.index(s)
            if s in pos and not math.isnan(O.iat[i, j]):
                sell(s, O.iat[i, j] * (1 - slip), t, "signal")
        pending_exit.clear()
        if pending_entry:
            eq_open = cash + sum(p["qty"] * (O.iat[i, syms.index(s)] if not math.isnan(O.iat[i, syms.index(s)])
                                             else Cff.iat[i - 1, syms.index(s)]) for s, p in pos.items())
            for s, sd, w in pending_entry:
                j = syms.index(s)
                if s in pos or len(pos) >= max_pos or math.isnan(O.iat[i, j]):
                    continue
                price = O.iat[i, j] * (1 + slip)
                qty = position_size(eq_open, cash - fee, price, sd, w, risk)
                if qty <= 0:
                    continue
                cash -= qty * price + fee
                stop = price - sd if (sd is not None and math.isfinite(sd)) else None
                tr = Trade(s, t, price, qty)
                trades.append(tr)
                pos[s] = {"qty": qty, "stop": stop, "trade": tr}
            pending_entry = []

        # 2) protective stops
        for s in list(pos):
            j = syms.index(s)
            stop = pos[s]["stop"]
            if stop is not None and not math.isnan(L.iat[i, j]) and L.iat[i, j] <= stop:
                sell(s, min(O.iat[i, j], stop) * (1 - slip), t, "stop")

        # 3) mark to market + circuit breakers
        equity = cash + sum(p["qty"] * Cff.iat[i, syms.index(s)] for s, p in pos.items())
        peak = max(peak, equity)
        if drawdown_breached(equity, peak, risk):
            liquidate_at_close(i, t, "drawdown_halt")
            halted_at = t
        elif daily_loss_breached(equity, day_start_equity, risk):
            liquidate_at_close(i, t, "daily_loss_limit")
            day_halted = True
        if strategy.flat_eod and last_of_session[i]:
            liquidate_at_close(i, t, "end_of_day")
        equity = cash + sum(p["qty"] * Cff.iat[i, syms.index(s)] for s, p in pos.items())
        curve.append(equity)
        invested.append(bool(pos))
        if halted_at is not None:
            break

        # 4) decide for tomorrow's open
        for s in pos:
            if X.iat[i, syms.index(s)]:
                pending_exit.add(s)
        if (day_halted and not last_of_session[i]) or (strategy.flat_eod and last_of_session[i]):
            continue
        cands = [s for j, s in enumerate(syms)
                 if E.iat[i, j] and not X.iat[i, j] and s not in pos and not math.isnan(C.iat[i, j])]
        cands.sort(key=lambda s: -np.nan_to_num(SC.iat[i, syms.index(s)], nan=-np.inf))
        slots = max_pos - (len(pos) - len(pending_exit))
        pending_entry = [(s, SD.iat[i, syms.index(s)], W.iat[i, syms.index(s)]) for s in cands[:max(slots, 0)]]

    # close anything still open at the final price for reporting
    if pos:
        last_i = lo + len(curve) - 1
        for s in list(pos):
            sell(s, Cff.iat[last_i, syms.index(s)], idx[last_i], "end_of_test")

    eq = pd.Series(curve, index=idx[lo:lo + len(curve)], name=strategy.name)
    res = Result(strategy.name, eq, trades, float(np.mean(invested)) if invested else 0.0, halted_at)
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill().dropna()
        res.benchmark = b / b.iloc[0] * capital if len(b) else None
    res.stats = compute_stats(res)
    return res


def _ts(x, idx):
    t = pd.Timestamp(x)
    if idx.tz is not None and t.tzinfo is None:
        t = t.tz_localize(idx.tz)
    return t


def _daily(eq: pd.Series) -> pd.Series:
    if eq.index.tz is not None:
        eq = eq.tz_localize(None)
    return eq.groupby(eq.index.normalize()).last()


def perf(eq: pd.Series) -> dict:
    d = _daily(eq)
    if len(d) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "mar": 0.0}
    rets = d.pct_change().dropna()
    years = max((d.index[-1] - d.index[0]).days / 365.25, 1 / 365.25)
    total = d.iloc[-1] / d.iloc[0] - 1
    cagr = (d.iloc[-1] / d.iloc[0]) ** (1 / years) - 1 if d.iloc[-1] > 0 else -1.0
    mdd = float((d / d.cummax() - 1).min())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {"total_return": float(total), "cagr": float(cagr), "max_drawdown": mdd,
            "sharpe": sharpe, "mar": float(cagr / abs(mdd)) if mdd < 0 else 0.0}


def compute_stats(res: Result) -> dict:
    s = perf(res.equity)
    closed = [t for t in res.trades if t.exit_price is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_win, gross_loss = sum(t.pnl for t in wins), -sum(t.pnl for t in losses)
    s.update({
        "start": str(res.equity.index[0].date()) if len(res.equity) else "",
        "end": str(res.equity.index[-1].date()) if len(res.equity) else "",
        "final_equity": float(res.equity.iloc[-1]) if len(res.equity) else 0.0,
        "trades": len(closed),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "avg_trade": float(np.mean([t.ret for t in closed])) if closed else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "exposure": res.exposure,
        "halted": str(res.halted_at) if res.halted_at is not None else "",
    })
    if res.benchmark is not None and len(res.benchmark) > 1:
        b = perf(res.benchmark)
        s.update({f"bench_{k}": v for k, v in b.items()})
    return s
