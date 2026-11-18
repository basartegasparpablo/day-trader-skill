"""The live/paper bot. One call to `run_once` = one decision cycle.

Daily strategies: run once per trading day shortly after the open (e.g. 09:45 ET).
It reads completed daily bars up to yesterday, so it trades exactly like the
backtest (signal at close of t, order at open of t+1).

Intraday strategies: run every 5 minutes during market hours (`run --loop`).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from . import journal
from .broker import AlpacaBroker
from .config import ROOT
from .data import NY, load_bars
from .risk import daily_loss_breached, drawdown_breached, position_size
from .strategies import make_strategy

STOP_FILE = ROOT / "STOP"        # touch this file to stop all new trading (kill switch)
HALT_FILE = ROOT / "HALTED"      # written automatically by the drawdown circuit breaker
LIVE_ACK_ENV = "TRADER_I_UNDERSTAND_LIVE_RISK"


def _log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def check_mode(cfg) -> bool:
    """Return paper flag; refuse live unless explicitly acknowledged."""
    if cfg["mode"] == "live":
        if os.environ.get(LIVE_ACK_ENV) != "yes":
            raise SystemExit(f"Refusing LIVE trading: config.mode=live but env {LIVE_ACK_ENV}=yes is not set.")
        return False
    return True


def completed_bars(bars: dict[str, pd.DataFrame], timeframe: str, now_ny: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Drop the bar that is still forming so live signals match the backtest."""
    out = {}
    for s, df in bars.items():
        if timeframe == "1Day":
            df = df[df.index < now_ny.tz_localize(None).normalize()]
        else:
            df = df[df.index + pd.Timedelta(minutes=5) <= now_ny]
        if len(df):
            out[s] = df
    return out


def run_once(cfg, dry_run=False, broker=None, now=None) -> dict:
    paper = check_mode(cfg)
    mode = "paper" if paper else "live"
    strat = make_strategy(cfg["strategy"], cfg)
    risk = cfg["risk"]
    if STOP_FILE.exists():
        _log("Kill switch file STOP present: no trading. Delete it to resume.")
        return {"status": "stopped"}
    if HALT_FILE.exists():
        _log(f"HALTED by drawdown breaker ({HALT_FILE.read_text().strip()}). Review, then `python -m trader resume`.")
        return {"status": "halted"}

    broker = broker or AlpacaBroker(paper=paper)
    clock = broker.clock()
    if not clock["is_open"]:
        _log(f"Market closed (next open {clock['next_open']}). Nothing to do.")
        return {"status": "market_closed"}
    now_ny = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=NY)
    minutes_to_close = (pd.Timestamp(clock["next_close"]).tz_convert(NY) - now_ny).total_seconds() / 60

    acct = broker.account()
    if acct.get("trading_blocked") or acct.get("account_blocked"):
        raise SystemExit("Alpaca account is blocked from trading.")
    equity, last_equity, cash = float(acct["equity"]), float(acct["last_equity"]), float(acct["cash"])
    positions = broker.positions()
    actions = []

    def close(sym, reason):
        p = positions.get(sym)
        qty = int(float(p["qty"])) if p else 0
        _log(f"SELL {sym} x{qty} ({reason})")
        bid = None
        if not dry_run:
            r = broker.close_position(sym)
            bid = (r or {}).get("id")
        journal.log_order(mode, sym, "sell", qty, float(p["current_price"]) if p else None, None, reason, bid, dry_run)
        actions.append(("sell", sym, reason))

    def flatten(reason):
        for sym in list(positions):
            close(sym, reason)

    # --- circuit breakers (account level, stateless: Alpaca keeps the numbers)
    peak = broker.equity_peak() or equity
    if drawdown_breached(equity, max(peak, equity), risk):
        flatten("drawdown_halt")
        if not dry_run:
            HALT_FILE.write_text(f"{journal.now()} equity={equity:.2f} peak={peak:.2f}\n")
        journal.log_run(mode, strat.name, equity, cash, 0, "drawdown_halt", {"peak": peak})
        return {"status": "drawdown_halt", "actions": actions}
    if daily_loss_breached(equity, last_equity, risk):
        flatten("daily_loss_limit")
        journal.log_run(mode, strat.name, equity, cash, 0, "daily_loss_limit", {"last_equity": last_equity})
        return {"status": "daily_loss_limit", "actions": actions}

    if strat.flat_eod and minutes_to_close <= 10:
        flatten("end_of_day")
        journal.log_run(mode, strat.name, equity, cash, 0, "end_of_day")
        return {"status": "end_of_day", "actions": actions}

    # --- signals on completed bars only
    universe = strat.universe(cfg)
    lookback = timedelta(days=5) if strat.timeframe != "1Day" else timedelta(days=int(strat.warmup_bars * 1.6) + 30)
    start = (now_ny - lookback).tz_localize(None)
    bars = load_bars(universe, strat.timeframe, start=start, source="alpaca", cfg=cfg, use_cache=False)
    bars = completed_bars(bars, strat.timeframe, now_ny)
    sig = strat.signals(bars)
    last = {s: f.iloc[-1] for s, f in sig.items() if len(f)}
    if strat.timeframe == "1Day":
        # only act on yesterday's bar, never on stale data
        newest = max(f.index[-1] for f in bars.values())
        if (now_ny.tz_localize(None).normalize() - newest).days > 4:
            raise SystemExit(f"Daily data is stale (newest bar {newest.date()}); refusing to trade.")

    # --- exits
    for sym in list(positions):
        if sym in last and bool(last[sym]["exit"]):
            close(sym, "signal")
            positions.pop(sym)

    # --- entries
    pending_buys = {o["symbol"] for o in broker.open_orders() if o["side"] == "buy"}
    slots = int(risk["max_positions"]) - len(positions) - len(pending_buys)
    if strat.flat_eod and datetime.strptime(str(strat.params.get("last_entry", "15:00")), "%H:%M").time() <= now_ny.time():
        slots = 0
    cands = [s for s, row in last.items()
             if bool(row["entry"]) and not bool(row["exit"]) and s not in positions and s not in pending_buys]
    cands.sort(key=lambda s: -float(last[s]["score"]) if pd.notna(last[s]["score"]) else float("inf"))
    tif = "day" if strat.flat_eod else "gtc"
    for sym in cands[:max(slots, 0)]:
        price = float(bars[sym]["close"].iloc[-1])
        sd = float(last[sym]["stop_dist"]) if pd.notna(last[sym]["stop_dist"]) else None
        w = float(last[sym]["weight"]) if pd.notna(last[sym]["weight"]) else None
        qty = position_size(equity, cash, price, sd, w, risk, max_order_value=risk.get("max_order_value"))
        if qty <= 0:
            _log(f"skip {sym}: position size rounds to 0 shares")
            continue
        stop = round(price - sd, 2) if sd else None
        _log(f"BUY {sym} x{qty} ~${price:.2f}" + (f" stop ${stop:.2f}" if stop else ""))
        bid = None
        if not dry_run:
            bid = broker.buy_with_stop(sym, qty, stop, tif).get("id")
        journal.log_order(mode, sym, "buy", qty, price, stop, "signal", bid, dry_run)
        actions.append(("buy", sym, qty))
        cash -= qty * price

    status = "dry_run" if dry_run else "ok"
    journal.log_run(mode, strat.name, equity, cash, len(positions), status, {"actions": actions})
    _log(f"{mode} equity ${equity:,.2f} | positions {sorted(positions)} | actions {actions or 'none'}")
    return {"status": status, "actions": actions}


def run_loop(cfg, dry_run=False, interval=300):
    """For intraday strategies: one cycle every `interval` seconds until the close."""
    while True:
        res = run_once(cfg, dry_run=dry_run)
        if res["status"] in ("market_closed", "stopped", "halted", "drawdown_halt", "end_of_day"):
            return res
        time.sleep(interval - (time.time() % interval) + 5)
