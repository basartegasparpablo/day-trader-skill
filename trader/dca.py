"""Monthly index investing (dollar-cost averaging): the slow, boring core.

Every month, buy a fixed dollar amount of broad index ETFs, no matter what the
news or the market is doing. Never sells. Idempotent: each month's order has a
unique client_order_id (dca-YYYY-MM-SYMBOL), so running it every day is safe;
it buys once per month, on the first run that finds the market open.

By default it uses a SEPARATE Alpaca account (env ALPACA_DCA_API_KEY_ID /
ALPACA_DCA_API_SECRET_KEY) so the trading bot can never touch these holdings.
If those aren't set it uses the main account, and the trading bot then treats
these symbols as off-limits "core" holdings.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import journal
from .backtest import perf


def core_symbols(cfg) -> set[str]:
    d = cfg.get("dca") or {}
    if not d.get("enabled") or uses_separate_account():
        return set()
    return set((d.get("allocations") or {}))


def uses_separate_account() -> bool:
    return bool(os.environ.get("ALPACA_DCA_API_KEY_ID") and os.environ.get("ALPACA_DCA_API_SECRET_KEY"))


def _validate(d):
    alloc = d.get("allocations") or {}
    if not alloc or abs(sum(alloc.values()) - 1) > 1e-6:
        raise ValueError("dca.allocations must be non-empty and sum to 1.0")
    if float(d["monthly_amount"]) <= 0:
        raise ValueError("dca.monthly_amount must be > 0")


def run_dca(cfg, broker=None, now=None, dry_run=False) -> dict:
    d = cfg.get("dca") or {}
    if not d.get("enabled"):
        print("DCA disabled in config.yaml (dca.enabled: false).")
        return {"status": "disabled"}
    _validate(d)
    from .live import check_mode, _log
    paper = check_mode(cfg)
    mode = "paper" if paper else "live"
    if broker is None:
        from .broker import AlpacaBroker
        broker = AlpacaBroker(paper=paper, key_env="ALPACA_DCA" if uses_separate_account() else "ALPACA")
    clock = broker.clock()
    if not clock["is_open"]:
        _log("DCA: market closed, will buy on the next run while the market is open.")
        return {"status": "market_closed"}
    month = (pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="America/New_York")).strftime("%Y-%m")
    cash = float(broker.account()["cash"])
    bought = []
    for sym, w in sorted(d["allocations"].items()):
        amount = round(float(d["monthly_amount"]) * float(w), 2)
        coid = f"dca-{month}-{sym}"
        if broker.order_by_client_id(coid):
            continue  # already bought this month
        if amount < 1:
            continue
        if cash < amount:
            _log(f"DCA: not enough cash for {sym} (${cash:,.2f} < ${amount:,.2f}). Deposit funds.")
            journal.log_order(mode, sym, "buy", 0, None, None, "dca_insufficient_cash", None, dry_run)
            continue
        _log(f"DCA: BUY ${amount:,.2f} of {sym} ({month})")
        bid = None if dry_run else broker.buy_notional(sym, amount, coid).get("id")
        journal.log_order(mode, sym, "buy", 0, amount, None, f"dca {month} ${amount:.2f}", bid, dry_run)
        cash -= amount
        bought.append((sym, amount))
    if not bought:
        _log(f"DCA: nothing to do for {month}.")
    return {"status": "ok", "bought": bought}


def backtest_dca(data: dict[str, pd.DataFrame], allocations: dict[str, float], monthly: float) -> dict:
    """Buy `monthly` dollars split by `allocations` at the open of the first trading day each month."""
    closes = pd.DataFrame({s: data[s]["close"] for s in allocations}).dropna()
    opens = pd.DataFrame({s: data[s]["open"] for s in allocations}).reindex(closes.index)
    month = pd.Series(closes.index.year * 12 + closes.index.month, index=closes.index)
    buy_days = closes.index[month.ne(month.shift()).to_numpy()]
    shares = pd.Series(0.0, index=list(allocations))
    contributed, values, contrib_series = 0.0, [], []
    buy_set = set(buy_days)
    for t in closes.index:
        if t in buy_set:
            for s, w in allocations.items():
                shares[s] += monthly * w / opens.at[t, s]
            contributed += monthly
        values.append(float((shares * closes.loc[t]).sum()))
        contrib_series.append(contributed)
    value = pd.Series(values, index=closes.index)
    contrib = pd.Series(contrib_series, index=closes.index)
    # unit-value index (strips out the contributions) for return/drawdown stats
    unit = (closes[list(allocations)] / closes[list(allocations)].iloc[0] * pd.Series(allocations)).sum(axis=1)
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    # money-weighted annual return (IRR) on the monthly cash flows
    flows = [(-monthly, t) for t in buy_days] + [(value.iloc[-1], closes.index[-1])]
    irr = _irr(flows)
    loss_vs_contrib = float(((value - contrib) / contrib.replace(0, np.nan)).min())
    return {
        "start": str(closes.index[0].date()), "end": str(closes.index[-1].date()), "years": round(years, 1),
        "months": len(buy_days), "contributed": round(contributed, 2), "final_value": round(value.iloc[-1], 2),
        "gain": round(value.iloc[-1] - contributed, 2), "irr": irr,
        "worst_point_vs_contributed": loss_vs_contrib,
        "index_cagr": perf(unit)["cagr"], "index_max_drawdown": perf(unit)["max_drawdown"],
        "value": value, "contributed_series": contrib,
    }


def _irr(flows) -> float:
    t0 = flows[0][1]

    def npv(r):
        return sum(cf / (1 + r) ** ((t - t0).days / 365.25) for cf, t in flows)

    lo, hi = -0.99, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
