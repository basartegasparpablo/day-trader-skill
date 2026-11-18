import copy
import math

import numpy as np
import pandas as pd
import pytest

from trader.backtest import run_backtest
from trader.config import load_config
from trader.data import NY, load_bars
from trader.indicators import rsi
from trader.live import completed_bars, run_once
from trader.risk import position_size
from trader.strategies import Strategy, make_strategy, STRATEGIES

CFG = load_config()


def cfg(**risk):
    c = copy.deepcopy(CFG)
    c["risk"].update(risk)
    c["costs"]["slippage_bps"] = 0
    return c


def bars(closes, start="2020-01-01", opens=None, lows=None):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else c
    lo = np.asarray(lows, float) if lows is not None else np.minimum(o, c)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": lo, "close": c, "volume": 1e6}, index=idx)


class Scripted(Strategy):
    """Enters on given bar numbers, exits on others."""
    name = "scripted"

    def __init__(self, entries, exits=(), stop_dist=np.nan, weight=1.0):
        super().__init__()
        self.e, self.x, self.sd, self.w = set(entries), set(exits), stop_dist, weight

    def signals(self, data):
        out = {}
        for s, df in data.items():
            n = len(df)
            out[s] = pd.DataFrame({
                "entry": [i in self.e for i in range(n)], "exit": [i in self.x for i in range(n)],
                "stop_dist": self.sd, "score": 0.0, "weight": self.w}, index=df.index)
        return out


# ------------------------------------------------------------------ risk
def test_position_size_takes_smallest_cap():
    r = {"risk_per_trade": 0.01, "max_position_pct": 0.5}
    # risk cap: 10000*1%/2 = 50 shares; pct cap: 5000/100 = 50; weight cap: 0.3 -> 30
    assert position_size(10_000, 10_000, 100, 2.0, 0.3, r) == 30
    assert position_size(10_000, 10_000, 100, 2.0, None, r) == 50
    assert position_size(10_000, 1_000, 100, 2.0, None, r) == 10        # cash-limited
    assert position_size(10_000, 10_000, 100, None, None, r, max_order_value=2_000) == 20
    assert position_size(10_000, 10_000, 100, float("nan"), None, r) == 50
    assert position_size(0, 0, 100, 1, None, r) == 0


def test_config_rejects_reckless_risk(tmp_path):
    import yaml
    c = copy.deepcopy(CFG)
    c["risk"]["risk_per_trade"] = 0.10
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        load_config(p)


# ------------------------------------------------------------------ backtest mechanics
def test_signal_executes_next_open_no_lookahead():
    df = bars([100, 100, 110, 120, 130], opens=[100, 100, 105, 115, 125])
    res = run_backtest({"AAA": df}, Scripted(entries=[1], exits=[3]), cfg(max_position_pct=1.0), capital=1000)
    t = res.trades[0]
    assert t.entry_time == df.index[2] and t.entry_price == 105   # signal bar 1 -> open of bar 2
    assert t.exit_time == df.index[4] and t.exit_price == 125     # signal bar 3 -> open of bar 4


def test_stop_fills_at_stop_or_gap_open():
    df = bars([100, 100, 100, 100, 100], opens=[100, 100, 100, 100, 100], lows=[99, 99, 99, 94, 99])
    res = run_backtest({"AAA": df}, Scripted(entries=[0], stop_dist=5.0), cfg(max_position_pct=1.0, max_drawdown_halt=0.5, daily_loss_limit=0.1), capital=1000)
    assert res.trades[0].reason == "stop" and res.trades[0].exit_price == 95
    df2 = bars([100, 100, 100, 90, 90], opens=[100, 100, 100, 90, 90], lows=[99, 99, 99, 89, 89])
    res2 = run_backtest({"AAA": df2}, Scripted(entries=[0], stop_dist=5.0), cfg(max_position_pct=1.0, max_drawdown_halt=0.5, daily_loss_limit=0.1), capital=1000)
    assert res2.trades[0].exit_price == 90  # gapped through the stop -> filled at the open


def test_drawdown_halt_stops_trading():
    closes = [100, 100, 95, 90, 85, 80, 80, 80]
    res = run_backtest({"AAA": bars(closes)}, Scripted(entries=[0, 6]),
                       cfg(max_position_pct=1.0, max_drawdown_halt=0.15, daily_loss_limit=0.1), capital=1000)
    assert res.halted_at is not None
    assert res.trades[-1].reason == "drawdown_halt"
    assert len(res.trades) == 1  # the later entry signal was ignored


def test_max_positions_respected():
    data = {s: bars([100] * 6) for s in "ABCDE"}
    res = run_backtest(data, Scripted(entries=[0]), cfg(max_positions=2, max_position_pct=0.3), capital=1000)
    assert len(res.trades) == 2


def test_intraday_flat_at_end_of_day():
    data = load_bars(["SPY"], "5Min", source="synthetic")
    strat = make_strategy("opening_range_breakout", CFG)
    res = run_backtest(data, strat, cfg())
    assert res.trades, "expected some breakout trades"
    for t in res.trades:
        assert t.exit_time.date() == t.entry_time.date()
        assert t.entry_time.time() >= pd.Timestamp("10:00").time()


# ------------------------------------------------------------------ strategies
def test_rsi_bounds():
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 300).cumsum() + 100)
    r = rsi(s, 2).dropna()
    assert ((r >= 0) & (r <= 100)).all()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_have_no_lookahead(name):
    """Signals up to bar t must not change when future bars are appended."""
    strat = make_strategy(name, CFG)
    tf = strat.timeframe
    full = load_bars(["SPY", "QQQ", "IWM", "TLT"], tf, source="synthetic")
    cut = {s: df.iloc[: len(df) * 2 // 3] for s, df in full.items()}
    a, b = strat.signals(cut), strat.signals(full)
    for s in cut:
        n = len(cut[s])
        pd.testing.assert_frame_equal(a[s][["entry", "exit"]], b[s][["entry", "exit"]].iloc[:n])


def test_momentum_rotation_trades_monthly_only():
    data = load_bars(CFG["universe"], "1Day", source="synthetic", start="2010-01-01")
    sig = make_strategy("momentum_rotation", CFG).signals(data)
    days = sig["SPY"].index[sig["SPY"]["entry"] | sig["SPY"]["exit"]]
    prev = sig["SPY"].index.to_series().shift()
    assert all(d.month != prev[d].month for d in days)


# ------------------------------------------------------------------ live bot (fake broker)
class FakeBroker:
    def __init__(self, equity=10_000, last_equity=10_000, peak=10_000, positions=None):
        self.eq, self.last, self.peak = equity, last_equity, peak
        self.pos = positions or {}
        self.orders, self.closed = [], []

    def clock(self):
        return {"is_open": True, "next_close": "2026-11-18T16:00:00-05:00", "next_open": ""}

    def account(self):
        return {"equity": str(self.eq), "last_equity": str(self.last), "cash": str(self.eq)}

    def positions(self):
        return dict(self.pos)

    def open_orders(self):
        return []

    def equity_peak(self):
        return self.peak

    def close_position(self, s):
        self.closed.append(s)
        return {"id": "x"}

    def buy_with_stop(self, s, q, stop, tif):
        self.orders.append((s, q, stop, tif))
        return {"id": "y"}


@pytest.fixture
def live_env(monkeypatch, tmp_path):
    import trader.journal as j
    import trader.live as live
    monkeypatch.setattr(j, "DB", tmp_path / "j.db")
    monkeypatch.setattr(live, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(live, "HALT_FILE", tmp_path / "HALTED")
    monkeypatch.setattr(live, "load_bars",
                        lambda syms, tf, **kw: load_bars(syms, tf, source="synthetic", start="2024-01-01", end="2026-11-18"))
    return tmp_path


NOW = pd.Timestamp("2026-11-18 09:45", tz=NY)


def test_live_daily_loss_limit_flattens(live_env):
    br = FakeBroker(equity=9_700, last_equity=10_000, positions={"SPY": {"qty": "5", "current_price": "100"}})
    res = run_once(cfg(), broker=br, now=NOW)
    assert res["status"] == "daily_loss_limit" and br.closed == ["SPY"] and not br.orders


def test_live_drawdown_halt_writes_halt_file(live_env):
    br = FakeBroker(equity=8_000, last_equity=8_000, peak=10_000)
    assert run_once(cfg(), broker=br, now=NOW)["status"] == "drawdown_halt"
    assert (live_env / "HALTED").exists()
    assert run_once(cfg(), broker=br, now=NOW)["status"] == "halted"


def test_live_kill_switch(live_env):
    (live_env / "STOP").write_text("x")
    assert run_once(cfg(), broker=FakeBroker(), now=NOW)["status"] == "stopped"


def test_live_refuses_live_mode_without_ack(live_env, monkeypatch):
    monkeypatch.delenv("TRADER_I_UNDERSTAND_LIVE_RISK", raising=False)
    c = cfg()
    c["mode"] = "live"
    with pytest.raises(SystemExit):
        run_once(c, broker=FakeBroker(), now=NOW)


def test_live_cycle_respects_limits(live_env):
    c = cfg()
    c["strategy"] = "trend_pullback"
    br = FakeBroker()
    res = run_once(c, broker=br, now=NOW)
    assert res["status"] == "ok"
    assert len(br.orders) <= c["risk"]["max_positions"]
    for s, q, stop, tif in br.orders:
        assert stop is not None and tif == "gtc" and q > 0


def test_completed_bars_drops_forming_daily_bar():
    df = bars([1, 2, 3], start="2026-11-16")
    out = completed_bars({"A": df}, "1Day", NOW)
    assert out["A"].index[-1] == pd.Timestamp("2026-11-17")
