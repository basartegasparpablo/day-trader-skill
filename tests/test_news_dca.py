import copy

import numpy as np
import pandas as pd

from trader.config import load_config
from trader.dca import backtest_dca, core_symbols, run_dca
from trader.data import load_bars
from trader.live import news_filter, run_once
from trader.news import daily_news_scores, effective_session, event_study, interpret, lexicon_score
from tests.test_system import FakeBroker, NOW, cfg, live_env  # noqa: F401  (fixture reuse)

CFG = load_config()


# ------------------------------------------------------------------ lexicon
def test_lexicon_tone():
    assert lexicon_score("Apple beats estimates, raises guidance") > 0
    assert lexicon_score("Tesla recalls 2 million cars amid probe") < 0
    assert lexicon_score("Nvidia stops shipping US-approved AI chips to China") <= 0
    assert lexicon_score("Company did not miss estimates") > 0     # negation flips
    assert lexicon_score("Here are the holdings of Berkshire") == 0


# ------------------------------------------------------------------ session mapping + study
def test_news_after_close_counts_for_next_session():
    sessions = pd.bdate_range("2026-11-16", periods=5)  # Mon..Fri
    times = pd.Series(pd.to_datetime([
        "2026-11-16 14:00", "2026-11-16 21:30",   # Mon 09:00 ET, Mon 16:30 ET
        "2026-11-21 15:00",                      # Saturday -> beyond range
    ], utc=True))
    eff = effective_session(times, sessions)
    assert eff[0] == pd.Timestamp("2026-11-16")
    assert eff[1] == pd.Timestamp("2026-11-17")
    assert pd.isna(eff[2])


def test_event_study_detects_planted_effect():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-03", periods=400)
    market = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    rets = rng.normal(0, 0.01, 400)
    tone = rng.choice([-1.0, 0.0, 1.0], size=400)
    rets += 0.03 * tone                                # news moves the stock on the day
    stock = market * np.exp(np.cumsum(rets))
    prices = {"AAA": pd.DataFrame({"close": stock})}
    daily = pd.DataFrame({"session": idx, "symbol": "AAA", "score": tone, "articles": 1})
    res = event_study(daily, prices, market)
    s = res["summary"]
    assert s["pos_minus_neg_reaction"] > 0.04 and s["tstat_reaction"] > 5
    assert abs(s["tstat_next_5d"]) < 3                 # no planted drift
    text = " ".join(interpret(s))
    assert "Same-day reaction is real" in text


# ------------------------------------------------------------------ live news filter
def test_news_filter_blocks_negative(live_env):
    c = cfg()
    c["news"]["filter"]["enabled"] = True
    tone = lambda syms, cfg: {"AAA": (-0.8, 3, ["AAA sued for fraud"]), "BBB": (0.4, 2, [])}
    assert news_filter(["AAA", "BBB", "CCC"], c, "paper", True, tone_fn=tone) == ["BBB", "CCC"]


def test_news_filter_fails_open(live_env):
    c = cfg()
    c["news"]["filter"]["enabled"] = True

    def boom(syms, cfg):
        raise RuntimeError("news API down")
    assert news_filter(["AAA"], c, "paper", True, tone_fn=boom) == ["AAA"]


# ------------------------------------------------------------------ DCA
class DcaBroker(FakeBroker):
    def __init__(self, cash=1000, existing=()):
        super().__init__()
        self.cash, self.existing, self.notional = cash, set(existing), []

    def account(self):
        return {"equity": "10000", "last_equity": "10000", "cash": str(self.cash)}

    def order_by_client_id(self, coid):
        return {"id": coid} if coid in self.existing else None

    def buy_notional(self, sym, dollars, coid):
        self.notional.append((sym, dollars, coid))
        self.existing.add(coid)
        return {"id": coid}


def dca_cfg(**d):
    c = copy.deepcopy(CFG)
    c["dca"].update(d)
    return c


def test_dca_buys_once_per_month(live_env):
    c = dca_cfg(monthly_amount=300, allocations={"VTI": 0.6, "BND": 0.4})
    br = DcaBroker()
    run_dca(c, broker=br, now=NOW)
    assert sorted(br.notional) == [("BND", 120.0, "dca-2026-11-BND"), ("VTI", 180.0, "dca-2026-11-VTI")]
    run_dca(c, broker=br, now=NOW)                      # second run same month: nothing
    assert len(br.notional) == 2
    run_dca(c, broker=br, now=NOW + pd.Timedelta(days=20))  # next month: buys again
    assert len(br.notional) == 4


def test_dca_insufficient_cash(live_env):
    br = DcaBroker(cash=50)
    run_dca(dca_cfg(monthly_amount=500, allocations={"VTI": 1.0}), broker=br, now=NOW)
    assert br.notional == []


def test_dca_backtest_accounting():
    data = load_bars(["VTI"], "1Day", source="synthetic", start="2020-01-01", end="2022-12-31")
    r = backtest_dca(data, {"VTI": 1.0}, 100)
    assert r["months"] == 36 and r["contributed"] == 3600


def test_trading_bot_never_touches_core_holdings(live_env, monkeypatch):
    monkeypatch.delenv("ALPACA_DCA_API_KEY_ID", raising=False)
    c = cfg(daily_loss_limit=0.02)
    c["dca"]["allocations"] = {"SPY": 1.0}
    assert core_symbols(c) == {"SPY"}
    br = FakeBroker(equity=9_700, last_equity=10_000, positions={
        "SPY": {"qty": "50", "current_price": "100", "market_value": "5000"},
        "QQQ": {"qty": "5", "current_price": "100", "market_value": "500"}})
    run_once(c, broker=br, now=NOW)                     # daily loss limit -> flatten
    assert br.closed == ["QQQ"]                          # core SPY untouched
