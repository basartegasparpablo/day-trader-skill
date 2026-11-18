"""Market data loaders. Every loader returns {symbol: DataFrame[open, high, low, close, volume]}.

Daily bars are indexed by naive dates; intraday bars by America/New_York timestamps
(regular session only). Prices are split/dividend adjusted where the source allows.
"""
from __future__ import annotations

import os
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ROOT

NY = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]
CACHE = ROOT / "data" / "cache"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def load_bars(symbols, timeframe="1Day", start=None, end=None, source="yahoo", cfg=None, use_cache=True):
    cfg = cfg or {}
    loaders = {"yahoo": _yahoo, "alpaca": _alpaca, "csv": _csv, "synthetic": _synthetic}
    if source not in loaders:
        raise ValueError(f"unknown data source {source!r}")
    out = {}
    for sym in symbols:
        cache_file = CACHE / f"{source}_{sym}_{timeframe}.csv"
        df = None
        if use_cache and source in ("yahoo", "alpaca") and cache_file.exists() \
                and _time.time() - cache_file.stat().st_mtime < 12 * 3600:
            df = _read_cache(cache_file, timeframe)
        if df is None:
            df = loaders[source](sym, timeframe, start, end, cfg)
            if source in ("yahoo", "alpaca") and len(df):
                CACHE.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file)
        if start is not None:
            df = df[df.index >= _ts(start, df.index)]
        if end is not None:
            df = df[df.index <= _ts(end, df.index)]
        if len(df):
            out[sym] = df
    if not out:
        raise RuntimeError(f"no data loaded for {symbols} from {source}")
    return out


def _ts(x, index):
    t = pd.Timestamp(x)
    if getattr(index, "tz", None) is not None and t.tzinfo is None:
        t = t.tz_localize(index.tz)
    return t


def _read_cache(path, timeframe):
    df = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(df.index, utc=timeframe != "1Day")
    df.index = idx.tz_convert(NY) if timeframe != "1Day" else idx
    return df[COLS].astype(float)


def _finish(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df[COLS].astype(float).dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if timeframe != "1Day":
        t = df.index.hour * 60 + df.index.minute
        df = df[(t >= 9 * 60 + 30) & (t < 16 * 60)]
    return df


# ----------------------------------------------------------------------------- Yahoo
def _yahoo(sym, timeframe, start, end, cfg):
    interval = {"1Day": "1d", "5Min": "5m"}[timeframe]
    params = {"interval": interval, "events": "div,splits", "includeAdjustedClose": "true"}
    if timeframe == "1Day":
        p1 = int(pd.Timestamp(start or "2000-01-01").timestamp())
        p2 = int(pd.Timestamp(end).timestamp()) if end else int(_time.time())
        params.update(period1=p1, period2=p2)
    else:
        params["range"] = "60d"  # Yahoo only serves ~60 days of 5-minute bars
    r = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
        params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in COLS}, index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
    if timeframe == "1Day":
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
        if adj is not None:
            ratio = pd.Series(adj, index=df.index) / df["close"]
            for c in ("open", "high", "low", "close"):
                df[c] = df[c] * ratio
        df.index = df.index.tz_convert(NY).tz_localize(None).normalize()
    else:
        df.index = df.index.tz_convert(NY)
    return _finish(df, timeframe)


# ----------------------------------------------------------------------------- Alpaca
def alpaca_headers(prefix: str = "ALPACA"):
    key = os.environ.get(f"{prefix}_API_KEY_ID")
    secret = os.environ.get(f"{prefix}_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(f"set {prefix}_API_KEY_ID and {prefix}_API_SECRET_KEY (see README)")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca(sym, timeframe, start, end, cfg):
    def iso(t):
        t = pd.Timestamp(t)
        return (t.tz_localize("UTC") if t.tzinfo is None else t).isoformat()

    params = {
        "symbols": sym, "timeframe": timeframe, "adjustment": "all", "limit": 10000,
        "feed": (cfg.get("data") or {}).get("alpaca_feed", "iex"),
        "start": iso(start or "2016-01-01"),
        # free IEX feed: the most recent 15 minutes are not available
        "end": iso(end or datetime.now(timezone.utc) - timedelta(minutes=16)),
    }
    rows, headers = [], alpaca_headers()
    while True:
        r = requests.get(ALPACA_DATA_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows += (js.get("bars") or {}).get(sym, [])
        if not js.get("next_page_token"):
            break
        params["page_token"] = js["next_page_token"]
    if not rows:
        return pd.DataFrame(columns=COLS, dtype=float)
    df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    idx = pd.to_datetime(df["t"], utc=True).dt.tz_convert(NY)
    df.index = idx.dt.tz_localize(None).dt.normalize() if timeframe == "1Day" else idx
    df.index.name = None
    return _finish(df, timeframe)


# ----------------------------------------------------------------------------- CSV
def _csv(sym, timeframe, start, end, cfg):
    d = Path((cfg.get("data") or {}).get("csv_dir", "data/csv"))
    path = d if d.is_absolute() else ROOT / d
    df = pd.read_csv(path / f"{sym}.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("date", "datetime", "timestamp", "time") if c in df.columns)
    idx = pd.to_datetime(df.pop(tcol))
    if timeframe != "1Day":
        idx = idx.dt.tz_localize(NY) if idx.dt.tz is None else idx.dt.tz_convert(NY)
    df.index = pd.DatetimeIndex(idx)
    if "volume" not in df:
        df["volume"] = 0.0
    return _finish(df, timeframe)


# ----------------------------------------------------------------------------- Synthetic
def _synthetic(sym, timeframe, start, end, cfg):
    """Random-walk prices for offline testing ONLY. Results on these mean nothing."""
    rng = np.random.default_rng(sum((i + 1) * ord(ch) for i, ch in enumerate(sym)) * 7919 + 17)
    start = pd.Timestamp(start or "2008-01-01")
    end = pd.Timestamp(end or "2026-01-01")
    days = pd.bdate_range(start, end)
    if timeframe == "1Day":
        idx = days
        n = len(idx)
        drift, vol = 0.07 / 252, 0.17 / np.sqrt(252)
    else:
        days = days[-60:]
        idx = pd.DatetimeIndex(
            [pd.Timestamp(d.date()).tz_localize(NY) + pd.Timedelta(minutes=570 + 5 * k) for d in days for k in range(78)]
        )
        n = len(idx)
        drift, vol = 0.07 / (252 * 78), 0.17 / np.sqrt(252 * 78)
    # regime-switching drift so trend strategies have something to find
    regime = np.repeat(rng.choice([-1.5, 1.0, 2.0], size=n // 60 + 1), 60)[:n]
    rets = rng.normal(drift * regime, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                       "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)
    return _finish(df, timeframe)
