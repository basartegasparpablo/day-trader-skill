from __future__ import annotations

import pandas as pd


def sma(s: pd.Series | pd.DataFrame, n: int):
    return s.rolling(n, min_periods=n).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    """Wilder's RSI."""
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + au / ad)
    return out.mask((ad == 0) & au.notna(), 100.0)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
