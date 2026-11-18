"""Company news: fetch headlines, score their tone, and measure how they move prices.

Three uses:
  1. `python -m trader news`        - today's headlines per symbol with a tone score
                                      next to today's price move (daily brief)
  2. `python -m trader news-study`  - event study: after positive/negative news days,
                                      what did the stock do on the day (reaction) and
                                      over the next 1 and 5 days (drift), relative to SPY?
                                      Only the *drift* is something you can trade on.
  3. optional live filter           - don't open a position in a symbol with strongly
                                      negative news in the last N hours (config news.filter)

Sources:  alpaca  - Benzinga feed via Alpaca, history back to 2015 (needs Alpaca keys)
          yahoo   - recent headlines only (~a few days), no key needed
Scorers:  lexicon - free, offline finance word lists (default)
          claude  - Claude reads each headline (needs ANTHROPIC_API_KEY, costs money)
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .data import NY, alpaca_headers

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# ----------------------------------------------------------------------------- fetching


def fetch_news(symbols, start=None, end=None, source="alpaca", limit_per_symbol=None) -> pd.DataFrame:
    """Return DataFrame[id, time (UTC), symbol, headline, summary, source]; one row per (article, symbol)."""
    fn = {"alpaca": _alpaca_news, "yahoo": _yahoo_news}[source]
    rows = []
    for sym in symbols:
        rows += fn(sym, start, end, limit_per_symbol)
    df = pd.DataFrame(rows, columns=["id", "time", "symbol", "headline", "summary", "source"])
    if len(df):
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.drop_duplicates(["id", "symbol"]).sort_values("time").reset_index(drop=True)
    return df


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _alpaca_news(sym, start, end, limit):
    params = {"symbols": sym, "limit": 50, "sort": "asc", "include_content": "false",
              "start": _utc(start if start is not None else datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if end is not None:
        params["end"] = _utc(end).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, headers = [], alpaca_headers()
    while True:
        r = requests.get(ALPACA_NEWS_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()
        for a in js.get("news", []):
            rows.append((str(a["id"]), a["created_at"], sym, a.get("headline", ""), a.get("summary", ""), a.get("source", "")))
        if not js.get("next_page_token") or (limit and len(rows) >= limit):
            break
        params["page_token"] = js["next_page_token"]
    return rows[:limit] if limit else rows


def _yahoo_news(sym, start, end, limit):
    r = requests.get(YAHOO_SEARCH_URL, params={"q": sym, "newsCount": limit or 20, "quotesCount": 0},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    rows = []
    for a in r.json().get("news", []):
        # Yahoo search returns loosely related stories; keep only ones tagged with the symbol
        if sym not in (a.get("relatedTickers") or []):
            continue
        t = datetime.fromtimestamp(a["providerPublishTime"], tz=timezone.utc)
        if start is not None and t < _utc(start):
            continue
        rows.append((a["uuid"], t.isoformat(), sym, a.get("title", ""), "", a.get("publisher", "")))
    return rows


# ----------------------------------------------------------------------------- scoring

POSITIVE = """beat beats beating exceed exceeds exceeded surpass surpasses record upgrade upgrades upgraded
outperform outperforms raise raises raised boost boosts boosted strong stronger strength surge surges surged
soar soars soared jump jumps jumped rally rallies rallied gain gains gained growth grows grew profit profitable
approval approved approves wins win won award awarded partnership expands expansion buyback dividend
bullish optimistic upbeat robust accelerate accelerates breakthrough launch launches milestone tops topped
rebound rebounds recovers recovery higher climbs climbed positive""".split()

NEGATIVE = """miss misses missed disappoint disappoints disappointing disappointed downgrade downgrades downgraded
underperform cut cuts cutting lower lowers lowered weak weaker weakness plunge plunges plunged drop drops dropped
fall falls fell slump slumps slumped tumble tumbles tumbled sink sinks sank loss losses lose losing decline
declines declined lawsuit sue sues sued probe investigation investigate fraud recall recalls warn warns warning
layoff layoffs bankruptcy default delay delays delayed halt halts halted fine fined penalty breach hack
bearish pessimistic concern concerns fears fear risk risks slowdown shortfall resign resigns resigned
crash crashes antitrust ban banned subpoena negative scandal stop stops stopped suspend suspends suspended
block blocks blocked restrict restricts restriction restrictions tariff tariffs outage outages""".split()

NEGATORS = {"not", "no", "never", "without", "fails", "failed", "despite"}
_pos, _neg = set(POSITIVE), set(NEGATIVE)


def lexicon_score(text: str) -> float:
    """Tone in [-1, 1]: (pos - neg) / (pos + neg) over finance words; a negator flips the next hit."""
    words = re.findall(r"[a-z]+", (text or "").lower())
    pos = neg = 0
    flip = 0
    for w in words:
        if w in NEGATORS:
            flip = 3
            continue
        hit = 1 if w in _pos else -1 if w in _neg else 0
        if hit:
            if flip:
                hit = -hit
            pos += hit > 0
            neg += hit < 0
        flip = max(flip - 1, 0)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def claude_scores(df: pd.DataFrame, model: str = "claude-opus-5-5", batch: int = 100) -> pd.Series:
    """Ask Claude to rate each headline's likely price impact on the tagged symbol, in [-1, 1].

    Requires `pip install anthropic pydantic` and an Anthropic API key. Costs money:
    roughly 40 tokens per headline in, 15 out.
    """
    import anthropic
    from pydantic import BaseModel

    class Item(BaseModel):
        i: int
        score: float

    class Scores(BaseModel):
        items: list[Item]

    client = anthropic.Anthropic()
    out = pd.Series(0.0, index=df.index)
    for k in range(0, len(df), batch):
        chunk = df.iloc[k:k + batch]
        lines = "\n".join(f"{i}\t{r.symbol}\t{r.headline}" for i, r in zip(range(len(chunk)), chunk.itertuples()))
        resp = client.messages.parse(
            model=model,
            max_tokens=16000,
            output_config={"effort": "low"},
            system=("You rate financial news headlines. For each line (index, ticker, headline) give a score "
                    "from -1 (clearly bad for that company's share price) to 1 (clearly good); 0 if neutral, "
                    "irrelevant to that ticker, or already-known market recap. Judge only the headline."),
            messages=[{"role": "user", "content": lines}],
            output_format=Scores,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            continue
        for it in resp.parsed_output.items:
            if 0 <= it.i < len(chunk):
                out.iloc[k + it.i] = float(np.clip(it.score, -1, 1))
    return out


def score_news(df: pd.DataFrame, scorer: str = "lexicon") -> pd.DataFrame:
    df = df.copy()
    if not len(df):
        df["score"] = []
        return df
    if scorer == "claude":
        df["score"] = claude_scores(df)
    else:
        df["score"] = [lexicon_score(f"{h}. {s}") for h, s in zip(df["headline"], df["summary"].fillna(""))]
    return df


# ----------------------------------------------------------------------------- daily aggregation + study


def effective_session(times_utc: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Map each article to the trading session whose close-to-close return it can affect:
    published before 16:00 ET -> that day's session (or the next one on weekends/holidays),
    published at/after 16:00 ET -> the next session."""
    ny = times_utc.dt.tz_convert(NY)
    day = ny.dt.tz_localize(None).dt.normalize()
    after_close = ny.dt.hour >= 16
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")
    pos = np.searchsorted(sessions.values, day.values.astype("datetime64[ns]"))
    return pd.Series([sessions[p] if p < len(sessions) else pd.NaT for p in pos], index=times_utc.index)


def daily_news_scores(scored: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Average tone and article count per (session, symbol)."""
    if not len(scored):
        return pd.DataFrame(columns=["session", "symbol", "score", "articles"])
    s = scored.assign(session=effective_session(scored["time"], sessions)).dropna(subset=["session"])
    return (s.groupby(["session", "symbol"])
              .agg(score=("score", "mean"), articles=("score", "size")).reset_index())


def event_study(daily: pd.DataFrame, prices: dict[str, pd.DataFrame], market: pd.Series,
                threshold: float = 0.2) -> dict:
    """Market-adjusted returns around news days, by tone bucket."""
    mret = market.pct_change()
    rows = []
    for sym, g in daily.groupby("symbol"):
        if sym not in prices:
            continue
        c = prices[sym]["close"]
        r = c.pct_change()
        f1 = c.shift(-1) / c - 1
        f5 = c.shift(-5) / c - 1
        m1 = market.shift(-1) / market - 1
        m5 = market.shift(-5) / market - 1
        for row in g.itertuples():
            t = row.session
            if t not in c.index:
                continue
            rows.append({
                "symbol": sym, "session": t, "score": row.score, "articles": row.articles,
                "reaction": r.get(t, np.nan) - mret.get(t, np.nan),
                "next_1d": f1.get(t, np.nan) - m1.get(t, np.nan),
                "next_5d": f5.get(t, np.nan) - m5.get(t, np.nan),
            })
    ev = pd.DataFrame(rows).dropna(subset=["reaction"])
    if not len(ev):
        return {"events": ev, "table": pd.DataFrame(), "summary": {}}
    ev["bucket"] = np.where(ev["score"] >= threshold, "positive", np.where(ev["score"] <= -threshold, "negative", "neutral"))
    table = ev.groupby("bucket").agg(
        days=("score", "size"),
        avg_tone=("score", "mean"),
        reaction=("reaction", "mean"),
        reaction_up=("reaction", lambda x: (x > 0).mean()),
        next_1d=("next_1d", "mean"),
        next_5d=("next_5d", "mean"),
    ).reindex(["negative", "neutral", "positive"]).dropna(how="all")
    summary = {"events": len(ev), "symbols": ev["symbol"].nunique(),
               "start": str(ev["session"].min().date()), "end": str(ev["session"].max().date())}
    for col in ("reaction", "next_1d", "next_5d"):
        sub = ev[["score", col]].dropna()
        summary[f"corr_{col}"] = float(sub["score"].corr(sub[col])) if len(sub) > 2 else float("nan")
        pos = ev.loc[ev.bucket == "positive", col].dropna()
        neg = ev.loc[ev.bucket == "negative", col].dropna()
        summary[f"pos_minus_neg_{col}"] = float(pos.mean() - neg.mean()) if len(pos) and len(neg) else float("nan")
        summary[f"tstat_{col}"] = _welch_t(pos, neg)
    return {"events": ev, "table": table, "summary": summary}


def _welch_t(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def interpret(summary: dict) -> list[str]:
    """Plain-language reading of an event study."""
    out = []
    if not summary:
        return ["No news events matched price data."]
    t_react, t_1, t_5 = summary.get("tstat_reaction"), summary.get("tstat_next_1d"), summary.get("tstat_next_5d")
    if t_react is not None and not math.isnan(t_react) and abs(t_react) >= 2:
        out.append(f"Same-day reaction is real: positive-news days beat negative-news days by "
                   f"{summary['pos_minus_neg_reaction']:+.2%} vs SPY (t={t_react:.1f}). But by the time a "
                   "headline is public, most of that move has happened; you cannot capture it.")
    else:
        out.append("Same-day reaction to news tone is weak/insignificant in this sample.")
    tested = [(n, k, t) for n, k, t in (("next day", "next_1d", t_1), ("next 5 days", "next_5d", t_5))
              if t is not None and not math.isnan(t)]
    tradable = [(n, t, summary[f"pos_minus_neg_{k}"]) for n, k, t in tested if abs(t) >= 2]
    if not tested:
        out.append("Not enough days after the news yet to measure any follow-through.")
    elif tradable:
        for n, t, d in tradable:
            out.append(f"Drift after the news ({n}): {d:+.2%} positive-minus-negative (t={t:.1f}). This part could "
                       "be tradable; confirm it on a later, separate period before relying on it.")
    else:
        out.append("No significant drift after the news (next day / next 5 days): the news was priced in. "
                   "Using news to time trades would not have helped here.")
    if summary["events"] < 300:
        out.append(f"Only {summary['events']} news-days: too small a sample to conclude much. "
                   "Use --source alpaca with a multi-year --start.")
    return out


# ----------------------------------------------------------------------------- live filter


def recent_tone(symbols, cfg, now=None) -> dict[str, tuple[float, int, list[str]]]:
    """{symbol: (avg tone, article count, headlines)} over the configured lookback window."""
    ncfg = cfg.get("news") or {}
    hours = float((ncfg.get("filter") or {}).get("lookback_hours", 24))
    now = _utc(now if now is not None else datetime.now(timezone.utc))
    df = fetch_news(symbols, start=now - pd.Timedelta(hours=hours), end=now, source=ncfg.get("source", "alpaca"))
    df = score_news(df, ncfg.get("scorer", "lexicon"))
    out = {}
    for s in symbols:
        g = df[df["symbol"] == s] if len(df) else df
        out[s] = (float(g["score"].mean()) if len(g) else 0.0, len(g), list(g["headline"])[-5:] if len(g) else [])
    return out
