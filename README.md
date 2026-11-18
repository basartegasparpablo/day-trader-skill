# day-trader-skill

A small, **risk-first** trading system for growing money slowly, plus a Claude skill
(`.claude/skills/day-trader/SKILL.md`) that operates it with you.

The path is: **backtest → pick honestly → paper trade automatically → review → (maybe) live.**

> ⚠️ Not financial advice. Most retail day traders lose money. This system is built to
> lose small and survive: hard limits on risk per trade, positions, daily loss and total
> drawdown, plus a kill switch. It cannot make a bad strategy profitable.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q                       # 27 tests, offline
```

Create a free Alpaca account, open **Paper Trading**, generate API keys, then:

```bash
export ALPACA_API_KEY_ID=...        # paper keys
export ALPACA_API_SECRET_KEY=...
```

## 0. The slow core: monthly index investing (start here)

The backtests below showed that, from 2008 to 2026, simply holding the index beat every
trading strategy here. So the default plan is: **buy a fixed amount of a broad index fund
every month, never sell, ignore the news.**

```bash
python -m trader dca-backtest --monthly 500    # what $500/month into VTI did since 2008
python -m trader dca --dry-run                 # what it would buy now
python -m trader dca                           # buy this month's amount (once per month; safe to run daily)
```

Real result (VTI, Jan 2008 - Nov 2026): $113,500 put in → ~$474,000, about **+13.5%/year**
on the money invested. The price: at the worst point (early 2009) the account was **41%
below** what had been put in. Staying in through that is the whole game.

Configure in `config.yaml → dca` (amount, `allocations` such as `{VTI: 0.6, VXUS: 0.3, BND: 0.1}`).
Best practice is a separate Alpaca account for it (`ALPACA_DCA_API_KEY_ID/SECRET`); in a shared
account the trading bot treats these symbols as untouchable.

## News: how daily headlines affect prices

```bash
python -m trader news                          # daily brief: headlines, tone (-1..+1), today's move
python -m trader news AAPL NVDA --hours 48
python -m trader news-study --start 2022-01-01 # needs Alpaca keys for history
```

`news-study` scores every headline for the watchlist companies, assigns it to the trading
day it could affect (after 16:00 ET → next day), and measures returns **relative to SPY**:

- **reaction**: the day the news hit. Usually clearly positive for good news and negative for
  bad news, but this move happens within minutes, before you can act.
- **next_1d / next_5d**: what happened *afterwards*. This is the only part a strategy could
  use. If it is not significant (|t| < 2), the news was already priced in.

Tone scoring is a free finance word list by default; set `news.scorer: claude` (needs
`pip install anthropic pydantic` and an Anthropic API key; costs money) for Claude to read
each headline, which handles cases like "stops shipping approved chips" far better.

Live: `news.filter.enabled: true` makes the bot skip buying a symbol with strongly negative
news in the last 24h. Leave it off unless `news-study` shows negative news keeps hurting
in the following days. The DCA core deliberately ignores news.

## 1. Choose a strategy

```bash
python -m trader compare                       # daily strategies on Yahoo data since 2008
python -m trader compare --source alpaca       # adds multi-year 5-min data for the intraday strategy
python -m trader backtest --strategy momentum_rotation --trades
```

| Strategy | Style | Trades | Idea |
|---|---|---|---|
| `momentum_rotation` | slow, monthly | ~1-3/month | Hold the 3 strongest ETFs that are above their 200-day average; cash otherwise |
| `trend_pullback` | swing, days | several/week | Buy oversold dips (RSI(2)<10) in uptrends, sell the bounce, 2.5×ATR stop |
| `opening_range_breakout` | intraday | ~daily | Buy a break of the first 30 min's high above VWAP, stop at the range low, flat by close |

Each strategy is tested on the full period and on a held-out **out-of-sample** slice
(last 40%). `compare` rejects anything that lost money out-of-sample, has too little
history or too few trades, or hit the drawdown halt, then picks the best
return-per-drawdown (MAR) and tells you if plain buy-and-hold SPY was better.
A report is saved in `reports/`. Set the winner as `strategy:` in `config.yaml`.

## 2. Paper trade automatically

```bash
python -m trader run --dry-run    # what it would do now
python -m trader run              # one cycle: daily strategies, once per day after the open
python -m trader run --loop       # intraday strategy: every 5 min until the close
```

**Hands-off option:** add `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` as GitHub
repository secrets. `.github/workflows/paper-trade.yml` then runs one cycle every
weekday at 14:50 UTC (daily strategies only; it refuses to run if `mode` is not `paper`).
To pause it, disable the workflow in the Actions tab. The intraday strategy needs an
always-on machine running `--loop`.

How a cycle works: check kill switch → check market open → **circuit breakers**
(daily loss vs. yesterday's close, drawdown vs. 1-year equity peak, both read from
Alpaca) → compute signals on *completed* bars only → exits → entries sized by the
risk rules, each with a broker-side stop order → everything logged to `journal.db`.

## 3. Review and control

```bash
python -m trader status           # equity, positions, open stops
python -m trader journal          # orders   (--runs: every cycle's outcome)
python -m trader stop             # kill switch: no new trades
python -m trader resume           # clear kill switch / drawdown halt (a human decision)
```

## Risk rules (`config.yaml`)

| Setting | Default | Meaning |
|---|---|---|
| `risk_per_trade` | 1% | A stopped-out trade loses at most ~1% of equity |
| `max_position_pct` | 34% | Cap per position |
| `max_positions` | 3 | Cap on open positions |
| `daily_loss_limit` | 2% | Down 2% on the day → flatten, no new trades today |
| `max_drawdown_halt` | 15% | Down 15% from peak → flatten and halt until you resume |
| `max_order_value` | $5,000 | Live sanity cap per order |

Config validation refuses obviously reckless values (e.g. >3% risk per trade).

## Going live (checklist)

- [ ] Strategy passed `compare` on **real** data, out-of-sample
- [ ] ≥ 3 months of paper trading with the same config, results in line with the backtest
- [ ] Only money you can afford to lose; emergency fund elsewhere
- [ ] Start small, keep `max_order_value` low, scale only after 3 more matching months
- [ ] Understand the tax treatment of short-term gains where you live

Then: `mode: live` in `config.yaml`, live API keys, and `export TRADER_I_UNDERSTAND_LIVE_RISK=yes`.
The bot refuses live trading without all of these, and the GitHub workflow never trades live.

## Layout

```
trader/strategies.py   signal logic (stateless, no look-ahead; tested)
trader/backtest.py     portfolio backtester using the same risk rules as live
trader/risk.py         position sizing + circuit breakers
trader/live.py         paper/live decision cycle
trader/broker.py       Alpaca REST client
trader/data.py         Yahoo / Alpaca / CSV / synthetic bars
trader/journal.py      SQLite audit log
trader/dca.py          monthly index investing (live + backtest)
trader/news.py         news fetch, tone scoring, event study, live filter
```
