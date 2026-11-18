---
name: day-trader
description: Run the risk-first investing/trading system in this repo - monthly index investing (DCA), company news briefs and news-impact studies, strategy backtests and comparison, automatic Alpaca paper trading, journal review, and the go-live checklist. Use when the user asks about investing, trading, stock news, backtests, strategy choice, their paper/live bot, positions, P&L, or risk limits.
---

# Day Trader (risk-first)

The goal of this system is **slow, survivable growth**, not big wins. Your job is to
operate it and to be the honest voice in the room. Never promise returns.

## Ground rules (never break these)

1. **Paper before live.** `config.yaml` stays `mode: paper` until every item of the
   go-live checklist below is true. Never set `TRADER_I_UNDERSTAND_LIVE_RISK=yes`
   yourself; the user must do it.
2. **Never loosen risk limits to make a backtest look better** or to "win back" losses.
   `risk_per_trade` > 1%, `daily_loss_limit` > 2% or `max_drawdown_halt` > 15% need
   an explicit request from the user, and you must state the consequence.
3. **Never clear a drawdown halt (`resume`) on your own.** First show the user what
   happened (journal + status), then let them decide.
4. **Judge strategies on out-of-sample results**, not the full period. If buy-and-hold
   of the benchmark beats the strategy risk-adjusted, say so plainly.
5. Synthetic data (`--source synthetic`) only proves the code works. Never base a
   decision on it.
6. This is not financial advice; the user owns every decision.

## Workflow

### 0. The core: monthly index investing (DCA)
On real data 2008-2026, holding the index beat every trading strategy here, so this
is the default recommendation for "make money slowly".
```bash
python -m trader dca-backtest --monthly 500
python -m trader dca --dry-run      # then `dca` (idempotent: one buy per month)
```
When presenting results, always show the worst point vs. money put in (about -41% in
2009) next to the final gain: the user must be ready to keep buying through that.
The DCA never sells and ignores news, by design. Don't add timing rules to it.

### News: daily brief and impact study
```bash
python -m trader news [SYMBOLS] [--hours 24]       # headlines, tone, today's move
python -m trader news-study --start 2022-01-01     # reaction vs. follow-through, vs SPY
```
When the user asks how news affects a stock:
1. Run `news` for the brief. You may also read the headlines yourself and explain them,
   since you understand them better than the word-list scorer; say when the scorer
   got a headline wrong.
2. Explain the difference between **reaction** (same day, already happened by the time
   anyone reads it) and **drift** (next 1/5 days, the only tradable part), using `news-study`.
3. Only suggest enabling `news.filter` if drift after negative news is significant
   (|t| >= 2) on a large sample (hundreds of news-days, multi-year, alpaca source).
Never present one headline as a reason to buy or sell; news-driven trading is where
retail traders are slowest relative to professionals.

### 1. Choose a strategy (backtests)
```bash
python -m trader compare                      # all strategies, real daily data (Yahoo)
python -m trader compare --source alpaca      # needed for multi-year intraday history
python -m trader backtest --strategy trend_pullback --trades
```
Read the table for each strategy: out-of-sample CAGR, max drawdown, MAR (CAGR/|DD|),
Sharpe, trade count, and the `bench_*` rows (buy-and-hold). The `compare` command
rejects anything that lost money out-of-sample, has < 12 months or < 20 trades of
out-of-sample history, or hit the drawdown halt, then picks the best MAR.
Explain the pick in plain language and set `strategy:` in `config.yaml`.

Strategies available:
- `momentum_rotation`: slow, monthly. Holds the top 3 trending ETFs, cash otherwise.
- `trend_pullback`: swing. Buys oversold dips in uptrends, holds days, ATR stop.
- `opening_range_breakout`: intraday on 5-min bars, always flat by the close.

### 2. Paper trade (automatic)
Needs `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` (paper keys) in the environment.
```bash
python -m trader run --dry-run   # see what it would do, no orders
python -m trader run             # one decision cycle (daily strategies: ~09:45 ET each trading day)
python -m trader run --loop      # intraday strategy: every 5 min until the close
```
Automation: `.github/workflows/paper-trade.yml` runs `run` every weekday after the
open (daily strategies only). Intraday needs an always-on machine running `--loop`.

### 3. Review
```bash
python -m trader status          # equity, positions, open stop orders
python -m trader journal         # recent orders;  --runs for each cycle's outcome
```
When reviewing, compare paper results with the backtest's expectations (trade
frequency, win rate, average trade, drawdown). Large divergence = something is wrong
(costs, data, fills) - investigate before continuing.

### 4. Emergency controls
```bash
python -m trader stop     # kill switch: no new trades (broker-side stops stay active)
python -m trader resume   # user decision only: clears STOP and the drawdown HALTED file
```

## Go-live checklist (all must be true; walk the user through it)
- [ ] Strategy passed `compare` on **real** data, out-of-sample.
- [ ] At least **3 months** of paper trading with the same strategy and config.
- [ ] Paper results are within the backtest's range (no worse drawdown, similar trade stats).
- [ ] No unexplained errors in `journal --runs` for the last month.
- [ ] Money going in is money the user can afford to lose entirely, and an
      emergency fund exists outside the account.
- [ ] Start live with a small amount and keep `max_order_value` low; scale up only
      after another 3 months that match paper.
- [ ] Taxes: in the US, frequent trading means short-term gains taxed as income;
      the user should know the rules in their country.

Then the user sets `mode: live`, uses live API keys, and sets
`TRADER_I_UNDERSTAND_LIVE_RISK=yes` themselves.

## Honest framing to give the user
- Most retail day traders lose money after costs. Slower strategies with fewer
  trades and hard risk limits have the best chance of a positive result.
- Expect losing months. A 10-15% drawdown is normal even for a good strategy;
  that is why the halt exists.
- A sensible structure for "make money slowly": most savings in a broad index fund
  bought regularly, and only a slice in this system.
