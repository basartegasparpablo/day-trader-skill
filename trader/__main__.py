"""CLI:  python -m trader <command>

  backtest  --strategy NAME     backtest one strategy (+ out-of-sample split)
  compare                       backtest every strategy, rank them, write a report
  run       [--dry-run] [--loop]  one paper/live decision cycle via Alpaca
  status                        account, positions, open orders
  dca       [--dry-run]         monthly index buy (once per month, safe to run daily)
  dca-backtest [--monthly 500]  what monthly index investing would have done
  news      [SYMBOLS...]        daily brief: today's headlines, tone, and price move
  news-study [--start DATE]     how did news tone move prices (reaction vs. drift)?
  journal   [--runs]            recent orders / runs from journal.db
  stop | resume                 kill switch on / off (also clears a drawdown halt)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date

import pandas as pd

from .backtest import run_backtest
from .config import ROOT, load_config
from .data import load_bars
from .strategies import STRATEGIES, make_strategy

PCT = ("total_return", "cagr", "max_drawdown", "win_rate", "avg_trade", "exposure",
       "bench_total_return", "bench_cagr", "bench_max_drawdown")


def _fmt(k, v):
    if isinstance(v, float):
        if k in PCT:
            return f"{v:+.1%}" if k != "win_rate" and k != "exposure" else f"{v:.0%}"
        return "inf" if math.isinf(v) else f"{v:.2f}"
    return str(v)


def evaluate(name, cfg, source=None, start=None):
    strat = make_strategy(name, cfg)
    src = source or cfg["data"]["source"]
    bt = cfg["backtest"]
    universe = strat.universe(cfg)
    syms = sorted(set(universe) | {cfg["benchmark"]})
    data = load_bars(syms, strat.timeframe, start=start or bt["start"], source=src, cfg=cfg)
    bench = data[cfg["benchmark"]]["close"] if cfg["benchmark"] in data else None
    tradable = {s: d for s, d in data.items() if s in universe}
    full = run_backtest(tradable, strat, cfg, benchmark=bench)
    out = {"strategy": name, "timeframe": strat.timeframe, "source": src, "full": full.stats}
    eq = full.equity
    if len(eq) > 20:
        cut = eq.index[int(len(eq) * float(bt.get("oos_split", 0.6)))]
        oos = run_backtest(tradable, strat, cfg, start=cut, benchmark=bench)
        out["oos"] = oos.stats
    return out, full


def print_eval(ev):
    print(f"\n=== {ev['strategy']} ({ev['timeframe']}, data: {ev['source']}) ===")
    keys = ["start", "end", "final_equity", "total_return", "cagr", "max_drawdown", "sharpe", "mar",
            "trades", "win_rate", "avg_trade", "profit_factor", "exposure", "halted",
            "bench_cagr", "bench_max_drawdown", "bench_sharpe"]
    rows = {"full period": ev["full"], **({"out-of-sample": ev["oos"]} if "oos" in ev else {})}
    print(pd.DataFrame({c: {k: _fmt(k, s.get(k, "")) for k in keys} for c, s in rows.items()}).to_string())


def recommend(evals, cfg):
    """Pick the strategy with the best out-of-sample return per unit of drawdown,
    among those that were profitable OOS, traded enough, and stayed within the
    drawdown halt. Honest about when plain buy-and-hold is the better choice."""
    max_dd = float(cfg["risk"]["max_drawdown_halt"])
    lines, ok = [], []
    for ev in evals:
        s = ev.get("oos") or ev["full"]
        why = []
        if s["cagr"] <= 0:
            why.append("lost money out-of-sample")
        years = (pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).days / 365.25 if s["start"] else 0
        if years < 1:
            why.append(f"only {years * 12:.0f} months of out-of-sample history (need >= 12; "
                       "for intraday use --source alpaca, Yahoo only has 60 days)")
        if s["trades"] < 20:
            why.append(f"only {s['trades']} trades out-of-sample (too few to trust)")
        if s["max_drawdown"] <= -max_dd:
            why.append(f"drawdown {s['max_drawdown']:.0%} hit the {max_dd:.0%} halt")
        if s["halted"]:
            why.append("triggered the drawdown halt")
        if why:
            lines.append(f"- REJECT {ev['strategy']}: " + "; ".join(why))
        else:
            ok.append((s["mar"], ev))
    if not ok:
        lines.append("\nNo strategy passed. Do not trade this. A plain index fund (e.g. SPY/VTI) "
                     "bought monthly is the honest default.")
        return None, "\n".join(lines)
    ok.sort(key=lambda x: -x[0])
    best = ok[0][1]
    s = best.get("oos") or best["full"]
    lines.append(f"- PICK {best['strategy']}: OOS CAGR {s['cagr']:+.1%}, max DD {s['max_drawdown']:.1%}, "
                 f"MAR {s['mar']:.2f}, Sharpe {s['sharpe']:.2f}, {s['trades']} trades")
    if "bench_sharpe" in s and s["sharpe"] < s["bench_sharpe"]:
        lines.append(f"  NOTE: buy-and-hold {cfg['benchmark']} had a better Sharpe ({s['bench_sharpe']:.2f}) "
                     "over the same period. The strategy's value is mainly the smaller drawdown; "
                     "consider keeping most money in the index and trading only a slice.")
    return best["strategy"], "\n".join(lines)


def cmd_compare(args, cfg):
    evals = []
    names = args.strategies or list(STRATEGIES)
    for name in names:
        try:
            ev, _ = evaluate(name, cfg, source=args.source)
            evals.append(ev)
            print_eval(ev)
        except Exception as e:  # e.g. intraday history unavailable
            print(f"\n=== {name}: skipped ({e})")
    pick, text = recommend(evals, cfg)
    print("\n=== Recommendation ===\n" + text)
    src = args.source or cfg["data"]["source"]
    if src == "synthetic":
        print("\n(!) SYNTHETIC random data: these numbers only prove the code runs. Use real data to decide.")
    report = ROOT / "reports" / f"compare_{date.today()}_{src}.md"
    report.parent.mkdir(exist_ok=True)
    with open(report, "w") as fh:
        fh.write(f"# Strategy comparison {date.today()} (data: {src})\n\n")
        for ev in evals:
            fh.write(f"## {ev['strategy']}\n\n```json\n{json.dumps(ev, indent=2, default=str)}\n```\n\n")
        fh.write("## Recommendation\n\n" + text + "\n")
    print(f"\nReport written to {report}")
    if pick and pick != cfg["strategy"]:
        print(f"To trade it, set `strategy: {pick}` in config.yaml.")


def cmd_dca_backtest(args, cfg):
    from .dca import backtest_dca
    d = cfg["dca"]
    alloc = d["allocations"]
    monthly = args.monthly or float(d["monthly_amount"])
    data = load_bars(sorted(alloc), "1Day", start=args.start or cfg["backtest"]["start"],
                     source=args.source or cfg["data"]["source"], cfg=cfg)
    r = backtest_dca(data, alloc, monthly)
    print(f"\n=== Monthly investing: ${monthly:,.0f}/month into {alloc} ===")
    print(f"period            {r['start']} -> {r['end']} ({r['years']} years, {r['months']} buys)")
    print(f"you put in        ${r['contributed']:,.0f}")
    print(f"worth at the end  ${r['final_value']:,.0f}   (gain ${r['gain']:,.0f})")
    print(f"annual return     {r['irr']:+.1%} on the money invested (money-weighted)")
    print(f"worst moment      {r['worst_point_vs_contributed']:+.1%} vs. what you had put in so far")
    print(f"index itself      CAGR {r['index_cagr']:+.1%}, worst drawdown {r['index_max_drawdown']:+.1%}")
    yearly = r["value"].groupby(r["value"].index.year).last()
    contrib = r["contributed_series"].groupby(r["contributed_series"].index.year).last()
    print("\nyear-end   invested      value")
    for y in yearly.index:
        print(f"  {y}   ${contrib[y]:>10,.0f}  ${yearly[y]:>10,.0f}")


def cmd_news(args, cfg):
    from .news import fetch_news, score_news
    ncfg = cfg.get("news") or {}
    syms = args.symbols or ncfg.get("watchlist") or cfg["universe"]
    src = args.source or ncfg.get("source", "alpaca")
    now = pd.Timestamp.now(tz="UTC")
    df = score_news(fetch_news(syms, start=now - pd.Timedelta(hours=args.hours), source=src),
                    args.scorer or ncfg.get("scorer", "lexicon"))
    try:
        px = load_bars(syms, "1Day", start=(now - pd.Timedelta(days=10)).tz_localize(None),
                       source="yahoo" if src == "yahoo" else "alpaca", cfg=cfg, use_cache=False)
    except Exception:
        px = {}
    print(f"News brief, last {args.hours:.0f}h ({src}, tone -1 bad ... +1 good)")
    for s in syms:
        g = df[df["symbol"] == s] if len(df) else df
        move = ""
        if s in px and len(px[s]) > 1:
            c = px[s]["close"]
            move = f"  last close {c.iloc[-1]:.2f} ({c.iloc[-1] / c.iloc[-2] - 1:+.2%} on {c.index[-1].date()})"
        tone = f"tone {g['score'].mean():+.2f} over {len(g)} articles" if len(g) else "no news"
        print(f"\n{s}: {tone}{move}")
        for r in g.tail(5).itertuples():
            print(f"   {r.score:+.2f}  {r.time.tz_convert('America/New_York'):%m-%d %H:%M}  {r.headline[:110]}")


def cmd_news_study(args, cfg):
    from .news import daily_news_scores, event_study, fetch_news, interpret, score_news
    ncfg = cfg.get("news") or {}
    syms = args.symbols or ncfg.get("watchlist")
    src = args.source or ncfg.get("source", "alpaca")
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp.now() - pd.Timedelta(days=730)
    print(f"Fetching {src} news for {len(syms)} symbols since {start.date()} ...")
    news = score_news(fetch_news(syms, start=start, source=src), args.scorer or ncfg.get("scorer", "lexicon"))
    if not len(news):
        print("No news found.")
        return
    px_start = min(start, news["time"].min().tz_localize(None)) - pd.Timedelta(days=10)
    prices = load_bars(sorted(set(syms) | {"SPY"}), "1Day", start=px_start,
                       source="yahoo" if cfg["data"]["source"] == "yahoo" else cfg["data"]["source"], cfg=cfg)
    sessions = prices["SPY"].index
    daily = daily_news_scores(news, sessions)
    res = event_study(daily, prices, prices["SPY"]["close"], float(ncfg.get("study_threshold", 0.2)))
    print(f"\n{len(news)} articles -> {res['summary'].get('events', 0)} symbol-news-days "
          f"({res['summary'].get('start')} .. {res['summary'].get('end')})")
    print("Returns are relative to SPY. reaction = the day the news hit; next_1d/next_5d = afterwards.\n")
    if len(res["table"]):
        t = res["table"].copy()
        for c in ("reaction", "next_1d", "next_5d"):
            t[c] = t[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "")
        t["reaction_up"] = t["reaction_up"].map(lambda v: f"{v:.0%}")
        t["avg_tone"] = t["avg_tone"].map(lambda v: f"{v:+.2f}")
        print(t.to_string())
        s = res["summary"]
        print("\ncorrelation of tone with:  reaction {:+.2f} | next day {:+.2f} | next 5 days {:+.2f}".format(
            s["corr_reaction"], s["corr_next_1d"], s["corr_next_5d"]))
    print("\nWhat this means:")
    for line in interpret(res["summary"]):
        print(" - " + line)
    report = ROOT / "reports" / f"news_study_{date.today()}_{src}.csv"
    res["events"].to_csv(report, index=False)
    print(f"\nEvery news-day with its returns: {report}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--strategy")
    b.add_argument("--source", choices=["yahoo", "alpaca", "csv", "synthetic"])
    b.add_argument("--start")
    b.add_argument("--trades", action="store_true", help="list every trade")
    c = sub.add_parser("compare")
    c.add_argument("--source", choices=["yahoo", "alpaca", "csv", "synthetic"])
    c.add_argument("--strategies", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true", help="decide and log, but place no orders")
    r.add_argument("--loop", action="store_true", help="repeat every 5 minutes until the close (intraday)")
    sub.add_parser("status")
    d = sub.add_parser("dca")
    d.add_argument("--dry-run", action="store_true")
    db = sub.add_parser("dca-backtest")
    db.add_argument("--monthly", type=float)
    db.add_argument("--start")
    db.add_argument("--source", choices=["yahoo", "alpaca", "csv", "synthetic"])
    n = sub.add_parser("news")
    n.add_argument("symbols", nargs="*")
    n.add_argument("--source", choices=["alpaca", "yahoo"])
    n.add_argument("--scorer", choices=["lexicon", "claude"])
    n.add_argument("--hours", type=float, default=24)
    ns = sub.add_parser("news-study")
    ns.add_argument("symbols", nargs="*")
    ns.add_argument("--start", default=None, help="default: 2 years ago (alpaca) / recent only (yahoo)")
    ns.add_argument("--source", choices=["alpaca", "yahoo"])
    ns.add_argument("--scorer", choices=["lexicon", "claude"])
    j = sub.add_parser("journal")
    j.add_argument("--runs", action="store_true")
    j.add_argument("-n", type=int, default=20)
    sub.add_parser("stop")
    sub.add_parser("resume")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "backtest":
        ev, full = evaluate(args.strategy or cfg["strategy"], cfg, source=args.source, start=args.start)
        print_eval(ev)
        if args.trades:
            for t in full.trades:
                print(f"{t.entry_time.date()} -> {t.exit_time.date() if t.exit_time is not None else '':<10} "
                      f"{t.symbol:<5} {t.qty:>5} @ {t.entry_price:8.2f} -> {t.exit_price or 0:8.2f} "
                      f"{t.ret:+7.2%} {t.reason}")
    elif args.cmd == "compare":
        cmd_compare(args, cfg)
    elif args.cmd == "run":
        from .live import run_loop, run_once
        (run_loop if args.loop else run_once)(cfg, dry_run=args.dry_run)
    elif args.cmd == "status":
        from .broker import AlpacaBroker
        from .live import check_mode
        br = AlpacaBroker(paper=check_mode(cfg))
        a = br.account()
        print(f"mode={cfg['mode']} strategy={cfg['strategy']} equity=${float(a['equity']):,.2f} "
              f"cash=${float(a['cash']):,.2f} prev_close_equity=${float(a['last_equity']):,.2f}")
        for s, p in br.positions().items():
            print(f"  {s:<5} {p['qty']:>6} avg {float(p['avg_entry_price']):8.2f} "
                  f"now {float(p['current_price']):8.2f} P/L {float(p['unrealized_plpc']):+.2%}")
        for o in br.open_orders():
            print(f"  open order: {o['side']} {o['qty']} {o['symbol']} {o['type']} {o.get('stop_price') or ''}")
    elif args.cmd == "dca":
        from .dca import run_dca
        run_dca(cfg, dry_run=args.dry_run)
    elif args.cmd == "dca-backtest":
        cmd_dca_backtest(args, cfg)
    elif args.cmd == "news":
        cmd_news(args, cfg)
    elif args.cmd == "news-study":
        cmd_news_study(args, cfg)
    elif args.cmd == "journal":
        from .journal import recent
        rows = recent("runs" if args.runs else "orders", args.n)
        print(pd.DataFrame(rows).to_string(index=False) if rows else "journal is empty")
    elif args.cmd == "stop":
        (ROOT / "STOP").write_text("manual stop\n")
        print("Kill switch ON: the bot will place no trades. Existing stops at the broker stay active.")
    elif args.cmd == "resume":
        for f in ("STOP", "HALTED"):
            (ROOT / f).unlink(missing_ok=True)
        print("Kill switch OFF and drawdown halt cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
