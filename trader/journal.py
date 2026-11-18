"""SQLite journal of every bot run and every order, so results can be audited."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .config import ROOT

DB = ROOT / "journal.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (ts TEXT, mode TEXT, strategy TEXT, equity REAL, cash REAL,
                                 positions INTEGER, status TEXT, details TEXT);
CREATE TABLE IF NOT EXISTS orders (ts TEXT, mode TEXT, symbol TEXT, side TEXT, qty INTEGER,
                                   est_price REAL, stop_price REAL, reason TEXT, broker_id TEXT, dry_run INTEGER);
"""


def _conn(path=None):
    c = sqlite3.connect(path or DB)
    c.executescript(SCHEMA)
    return c


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_run(mode, strategy, equity, cash, positions, status, details=None, path=None):
    with _conn(path) as c:
        c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                  (now(), mode, strategy, equity, cash, positions, status, json.dumps(details or {})))


def log_order(mode, symbol, side, qty, est_price, stop_price, reason, broker_id, dry_run, path=None):
    with _conn(path) as c:
        c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (now(), mode, symbol, side, qty, est_price, stop_price, reason, broker_id, int(dry_run)))


def recent(table="orders", n=20, path=None):
    with _conn(path) as c:
        cur = c.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (n,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
