"""Minimal Alpaca Trading API client (REST, no SDK dependency)."""
from __future__ import annotations

import requests

from .data import alpaca_headers

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaBroker:
    def __init__(self, paper: bool = True, session: requests.Session | None = None):
        self.paper = paper
        self.base = PAPER_URL if paper else LIVE_URL
        self.s = session or requests.Session()
        self.s.headers.update(alpaca_headers())

    def _req(self, method, path, **kw):
        r = self.s.request(method, self.base + path, timeout=30, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def account(self):
        return self._req("GET", "/v2/account")

    def clock(self):
        return self._req("GET", "/v2/clock")

    def positions(self) -> dict[str, dict]:
        return {p["symbol"]: p for p in self._req("GET", "/v2/positions")}

    def open_orders(self):
        return self._req("GET", "/v2/orders", params={"status": "open", "limit": 500, "nested": "true"})

    def equity_peak(self) -> float | None:
        """Highest end-of-day equity over the last year (for the drawdown halt)."""
        h = self._req("GET", "/v2/account/portfolio/history", params={"period": "1A", "timeframe": "1D"})
        vals = [v for v in (h or {}).get("equity") or [] if v]
        return max(vals) if vals else None

    def cancel_orders_for(self, symbol: str):
        for o in self.open_orders():
            if o["symbol"] == symbol:
                self._req("DELETE", f"/v2/orders/{o['id']}")

    def close_position(self, symbol: str):
        self.cancel_orders_for(symbol)
        return self._req("DELETE", f"/v2/positions/{symbol}")

    def close_all(self):
        return self._req("DELETE", "/v2/positions", params={"cancel_orders": "true"})

    def buy_with_stop(self, symbol: str, qty: int, stop_price: float | None, tif: str):
        order = {"symbol": symbol, "qty": str(qty), "side": "buy", "type": "market", "time_in_force": tif}
        if stop_price is not None:
            order["order_class"] = "oto"
            order["stop_loss"] = {"stop_price": f"{stop_price:.2f}"}
        return self._req("POST", "/v2/orders", json=order)
