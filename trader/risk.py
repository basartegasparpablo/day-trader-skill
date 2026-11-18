"""Position sizing and account-level circuit breakers. Shared by backtest and live."""
from __future__ import annotations

import math
from typing import Any


def position_size(
    equity: float,
    cash: float,
    price: float,
    stop_dist: float | None,
    weight: float | None,
    risk: dict[str, Any],
    max_order_value: float | None = None,
) -> int:
    """Whole shares to buy. The smallest of every applicable cap wins."""
    if price <= 0 or equity <= 0:
        return 0
    caps = [
        equity * float(risk["max_position_pct"]) / price,
        max(cash, 0.0) / price,
    ]
    if stop_dist is not None and math.isfinite(stop_dist) and stop_dist > 0:
        caps.append(equity * float(risk["risk_per_trade"]) / stop_dist)
    if weight is not None and math.isfinite(weight) and weight > 0:
        caps.append(equity * weight / price)
    if max_order_value:
        caps.append(float(max_order_value) / price)
    return max(int(math.floor(min(caps))), 0)


def daily_loss_breached(equity: float, start_of_day_equity: float, risk: dict[str, Any]) -> bool:
    return equity <= start_of_day_equity * (1 - float(risk["daily_loss_limit"]))


def drawdown_breached(equity: float, peak_equity: float, risk: dict[str, Any]) -> bool:
    return equity <= peak_equity * (1 - float(risk["max_drawdown_halt"]))
