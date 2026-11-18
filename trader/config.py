from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    if cfg.get("mode") not in ("paper", "live"):
        raise ValueError("config.mode must be 'paper' or 'live'")
    r = cfg["risk"]
    limits = {
        "risk_per_trade": (0, 0.03),
        "max_position_pct": (0, 1.0),
        "daily_loss_limit": (0, 0.10),
        "max_drawdown_halt": (0, 0.50),
    }
    for key, (lo, hi) in limits.items():
        v = float(r[key])
        if not lo < v <= hi:
            raise ValueError(f"risk.{key}={v} outside the sane range ({lo}, {hi}]")
    if int(r["max_positions"]) < 1:
        raise ValueError("risk.max_positions must be >= 1")
