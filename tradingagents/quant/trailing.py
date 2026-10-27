"""Pure trailing-stop rules, shared by the live position manager and the backtester.

- At +breakeven_r: SL -> entry + spread (buy) / entry - spread (sell)
- After +trail_start_r: SL trails trail_atr_multiplier x ATR behind the highest high (buy)
  or lowest low (sell) since entry
- SL only ever moves in the profit direction, and only by a meaningful amount
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrailParams:
    breakeven_r: float = 1.0
    trail_start_r: float = 1.5
    trail_atr_multiplier: float = 2.0
    min_change_atr: float = 0.05
    partial_enabled: bool = False
    partial_at_r: float = 2.0
    partial_fraction: float = 0.5

    @classmethod
    def from_config(cls, c: dict) -> "TrailParams":
        pc = c.get("partial_close") or {}
        return cls(
            breakeven_r=float(c.get("breakeven_r", 1.0)),
            trail_start_r=float(c.get("trail_start_r", 1.5)),
            trail_atr_multiplier=float(c.get("trail_atr_multiplier", 2.0)),
            min_change_atr=float(c.get("min_change_atr", 0.05)),
            partial_enabled=bool(pc.get("enabled", False)),
            partial_at_r=float(pc.get("at_r", 2.0)),
            partial_fraction=float(pc.get("fraction", 0.5)),
        )


def profit_r(direction: str, entry: float, initial_sl: float, price: float) -> float:
    risk = abs(entry - initial_sl)
    if risk <= 0:
        return 0.0
    return (price - entry) / risk if direction == "buy" else (entry - price) / risk


def is_improvement(direction: str, current_sl: float, candidate: float, atr: float, min_change_atr: float) -> bool:
    if not current_sl:  # no SL at all: anything is better
        return True
    delta = candidate - current_sl if direction == "buy" else current_sl - candidate
    return delta > 0 and delta >= min_change_atr * atr


def new_stop_loss(direction: str, entry: float, initial_sl: float, current_sl: float, price: float,
                  extreme: float, atr: float, spread: float, p: TrailParams) -> float | None:
    """Return a better SL, or None if it should stay where it is.

    `price` is the price the position would close at (bid for buys, ask for sells);
    `extreme` is the highest high (buy) / lowest low (sell) since entry.
    """
    buy = direction == "buy"
    r = profit_r(direction, entry, initial_sl, price)
    candidates = []
    if r >= p.breakeven_r:
        candidates.append(entry + spread if buy else entry - spread)
    if r >= p.trail_start_r and atr > 0:
        candidates.append(extreme - p.trail_atr_multiplier * atr if buy
                          else extreme + p.trail_atr_multiplier * atr)
    if not candidates:
        return None
    candidate = max(candidates) if buy else min(candidates)
    return candidate if is_improvement(direction, current_sl, candidate, atr, p.min_change_atr) else None


def partial_due(direction: str, entry: float, initial_sl: float, price: float, done: bool, p: TrailParams) -> bool:
    return p.partial_enabled and not done and profit_r(direction, entry, initial_sl, price) >= p.partial_at_r
