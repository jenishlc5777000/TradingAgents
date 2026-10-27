"""Trade levels for one signal: stop, targets, and the support/resistance around it.

One place computes these so the chart, the signal detail page and the Telegram alert can
never disagree about where the stop or a target sits.

Honesty about the targets matters here. The bot's default exit is the trailing stop, not a
fixed take-profit, so a target drawn on a chart is a *projection* unless the configuration
actually places that order:

* `take_profit_r` set            -> target 2 is a real order sent with the trade.
* `partial_close.enabled: true`  -> target 1 is where half the position is really closed.

When neither is configured the levels are still returned, because knowing where +1R and
+2R sit is useful, but `real` says False and callers label them as projections.
"""
from __future__ import annotations

import math

import pandas as pd

DEFAULT_T1_R = 1.5
DEFAULT_T2_R = 3.0


def _num(value):
    """Floats only, and never NaN - a NaN would silently draw a line at the wrong place."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def target_multiples(cfg: dict) -> tuple[float, float, bool, bool]:
    """(target1 R, target2 R, target1 is a real exit, target2 is a real exit)."""
    risk = cfg.get("risk") or {}
    partial = (cfg.get("trailing") or {}).get("partial_close") or {}
    tp_r = _num(risk.get("take_profit_r"))

    t1_real = bool(partial.get("enabled"))
    t1 = _num(partial.get("at_r")) if t1_real else _num(risk.get("target1_r")) or DEFAULT_T1_R
    t2_real = tp_r is not None
    t2 = tp_r if t2_real else (_num(risk.get("target2_r")) or DEFAULT_T2_R)
    # A first target beyond the second one would draw a nonsense ladder.
    if t1 and t2 and t1 > t2:
        t1, t2 = t2, t1
        t1_real, t2_real = t2_real, t1_real
    return float(t1 or DEFAULT_T1_R), float(t2 or DEFAULT_T2_R), t1_real, t2_real


def trade_levels(row: pd.Series, direction: str, cfg: dict, entry: float | None = None) -> dict:
    """Every level for a signal on `row`.

    `entry` defaults to the row's close; the live bot and the backtester pass the real fill
    once they have it, so the levels shown match the trade that was actually taken.
    """
    risk = cfg.get("risk") or {}
    atr = _num(row.get("atr"))
    price = _num(entry if entry is not None else row.get("close"))
    if atr is None or price is None:
        return {"entry": price, "sl": None, "tp1": None, "tp2": None, "risk_distance": None,
                "support": _num(row.get("support")), "resistance": _num(row.get("resistance")),
                "breakout_up": _num(row.get("dc_high")), "breakout_down": _num(row.get("dc_low")),
                "t1_r": None, "t2_r": None, "t1_real": False, "t2_real": False}

    dist = float(risk.get("sl_atr_multiplier", 1.5)) * atr
    buy = direction == "buy"
    t1_r, t2_r, t1_real, t2_real = target_multiples(cfg)

    return {
        "entry": price,
        "sl": price - dist if buy else price + dist,
        "risk_distance": dist,
        "tp1": price + t1_r * dist if buy else price - t1_r * dist,
        "tp2": price + t2_r * dist if buy else price - t2_r * dist,
        "t1_r": t1_r,
        "t2_r": t2_r,
        "t1_real": t1_real,
        "t2_real": t2_real,
        # Levels the market itself drew, independent of the trade.
        "support": _num(row.get("support")),
        "resistance": _num(row.get("resistance")),
        "breakout_up": _num(row.get("dc_high")),
        "breakout_down": _num(row.get("dc_low")),
    }
