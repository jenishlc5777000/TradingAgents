"""Reversal: stochastic turns back out of an extreme, near a confirmed swing level.

Mean reversion already fades RSI at the Bollinger Bands; this one waits for the turn to
actually happen (stochastic crossing back out of oversold) and wants price to be near a
level the market has respected before, which keeps it out of the middle of the range.
"""
from __future__ import annotations

import pandas as pd

from ..regime import NEUTRAL, RANGE
from .base import Signal, Strategy


class StochReversal(Strategy):
    name = "stoch_reversal"
    regimes = (RANGE, NEUTRAL)

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        if len(df) < 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        if not self._has(row, "stoch", "atr") or not self._has(prev, "stoch"):
            return None
        low_level = float(self.p.get("oversold", 20))
        high_level = float(self.p.get("overbought", 80))
        near = float(self.p.get("level_atr_distance", 1.0)) * row["atr"]
        use_levels = bool(self.p.get("require_level", True))

        at_support = not use_levels or (
            self._has(row, "support") and abs(row["low"] - row["support"]) <= near)
        at_resistance = not use_levels or (
            self._has(row, "resistance") and abs(row["high"] - row["resistance"]) <= near)

        if prev["stoch"] < low_level <= row["stoch"] and at_support:
            where = f" near support {row['support']:.5g}" if use_levels and self._has(row, "support") else ""
            return self._signal(symbol, row, "buy", regime,
                                f"stochastic turned up out of oversold "
                                f"({prev['stoch']:.0f} to {row['stoch']:.0f}){where}")
        if prev["stoch"] > high_level >= row["stoch"] and at_resistance:
            where = f" near resistance {row['resistance']:.5g}" if use_levels and self._has(row, "resistance") else ""
            return self._signal(symbol, row, "sell", regime,
                                f"stochastic turned down out of overbought "
                                f"({prev['stoch']:.0f} to {row['stoch']:.0f}){where}")
        return None
