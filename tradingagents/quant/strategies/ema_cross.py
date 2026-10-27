"""Momentum: the fast EMA crosses the slow EMA with MACD agreeing.

A cross on its own fires constantly in chop, so MACD must confirm the same direction.
Optionally the higher timeframe must agree too, which cuts the trade count roughly in
half and raises the win rate - worth testing both ways in the optimizer.
"""
from __future__ import annotations

import pandas as pd

from ..regime import NEUTRAL, TREND
from .base import Signal, Strategy


class EmaCross(Strategy):
    name = "ema_cross"
    regimes = (TREND, NEUTRAL)

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        if len(df) < 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        if not self._has(row, "ema_fast", "ema_slow", "macd_hist", "atr") \
                or not self._has(prev, "ema_fast", "ema_slow"):
            return None
        confirm = bool(self.p.get("require_macd", True))
        align = bool(self.p.get("require_htf_alignment", True))

        crossed_up = prev["ema_fast"] <= prev["ema_slow"] and row["ema_fast"] > row["ema_slow"]
        crossed_down = prev["ema_fast"] >= prev["ema_slow"] and row["ema_fast"] < row["ema_slow"]
        htf_up = not align or (self._has(row, "htf_ema_trend") and row["close"] > row["htf_ema_trend"])
        htf_down = not align or (self._has(row, "htf_ema_trend") and row["close"] < row["htf_ema_trend"])

        if crossed_up and htf_up and (not confirm or row["macd_hist"] > 0):
            return self._signal(symbol, row, "buy", regime,
                                f"EMA{self.p.get('ema_fast_len', 20)} crossed above "
                                f"EMA{self.p.get('ema_slow_len', 50)} at {row['ema_fast']:.5g}, "
                                f"MACD hist {row['macd_hist']:+.5g}")
        if crossed_down and htf_down and (not confirm or row["macd_hist"] < 0):
            return self._signal(symbol, row, "sell", regime,
                                f"EMA{self.p.get('ema_fast_len', 20)} crossed below "
                                f"EMA{self.p.get('ema_slow_len', 50)} at {row['ema_fast']:.5g}, "
                                f"MACD hist {row['macd_hist']:+.5g}")
        return None
