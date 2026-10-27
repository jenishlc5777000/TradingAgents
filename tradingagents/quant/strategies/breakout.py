"""Breakout: price closes beyond the range it has held for the last N bars.

The classic failure of a breakout system is buying every poke above the high in a dead
market, so two filters apply: the move must clear the channel by a fraction of ATR, and
volatility must not be collapsing. It trades in the trend and neutral regimes - a
breakout usually *begins* while ADX is still middling, which is exactly the band the
other strategies sit out.
"""
from __future__ import annotations

import pandas as pd

from ..regime import NEUTRAL, TREND
from .base import Signal, Strategy


class Breakout(Strategy):
    name = "breakout"
    regimes = (TREND, NEUTRAL)

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        row = df.iloc[-1]
        if not self._has(row, "dc_high", "dc_low", "atr", "atr_avg"):
            return None
        buffer = float(self.p.get("breakout_atr_buffer", 0.1)) * row["atr"]
        # A breakout into falling volatility tends to be the fake one.
        min_vol = float(self.p.get("min_atr_ratio", 0.8))
        if pd.notna(row["atr_avg"]) and row["atr_avg"] > 0 and row["atr"] / row["atr_avg"] < min_vol:
            return None
        align = bool(self.p.get("require_htf_alignment", False))
        htf_ok_up = not align or (self._has(row, "htf_ema_trend") and row["close"] > row["htf_ema_trend"])
        htf_ok_down = not align or (self._has(row, "htf_ema_trend") and row["close"] < row["htf_ema_trend"])

        if row["close"] > row["dc_high"] + buffer and htf_ok_up:
            return self._signal(symbol, row, "buy", regime,
                                f"broke above the {self.p.get('donchian_length', 20)}-bar high "
                                f"{row['dc_high']:.5g} (close {row['close']:.5g}), ADX {row['adx']:.1f}")
        if row["close"] < row["dc_low"] - buffer and htf_ok_down:
            return self._signal(symbol, row, "sell", regime,
                                f"broke below the {self.p.get('donchian_length', 20)}-bar low "
                                f"{row['dc_low']:.5g} (close {row['close']:.5g}), ADX {row['adx']:.1f}")
        return None
