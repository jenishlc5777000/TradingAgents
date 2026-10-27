"""Squeeze: a quiet market that suddenly expands.

Bollinger width is compared with its own recent average, so "quiet" means quiet *for this
instrument* rather than a fixed number of points. When width has been below the threshold
and price then closes outside the band, the market is leaving its range.
"""
from __future__ import annotations

import pandas as pd

from ..regime import NEUTRAL, RANGE
from .base import Signal, Strategy


class SqueezeBreakout(Strategy):
    name = "squeeze_breakout"
    regimes = (RANGE, NEUTRAL)

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        if len(df) < 2:
            return None
        row, prev = df.iloc[-1], df.iloc[-2]
        if not self._has(row, "bb_upper", "bb_lower", "atr") or not self._has(prev, "bb_squeeze"):
            return None
        limit = float(self.p.get("squeeze_threshold", 0.85))
        # The squeeze must have been in place on the PREVIOUS bar; this bar is the escape.
        if not (pd.notna(prev["bb_squeeze"]) and prev["bb_squeeze"] < limit):
            return None

        if row["close"] > row["bb_upper"]:
            return self._signal(symbol, row, "buy", regime,
                                f"squeeze ({prev['bb_squeeze']:.2f} of normal width) broke up through "
                                f"the upper band {row['bb_upper']:.5g}")
        if row["close"] < row["bb_lower"]:
            return self._signal(symbol, row, "sell", regime,
                                f"squeeze ({prev['bb_squeeze']:.2f} of normal width) broke down through "
                                f"the lower band {row['bb_lower']:.5g}")
        return None
