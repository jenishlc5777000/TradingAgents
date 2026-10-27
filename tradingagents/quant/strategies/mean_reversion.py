"""Mean reversion: fade RSI extremes at the Bollinger Bands in a ranging market."""
from __future__ import annotations

import pandas as pd

from ..regime import RANGE
from .base import Signal, Strategy


class MeanReversion(Strategy):
    name = "mean_reversion"
    regimes = (RANGE,)   # only when ADX says the market is ranging

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        row = df.iloc[-1]
        if not self._has(row, "rsi", "bb_lower", "bb_upper", "atr"):
            return None
        rsi_buy = float(self.p.get("rsi_buy", 30))
        rsi_sell = float(self.p.get("rsi_sell", 70))

        if row["rsi"] < rsi_buy and row["low"] <= row["bb_lower"]:
            return self._signal(symbol, row, "buy", regime,
                                f"RSI {row['rsi']:.1f} < {rsi_buy:g} at lower BB {row['bb_lower']:.5g}, "
                                f"ADX {row['adx']:.1f}")
        if row["rsi"] > rsi_sell and row["high"] >= row["bb_upper"]:
            return self._signal(symbol, row, "sell", regime,
                                f"RSI {row['rsi']:.1f} > {rsi_sell:g} at upper BB {row['bb_upper']:.5g}, "
                                f"ADX {row['adx']:.1f}")
        return None
