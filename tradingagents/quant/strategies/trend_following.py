"""Trend following: trade pullbacks to EMA20 in the direction of the higher-timeframe EMA200."""
from __future__ import annotations

import pandas as pd

from ..regime import TREND
from .base import Signal, Strategy


class TrendFollowing(Strategy):
    name = "trend_following"
    regimes = (TREND,)   # only when ADX says the market is trending

    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None:
        row = df.iloc[-1]
        if not self._has(row, "ema_fast", "atr", "htf_close", "htf_ema_trend"):
            return None
        tol = float(self.p.get("pullback_atr_tolerance", 0.2)) * row["atr"]
        confirm = bool(self.p.get("require_confirmation_candle", True))
        ema_fast, htf_ema = row["ema_fast"], row["htf_ema_trend"]

        uptrend = row["htf_close"] > htf_ema and row["close"] > htf_ema
        downtrend = row["htf_close"] < htf_ema and row["close"] < htf_ema

        if uptrend and row["low"] <= ema_fast + tol and row["close"] > ema_fast \
                and (not confirm or row["close"] > row["open"]):
            return self._signal(symbol, row, "buy", regime,
                                f"HTF uptrend (close {row['htf_close']:.5g} > EMA200 {htf_ema:.5g}); "
                                f"pullback to EMA20 {ema_fast:.5g}, ADX {row['adx']:.1f}")
        if downtrend and row["high"] >= ema_fast - tol and row["close"] < ema_fast \
                and (not confirm or row["close"] < row["open"]):
            return self._signal(symbol, row, "sell", regime,
                                f"HTF downtrend (close {row['htf_close']:.5g} < EMA200 {htf_ema:.5g}); "
                                f"pullback to EMA20 {ema_fast:.5g}, ADX {row['adx']:.1f}")
        return None
