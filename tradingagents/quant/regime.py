"""Market regime detector: decides which strategy family may trade on this candle."""
from __future__ import annotations

import pandas as pd

TREND = "trend"
RANGE = "range"
NEUTRAL = "neutral"


def detect_regime(row: pd.Series, cfg: dict) -> str:
    """ADX above `adx_trend` = trend, below `adx_range` = range, in between = no trading."""
    value = row.get("adx")
    if value is None or pd.isna(value):
        return NEUTRAL
    if value > cfg["adx_trend"]:
        return TREND
    if value < cfg["adx_range"]:
        return RANGE
    return NEUTRAL
