"""Intraday and daily bars for the quant engine, in the shape its indicators read.

``load_ohlcv`` serves daily closes to the analysts; the strategy engine needs bars of
a chosen timeframe with the open time as the index of each row. Yahoo keeps only a
limited history for fast timeframes, so the window asked for is clamped to what it
holds and the caller is told what was actually covered.
"""

import logging

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry

logger = logging.getLogger(__name__)

# timeframe -> (yfinance interval, longest history Yahoo serves in days, resample rule)
_TIMEFRAMES = {
    "M1": ("1m", 7, None),
    "M5": ("5m", 59, None),
    "M15": ("15m", 59, None),
    "M30": ("30m", 59, None),
    "H1": ("1h", 729, None),
    "H4": ("1h", 729, "4h"),
    "D1": ("1d", 36500, None),
}


_BAR_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}


def max_history_days(timeframe: str) -> int:
    """How many days back Yahoo serves bars of this timeframe."""
    return _TIMEFRAMES[timeframe][1]


def load_bars(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Bars with columns time, open, high, low, close, volume; ``time`` is each bar's open.

    ``start`` is moved forward when Yahoo holds less history than asked for; compare
    the first row with ``start`` to see it. Raises NoMarketDataError when nothing comes back.
    """
    if timeframe not in _TIMEFRAMES:
        raise ValueError(f"unknown timeframe {timeframe!r}; use one of {', '.join(_TIMEFRAMES)}")
    interval, days, rule = _TIMEFRAMES[timeframe]
    canonical = normalize_symbol(symbol)
    now = pd.Timestamp.now().normalize()
    first = max(pd.Timestamp(start).normalize(), now - pd.Timedelta(days=days))
    last = min(pd.Timestamp(end).normalize(), now) + pd.Timedelta(days=1)  # Yahoo's end is exclusive
    if last <= first:
        raise NoMarketDataError(symbol, canonical, f"no {timeframe} history reaches back to {start:%Y-%m-%d}")

    raw = yf_retry(lambda: yf.Ticker(canonical).history(
        start=first.strftime("%Y-%m-%d"), end=last.strftime("%Y-%m-%d"),
        interval=interval, auto_adjust=True, actions=False))
    if raw is None or raw.empty:
        raise_for_empty(symbol, canonical, f"{timeframe} bars")

    df = raw.reset_index()
    df = df.rename(columns={df.columns[0]: "time", "Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    times = pd.to_datetime(df["time"])
    df["time"] = times.dt.tz_localize(None) if times.dt.tz is not None else times
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])

    if rule:
        df = (df.set_index("time").resample(rule, origin="start_day")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
              .dropna(subset=["open"]).reset_index())
    # Yahoo marks the forming bar with a partial close; a closed-bar engine must not trade it.
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if len(df) and timeframe != "D1":
        if df["time"].iloc[-1] + pd.Timedelta(seconds=_BAR_SECONDS[timeframe]) > pd.Timestamp.now():
            df = df.iloc[:-1].reset_index(drop=True)
    elif len(df) and df["time"].iloc[-1].normalize() >= now:
        df = df.iloc[:-1].reset_index(drop=True)
    if df.empty:
        raise NoMarketDataError(symbol, canonical, f"no closed {timeframe} bars in range")
    return df
