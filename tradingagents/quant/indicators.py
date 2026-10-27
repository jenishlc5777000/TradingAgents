"""Indicator maths shared by the live scan and the backtester.

Plain pandas, so no extra dependency. Every column is computed from bars up to and
including its own row, except where noted (Donchian and swing levels are shifted so a
bar never sees its own high or a pivot the market had not yet confirmed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Seconds per bar for each timeframe name the settings accept.
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    gain, loss = _rma(d.clip(lower=0), n), _rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift()
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return _rma(_true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    a = atr(high, low, close, n)
    plus_di = 100 * _rma(plus_dm, n) / a
    minus_di = 100 * _rma(minus_dm, n) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _rma(dx, n)


def bbands(close: pd.Series, n: int, k: float):
    """(lower, middle, upper)."""
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    """(macd line, signal line, histogram)."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int, smooth: int) -> pd.Series:
    """Stochastic %K, smoothed. 0 = at the n-bar low, 100 = at the n-bar high."""
    hh, ll = high.rolling(n).max(), low.rolling(n).min()
    raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return raw.rolling(smooth).mean()


def donchian(high: pd.Series, low: pd.Series, n: int):
    """Highest high / lowest low of the `n` bars BEFORE each bar.

    Shifted by one on purpose: a bar may not use its own high to decide whether it broke
    out of a range, or every bar trivially touches its own channel.
    """
    return high.rolling(n).max().shift(1), low.rolling(n).min().shift(1)


def swing_levels(high: pd.Series, low: pd.Series, left: int, right: int):
    """Confirmed swing support/resistance: a pivot with `left` bars each side.

    Carried forward only from the bar where the pivot becomes known (pivot + right bars),
    so nothing here reveals a level the market had not yet confirmed.
    """
    span = left + right + 1
    # Not rolling(center=True): pandas splits an even-sized window asymmetrically, which
    # looks further ahead than `right` bars. rolling(span).max() at k covers
    # [k-span+1, k]; shifting by -right aligns it to [j-left, j+right] at j.
    win_high = high.rolling(span).max().shift(-right)
    win_low = low.rolling(span).min().shift(-right)
    ph, pl = high.eq(win_high), low.eq(win_low)
    # Publish the level only once the confirming bars exist, i.e. `right` bars later.
    res = high.where(ph).shift(right).ffill()
    sup = low.where(pl).shift(right).ffill()
    return sup, res


def add_entry_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], p["ema_fast"])
    df["atr"] = atr(df["high"], df["low"], df["close"], p["atr_length"])
    df["atr_avg"] = df["atr"].rolling(p["atr_avg_length"]).mean()
    df["adx"] = adx(df["high"], df["low"], df["close"], p["adx_length"])
    df["rsi"] = rsi(df["close"], p["rsi_length"])
    df["bb_lower"], df["bb_mid"], df["bb_upper"] = bbands(df["close"], p["bb_length"], float(p["bb_std"]))
    df["ema_slow"] = ema(df["close"], p.get("ema_slow", 50))
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(
        df["close"], p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_signal", 9))
    df["stoch"] = stoch_k(df["high"], df["low"], df["close"], p.get("stoch_length", 14), p.get("stoch_smooth", 3))
    df["dc_high"], df["dc_low"] = donchian(df["high"], df["low"], p.get("donchian_length", 20))
    # Band width against its own recent average: below 1 means an unusually quiet (squeezed) market.
    width = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_width"] = width
    df["bb_squeeze"] = width / width.rolling(p.get("squeeze_lookback", 50)).mean()
    df["support"], df["resistance"] = swing_levels(
        df["high"], df["low"], p.get("swing_left", 10), p.get("swing_right", 3))
    return df


def add_trend_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["ema_trend"] = ema(df["close"], p["ema_trend"])
    return df


def align_htf(entry: pd.DataFrame, trend: pd.DataFrame, entry_tf: str, trend_tf: str) -> pd.DataFrame:
    """Attach the latest *closed* higher-timeframe bar to each entry bar (no lookahead).

    Bar time is the open time, so match on close times: an H1 bar closing at 12:00 sees
    the D1 bar that closed at or before 12:00, never the one still forming.
    """
    e = entry.copy()
    e["close_time"] = e["time"] + pd.Timedelta(seconds=TF_SECONDS[entry_tf])
    t = trend[["time", "close", "ema_trend"]].copy()
    t["close_time"] = t["time"] + pd.Timedelta(seconds=TF_SECONDS[trend_tf])
    t = t.rename(columns={"close": "htf_close", "ema_trend": "htf_ema_trend"}).drop(columns="time")
    out = pd.merge_asof(e.sort_values("close_time"), t.sort_values("close_time"),
                        on="close_time", direction="backward")
    return out.reset_index(drop=True)


def prepare(entry_raw: pd.DataFrame, trend_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Entry-timeframe indicators plus the aligned higher-timeframe trend columns."""
    p, tf = cfg["indicators"], cfg["timeframes"]
    return align_htf(add_entry_indicators(entry_raw, p), add_trend_indicators(trend_raw, p),
                     tf["entry"], tf["trend"])
