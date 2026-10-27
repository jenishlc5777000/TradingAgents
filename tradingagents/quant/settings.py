"""Quant settings: one nested dict, saved to ``~/.tradingagents/quant_settings.json``.

The sections match the trading bot this engine came from, so a setup tuned there
carries over key for key: ``timeframes``, ``indicators``, ``regime``, ``strategies``,
``risk`` (with ``martingale``), ``trailing`` and ``backtest``. ``costs`` is new: a bar
feed from Yahoo has no quoted spread, so the cost of crossing it is a share of price.

Anything not saved falls back to ``DEFAULTS``, so a settings file written by an
older version keeps working when a key is added.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from tradingagents.dataflows.files import replace_file
from tradingagents.quant.indicators import TF_SECONDS

SETTINGS_PATH = Path(os.path.expanduser("~")) / ".tradingagents" / "quant_settings.json"

DEFAULTS: dict = {
    # Entry bars decide when to trade; the trend timeframe supplies the EMA200 filter.
    # H4 is built from H1 bars, as Yahoo serves no 4-hour candles.
    "timeframes": {"entry": "H1", "trend": "D1"},
    "indicators": {
        "ema_fast": 20,
        "ema_trend": 200,           # computed on the trend timeframe
        "atr_length": 14,
        "atr_avg_length": 50,       # baseline for "abnormal volatility" context
        "adx_length": 14,
        "rsi_length": 14,
        "bb_length": 20,
        "bb_std": 2.0,
        "ema_slow": 50,             # momentum cross: the slow leg
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_length": 14,
        "stoch_smooth": 3,
        "donchian_length": 20,      # breakout channel: the range price must clear
        "squeeze_lookback": 50,     # band width is compared with its own average over this many bars
        "swing_left": 10,           # support/resistance pivot: bars required either side
        "swing_right": 3,
    },
    # The regime detector picks which strategies may fire on a bar.
    "regime": {"adx_trend": 25, "adx_range": 20},
    # Order is priority: the first strategy to fire on a bar wins.
    "strategies": {
        "trend_following": {"enabled": True, "pullback_atr_tolerance": 0.2,
                            "require_confirmation_candle": True},
        "mean_reversion": {"enabled": True, "rsi_buy": 30, "rsi_sell": 70},
        "breakout": {"enabled": True, "breakout_atr_buffer": 0.1, "min_atr_ratio": 0.8,
                     "require_htf_alignment": False},
        "ema_cross": {"enabled": True, "require_macd": True, "require_htf_alignment": True},
        "squeeze_breakout": {"enabled": True, "squeeze_threshold": 0.85},
        "stoch_reversal": {"enabled": True, "oversold": 20, "overbought": 80,
                           "require_level": True, "level_atr_distance": 1.0},
    },
    "risk": {
        # money = a fixed cash amount per trade; percent = a share of the balance.
        "risk_mode": "money",
        "risk_per_trade_money": 10.0,
        "risk_per_trade_pct": 1.0,
        # Size from the balance you have now (true) or always the starting balance (false).
        "compounding": True,
        # DANGEROUS: raises the stake after a loss to chase it back. Off by default; max_steps
        # and the max_risk caps are hard limits.
        "martingale": {"enabled": False, "mode": "after_loss", "multiplier": 2.0, "max_steps": 3,
                       "max_risk_money": 60.0, "max_risk_pct": 6.0, "reset_on_win": True},
        "sl_atr_multiplier": 1.5,
        "take_profit_r": None,      # null = no fixed target, exit through the trailing stop
        "target1_r": 1.5,           # projections drawn in a scan when no real target is set
        "target2_r": 3.0,
        "max_daily_loss_money": 30.0,
        "max_daily_loss_pct": 3.0,
        "max_trades_per_day": 5,    # 0 = no cap
        "max_open_trades": 2,
        "max_trades_per_symbol": 1,
    },
    "trailing": {
        "breakeven_r": 1.0,         # at +1R: stop moves to entry plus the spread
        "trail_start_r": 1.5,       # after +1.5R: stop trails behind the extreme
        "trail_atr_multiplier": 2.0,
        "min_change_atr": 0.05,     # only move the stop by at least this much ATR
        "partial_close": {"enabled": False, "at_r": 2.0, "fraction": 0.5},
    },
    # Cost of crossing the spread, as a percent of the bar's open. Yahoo quotes no spread.
    "costs": {"spread_pct": 0.02},
    "backtest": {
        "start": "2024-01-01",
        "end": None,                # None = today
        "in_sample_ratio": 0.7,     # first 70% of the period is in-sample, the rest out-of-sample
        "initial_balance": 1000.0,
    },
}

# Named timeframe pairs for `--preset`. Yahoo keeps about 60 days of 5/15-minute bars and
# about two years of hourly bars, so the faster presets can only look back that far.
TIMEFRAME_PRESETS = {
    "scalp": {"entry": "M5", "trend": "H1"},
    "intraday": {"entry": "M15", "trend": "H4"},
    "swing": {"entry": "H1", "trend": "D1"},
    "position": {"entry": "D1", "trend": "D1"},
}


def merge(base: dict, override: dict | None) -> dict:
    """``base`` with ``override`` laid over it, key by key through nested dicts."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def set_path(settings: dict, path: str, value) -> None:
    """Set ``a.b.c`` inside ``settings``, creating the nodes on the way."""
    keys = path.split(".")
    node = settings
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def get_path(settings: dict, path: str):
    node = settings
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(path)
        node = node[key]
    return node


def parse_value(text: str):
    """A command-line value as JSON when it is one (1.5, true, null), else the string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def apply_overrides(settings: dict, overrides: dict | None) -> dict:
    """A copy of ``settings`` with dotted-path overrides applied, e.g. {"risk.compounding": False}."""
    out = copy.deepcopy(settings)
    for path, value in (overrides or {}).items():
        set_path(out, path, value)
    return out


def validate(settings: dict) -> dict:
    """Return ``settings`` unchanged when usable, else raise ValueError saying what is wrong."""
    from tradingagents.quant.strategies import REGISTRY

    tf = settings["timeframes"]
    for role in ("entry", "trend"):
        if tf.get(role) not in TF_SECONDS:
            raise ValueError(f"timeframes.{role} must be one of {', '.join(TF_SECONDS)}, got {tf.get(role)!r}")
    if TF_SECONDS[tf["trend"]] < TF_SECONDS[tf["entry"]]:
        raise ValueError("timeframes.trend must be the same as or slower than timeframes.entry")
    unknown = sorted(set(settings["strategies"]) - set(REGISTRY))
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; known: {', '.join(REGISTRY)}")
    risk = settings["risk"]
    if risk["risk_mode"] not in ("money", "percent"):
        raise ValueError("risk.risk_mode must be 'money' or 'percent'")
    if not 0 < float(risk["risk_per_trade_pct"]) <= 5:
        raise ValueError("risk.risk_per_trade_pct must be in (0, 5]")
    if float(risk["risk_per_trade_money"]) <= 0:
        raise ValueError("risk.risk_per_trade_money must be > 0")
    if float(risk["sl_atr_multiplier"]) <= 0:
        raise ValueError("risk.sl_atr_multiplier must be > 0")
    if risk.get("take_profit_r") is not None and float(risk["take_profit_r"]) <= 0:
        raise ValueError("risk.take_profit_r must be > 0, or null for no fixed target")
    if not 0 < float(settings["backtest"]["in_sample_ratio"]) < 1:
        raise ValueError("backtest.in_sample_ratio must be between 0 and 1")
    if float(settings["backtest"]["initial_balance"]) <= 0:
        raise ValueError("backtest.initial_balance must be > 0")
    if float(settings["regime"]["adx_range"]) > float(settings["regime"]["adx_trend"]):
        raise ValueError("regime.adx_range must not exceed regime.adx_trend")
    return settings


def load_settings(path: str | os.PathLike | None = None) -> dict:
    """The saved settings over the defaults. A missing or unreadable file means defaults."""
    target = Path(path) if path else SETTINGS_PATH
    try:
        saved = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            saved = {}
    except (OSError, ValueError):
        saved = {}
    return merge(DEFAULTS, saved)


def save_settings(settings: dict, path: str | os.PathLike | None = None) -> Path:
    """Validate and write ``settings``; a rejected value never reaches the file."""
    validate(settings)
    target = Path(path) if path else SETTINGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_file(target, lambda temp: Path(temp).write_text(json.dumps(settings, indent=2), encoding="utf-8"))
    return target


def reset_settings(path: str | os.PathLike | None = None) -> dict:
    (Path(path) if path else SETTINGS_PATH).unlink(missing_ok=True)
    return copy.deepcopy(DEFAULTS)


def enabled_strategies(settings: dict) -> list[str]:
    return [name for name, params in settings["strategies"].items() if (params or {}).get("enabled")]


def engine_config(settings: dict) -> dict:
    """The shape the indicator, strategy and backtest code reads."""
    return {key: settings[key] for key in ("timeframes", "indicators", "regime", "strategies",
                                           "risk", "trailing", "costs")}
