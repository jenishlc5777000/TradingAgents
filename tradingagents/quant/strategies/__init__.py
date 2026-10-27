"""Strategy registry. To add a strategy: create a Strategy subclass in this folder,
add it to REGISTRY, and give it a block (with `enabled: true`) under `strategies` in the quant settings (settings.py)."""
from __future__ import annotations

import pandas as pd

from ..regime import detect_regime
from .base import Signal, Strategy
from .breakout import Breakout
from .ema_cross import EmaCross
from .mean_reversion import MeanReversion
from .squeeze_breakout import SqueezeBreakout
from .stoch_reversal import StochReversal
from .trend_following import TrendFollowing

REGISTRY: dict[str, type[Strategy]] = {cls.name: cls for cls in (
    TrendFollowing, MeanReversion, Breakout, EmaCross, SqueezeBreakout, StochReversal)}

__all__ = ["Signal", "Strategy", "REGISTRY", "catalog", "load_strategies", "evaluate"]


def catalog() -> list[dict]:
    """What each strategy is and when it fires, for the dashboard's on/off list.

    The description is the strategy module's own docstring, so there is one place to edit.
    """
    import sys

    out = []
    for name, cls in REGISTRY.items():
        doc = (sys.modules[cls.__module__].__doc__ or "").strip()
        head, _, rest = doc.partition("\n")
        title, _, summary = head.partition(":")
        out.append({
            "name": name,
            "title": title.strip() or name.replace("_", " ").capitalize(),
            "summary": summary.strip().rstrip(".") or head,
            "detail": " ".join(rest.split()),
            "regimes": list(cls.regimes),
        })
    return out


def load_strategies(cfg: dict) -> list[Strategy]:
    out = []
    for name, params in (cfg.get("strategies") or {}).items():
        if not params or not params.get("enabled", False):
            continue
        if name not in REGISTRY:
            raise ValueError(f"Unknown strategy '{name}' in the quant settings (known: {', '.join(REGISTRY)})")
        out.append(REGISTRY[name](params))
    return out


def evaluate(symbol: str, df: pd.DataFrame, strategies: list[Strategy],
             regime_cfg: dict) -> tuple[str, Signal | None]:
    """Detect the regime on the last closed candle and run the strategies built for it.

    The first strategy to fire wins, so the order in the quant settings is the priority order.
    """
    regime = detect_regime(df.iloc[-1], regime_cfg)
    for strategy in strategies:
        if not strategy.trades_in(regime):
            continue
        signal = strategy.check(symbol, df, regime)
        if signal is not None:
            return regime, signal
    return regime, None
