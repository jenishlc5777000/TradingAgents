"""Strategy interface. A strategy only looks at a prepared DataFrame and returns a Signal or None."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar

import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str          # "buy" | "sell"
    strategy: str
    regime: str
    price: float            # close of the signal candle
    atr: float
    bar_time: pd.Timestamp  # open time of the signal candle (server time)
    reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bar_time"] = str(self.bar_time)
        return d


class Strategy(ABC):
    """Subclasses set `name` (the key under `strategies` in the settings) and `regimes` (the regimes they trade in).

    `check` receives a DataFrame whose last row is the candle that just closed, with the
    columns built by indicators.prepare(). Only the last three rows are guaranteed: every
    caller passes a short window, so look no further back than `df.iloc[-3]`.

    A strategy can trade in more than one regime. "neutral" (ADX between the two
    thresholds) is a real choice, not a gap: a breakout often starts there.
    `regimes: [trend, neutral]` in the quant settings overrides the class default per setup.
    """

    name: ClassVar[str]
    regimes: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: dict):
        self.p = params
        override = params.get("regimes")
        self.regimes = tuple(override) if override else self.regimes

    def trades_in(self, regime: str) -> bool:
        return regime in self.regimes

    @abstractmethod
    def check(self, symbol: str, df: pd.DataFrame, regime: str) -> Signal | None: ...

    def _signal(self, symbol: str, row: pd.Series, direction: str, regime: str, reason: str) -> Signal:
        return Signal(symbol=symbol, direction=direction, strategy=self.name, regime=regime,
                      price=float(row["close"]), atr=float(row["atr"]), bar_time=row["time"], reason=reason)

    @staticmethod
    def _has(row: pd.Series, *cols: str) -> bool:
        return all(pd.notna(row.get(c)) for c in cols)
