"""Replay the rule-based strategies on history: entries, stops, trailing, account rules.

Modelling assumptions (the same ones the live scan relies on):

* A signal forms on a bar's close and fills at the next bar's open, plus the spread for buys.
* The stop is checked inside the bar before anything else (pessimistic), and a gap through
  it fills at the open. The trailing stop is recomputed once per bar, at the close.
* P&L is measured in R, the multiple of the money risked. Money is R times the stake sized
  by ``tradingagents.quant.sizing``. Lot or share rounding is ignored.
* The LLM agents are not consulted: a model that has read the history may know how it ended.

This simulates a portfolio, which ``tradingagents.backtest`` deliberately does not: that
module scores LLM decisions, this one scores rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tradingagents.quant.indicators import TF_SECONDS, prepare
from tradingagents.quant.settings import engine_config
from tradingagents.quant.sizing import SizingState, after_trade, daily_loss_limit, risk_amount
from tradingagents.quant.strategies import evaluate, load_strategies
from tradingagents.quant.trailing import TrailParams, new_stop_loss, profit_r

BarsLoader = Callable[[str, str, pd.Timestamp, pd.Timestamp], pd.DataFrame]


@dataclass
class BtTrade:
    symbol: str
    direction: str
    strategy: str
    regime: str
    entry_time: pd.Timestamp
    entry: float
    initial_sl: float
    sl: float
    extreme: float
    tp: float | None = None
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    exit_reason: str = ""
    r: float = 0.0
    partial_done: bool = False
    partial_r: float = 0.0
    pnl: float = 0.0


def warmup_days(settings: dict) -> int:
    """Calendar days of history before the start, so the trend EMA is valid on day one."""
    bars = int(settings["indicators"]["ema_trend"]) + 10
    return max(30, int(bars * TF_SECONDS[settings["timeframes"]["trend"]] / 86400 * 1.6) + 5)


# --------------------------------------------------------------------------- simulation
def simulate_symbol(symbol: str, df: pd.DataFrame, cfg: dict, start: pd.Timestamp) -> list[BtTrade]:
    """Every trade the enabled strategies would have taken on one prepared frame."""
    strategies = load_strategies(cfg)
    p = TrailParams.from_config(cfg["trailing"])
    sl_mult = float(cfg["risk"]["sl_atr_multiplier"])
    tp_r = cfg["risk"].get("take_profit_r")
    spread_frac = float(cfg.get("costs", {}).get("spread_pct", 0.0)) / 100

    t = df["time"].to_numpy()
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    spr = o * spread_frac
    atr = df["atr"].to_numpy(dtype=float)
    first = int(np.searchsorted(t, np.datetime64(start)))

    trades: list[BtTrade] = []
    pos: BtTrade | None = None
    pending = None
    for i in range(max(first, 3), len(df)):
        # 1) fill a signal from the previous close at this bar's open
        if pending is not None:
            sig, pending = pending, None
            buy = sig.direction == "buy"
            entry = o[i] + spr[i] if buy else o[i]
            dist = sl_mult * sig.atr
            sl = entry - dist if buy else entry + dist
            tp = (entry + tp_r * dist if buy else entry - tp_r * dist) if tp_r else None
            pos = BtTrade(symbol, sig.direction, sig.strategy, sig.regime, pd.Timestamp(t[i]), entry, sl, sl,
                          extreme=o[i], tp=tp)

        # 2) manage the open position through this bar
        if pos is not None:
            buy = pos.direction == "buy"
            exit_price, reason = None, ""
            if buy and l[i] <= pos.sl:
                exit_price, reason = min(o[i], pos.sl), "sl"
            elif not buy and h[i] + spr[i] >= pos.sl:
                exit_price, reason = max(o[i] + spr[i], pos.sl), "sl"
            elif pos.tp is not None and buy and h[i] >= pos.tp:
                exit_price, reason = max(o[i], pos.tp), "tp"
            elif pos.tp is not None and not buy and l[i] + spr[i] <= pos.tp:
                exit_price, reason = min(o[i] + spr[i], pos.tp), "tp"

            if exit_price is not None:
                _finish(pos, pd.Timestamp(t[i]), exit_price, reason, p)
                trades.append(pos)
                pos = None
            else:
                risk = abs(pos.entry - pos.initial_sl)
                if p.partial_enabled and not pos.partial_done:
                    target = pos.entry + p.partial_at_r * risk if buy else pos.entry - p.partial_at_r * risk
                    if (buy and h[i] >= target) or (not buy and l[i] + spr[i] <= target):
                        pos.partial_done = True
                        pos.partial_r = p.partial_fraction * p.partial_at_r
                pos.extreme = max(pos.extreme, h[i]) if buy else min(pos.extreme, l[i])
                price = c[i] if buy else c[i] + spr[i]
                new_sl = new_stop_loss(pos.direction, pos.entry, pos.initial_sl, pos.sl, price, pos.extreme,
                                       atr[i], spr[i], p)
                if new_sl is not None:
                    pos.sl = new_sl

        # 3) look for a new signal on this bar's close
        if pos is None and pending is None and i + 1 < len(df):
            _, sig = evaluate(symbol, df.iloc[i - 2:i + 1], strategies, cfg["regime"])
            if sig is not None and not math.isnan(sig.atr):
                pending = sig

    if pos is not None:
        _finish(pos, pd.Timestamp(t[-1]), c[-1], "end_of_data", p)
        trades.append(pos)
    return trades


def _finish(pos: BtTrade, when, price: float, reason: str, p: TrailParams) -> None:
    pos.exit_time, pos.exit, pos.exit_reason = when, price, reason
    remaining = 1 - p.partial_fraction if pos.partial_done else 1.0
    pos.r = pos.partial_r + remaining * profit_r(pos.direction, pos.entry, pos.initial_sl, price)


# --------------------------------------------------------------------------- portfolio
def run_portfolio(trades: list[BtTrade], cfg: dict, balance0: float):
    """Apply account-level rules across symbols: max open trades, daily loss stop, compounding risk.

    Returns the trades that were taken, with ``pnl`` filled in, and the equity curve as
    ``(time, balance)`` points.
    """
    rc = cfg["risk"]
    max_open = int(rc["max_open_trades"])
    max_per_symbol = int(rc.get("max_trades_per_symbol", 0) or 0)
    max_per_day = int(rc.get("max_trades_per_day", 0) or 0)
    state = SizingState()
    balance = balance0
    open_: list[tuple[BtTrade, float]] = []
    taken: list[BtTrade] = []
    day_start: dict = {}
    day_pnl: dict = {}
    day_count: dict = {}
    curve = [(min(t.entry_time for t in trades) if trades else pd.Timestamp.now(), balance0)]

    def realize_until(ts):
        nonlocal balance, state
        open_.sort(key=lambda x: x[0].exit_time)
        while open_ and open_[0][0].exit_time <= ts:
            tr, risk_money = open_.pop(0)
            d = tr.exit_time.date()
            day_start.setdefault(d, balance)
            tr.pnl = risk_money * tr.r
            balance += tr.pnl
            state = after_trade(rc, state, tr.r)
            day_pnl[d] = day_pnl.get(d, 0.0) + tr.pnl
            curve.append((tr.exit_time, balance))

    for tr in sorted(trades, key=lambda x: x.entry_time):
        realize_until(tr.entry_time)
        d = tr.entry_time.date()
        day_start.setdefault(d, balance)
        limit, _ = daily_loss_limit(rc, day_start[d])
        if limit > 0 and -day_pnl.get(d, 0.0) >= limit:
            continue
        if len(open_) >= max_open:
            continue
        if max_per_symbol and sum(1 for o, _ in open_ if o.symbol == tr.symbol) >= max_per_symbol:
            continue
        if max_per_day and day_count.get(d, 0) >= max_per_day:
            continue
        risk_money, _ = risk_amount(rc, balance, balance0, state)
        day_count[d] = day_count.get(d, 0) + 1
        open_.append((tr, risk_money))
        taken.append(tr)
    realize_until(pd.Timestamp.max)
    return taken, curve


def metrics(taken: list[BtTrade], curve, balance0: float) -> dict:
    if not taken:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in taken])
    rs = np.array([t.r for t in taken])
    gross_win, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = np.array([b for _, b in curve])
    peak = np.maximum.accumulate(eq)
    return {
        "trades": len(taken),
        "win_rate_pct": round(100 * float((pnl > 0).mean()), 1),
        "profit_factor": round(float(gross_win / gross_loss), 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(100 * float(((eq - peak) / peak).min()), 2),
        "avg_r": round(float(rs.mean()), 3),
        "total_r": round(float(rs.sum()), 2),
        "net_profit": round(float(eq[-1] - balance0), 2),
        "return_pct": round(100 * float(eq[-1] / balance0 - 1), 2),
    }


def by_strategy(taken: list[BtTrade]) -> list[dict]:
    """Per symbol and strategy: trades, win rate, average and total R."""
    if not taken:
        return []
    df = pd.DataFrame([asdict(t) for t in taken])
    g = df.groupby(["symbol", "strategy"])["r"]
    out = pd.DataFrame({"trades": g.size(), "win_rate_pct": g.apply(lambda s: (s > 0).mean() * 100).round(1),
                        "avg_r": g.mean().round(3), "total_r": g.sum().round(2)})
    return [{"symbol": idx[0], "strategy": idx[1], **{k: float(v) for k, v in row.items()}}
            for idx, row in out.iterrows()]


# --------------------------------------------------------------------------- driver
def default_loader(symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from tradingagents.dataflows.vendors.yahoo.bars import load_bars
    return load_bars(symbol, timeframe, start, end)


def load_history(symbols: list[str], settings: dict, start: pd.Timestamp, end: pd.Timestamp,
                 loader: BarsLoader | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Entry and trend bars for each symbol, with the warm-up before ``start``."""
    loader = loader or default_loader
    tf = settings["timeframes"]
    fetch_start = start - pd.Timedelta(days=warmup_days(settings))
    out = {}
    for symbol in symbols:
        entry = loader(symbol, tf["entry"], fetch_start, end)
        trend = entry if tf["trend"] == tf["entry"] else loader(symbol, tf["trend"], fetch_start, end)
        out[symbol] = (entry, trend)
    return out


def simulate_all(raw: dict, settings: dict, start: pd.Timestamp) -> tuple[list[BtTrade], dict]:
    cfg = engine_config(settings)
    trades: list[BtTrade] = []
    for symbol, (entry, trend) in raw.items():
        trades += simulate_symbol(symbol, prepare(entry, trend, cfg), cfg, start)
    return trades, cfg


def split_point(raw: dict, start: pd.Timestamp, end: pd.Timestamp, is_ratio: float):
    """The in-sample/out-of-sample boundary, drawn on the data that is actually there.

    A feed with a short history would otherwise leave the in-sample half empty.
    """
    first = max((pd.Timestamp(e["time"].iloc[0]) for e, _ in raw.values()), default=start)
    last = min((pd.Timestamp(e["time"].iloc[-1]) for e, _ in raw.values()), default=end)
    lo, hi = max(start, first), min(end, last)
    if hi <= lo:
        lo, hi = start, end
    return lo + (hi - lo) * is_ratio, lo, hi


def evaluate_periods(trades: list[BtTrade], cfg: dict, balance: float, split: pd.Timestamp) -> dict:
    """In-sample, out-of-sample and full-period results under the account rules."""
    periods = {}
    for label, subset in (("in_sample", [t for t in trades if t.entry_time < split]),
                          ("out_of_sample", [t for t in trades if t.entry_time >= split]),
                          ("full", trades)):
        taken, curve = run_portfolio(subset, cfg, balance)
        periods[label] = {"metrics": metrics(taken, curve, balance), "by_strategy": by_strategy(taken),
                          "taken": taken, "curve": curve}
    return periods


def run_quant_backtest(symbols: list[str], settings: dict, start: str | pd.Timestamp | None = None,
                       end: str | pd.Timestamp | None = None, balance: float | None = None,
                       is_ratio: float | None = None, loader: BarsLoader | None = None,
                       raw: dict | None = None) -> dict:
    """Backtest ``symbols`` under ``settings``; the result carries every period and the coverage.

    ``raw`` (from ``load_history``) skips the download, so a sweep or a search reuses one fetch.
    """
    bt = settings["backtest"]
    start = pd.Timestamp(start or bt["start"])
    end = pd.Timestamp(end or bt["end"] or pd.Timestamp.now().normalize())
    balance = float(balance if balance is not None else bt["initial_balance"])
    is_ratio = float(is_ratio if is_ratio is not None else bt["in_sample_ratio"])
    if end <= start:
        raise ValueError(f"the backtest ends before it starts: {end:%Y-%m-%d} is not after {start:%Y-%m-%d}")

    raw = raw or load_history(symbols, settings, start, end, loader)
    trades, cfg = simulate_all(raw, settings, start)
    split, lo, hi = split_point(raw, start, end, is_ratio)
    periods = evaluate_periods(trades, cfg, balance, split)
    coverage = [{"symbol": s, "bars": int(len(e)), "first": pd.Timestamp(e["time"].iloc[0]),
                 "last": pd.Timestamp(e["time"].iloc[-1])} for s, (e, _) in raw.items()]
    return {"settings": settings, "balance": balance, "start": start, "end": end, "split": split,
            "tested_from": lo, "tested_to": hi, "periods": periods, "coverage": coverage,
            "raw_trades": len(trades)}


def sweep(symbols: list[str], settings: dict, path: str, values: list, raw: dict, **kwargs) -> list[dict]:
    """Re-run the backtest once per value of one setting, e.g. ``risk.sl_atr_multiplier``."""
    from tradingagents.quant.settings import apply_overrides
    rows = []
    for value in values:
        result = run_quant_backtest(symbols, apply_overrides(settings, {path: value}), raw=raw, **kwargs)
        rows.append({"value": value, "full": result["periods"]["full"]["metrics"],
                     "out_of_sample": result["periods"]["out_of_sample"]["metrics"]})
    return rows
