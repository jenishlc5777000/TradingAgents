"""What the enabled strategies say about an instrument right now, or on a past date.

Used by the ``scan`` command and by the market analyst's ``get_strategy_signals`` tool, so
the LLM agents and the rule engine read the same numbers. ``as_of`` cuts the bars before
a date, so a dated agent run never sees a signal formed after it.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.quant.backtest import BarsLoader, default_loader, warmup_days
from tradingagents.quant.indicators import TF_SECONDS, prepare
from tradingagents.quant.levels import trade_levels
from tradingagents.quant.regime import detect_regime
from tradingagents.quant.settings import engine_config
from tradingagents.quant.strategies import evaluate, load_strategies


def scan_symbol(symbol: str, settings: dict, as_of: str | pd.Timestamp | None = None,
                loader: BarsLoader | None = None) -> dict:
    """Latest closed bar's regime, every strategy's verdict, and the trade levels of the winner."""
    loader = loader or default_loader
    cfg = engine_config(settings)
    tf = cfg["timeframes"]
    end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1) if as_of else pd.Timestamp.now()
    # Enough entry bars for the slowest indicator to settle, not the whole warm-up.
    lookback = max(warmup_days(settings), int(300 * TF_SECONDS[tf["entry"]] / 86400 * 1.6) + 5)
    start = end - pd.Timedelta(days=lookback)
    entry = loader(symbol, tf["entry"], start, end)
    trend = entry if tf["trend"] == tf["entry"] else loader(symbol, tf["trend"], start, end)
    if as_of:
        # A bar belongs to the run only once it has closed by the end of the as-of day.
        cutoff = end
        entry = entry[entry["time"] + pd.Timedelta(seconds=TF_SECONDS[tf["entry"]]) <= cutoff]
        trend = trend[trend["time"] + pd.Timedelta(seconds=TF_SECONDS[tf["trend"]]) <= cutoff]
    if len(entry) < 4:
        raise ValueError(f"not enough {tf['entry']} bars for {symbol} to scan")

    df = prepare(entry, trend, cfg)
    window = df.iloc[-3:]
    row = df.iloc[-1]
    regime = detect_regime(row, cfg["regime"])
    strategies = load_strategies(cfg)
    verdicts = []
    for strategy in strategies:
        signal = strategy.check(symbol, window, regime) if strategy.trades_in(regime) else None
        verdicts.append({"strategy": strategy.name, "regimes": list(strategy.regimes),
                         "allowed_now": strategy.trades_in(regime),
                         "direction": signal.direction if signal else None,
                         "reason": signal.reason if signal else None})
    _, winner = evaluate(symbol, window, strategies, cfg["regime"])
    return {
        "symbol": symbol, "bar_time": pd.Timestamp(row["time"]), "timeframe": tf["entry"],
        "close": float(row["close"]), "regime": regime,
        "adx": None if pd.isna(row.get("adx")) else float(row["adx"]),
        "rsi": None if pd.isna(row.get("rsi")) else float(row["rsi"]),
        "signal": None if winner is None else {"direction": winner.direction, "strategy": winner.strategy,
                                               "reason": winner.reason},
        "levels": trade_levels(row, winner.direction, cfg) if winner else None,
        "strategies": verdicts,
    }


def render_scan(scan: dict) -> str:
    """The scan as text an analyst can quote."""
    lines = [f"Rule-based strategy scan for {scan['symbol']} on {scan['timeframe']} bars, "
             f"last closed bar {scan['bar_time']:%Y-%m-%d %H:%M} (close {scan['close']:.5g}).",
             f"Regime: {scan['regime']}" + (f" (ADX {scan['adx']:.1f})" if scan["adx"] is not None else "")
             + (f", RSI {scan['rsi']:.1f}" if scan["rsi"] is not None else "") + "."]
    sig = scan["signal"]
    if sig:
        lv = scan["levels"] or {}
        lines.append(f"SIGNAL: {sig['direction'].upper()} from {sig['strategy']} - {sig['reason']}.")
        if lv.get("sl") is not None:
            lines.append(f"Levels: entry {lv['entry']:.5g}, stop {lv['sl']:.5g}, "
                         f"{lv['t1_r']:g}R target {lv['tp1']:.5g}, {lv['t2_r']:g}R target {lv['tp2']:.5g}"
                         + ("" if lv.get("t2_real") else " (targets are projections; the exit is the trailing stop)")
                         + ".")
    else:
        lines.append("SIGNAL: none. No enabled strategy fires on this bar.")
    for v in scan["strategies"]:
        state = (f"fires {v['direction']}" if v["direction"]
                 else "no setup" if v["allowed_now"] else f"sits out the {scan['regime']} regime")
        lines.append(f"- {v['strategy']}: {state}")
    return "\n".join(lines)
