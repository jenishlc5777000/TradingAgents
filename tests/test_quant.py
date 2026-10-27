"""The rule-based engine: indicators without lookahead, strategies, settings, backtest, search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant import settings as qs
from tradingagents.quant.backtest import (
    BtTrade, metrics, run_portfolio, run_quant_backtest, simulate_symbol, sweep, warmup_days,
)
from tradingagents.quant.indicators import donchian, prepare, swing_levels
from tradingagents.quant.optimizer import candidates, optimize, score_of, strategy_combos
from tradingagents.quant.scan import render_scan, scan_symbol
from tradingagents.quant.strategies import REGISTRY, catalog, load_strategies


def make_bars(n: int, timeframe_seconds: int, seed: int = 1, start="2024-01-01") -> pd.DataFrame:
    """A trending-then-ranging random walk, enough to fire several strategies."""
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(n) % 400 < 200, 0.0006, -0.0005)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq=f"{timeframe_seconds}s"),
        "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": 1000.0,
    })


def fake_loader(symbol, timeframe, start, end):
    seconds = {"H1": 3600, "D1": 86400}[timeframe]
    n = 6000 if timeframe == "H1" else 800
    df = make_bars(n, seconds, seed=len(symbol), start="2023-01-01")
    return df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)


@pytest.mark.unit
def test_defaults_validate_and_round_trip(tmp_path):
    s = qs.load_settings(tmp_path / "none.json")
    assert qs.validate(s) is s
    path = qs.save_settings(s, tmp_path / "q.json")
    assert qs.load_settings(path) == s
    assert set(s["strategies"]) == set(REGISTRY)


@pytest.mark.unit
def test_saved_file_missing_a_key_falls_back_to_default(tmp_path):
    (tmp_path / "q.json").write_text('{"risk": {"sl_atr_multiplier": 2.5}}', encoding="utf-8")
    s = qs.load_settings(tmp_path / "q.json")
    assert s["risk"]["sl_atr_multiplier"] == 2.5
    assert s["risk"]["risk_per_trade_money"] == qs.DEFAULTS["risk"]["risk_per_trade_money"]


@pytest.mark.unit
@pytest.mark.parametrize("path, value", [
    ("risk.sl_atr_multiplier", 0), ("risk.risk_mode", "lots"), ("timeframes.entry", "H3"),
    ("regime.adx_range", 99), ("backtest.in_sample_ratio", 1.2), ("risk.take_profit_r", -1),
])
def test_invalid_values_are_rejected_and_not_saved(tmp_path, path, value):
    s = qs.load_settings(tmp_path / "none.json")
    qs.set_path(s, path, value)
    with pytest.raises(ValueError):
        qs.save_settings(s, tmp_path / "q.json")
    assert not (tmp_path / "q.json").exists()


@pytest.mark.unit
def test_overrides_do_not_touch_the_original():
    s = qs.load_settings("/nonexistent/q.json")
    out = qs.apply_overrides(s, {"risk.compounding": False, "strategies.breakout.enabled": False})
    assert out["risk"]["compounding"] is False and s["risk"]["compounding"] is True
    assert qs.parse_value("1.5") == 1.5 and qs.parse_value("null") is None and qs.parse_value("M15") == "M15"


@pytest.mark.unit
def test_catalog_describes_every_strategy():
    assert [c["name"] for c in catalog()] == list(REGISTRY)
    assert all(c["summary"] and c["regimes"] for c in catalog())


@pytest.mark.unit
def test_donchian_and_swing_levels_never_see_the_future():
    bars = make_bars(300, 3600)
    hi, lo = bars["high"], bars["low"]
    dh, _ = donchian(hi, lo, 20)
    assert dh.iloc[30] == hi.iloc[10:30].max()  # the 20 bars before bar 30, not including it

    base_sup, base_res = swing_levels(hi, lo, 10, 3)
    changed = bars.copy()
    changed.loc[200:, ["high", "low"]] *= 1.5  # rewrite the future
    sup, res = swing_levels(changed["high"], changed["low"], 10, 3)
    # Until a changed bar is within `right` bars of a pivot, nothing earlier moves.
    assert res.iloc[:197].equals(base_res.iloc[:197]) and sup.iloc[:197].equals(base_sup.iloc[:197])


@pytest.mark.unit
def test_prepare_attaches_only_closed_higher_timeframe_bars():
    s = qs.load_settings("/nonexistent/q.json")
    entry, trend = make_bars(2400, 3600), make_bars(100, 86400)
    df = prepare(entry, trend, qs.engine_config(s))
    row = df[df["time"] == pd.Timestamp("2024-01-20 10:00")].iloc[0]  # closes 11:00 on day 20
    day19 = trend[trend["time"] == pd.Timestamp("2024-01-19")].iloc[0]  # closed Jan 20 00:00
    assert row["htf_close"] == day19["close"]


@pytest.mark.unit
def test_backtest_runs_and_the_periods_add_up():
    s = qs.load_settings("/nonexistent/q.json")
    result = run_quant_backtest(["AAA"], s, "2023-06-01", "2023-09-30", loader=fake_loader)
    full = result["periods"]["full"]["metrics"]
    assert full["trades"] > 0
    ins = result["periods"]["in_sample"]["metrics"].get("trades", 0)
    oos = result["periods"]["out_of_sample"]["metrics"].get("trades", 0)
    assert ins + oos <= result["raw_trades"] and ins > 0 and oos > 0
    for trade in result["periods"]["full"]["taken"]:
        assert trade.exit_time >= trade.entry_time and trade.exit_reason


@pytest.mark.unit
def test_stop_is_hit_before_anything_else_and_gaps_fill_at_the_open():
    s = qs.load_settings("/nonexistent/q.json")
    s["costs"]["spread_pct"] = 0
    cfg = qs.engine_config(s)
    entry, trend = make_bars(2400, 3600), make_bars(100, 86400)
    df = prepare(entry, trend, cfg)
    trades = simulate_symbol("AAA", df, cfg, pd.Timestamp("2024-02-01"))
    assert trades
    for t in trades:
        if t.exit_reason == "sl" and t.direction == "buy":
            bar = df[df["time"] == t.exit_time].iloc[0]
            assert t.exit == min(bar["open"], t.sl) and t.exit <= bar["open"]


@pytest.mark.unit
def test_disabling_every_strategy_gives_no_trades():
    s = qs.load_settings("/nonexistent/q.json")
    for params in s["strategies"].values():
        params["enabled"] = False
    result = run_quant_backtest(["AAA"], s, "2023-06-01", "2023-09-30", loader=fake_loader)
    assert result["raw_trades"] == 0 and result["periods"]["full"]["metrics"] == {"trades": 0}


def _trade(day, hours, r, symbol="AAA"):
    t0 = pd.Timestamp(day) + pd.Timedelta(hours=hours)
    return BtTrade(symbol, "buy", "x", "trend", t0, 100, 99, 99, 100, exit_time=t0 + pd.Timedelta(hours=1),
                   exit=100 + r, r=r)


@pytest.mark.unit
def test_account_rules_cap_open_trades_daily_loss_and_trades_per_day():
    cfg = qs.engine_config(qs.load_settings("/nonexistent/q.json"))
    cfg["risk"].update(risk_mode="money", risk_per_trade_money=10.0, max_daily_loss_money=15.0,
                       max_daily_loss_pct=3.0, max_trades_per_day=0, max_open_trades=1, max_trades_per_symbol=1)
    # Two trades overlap in time: only the first is taken.
    a, b = _trade("2024-01-02", 9, -1.0), _trade("2024-01-02", 9, 2.0, "BBB")
    b.entry_time = a.entry_time + pd.Timedelta(minutes=10)
    taken, _ = run_portfolio([a, b], cfg, 1000)
    assert taken == [a]

    # Two -1R losses reach the 15 daily limit after the second (20 > 15): the third is skipped.
    losers = [_trade("2024-01-03", h, -1.0) for h in (9, 11, 13)]
    taken, curve = run_portfolio(losers, cfg, 1000)
    assert len(taken) == 2 and curve[-1][1] == 980

    cfg["risk"]["max_trades_per_day"] = 1
    taken, _ = run_portfolio([_trade("2024-01-04", h, 1.0) for h in (9, 11)], cfg, 1000)
    assert len(taken) == 1


@pytest.mark.unit
def test_martingale_raises_the_stake_after_a_loss_up_to_its_cap():
    cfg = qs.engine_config(qs.load_settings("/nonexistent/q.json"))
    cfg["risk"].update(risk_per_trade_money=10.0, max_trades_per_day=0, max_daily_loss_money=1e9, max_open_trades=1)
    cfg["risk"]["martingale"].update(enabled=True, multiplier=2.0, max_steps=2, max_risk_money=35.0)
    taken, _ = run_portfolio([_trade("2024-01-05", h, -1.0) for h in (9, 11, 13, 15)], cfg, 1000)
    assert [round(t.pnl) for t in taken] == [-10, -20, -35, -35]  # 10, x2, x4 capped at 35


@pytest.mark.unit
def test_metrics_of_a_known_sequence():
    cfg = qs.engine_config(qs.load_settings("/nonexistent/q.json"))
    cfg["risk"].update(risk_per_trade_money=10.0, max_trades_per_day=0, max_open_trades=1)
    taken, curve = run_portfolio([_trade("2024-01-08", 9, 2.0), _trade("2024-01-08", 11, -1.0)], cfg, 1000)
    m = metrics(taken, curve, 1000)
    assert (m["trades"], m["win_rate_pct"], m["net_profit"], m["profit_factor"], m["total_r"]) == (2, 50.0, 10.0, 2.0, 1.0)


@pytest.mark.unit
def test_warmup_covers_the_trend_ema():
    s = qs.load_settings("/nonexistent/q.json")
    assert warmup_days(s) >= 200  # D1 EMA200 needs 200 daily bars
    s["timeframes"] = {"entry": "M15", "trend": "H4"}
    assert warmup_days(s) < 200


@pytest.mark.unit
def test_scan_reports_regime_and_every_strategy_and_respects_as_of():
    s = qs.load_settings("/nonexistent/q.json")
    scan = scan_symbol("AAA", s, as_of="2023-08-01", loader=fake_loader)
    assert scan["bar_time"] < pd.Timestamp("2023-08-02")
    assert [v["strategy"] for v in scan["strategies"]] == list(REGISTRY)
    text = render_scan(scan)
    assert "Regime:" in text and "SIGNAL:" in text


@pytest.mark.unit
def test_sweep_returns_one_row_per_value():
    s = qs.load_settings("/nonexistent/q.json")
    from tradingagents.quant.backtest import load_history
    start, end = pd.Timestamp("2023-06-01"), pd.Timestamp("2023-09-30")
    raw = load_history(["AAA"], s, start, end, fake_loader)
    rows = sweep(["AAA"], s, "risk.sl_atr_multiplier", [1.0, 2.0], raw, start=start, end=end)
    assert [r["value"] for r in rows] == [1.0, 2.0]


@pytest.mark.unit
def test_search_candidates_are_distinct_and_scoring_punishes_small_samples():
    combos = strategy_combos(list(REGISTRY), 2)
    assert len(combos) == 6 + 15
    cands = candidates({"risk.sl_atr_multiplier": [1.0, 2.0]}, combos, 20, seed=3)
    keys = {(tuple(c["strategies"]), str(c["params"])) for c in cands}
    assert len(keys) == len(cands) == 20
    assert score_of({"trades": 3, "return_pct": 50, "max_drawdown_pct": -2}, 10) == float("-inf")
    few = score_of({"trades": 10, "return_pct": 20, "max_drawdown_pct": -5}, 10)
    many = score_of({"trades": 20, "return_pct": 20, "max_drawdown_pct": -5}, 10)
    assert 0 < few < many


@pytest.mark.unit
def test_optimize_ranks_and_reports_a_baseline():
    from tradingagents.quant.backtest import load_history
    s = qs.load_settings("/nonexistent/q.json")
    start, end = pd.Timestamp("2023-06-01"), pd.Timestamp("2023-09-30")
    raw = load_history(["AAA"], s, start, end, fake_loader)
    result = optimize(raw, s, start, end, iterations=6, min_trades=1, max_strategies=2, top=3)
    assert result["tested"] == 6 and len(result["top"]) <= 3
    scores = [(r["robust"], r["score"]) for r in result["top"]]
    assert scores == sorted(scores, reverse=True)
    assert result["baseline"]["metrics"]["full"]["trades"] >= 0


@pytest.mark.unit
def test_report_writes_files(tmp_path):
    from tradingagents.quant.report import save_report
    s = qs.load_settings("/nonexistent/q.json")
    result = run_quant_backtest(["AAA"], s, "2023-06-01", "2023-09-30", loader=fake_loader)
    paths = save_report(result, tmp_path)
    assert "<svg" in paths["html"].read_text(encoding="utf-8")
    assert paths["trades"].read_text(encoding="utf-8").startswith("symbol,")


@pytest.mark.unit
def test_load_strategies_skips_disabled_and_rejects_unknown():
    cfg = qs.engine_config(qs.load_settings("/nonexistent/q.json"))
    cfg["strategies"]["breakout"]["enabled"] = False
    assert "breakout" not in [x.name for x in load_strategies(cfg)]
    cfg["strategies"]["bogus"] = {"enabled": True}
    with pytest.raises(ValueError, match="Unknown strategy"):
        load_strategies(cfg)
