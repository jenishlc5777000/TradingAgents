"""`tradingagents quant ...`: settings commands persist, runs take --set without saving."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cli.main as m
from tests.test_quant import fake_loader
from tradingagents.quant import settings as qs


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setattr(qs, "SETTINGS_PATH", tmp_path / "quant_settings.json")
    monkeypatch.setattr("tradingagents.quant.backtest.default_loader", fake_loader)
    monkeypatch.setattr("tradingagents.quant.scan.default_loader", fake_loader)
    return CliRunner()


def run(runner, *args):
    return runner.invoke(m.app, ["quant", *args], env={"COLUMNS": "200"})


@pytest.mark.unit
def test_strategies_lists_all_six_and_toggling_is_saved(runner):
    out = run(runner, "strategies")
    assert out.exit_code == 0 and "stoch_reversal" in out.output
    assert run(runner, "disable", "breakout").exit_code == 0
    assert qs.load_settings()["strategies"]["breakout"]["enabled"] is False
    assert run(runner, "enable", "breakout").exit_code == 0
    assert qs.load_settings()["strategies"]["breakout"]["enabled"] is True


@pytest.mark.unit
def test_unknown_strategy_is_refused(runner):
    out = run(runner, "enable", "nope")
    assert out.exit_code == 1 and "unknown strategy" in out.output


@pytest.mark.unit
def test_set_saves_valid_values_and_refuses_invalid_ones(runner):
    assert run(runner, "set", "risk.sl_atr_multiplier", "2.5").exit_code == 0
    assert qs.load_settings()["risk"]["sl_atr_multiplier"] == 2.5
    bad = run(runner, "set", "risk.sl_atr_multiplier", "0")
    assert bad.exit_code == 1 and qs.load_settings()["risk"]["sl_atr_multiplier"] == 2.5
    assert run(runner, "set", "risk.no_such_thing", "1").exit_code == 1


@pytest.mark.unit
def test_preset_and_reset(runner):
    assert run(runner, "preset", "intraday").exit_code == 0
    assert qs.load_settings()["timeframes"] == {"entry": "M15", "trend": "H4"}
    assert run(runner, "reset").exit_code == 0
    assert qs.load_settings()["timeframes"] == qs.DEFAULTS["timeframes"]


@pytest.mark.unit
def test_backtest_prints_results_saves_a_report_and_set_is_not_persisted(runner, tmp_path):
    out = run(runner, "backtest", "AAA", "--start", "2023-06-01", "--end", "2023-09-30",
              "--set", "risk.sl_atr_multiplier=2", "--sweep", "risk.sl_atr_multiplier=1,2",
              "--out", str(tmp_path / "rep"))
    assert out.exit_code == 0, out.output
    assert "Out-of-sample" in out.output and "Sweep of risk.sl_atr_multiplier" in out.output
    assert (tmp_path / "rep" / "report.html").is_file()
    assert qs.load_settings()["risk"]["sl_atr_multiplier"] == qs.DEFAULTS["risk"]["sl_atr_multiplier"]


@pytest.mark.unit
def test_backtest_without_enabled_strategies_says_so(runner):
    for name in qs.DEFAULTS["strategies"]:
        run(runner, "disable", name)
    out = run(runner, "backtest", "AAA", "--start", "2023-06-01", "--end", "2023-09-30")
    assert out.exit_code == 1 and "No strategy is enabled" in out.output


@pytest.mark.unit
def test_scan_and_optimize_run(runner):
    scan = run(runner, "scan", "AAA", "--date", "2023-08-01")
    assert scan.exit_code == 0 and "Regime:" in scan.output
    opt = run(runner, "optimize", "AAA", "--start", "2023-06-01", "--end", "2023-09-30",
              "--iterations", "4", "--min-trades", "1", "--max-strategies", "2")
    assert opt.exit_code == 0, opt.output
    assert "Tested 4 setups" in opt.output
