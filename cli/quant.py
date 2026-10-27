"""`tradingagents quant ...`: the rule-based strategy engine, beside the LLM agents.

Settings live in ``~/.tradingagents/quant_settings.json`` and use the same sections as the
bot the engine came from. Any command takes ``--set path=value`` to try a change for one
run without saving it.
"""

import json
from datetime import datetime

import typer
from rich.markup import escape
from rich.table import Table

from cli.display import console

quant_app = typer.Typer(name="quant", help="Rule-based strategies, settings, scans, backtests and a setup search.")


def _fail(message) -> "typer.Exit":
    console.print(f"[red]{escape(str(message))}[/red]")
    return typer.Exit(code=1)


def _settings(overrides: list[str] | None = None, preset: str | None = None) -> dict:
    """Saved settings with this run's ``--set`` and ``--preset`` laid over them, validated."""
    from tradingagents.quant import settings as qs

    base = qs.load_settings()
    if preset:
        if preset not in qs.TIMEFRAME_PRESETS:
            raise _fail(f"unknown preset {preset!r}; use one of {', '.join(qs.TIMEFRAME_PRESETS)}")
        base = qs.apply_overrides(base, {"timeframes": qs.TIMEFRAME_PRESETS[preset]})
    pairs = {}
    for item in overrides or []:
        path, sep, value = item.partition("=")
        if not sep:
            raise _fail(f"--set expects path=value, got {item!r}")
        pairs[path.strip()] = qs.parse_value(value.strip())
    try:
        return qs.validate(qs.apply_overrides(base, pairs))
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"settings are not usable: {exc}") from None


def _names(tickers: str) -> list[str]:
    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        raise _fail("No ticker given; pass them comma-separated, e.g. AAPL,GC=F,BTC-USD")
    return names


SET_OPTION = typer.Option(None, "--set", help="Override a setting for this run, e.g. --set risk.sl_atr_multiplier=2")
PRESET_OPTION = typer.Option(None, "--preset", help="Timeframes: scalp (M5/H1), intraday (M15/H4), swing (H1/D1), position (D1)")


# --------------------------------------------------------------------------- settings
@quant_app.command("strategies")
def strategies_cmd():
    """List the strategies, whether each is on, and the regimes it trades in."""
    from tradingagents.quant.settings import load_settings
    from tradingagents.quant.strategies import catalog

    saved = load_settings()["strategies"]
    table = Table(title="Strategies (order = priority; the first to fire on a bar wins)")
    for column in ("", "Name", "On", "Regimes", "What it does"):
        table.add_column(column, overflow="fold")
    for i, entry in enumerate(catalog(), 1):
        on = bool((saved.get(entry["name"]) or {}).get("enabled"))
        table.add_row(str(i), entry["name"], "[green]yes[/green]" if on else "[dim]no[/dim]",
                      ", ".join(entry["regimes"]), entry["summary"])
    console.print(table)
    console.print("Turn one on or off: [bold]tradingagents quant enable NAME[/bold] / [bold]disable NAME[/bold]. "
                  "Tune one: [bold]quant set strategies.NAME.PARAM VALUE[/bold].")


def _toggle(name: str, enabled: bool) -> None:
    from tradingagents.quant import settings as qs
    from tradingagents.quant.strategies import REGISTRY

    if name not in REGISTRY:
        raise _fail(f"unknown strategy {name!r}; known: {', '.join(REGISTRY)}")
    s = qs.load_settings()
    s["strategies"][name]["enabled"] = enabled
    qs.save_settings(s)
    console.print(f"{name}: [bold]{'on' if enabled else 'off'}[/bold]. Enabled now: "
                  f"{', '.join(qs.enabled_strategies(s)) or 'none'}")


@quant_app.command("enable")
def enable_cmd(name: str = typer.Argument(..., help="Strategy name")):
    """Turn a strategy on."""
    _toggle(name, True)


@quant_app.command("disable")
def disable_cmd(name: str = typer.Argument(..., help="Strategy name")):
    """Turn a strategy off."""
    _toggle(name, False)


@quant_app.command("settings")
def settings_cmd(path: str = typer.Argument(None, help="Section or setting to show, e.g. risk or risk.sl_atr_multiplier")):
    """Show the settings (all, or one section / value)."""
    from tradingagents.quant.settings import SETTINGS_PATH, get_path, load_settings

    s = load_settings()
    try:
        shown = get_path(s, path) if path else s
    except KeyError:
        raise _fail(f"no setting {path!r}; sections: {', '.join(s)}") from None
    console.print_json(json.dumps(shown))
    console.print(f"[dim]Saved in {SETTINGS_PATH}[/dim]")


@quant_app.command("set")
def set_cmd(path: str = typer.Argument(..., help="e.g. risk.risk_per_trade_money"),
            value: str = typer.Argument(..., help="JSON value: 1.5, true, null, \"M15\"")):
    """Change one setting and save it."""
    from tradingagents.quant import settings as qs

    s = qs.load_settings()
    try:
        qs.get_path(s, path)
    except KeyError:
        raise _fail(f"no setting {path!r}; run `tradingagents quant settings` to see them") from None
    qs.set_path(s, path, qs.parse_value(value))
    try:
        qs.save_settings(s)
    except (KeyError, ValueError, TypeError) as exc:
        raise _fail(f"not saved: {exc}") from None
    console.print(f"{path} = {json.dumps(qs.get_path(s, path))}")


@quant_app.command("preset")
def preset_cmd(name: str = typer.Argument(..., help="scalp, intraday, swing or position")):
    """Set the entry and trend timeframes from a named pair and save them."""
    from tradingagents.quant import settings as qs

    if name not in qs.TIMEFRAME_PRESETS:
        raise _fail(f"unknown preset {name!r}; use one of {', '.join(qs.TIMEFRAME_PRESETS)}")
    s = qs.load_settings()
    s["timeframes"] = dict(qs.TIMEFRAME_PRESETS[name])
    qs.save_settings(s)
    console.print(f"Timeframes: entry {s['timeframes']['entry']}, trend {s['timeframes']['trend']}")
    if name in ("scalp", "intraday"):
        console.print("[yellow]Yahoo keeps only about 60 days of 5/15-minute bars; a longer backtest is cut to that.[/yellow]")


@quant_app.command("reset")
def reset_cmd():
    """Delete the saved settings and go back to the defaults."""
    from tradingagents.quant.settings import reset_settings

    reset_settings()
    console.print("Quant settings reset to the defaults.")


# --------------------------------------------------------------------------- scan
@quant_app.command("scan")
def scan_cmd(tickers: str = typer.Argument(..., help="Comma-separated, e.g. AAPL,GC=F"),
             date: str = typer.Option(None, "--date", help="Scan as of this date, YYYY-MM-DD (default: now)"),
             set_: list[str] = SET_OPTION, preset: str = PRESET_OPTION):
    """What the enabled strategies say about each instrument on its last closed bar."""
    from tradingagents.quant.scan import render_scan, scan_symbol

    settings = _settings(set_, preset)
    for symbol in _names(tickers):
        try:
            console.print(escape(render_scan(scan_symbol(symbol, settings, as_of=date))) + "\n")
        except Exception as exc:  # a missing symbol must not hide the others
            console.print(f"[yellow]{escape(symbol)}: {escape(str(exc))}[/yellow]")


# --------------------------------------------------------------------------- backtest
def _metrics_table(result: dict) -> Table:
    table = Table(title="Results (each period starts from the same balance)")
    cols = ("trades", "win_rate_pct", "profit_factor", "max_drawdown_pct", "avg_r", "total_r", "net_profit", "return_pct")
    heads = ("Trades", "Win %", "PF", "Max DD %", "Avg R", "Total R", "Net", "Return %")
    table.add_column("Period", no_wrap=True)
    for head in heads:
        table.add_column(head, justify="right", no_wrap=True)
    from tradingagents.quant.report import PERIOD_LABELS
    for key, label in PERIOD_LABELS.items():
        m = result["periods"][key]["metrics"]
        table.add_row(label, *[str(m.get(c, "-")) for c in cols])
    return table


def _coverage_notes(result: dict) -> None:
    for c in result["coverage"]:
        console.print(f"[dim]{escape(c['symbol'])}: {c['bars']} bars, {c['first']:%Y-%m-%d} to {c['last']:%Y-%m-%d}[/dim]")
    if result["tested_from"] > result["start"] + (result["end"] - result["start"]) * 0.05:
        console.print(f"[yellow]The feed only reaches back to {result['tested_from']:%Y-%m-%d}, so that is where the test starts "
                      f"(you asked for {result['start']:%Y-%m-%d}).[/yellow]")


@quant_app.command("backtest")
def backtest_cmd(tickers: str = typer.Argument(..., help="Comma-separated, e.g. AAPL,GC=F,BTC-USD"),
                 start: str = typer.Option(None, "--start", help="YYYY-MM-DD (default: settings backtest.start)"),
                 end: str = typer.Option(None, "--end", help="YYYY-MM-DD (default: today)"),
                 balance: float = typer.Option(None, "--balance", help="Starting balance"),
                 is_ratio: float = typer.Option(None, "--is-ratio", help="Share of the period that is in-sample"),
                 sweep: str = typer.Option(None, "--sweep", help="Re-run over values of one setting: path=v1,v2,v3"),
                 out: str = typer.Option(None, "--out", help="Folder for report.html, trades.csv, summary.json"),
                 set_: list[str] = SET_OPTION, preset: str = PRESET_OPTION):
    """Replay the enabled strategies on history, with an in-sample / out-of-sample split."""
    from tradingagents.quant.backtest import load_history, run_quant_backtest
    from tradingagents.quant.backtest import sweep as run_sweep
    from tradingagents.quant.report import save_report
    from tradingagents.quant.settings import enabled_strategies, parse_value

    settings = _settings(set_, preset)
    names = _names(tickers)
    if not enabled_strategies(settings):
        raise _fail("No strategy is enabled; run `tradingagents quant enable NAME`")
    import pandas as pd
    bt = settings["backtest"]
    try:
        t0 = pd.Timestamp(start or bt["start"])
        t1 = pd.Timestamp(end or bt["end"] or datetime.now().strftime("%Y-%m-%d"))
        if t1 <= t0:
            raise ValueError(f"the backtest ends before it starts ({t1:%Y-%m-%d} is not after {t0:%Y-%m-%d})")
        console.print(f"[dim]Downloading {settings['timeframes']['entry']} / {settings['timeframes']['trend']} bars for {', '.join(names)}...[/dim]")
        raw = load_history(names, settings, t0, t1)
        result = run_quant_backtest(names, settings, t0, t1, balance, is_ratio, raw=raw)
        rows = []
        if sweep:
            path, _, values = sweep.partition("=")
            if not values:
                raise ValueError("--sweep expects path=v1,v2,v3")
            rows = run_sweep(names, settings, path.strip(), [parse_value(v.strip()) for v in values.split(",")],
                             raw, start=t0, end=t1, balance=balance, is_ratio=is_ratio)
    except Exception as exc:
        raise _fail(exc) from None

    console.print(f"Strategies: {', '.join(enabled_strategies(settings))}")
    _coverage_notes(result)
    console.print(_metrics_table(result))
    detail = result["periods"]["full"]["by_strategy"]
    if detail:
        table = Table(title="By strategy (full period)")
        for c in ("Symbol", "Strategy", "Trades", "Win %", "Avg R", "Total R"):
            table.add_column(c, justify="left" if c in ("Symbol", "Strategy") else "right", no_wrap=True)
        for r in detail:
            table.add_row(r["symbol"], r["strategy"], f"{r['trades']:.0f}", str(r["win_rate_pct"]), str(r["avg_r"]), str(r["total_r"]))
        console.print(table)
    else:
        console.print("[yellow]No trades. Enable more strategies, widen the period, or loosen the regime thresholds.[/yellow]")
    if rows:
        table = Table(title=f"Sweep of {escape(path.strip())}")
        for c in ("Value", "Trades", "Return %", "PF", "Max DD %", "OOS ret %", "OOS PF"):
            table.add_column(c, justify="right", no_wrap=True)
        for r in rows:
            f, o = r["full"], r["out_of_sample"]
            table.add_row(str(r["value"]), str(f.get("trades")), str(f.get("return_pct")), str(f.get("profit_factor")),
                          str(f.get("max_drawdown_pct")), str(o.get("return_pct")), str(o.get("profit_factor")))
        console.print(table)
    if out:
        paths = save_report(result, out)
        console.print(f"Saved {paths['html']}, {paths['trades']} and {paths['summary']}")
    console.print("[dim]Past results do not predict live results. The LLM agents are not consulted in this replay.[/dim]")


# --------------------------------------------------------------------------- optimize
@quant_app.command("optimize")
def optimize_cmd(tickers: str = typer.Argument(..., help="Comma-separated tickers"),
                 start: str = typer.Option(None, "--start"), end: str = typer.Option(None, "--end"),
                 iterations: int = typer.Option(150, "--iterations", help="Setups to test (max 600)"),
                 min_trades: int = typer.Option(10, "--min-trades", help="Out-of-sample trades needed to rank a setup"),
                 max_strategies: int = typer.Option(4, "--max-strategies", help="Most strategies switched on together"),
                 strategies: str = typer.Option(None, "--strategies", help="Comma-separated pool; default: the enabled ones"),
                 top: int = typer.Option(10, "--top"), seed: int = typer.Option(7, "--seed"),
                 workers: int = typer.Option(1, "--workers", help="Worker processes"),
                 apply: bool = typer.Option(False, "--apply", help="Save the best robust setup as your settings"),
                 set_: list[str] = SET_OPTION, preset: str = PRESET_OPTION):
    """Search strategy combinations and risk settings, ranked by out-of-sample results."""
    import pandas as pd

    from tradingagents.quant import settings as qs
    from tradingagents.quant.backtest import load_history
    from tradingagents.quant.optimizer import optimize

    settings = _settings(set_, preset)
    names = _names(tickers)
    bt = settings["backtest"]
    try:
        t0 = pd.Timestamp(start or bt["start"])
        t1 = pd.Timestamp(end or bt["end"] or datetime.now().strftime("%Y-%m-%d"))
        console.print(f"[dim]Downloading bars for {', '.join(names)}...[/dim]")
        raw = load_history(names, settings, t0, t1)
        pool = [s.strip() for s in strategies.split(",")] if strategies else None
        result = optimize(raw, settings, t0, t1, iterations=iterations, min_trades=min_trades,
                          max_strategies=max_strategies, strategies=pool, top=top, seed=seed, workers=workers,
                          progress=lambda d, n: console.print(f"[dim]tested {d}/{n}[/dim]") if d % 25 == 0 or d == n else None)
    except Exception as exc:
        raise _fail(exc) from None

    base = result["baseline"]["metrics"]["out_of_sample"]
    console.print(f"Tested {result['tested']} setups; {result['kept']} had at least {min_trades} out-of-sample trades.")
    console.print(f"Your current settings, out-of-sample: return {base.get('return_pct')}%, "
                  f"profit factor {base.get('profit_factor')}, trades {base.get('trades')}")
    table = Table(title="Best setups (robust = profitable in both halves)")
    for c in ("#", "robust", "score", "strategies", "oos ret %", "oos PF", "oos DD %", "oos trades", "is ret %"):
        table.add_column(c, overflow="fold")
    for i, r in enumerate(result["top"], 1):
        o, n = r["metrics"]["out_of_sample"], r["metrics"]["in_sample"]
        table.add_row(str(i), "yes" if r["robust"] else "no", f"{r['score']:.2f}", ", ".join(r["strategies"]),
                      str(o.get("return_pct")), str(o.get("profit_factor")), str(o.get("max_drawdown_pct")),
                      str(o.get("trades")), str(n.get("return_pct")))
    console.print(table)
    if result["top"]:
        console.print("Top setup parameters: " + json.dumps(result["top"][0]["params"]))
    console.print("[dim]A search picks what fit this history. Trust a setup that is robust and has many out-of-sample trades; "
                  "re-test it on other symbols or dates before relying on it.[/dim]")
    if apply:
        best = next((r for r in result["top"] if r["robust"]), None)
        if best is None:
            console.print("[yellow]Nothing robust to apply; settings unchanged.[/yellow]")
        else:
            saved = qs.apply_overrides(qs.load_settings(), best["overrides"])
            qs.save_settings(saved)
            console.print(f"[green]Applied setup #{result['top'].index(best) + 1} to your settings.[/green]")

