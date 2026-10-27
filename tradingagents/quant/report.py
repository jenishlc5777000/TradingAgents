"""Save a quant backtest: trades.csv, summary.json and a self-contained report.html."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PERIOD_LABELS = {"in_sample": "In-sample", "out_of_sample": "Out-of-sample", "full": "Full period"}


def _num(v):
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return None
    return v


def summary(result: dict) -> dict:
    return {
        "symbols": [c["symbol"] for c in result["coverage"]],
        "start": str(result["start"]), "end": str(result["end"]), "split": str(result["split"]),
        "tested_from": str(result["tested_from"]), "tested_to": str(result["tested_to"]),
        "balance": result["balance"],
        "settings": result["settings"],
        "coverage": [{**c, "first": str(c["first"]), "last": str(c["last"])} for c in result["coverage"]],
        "periods": {k: {"metrics": {m: _num(v) for m, v in p["metrics"].items()},
                        "by_strategy": p["by_strategy"]} for k, p in result["periods"].items()},
    }


def equity_svg(curve, split, width: int = 900, height: int = 280) -> str:
    """A step line of the balance with the out-of-sample stretch shaded."""
    if len(curve) < 2:
        return ""
    times = [pd.Timestamp(t).value for t, _ in curve]
    vals = [float(v) for _, v in curve]
    t0, t1 = min(times), max(times)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad

    def x(t):
        return 40 + (t - t0) / ((t1 - t0) or 1) * (width - 50)

    def y(v):
        return 10 + (hi - v) / (hi - lo) * (height - 30)

    pts, prev = [], None
    for t, v in zip(times, vals):
        if prev is not None:
            pts.append(f"{x(t):.1f},{y(prev):.1f}")
        pts.append(f"{x(t):.1f},{y(v):.1f}")
        prev = v
    sx = x(min(max(pd.Timestamp(split).value, t0), t1))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve" '
            f'style="width:100%;height:auto">'
            f'<rect x="{sx:.1f}" y="10" width="{width - 10 - sx:.1f}" height="{height - 30}" fill="currentColor" opacity=".07"/>'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="1.6"/>'
            f'<text x="42" y="{height - 6}" font-size="11" fill="currentColor">{html.escape(f"{lo + pad:,.0f}")} low · '
            f'{html.escape(f"{hi - pad:,.0f}")} high · shaded = out-of-sample</text></svg>')


def render_html(result: dict) -> str:
    s = summary(result)
    rows = []
    for key, label in PERIOD_LABELS.items():
        m = s["periods"][key]["metrics"]
        cells = "".join(f"<td>{html.escape(str(m.get(c, '-')))}</td>" for c in
                        ("trades", "win_rate_pct", "profit_factor", "max_drawdown_pct", "avg_r", "total_r",
                         "net_profit", "return_pct"))
        rows.append(f"<tr><th>{label}</th>{cells}</tr>")
    strat = "".join(
        f"<tr><td>{html.escape(r['symbol'])}</td><td>{html.escape(r['strategy'])}</td><td>{r['trades']:.0f}</td>"
        f"<td>{r['win_rate_pct']}</td><td>{r['avg_r']}</td><td>{r['total_r']}</td></tr>"
        for r in s["periods"]["full"]["by_strategy"])
    cover = "".join(f"<li>{html.escape(c['symbol'])}: {c['bars']} bars, {c['first'][:16]} to {c['last'][:16]}</li>"
                    for c in s["coverage"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quant backtest</title>
<style>:root{{color-scheme:light dark}}body{{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}td,th{{border-bottom:1px solid #8884;padding:4px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>Quant backtest: {html.escape(", ".join(s["symbols"]))}</h1>
<p>{html.escape(s["start"][:10])} to {html.escape(s["end"][:10])}, starting balance {s["balance"]:,.2f}.
Split at {html.escape(s["split"][:16])}.</p>
{equity_svg(result["periods"]["full"]["curve"], result["split"])}
<table><tr><th></th><th>Trades</th><th>Win %</th><th>Profit factor</th><th>Max DD %</th><th>Avg R</th><th>Total R</th><th>Net</th><th>Return %</th></tr>{"".join(rows)}</table>
<h2>By strategy (full period)</h2>
<table><tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>Win %</th><th>Avg R</th><th>Total R</th></tr>{strat}</table>
<h2>Data covered</h2><ul>{cover}</ul>
<p>Past results do not predict live results. Costs are a flat spread share; fills, lots and slippage are simplified.</p>
</body></html>"""


def save_report(result: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"summary": out / "summary.json", "trades": out / "trades.csv", "html": out / "report.html"}
    paths["summary"].write_text(json.dumps(summary(result), indent=2, default=str), encoding="utf-8")
    taken = result["periods"]["full"]["taken"]
    pd.DataFrame([asdict(t) for t in taken]).to_csv(paths["trades"], index=False)
    paths["html"].write_text(render_html(result), encoding="utf-8")
    return paths
