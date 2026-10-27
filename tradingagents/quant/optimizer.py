"""Parameter search: run many configurations through the real backtest and rank them.

Every candidate is simulated in full, because changing a strategy, a stop or a regime
threshold changes which signals exist at all. Ranking leans on the out-of-sample half:
a setup that shines on the data it was picked from and falls apart afterwards is the
normal result of a search like this, so ``score`` reads out-of-sample numbers only and
``robust`` also requires the in-sample half to agree.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from tradingagents.quant.backtest import evaluate_periods, simulate_all, split_point
from tradingagents.quant.settings import apply_overrides, enabled_strategies
from tradingagents.quant.strategies import REGISTRY

logger = logging.getLogger(__name__)

# What the search may change: a settings path, and the values to try for it.
SPACE: dict[str, list] = {
    "risk.sl_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
    "risk.take_profit_r": [None, 1.5, 2.0, 3.0],
    "risk.max_trades_per_day": [0, 3, 5, 8],
    "trailing.breakeven_r": [0.5, 1.0, 1.5],
    "trailing.trail_start_r": [1.0, 1.5, 2.0],
    "trailing.trail_atr_multiplier": [1.5, 2.0, 3.0],
    "regime.adx_trend": [20, 25, 30],
    "regime.adx_range": [15, 20, 25],
}


def strategy_combos(names: list[str], max_on: int = 4) -> list[tuple[str, ...]]:
    """Every non-empty combination of up to ``max_on`` of these strategies."""
    out: list[tuple[str, ...]] = []
    for n in range(1, min(max_on, len(names)) + 1):
        out += list(itertools.combinations(names, n))
    return out


def candidates(space: dict[str, list], combos: list[tuple[str, ...]], limit: int, seed: int = 7) -> list[dict]:
    """Sample ``limit`` distinct configurations.

    The full grid runs to millions of combinations, so this samples at random, which
    covers the space evenly instead of exhausting the first parameter.
    """
    rng = random.Random(seed)
    seen: set = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < limit and attempts < limit * 80:
        attempts += 1
        combo = rng.choice(combos)
        pick = {k: rng.choice(v) for k, v in space.items()}
        key = (combo, tuple(sorted((k, str(v)) for k, v in pick.items())))
        if key in seen:
            continue
        seen.add(key)
        overrides = {f"strategies.{n}.enabled": n in combo for n in REGISTRY}
        overrides.update(pick)
        out.append({"strategies": list(combo), "params": pick, "overrides": overrides})
    return out


def score_of(oos: dict, min_trades: int) -> float:
    """Out-of-sample return per unit of drawdown, damped when there are few trades.

    A setup with three trades can show any number at all, so the confidence multiplier
    pulls small samples towards zero rather than letting luck top the table.
    """
    trades = int(oos.get("trades") or 0)
    if trades < min_trades:
        return float("-inf")
    ret = float(oos.get("return_pct") or 0.0)
    dd = abs(float(oos.get("max_drawdown_pct") or 0.0))
    confidence = min(1.0, trades / max(1, min_trades * 2))
    return (ret / max(5.0, dd)) * confidence


def run_candidate(raw: dict, base: dict, start, end, is_ratio: float, balance: float,
                  overrides: dict, min_trades: int) -> dict:
    """One configuration, simulated start to finish. Top level so a worker process can run it."""
    settings = apply_overrides(base, overrides)
    trades, cfg = simulate_all(raw, settings, start)
    split, lo, hi = split_point(raw, start, end, is_ratio)
    days = max(1.0, (hi - lo).total_seconds() / 86400)
    periods = evaluate_periods(trades, cfg, balance, split)
    out = {label: dict(p["metrics"]) for label, p in periods.items()}
    for label, p in periods.items():
        out[label]["trades_per_day"] = round(len(p["taken"]) / days, 2)
    out["score"] = score_of(out["out_of_sample"], min_trades)
    ins, oos = out["in_sample"], out["out_of_sample"]
    out["robust"] = bool(
        (ins.get("return_pct") or 0) > 0 and (oos.get("return_pct") or 0) > 0
        and (ins.get("profit_factor") or 0) > 1.1 and (oos.get("profit_factor") or 0) > 1.1)
    return out


def optimize(raw: dict, settings: dict, start, end, *, iterations: int = 150, min_trades: int = 10,
             max_strategies: int = 4, strategies: list[str] | None = None, top: int = 10, seed: int = 7,
             is_ratio: float | None = None, balance: float | None = None, workers: int = 1,
             progress: Callable[[int, int], None] | None = None) -> dict:
    """Search strategy combinations and risk parameters; return the best by out-of-sample score."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    is_ratio = float(is_ratio if is_ratio is not None else settings["backtest"]["in_sample_ratio"])
    balance = float(balance if balance is not None else settings["backtest"]["initial_balance"])
    enabled = enabled_strategies(settings)
    pool = [n for n in REGISTRY if n in (strategies or enabled or list(REGISTRY))]
    if not pool:
        raise ValueError("no strategies to search over; name some, or enable at least one")
    cands = candidates(SPACE, strategy_combos(pool, max_strategies), max(1, min(iterations, 600)), seed)

    args = (raw, settings, start, end, is_ratio, balance)
    results: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [(c, ex.submit(run_candidate, *args, c["overrides"], min_trades)) for c in cands]
            for done, (c, fut) in enumerate(futures, 1):
                results.append(_collect(c, fut.result))
                if progress:
                    progress(done, len(cands))
    else:
        for done, c in enumerate(cands, 1):
            results.append(_collect(c, lambda c=c: run_candidate(*args, c["overrides"], min_trades)))
            if progress:
                progress(done, len(cands))
    results = [r for r in results if r is not None]

    # The saved settings, for an honest "is this actually better?" comparison.
    base_overrides = {f"strategies.{n}.enabled": n in enabled for n in REGISTRY}
    baseline = run_candidate(*args, base_overrides, 1)

    ranked = sorted((r for r in results if r["score"] > float("-inf")),
                    key=lambda r: (r["robust"], r["score"]), reverse=True)
    return {"top": ranked[:top], "tested": len(results), "kept": len(ranked), "min_trades": min_trades,
            "baseline": {"strategies": enabled, "metrics": baseline}}


def _collect(candidate: dict, run: Callable[[], dict]) -> dict | None:
    try:
        metrics = run()
    except Exception as exc:  # one bad combination must not end the search
        logger.warning("candidate failed: %s", exc)
        return None
    return {**candidate, "metrics": metrics, "score": metrics["score"], "robust": metrics["robust"]}
