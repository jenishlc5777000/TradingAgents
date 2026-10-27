"""How much money to risk on the next trade: compounding and (optionally) martingale.

**Risk mode** decides what "risk per trade" means:

* `money` - risk a fixed cash amount, e.g. 100 USD per trade. The stake never changes on its own,
  so compounding has no effect in this mode.
* `percent` - risk a share of the balance, e.g. 1%.

Two further switches:

* **Compounding** - size from the balance you have now, so positions grow as the account grows and
  shrink after losses. Switch it off to always size from the starting balance (constant position size).

* **Martingale** - increase the risk after a loss, hoping one win recovers the run of losses.
  It is off by default, and for good reason: the stake doubles with every loss, so a long enough
  losing run wipes the account. Eight losses in a row at 2x is 256 times the original risk.
  `max_steps` and `max_risk_pct` are hard caps that stop the sequence long before that; they are
  what makes this safe enough to experiment with on a practice account.
  The reverse (`after_win`, "anti-martingale") raises the stake after wins instead, which risks
  profits rather than capital.

Every function here is pure, so the live bot, the paper account and the backtester all size trades
the same way.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 0.02          # |R| below this counts as break-even, not a win or a loss


@dataclass
class SizingState:
    """Carried between trades. Persist it so a restart doesn't reset a sequence."""
    step: int = 0          # how many times the multiplier is currently applied
    losses: int = 0        # consecutive losses
    wins: int = 0          # consecutive wins

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "SizingState":
        d = d or {}
        return cls(step=int(d.get("step", 0)), losses=int(d.get("losses", 0)), wins=int(d.get("wins", 0)))


def _martingale(risk_cfg: dict) -> dict:
    m = risk_cfg.get("martingale") or {}
    return {
        "enabled": bool(m.get("enabled", False)),
        "mode": str(m.get("mode", "after_loss")),
        "multiplier": max(1.0, float(m.get("multiplier", 2.0))),
        "max_steps": max(0, int(m.get("max_steps", 3))),
        "max_risk_pct": float(m.get("max_risk_pct", 6.0)),
        "max_risk_money": float(m.get("max_risk_money", 600.0)),
        "reset_on_win": bool(m.get("reset_on_win", True)),
    }


def in_money(risk_cfg: dict) -> bool:
    """True when risk is set as a cash amount rather than a share of the balance."""
    return str(risk_cfg.get("risk_mode", "money")) == "money"


def daily_loss_limit(risk_cfg: dict, day_start_balance: float) -> tuple[float, str]:
    """How much may be lost in one day, as money, plus a label for messages."""
    mode = str(risk_cfg.get("daily_loss_mode", risk_cfg.get("risk_mode", "money")))
    if mode == "money":
        amount = float(risk_cfg.get("max_daily_loss_money", 300.0))
        return amount, f"{amount:,.2f}"
    pct = float(risk_cfg.get("max_daily_loss_pct", 3.0))
    return day_start_balance * pct / 100, f"{pct:g}%"


def base_balance(risk_cfg: dict, balance: float, initial_balance: float) -> float:
    """The balance position size is measured against."""
    return balance if risk_cfg.get("compounding", True) else (initial_balance or balance)


def risk_amount(risk_cfg: dict, balance: float, initial_balance: float,
                state: SizingState | None = None) -> tuple[float, str]:
    """Money to risk on the next trade, plus a short explanation for the log and the UI."""
    money_mode = in_money(risk_cfg)
    ref = base_balance(risk_cfg, balance, initial_balance)
    if money_mode:
        amount = float(risk_cfg.get("risk_per_trade_money", 100.0))
        note = f"fixed {amount:,.2f} per trade"
    else:
        base_pct = float(risk_cfg.get("risk_per_trade_pct", 1.0))
        amount = ref * base_pct / 100
        compounding = risk_cfg.get("compounding", True)
        note = f"{base_pct:g}% of {'balance' if compounding else 'starting balance'} {ref:,.2f}"

    m = _martingale(risk_cfg)
    step = min((state or SizingState()).step, m["max_steps"])
    if not m["enabled"] or step <= 0:
        return max(0.0, amount), note

    multiplied = amount * (m["multiplier"] ** step)
    cap = m["max_risk_money"] if money_mode else ref * m["max_risk_pct"] / 100
    final = min(multiplied, cap)
    word = "loss" if m["mode"] == "after_loss" else "win"
    note += f" x{m['multiplier'] ** step:g} (martingale step {step} after {step} {word}{'es' if step > 1 and word == 'loss' else 's' if step > 1 else ''})"
    if final < multiplied:
        note += (f", capped at {m['max_risk_money']:,.2f}" if money_mode
                 else f", capped at {m['max_risk_pct']:g}%")
    return max(0.0, final), note


def after_trade(risk_cfg: dict, state: SizingState, r_multiple: float) -> SizingState:
    """New state once a trade has closed. `r_multiple` is the result in R (+1 = made what was risked)."""
    m = _martingale(risk_cfg)
    win = r_multiple > EPS
    loss = r_multiple < -EPS
    wins = state.wins + 1 if win else 0
    losses = state.losses + 1 if loss else 0
    if not m["enabled"]:
        return SizingState(step=0, losses=losses, wins=wins)

    step = state.step
    if m["mode"] == "after_loss":
        if loss:
            step = min(step + 1, m["max_steps"])
        elif win and m["reset_on_win"]:
            step = 0
    else:                                   # after_win: raise the stake while winning
        if win:
            step = min(step + 1, m["max_steps"])
        elif loss:
            step = 0
    return SizingState(step=step, losses=losses, wins=wins)


def describe(risk_cfg: dict, currency: str = "") -> str:
    """One line describing the sizing rules, for logs and tooltips."""
    cur = f" {currency}" if currency else ""
    money_mode = in_money(risk_cfg)
    if money_mode:
        parts = [f"risk {float(risk_cfg.get('risk_per_trade_money', 100)):,.2f}{cur} per trade (fixed)"]
    else:
        parts = [f"risk {float(risk_cfg.get('risk_per_trade_pct', 1)):g}% per trade",
                 "compounding" if risk_cfg.get("compounding", True) else "fixed stake (no compounding)"]
    m = _martingale(risk_cfg)
    if m["enabled"]:
        cap = f"{m['max_risk_money']:,.2f}{cur}" if money_mode else f"{m['max_risk_pct']:g}%"
        parts.append(f"martingale x{m['multiplier']:g} after a {'loss' if m['mode'] == 'after_loss' else 'win'}, "
                     f"max {m['max_steps']} steps, capped at {cap}")
    return " · ".join(parts)
