"""Rule-based strategy engine: six strategies, a regime filter, risk sizing, a trailing
stop, a backtester with an in-sample/out-of-sample split, and a parameter search.

This is the deterministic half of the framework. The LLM agents reason about a
company; this package turns price action into entries, stops and sized trades
that can be replayed on history. Both read the same instruments.
"""
