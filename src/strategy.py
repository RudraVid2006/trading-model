"""
strategy.py — transparent signal logic for the baseline benchmark.

Moving-average crossover (default 50-day vs 200-day):
  - **Buy**  : fast MA > slow MA  (bullish regime → hold long)
  - **Sell** : fast MA < slow MA  (bearish regime → exit to cash)
  - **Hold** : fast MA == slow MA, or no change vs prior signal

Lookahead guard
---------------
Signals for calendar day *t* are computed using closes through *t − 1* only.
We shift the regime by one bar so `signal[t]` is actionable at *t*'s open
(aligned with backtesting.py's default: orders from `next()` fill next bar).

This module is usable on its own (vectorized signals for analysis) and via
`MovingAverageCrossover`, a backtesting.py `Strategy` for simulation.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
from backtesting import Strategy

# Defaults for the baseline; tune only on in-sample data in backtest.py.
FAST_WINDOW = 50
SLOW_WINDOW = 200


class Signal(IntEnum):
    """Discrete action labels (long-only ETF universe)."""

    SELL = -1  # exit to cash
    HOLD = 0  # no change
    BUY = 1  # enter / stay long


def moving_average_crossover_signals(
    close: pd.Series,
    *,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
) -> pd.DataFrame:
    """
    Build shifted buy/hold/sell signals from a close price series.

    Returns a DataFrame indexed like `close` with:
      fast_ma, slow_ma — simple moving averages (unshifted, for charts)
      regime         — 1 bullish, -1 bearish, 0 tie (shifted, no lookahead)
      signal         — Signal enum values (shifted)
      signal_change  — True on days the shifted signal changes
    """
    if fast >= slow:
        raise ValueError(f"fast window ({fast}) must be smaller than slow ({slow})")
    if fast < 1 or slow < 1:
        raise ValueError("windows must be positive integers")

    close = close.sort_index()
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()

    raw_regime = np.where(
        fast_ma > slow_ma,
        1,
        np.where(fast_ma < slow_ma, -1, 0),
    )
    # Regime known after prior close → shift by 1 bar.
    regime = pd.Series(raw_regime, index=close.index).shift(1)

    signal = regime.map({1: Signal.BUY, -1: Signal.SELL, 0: Signal.HOLD})
    prev = signal.shift(1)
    signal_change = signal.notna() & prev.notna() & (signal != prev)

    return pd.DataFrame(
        {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "regime": regime,
            "signal": signal,
            "signal_change": signal_change,
        },
        index=close.index,
    )


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average for backtesting.py indicators."""
    series = pd.Series(values, dtype=float)
    return series.rolling(period, min_periods=period).mean().to_numpy()


class MovingAverageCrossover(Strategy):
    """
    Long-only MA crossover for backtesting.py.

    Decisions at bar *t* use fast/slow MAs through bar *t − 1* (`[-2]`).
    Orders are long-only (buy to enter, close to exit); no shorting.
    """

    fast = FAST_WINDOW
    slow = SLOW_WINDOW

    def init(self) -> None:
        close = self.data.Close
        self.sma_fast = self.I(_sma, close, self.fast)
        self.sma_slow = self.I(_sma, close, self.slow)

    def next(self) -> None:
        # Need prior bar MAs (through yesterday's close).
        if len(self.data) < self.slow + 1:
            return
        if np.isnan(self.sma_fast[-2]) or np.isnan(self.sma_slow[-2]):
            return

        fast_prev, slow_prev = self.sma_fast[-2], self.sma_slow[-2]

        if fast_prev > slow_prev:
            if not self.position:
                self.buy()
        elif fast_prev < slow_prev:
            if self.position:
                self.position.close()
