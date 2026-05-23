"""
backtest.py — run and evaluate the baseline MA crossover on cached ETF data.

Workflow
--------
1. Load OHLCV from data_loader (parquet cache).
2. Split each series into in-sample (older) and out-of-sample (recent) by date.
3. Run backtesting.py separately on each slice (no parameter tuning on OOS).
4. Print risk metrics: total return, Sharpe, max drawdown, win rate.

Transaction costs are modeled via `commission` (fraction per trade, e.g. 0.001 = 0.1%).

Example
-------
    python src/backtest.py --ticker VOO
    python src/backtest.py --all
    python src/backtest.py --ticker VTI --plot
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from backtesting import Backtest

# Allow `python src/backtest.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import UNIVERSE, load_prices  # noqa: E402
from src.strategy import FAST_WINDOW, SLOW_WINDOW, MovingAverageCrossover  # noqa: E402

# --- Defaults (change here or via CLI; tune only on in-sample) ---
DEFAULT_CASH = 10_000.0
DEFAULT_COMMISSION = 0.001  # 0.1% per trade side (~ETF brokerage)
IN_SAMPLE_FRAC = 0.70  # first 70% of trading days = in-sample


@dataclass(frozen=True)
class Period:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class BacktestMetrics:
    ticker: str
    period: str
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    exposure_pct: float
    strategy: str = "MA"

    def format_row(self) -> str:
        return (
            f"{self.ticker:5s}  {self.period:12s}  "
            f"return={self.total_return_pct:>7.2f}%  "
            f"sharpe={self.sharpe:>6.2f}  "
            f"max_dd={self.max_drawdown_pct:>7.2f}%  "
            f"win_rate={self.win_rate_pct:>6.2f}%  "
            f"trades={self.num_trades:>4d}  "
            f"exposure={self.exposure_pct:>6.2f}%"
        )


def split_in_out_sample(
    df: pd.DataFrame,
    in_sample_frac: float = IN_SAMPLE_FRAC,
    *,
    min_warmup: int = SLOW_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, Period, Period]:
    """Split by time: older rows in-sample, recent rows out-of-sample."""
    if not 0.0 < in_sample_frac < 1.0:
        raise ValueError("in_sample_frac must be between 0 and 1")
    df = df.sort_index()
    split_idx = int(len(df) * in_sample_frac)
    if split_idx < min_warmup + 5 or len(df) - split_idx < 30:
        raise ValueError(
            f"Not enough rows to split (have {len(df)}, need room for "
            f"{min_warmup}-bar warmup and OOS window)"
        )
    in_df = df.iloc[:split_idx].copy()
    out_df = df.iloc[split_idx:].copy()
    in_period = Period("in_sample", in_df.index.min(), in_df.index.max())
    out_period = Period("out_of_sample", out_df.index.min(), out_df.index.max())
    return in_df, out_df, in_period, out_period


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """backtesting.py expects capitalized OHLCV columns."""
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out.dropna()


def run_single_backtest(
    ohlcv: pd.DataFrame,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
) -> tuple[Backtest, pd.Series]:
    """Run one backtest; return the Backtest instance and stats Series."""
    data = _prepare_ohlcv(ohlcv)
    bt = Backtest(
        data,
        MovingAverageCrossover,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=False,  # signal from prior close → fill next bar open
        finalize_trades=True,  # mark-to-market open positions at end of window
    )
    stats = bt.run(fast=fast, slow=slow)
    return bt, stats


def extract_metrics(
    ticker: str,
    period_label: str,
    stats: pd.Series,
    *,
    strategy: str = "MA",
) -> BacktestMetrics:
    """Pull the metrics we care about from backtesting.py output."""
    return BacktestMetrics(
        ticker=ticker,
        period=period_label,
        total_return_pct=float(stats["Return [%]"]),
        sharpe=float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else float("nan"),
        max_drawdown_pct=float(stats["Max. Drawdown [%]"]),
        win_rate_pct=float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else float("nan"),
        num_trades=int(stats["# Trades"]),
        exposure_pct=float(stats["Exposure Time [%]"]),
        strategy=strategy,
    )


def evaluate_ticker(
    ticker: str,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
    in_sample_frac: float = IN_SAMPLE_FRAC,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
    plot: bool = False,
) -> list[BacktestMetrics]:
    """Backtest one ETF on in-sample and out-of-sample windows."""
    full = load_prices(ticker)
    in_df, out_df, in_period, out_period = split_in_out_sample(full, in_sample_frac)

    results: list[BacktestMetrics] = []
    last_bt: Backtest | None = None

    for label, slice_df, period in (
        ("in_sample", in_df, in_period),
        ("out_of_sample", out_df, out_period),
    ):
        bt, stats = run_single_backtest(
            slice_df,
            cash=cash,
            commission=commission,
            fast=fast,
            slow=slow,
        )
        results.append(extract_metrics(ticker, label, stats))
        last_bt = bt
        print(
            f"  [{label}] {period.start.date()} → {period.end.date()}  "
            f"({len(slice_df)} bars)"
        )

    print()
    for m in results:
        print(m.format_row())

    if plot and last_bt is not None:
        last_bt.plot(open_browser=False)
        plt.show()

    return results


def evaluate_universe(
    tickers: tuple[str, ...] = UNIVERSE,
    **kwargs,
) -> list[BacktestMetrics]:
    """Run evaluate_ticker for each symbol and print a summary table."""
    all_metrics: list[BacktestMetrics] = []
    for ticker in tickers:
        print(f"\n{'=' * 72}")
        print(f"  {ticker}  —  MA({kwargs.get('fast', FAST_WINDOW)}/"
              f"{kwargs.get('slow', SLOW_WINDOW)})  "
              f"commission={kwargs.get('commission', DEFAULT_COMMISSION):.4f}")
        print(f"{'=' * 72}")
        all_metrics.extend(evaluate_ticker(ticker, **kwargs))
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest MA crossover on cached ETF data (in-sample + OOS)."
    )
    parser.add_argument(
        "--ticker",
        default="VOO",
        help=f"ETF symbol (default VOO). Universe: {', '.join(UNIVERSE)}",
    )
    parser.add_argument("--all", action="store_true", help="Run all universe tickers")
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    parser.add_argument("--in-sample-frac", type=float, default=IN_SAMPLE_FRAC)
    parser.add_argument("--fast", type=int, default=FAST_WINDOW)
    parser.add_argument("--slow", type=int, default=SLOW_WINDOW)
    parser.add_argument("--plot", action="store_true", help="Show backtest chart (last run)")
    args = parser.parse_args()

    kwargs = {
        "cash": args.cash,
        "commission": args.commission,
        "in_sample_frac": args.in_sample_frac,
        "fast": args.fast,
        "slow": args.slow,
        "plot": args.plot,
    }

    if args.all:
        evaluate_universe(**kwargs)
    else:
        ticker = args.ticker.upper()
        if ticker not in UNIVERSE:
            print(f"Warning: {ticker} is not in the fixed universe {UNIVERSE}")
        evaluate_ticker(ticker, **kwargs)


if __name__ == "__main__":
    main()
