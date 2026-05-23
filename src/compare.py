"""
compare.py — MA baseline vs ML (logreg or XGBoost) on in-sample and OOS.

Example
-------
    python src/compare.py --ticker VOO
    python src/compare.py --all --model xgb --threshold 0.55
    python src/compare.py --ticker VOO --show-importance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from backtesting import Backtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (  # noqa: E402
    DEFAULT_CASH,
    DEFAULT_COMMISSION,
    IN_SAMPLE_FRAC,
    BacktestMetrics,
    extract_metrics,
    split_in_out_sample,
)
from src.data_loader import UNIVERSE, load_prices  # noqa: E402
from src.features import WARMUP_BARS  # noqa: E402
from src.ml_strategy import (  # noqa: E402
    DEFAULT_MODEL_TYPE,
    DEFAULT_PROB_THRESHOLD,
    MLLogisticStrategy,
    ModelType,
    attach_signals,
    feature_importance,
    in_sample_cv_signals,
    oos_signals,
    train_model,
)
from src.strategy import FAST_WINDOW, SLOW_WINDOW, MovingAverageCrossover  # noqa: E402


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index().dropna()


def run_ma_backtest(
    ohlcv: pd.DataFrame,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
) -> pd.Series:
    data = _prepare_ohlcv(ohlcv)
    bt = Backtest(
        data,
        MovingAverageCrossover,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=False,
        finalize_trades=True,
    )
    return bt.run(fast=fast, slow=slow)


def run_ml_backtest(
    ohlcv: pd.DataFrame,
    signals: pd.Series,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
) -> pd.Series:
    data = attach_signals(_prepare_ohlcv(ohlcv), signals)
    bt = Backtest(
        data,
        MLLogisticStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=False,
        finalize_trades=True,
    )
    return bt.run()


def _print_metrics(metrics: list[BacktestMetrics], strategy: str) -> None:
    for m in metrics:
        print(f"  [{strategy:2s}] {m.format_row()}")


def compare_ticker(
    ticker: str,
    *,
    cash: float = DEFAULT_CASH,
    commission: float = DEFAULT_COMMISSION,
    in_sample_frac: float = IN_SAMPLE_FRAC,
    fast: int = FAST_WINDOW,
    slow: int = SLOW_WINDOW,
    model_type: ModelType = DEFAULT_MODEL_TYPE,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    show_importance: bool = False,
) -> None:
    full = load_prices(ticker)
    in_df, out_df, in_period, out_period = split_in_out_sample(
        full,
        in_sample_frac,
        min_warmup=WARMUP_BARS,
    )

    print(f"  [in_sample]     {in_period.start.date()} → {in_period.end.date()}  ({len(in_df)} bars)")
    print(f"  [out_of_sample] {out_period.start.date()} → {out_period.end.date()}  ({len(out_df)} bars)")
    print(f"  ML model={model_type}  prob_threshold={prob_threshold:.2f}")
    print()

    ma_metrics: list[BacktestMetrics] = []
    ml_metrics: list[BacktestMetrics] = []

    ma_in = run_ma_backtest(in_df, cash=cash, commission=commission, fast=fast, slow=slow)
    ma_oos = run_ma_backtest(out_df, cash=cash, commission=commission, fast=fast, slow=slow)
    ma_metrics.append(extract_metrics(ticker, "in_sample", ma_in, strategy="MA"))
    ma_metrics.append(extract_metrics(ticker, "out_of_sample", ma_oos, strategy="MA"))

    ml_in_signals = in_sample_cv_signals(
        in_df,
        prob_threshold=prob_threshold,
        model_type=model_type,
    )
    trained = train_model(
        in_df,
        ticker,
        prob_threshold=prob_threshold,
        model_type=model_type,
    )
    ml_oos_signals = oos_signals(trained, out_df)

    if show_importance:
        print("  Feature importance (in-sample fit):")
        for name, value in feature_importance(trained).items():
            print(f"    {name:15s}  {value:.4f}")
        print()

    ml_in = run_ml_backtest(in_df, ml_in_signals, cash=cash, commission=commission)
    ml_oos = run_ml_backtest(out_df, ml_oos_signals, cash=cash, commission=commission)
    ml_label = model_type.upper()
    ml_metrics.append(extract_metrics(ticker, "in_sample", ml_in, strategy=ml_label))
    ml_metrics.append(extract_metrics(ticker, "out_of_sample", ml_oos, strategy=ml_label))

    header = (
        f"{'':7s}  {'period':12s}  {'return':>10s}  {'sharpe':>7s}  "
        f"{'max_dd':>10s}  {'win_rate':>9s}  {'trades':>6s}  {'exposure':>9s}"
    )
    print(header)
    _print_metrics(ma_metrics, "MA")
    _print_metrics(ml_metrics, ml_label)
    print()


def compare_universe(
    tickers: tuple[str, ...] = UNIVERSE,
    **kwargs,
) -> None:
    model_type = kwargs.get("model_type", DEFAULT_MODEL_TYPE)
    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(
            f"  {ticker}  —  MA({kwargs.get('fast', FAST_WINDOW)}/"
            f"{kwargs.get('slow', SLOW_WINDOW)}) vs ML({model_type})  "
            f"threshold={kwargs.get('prob_threshold', DEFAULT_PROB_THRESHOLD):.2f}  "
            f"commission={kwargs.get('commission', DEFAULT_COMMISSION):.4f}"
        )
        print(f"{'=' * 80}")
        compare_ticker(ticker, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MA baseline vs ML on in-sample + OOS.")
    parser.add_argument("--ticker", default="VOO")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH)
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION)
    parser.add_argument("--in-sample-frac", type=float, default=IN_SAMPLE_FRAC)
    parser.add_argument("--fast", type=int, default=FAST_WINDOW)
    parser.add_argument("--slow", type=int, default=SLOW_WINDOW)
    parser.add_argument(
        "--model",
        choices=("logreg", "xgb"),
        default=DEFAULT_MODEL_TYPE,
        help="ML classifier (default logreg)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PROB_THRESHOLD,
        help="Min predicted prob to go/stay long (default 0.55)",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Print feature importance before metrics",
    )
    args = parser.parse_args()

    kwargs = {
        "cash": args.cash,
        "commission": args.commission,
        "in_sample_frac": args.in_sample_frac,
        "fast": args.fast,
        "slow": args.slow,
        "model_type": args.model,
        "prob_threshold": args.threshold,
        "show_importance": args.show_importance,
    }

    if args.all:
        compare_universe(**kwargs)
    else:
        ticker = args.ticker.upper()
        if ticker not in UNIVERSE:
            print(f"Warning: {ticker} is not in the fixed universe {UNIVERSE}")
        compare_ticker(ticker, **kwargs)


if __name__ == "__main__":
    main()
