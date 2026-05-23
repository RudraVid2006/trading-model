"""
train.py — fit and save in-sample ML models per ETF.

Models are stored under `models/{TICKER}_{model}.joblib`.

Example
-------
    python src/train.py --ticker VOO
    python src/train.py --all --model xgb --threshold 0.55
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import IN_SAMPLE_FRAC, split_in_out_sample  # noqa: E402
from src.data_loader import UNIVERSE, load_prices  # noqa: E402
from src.features import WARMUP_BARS  # noqa: E402
from src.ml_strategy import (  # noqa: E402
    DEFAULT_MODEL_TYPE,
    DEFAULT_PROB_THRESHOLD,
    ModelType,
    feature_importance,
    save_model,
    train_model,
)


def train_ticker(
    ticker: str,
    *,
    in_sample_frac: float = IN_SAMPLE_FRAC,
    model_type: ModelType = DEFAULT_MODEL_TYPE,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    show_importance: bool = False,
) -> Path:
    full = load_prices(ticker)
    in_df, _, in_period, _ = split_in_out_sample(
        full,
        in_sample_frac,
        min_warmup=WARMUP_BARS,
    )
    trained = train_model(
        in_df,
        ticker,
        prob_threshold=prob_threshold,
        model_type=model_type,
    )
    path = save_model(trained)
    print(
        f"{ticker.upper():5s}  {model_type}  threshold={prob_threshold:.2f}  "
        f"in-sample {in_period.start.date()} → {in_period.end.date()}  "
        f"→  {path}"
    )
    if show_importance:
        for name, value in feature_importance(trained).items():
            print(f"         {name:15s}  {value:.4f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models on in-sample ETF data.")
    parser.add_argument("--ticker", default="VOO", help="ETF symbol (default VOO)")
    parser.add_argument("--all", action="store_true", help="Train all universe tickers")
    parser.add_argument("--in-sample-frac", type=float, default=IN_SAMPLE_FRAC)
    parser.add_argument("--model", choices=("logreg", "xgb"), default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_PROB_THRESHOLD)
    parser.add_argument("--show-importance", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "in_sample_frac": args.in_sample_frac,
        "model_type": args.model,
        "prob_threshold": args.threshold,
        "show_importance": args.show_importance,
    }

    if args.all:
        for ticker in UNIVERSE:
            train_ticker(ticker, **kwargs)
    else:
        ticker = args.ticker.upper()
        if ticker not in UNIVERSE:
            print(f"Warning: {ticker} is not in the fixed universe {UNIVERSE}")
        train_ticker(ticker, **kwargs)


if __name__ == "__main__":
    main()
