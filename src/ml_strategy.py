"""
ml_strategy.py — ML signals for long-horizon ETF trading (logreg or XGBoost).

Training discipline
-------------------
- Fit scaler + model on **in-sample** rows only.
- **In-sample backtest** uses TimeSeriesSplit out-of-fold predictions (no
  training-set leakage in the simulated trades).
- **Out-of-sample backtest** applies the model trained on all in-sample rows
  to OOS features the model never saw during fit.

Signals: 1 = long, 0 = cash (long-only). Features are already lagged in
`features.py`; backtests use `trade_on_close=False`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from backtesting import Strategy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import FEATURE_COLUMNS, WARMUP_BARS, build_dataset  # noqa: E402

ModelType = Literal["logreg", "xgb"]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_PROB_THRESHOLD = 0.55  # higher → fewer trades, less commission drag
DEFAULT_MODEL_TYPE: ModelType = "logreg"
CV_SPLITS = 5


@dataclass(frozen=True)
class TrainedModel:
    ticker: str
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    prob_threshold: float
    model_type: ModelType = DEFAULT_MODEL_TYPE


class MLLogisticStrategy(Strategy):
    """Long-only strategy driven by a precomputed `MLSignal` column on the data."""

    def init(self) -> None:
        pass

    def next(self) -> None:
        if len(self.data) < 2:
            return
        signal = int(self.data.MLSignal[-1])
        if signal == 1:
            if not self.position:
                self.buy()
        elif self.position:
            self.position.close()


def _make_pipeline(model_type: ModelType = DEFAULT_MODEL_TYPE) -> Pipeline:
    if model_type == "logreg":
        clf: Any = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Run: pip install xgboost"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "XGBoost failed to load. On macOS, try: brew install libomp"
            ) from exc
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}; use 'logreg' or 'xgb'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _model_path(ticker: str, model_type: ModelType = DEFAULT_MODEL_TYPE) -> Path:
    return MODELS_DIR / f"{ticker.upper()}_{model_type}.joblib"


def train_model(
    ohlcv: pd.DataFrame,
    ticker: str,
    *,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    model_type: ModelType = DEFAULT_MODEL_TYPE,
) -> TrainedModel:
    """Fit classifier on all rows in `ohlcv` (caller passes in-sample slice)."""
    dataset = build_dataset(ohlcv)
    if len(dataset) < CV_SPLITS * 10:
        raise ValueError(f"Not enough in-sample rows to train {ticker!r} ({len(dataset)} rows)")

    X = dataset[list(FEATURE_COLUMNS)]
    y = dataset["label"]
    pipeline = _make_pipeline(model_type)
    pipeline.fit(X, y)

    return TrainedModel(
        ticker=ticker.upper(),
        pipeline=pipeline,
        feature_columns=FEATURE_COLUMNS,
        prob_threshold=prob_threshold,
        model_type=model_type,
    )


def save_model(trained: TrainedModel) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(trained.ticker, trained.model_type)
    joblib.dump(
        {
            "pipeline": trained.pipeline,
            "feature_columns": trained.feature_columns,
            "prob_threshold": trained.prob_threshold,
            "ticker": trained.ticker,
            "model_type": trained.model_type,
        },
        path,
    )
    return path


def load_model(ticker: str, model_type: ModelType = DEFAULT_MODEL_TYPE) -> TrainedModel:
    path = _model_path(ticker, model_type)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model for {ticker!r} ({model_type}). "
            f"Run: python src/train.py --ticker {ticker} --model {model_type}"
        )
    payload: dict[str, Any] = joblib.load(path)
    return TrainedModel(
        ticker=payload["ticker"],
        pipeline=payload["pipeline"],
        feature_columns=tuple(payload["feature_columns"]),
        prob_threshold=float(payload["prob_threshold"]),
        model_type=payload.get("model_type", model_type),
    )


def feature_importance(trained: TrainedModel) -> pd.Series:
    """
    Return feature importance as a sorted Series.

    Logreg: absolute standardized coefficients. XGBoost: gain importances.
    """
    clf = trained.pipeline.named_steps["clf"]
    names = list(trained.feature_columns)
    if isinstance(clf, LogisticRegression):
        values = np.abs(clf.coef_[0])
        label = "abs_coef"
    elif type(clf).__name__ == "XGBClassifier":
        values = clf.feature_importances_
        label = "importance"
    else:
        raise TypeError(f"Unsupported classifier type: {type(clf)}")
    return (
        pd.Series(values, index=names, name=label)
        .sort_values(ascending=False)
    )


def probabilities_for_index(
    trained: TrainedModel,
    ohlcv: pd.DataFrame,
) -> pd.Series:
    """Model probability of "long" for each date (NaN where features unavailable)."""
    features = build_dataset(ohlcv)[list(trained.feature_columns)]
    if features.empty:
        return pd.Series(dtype=float, name="prob_long")

    probs = trained.pipeline.predict_proba(features)[:, 1]
    return pd.Series(probs, index=features.index, name="prob_long")


def signals_from_probabilities(
    prob_long: pd.Series,
    *,
    threshold: float = DEFAULT_PROB_THRESHOLD,
) -> pd.Series:
    """Convert probabilities to 1/0 long signal; NaN → 0 (cash)."""
    signals = (prob_long >= threshold).astype(int)
    return signals.fillna(0).astype(int).rename("MLSignal")


def in_sample_cv_signals(
    ohlcv: pd.DataFrame,
    *,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    model_type: ModelType = DEFAULT_MODEL_TYPE,
) -> pd.Series:
    """
    Out-of-fold predicted probabilities on the in-sample window.

    Each day's signal comes from a model that did not train on that day.
    """
    dataset = build_dataset(ohlcv)
    if len(dataset) < CV_SPLITS * 10:
        raise ValueError(f"Not enough rows for time-series CV ({len(dataset)})")

    X = dataset[list(FEATURE_COLUMNS)]
    y = dataset["label"]
    pipeline = _make_pipeline(model_type)
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    probs = np.full(len(X), np.nan)
    for train_idx, test_idx in tscv.split(X):
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs[test_idx] = pipeline.predict_proba(X.iloc[test_idx])[:, 1]

    prob_series = pd.Series(probs, index=X.index, name="prob_long")
    return signals_from_probabilities(prob_series, threshold=prob_threshold)


def oos_signals(trained: TrainedModel, ohlcv: pd.DataFrame) -> pd.Series:
    """Apply a model trained on in-sample data to an out-of-sample OHLCV slice."""
    probs = probabilities_for_index(trained, ohlcv)
    return signals_from_probabilities(probs, threshold=trained.prob_threshold)


def attach_signals(ohlcv: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """Add MLSignal column for backtesting.py (0 where signal unknown)."""
    data = ohlcv.copy()
    aligned = signals.reindex(data.index).fillna(0).astype(int)
    data["MLSignal"] = aligned
    return data
