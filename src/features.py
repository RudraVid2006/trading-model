"""
features.py — engineered daily features and labels for phase 3 ML.

Every feature row at date *t* uses OHLCV only through *t − 1* (`.shift(1)` at
the end). Labels use forward returns and are for training only — never fed
into the model at prediction time.

Default label: 1 if the next `LABEL_HORIZON` daily returns are positive, else 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Longest rolling window used below; rows before this are dropped after shift.
WARMUP_BARS = 201
LABEL_HORIZON = 5  # predict 5-day forward direction (long-term-ish horizon)

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "ma_spread",
    "rsi_14",
    "volume_z_20",
)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Return lagged feature matrix aligned to `ohlcv.index`.

    All columns are shifted by one bar so row *t* is actionable at *t*'s open.
    """
    close = ohlcv["Close"].sort_index()
    volume = ohlcv["Volume"].sort_index()

    ret_1 = close.pct_change(1)
    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "vol_20": ret_1.rolling(20, min_periods=20).std(),
            "ma_spread": close.rolling(50, min_periods=50).mean()
            / close.rolling(200, min_periods=200).mean()
            - 1.0,
            "rsi_14": _rsi(close, 14),
            "volume_z_20": (
                (volume - volume.rolling(20, min_periods=20).mean())
                / volume.rolling(20, min_periods=20).std()
            ),
        },
        index=close.index,
    )
    return features.shift(1)


def build_labels(
    close: pd.Series,
    *,
    horizon: int = LABEL_HORIZON,
) -> pd.Series:
    """
    Binary label at *t*: 1 if close[t+horizon] / close[t] - 1 > 0, else 0.

    Uses future prices — training/evaluation target only.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    close = close.sort_index()
    fwd_ret = close.shift(-horizon) / close - 1.0
    return (fwd_ret > 0).astype(int).rename("label")


def build_dataset(
    ohlcv: pd.DataFrame,
    *,
    horizon: int = LABEL_HORIZON,
) -> pd.DataFrame:
    """Features + label in one frame; drops rows with NaN from warmup/label tail."""
    features = build_features(ohlcv)
    labels = build_labels(ohlcv["Close"], horizon=horizon)
    dataset = features.join(labels, how="inner")
    return dataset.dropna()
