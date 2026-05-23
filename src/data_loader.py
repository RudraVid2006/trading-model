"""
data_loader.py — pull and cache daily OHLCV price data for the ETF universe.

Each ticker is downloaded once via yfinance with auto_adjust=True (so OHLC are
already adjusted for dividends and splits) and cached to a parquet file under
`data/`. Subsequent calls read from the cache, so reruns are fast and work
offline. Pass refresh=True to force a re-download.

The cache always stores **max-available history** per ticker. Optional start/end
arguments slice the returned DataFrame at read time — they do NOT shrink the
on-disk cache. That way one cached file serves every date-range experiment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

# Fixed universe. Symbol -> fund identity (verified):
#   VOO  : Vanguard S&P 500 ETF
#   VTI  : Vanguard Total Stock Market ETF
#   QQQM : Invesco NASDAQ-100 ETF (lower-fee successor to QQQ)
#   SCHD : Schwab U.S. Dividend Equity ETF
UNIVERSE: tuple[str, ...] = ("VOO", "VTI", "QQQM", "SCHD")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.upper()}.parquet"


def _download(ticker: str) -> pd.DataFrame:
    """Pull max-available daily history for one ticker from Yahoo."""
    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker!r}")
    # Strip tz so parquet round-trips cleanly and downstream date math is simple.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df[_OHLCV_COLS].copy()


def load_prices(
    ticker: str,
    *,
    refresh: bool = False,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Return cached OHLCV for `ticker`. Downloads via yfinance if the cache is
    missing or `refresh=True`. `start`/`end` slice the returned frame only.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    if refresh or not path.exists():
        df = _download(ticker)
        df.to_parquet(path)
    else:
        df = pd.read_parquet(path)
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def load_universe(
    tickers: Iterable[str] = UNIVERSE,
    *,
    refresh: bool = False,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every ticker into a {symbol: DataFrame} dict."""
    return {t: load_prices(t, refresh=refresh, start=start, end=end) for t in tickers}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load/refresh the ETF universe cache.")
    parser.add_argument("--refresh", action="store_true", help="Force re-download from Yahoo")
    args = parser.parse_args()

    frames = load_universe(refresh=args.refresh)
    for sym, df in frames.items():
        first = df.index.min().date()
        last = df.index.max().date()
        print(
            f"{sym:5s} {len(df):>5} rows  "
            f"{first} -> {last}  "
            f"last close={df['Close'].iloc[-1]:.2f}"
        )
