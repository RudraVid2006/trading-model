# trading-model

Personal learning project: backtest trading signals on a **fixed universe of four ETFs** locally. No live trading, no real money.

## Universe (fixed)

| Symbol | Fund |
|--------|------|
| VOO | Vanguard S&P 500 ETF |
| VTI | Vanguard Total Stock Market ETF |
| QQQM | Invesco NASDAQ-100 ETF |
| SCHD | Schwab U.S. Dividend Equity ETF |

Chosen for liquidity and clean data, not as investment recommendations.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
trading-model/
  data/              # Cached parquet OHLCV (gitignored)
  models/            # Saved ML models (gitignored)
  notebooks/         # Exploratory analysis
  src/
    data_loader.py   # Download + cache prices (yfinance)
    strategy.py      # MA crossover baseline (phase 2)
    backtest.py      # Backtest MA strategy + metrics
    features.py      # Lagged features + labels (phase 3)
    ml_strategy.py   # Logistic regression train/predict/signals
    train.py         # Fit models on in-sample data
    compare.py       # MA vs ML side-by-side
  requirements.txt
```

### Data flow

1. **data_loader.py** — Pull daily adjusted OHLCV from Yahoo, cache under `data/*.parquet`.
2. **strategy.py** — MA crossover buy/hold/sell (50/200 day). Signals for day *t* use closes through *t − 1* only.
3. **features.py** — Engineered inputs (returns, volatility, MA spread, RSI, …) + 5-day forward label.
4. **ml_strategy.py** — Train logistic regression on in-sample; generate long/cash signals.
5. **compare.py** — Run MA and ML through the same backtest engine and print metrics.

## Commands

```bash
# Refresh price cache
python src/data_loader.py --refresh

# Phase 2 — MA baseline only
python src/backtest.py --ticker VOO
python src/backtest.py --all

# Phase 3 — train ML on in-sample (optional; compare.py also trains inline)
python src/train.py --ticker VOO
python src/train.py --all --model xgb --threshold 0.55

# Phase 3 — MA vs ML comparison (main entry point)
python src/compare.py --ticker VOO
python src/compare.py --ticker VOO --model xgb --threshold 0.55 --show-importance
python src/compare.py --all --model logreg --threshold 0.55

# Feature importance notebook
jupyter notebook notebooks/phase3_ml.ipynb
```

## Phase status

| Phase | Status |
|-------|--------|
| 1 — Data loader | Done |
| 2 — MA baseline + backtest | Done |
| 3 — ML vs MA comparison | Done |

## Lookahead & evaluation

- Features and MA signals use **prior close** only; backtests use `trade_on_close=False` (fills on next bar open).
- **In-sample:** MA backtest; ML uses time-series CV out-of-fold predictions (not training-set leakage).
- **Out-of-sample:** ML model fit on in-sample only, evaluated on recent 30% — same split as MA.
- Default commission: `0.001` (0.1% per trade side).
- **ML defaults:** `--model logreg`, `--threshold 0.55` (fewer trades vs 0.50).
- **XGBoost on macOS:** if `import xgboost` fails, run `brew install libomp`, then retry.
- **Focus on OOS rows** when deciding if ML beats the MA baseline.
