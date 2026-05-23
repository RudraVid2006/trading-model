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
  notebooks/         # Exploratory analysis
  src/
    data_loader.py   # Download + cache prices (yfinance)
    strategy.py      # Signal logic (MA crossover baseline)
    backtest.py      # Run backtests + report metrics
  requirements.txt
```

### Data flow

1. **data_loader.py** — Pull daily adjusted OHLCV from Yahoo, cache under `data/*.parquet`.
2. **strategy.py** — Turn prices into buy/hold/sell signals (50/200 MA crossover). Signals for day *t* use closes through *t − 1* only.
3. **backtest.py** — Simulate trades with [backtesting.py](https://kernc.github.io/backtesting.py/), split **in-sample** (older 70%) vs **out-of-sample** (recent 30%), report return, Sharpe, max drawdown, win rate.

## Commands

```bash
# Refresh price cache
python src/data_loader.py --refresh

# Backtest one ETF (in-sample + out-of-sample)
python src/backtest.py --ticker VOO

# All four ETFs
python src/backtest.py --all

# Chart (last OOS run)
python src/backtest.py --ticker VOO --plot
```

## Phase status

| Phase | Status |
|-------|--------|
| 1 — Data loader | Done |
| 2 — MA baseline + backtest | Done |
| 3 — ML strategy vs baseline | Not started (commit baseline first) |

## Lookahead & evaluation

- Signals are shifted so decisions use **prior close** only; backtests use `trade_on_close=False` (fills on next bar open).
- Tune parameters (MA windows, etc.) on **in-sample** data only. **Out-of-sample** is for honest comparison, not tuning.
- Default commission: `0.001` (0.1% per trade side).
