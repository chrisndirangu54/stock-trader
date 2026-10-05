# stock-trader
A stock trading app using Python 3, SQLite, Flask, Jinja, HTML, and CSS.


## Quantitative statistical-arbitrage engine

`trading_system_sota.py` is a multi-asset, cross-sectional statistical-arbitrage research engine inspired by publicly known institutional quant principles. It uses causal features, walk-forward retraining with an embargo, ensemble prediction, residual momentum, regime-aware alpha blending, Ledoit-Wolf covariance shrinkage, cost-aware constrained portfolio optimization, volatility targeting, and bootstrap/deflated-Sharpe-style diagnostics.

### Install

```bash
pip install -r requirements-quant.txt
python trading_system_sota.py selftest
```

### Research universes

```bash
python trading_system_sota.py backtest --universe stocks
python trading_system_sota.py backtest --universe crypto
python trading_system_sota.py backtest --universe forex
python trading_system_sota.py backtest --universe commodities
python trading_system_sota.py backtest --universe commodity_etfs
```

Custom Yahoo Finance symbols are also accepted:

```bash
python trading_system_sota.py backtest --tickers AAPL MSFT NVDA AMZN
python trading_system_sota.py backtest --tickers BTC-USD ETH-USD SOL-USD
python trading_system_sota.py backtest --tickers EURUSD=X GBPUSD=X USDJPY=X
python trading_system_sota.py backtest --tickers GC=F SI=F CL=F NG=F
```

### Live PAPER execution

The included Alpaca adapter supports US equities/ETFs and supported crypto pairs. Spot FX and commodity futures remain research/backtest-only until a compatible broker adapter is configured. Commodity ETFs can be paper-traded through the equity route.

```bash
export ALPACA_API_KEY="..."
export ALPACA_SECRET_KEY="..."

python trading_system_sota.py live --universe stocks
python trading_system_sota.py live --universe crypto
python trading_system_sota.py live --universe crypto --execute
```

The system is research software, not a claim to reproduce Renaissance Technologies' proprietary models and not investment advice.
