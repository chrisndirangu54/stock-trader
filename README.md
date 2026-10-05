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


## Institutional multi-market layer

The repository now includes a modular paper/practice architecture around the statistical-arbitrage core:

- `multi_broker.py` — Alpaca PAPER, OANDA practice, IBKR paper, and IBKR futures-paper adapters.
- `cross_asset_risk.py` — synchronized UTC decision grids, asset-class margin rules, risk contributions, class risk budgets, and a dollar/beta-neutral risk-budget projection.
- `institutional_platform.py` — applies the cross-asset risk-budget and margin overlays to the core walk-forward engine and recomputes strategy PnL/costs.
- `ai_engine.py` — Isolation Forest anomaly detection, nonlinear risk forecasting, confidence scoring, and human-readable signal explanations.
- `streamlit_app.py` — interactive UI for portfolios, backtests, AI insights, margin/risk diagnostics, and broker configuration.

### UI

```bash
pip install -r requirements-quant.txt
streamlit run streamlit_app.py
```

### Broker configuration

Alpaca PAPER:

```bash
export ALPACA_API_KEY="..."
export ALPACA_SECRET_KEY="..."
```

OANDA practice:

```bash
export OANDA_ACCESS_TOKEN="..."
export OANDA_ACCOUNT_ID="..."
```

IBKR paper / TWS or IB Gateway:

```bash
export IBKR_HOST="127.0.0.1"
export IBKR_PORT="7497"
export IBKR_CLIENT_ID="71"
```

Futures require explicit contract metadata, especially contract multiplier, exchange and expiry. Broker-reported margin and contract details should override the conservative local research defaults.

The execution adapters are intentionally paper/practice-oriented. Real-money deployment should be separated from research and enabled only after validating broker contract mapping, trading calendars, order lifecycle, margin behavior, reconciliation, slippage and failure recovery.
