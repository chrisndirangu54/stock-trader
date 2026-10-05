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


## Quant desk dashboard

Launch the professional Streamlit desk:

```bash
pip install -r requirements-quant.txt
streamlit run streamlit_app.py
```

The desk now includes:

- normalized live market tape for the loaded universe;
- strategy equity, cumulative P&L, turnover and modeled trading costs;
- target book with margin-safe weights and current risk statistics;
- asset-class workspaces for equities, crypto, FX and futures research;
- 126-day correlation heatmap and rolling volatility/drawdown panels;
- connected paper/practice broker status and positions;
- session order blotter;
- paper/practice rebalance preview and guarded submission controls;
- AI anomaly monitor, nonlinear risk forecast and grounded copilot chat.

Paper submission requires an explicit confirmation phrase in the UI. Continuous Yahoo futures remain research-only; IBKR futures execution requires a dated contract with expiry and exchange metadata.

The UI intentionally does not provide unrestricted real-money execution.


## Production security gate

The dashboard is now protected by Firebase Authentication and role-based access control. The production path is fail-closed and remains unavailable unless all required controls pass.

### Roles

- `viewer` — read-only research access.
- `trader` — may create trade proposals.
- `risk_approver` — may perform stage-1 risk approval.
- `execution_approver` — may perform stage-2 execution approval.
- `admin` — manages users/kill switch/secrets, but approval separation is still enforced.

A production ticket requires three distinct identities: proposer, risk approver and execution approver.

### Firebase setup

Enable Email/Password authentication in Firebase Authentication and create a Firestore database. Provide a server-side Firebase Admin credential using either:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/secure/path/service-account.json"
```

or:

```bash
export FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

The Streamlit sign-in flow also requires the Firebase Web API key:

```bash
export FIREBASE_WEB_API_KEY="..."
```

Deploy `firestore.rules`. Sensitive trading collections deny all browser/client access; the server-side Firebase Admin SDK performs authorized access.

Bootstrap the first role after creating the Auth user:

```bash
python bootstrap_firebase.py FIREBASE_UID user@example.com admin
```

### Encrypted secrets

Generate a Fernet master key once:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Store it outside Firestore and outside Git:

```bash
export QUANT_MASTER_KEY="..."
```

Broker secrets stored through the admin UI are encrypted before being written to Firestore. Only ciphertext is persisted there. For production, keep `QUANT_MASTER_KEY` in a managed cloud secret manager rather than a local `.env` file.

### Mandatory production controls

Before any ticket can become production-eligible, the gate checks:

1. verified Firebase identity and authorized role;
2. global kill switch is disengaged;
3. daily account loss is above `MAX_DAILY_LOSS_PCT`;
4. market data is within `MAX_MARKET_DATA_AGE_MIN`;
5. local target positions reconcile with broker-reported positions;
6. order sizes satisfy broker/equity risk limits;
7. trade payload has completed two approval stages without mutation or expiry.

Additional broker-native state is available through `account_risk()` and `open_orders()` for post-trade/order-state reconciliation.

Useful settings:

```bash
export MAX_DAILY_LOSS_PCT="0.02"
export MAX_MARKET_DATA_AGE_MIN="30"
export MAX_ORDER_PCT_EQUITY="0.10"
export RECON_TOLERANCE_USD="250"
export MIN_BROKER_EQUITY="100"
```

Daily research bars intentionally fail the intraday production freshness test. A real-time broker or institutional market-data feed must replace Yahoo daily bars before a production ticket can pass.

### Audit and execution ledger

Security-sensitive actions are written to Firestore with timestamp, actor, role, request ID and SHA-256 payload hash. This includes:

- kill-switch changes;
- secret rotation/deletion;
- trade proposal;
- risk approval;
- execution approval;
- rejection;
- gate evaluation;
- daily-loss breaches;
- execution ledger records.

The admin desk exposes recent audit events, while `ExecutionLedger` stores hashed execution payload/result pairs for reconciliation.

The repository still intentionally omits an unrestricted real-money submit button. Passing all controls means a ticket is **eligible**, not automatically executed.


## React web investor portal

The project now has a dedicated React/Vite investor portal in `web/` and a FastAPI backend in `web_api.py`.

### Web architecture

```text
Firebase Authentication
        │
        ▼
React / Vite investor portal
        │  Firebase ID token
        ▼
FastAPI /api
        │
        ├── investor accounts / KYC
        ├── subscriptions / redemptions
        ├── units / NAV / statements
        ├── admin user management
        ├── audit / approvals / risk controls
        └── quant platform integration
        │
        ▼
Firebase Admin SDK + Firestore
```

The investor-facing product is intentionally modeled as an MMF/asset-management style account system—units, NAV, subscriptions, redemptions, statements, beneficiaries and KYC workflows—but this software does **not** by itself make the trading strategy or business a licensed money-market fund.

### Investor portal features

- Firebase email/password login;
- separate `investor` and internal staff roles;
- investor onboarding and account creation;
- KYC/account-status visibility;
- unit balance, latest NAV, portfolio value, net contributions and gain/loss;
- subscription funding requests;
- redemption requests;
- account transaction history and printable statements;
- responsive desktop/mobile institutional UI.

Subscriptions are fail-safe: a client payment reference creates only a **pending subscription request**. Units are posted only after an authorized operations user confirms cleared cash. The posted subscription is repriced at the current published NAV at approval time.

### Fund operations console

Admin users get additional React operations tabs for:

- user creation, disable/enable and role management;
- investor/KYC queue;
- pending subscription cash-verification queue;
- pending redemption approval queue;
- NAV publication and NAV history.

Internal roles are:

```text
investor
viewer
trader
risk_approver
execution_approver
admin
```

### Local development

Backend:

```bash
pip install -r requirements-quant.txt
uvicorn web_api:app --reload --port 8000
```

Frontend:

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000` during development.

### Firebase Hosting + Cloud Run deployment

The repository includes:

- `Dockerfile.web-api` for the FastAPI service;
- `firebase.json` for Hosting and the `/api/**` Cloud Run rewrite;
- `firestore.rules`;
- `firestore.indexes.json`;
- `web/.env.example`.

Build and deploy the API to a Cloud Run service named `quant-fund-api` in `us-central1`, or update `firebase.json` to match your chosen service/region.

Example:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/quant-fund-api -f Dockerfile.web-api .
gcloud run deploy quant-fund-api \
  --image gcr.io/PROJECT_ID/quant-fund-api \
  --region us-central1
```

Build the React app and deploy Firebase Hosting/Firestore configuration:

```bash
cd web
npm install
npm run build
cd ..

firebase deploy --only hosting,firestore
```

Set the Firebase client variables from `web/.env.example` before building the frontend. Keep Firebase Admin credentials and `QUANT_MASTER_KEY` server-side only; never expose them through `VITE_*` variables.

### Regulatory boundary

Before offering this to real investors, connect the software to the legally required fund manager/custodian/trustee/administrator, KYC/AML process, payment settlement rails, valuation policy, disclosures and jurisdiction-specific licensing/approval requirements. The code currently provides operational controls and ledger mechanics, not regulatory authorization.
