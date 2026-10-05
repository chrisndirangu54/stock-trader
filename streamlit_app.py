"""Streamlit UI for the multi-asset statistical-arbitrage research platform."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trading_system_sota import (ASSET_UNIVERSES, Config, build_feature_cube, infer_regime,
                                 latest_weights, load_panel, performance_metrics, run_backtest,
                                 universe_tickers)
from ai_engine import MarketAICopilot
from cross_asset_risk import enforce_margin

st.set_page_config(page_title="StatArb Research Console", layout="wide")
st.title("Multi-Asset Statistical-Arbitrage Research Console")
st.caption("Research/paper-trading interface. Broker execution remains dry-run unless explicitly enabled in code/config.")

with st.sidebar:
    universe = st.selectbox("Universe", sorted(ASSET_UNIVERSES), index=sorted(ASSET_UNIVERSES).index("etf_stat_arb"))
    period = st.selectbox("History", ["2y", "5y", "10y", "12y", "max"], index=3)
    custom = st.text_input("Custom Yahoo symbols (comma separated)", "")
    run = st.button("Load & analyze", type="primary")

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if run:
    tickers = tuple(x.strip() for x in custom.split(",") if x.strip()) or universe_tickers(universe)
    cfg = Config(tickers=tickers, period=period)
    with st.spinner("Downloading data and running walk-forward analysis..."):
        panel = load_panel(tickers, period)
        out = run_backtest(panel, cfg)
        weights = latest_weights(panel, cfg)
        features, close, _ = build_feature_cube(panel, cfg)
        st.session_state.analysis = (cfg, panel, out, weights, features, close)

if st.session_state.analysis is None:
    st.info("Choose a universe and run the analysis. The UI does not invent results before data is loaded.")
    st.stop()

cfg, panel, out, weights, features, close = st.session_state.analysis
m = performance_metrics(out["net"])
cols = st.columns(5)
cols[0].metric("Sharpe", f"{m['sharpe']:.2f}")
cols[1].metric("CAGR", f"{m['cagr']:.1%}")
cols[2].metric("Volatility", f"{m['vol']:.1%}")
cols[3].metric("Max drawdown", f"{m['max_dd']:.1%}")
cols[4].metric("Assets", str(len(cfg.tickers)))

t1, t2, t3, t4, t5 = st.tabs(["Portfolio", "Backtest", "AI Insights", "Risk", "Brokers"])

with t1:
    st.subheader("Latest target weights")
    st.dataframe(weights.sort_values().rename("target_weight").to_frame(), use_container_width=True)
    st.bar_chart(weights.sort_values())

with t2:
    equity = (1 + out["net"]).cumprod()
    st.line_chart(equity.rename("model_equity"))
    comp = pd.DataFrame({"model": out["net"], "equal_weight": out["equal_weight"],
                         "momentum_ls": out["momentum_ls"]}).cumsum()
    st.line_chart(comp)
    st.dataframe(out["weights"].tail(30), use_container_width=True)

with t3:
    ai = MarketAICopilot(cfg.seed)
    date = features.index.get_level_values("date").max()
    today = features.xs(date, level="date").reindex(cfg.tickers).fillna(0.0)
    scan = ai.anomaly_scan(today)
    regime = infer_regime(close, date)
    risk_fc = ai.forecast_risk(close.pct_change())
    rows = []
    for s in cfg.tickers:
        conf = min(1.0, abs(float(weights.get(s, 0.0))) / max(cfg.name_cap, 1e-9))
        explanation = ai.explain_signal(s, float(weights.get(s, 0.0)), today.loc[s], regime,
                                        bool(scan.loc[s, "anomaly"]), conf)
        rows.append({"symbol": s, "weight": float(weights.get(s, 0.0)),
                     "anomaly": bool(scan.loc[s, "anomaly"]),
                     "anomaly_score": float(scan.loc[s, "anomaly_score"]),
                     "risk_forecast": risk_fc.get(s, {}).get("predicted_abs_return"),
                     "explanation": explanation})
    st.dataframe(pd.DataFrame(rows).set_index("symbol"), use_container_width=True)
    for r in rows:
        st.write(r["explanation"])

with t4:
    asset_classes = {}
    from trading_system_sota import infer_asset_class
    for s in cfg.tickers:
        c = infer_asset_class(s)
        asset_classes[s] = "future" if c == "commodity_future" else c
    margin_safe = enforce_margin(weights, asset_classes)
    st.write("Margin-aware scaled weights")
    st.dataframe(pd.DataFrame({"raw": weights, "margin_safe": margin_safe}), use_container_width=True)
    st.caption("Margin assumptions are conservative defaults for research. Broker-reported margin always overrides local estimates.")

with t5:
    st.write("Configured adapters")
    st.code("Alpaca PAPER: ALPACA_API_KEY / ALPACA_SECRET_KEY\n"
            "OANDA practice: OANDA_ACCESS_TOKEN / OANDA_ACCOUNT_ID\n"
            "IBKR paper: IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID")
    st.warning("This UI intentionally does not expose a real-money execution button. Validate fills, margin, contract metadata, calendars and reconciliation in paper/practice first.")
