"""Bloomberg/quant-desk-style Streamlit dashboard for the multi-asset stat-arb platform."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_engine import MarketAICopilot, copilot_answer
from cross_asset_risk import enforce_margin, portfolio_margin
from desk_runtime import all_broker_snapshots, append_blotter, blotter_frame, positions_frame, snapshots_frame
from desk_auth import require_authentication, auth_sidebar, secret_vault_panel, approval_panel, kill_switch_panel
from production_controls import ProductionGate, AuditLogger
from institutional_platform import latest_institutional_weights, run_institutional_backtest
from multi_broker import AlpacaPaperAdapter, ContractSpec, IBKRPaperAdapter, OandaPracticeAdapter
from trading_system_sota import (
    ASSET_UNIVERSES, Config, build_feature_cube, infer_asset_class, infer_regime,
    load_panel, performance_metrics, universe_tickers,
)

st.set_page_config(page_title="Quant Desk", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
auth_user = require_authentication()
auth_sidebar(auth_user)

st.markdown(
    """
    <style>
      .stApp {background:#070b11;color:#d8e2ef;}
      [data-testid="stSidebar"] {background:#0b111a;border-right:1px solid #1b2a3a;}
      [data-testid="stMetric"] {background:#0c1420;border:1px solid #1b2a3a;padding:10px 12px;border-radius:3px;}
      [data-testid="stMetricLabel"] {font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#7f94aa;}
      [data-testid="stMetricValue"] {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
      .desk-title {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.05rem;letter-spacing:.14em;color:#dce9f5;}
      .desk-sub {font-size:.78rem;color:#70869c;margin-top:-8px;margin-bottom:12px;}
      .status-ok {color:#7ee787;font-weight:700;}
      .status-bad {color:#ff7b72;font-weight:700;}
      div[data-testid="stTabs"] button {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.78rem;}
      .stDataFrame {border:1px solid #182536;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="desk-title">◈ MULTI-ASSET QUANT DESK / STATISTICAL ARBITRAGE</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="desk-sub">Research + paper/practice execution • UTC {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} • No unrestricted live-money control</div>',
    unsafe_allow_html=True,
)

for key, default in (("analysis", None), ("order_blotter", []), ("chat", []), ("production_ticket", None)):
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### DESK CONTROL")
    universe = st.selectbox("Universe", sorted(ASSET_UNIVERSES), index=sorted(ASSET_UNIVERSES).index("etf_stat_arb"))
    period = st.selectbox("History", ["2y", "5y", "10y", "12y", "max"], index=3)
    custom = st.text_input("Custom Yahoo symbols", placeholder="AAPL,MSFT,BTC-USD,EURUSD=X")
    refresh_seconds = st.select_slider("Broker refresh", options=[5, 10, 15, 30, 60], value=15)
    run = st.button("RUN WALK-FORWARD", type="primary", use_container_width=True)
    if st.button("REFRESH SCREEN", use_container_width=True):
        st.rerun()
    st.divider()
    st.caption("Execution adapters are paper/practice only. Futures require an explicit dated contract before submission.")

if run:
    tickers = tuple(x.strip() for x in custom.split(",") if x.strip()) or universe_tickers(universe)
    cfg = Config(tickers=tickers, period=period)
    with st.spinner("Loading market data, fitting walk-forward models and optimizing cross-asset risk..."):
        panel = load_panel(tickers, period)
        out = run_institutional_backtest(panel, cfg)
        weights = latest_institutional_weights(panel, cfg)
        features, close, dollar_vol = build_feature_cube(panel, cfg)
        st.session_state.analysis = (cfg, panel, out, weights, features, close, dollar_vol)

if st.session_state.analysis is None:
    st.info("Select a universe and run the desk. No synthetic performance is shown before real data is loaded.")
    st.stop()

cfg, panel, out, weights, features, close, dollar_vol = st.session_state.analysis
metrics = performance_metrics(out["net"])
last_date = close.index[-1]
regime = infer_regime(close, last_date)
ret = close.pct_change()
asset_classes = {
    s: ("future" if infer_asset_class(s) == "commodity_future" else infer_asset_class(s))
    for s in cfg.tickers
}
margin_safe = enforce_margin(weights, asset_classes)
margin_used = portfolio_margin(margin_safe, asset_classes)
daily_pnl = float(out["net"].iloc[-1]) if len(out["net"]) else 0.0
equity_curve = (1 + out["net"]).cumprod()
gross = float(weights.abs().sum())
net = float(weights.sum())

k = st.columns(9)
k[0].metric("Regime", regime.upper())
k[1].metric("Sharpe", f"{metrics['sharpe']:.2f}")
k[2].metric("CAGR", f"{metrics['cagr']:.1%}")
k[3].metric("Vol", f"{metrics['vol']:.1%}")
k[4].metric("Max DD", f"{metrics['max_dd']:.1%}")
k[5].metric("Daily P&L", f"{daily_pnl:+.2%}")
k[6].metric("Gross", f"{gross:.2f}x")
k[7].metric("Net", f"{net:+.3f}")
k[8].metric("Margin", f"{margin_used:.1%}")

st.markdown("#### MARKET TAPE")
norm = close.tail(120) / close.tail(120).iloc[0] * 100.0
fig_tape = go.Figure()
for c in norm.columns[:12]:
    fig_tape.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c, line={"width": 1.2}))
fig_tape.update_layout(
    height=300, margin=dict(l=5, r=5, t=10, b=5), template="plotly_dark",
    paper_bgcolor="#070b11", plot_bgcolor="#070b11", legend_orientation="h",
)
st.plotly_chart(fig_tape, use_container_width=True)

tabs = st.tabs(["OVERVIEW", "ASSET CLASSES", "RISK", "POSITIONS + P&L", "ORDER BLOTTER", "BROKERS", "AI COPILOT", "PRODUCTION GATE", "SECRETS"])

with tabs[0]:
    a, b = st.columns([1.35, 1])
    with a:
        st.markdown("##### Strategy equity / cumulative P&L")
        comparison = pd.DataFrame({
            "StatArb": (1 + out["net"]).cumprod(),
            "Equal weight": (1 + out["equal_weight"].reindex(out["net"].index).fillna(0)).cumprod(),
            "Momentum L/S": (1 + out["momentum_ls"].reindex(out["net"].index).fillna(0)).cumprod(),
        })
        st.line_chart(comparison, height=350)
    with b:
        st.markdown("##### Target book")
        book = pd.DataFrame({
            "weight": weights,
            "margin_safe": margin_safe,
            "asset_class": pd.Series(asset_classes),
            "last": close.iloc[-1],
            "1d": ret.iloc[-1],
            "20d_vol": ret.tail(20).std() * np.sqrt(252),
        }).sort_values("weight", ascending=False)
        st.dataframe(
            book.style.format({
                "weight": "{:+.2%}", "margin_safe": "{:+.2%}", "last": "{:,.4f}",
                "1d": "{:+.2%}", "20d_vol": "{:.1%}",
            }),
            use_container_width=True, height=350,
        )
    p1, p2, p3 = st.columns(3)
    p1.metric("Backtest cumulative", f"{equity_curve.iloc[-1]-1:+.1%}")
    p2.metric("Annualized turnover", f"{out['weights'].diff().abs().sum(axis=1).mean()*252:.1f}x")
    p3.metric("Mean modeled cost/day", f"{out['costs'].mean():.4%}")

with tabs[1]:
    classes = sorted(set(asset_classes.values()))
    class_tabs = st.tabs([c.upper() for c in classes])
    for ct, cls in zip(class_tabs, classes):
        with ct:
            syms = [s for s in cfg.tickers if asset_classes[s] == cls]
            if not syms:
                st.info("No assets in this class.")
                continue
            c1, c2 = st.columns([1.4, 1])
            with c1:
                rel = close[syms].tail(180) / close[syms].tail(180).iloc[0] * 100
                st.line_chart(rel, height=360)
            with c2:
                class_book = pd.DataFrame({
                    "target": weights.reindex(syms),
                    "last": close[syms].iloc[-1],
                    "1d": ret[syms].iloc[-1],
                    "vol20": ret[syms].tail(20).std() * np.sqrt(252),
                })
                st.dataframe(
                    class_book.style.format({"target": "{:+.2%}", "1d": "{:+.2%}", "vol20": "{:.1%}"}),
                    use_container_width=True,
                )

with tabs[2]:
    st.markdown("##### Correlation heatmap")
    corr = ret.tail(126).corr()
    heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    heat.update_layout(height=max(420, 32 * len(corr)), template="plotly_dark", paper_bgcolor="#070b11")
    st.plotly_chart(heat, use_container_width=True)
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("##### Risk concentration")
        vol20 = ret.tail(20).std() * np.sqrt(252)
        proxy = (weights.abs() * vol20).sort_values(ascending=False)
        risk_df = pd.DataFrame({
            "risk_proxy": proxy,
            "weight": weights.reindex(proxy.index),
            "class": pd.Series(asset_classes).reindex(proxy.index),
        })
        st.dataframe(risk_df.style.format({"risk_proxy": "{:.3f}", "weight": "{:+.2%}"}), use_container_width=True)
    with r2:
        st.markdown("##### Rolling strategy risk")
        roll = pd.DataFrame({
            "20d vol": out["net"].rolling(20).std() * np.sqrt(252),
            "63d vol": out["net"].rolling(63).std() * np.sqrt(252),
            "drawdown": equity_curve / equity_curve.cummax() - 1,
        })
        st.line_chart(roll, height=330)

with tabs[3]:
    st.markdown("##### Model / strategy P&L")
    pnl_frame = pd.DataFrame({
        "daily_return": out["net"],
        "cumulative_return": equity_curve - 1,
        "cost": out["costs"].reindex(out["net"].index),
    }).tail(120)
    st.dataframe(pnl_frame.style.format("{:+.4%}"), use_container_width=True, height=300)

    st.markdown("##### Paper/practice broker positions")
    @st.fragment(run_every=f"{refresh_seconds}s")
    def live_positions_panel():
        snaps = all_broker_snapshots()
        pf = positions_frame(snaps)
        if pf.empty:
            st.caption("No connected paper/practice positions available.")
        else:
            st.dataframe(pf, use_container_width=True)
        total_paper_equity = sum(s.equity or 0.0 for s in snaps if s.connected)
        st.metric("Connected paper/practice equity", f"USD {total_paper_equity:,.2f}")
    live_positions_panel()

with tabs[4]:
    st.markdown("##### Paper order blotter")
    blotter = blotter_frame(st.session_state)
    if blotter.empty:
        st.caption("No paper orders generated in this UI session.")
    else:
        st.dataframe(blotter.sort_values("timestamp", ascending=False), use_container_width=True, height=320)

    st.markdown("##### Rebalance control")
    broker_choice = st.selectbox("Paper/practice broker", ["Alpaca PAPER", "OANDA Practice", "IBKR Paper"])
    execute_paper = st.checkbox("Submit to connected PAPER/PRACTICE account", value=False)
    confirmation = st.text_input("Type PAPER to unlock submission", type="password") if execute_paper else ""

    def map_symbol(symbol: str, broker: str) -> tuple[str, ContractSpec]:
        cls = asset_classes[symbol]
        if broker == "OANDA Practice":
            if cls != "forex":
                raise ValueError("OANDA Practice accepts FX symbols only.")
            raw = symbol.replace("=X", "")
            b = raw[:3] + "_" + raw[3:6]
            return b, ContractSpec(b, "forex", min_qty=1, qty_step=1, initial_margin_rate=0.0333)
        if broker == "Alpaca PAPER":
            if cls not in {"equity", "crypto"}:
                raise ValueError("Alpaca PAPER accepts equities/ETFs and crypto only.")
            from trading_system_sota import alpaca_symbol
            b = alpaca_symbol(symbol)
            return b, ContractSpec(
                b, cls,
                min_qty=0.00000001 if cls == "crypto" else 1,
                qty_step=0.00000001 if cls == "crypto" else 1,
                initial_margin_rate=1.0 if cls == "crypto" else 0.5,
            )
        if cls == "future":
            raise ValueError("Continuous Yahoo futures are research-only. Select an explicit dated contract for IBKR paper.")
        raw = symbol.replace("=X", "")
        b = raw
        return b, ContractSpec(b, cls, min_qty=1, qty_step=1, initial_margin_rate=0.0333 if cls == "forex" else 0.5)

    if st.button("GENERATE / SUBMIT PAPER REBALANCE", type="primary"):
        if execute_paper and confirmation != "PAPER":
            st.error("Submission blocked. Type PAPER exactly to confirm the paper/practice action.")
        else:
            try:
                adapter = {
                    "Alpaca PAPER": AlpacaPaperAdapter,
                    "OANDA Practice": OandaPracticeAdapter,
                    "IBKR Paper": IBKRPaperAdapter,
                }[broker_choice]()
                mapped_weights, mapped_prices, specs = {}, {}, {}
                for s, w in margin_safe.items():
                    if abs(float(w)) < 1e-6:
                        continue
                    try:
                        bs, spec = map_symbol(s, broker_choice)
                    except ValueError:
                        continue
                    mapped_weights[bs] = float(w)
                    mapped_prices[bs] = float(close[s].iloc[-1])
                    specs[bs] = spec
                orders = adapter.rebalance(mapped_weights, mapped_prices, specs, execute=execute_paper)
                append_blotter(st.session_state, orders)
                st.session_state.production_ticket = {
                    "broker": broker_choice,
                    "orders": orders,
                    "prices": mapped_prices,
                    "asset_classes": {k: specs[k].asset_class for k in specs},
                    "expected_qty": {o["symbol"]: float(o.get("target_qty", 0.0)) for o in orders},
                }
                st.success(f"{len(orders)} paper/practice order actions generated and staged for approval.")
                st.dataframe(pd.DataFrame(orders), use_container_width=True)
            except Exception as e:
                st.error(f"Paper rebalance failed safely: {e}")

with tabs[5]:
    st.markdown("##### Broker health")
    @st.fragment(run_every=f"{refresh_seconds}s")
    def broker_health_panel():
        snaps = all_broker_snapshots()
        sf = snapshots_frame(snaps)
        st.dataframe(sf, use_container_width=True)
        cols = st.columns(3)
        for i, snap in enumerate(snaps):
            with cols[i]:
                if snap.connected:
                    st.markdown(f'<span class="status-ok">● {snap.broker} CONNECTED</span>', unsafe_allow_html=True)
                    st.metric("Equity", f"USD {(snap.equity or 0):,.2f}")
                    st.caption(f"{len(snap.positions)} open position(s)")
                else:
                    st.markdown(f'<span class="status-bad">● {snap.broker} OFFLINE</span>', unsafe_allow_html=True)
                    st.caption(snap.message[:180])
    broker_health_panel()
    st.code(
        "ALPACA_API_KEY / ALPACA_SECRET_KEY\\n"
        "OANDA_ACCESS_TOKEN / OANDA_ACCOUNT_ID\\n"
        "IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID",
        language="text",
    )

with tabs[6]:
    ai = MarketAICopilot(cfg.seed)
    fdate = features.index.get_level_values("date").max()
    today = features.xs(fdate, level="date").reindex(cfg.tickers).fillna(0.0)
    scan = ai.anomaly_scan(today)
    risk_fc = ai.forecast_risk(ret)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### AI signal monitor")
        rows = []
        for s in cfg.tickers:
            confidence = min(1.0, abs(float(weights.get(s, 0.0))) / max(cfg.name_cap, 1e-9))
            rows.append({
                "symbol": s,
                "class": asset_classes[s],
                "target": float(weights.get(s, 0.0)),
                "confidence": confidence,
                "anomaly": bool(scan.loc[s, "anomaly"]),
                "anomaly_score": float(scan.loc[s, "anomaly_score"]),
                "risk_forecast": risk_fc.get(s, {}).get("predicted_abs_return"),
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("symbol").style.format({
                "target": "{:+.2%}", "confidence": "{:.0%}",
                "anomaly_score": "{:.3f}", "risk_forecast": "{:.2%}",
            }),
            use_container_width=True, height=370,
        )

    with right:
        st.markdown("##### Copilot")
        for msg in st.session_state.chat[-8:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        prompt = st.chat_input("Ask about risk, positions, longs/shorts, or a signal driver...")
        if prompt:
            st.session_state.chat.append({"role": "user", "content": prompt})
            answer = copilot_answer(
                prompt, metrics=metrics, weights=weights, features_today=today,
                regime=regime, risk_forecast=risk_fc,
            )
            st.session_state.chat.append({"role": "assistant", "content": answer})
            st.rerun()



with tabs[7]:
    st.markdown("##### Production readiness")
    st.caption(
        "This panel does not expose unrestricted real-money execution. It proves whether a staged ticket "
        "would satisfy the production gate. Every control is fail-closed."
    )
    kill_switch_panel(auth_user)
    ticket = st.session_state.get("production_ticket")
    if not ticket:
        st.info("Generate a rebalance ticket in ORDER BLOTTER first.")
    else:
        st.write(f"Staged broker: **{ticket['broker']}**")
        st.dataframe(pd.DataFrame(ticket["orders"]), use_container_width=True)
        approval_panel(auth_user, ticket["orders"], ticket["broker"])

        approval_id = st.session_state.get("approval_request_id", "")
        if approval_id and st.button("EVALUATE PRODUCTION GATE", type="primary"):
            try:
                snaps = all_broker_snapshots()
                snap = next((x for x in snaps if x.broker == ticket["broker"]), None)
                if snap is None or not snap.connected:
                    raise RuntimeError("Selected broker is not connected")
                # Yahoo daily bars intentionally fail the intraday freshness test; a production market-data
                # feed must replace them before live-money enablement.
                ts_map = {}
                class_map = {}
                for original in cfg.tickers:
                    cls = asset_classes[original]
                    # map known broker symbol if present, otherwise retain original
                    broker_symbol = next((o["symbol"] for o in ticket["orders"] if o["symbol"].replace("/", "") in original.replace("-USD","").replace("=X","").replace("/","")), original)
                    ts_map[broker_symbol] = close[original].index[-1]
                    class_map[broker_symbol] = cls
                gate = ProductionGate()
                results = gate.evaluate(
                    actor=auth_user,
                    broker_name=ticket["broker"],
                    broker_equity=float(snap.equity or 0.0),
                    orders=ticket["orders"],
                    prices=ticket["prices"],
                    market_timestamps={k: ts_map.get(k, close.index[-1]) for k in ticket["prices"]},
                    asset_classes={k: ticket["asset_classes"].get(k, class_map.get(k, "equity")) for k in ticket["prices"]},
                    expected_qty=ticket["expected_qty"],
                    broker_qty=snap.positions,
                    approval_request_id=approval_id,
                )
                gate_df = pd.DataFrame([{"control": r.name, "passed": r.passed, "detail": r.detail} for r in results])
                st.dataframe(gate_df, use_container_width=True)
                if gate.all_pass(results):
                    st.success("All production gates pass. The repository still has no unrestricted real-money submit button.")
                else:
                    st.error("Production blocked. One or more controls failed.")
            except Exception as e:
                st.error(f"Production gate failed closed: {e}")

with tabs[8]:
    secret_vault_panel(auth_user)
    if auth_user.role == "admin":
        st.divider()
        st.markdown("##### Recent audit events")
        try:
            events = AuditLogger().recent(100)
            if events:
                audit_df = pd.DataFrame(events)
                cols = [c for c in ["timestamp","actor_email","actor_role","action","severity","request_id","payload_hash"] if c in audit_df.columns]
                st.dataframe(audit_df[cols], use_container_width=True, height=320)
            else:
                st.caption("No audit events yet.")
        except Exception as e:
            st.error(f"Audit log unavailable: {e}")
    st.divider()
    st.markdown("##### Security posture")
    st.write(
        "Broker secrets are encrypted with Fernet before Firestore storage. The QUANT_MASTER_KEY must be "
        "kept outside Firestore (for example in a cloud secret manager or deployment environment). "
        "Admins can rotate secrets; non-admin users cannot retrieve plaintext through the UI."
    )

st.caption(
    "Research software. Performance is historical/out-of-sample backtest output, not a guarantee. "
    "Production eligibility is fail-closed behind Firebase identity, RBAC, audit, kill switch, "
    "daily loss, stale-data, reconciliation, broker-risk and multi-person approval gates."
)
