"""Quant-desk runtime helpers for broker health, paper positions, blotter and P&L."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from multi_broker import AlpacaPaperAdapter, IBKRPaperAdapter, OandaPracticeAdapter


@dataclass
class BrokerSnapshot:
    broker: str
    configured: bool
    connected: bool
    equity: float | None
    positions: Dict[str, float]
    message: str
    timestamp: str


def env_configured(name: str) -> bool:
    if name == "Alpaca PAPER":
        return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))
    if name == "OANDA Practice":
        return bool(os.getenv("OANDA_ACCESS_TOKEN") and os.getenv("OANDA_ACCOUNT_ID"))
    if name == "IBKR Paper":
        return True  # local TWS/Gateway may be present even without env vars
    return False


def broker_snapshot(name: str) -> BrokerSnapshot:
    configured = env_configured(name)
    now = datetime.now(timezone.utc).isoformat()
    try:
        if name == "Alpaca PAPER":
            adapter = AlpacaPaperAdapter()
        elif name == "OANDA Practice":
            adapter = OandaPracticeAdapter()
        elif name == "IBKR Paper":
            adapter = IBKRPaperAdapter()
        else:
            raise ValueError(name)
        equity = float(adapter.account_equity())
        positions = adapter.positions()
        return BrokerSnapshot(name, configured, True, equity, positions, "Connected", now)
    except Exception as e:
        return BrokerSnapshot(name, configured, False, None, {}, str(e), now)


def all_broker_snapshots() -> list[BrokerSnapshot]:
    return [broker_snapshot(x) for x in ("Alpaca PAPER", "OANDA Practice", "IBKR Paper")]


def snapshots_frame(snaps: list[BrokerSnapshot]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        rows.append({
            "broker": s.broker,
            "configured": s.configured,
            "connected": s.connected,
            "equity": s.equity,
            "positions": len(s.positions),
            "status": s.message,
            "timestamp": s.timestamp,
        })
    return pd.DataFrame(rows)


def positions_frame(snaps: list[BrokerSnapshot]) -> pd.DataFrame:
    rows = []
    for snap in snaps:
        for symbol, qty in snap.positions.items():
            rows.append({"broker": snap.broker, "symbol": symbol, "qty": qty})
    return pd.DataFrame(rows, columns=["broker", "symbol", "qty"])


def estimate_live_pnl(position_df: pd.DataFrame, prices: Dict[str, float]) -> pd.DataFrame:
    """Mark positions using supplied prices; this is exposure, not broker official realized P&L."""
    if position_df.empty:
        return position_df.assign(mark_price=pd.Series(dtype=float), market_value=pd.Series(dtype=float))
    out = position_df.copy()
    out["mark_price"] = out["symbol"].map(prices)
    out["market_value"] = out["qty"] * out["mark_price"]
    return out


def append_blotter(session_state: Any, orders: list[dict]) -> None:
    if "order_blotter" not in session_state:
        session_state.order_blotter = []
    ts = datetime.now(timezone.utc).isoformat()
    for o in orders:
        row = dict(o)
        row["timestamp"] = ts
        session_state.order_blotter.append(row)


def blotter_frame(session_state: Any) -> pd.DataFrame:
    rows = getattr(session_state, "order_blotter", [])
    return pd.DataFrame(rows)
