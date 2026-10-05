"""Paper/practice broker adapters with contract-aware sizing.

Adapters are deliberately paper/practice-only. Real-money enablement should be a
separate deployment decision after broker-specific validation and reconciliation.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    asset_class: str
    multiplier: float = 1.0
    lot_size: float = 1.0
    min_qty: float = 1.0
    qty_step: float = 1.0
    currency: str = "USD"
    initial_margin_rate: float = 1.0
    maintenance_margin_rate: float = 1.0
    exchange: str = "SMART"
    expiry: Optional[str] = None


FUTURES_SPECS: Dict[str, ContractSpec] = {
    "ES": ContractSpec("ES", "future", multiplier=50, lot_size=1, min_qty=1, qty_step=1,
                       initial_margin_rate=0.12, maintenance_margin_rate=0.10, exchange="CME"),
    "NQ": ContractSpec("NQ", "future", multiplier=20, lot_size=1, min_qty=1, qty_step=1,
                       initial_margin_rate=0.14, maintenance_margin_rate=0.12, exchange="CME"),
    "GC": ContractSpec("GC", "future", multiplier=100, lot_size=1, min_qty=1, qty_step=1,
                       initial_margin_rate=0.10, maintenance_margin_rate=0.08, exchange="COMEX"),
    "CL": ContractSpec("CL", "future", multiplier=1000, lot_size=1, min_qty=1, qty_step=1,
                       initial_margin_rate=0.14, maintenance_margin_rate=0.12, exchange="NYMEX"),
    "SI": ContractSpec("SI", "future", multiplier=5000, lot_size=1, min_qty=1, qty_step=1,
                       initial_margin_rate=0.12, maintenance_margin_rate=0.10, exchange="COMEX"),
}


def round_qty(qty: float, spec: ContractSpec) -> float:
    if qty <= 0:
        return 0.0
    steps = math.floor(qty / spec.qty_step + 1e-12)
    out = steps * spec.qty_step
    if out < spec.min_qty:
        return 0.0
    return round(out, 8)


def target_quantity(equity: float, target_weight: float, price: float, spec: ContractSpec,
                    max_margin_fraction: float = 0.50) -> float:
    """Convert portfolio weight to broker quantity while respecting contract multiplier/margin."""
    if equity <= 0 or price <= 0:
        return 0.0
    target_notional = abs(equity * target_weight)
    unit_notional = price * spec.multiplier
    raw = target_notional / unit_notional
    if spec.initial_margin_rate > 0:
        margin_per_unit = unit_notional * spec.initial_margin_rate
        max_units = equity * max_margin_fraction / margin_per_unit
        raw = min(raw, max_units)
    return round_qty(raw, spec)


class BrokerAdapter(ABC):
    name = "abstract"

    @abstractmethod
    def account_equity(self) -> float: ...

    @abstractmethod
    def positions(self) -> Dict[str, float]: ...

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: str,
                            spec: ContractSpec) -> dict: ...

    def position_details(self) -> list[dict]:
        return [{"symbol": k, "qty": v} for k, v in self.positions().items()]

    def open_orders(self) -> list[dict]:
        return []

    def account_risk(self) -> dict:
        return {"equity": self.account_equity()}

    def rebalance(self, weights: Dict[str, float], prices: Dict[str, float],
                  specs: Dict[str, ContractSpec], execute: bool = False) -> list[dict]:
        equity = self.account_equity()
        held = self.positions()
        orders = []
        for symbol, weight in weights.items():
            spec = specs[symbol]
            px = float(prices[symbol])
            target = target_quantity(equity, float(weight), px, spec)
            if weight < 0:
                target *= -1
            current = float(held.get(symbol, held.get(symbol.replace("/", ""), 0.0)))
            delta = target - current
            if abs(delta) < spec.min_qty:
                continue
            side = "buy" if delta > 0 else "sell"
            item = {"broker": self.name, "symbol": symbol, "side": side,
                    "qty": abs(delta), "target_qty": target, "current_qty": current,
                    "execute": execute}
            if execute:
                item.update(self.submit_market_order(symbol, abs(delta), side, spec))
            orders.append(item)
        return orders


class AlpacaPaperAdapter(BrokerAdapter):
    name = "alpaca-paper"
    def __init__(self, key: str | None = None, secret: str | None = None):
        from alpaca.trading.client import TradingClient
        key = key or os.getenv("ALPACA_API_KEY")
        secret = secret or os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        self.client = TradingClient(key, secret, paper=True)

    def account_equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> Dict[str, float]:
        out = {}
        for p in self.client.get_all_positions():
            out[p.symbol] = float(p.qty)
            out[p.symbol.replace("/", "")] = float(p.qty)
        return out

    def position_details(self) -> list[dict]:
        rows = []
        for p in self.client.get_all_positions():
            rows.append({
                "symbol": p.symbol, "qty": float(p.qty),
                "market_value": float(getattr(p, "market_value", 0) or 0),
                "avg_entry_price": float(getattr(p, "avg_entry_price", 0) or 0),
                "unrealized_pnl": float(getattr(p, "unrealized_pl", 0) or 0),
                "unrealized_pnl_pct": float(getattr(p, "unrealized_plpc", 0) or 0),
            })
        return rows

    def open_orders(self) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        out = []
        for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
            out.append({"order_id": str(o.id), "symbol": o.symbol, "qty": float(o.qty or 0),
                        "side": str(o.side), "status": str(o.status)})
        return out

    def account_risk(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity), "buying_power": float(a.buying_power),
                "cash": float(a.cash), "trading_blocked": bool(a.trading_blocked)}

    def submit_market_order(self, symbol: str, qty: float, side: str, spec: ContractSpec) -> dict:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        tif = TimeInForce.GTC if spec.asset_class == "crypto" else TimeInForce.DAY
        req = MarketOrderRequest(symbol=symbol, qty=qty,
                                 side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                                 time_in_force=tif)
        order = self.client.submit_order(order_data=req)
        return {"submitted": True, "order_id": str(order.id)}


class OandaPracticeAdapter(BrokerAdapter):
    name = "oanda-practice"
    def __init__(self, token: str | None = None, account_id: str | None = None):
        import oandapyV20
        token = token or os.getenv("OANDA_ACCESS_TOKEN")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        if not token or not self.account_id:
            raise RuntimeError("Set OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID")
        self.api = oandapyV20.API(access_token=token, environment="practice")

    def account_equity(self) -> float:
        from oandapyV20.endpoints.accounts import AccountSummary
        r = AccountSummary(self.account_id)
        self.api.request(r)
        return float(r.response["account"]["NAV"])

    def positions(self) -> Dict[str, float]:
        from oandapyV20.endpoints.positions import OpenPositions
        r = OpenPositions(self.account_id)
        self.api.request(r)
        out = {}
        for p in r.response.get("positions", []):
            out[p["instrument"]] = float(p["long"]["units"]) + float(p["short"]["units"])
        return out

    def position_details(self) -> list[dict]:
        from oandapyV20.endpoints.positions import OpenPositions
        r = OpenPositions(self.account_id)
        self.api.request(r)
        rows = []
        for p in r.response.get("positions", []):
            qty = float(p["long"]["units"]) + float(p["short"]["units"])
            rows.append({
                "symbol": p["instrument"], "qty": qty,
                "market_value": None,
                "avg_entry_price": None,
                "unrealized_pnl": float(p.get("unrealizedPL", 0) or 0),
                "unrealized_pnl_pct": None,
            })
        return rows

    def open_orders(self) -> list[dict]:
        from oandapyV20.endpoints.orders import OrdersPending
        r = OrdersPending(self.account_id)
        self.api.request(r)
        return [{"order_id": str(o.get("id")), "symbol": o.get("instrument"),
                 "qty": float(o.get("units", 0) or 0), "side": "buy" if float(o.get("units",0) or 0) > 0 else "sell",
                 "status": o.get("state", "PENDING")} for o in r.response.get("orders", [])]

    def account_risk(self) -> dict:
        from oandapyV20.endpoints.accounts import AccountSummary
        r = AccountSummary(self.account_id)
        self.api.request(r)
        a = r.response["account"]
        return {"equity": float(a["NAV"]), "margin_available": float(a.get("marginAvailable", 0)),
                "margin_used": float(a.get("marginUsed", 0)), "margin_rate": float(a.get("marginRate", 0))}

    def submit_market_order(self, symbol: str, qty: float, side: str, spec: ContractSpec) -> dict:
        from oandapyV20.endpoints.orders import OrderCreate
        units = int(qty) * (1 if side == "buy" else -1)
        data = {"order": {"units": str(units), "instrument": symbol,
                           "timeInForce": "FOK", "type": "MARKET", "positionFill": "DEFAULT"}}
        r = OrderCreate(self.account_id, data=data)
        self.api.request(r)
        tx = r.response.get("orderFillTransaction", {})
        return {"submitted": True, "order_id": tx.get("id")}


class IBKRPaperAdapter(BrokerAdapter):
    """IBKR paper adapter for equities, FX and futures via TWS/IB Gateway."""
    name = "ibkr-paper"
    def __init__(self, host: str | None = None, port: int | None = None, client_id: int | None = None):
        from ib_insync import IB
        self.ib = IB()
        host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        port = int(port or os.getenv("IBKR_PORT", "7497"))
        client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "71"))
        self.ib.connect(host, port, clientId=client_id, readonly=False)

    def account_equity(self) -> float:
        vals = {v.tag: v.value for v in self.ib.accountValues() if v.currency in ("USD", "BASE")}
        return float(vals.get("NetLiquidation", vals.get("EquityWithLoanValue", 0.0)))

    def positions(self) -> Dict[str, float]:
        return {p.contract.localSymbol or p.contract.symbol: float(p.position) for p in self.ib.positions()}

    def position_details(self) -> list[dict]:
        rows = []
        for p in self.ib.portfolio():
            rows.append({
                "symbol": p.contract.localSymbol or p.contract.symbol,
                "qty": float(p.position),
                "market_value": float(p.marketValue),
                "avg_entry_price": float(p.averageCost),
                "unrealized_pnl": float(p.unrealizedPNL),
                "unrealized_pnl_pct": None,
            })
        return rows

    def open_orders(self) -> list[dict]:
        rows = []
        for t in self.ib.openTrades():
            rows.append({"order_id": str(t.order.orderId),
                         "symbol": t.contract.localSymbol or t.contract.symbol,
                         "qty": float(t.order.totalQuantity),
                         "side": str(t.order.action).lower(),
                         "status": str(t.orderStatus.status)})
        return rows

    def account_risk(self) -> dict:
        vals = {v.tag: v.value for v in self.ib.accountValues() if v.currency in ("USD", "BASE")}
        return {"equity": float(vals.get("NetLiquidation", 0) or 0),
                "buying_power": float(vals.get("BuyingPower", 0) or 0),
                "available_funds": float(vals.get("AvailableFunds", 0) or 0),
                "maint_margin_req": float(vals.get("MaintMarginReq", 0) or 0)}

    def _contract(self, symbol: str, spec: ContractSpec):
        from ib_insync import Stock, Forex, Future, Crypto
        if spec.asset_class == "equity":
            return Stock(symbol, "SMART", spec.currency)
        if spec.asset_class == "forex":
            return Forex(symbol.replace("_", ""))
        if spec.asset_class == "crypto":
            return Crypto(symbol, "PAXOS", spec.currency)
        if spec.asset_class == "future":
            if not spec.expiry:
                raise ValueError(f"Futures expiry required for {symbol}")
            return Future(symbol, spec.expiry, spec.exchange, currency=spec.currency, multiplier=str(spec.multiplier))
        raise ValueError(spec.asset_class)

    def submit_market_order(self, symbol: str, qty: float, side: str, spec: ContractSpec) -> dict:
        from ib_insync import MarketOrder
        contract = self._contract(symbol, spec)
        self.ib.qualifyContracts(contract)
        trade = self.ib.placeOrder(contract, MarketOrder("BUY" if side == "buy" else "SELL", qty))
        self.ib.sleep(0.2)
        return {"submitted": True, "order_id": str(trade.order.orderId)}


class IBKRFuturesPaperAdapter(IBKRPaperAdapter):
    name = "ibkr-futures-paper"
