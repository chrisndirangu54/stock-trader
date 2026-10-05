"""Credential-free smoke tests for production safety primitives."""
from __future__ import annotations

import pandas as pd

from firebase_security import AuthUser, SecretVault
from production_controls import (
    BrokerRiskVerifier, MarketDataGuard, OrderStateReconciler,
    ReconciliationEngine, canonical_json, sha256_payload,
)


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise AssertionError(name)


def main():
    trader = AuthUser("u1", "t@example.com", "trader")
    viewer = AuthUser("u2", "v@example.com", "viewer")
    risk = AuthUser("u3", "r@example.com", "risk_approver")
    execution = AuthUser("u4", "e@example.com", "execution_approver")
    check("RBAC trader", trader.can_trade and not viewer.can_trade)
    check("RBAC separated approvers", risk.can_risk_approve and execution.can_execution_approve)

    p1 = {"b": 2, "a": 1}
    p2 = {"a": 1, "b": 2}
    check("canonical hashing stable", sha256_payload(p1) == sha256_payload(p2))
    check("canonical JSON stable", canonical_json(p1) == canonical_json(p2))

    fresh = MarketDataGuard(max_age_minutes=30)
    now = pd.Timestamp.now(tz="UTC")
    check("fresh equity data passes",
          fresh.check({"AAPL": now - pd.Timedelta(minutes=5)}, {"AAPL": "equity"}).passed)
    check("stale equity data blocked",
          not fresh.check({"AAPL": now - pd.Timedelta(hours=2)}, {"AAPL": "equity"}).passed)
    check("stale crypto data blocked aggressively",
          not fresh.check({"BTC/USD": now - pd.Timedelta(minutes=20)}, {"BTC/USD": "crypto"}).passed)

    rec = ReconciliationEngine(tolerance_notional=100)
    check("position reconciliation passes",
          rec.check({"AAPL": 10}, {"AAPL": 10}, {"AAPL": 200}).passed)
    check("position reconciliation blocks notional mismatch",
          not rec.check({"AAPL": 10}, {"AAPL": 11}, {"AAPL": 200}).passed)

    broker = BrokerRiskVerifier(max_order_pct_equity=0.10)
    check("broker order size passes",
          broker.check(100000, [{"symbol": "AAPL", "qty": 10}], {"AAPL": 200}).passed)
    check("broker order size blocked",
          not broker.check(100000, [{"symbol": "AAPL", "qty": 100}], {"AAPL": 200}).passed)

    ored = OrderStateReconciler()
    check("order state reconciliation passes",
          ored.reconcile([{"order_id": "1", "symbol": "AAPL", "status": "submitted"}],
                         [{"order_id": "1", "symbol": "AAPL", "status": "new"}]).passed)
    check("missing submitted order blocks",
          not ored.reconcile([{"order_id": "1", "symbol": "AAPL", "status": "submitted"}], []).passed)

    key = SecretVault.generate_key()
    check("Fernet key generated", isinstance(key, str) and len(key) > 20)
    print("SECURITY SELFTEST PASSED")


if __name__ == "__main__":
    main()
