"""Production execution safety controls: audit, kill switch, stale-data, loss limits,
reconciliation, broker checks, and two-stage approvals.

All controls are fail-closed. If Firestore is unavailable or a required check cannot be
verified, production eligibility is denied.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from firebase_security import AuthUser, firestore_client


def utcnow():
    return datetime.now(timezone.utc)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_payload(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


class AuditLogger:
    COLLECTION = "audit_logs"

    def __init__(self):
        self.db = firestore_client()

    def log(self, actor: AuthUser | None, action: str, payload: dict,
            severity: str = "info", request_id: str | None = None) -> str:
        doc = self.db.collection(self.COLLECTION).document()
        event = {
            "timestamp": utcnow(),
            "actor_uid": actor.uid if actor else None,
            "actor_email": actor.email if actor else None,
            "actor_role": actor.role if actor else None,
            "action": action,
            "severity": severity,
            "request_id": request_id,
            "payload": payload,
            "payload_hash": sha256_payload(payload),
        }
        doc.set(event)
        return doc.id


class KillSwitch:
    DOC = ("risk_controls", "global")

    def __init__(self, audit: AuditLogger | None = None):
        self.db = firestore_client()
        self.audit = audit or AuditLogger()

    def status(self) -> dict:
        snap = self.db.collection(self.DOC[0]).document(self.DOC[1]).get()
        data = snap.to_dict() if snap.exists else {}
        return {
            "engaged": bool(data.get("kill_switch", True)),  # fail closed until explicitly initialized
            "reason": data.get("kill_switch_reason", "Uninitialized production gate"),
            "updated_at": data.get("updated_at"),
            "updated_by": data.get("updated_by"),
        }

    def set(self, engaged: bool, reason: str, actor: AuthUser) -> None:
        if actor.role != "admin":
            raise PermissionError("Only admin can change the global kill switch")
        payload = {
            "kill_switch": bool(engaged),
            "kill_switch_reason": reason,
            "updated_at": utcnow(),
            "updated_by": actor.uid,
        }
        self.db.collection(self.DOC[0]).document(self.DOC[1]).set(payload, merge=True)
        self.audit.log(actor, "kill_switch_changed", payload, severity="critical" if engaged else "warning")


class DailyLossGuard:
    """Persist start-of-day equity and block when loss threshold is breached."""

    def __init__(self, max_loss_pct: float = 0.02, audit: AuditLogger | None = None):
        self.max_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", max_loss_pct))
        self.db = firestore_client()
        self.audit = audit or AuditLogger()

    def _doc(self, broker: str, day: str):
        return self.db.collection("daily_risk").document(f"{broker}__{day}")

    def ensure_start_equity(self, broker: str, equity: float, actor: AuthUser | None = None) -> dict:
        day = utcnow().date().isoformat()
        ref = self._doc(broker, day)
        snap = ref.get()
        if not snap.exists:
            data = {"broker": broker, "date": day, "start_equity": float(equity),
                    "created_at": utcnow(), "created_by": actor.uid if actor else None}
            ref.set(data)
            self.audit.log(actor, "daily_equity_baseline_created", data)
            return data
        return snap.to_dict()

    def check(self, broker: str, equity: float, actor: AuthUser | None = None) -> GateResult:
        base = self.ensure_start_equity(broker, equity, actor)
        start = float(base["start_equity"])
        if start <= 0:
            return GateResult("daily_loss_limit", False, "Invalid start-of-day equity")
        pnl_pct = (float(equity) - start) / start
        passed = pnl_pct > -self.max_loss_pct
        detail = f"daily P&L {pnl_pct:+.2%}; limit {-self.max_loss_pct:.2%}"
        if not passed:
            self.audit.log(actor, "daily_loss_limit_breached",
                           {"broker": broker, "equity": equity, "start_equity": start,
                            "pnl_pct": pnl_pct, "limit": self.max_loss_pct},
                           severity="critical")
        return GateResult("daily_loss_limit", passed, detail)


class MarketDataGuard:
    def __init__(self, max_age_minutes: int = 30):
        self.max_age_minutes = int(os.getenv("MAX_MARKET_DATA_AGE_MIN", max_age_minutes))

    def check(self, timestamps: Dict[str, Any], asset_classes: Dict[str, str]) -> GateResult:
        now = pd.Timestamp.now(tz="UTC")
        stale = []
        for symbol, ts in timestamps.items():
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            age = (now - t).total_seconds() / 60
            cls = asset_classes.get(symbol, "equity")
            # Daily research bars are not acceptable for production intraday execution.
            max_age = 10 if cls in {"crypto", "forex", "future"} else self.max_age_minutes
            if age > max_age:
                stale.append(f"{symbol}:{age:.0f}m")
        return GateResult(
            "market_data_freshness",
            len(stale) == 0,
            "fresh" if not stale else "stale " + ", ".join(stale[:8]),
        )


class ReconciliationEngine:
    """Compare intended/local positions against broker-reported positions."""

    def __init__(self, tolerance_notional: float = 250.0):
        self.tolerance_notional = float(os.getenv("RECON_TOLERANCE_USD", tolerance_notional))

    def check(self, expected_qty: Dict[str, float], broker_qty: Dict[str, float],
              prices: Dict[str, float]) -> GateResult:
        mismatches = []
        symbols = set(expected_qty) | set(broker_qty)
        for s in symbols:
            dq = float(broker_qty.get(s, 0.0)) - float(expected_qty.get(s, 0.0))
            px = float(prices.get(s, 0.0) or 0.0)
            notional = abs(dq * px)
            if notional > self.tolerance_notional:
                mismatches.append(f"{s}:{notional:.0f}USD")
        return GateResult(
            "position_reconciliation",
            len(mismatches) == 0,
            "matched" if not mismatches else "mismatch " + ", ".join(mismatches[:8]),
        )


class BrokerRiskVerifier:
    """Broker-side sanity checks before order release."""

    def __init__(self, max_order_pct_equity: float = 0.10, min_equity: float = 100.0):
        self.max_order_pct_equity = float(os.getenv("MAX_ORDER_PCT_EQUITY", max_order_pct_equity))
        self.min_equity = float(os.getenv("MIN_BROKER_EQUITY", min_equity))

    def check(self, broker_equity: float, orders: Iterable[dict], prices: Dict[str, float]) -> GateResult:
        if not math.isfinite(broker_equity) or broker_equity < self.min_equity:
            return GateResult("broker_risk", False, f"equity {broker_equity} below minimum")
        breaches = []
        for o in orders:
            symbol = o["symbol"]
            qty = abs(float(o.get("qty", 0.0)))
            px = float(prices.get(symbol, 0.0) or 0.0)
            multiplier = float(o.get("multiplier", 1.0) or 1.0)
            order_notional = qty * px * multiplier
            if order_notional > broker_equity * self.max_order_pct_equity:
                breaches.append(f"{symbol}:{order_notional/broker_equity:.1%}")
        return GateResult(
            "broker_risk",
            len(breaches) == 0,
            "within broker limits" if not breaches else "oversized " + ", ".join(breaches[:8]),
        )


class ApprovalWorkflow:
    """Proposal -> risk approval -> execution approval. Distinct users required."""

    COLLECTION = "trade_approvals"
    TERMINAL_STATES = {"rejected", "executed", "expired", "cancelled"}

    def __init__(self, audit: AuditLogger | None = None):
        self.db = firestore_client()
        self.audit = audit or AuditLogger()

    def propose(self, actor: AuthUser, broker: str, orders: list[dict],
                rationale: str, expires_minutes: int = 30) -> str:
        if not actor.can_trade:
            raise PermissionError("Role cannot propose trades")
        request_id = str(uuid.uuid4())
        now = utcnow()
        payload = {
            "request_id": request_id,
            "broker": broker,
            "orders": orders,
            "order_hash": sha256_payload(orders),
            "rationale": rationale,
            "state": "proposed",
            "proposer_uid": actor.uid,
            "proposer_email": actor.email,
            "risk_approver_uid": None,
            "execution_approver_uid": None,
            "created_at": now,
            "expires_at": now + pd.Timedelta(minutes=expires_minutes).to_pytimedelta(),
            "updated_at": now,
        }
        self.db.collection(self.COLLECTION).document(request_id).set(payload)
        self.audit.log(actor, "trade_proposed", payload, request_id=request_id)
        return request_id

    def get(self, request_id: str) -> dict:
        snap = self.db.collection(self.COLLECTION).document(request_id).get()
        if not snap.exists:
            raise KeyError(request_id)
        return snap.to_dict()

    def risk_approve(self, request_id: str, actor: AuthUser) -> None:
        req = self.get(request_id)
        if not actor.can_risk_approve:
            raise PermissionError("Role cannot risk-approve")
        if actor.uid == req["proposer_uid"]:
            raise PermissionError("Proposer cannot risk-approve own trade")
        if req["state"] != "proposed":
            raise RuntimeError(f"Request state is {req['state']}")
        self.db.collection(self.COLLECTION).document(request_id).update({
            "state": "risk_approved",
            "risk_approver_uid": actor.uid,
            "risk_approved_at": utcnow(),
            "updated_at": utcnow(),
        })
        self.audit.log(actor, "trade_risk_approved", {"request_id": request_id}, request_id=request_id)

    def execution_approve(self, request_id: str, actor: AuthUser) -> None:
        req = self.get(request_id)
        if not actor.can_execution_approve:
            raise PermissionError("Role cannot execution-approve")
        if req["state"] != "risk_approved":
            raise RuntimeError(f"Request state is {req['state']}")
        distinct = {req["proposer_uid"], req["risk_approver_uid"], actor.uid}
        if len(distinct) < 3:
            raise PermissionError("Execution approver must be distinct from proposer and risk approver")
        self.db.collection(self.COLLECTION).document(request_id).update({
            "state": "execution_approved",
            "execution_approver_uid": actor.uid,
            "execution_approved_at": utcnow(),
            "updated_at": utcnow(),
        })
        self.audit.log(actor, "trade_execution_approved", {"request_id": request_id}, request_id=request_id)

    def reject(self, request_id: str, actor: AuthUser, reason: str) -> None:
        self.db.collection(self.COLLECTION).document(request_id).update({
            "state": "rejected", "rejected_by": actor.uid, "rejection_reason": reason,
            "updated_at": utcnow(),
        })
        self.audit.log(actor, "trade_rejected", {"request_id": request_id, "reason": reason},
                       severity="warning", request_id=request_id)

    def execution_eligible(self, request_id: str, current_orders: list[dict]) -> GateResult:
        req = self.get(request_id)
        if req["state"] != "execution_approved":
            return GateResult("two_stage_approval", False, f"state={req['state']}")
        exp = req.get("expires_at")
        if exp is not None:
            exp_ts = pd.Timestamp(exp)
            now = pd.Timestamp.now(tz="UTC")
            if exp_ts.tzinfo is None:
                exp_ts = exp_ts.tz_localize("UTC")
            if now > exp_ts:
                return GateResult("two_stage_approval", False, "approval expired")
        if sha256_payload(current_orders) != req["order_hash"]:
            return GateResult("two_stage_approval", False, "order payload changed after approval")
        return GateResult("two_stage_approval", True, "proposer + risk approver + execution approver complete")

    def mark_executed(self, request_id: str, actor: AuthUser, broker_result: list[dict]) -> None:
        self.db.collection(self.COLLECTION).document(request_id).update({
            "state": "executed", "executed_at": utcnow(), "executed_by": actor.uid,
            "broker_result_hash": sha256_payload(broker_result), "updated_at": utcnow(),
        })
        self.audit.log(actor, "trade_executed", {"request_id": request_id, "broker_result": broker_result},
                       severity="critical", request_id=request_id)


class ProductionGate:
    def __init__(self):
        self.audit = AuditLogger()
        self.kill = KillSwitch(self.audit)
        self.loss = DailyLossGuard(audit=self.audit)
        self.fresh = MarketDataGuard()
        self.recon = ReconciliationEngine()
        self.broker = BrokerRiskVerifier()
        self.approvals = ApprovalWorkflow(self.audit)

    def evaluate(self, *, actor: AuthUser, broker_name: str, broker_equity: float,
                 orders: list[dict], prices: Dict[str, float], market_timestamps: Dict[str, Any],
                 asset_classes: Dict[str, str], expected_qty: Dict[str, float],
                 broker_qty: Dict[str, float], approval_request_id: str) -> list[GateResult]:
        ks = self.kill.status()
        results = [
            GateResult("authenticated_role", actor.can_trade, f"role={actor.role}"),
            GateResult("kill_switch", not ks["engaged"], ks["reason"]),
            self.loss.check(broker_name, broker_equity, actor),
            self.fresh.check(market_timestamps, asset_classes),
            self.recon.check(expected_qty, broker_qty, prices),
            self.broker.check(broker_equity, orders, prices),
            self.approvals.execution_eligible(approval_request_id, orders),
        ]
        self.audit.log(actor, "production_gate_evaluated", {
            "broker": broker_name,
            "approval_request_id": approval_request_id,
            "results": [asdict(r) for r in results],
        }, severity="warning" if not all(r.passed for r in results) else "info",
           request_id=approval_request_id)
        return results

    @staticmethod
    def all_pass(results: list[GateResult]) -> bool:
        return bool(results) and all(r.passed for r in results)
