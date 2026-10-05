"""MMF-style client account ledger for the web platform.

This models units, NAV, subscriptions, redemptions, beneficiaries, KYC status,
statements and operations workflows. It is a software ledger, not a representation
that the product is itself a licensed money-market fund.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Iterable, Optional

from firebase_security import AuthUser, firestore_client
from production_controls import AuditLogger


D = Decimal


def utcnow():
    return datetime.now(timezone.utc)


def q8(x: Decimal) -> Decimal:
    return x.quantize(D("0.00000001"), rounding=ROUND_DOWN)


@dataclass(frozen=True)
class FundConfig:
    fund_id: str = "quant-growth"
    base_currency: str = "USD"
    unit_precision: int = 8
    min_subscription: Decimal = D("10.00")
    min_redemption: Decimal = D("10.00")


class InvestorAccountService:
    """Firestore-backed client-account service using immutable cash/unit transactions."""

    def __init__(self, cfg: FundConfig | None = None, audit: AuditLogger | None = None):
        self.cfg = cfg or FundConfig()
        self.db = firestore_client()
        self.audit = audit or AuditLogger()

    def create_account(self, user: AuthUser, legal_name: str, phone: str = "",
                       country: str = "", risk_profile: str = "moderate") -> str:
        ref = self.db.collection("investor_accounts").document()
        payload = {
            "account_id": ref.id,
            "uid": user.uid,
            "email": user.email,
            "legal_name": legal_name,
            "phone": phone,
            "country": country,
            "risk_profile": risk_profile,
            "kyc_status": "not_started",
            "account_status": "pending_kyc",
            "fund_id": self.cfg.fund_id,
            "base_currency": self.cfg.base_currency,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        ref.set(payload)
        self.audit.log(user, "investor_account_created", {"account_id": ref.id})
        return ref.id

    def list_accounts(self, user: AuthUser) -> list[dict]:
        q = self.db.collection("investor_accounts")
        if user.role == "admin":
            docs = q.limit(500).stream()
        else:
            docs = q.where("uid", "==", user.uid).stream()
        out = []
        for d in docs:
            row = d.to_dict()
            row["id"] = d.id
            out.append(row)
        return out

    def get_account(self, account_id: str, user: AuthUser) -> dict:
        snap = self.db.collection("investor_accounts").document(account_id).get()
        if not snap.exists:
            raise KeyError(account_id)
        data = snap.to_dict()
        if user.role != "admin" and data.get("uid") != user.uid:
            raise PermissionError("Account access denied")
        return data

    def set_kyc_status(self, account_id: str, status: str, actor: AuthUser, note: str = "") -> None:
        if actor.role not in {"admin", "risk_approver"}:
            raise PermissionError("Role cannot update KYC")
        if status not in {"not_started", "submitted", "in_review", "verified", "rejected", "expired"}:
            raise ValueError("Invalid KYC status")
        updates = {
            "kyc_status": status,
            "account_status": "active" if status == "verified" else "pending_kyc",
            "kyc_note": note,
            "updated_at": utcnow(),
        }
        self.db.collection("investor_accounts").document(account_id).update(updates)
        self.audit.log(actor, "kyc_status_changed", {"account_id": account_id, **updates})

    def add_beneficiary(self, account_id: str, user: AuthUser, name: str,
                        relationship: str, allocation_pct: float) -> str:
        self.get_account(account_id, user)
        if not 0 < allocation_pct <= 100:
            raise ValueError("allocation_pct must be in (0,100]")
        ref = self.db.collection("investor_accounts").document(account_id).collection("beneficiaries").document()
        ref.set({
            "beneficiary_id": ref.id,
            "name": name,
            "relationship": relationship,
            "allocation_pct": float(allocation_pct),
            "created_at": utcnow(),
        })
        self.audit.log(user, "beneficiary_added", {"account_id": account_id, "beneficiary_id": ref.id})
        return ref.id

    def current_nav(self) -> Decimal:
        snap = self.db.collection("fund_nav").document(self.cfg.fund_id).get()
        if not snap.exists:
            raise RuntimeError("NAV not published")
        return D(str(snap.to_dict()["nav_per_unit"]))

    def publish_nav(self, nav_per_unit: Decimal, actor: AuthUser, valuation_date: str,
                    source_equity: Decimal, notes: str = "") -> None:
        if actor.role not in {"admin", "risk_approver"}:
            raise PermissionError("Role cannot publish NAV")
        if nav_per_unit <= 0:
            raise ValueError("NAV must be positive")
        payload = {
            "fund_id": self.cfg.fund_id,
            "nav_per_unit": str(nav_per_unit),
            "valuation_date": valuation_date,
            "source_equity": str(source_equity),
            "notes": notes,
            "published_by": actor.uid,
            "published_at": utcnow(),
        }
        self.db.collection("fund_nav").document(self.cfg.fund_id).set(payload)
        self.db.collection("fund_nav_history").document(f"{self.cfg.fund_id}__{valuation_date}").set(payload)
        self.audit.log(actor, "nav_published", payload, severity="warning")

    def _transaction(self, account_id: str, kind: str, amount: Decimal, units: Decimal,
                     nav: Decimal, actor: AuthUser, status: str = "posted",
                     external_ref: str = "", metadata: Optional[dict] = None) -> str:
        tx_id = str(uuid.uuid4())
        payload = {
            "transaction_id": tx_id,
            "account_id": account_id,
            "fund_id": self.cfg.fund_id,
            "kind": kind,
            "amount": str(amount),
            "units": str(units),
            "nav": str(nav),
            "currency": self.cfg.base_currency,
            "status": status,
            "external_ref": external_ref,
            "metadata": metadata or {},
            "created_by": actor.uid,
            "created_at": utcnow(),
        }
        self.db.collection("investment_transactions").document(tx_id).set(payload)
        self.audit.log(actor, "investment_transaction_created",
                       {"transaction_id": tx_id, "account_id": account_id, "kind": kind,
                        "amount": str(amount), "units": str(units)})
        return tx_id

    def subscribe(self, account_id: str, amount: Decimal, user: AuthUser,
                  external_ref: str = "") -> str:
        account = self.get_account(account_id, user)
        if account.get("account_status") != "active" or account.get("kyc_status") != "verified":
            raise PermissionError("Verified active account required")
        if amount < self.cfg.min_subscription:
            raise ValueError("Below minimum subscription")
        nav = self.current_nav()
        units = q8(amount / nav)
        return self._transaction(account_id, "subscription", amount, units, nav, user,
                                 external_ref=external_ref)

    def request_redemption(self, account_id: str, amount: Decimal, user: AuthUser) -> str:
        account = self.get_account(account_id, user)
        if account.get("account_status") != "active":
            raise PermissionError("Active account required")
        if amount < self.cfg.min_redemption:
            raise ValueError("Below minimum redemption")
        nav = self.current_nav()
        available_units = D(str(self.holdings(account_id)["units"]))
        requested_units = q8(amount / nav)
        if requested_units > available_units:
            raise ValueError("Insufficient units")
        return self._transaction(account_id, "redemption_request", amount, -requested_units,
                                 nav, user, status="pending")

    def approve_redemption(self, transaction_id: str, actor: AuthUser) -> None:
        if actor.role not in {"admin", "execution_approver"}:
            raise PermissionError("Role cannot approve redemption")
        ref = self.db.collection("investment_transactions").document(transaction_id)
        snap = ref.get()
        if not snap.exists:
            raise KeyError(transaction_id)
        tx = snap.to_dict()
        if tx.get("kind") != "redemption_request" or tx.get("status") != "pending":
            raise RuntimeError("Transaction is not a pending redemption")
        ref.update({"kind": "redemption", "status": "posted", "approved_by": actor.uid,
                    "approved_at": utcnow()})
        self.audit.log(actor, "redemption_approved", {"transaction_id": transaction_id},
                       severity="warning")

    def holdings(self, account_id: str) -> dict:
        docs = (self.db.collection("investment_transactions")
                .where("account_id", "==", account_id)
                .where("status", "==", "posted").stream())
        units = D("0")
        net_cash = D("0")
        for d in docs:
            tx = d.to_dict()
            units += D(str(tx["units"]))
            amount = D(str(tx["amount"]))
            if tx["kind"] == "subscription":
                net_cash += amount
            elif tx["kind"] == "redemption":
                net_cash -= amount
        nav = self.current_nav()
        value = q8(units * nav)
        gain = value - net_cash
        return {
            "account_id": account_id,
            "units": str(q8(units)),
            "nav": str(nav),
            "market_value": str(value),
            "net_contributions": str(net_cash),
            "gain_loss": str(gain),
        }

    def transactions(self, account_id: str, user: AuthUser, limit: int = 200) -> list[dict]:
        self.get_account(account_id, user)
        docs = (self.db.collection("investment_transactions")
                .where("account_id", "==", account_id)
                .limit(limit).stream())
        out = []
        for d in docs:
            row = d.to_dict()
            row["id"] = d.id
            out.append(row)
        out.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return out

    def statement(self, account_id: str, user: AuthUser) -> dict:
        account = self.get_account(account_id, user)
        return {
            "account": account,
            "holdings": self.holdings(account_id),
            "transactions": self.transactions(account_id, user),
            "generated_at": utcnow().isoformat(),
        }
