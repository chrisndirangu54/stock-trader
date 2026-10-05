"""FastAPI web API for the React client portal and admin desk."""
from __future__ import annotations

import os
from decimal import Decimal
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from firebase_security import FirebaseAuthManager, AuthUser
from investment_accounts import InvestorAccountService
from production_controls import AuditLogger, KillSwitch, ApprovalWorkflow
from user_management import UserManagementService

app = FastAPI(title="Quant Fund Web API", version="1.0.0")

origins = [x.strip() for x in os.getenv("WEB_ORIGINS", "http://localhost:5173").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

auth = FirebaseAuthManager()
accounts = InvestorAccountService()
audit = AuditLogger()
users = UserManagementService(audit)


def current_user(authorization: Optional[str] = Header(default=None)) -> AuthUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Firebase bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        return auth.verify(token)
    except Exception as e:
        raise HTTPException(401, f"Invalid session: {e}")


class AccountCreate(BaseModel):
    legal_name: str = Field(min_length=2)
    phone: str = ""
    country: str = ""
    risk_profile: str = "moderate"


class SubscribeRequest(BaseModel):
    account_id: str
    amount: Decimal
    external_ref: str = ""


class RedemptionRequest(BaseModel):
    account_id: str
    amount: Decimal


class KycUpdate(BaseModel):
    status: str
    note: str = ""


class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)
    role: str = "investor"
    display_name: str = ""


class UserRoleUpdate(BaseModel):
    role: str


class UserDisabledUpdate(BaseModel):
    disabled: bool


class NavPublish(BaseModel):
    nav_per_unit: Decimal
    valuation_date: str
    source_equity: Decimal
    notes: str = ""


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/me")
def me(user: AuthUser = Depends(current_user)):
    return {"uid": user.uid, "email": user.email, "role": user.role}


@app.get("/api/accounts")
def list_accounts(user: AuthUser = Depends(current_user)):
    return accounts.list_accounts(user)


@app.post("/api/accounts")
def create_account(body: AccountCreate, user: AuthUser = Depends(current_user)):
    account_id = accounts.create_account(
        user, body.legal_name, body.phone, body.country, body.risk_profile
    )
    return {"account_id": account_id}


@app.get("/api/accounts/{account_id}")
def get_account(account_id: str, user: AuthUser = Depends(current_user)):
    return accounts.get_account(account_id, user)


@app.get("/api/accounts/{account_id}/holdings")
def holdings(account_id: str, user: AuthUser = Depends(current_user)):
    accounts.get_account(account_id, user)
    return accounts.holdings(account_id)


@app.get("/api/accounts/{account_id}/transactions")
def transactions(account_id: str, user: AuthUser = Depends(current_user)):
    return accounts.transactions(account_id, user)


@app.get("/api/accounts/{account_id}/statement")
def statement(account_id: str, user: AuthUser = Depends(current_user)):
    return accounts.statement(account_id, user)


@app.post("/api/subscriptions")
def subscribe(body: SubscribeRequest, user: AuthUser = Depends(current_user)):
    tx = accounts.subscribe(body.account_id, body.amount, user, body.external_ref)
    return {"transaction_id": tx}


@app.post("/api/redemptions")
def redemption(body: RedemptionRequest, user: AuthUser = Depends(current_user)):
    tx = accounts.request_redemption(body.account_id, body.amount, user)
    return {"transaction_id": tx}


@app.post("/api/admin/accounts/{account_id}/kyc")
def update_kyc(account_id: str, body: KycUpdate, user: AuthUser = Depends(current_user)):
    try:
        accounts.set_kyc_status(account_id, body.status, user, body.note)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


@app.post("/api/admin/nav")
def publish_nav(body: NavPublish, user: AuthUser = Depends(current_user)):
    try:
        accounts.publish_nav(
            body.nav_per_unit, user, body.valuation_date, body.source_equity, body.notes
        )
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


@app.post("/api/admin/redemptions/{transaction_id}/approve")
def approve_redemption(transaction_id: str, user: AuthUser = Depends(current_user)):
    try:
        accounts.approve_redemption(transaction_id, user)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}


@app.get("/api/admin/audit")
def audit_log(user: AuthUser = Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin required")
    return audit.recent(200)


@app.get("/api/admin/kill-switch")
def kill_switch(user: AuthUser = Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin required")
    return KillSwitch().status()


@app.get("/api/admin/approvals/{request_id}")
def approval(request_id: str, user: AuthUser = Depends(current_user)):
    if user.role not in {"admin", "risk_approver", "execution_approver", "trader"}:
        raise HTTPException(403, "Approval role required")
    return ApprovalWorkflow().get(request_id)


@app.get("/api/admin/users")
def list_users(user: AuthUser = Depends(current_user)):
    try:
        return users.list_users(user)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.post("/api/admin/users")
def create_user(body: UserCreate, user: AuthUser = Depends(current_user)):
    try:
        uid = users.create_user(body.email, body.password, body.role, user, body.display_name)
        return {"uid": uid}
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.patch("/api/admin/users/{uid}/role")
def update_role(uid: str, body: UserRoleUpdate, user: AuthUser = Depends(current_user)):
    try:
        users.set_role(uid, body.role, user)
        return {"ok": True}
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.patch("/api/admin/users/{uid}/disabled")
def update_disabled(uid: str, body: UserDisabledUpdate, user: AuthUser = Depends(current_user)):
    try:
        users.set_disabled(uid, body.disabled, user)
        return {"ok": True}
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/fund/nav")
def current_nav(user: AuthUser = Depends(current_user)):
    return accounts.nav_snapshot()


@app.get("/api/fund/nav/history")
def nav_history(user: AuthUser = Depends(current_user)):
    return accounts.nav_history()


@app.get("/api/admin/redemptions")
def pending_redemptions(user: AuthUser = Depends(current_user)):
    try:
        return accounts.pending_redemptions(user)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/admin/subscriptions")
def pending_subscriptions(user: AuthUser = Depends(current_user)):
    try:
        return accounts.pending_subscriptions(user)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.post("/api/admin/subscriptions/{transaction_id}/approve")
def approve_subscription(transaction_id: str, user: AuthUser = Depends(current_user)):
    try:
        accounts.approve_subscription(transaction_id, user)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"ok": True}
