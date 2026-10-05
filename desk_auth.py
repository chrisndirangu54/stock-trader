"""Streamlit authentication and production-control UI helpers."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from firebase_security import AuthUser, FirebaseAuthManager, SecretVault
from production_controls import ApprovalWorkflow, KillSwitch, AuditLogger


def require_authentication() -> AuthUser:
    """Require Firebase login. Optional dev bypass is viewer-only and cannot unlock production."""
    if os.getenv("DEV_ALLOW_UNAUTHENTICATED", "0") == "1":
        user = AuthUser(uid="dev-viewer", email="dev@local", role="viewer")
        st.session_state.auth_user = user
        return user

    if "auth_user" in st.session_state and "firebase_id_token" in st.session_state:
        try:
            mgr = FirebaseAuthManager()
            user = mgr.verify(st.session_state.firebase_id_token)
            st.session_state.auth_user = user
            return user
        except Exception:
            st.session_state.pop("auth_user", None)
            st.session_state.pop("firebase_id_token", None)

    st.markdown("### Secure sign-in")
    st.caption("Firebase Authentication is required. Production trading controls are fail-closed without verified identity.")
    with st.form("firebase_login", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("SIGN IN", type="primary")
    if submit:
        try:
            mgr = FirebaseAuthManager()
            tokens = mgr.sign_in(email, password)
            user = mgr.verify(tokens["idToken"])
            st.session_state.firebase_id_token = tokens["idToken"]
            st.session_state.auth_user = user
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    st.stop()


def auth_sidebar(user: AuthUser):
    with st.sidebar:
        st.markdown("### IDENTITY")
        st.write(user.email)
        st.caption(f"Role: {user.role}")
        if st.button("SIGN OUT", use_container_width=True):
            FirebaseAuthManager.logout(st.session_state)
            st.rerun()


def secret_vault_panel(user: AuthUser):
    st.markdown("##### Encrypted broker secret vault")
    if user.role != "admin":
        st.info("Only admins can add or rotate broker secrets. Other roles never see decrypted credentials.")
        return
    try:
        vault = SecretVault()
    except Exception as e:
        st.error(f"Secret vault unavailable: {e}")
        return

    scope = st.selectbox("Secret scope", ["alpaca-paper", "oanda-practice", "ibkr-paper", "production"])
    name = st.text_input("Secret name", placeholder="API_KEY")
    value = st.text_input("Secret value", type="password")
    c1, c2 = st.columns(2)
    if c1.button("STORE / ROTATE SECRET", type="primary"):
        if not name or not value:
            st.error("Name and value are required.")
        else:
            vault.put(scope, name, value, user.uid)
            AuditLogger().log(user, "secret_stored_or_rotated", {"scope": scope, "name": name})
            st.success("Secret encrypted and stored. Plaintext was not written to Firestore.")
    if c2.button("DELETE SECRET"):
        if name:
            vault.delete(scope, name)
            AuditLogger().log(user, "secret_deleted", {"scope": scope, "name": name}, severity="warning")
            st.success("Secret deleted.")


def approval_panel(user: AuthUser, proposed_orders: list[dict] | None = None, broker: str = ""):
    workflow = ApprovalWorkflow()
    st.markdown("##### Two-stage production approval")
    st.caption("Sequence: proposer → distinct risk approver → distinct execution approver. Any order change invalidates approval.")

    if proposed_orders is not None and user.can_trade:
        rationale = st.text_input("Trade rationale / ticket note", key="approval_rationale")
        if st.button("CREATE APPROVAL REQUEST"):
            if not proposed_orders:
                st.error("No proposed orders.")
            else:
                rid = workflow.propose(user, broker, proposed_orders, rationale or "Quant desk rebalance")
                st.session_state.approval_request_id = rid
                st.success(f"Approval request created: {rid}")

    rid = st.text_input("Approval request ID", value=st.session_state.get("approval_request_id", ""))
    if not rid:
        return
    try:
        req = workflow.get(rid)
        st.json({
            "request_id": req.get("request_id"),
            "broker": req.get("broker"),
            "state": req.get("state"),
            "proposer": req.get("proposer_email"),
            "risk_approver_uid": req.get("risk_approver_uid"),
            "execution_approver_uid": req.get("execution_approver_uid"),
            "expires_at": str(req.get("expires_at")),
            "order_hash": req.get("order_hash"),
        })
        a, b, c = st.columns(3)
        if a.button("RISK APPROVE", disabled=not user.can_risk_approve):
            workflow.risk_approve(rid, user)
            st.success("Risk approval recorded.")
            st.rerun()
        if b.button("EXECUTION APPROVE", disabled=not user.can_execution_approve):
            workflow.execution_approve(rid, user)
            st.success("Execution approval recorded.")
            st.rerun()
        reject_reason = st.text_input("Reject reason", key="reject_reason")
        if c.button("REJECT"):
            workflow.reject(rid, user, reject_reason or "Rejected from desk")
            st.warning("Request rejected.")
            st.rerun()
    except Exception as e:
        st.error(f"Approval request error: {e}")


def kill_switch_panel(user: AuthUser):
    ks = KillSwitch()
    status = ks.status()
    st.metric("Global kill switch", "ENGAGED" if status["engaged"] else "DISENGAGED")
    st.caption(status.get("reason") or "")
    if user.role == "admin":
        reason = st.text_input("Kill-switch reason", value="Manual operator action")
        c1, c2 = st.columns(2)
        if c1.button("ENGAGE KILL SWITCH", type="primary"):
            ks.set(True, reason, user)
            st.error("Global kill switch engaged.")
            st.rerun()
        if c2.button("DISENGAGE KILL SWITCH"):
            ks.set(False, reason, user)
            st.warning("Kill switch disengaged. Other production gates still apply.")
            st.rerun()
