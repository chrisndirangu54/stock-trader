"""Firebase authentication, RBAC and encrypted secret storage for the quant desk.

Fail-closed: production controls require Firebase Admin + Firestore + a Fernet master key.
Paper research can remain separate, but no production eligibility is granted without this layer.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

PLATFORM_OWNER_EMAIL = os.getenv("PLATFORM_OWNER_EMAIL", "chrisndirangu54@gmail.com").strip().lower()
from cryptography.fernet import Fernet, InvalidToken

_FIREBASE_APP = None


def _init_firebase():
    global _FIREBASE_APP
    if _FIREBASE_APP is not None:
        return _FIREBASE_APP
    import firebase_admin
    from firebase_admin import credentials
    try:
        _FIREBASE_APP = firebase_admin.get_app()
        return _FIREBASE_APP
    except ValueError:
        pass

    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if raw:
        info = json.loads(raw)
        cred = credentials.Certificate(info)
        _FIREBASE_APP = firebase_admin.initialize_app(cred)
    else:
        # Uses GOOGLE_APPLICATION_CREDENTIALS / ADC when available.
        _FIREBASE_APP = firebase_admin.initialize_app()
    return _FIREBASE_APP


def firestore_client():
    _init_firebase()
    from firebase_admin import firestore
    return firestore.client()


@dataclass(frozen=True)
class AuthUser:
    uid: str
    email: str
    role: str
    disabled: bool = False

    @property
    def can_trade(self) -> bool:
        return self.role in {"trader", "risk_approver", "execution_approver", "admin"}

    @property
    def can_risk_approve(self) -> bool:
        return self.role in {"risk_approver", "admin"}

    @property
    def can_execution_approve(self) -> bool:
        return self.role in {"execution_approver", "admin"}


class FirebaseAuthManager:
    """Email/password sign-in via Firebase Auth REST; ID tokens verified server-side."""

    def __init__(self):
        self.web_api_key = os.getenv("FIREBASE_WEB_API_KEY")
        _init_firebase()

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        if not self.web_api_key:
            raise RuntimeError("Set FIREBASE_WEB_API_KEY for server-side email/password sign-in")
        url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        r = requests.post(
            url,
            params={"key": self.web_api_key},
            json={"email": email, "password": password, "returnSecureToken": True},
            timeout=15,
        )
        if r.status_code >= 400:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = "Firebase sign-in failed"
            raise RuntimeError(msg)
        return r.json()

    def verify(self, id_token: str) -> AuthUser:
        from firebase_admin import auth
        decoded = auth.verify_id_token(id_token, check_revoked=True)
        uid = decoded["uid"]
        email = decoded.get("email", "")
        user_record = auth.get_user(uid)
        db = firestore_client()
        snap = db.collection("users").document(uid).get()
        profile = snap.to_dict() if snap.exists else {}
        is_owner = email.strip().lower() == PLATFORM_OWNER_EMAIL
        role = "admin" if is_owner else profile.get("role", "viewer")
        disabled = False if is_owner else bool(user_record.disabled or profile.get("disabled", False))
        if disabled:
            raise PermissionError("User account is disabled")
        if is_owner:
            db.collection("users").document(uid).set({
                "email": email,
                "role": "admin",
                "platform_owner": True,
                "disabled": False,
                "updated_at": datetime.now(timezone.utc),
            }, merge=True)
        return AuthUser(uid=uid, email=email, role=role, disabled=False)

    @staticmethod
    def logout(session_state: Any) -> None:
        for key in ("firebase_id_token", "auth_user"):
            if key in session_state:
                del session_state[key]


class SecretVault:
    """Fernet-encrypted secrets stored in Firestore. Master key must live outside Firestore."""

    COLLECTION = "encrypted_secrets"

    def __init__(self, master_key: Optional[str] = None):
        key = master_key or os.getenv("QUANT_MASTER_KEY")
        if not key:
            raise RuntimeError("Set QUANT_MASTER_KEY to a Fernet key")
        try:
            self.fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as e:
            raise RuntimeError("QUANT_MASTER_KEY must be a valid Fernet key") from e
        self.db = firestore_client()

    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

    def put(self, scope: str, name: str, plaintext: str, actor_uid: str) -> None:
        token = self.fernet.encrypt(plaintext.encode()).decode()
        self.db.collection(self.COLLECTION).document(f"{scope}__{name}").set({
            "scope": scope,
            "name": name,
            "ciphertext": token,
            "updated_by": actor_uid,
            "updated_at": datetime.now(timezone.utc),
        })

    def get(self, scope: str, name: str) -> str:
        snap = self.db.collection(self.COLLECTION).document(f"{scope}__{name}").get()
        if not snap.exists:
            raise KeyError(f"Secret {scope}/{name} not found")
        token = snap.to_dict()["ciphertext"]
        try:
            return self.fernet.decrypt(token.encode()).decode()
        except InvalidToken as e:
            raise RuntimeError("Secret decryption failed; master key mismatch or ciphertext corrupted") from e

    def delete(self, scope: str, name: str) -> None:
        self.db.collection(self.COLLECTION).document(f"{scope}__{name}").delete()


def bootstrap_user(uid: str, email: str, role: str = "investor") -> None:
    """Admin-only helper intended for initial setup scripts, not the public UI."""
    if role not in {"viewer", "investor", "trader", "risk_approver", "execution_approver", "admin"}:
        raise ValueError("Invalid role")
    db = firestore_client()
    db.collection("users").document(uid).set({
        "email": email,
        "role": role,
        "disabled": False,
        "updated_at": datetime.now(timezone.utc),
    }, merge=True)


def bootstrap_platform_owner(email: str = PLATFORM_OWNER_EMAIL) -> str:
    """Promote the existing Firebase Auth user for the configured owner email.

    The Auth user must already exist; this function intentionally does not invent
    or reset an owner password.
    """
    _init_firebase()
    from firebase_admin import auth
    rec = auth.get_user_by_email(email)
    db = firestore_client()
    db.collection("users").document(rec.uid).set({
        "email": rec.email,
        "role": "admin",
        "platform_owner": True,
        "disabled": False,
        "updated_at": datetime.now(timezone.utc),
    }, merge=True)
    if rec.disabled:
        auth.update_user(rec.uid, disabled=False)
    return rec.uid
