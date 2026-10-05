"""Administrative user management for the web investor platform."""
from __future__ import annotations

from datetime import datetime, timezone

from firebase_security import AuthUser, firestore_client, _init_firebase, PLATFORM_OWNER_EMAIL
from production_controls import AuditLogger


VALID_ROLES = {"viewer", "investor", "trader", "risk_approver", "execution_approver", "admin"}


class UserManagementService:
    def __init__(self, audit: AuditLogger | None = None):
        self.db = firestore_client()
        self.audit = audit or AuditLogger()
        _init_firebase()

    @staticmethod
    def _require_admin(actor: AuthUser):
        if actor.role != "admin":
            raise PermissionError("Admin required")

    def list_users(self, actor: AuthUser, limit: int = 500) -> list[dict]:
        self._require_admin(actor)
        from firebase_admin import auth
        rows = []
        for u in auth.list_users(max_results=min(limit, 1000)).iterate_all():
            snap = self.db.collection("users").document(u.uid).get()
            profile = snap.to_dict() if snap.exists else {}
            is_owner = (u.email or "").strip().lower() == PLATFORM_OWNER_EMAIL
            rows.append({
                "uid": u.uid,
                "email": u.email or "",
                "display_name": u.display_name or "",
                "disabled": bool(u.disabled),
                "email_verified": bool(u.email_verified),
                "role": "admin" if is_owner else profile.get("role", "investor"),
                "platform_owner": is_owner,
                "created_at": u.user_metadata.creation_timestamp,
                "last_sign_in_at": u.user_metadata.last_sign_in_timestamp,
            })
        return rows

    def set_role(self, uid: str, role: str, actor: AuthUser) -> None:
        self._require_admin(actor)
        if role not in VALID_ROLES:
            raise ValueError("Invalid role")
        from firebase_admin import auth
        target = auth.get_user(uid)
        if (target.email or "").strip().lower() == PLATFORM_OWNER_EMAIL and role != "admin":
            raise PermissionError("Platform owner cannot be demoted")
        self.db.collection("users").document(uid).set({
            "role": role,
            "updated_at": datetime.now(timezone.utc),
            "updated_by": actor.uid,
        }, merge=True)
        self.audit.log(actor, "user_role_changed", {"uid": uid, "role": role}, severity="warning")

    def set_disabled(self, uid: str, disabled: bool, actor: AuthUser) -> None:
        self._require_admin(actor)
        from firebase_admin import auth
        target = auth.get_user(uid)
        if (target.email or "").strip().lower() == PLATFORM_OWNER_EMAIL and disabled:
            raise PermissionError("Platform owner cannot be disabled")
        auth.update_user(uid, disabled=bool(disabled))
        self.db.collection("users").document(uid).set({
            "disabled": bool(disabled),
            "updated_at": datetime.now(timezone.utc),
            "updated_by": actor.uid,
        }, merge=True)
        self.audit.log(actor, "user_disabled_changed", {"uid": uid, "disabled": bool(disabled)}, severity="warning")

    def create_user(self, email: str, password: str, role: str, actor: AuthUser,
                    display_name: str = "") -> str:
        self._require_admin(actor)
        if role not in VALID_ROLES:
            raise ValueError("Invalid role")
        from firebase_admin import auth
        rec = auth.create_user(email=email, password=password, display_name=display_name or None)
        self.db.collection("users").document(rec.uid).set({
            "email": email,
            "role": role,
            "disabled": False,
            "created_at": datetime.now(timezone.utc),
            "created_by": actor.uid,
        })
        self.audit.log(actor, "user_created", {"uid": rec.uid, "email": email, "role": role}, severity="warning")
        return rec.uid
