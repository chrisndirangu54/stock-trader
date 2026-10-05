"""Bootstrap an initial Firebase Quant Desk user role.

Usage:
  export FIREBASE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
  python bootstrap_firebase.py <firebase_uid> <email> admin

The Firebase Auth account itself should already exist (for example via Firebase Console).
"""
from __future__ import annotations

import sys

from firebase_security import bootstrap_user

VALID = {"viewer", "investor", "trader", "risk_approver", "execution_approver", "admin"}

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: python bootstrap_firebase.py UID EMAIL ROLE")
    uid, email, role = sys.argv[1:4]
    if role not in VALID:
        raise SystemExit(f"role must be one of {sorted(VALID)}")
    bootstrap_user(uid, email, role)
    print(f"Configured {email} ({uid}) as {role}")

if __name__ == "__main__":
    main()
