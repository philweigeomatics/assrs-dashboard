"""
api/auth.py — who is calling, and which app_users row that is.

Two identities exist during the migration, and this file is the bridge:

  * Supabase Auth issues the login for the new site. Its access token is an
    ES256-signed JWT; we verify it locally against the project's published
    JWKS, so no request needs a round trip to Supabase.
  * Every existing table (watchlist, search_history, portfolios…) is keyed by
    the Streamlit app's integer app_users.id.

On a user's first call we find their app_users row by email, confirm with
Supabase's admin API that the email on the login is actually VERIFIED, and
record the pair in auth_user_link. After that the link is authoritative.

Why the verification step exists: matching by email alone would let anyone
who can create a Supabase login with your address inherit your data. Sign-ups
are off today, but a setting flipped later must not turn into an account
takeover, so the link is only created for a confirmed, non-anonymous email.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import jwt
import requests
from fastapi import Header, HTTPException

LINK_TABLE = "auth_user_link"
LINK_CACHE_TTL_S = 300


def _supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL is not set")
    # People paste the REST endpoint; the project URL is the bare origin.
    for suffix in ("/rest/v1", "/auth/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url


def _jwks_url() -> str:
    return os.environ.get("SUPABASE_JWKS_URL") or f"{_supabase_url()}/auth/v1/.well-known/jwks.json"


def _issuer() -> str:
    return os.environ.get("SUPABASE_JWT_ISSUER") or f"{_supabase_url()}/auth/v1"


_jwk_client: jwt.PyJWKClient | None = None
_jwk_lock = threading.Lock()


def _jwks() -> jwt.PyJWKClient:
    global _jwk_client
    with _jwk_lock:
        if _jwk_client is None:
            # Keys are cached in-process and re-fetched on an unknown `kid`,
            # which is what makes Supabase key rotation transparent.
            _jwk_client = jwt.PyJWKClient(_jwks_url(), cache_keys=True, lifespan=3600)
        return _jwk_client


def verify_token(token: str) -> dict:
    """Verified claims, or HTTPException(401). Signature, expiry, audience, issuer."""
    try:
        key = _jwks().get_signing_key_from_jwt(token).key
        claims = jwt.decode(
            token, key,
            algorithms=["ES256", "RS256"],
            audience="authenticated",
            issuer=_issuer(),
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "token expired")
    except (jwt.PyJWTError, jwt.PyJWKClientError) as exc:
        raise HTTPException(401, f"invalid token: {exc.__class__.__name__}")
    if claims.get("role") != "authenticated" or claims.get("is_anonymous"):
        raise HTTPException(401, "not an authenticated user")
    return claims


@dataclass(frozen=True)
class AppUser:
    id: int
    username: str
    email: str
    role: str
    auth_user_id: str

    def as_session_user(self) -> dict:
        """The shape auth_manager.get_current_user() returns for Streamlit."""
        return {"id": self.id, "username": self.username,
                "email": self.email, "role": self.role}


_link_cache: dict[str, tuple[float, AppUser]] = {}
_link_lock = threading.Lock()


def _app_user_by_id(app_user_id: int) -> dict | None:
    from db_manager import db
    df = db.read_table("app_users", filters={"id": int(app_user_id)},
                       columns="id,username,email,role,is_active", limit=1)
    return None if df is None or df.empty else df.iloc[0].to_dict()


def _email_confirmed(auth_user_id: str, email: str) -> bool:
    """Ask Supabase (with the service key) whether this login's email is verified."""
    key = os.environ.get("SUPABASE_KEY", "")
    r = requests.get(f"{_supabase_url()}/auth/v1/admin/users/{auth_user_id}",
                     headers={"apikey": key, "Authorization": f"Bearer {key}"},
                     timeout=10)
    if r.status_code != 200:
        return False
    u = r.json()
    return (bool(u.get("email_confirmed_at"))
            and not u.get("is_anonymous")
            and (u.get("email") or "").strip().lower() == email)


def resolve_app_user(claims: dict) -> AppUser:
    """The app_users row behind a verified Supabase login. 403 if there is none."""
    from db_manager import db

    sub = str(claims["sub"])
    now = time.time()
    with _link_lock:
        hit = _link_cache.get(sub)
        if hit and now - hit[0] < LINK_CACHE_TTL_S:
            return hit[1]

    row = None
    link = db.read_table(LINK_TABLE, filters={"auth_user_id": sub},
                         columns="app_user_id", limit=1)
    if link is not None and not link.empty:
        row = _app_user_by_id(int(link.iloc[0]["app_user_id"]))
    else:
        email = (claims.get("email") or "").strip().lower()
        if not email:
            raise HTTPException(403, "login has no email to match an account")
        users = db.read_table("app_users", columns="id,username,email,role,is_active")
        matches = [] if users is None or users.empty else [
            u for u in users.to_dict("records")
            if (u.get("email") or "").strip().lower() == email]
        if len(matches) != 1:
            raise HTTPException(403, "no matching account for this login — ask an admin")
        if not _email_confirmed(sub, email):
            raise HTTPException(403, "email not verified on this login")
        row = matches[0]
        db.insert_records(LINK_TABLE, [{
            "auth_user_id": sub, "app_user_id": int(row["id"]), "email": email,
        }], upsert=False)

    if row is None or not row.get("is_active", True):
        raise HTTPException(403, "account is disabled")

    user = AppUser(id=int(row["id"]), username=str(row.get("username") or ""),
                   email=str(row.get("email") or ""), role=str(row.get("role") or "user"),
                   auth_user_id=sub)
    with _link_lock:
        _link_cache[sub] = (now, user)
    return user


def current_user(authorization: str = Header(default="")) -> AppUser:
    """FastAPI dependency: `user: AppUser = Depends(current_user)`."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    claims = verify_token(authorization.split(" ", 1)[1].strip())
    return resolve_app_user(claims)
