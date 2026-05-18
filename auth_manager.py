"""auth_manager.py — Invitation-only login system with persistent cookie sessions."""
import secrets
import string
from datetime import datetime, timedelta, timezone

import bcrypt
import streamlit as st
from db_manager import db


# ── Cookie / session constants ─────────────────────────────────────────────────

COOKIE_NAME  = "assrs_session"   # browser cookie key
SESSION_DAYS = 30                 # how long a login persists across tabs/restarts


# ── DB migration helpers ───────────────────────────────────────────────────────

def ensure_must_change_password_column() -> None:
    """
    Idempotent migration: adds must_change_password column to app_users if missing.
    Safe to call on every startup — no-ops if the column already exists.
    SQLite only; Supabase column must be added via the dashboard.
    """
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return

    from db_config import DBNAME
    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(app_users)").fetchall()]
        if "must_change_password" not in cols:
            conn.execute(
                "ALTER TABLE app_users ADD COLUMN must_change_password BOOLEAN DEFAULT 0"
            )
            conn.commit()
            print("✅ Added must_change_password column to app_users")


def ensure_sessions_table() -> None:
    """
    Idempotent migration: creates app_sessions in SQLite if missing.
    SQLite only — for Supabase run the SQL shown in the UI/README instead.
    """
    from db_config import USE_SQLITE
    if not USE_SQLITE:
        return

    from db_config import DBNAME
    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_sessions (
                token      TEXT    PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                expires_at TEXT    NOT NULL,
                created_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sess_uid ON app_sessions(user_id)"
        )
        conn.commit()
    print("✅ app_sessions table ready (SQLite)")


# ── Password utilities ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def generate_temp_password(length: int = 12) -> str:
    """Generates a random alphanumeric temp password (no ambiguous chars)."""
    alphabet = string.ascii_letters + string.digits
    alphabet = alphabet.translate(str.maketrans("", "", "0OIl"))
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ── User management ────────────────────────────────────────────────────────────

def create_user(username: str, password: str, email: str = None, role: str = "user"):
    """Create a user directly (manual/CLI use). Password is not temporary."""
    hashed = hash_password(password)
    db.insert_records("app_users", [{
        "username":            username,
        "email":               email,
        "password_hash":       hashed,
        "role":                role,
        "is_active":           True,
        "must_change_password": False,
    }])
    print(f"✅ User '{username}' created with role '{role}'")


def invite_user(username: str, email: str, role: str = "user") -> str:
    """
    Create an invited user with a temporary password.
    Returns the plain-text temp password so the caller can email it.
    Raises ValueError if username or email already exists.
    """
    existing_username = db.read_table("app_users", filters={"username": username})
    if not existing_username.empty:
        raise ValueError(f"Username '{username}' is already taken.")

    existing_email = db.read_table("app_users", filters={"email": email})
    if not existing_email.empty:
        raise ValueError(f"Email '{email}' is already registered.")

    temp_pw = generate_temp_password()
    hashed  = hash_password(temp_pw)

    db.insert_records("app_users", [{
        "username":             username,
        "email":                email,
        "password_hash":        hashed,
        "role":                 role,
        "is_active":            True,
        "must_change_password": True,
    }])

    return temp_pw


def change_password(username: str, new_password: str) -> None:
    """
    Update password hash, clear must_change_password, and revoke all existing
    sessions so old cookies can no longer be used.
    """
    user_df = db.read_table("app_users", filters={"username": username})
    if not user_df.empty:
        revoke_all_sessions(int(user_df.iloc[0]["id"]))

    db.update_records(
        "app_users",
        update_values={
            "password_hash":        hash_password(new_password),
            "must_change_password": False,
        },
        filters={"username": username},
    )


def set_user_active(username: str, is_active: bool) -> None:
    db.update_records(
        "app_users",
        update_values={"is_active": is_active},
        filters={"username": username},
    )


def set_user_role(username: str, role: str) -> None:
    db.update_records(
        "app_users",
        update_values={"role": role},
        filters={"username": username},
    )


def get_all_users():
    """Return all user rows (excluding password_hash) for admin display."""
    df = db.read_table(
        "app_users",
        columns="id, username, email, role, is_active, must_change_password",
        order_by="username",
    )
    if df.empty:
        return []
    return df.to_dict("records")


# ── Login ──────────────────────────────────────────────────────────────────────

def login(username_or_email: str, password: str):
    """
    Returns user dict on success, None on failure.
    The returned dict includes must_change_password so Login.py can branch.
    """
    df = db.read_table("app_users", filters={"username": username_or_email})
    if df.empty:
        df = db.read_table("app_users", filters={"email": username_or_email})
    if df.empty:
        return None

    user = df.iloc[0].to_dict()
    if not user.get("is_active", True):
        return None
    if verify_password(password, user["password_hash"]):
        return user
    return None


# ── Persistent session management ─────────────────────────────────────────────

def create_session(user_id: int) -> str:
    """
    Mint a cryptographically random token, persist it in app_sessions, and
    return the token string. Call immediately after login; pass the token to
    write_session_cookie() before st.rerun().
    """
    token   = secrets.token_hex(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    db.insert_records("app_sessions", [{
        "token":      token,
        "user_id":    int(user_id),
        "expires_at": expires.isoformat(),
    }])
    return token


def validate_session(token: str) -> dict | None:
    """
    Validate a session token: checks existence, expiry, and user status.
    Deletes expired tokens on the fly.
    Returns the user dict on success, None otherwise.
    """
    if not token:
        return None
    try:
        df = db.read_table("app_sessions", filters={"token": token})
        if df.empty:
            return None

        row     = df.iloc[0]
        exp_raw = str(row["expires_at"]).replace("Z", "+00:00")
        exp     = datetime.fromisoformat(exp_raw)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) > exp:
            db.delete_records("app_sessions", {"token": token})
            return None

        user_df = db.read_table("app_users", filters={"id": int(row["user_id"])})
        if user_df.empty:
            return None

        user = user_df.iloc[0].to_dict()
        return user if user.get("is_active", True) else None

    except Exception:
        return None


def revoke_session(token: str) -> None:
    """Delete a specific session token from the DB."""
    if not token:
        return
    try:
        db.delete_records("app_sessions", {"token": token})
    except Exception:
        pass


def revoke_all_sessions(user_id: int) -> None:
    """Delete every session for a user (called on password change)."""
    try:
        db.delete_records("app_sessions", {"user_id": int(user_id)})
    except Exception:
        pass


# ── Cookie read / write ────────────────────────────────────────────────────────

def _read_cookie_from_headers() -> str | None:
    """
    Read the session cookie from the HTTP request headers (st.context.headers,
    Streamlit ≥ 1.33). This is synchronous — the cookie is present on the very
    first render of a new tab with no async delay or extra rerun required.
    """
    try:
        cookie_header = st.context.headers.get("Cookie", "")
        if not cookie_header:
            return None
        for part in cookie_header.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() == COOKIE_NAME:
                    return v.strip()
        return None
    except AttributeError:
        return None  # st.context not available (Streamlit < 1.33)


def _cookie_manager():
    """
    Return the CookieManager component. Fixed key so multiple calls in the
    same render share state. CookieManager is a real Streamlit component
    loaded from Streamlit's component endpoint, so it's whitelisted by
    Streamlit Cloud's Content Security Policy (unlike inline <script> /
    onerror handlers, which CSP blocks).
    """
    from extra_streamlit_components import CookieManager
    return CookieManager(key="assrs_cm")


def write_session_cookie(token: str) -> None:
    """
    Write the 30-day session cookie via the CookieManager component.

    IMPORTANT: do NOT call st.rerun() immediately after this. The component
    iframe needs the current render to remain mounted long enough for its JS
    to execute document.cookie. Callers should set st.session_state for the
    user and let the natural render cycle (or a deferred rerun on the next
    user action) flush the cookie.
    """
    try:
        cm      = _cookie_manager()
        expires = datetime.now() + timedelta(days=SESSION_DAYS)
        cm.set(COOKIE_NAME, token, expires_at=expires)
    except Exception:
        pass


def _delete_session_cookie() -> None:
    """Delete the session cookie via the CookieManager component."""
    try:
        cm = _cookie_manager()
        cm.delete(COOKIE_NAME)
    except Exception:
        pass


def restore_session_from_cookie() -> bool:
    """
    Try to restore the current_user from the browser session cookie.
    Call this in streamlit_app.py BEFORE the is_logged_in() routing check.

    Reading strategy (fast path first, reliable fallback):
      1. st.context.headers["Cookie"] — synchronous, available on the very
         first render. Works on localhost. On Streamlit Cloud the reverse
         proxy may not pass the Cookie header through to the WebSocket app,
         in which case this returns None.
      2. CookieManager.get(...) — reads document.cookie via an iframe and
         sends the value back to Python. Async: returns None on the first
         render, then the iframe's response triggers a Streamlit rerun, and
         on that next render the value is cached and available. There is a
         brief flash of the login page on the very first new-tab render, but
         subsequent renders restore the session correctly.
    """
    if is_logged_in():
        return True
    try:
        token = _read_cookie_from_headers()

        if not token:
            cm    = _cookie_manager()
            token = cm.get(COOKIE_NAME)

        if not token:
            return False

        user = validate_session(token)
        if user:
            st.session_state["current_user"]        = user
            st.session_state["assrs_session_token"] = token
            return True

        _delete_session_cookie()
        return False
    except Exception:
        return False


# ── Session-state helpers ──────────────────────────────────────────────────────

def get_current_user():
    return st.session_state.get("current_user", None)


def get_current_user_id():
    user = get_current_user()
    return user["id"] if user else None


def is_logged_in() -> bool:
    return get_current_user() is not None


def is_admin() -> bool:
    user = get_current_user()
    return user is not None and user.get("role") == "admin"


def logout() -> None:
    """
    Full logout: revoke the DB session token, delete the browser cookie,
    and clear session state. The caller should call st.rerun() after this.
    """
    token = st.session_state.pop("assrs_session_token", None)
    if token:
        revoke_session(token)
    st.session_state.pop("current_user", None)
    _delete_session_cookie()


def require_login() -> None:
    """
    Call at the top of any protected page.
    Stops rendering (st.stop()) if the user is not authenticated and no
    valid browser cookie can restore the session.
    """
    if is_logged_in():
        return

    # Fallback: try restoring from cookie in case this page was opened
    # before streamlit_app.py had a chance to run restore_session_from_cookie().
    if restore_session_from_cookie():
        return

    st.warning("🔒 Please log in to access this page.")
    st.page_link("Login.py", label="Go to Login", icon="🔐")
    st.stop()


def require_admin() -> None:
    """Call at top of admin-only pages."""
    require_login()
    if not is_admin():
        st.error("⛔ Admin access required.")
        st.stop()
