"""auth_manager.py — Invitation-only login system"""
import secrets
import string
import bcrypt
import streamlit as st
from db_manager import db


# ── Password utilities ─────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def generate_temp_password(length: int = 12) -> str:
    """Generates a random alphanumeric temp password (no ambiguous chars)."""
    alphabet = string.ascii_letters + string.digits
    # Remove visually ambiguous characters: 0, O, I, l
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
    # Uniqueness checks
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
    """Update password hash and clear the must_change_password flag."""
    db.update_records(
        "app_users",
        update_values={
            "password_hash":        hash_password(new_password),
            "must_change_password": False,
        },
        filters={"username": username},
    )


def set_user_active(username: str, is_active: bool) -> None:
    """Activate or deactivate a user account."""
    db.update_records(
        "app_users",
        update_values={"is_active": is_active},
        filters={"username": username},
    )


def set_user_role(username: str, role: str) -> None:
    """Change a user's role (user / admin)."""
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


# ── Session helpers ────────────────────────────────────────────────────────────

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


def logout():
    st.session_state.pop("current_user", None)


def require_login():
    """Call at top of any protected page. Stops rendering if not logged in."""
    if not is_logged_in():
        st.warning("🔒 Please log in to access this page.")
        st.page_link("Login.py", label="Go to Login", icon="🔐")
        st.stop()


def require_admin():
    """Call at top of admin-only pages."""
    require_login()
    if not is_admin():
        st.error("⛔ Admin access required.")
        st.stop()


# ── DB migration helper ────────────────────────────────────────────────────────

def ensure_must_change_password_column() -> None:
    """
    Idempotent migration: adds must_change_password column to app_users if missing.
    Safe to call on every startup — no-ops if the column already exists.
    SQLite only; Supabase column must be added via the dashboard.
    """
    from db_config import USE_SQLITE, DBNAME
    if not USE_SQLITE:
        return  # Supabase: add column manually in dashboard

    import sqlite3
    with sqlite3.connect(DBNAME) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(app_users)").fetchall()]
        if "must_change_password" not in cols:
            conn.execute(
                "ALTER TABLE app_users ADD COLUMN must_change_password BOOLEAN DEFAULT 0"
            )
            conn.commit()
            print("✅ Added must_change_password column to app_users")
