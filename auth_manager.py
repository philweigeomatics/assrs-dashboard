"""auth_manager.py — Invitation-only login system"""
import bcrypt
import streamlit as st
from db_manager import db


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_user(username: str, password: str, email: str = None, role: str = "user"):
    """Call this once manually to invite a new user."""
    hashed = hash_password(password)
    db.insert_records("app_users", [{
        "username": username,
        "email":    email,
        "password_hash": hashed,
        "role":     role,
        "is_active": True
    }])
    print(f"✅ User '{username}' created with role '{role}'")


def login(username_or_email: str, password: str):
    """Returns user dict on success, None on failure."""
    # Try username first
    df = db.read_table("app_users", filters={"username": username_or_email})
    if df.empty:
        # Try email
        df = db.read_table("app_users", filters={"email": username_or_email})
    if df.empty:
        return None
    user = df.iloc[0].to_dict()
    if not user.get("is_active", True):
        return None
    if verify_password(password, user["password_hash"]):
        return user
    return None


def get_current_user():
    """Returns logged-in user dict from session, or None."""
    return st.session_state.get("current_user", None)


def get_current_user_id():
    """Returns logged-in user's id, or None."""
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
