"""Login.py — Invitation-only login page with first-login password change."""
import streamlit as st
import auth_manager

# Run idempotent migrations on every start (no-ops after first run)
auth_manager.ensure_must_change_password_column()
auth_manager.ensure_sessions_table()

st.set_page_config(page_title="Login | 登录", page_icon="🔐", layout="centered")

# ── Already fully logged in ────────────────────────────────────────────────────
if auth_manager.is_logged_in():
    user = auth_manager.get_current_user()
    st.success(f"✅ Already logged in as **{user['username']}**")
    st.stop()

# ── First-login password change (session bridge state) ────────────────────────
# After credentials are verified but before the session is established,
# we store the pending user in a separate key so the form can finish setup.
pending = st.session_state.get("_pending_user")

if pending:
    st.title("🔑 Set Your Password")
    st.caption(f"Welcome, **{pending['username']}**! Choose a permanent password to continue.")
    st.markdown("---")

    with st.form("set_password_form"):
        pw1 = st.text_input("New password", type="password", placeholder="At least 8 characters")
        pw2 = st.text_input("Confirm password", type="password", placeholder="Repeat your password")
        submitted = st.form_submit_button("Set password & log in", type="primary",
                                          use_container_width=True)

    if submitted:
        if not pw1 or not pw2:
            st.warning("⚠️ Please fill in both fields.")
        elif pw1 != pw2:
            st.error("❌ Passwords do not match.")
        elif len(pw1) < 8:
            st.error("❌ Password must be at least 8 characters.")
        else:
            try:
                # change_password() revokes all old sessions before updating hash
                auth_manager.change_password(pending["username"], pw1)
                updated = auth_manager.login(pending["username"], pw1)
                token   = auth_manager.create_session(updated["id"])
                st.session_state.pop("_pending_user", None)
                st.session_state["current_user"]        = updated
                st.session_state["assrs_session_token"] = token
                # Defer cookie write to next render (see streamlit_app.py) to
                # avoid the iframe race with st.rerun().
                st.session_state["_pending_cookie_write"] = token
                st.success("✅ Password set! Logging you in…")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Could not update password: {exc}")
    st.stop()

# ── Normal login form ──────────────────────────────────────────────────────────
st.title("🔐 Login | 登录")
st.caption("Invitation-only access | 仅限受邀用户")
st.markdown("---")

with st.form("login_form"):
    username = st.text_input("Username or Email | 用户名或邮箱",
                              placeholder="Enter your username or email")
    password = st.text_input("Password | 密码", type="password",
                              placeholder="Enter your password")
    submitted = st.form_submit_button("Login | 登录", type="primary",
                                      use_container_width=True)

if submitted:
    if not username or not password:
        st.warning("⚠️ Please enter both username and password.")
    else:
        user = auth_manager.login(username.strip(), password)
        if user is None:
            st.error("❌ Invalid credentials. Contact admin for access.")
        elif user.get("must_change_password"):
            # Park the user dict and re-render the change-password form
            st.session_state["_pending_user"] = user
            st.rerun()
        else:
            token = auth_manager.create_session(user["id"])
            st.session_state["current_user"]        = user
            st.session_state["assrs_session_token"] = token
            # Defer cookie write to next render (see streamlit_app.py) to
            # avoid the iframe race with st.rerun().
            st.session_state["_pending_cookie_write"] = token
            st.success(f"✅ Welcome, **{user['username']}**!")
            st.rerun()

st.markdown("---")
st.caption("No account? Contact your administrator for an invitation.")
