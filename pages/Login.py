"""0_Login.py — Invitation-only login page"""
import streamlit as st
import auth_manager

st.set_page_config(page_title="Login | 登录", page_icon="🔐", layout="centered")

# Already logged in
if auth_manager.is_logged_in():
    user = auth_manager.get_current_user()
    st.success(f"✅ Already logged in as **{user['username']}**")
    st.stop()

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
        if user:
            st.session_state["current_user"] = user
            st.success(f"✅ Welcome, **{user['username']}**!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Contact admin for access.")

st.markdown("---")
st.caption("No account? Contact your administrator for an invitation.")
