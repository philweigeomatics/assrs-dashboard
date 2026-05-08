"""Admin_Invite_Users.py — Invite users and manage accounts (admin only)."""
import streamlit as st
import auth_manager
import email_sender

auth_manager.require_admin()

st.set_page_config(page_title="Invite Users | 用户管理", page_icon="👥", layout="wide")
st.title("👥 User Management | 用户管理")
st.caption("Admin only — invite new users and manage existing accounts.")
st.markdown("---")

# ── Section 1: Invite a new user ──────────────────────────────────────────────
st.subheader("✉️ Invite New User")

with st.form("invite_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_username = st.text_input("Username", placeholder="e.g. john_doe")
    with col2:
        new_email = st.text_input("Email address", placeholder="e.g. john@example.com")
    with col3:
        new_role = st.selectbox("Role", ["user", "admin"])

    app_url = st.text_input(
        "App URL (optional — included in the email)",
        placeholder="https://your-app.streamlit.app",
        help="If provided, the invitation email will include a direct login link.",
    )

    send = st.form_submit_button("Send invitation", type="primary", use_container_width=False)

if send:
    if not new_username.strip() or not new_email.strip():
        st.warning("⚠️ Username and email are required.")
    else:
        with st.spinner("Creating user and sending email…"):
            try:
                temp_pw = auth_manager.invite_user(
                    username=new_username.strip(),
                    email=new_email.strip(),
                    role=new_role,
                )
                email_sender.send_invite_email(
                    to_email=new_email.strip(),
                    username=new_username.strip(),
                    temp_password=temp_pw,
                    app_url=app_url.strip(),
                )
                st.success(
                    f"✅ **{new_username}** invited successfully. "
                    f"An email with their temporary password has been sent to **{new_email}**."
                )
            except ValueError as exc:
                # Duplicate username / email
                st.error(f"❌ {exc}")
            except RuntimeError as exc:
                # User was created but email failed — show temp password so admin can share it manually
                st.warning(
                    f"⚠️ User **{new_username}** was created but the email failed to send.\n\n"
                    f"**Share this temporary password manually:** `{exc}`\n\n"
                    f"Error: {exc}"
                )
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")

st.markdown("---")

# ── Section 2: User list ──────────────────────────────────────────────────────
st.subheader("🗂️ All Users")

users = auth_manager.get_all_users()

if not users:
    st.info("No users found.")
else:
    # Build display table + per-row action controls
    for user in users:
        uname  = user["username"]
        email  = user.get("email") or "—"
        role   = user.get("role", "user")
        active = bool(user.get("is_active", True))
        must_change = bool(user.get("must_change_password", False))

        is_self = uname == auth_manager.get_current_user().get("username")

        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2, 2.5, 1.2, 1.2, 1.2])

            with c1:
                label = f"**{uname}**"
                if is_self:
                    label += " *(you)*"
                st.markdown(label)
                if must_change:
                    st.caption("⏳ Awaiting first login")

            with c2:
                st.caption(email)

            with c3:
                # Role toggle (not self, to prevent self-demotion)
                if not is_self:
                    new_role = st.selectbox(
                        "Role", ["user", "admin"],
                        index=0 if role == "user" else 1,
                        key=f"role_{uname}",
                        label_visibility="collapsed",
                    )
                    if new_role != role:
                        auth_manager.set_user_role(uname, new_role)
                        st.rerun()
                else:
                    st.caption(f"Role: **{role}**")

            with c4:
                # Active badge
                if active:
                    st.markdown(
                        "<span style='color:#22c55e;font-weight:600'>● Active</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#ef4444;font-weight:600'>● Inactive</span>",
                        unsafe_allow_html=True,
                    )

            with c5:
                if not is_self:
                    if active:
                        if st.button("Deactivate", key=f"deact_{uname}", use_container_width=True):
                            auth_manager.set_user_active(uname, False)
                            st.rerun()
                    else:
                        if st.button("Reactivate", key=f"react_{uname}",
                                     use_container_width=True, type="primary"):
                            auth_manager.set_user_active(uname, True)
                            st.rerun()
