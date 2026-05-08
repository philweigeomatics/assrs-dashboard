"""
email_sender.py — Thin Gmail SMTP wrapper for transactional emails.

Secrets required in .streamlit/secrets.toml (and Streamlit Cloud):
    SMTP_EMAIL        = "youraddress@gmail.com"
    SMTP_APP_PASSWORD = "xxxx xxxx xxxx xxxx"   # Gmail App Password (16 chars, spaces ok)
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from api_config import _get_secret


def _credentials():
    return _get_secret("SMTP_EMAIL"), _get_secret("SMTP_APP_PASSWORD")


def send_invite_email(to_email: str, username: str, temp_password: str, app_url: str = "") -> None:
    """
    Send an invitation email with a temporary password.

    Raises RuntimeError on SMTP failure so the caller can surface it to the UI.
    """
    sender, app_password = _credentials()
    app_password = app_password.replace(" ", "")  # Gmail allows spaces in display; strip them

    subject = "You've been invited to ASSRS Dashboard"

    login_url = app_url.rstrip("/") if app_url else "the app"

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#1e293b;max-width:560px;margin:auto">
      <h2 style="color:#3b82f6">Welcome to ASSRS Dashboard</h2>
      <p>Hi <strong>{username}</strong>,</p>
      <p>You've been granted access to the ASSRS quantitative dashboard.
         Use the credentials below to log in for the first time.</p>

      <table style="background:#f1f5f9;border-radius:8px;padding:16px 24px;margin:24px 0">
        <tr><td style="padding:4px 0"><strong>Username:</strong></td>
            <td style="padding:4px 0 4px 16px">{username}</td></tr>
        <tr><td style="padding:4px 0"><strong>Temporary&nbsp;password:</strong></td>
            <td style="padding:4px 0 4px 16px;font-family:monospace;font-size:15px">{temp_password}</td></tr>
      </table>

      <p>You will be prompted to set your own password immediately after your first login.</p>

      {"<p><a href='" + login_url + "' style='color:#3b82f6'>Click here to log in →</a></p>" if app_url else ""}

      <p style="color:#64748b;font-size:13px;margin-top:32px">
        This is an automated message from ASSRS Dashboard. Do not reply.
      </p>
    </body></html>
    """

    plain = (
        f"Welcome to ASSRS Dashboard, {username}!\n\n"
        f"Username: {username}\n"
        f"Temporary password: {temp_password}\n\n"
        f"You will be asked to set your own password on first login.\n"
        + (f"\nLogin: {login_url}\n" if app_url else "")
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = to_email
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html,  "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, app_password)
            server.sendmail(sender, to_email, msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            "Gmail authentication failed. Check SMTP_EMAIL and SMTP_APP_PASSWORD in secrets."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to send email: {exc}") from exc
