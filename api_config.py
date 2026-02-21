import os

def _get_secret(key: str, fallback_env: str = None) -> str:
    """
    Same fallback chain as db_config:
    1. Streamlit secrets  → local .streamlit/secrets.toml + Streamlit Cloud
    2. Environment variable → GitHub Actions + any shell
    3. Raises clear error if neither found
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    env_key = fallback_env or key
    value = os.environ.get(env_key)
    if value:
        return value

    raise ValueError(
        f"❌ Secret '{key}' not found in Streamlit secrets or environment variable '{env_key}'.\n"
        f"   Local: add to .streamlit/secrets.toml\n"
        f"   GitHub Actions: add to repo Settings → Actions secrets\n"
        f"   Streamlit Cloud: add to app Settings → Secrets"
    )


# ── API tokens ────────────────────────────────────────────────────────
TUSHARE_TOKEN = _get_secret("TUSHARE_TOKEN")
