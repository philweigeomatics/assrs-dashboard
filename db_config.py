import os


def _get_secret(key: str, fallback_env: str = None) -> str:
    """
    Fetch a secret with fallback chain:
    1. Streamlit secrets (st.secrets)  → works in Streamlit Cloud + local .streamlit/secrets.toml
    2. Environment variable             → works in GitHub Actions + any shell
    3. Raises clear error if neither found
    """
    # Try Streamlit secrets first
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # st not available (e.g. running as a script)

    # Fall back to environment variable
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



# Simple binary detection: local or production
IS_LOCAL = os.environ.get('USER') == "phil" or os.environ.get('USERNAME') == "phil"

# Local = SQLite, Not Local = Supabase
USE_SQLITE = IS_LOCAL
USE_SUPABASE = not IS_LOCAL

print(f"🔧 Environment: {'Local' if IS_LOCAL else 'Production'}")
print(f"   - Database: {'SQLite' if USE_SQLITE else 'Supabase'}")

# Database configurations
if USE_SQLITE:
    # Local development - use SQLite
    DBNAME = 'assrs_tushare_local_dev.db'
    print(f"   - DB File: {DBNAME}")
else:
    # Production (GitHub Actions + Streamlit Cloud) - use Supabase
    SUPABASE_URL = _get_secret("SUPABASE_URL")
    SUPABASE_KEY = _get_secret("SUPABASE_KEY")
    print(f"   Supabase URL: {SUPABASE_URL}")