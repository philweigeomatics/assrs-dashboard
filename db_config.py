import os

# Simple binary detection: local or production
IS_LOCAL = os.environ.get('USER') == "phil-" or os.environ.get('USERNAME') == "phil-"

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
    SUPABASE_URL = "https://cttiqtuqywekemlaexop.supabase.co"
    SUPABASE_KEY = os.environ.get(
        "SUPABASE_SECRET_KEY",
        "sb_secret_-1qTazqAH3SGE4aAqU69dA_e0Hyspqg"
    )
    print(f"   - Supabase URL: {SUPABASE_URL}")
