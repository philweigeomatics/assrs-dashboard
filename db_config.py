import os

# Simple binary detection: local or production
IS_LOCAL = os.environ.get('USER') == "phil-f" or os.environ.get('USERNAME') == "phil-f"

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
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN0dGlxdHVxeXdla2VtbGFleG9wIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDA4ODQ0NywiZXhwIjoyMDg1NjY0NDQ3fQ.WDo6JqlpCl45N02_dc4nZow21Q2Qr513r55z4eDcvnI"
    print(f"   - Supabase URL: {SUPABASE_URL}")
