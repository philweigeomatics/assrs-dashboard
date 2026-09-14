import { createClient } from "@supabase/supabase-js";

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const key = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

/**
 * Browser Supabase client — auth only.
 *
 * It holds the PUBLISHABLE key, never the service_role key. Nothing in the
 * browser reads app tables directly during the migration: all data goes
 * through the API, which checks this client's access token. That keeps row-
 * level security off the existing tables, so the Streamlit app is untouched.
 */
export const supabase = createClient(
  // A placeholder keeps the module importable in dev-bypass mode and in unit
  // tests, where no Supabase project is configured.
  (url || "https://placeholder.supabase.co").replace(/\/(rest|auth)\/v1\/?$/, ""),
  key || "public-anon-placeholder",
  { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } },
);

export const supabaseConfigured = Boolean(url && key);

/** Dev-only login bypass. `import.meta.env.DEV` is false in `vite build`. */
export const devFakeToken: string | null =
  import.meta.env.DEV && import.meta.env.VITE_DEV_FAKE_TOKEN
    ? String(import.meta.env.VITE_DEV_FAKE_TOKEN)
    : null;
