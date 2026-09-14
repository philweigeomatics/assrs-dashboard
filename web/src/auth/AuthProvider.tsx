import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { devFakeToken, supabase } from "../lib/supabase";

type AuthState = { ready: boolean; session: Session | null; dev: boolean };

const Ctx = createContext<AuthState>({ ready: false, session: null, dev: false });

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ ready: Boolean(devFakeToken), session: null, dev: Boolean(devFakeToken) });

  useEffect(() => {
    if (devFakeToken) return;
    supabase.auth.getSession().then(({ data }) =>
      setState({ ready: true, session: data.session, dev: false }),
    );
    const { data } = supabase.auth.onAuthStateChange((_event, session) =>
      setState({ ready: true, session, dev: false }),
    );
    return () => data.subscription.unsubscribe();
  }, []);

  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
