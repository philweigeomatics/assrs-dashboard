import { devFakeToken, supabase } from "./supabase";
import type {
  AlertFeed, Analysis, BasketStats, CompareResult, HistoryRef, PairStats, SectorAnalysis,
  SimResult, StockRef, StrategyResult, WhatIfAi,
} from "./types";

const API = ((import.meta.env.VITE_API_URL as string | undefined) || "http://127.0.0.1:8000")
  .replace(/\/$/, "");

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function token(): Promise<string> {
  if (devFakeToken) return devFakeToken;
  // getSession refreshes an expired access token before handing it back.
  const { data } = await supabase.auth.getSession();
  const t = data.session?.access_token;
  if (!t) throw new ApiError(401, "not signed in");
  return t;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { ...(init.headers || {}), Authorization: `Bearer ${await token()}` },
  });
  if (res.status === 401 && !devFakeToken) {
    // Token rejected server-side (revoked, or a different project): drop the
    // local session so the app returns to the login screen instead of
    // failing every request forever.
    await supabase.auth.signOut();
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg);
  }
  return (await res.json()) as T;
}

export const api = {
  me: () => call<{ app_user_id: number; username: string; email: string; role: string }>("/me"),
  stocks: () => call<StockRef[]>("/stocks"),
  search: (q: string, market: "US" | "CA") =>
    call<StockRef[]>(`/search?q=${encodeURIComponent(q)}&market=${market}`),
  history: () => call<HistoryRef[]>("/history"),
  addHistory: (t: string) => call<StockRef>(`/history/${t}`, { method: "POST" }),
  analysis: (t: string) => call<Analysis>(`/analysis/${t}`),
  simulate: (t: string, body: { pct: number; volume: number; open?: number; high?: number; low?: number }) =>
    call<SimResult>(`/simulate/${t}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  compare: (t: string, other: string) => call<CompareResult>(`/compare/${t}?with=${other}`),
  alerts: (date?: string) => call<AlertFeed>(`/alerts${date ? `?date=${date}` : ""}`),
  basket: (symbols: string[], window: string) =>
    call<BasketStats>(`/basket?symbols=${symbols.map(encodeURIComponent).join(",")}&window=${window}`),
  strategy: (name: string) => call<StrategyResult>(`/strategies/${name}`),
  strategyScan: (name: string) =>
    call<StrategyResult>(`/strategies/${name}/scan`, { method: "POST" }),
  compareStats: (t: string, other: string, window: string) =>
    call<PairStats>(`/compare-stats/${t}?with=${other}&window=${window}`),
  sectors: (t: string, window: number) => call<SectorAnalysis>(`/sectors/${t}?window=${window}`),
  whatifAi: (t: string, body: Record<string, unknown>) =>
    call<WhatIfAi>(`/whatif-ai/${t}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
