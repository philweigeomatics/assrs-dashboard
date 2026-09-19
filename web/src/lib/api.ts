import { devFakeToken, supabase } from "./supabase";
import type {
  AlertFeed, Analysis, BasketStats, CompareResult, HistoryRef, PairStats, SectorAnalysis,
  AlertNote, EquityBrief, NoteScorecard, PairTradeResult, SimResult, StockRef,
  StrategyResult, WhatIfAi,
  Breadth, Heatmap, Leverage, Rotation, TopList, Wyckoff,
  QtBook, QtExposure, QtOptimise, QtRisk, QtScope, QtStatus,
  ChainGraphPayload, WatchRef, LeadLagResult, DiscoverResult,
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
  pairTrade: (symbols: string[], zWindow: number, olsWindow: number) =>
    call<PairTradeResult>("/strategies/pair-trade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols, z_window: zWindow, ols_window: olsWindow }),
    }),
  notes: (ticker?: string) =>
    call<{ notes: AlertNote[]; scorecard: NoteScorecard }>(
      `/alerts/notes${ticker ? `?ticker=${ticker}` : ""}`),
  noteCreate: (body: {
    ticker: string; scan_date: string; note: string; predictions: unknown[];
  }) => call<AlertNote>("/alerts/notes", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }),
  notesResolve: () =>
    call<{ resolved: number; pending: number; notes: AlertNote[]; scorecard: NoteScorecard }>(
      "/alerts/notes/resolve", { method: "POST" }),
  noteDelete: (id: number) =>
    call<{ deleted: number }>(`/alerts/notes/${id}`, { method: "DELETE" }),
  equity: (t: string) => call<EquityBrief>(`/equity/${t}`),
  equityGenerate: (t: string, section: string, force = false) =>
    call<{ section: string }>(`/equity/${t}/generate/${section}?force=${force}`, { method: "POST" }),
  equitySavePeers: (t: string, competitors: { ticker: string; name: string; why: string }[]) =>
    call<{ saved: number }>(`/equity/${t}/peers`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ competitors }),
    }),
  equitySaveSector: (t: string, sector: string, tickers: string[]) =>
    call<{ sector: string; added: number; created: boolean }>(`/equity/${t}/sector`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sector, tickers }),
    }),
  sectorNames: () => call<string[]>("/sectors"),
  watchlist: (market?: "CN" | "NA") =>
    call<WatchRef[]>(`/watchlist${market ? `?market=${market}` : ""}`),
  supplyChain: (t: string) =>
    call<ChainGraphPayload>(`/supply-chain/${encodeURIComponent(t)}`),
  watchlistAdd: (t: string) =>
    call<{ t: string; message: string }>(`/watchlist/${t}`, { method: "POST" }),
  watchlistRemove: (t: string) =>
    call<{ t: string; message: string }>(`/watchlist/${t}`, { method: "DELETE" }),
  leadLag: (body: { ticker: string; peers: string[];
                    lookback_days: number; max_lag: number }) =>
    call<LeadLagResult>("/strategies/lead-lag", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  discover: (body: { kind: "pair-trade" | "lead-lag"; lookback_days: number;
                     min_corr: number; within_sector: boolean }) =>
    call<DiscoverResult>("/strategies/discover", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  strategyScan: (name: string) =>
    call<StrategyResult>(`/strategies/${name}/scan`, { method: "POST" }),
  compareStats: (t: string, other: string, window: string) =>
    call<PairStats>(`/compare-stats/${t}?with=${other}&window=${window}`),
  sectors: (t: string, window: number) => call<SectorAnalysis>(`/sectors/${t}?window=${window}`),
  qtStatus: () => call<QtStatus>("/questrade/status"),
  // The token is POSTed and never returned, logged or stored client-side.
  qtConnect: (refresh_token: string) =>
    call<QtStatus>("/questrade/connect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token }),
    }),
  qtDisconnect: () => call<{ connected: false }>("/questrade/connect", { method: "DELETE" }),
  qtBook: (base: "CAD" | "USD") => call<QtBook>(`/questrade/portfolio?base=${base}`),
  qtRisk: (benchmark: string, base: "CAD" | "USD", scope: QtScope) =>
    call<QtRisk>(
      `/questrade/risk?benchmark=${encodeURIComponent(benchmark)}&base=${base}&scope=${scope}`),
  qtExposure: (base: "CAD" | "USD") => call<QtExposure>(`/questrade/exposure?base=${base}`),
  qtOptimise: (base: "CAD" | "USD", scope: QtScope, method: string, cap: number) =>
    call<QtOptimise>(
      `/questrade/optimise?base=${base}&scope=${scope}&method=${method}&cap=${cap}`),
  heatmap: () => call<Heatmap>("/market/heatmap"),
  breadth: (days = 60) => call<Breadth>(`/market/breadth?days=${days}`),
  leverage: () => call<Leverage[]>("/market/leverage"),
  topList: () => call<TopList>("/market/toplist"),
  wyckoff: (index: string) => call<Wyckoff>(`/market/wyckoff?index=${index}`),
  rotation: (freq: "w" | "d") => call<Rotation>(`/market/rotation?freq=${freq}`),
  // `force` regenerates a real-bar read instead of serving the cached one.
  whatifAi: (t: string, body: Record<string, unknown>, force = false) =>
    call<WhatIfAi>(`/whatif-ai/${t}?force=${force}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};
