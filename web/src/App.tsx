import type { ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import { LoginPage } from "./auth/LoginPage";
import { SetPasswordPage } from "./auth/SetPasswordPage";
import { Dashboard } from "./routes/Dashboard";
import { MyQuestrade } from "./routes/MyQuestrade";
import { Watchlist } from "./routes/Watchlist";
import { TechnicalAnalysis } from "./routes/TechnicalAnalysis";
import { Alerts } from "./routes/Alerts";
import { Basket } from "./routes/Basket";
import { Strategies } from "./routes/Strategies";
import { Admin } from "./routes/Admin";
import { ErrorBoundary } from "./components/ErrorBoundary";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

function RequireAuth({ children }: { children: ReactNode }) {
  const { ready, session, dev } = useAuth();
  const { pathname } = useLocation();
  if (!ready) return <div className="p-10 text-center label">加载中…</div>;
  if (!session && !dev) return <Navigate to="/login" replace />;
  // Inside the router, so a page that throws costs you that page and not the
  // navigation you need to leave it. Keyed on the path so walking away from a
  // broken route clears the error rather than carrying it to the next one.
  return (
    <ErrorBoundary key={pathname} label={pathname}>
      {children}
    </ErrorBoundary>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/set-password" element={<SetPasswordPage />} />
            <Route
              path="/"
              element={
                <RequireAuth>
                  <TechnicalAnalysis />
                </RequireAuth>
              }
            />
            <Route
              path="/market"
              element={
                <RequireAuth>
                  <Dashboard />
                </RequireAuth>
              }
            />
            <Route
              path="/questrade"
              element={
                <RequireAuth>
                  <MyQuestrade />
                </RequireAuth>
              }
            />
            <Route
              path="/watchlist"
              element={
                <RequireAuth>
                  <Watchlist />
                </RequireAuth>
              }
            />
            <Route
              path="/alerts"
              element={
                <RequireAuth>
                  <Alerts />
                </RequireAuth>
              }
            />
            <Route
              path="/basket"
              element={
                <RequireAuth>
                  <Basket />
                </RequireAuth>
              }
            />
            <Route
              path="/admin"
              element={
                <RequireAuth>
                  <Admin />
                </RequireAuth>
              }
            />
            <Route
              path="/strategies"
              element={
                <RequireAuth>
                  <Strategies />
                </RequireAuth>
              }
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
