import type { ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import { LoginPage } from "./auth/LoginPage";
import { SetPasswordPage } from "./auth/SetPasswordPage";
import { Dashboard } from "./routes/Dashboard";
import { TechnicalAnalysis } from "./routes/TechnicalAnalysis";
import { Alerts } from "./routes/Alerts";
import { Basket } from "./routes/Basket";
import { Strategies } from "./routes/Strategies";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

function RequireAuth({ children }: { children: ReactNode }) {
  const { ready, session, dev } = useAuth();
  if (!ready) return <div className="p-10 text-center label">加载中…</div>;
  if (!session && !dev) return <Navigate to="/login" replace />;
  return <>{children}</>;
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
