/**
 * The app's one header: page links on the left, whatever the page needs in
 * the middle, session controls on the right.
 *
 * Shared rather than copied into each route, so a new page cannot end up with
 * a subtly different header, and so the links stay in one list.
 */

import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { useAuth } from "../auth/AuthProvider";

const PAGES = [
  { to: "/market", label: "🗺️ 市场看板" },
  { to: "/", label: "📈 个股分析" },
  { to: "/watchlist", label: "⭐ 自选股" },
  { to: "/alerts", label: "🔔 今日提醒" },
  { to: "/basket", label: "🧺 多股对比" },
  { to: "/strategies", label: "⚡ 策略" },
  { to: "/questrade", label: "🏦 MyQuestrade" },
];

export function NavBar({ children }: { children?: ReactNode }) {
  const { dev } = useAuth();
  return (
    <header className="sticky top-0 z-40 bg-canvas/90 backdrop-blur border-b border-line">
      <div className="max-w-[1800px] mx-auto px-3 h-14 flex items-center gap-3">
        <nav className="flex items-center gap-1 shrink-0">
          {PAGES.map((p) => (
            <NavLink key={p.to} to={p.to} end={p.to === "/"}
              className={({ isActive }) =>
                `px-2 h-8 flex items-center rounded-lg text-[14px] font-semibold transition-colors ${
                  isActive ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                }`}>
              {p.label}
            </NavLink>
          ))}
        </nav>

        {children}

        <div className="ml-auto flex items-center gap-3 shrink-0">
          {dev && <span className="text-[12px] text-brand-ink">本地开发模式</span>}
          {!dev && (
            <button onClick={() => supabase.auth.signOut()}
              className="text-[13px] text-ink-mute hover:text-ink">
              退出
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
