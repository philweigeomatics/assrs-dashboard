/**
 * The app's one header: page links on the left, whatever the page needs in
 * the middle, session controls on the right.
 *
 * Shared rather than copied into each route, so a new page cannot end up with
 * a subtly different header, and so the links stay in one list.
 *
 * Below `lg` the links fold into a ☰ menu. Nine links are 819px wide; on a
 * phone they scrolled sideways under a thumb with most of them out of sight,
 * and left no room for what the page puts in the middle (the stock search).
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { NavLink, useLocation } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { useAuth } from "../auth/AuthProvider";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { TodayBell } from "./TodayBell";

export const PAGES = [
  { to: "/market", label: "🗺️ 市场看板" },
  { to: "/", label: "📈 个股分析" },
  { to: "/watchlist", label: "⭐ 自选股" },
  { to: "/alerts", label: "🔔 今日提醒" },
  { to: "/calendar", label: "🗓️ 日历" },
  { to: "/basket", label: "🧺 多股对比" },
  { to: "/strategies", label: "⚡ 策略" },
  { to: "/portfolio", label: "💼 组合" },
  { to: "/questrade", label: "🏦 MyQuestrade" },
];

/** Appended for admins only — a convenience, not the gate. */
const ADMIN_PAGE = { to: "/admin", label: "⚙️ 管理" };

type Page = (typeof PAGES)[number];

/** The page a path belongs to — "/" only matches itself. */
export function pageFor(pathname: string, pages: Page[] = [...PAGES, ADMIN_PAGE]): Page | undefined {
  return pages.find((p) => (p.to === "/" ? pathname === "/" : pathname.startsWith(p.to)));
}

export function NavBar({ children }: { children?: ReactNode }) {
  const { dev } = useAuth();
  // Cosmetic only. The server checks the role on every /admin route — see
  // _require_admin — because hiding a link does not stop a request.
  const me = useQuery({ queryKey: ["me"], queryFn: api.me,
                        staleTime: 10 * 60_000, retry: false });
  const isAdmin = (me.data?.role || "").toLowerCase() === "admin";
  const pages = isAdmin ? [...PAGES, ADMIN_PAGE] : PAGES;
  const { pathname } = useLocation();
  const here = pageFor(pathname, pages);
  const [open, setOpen] = useState(false);
  const shut = useCallback(() => setOpen(false), []);

  return (
    <header className="sticky top-0 z-40 bg-canvas/90 backdrop-blur border-b border-line">
      <div className="max-w-[1800px] mx-auto px-3 h-14 flex items-center gap-2 lg:gap-3">
        <button onClick={() => setOpen(true)} aria-label="打开菜单" aria-expanded={open}
          className="lg:hidden -ml-1 w-10 h-10 shrink-0 grid place-items-center rounded-lg
            text-ink hover:bg-elevated active:bg-elevated">
          <Burger />
        </button>

        <nav className="hidden lg:flex items-center gap-1 min-w-0 overflow-x-auto
          [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {pages.map((p) => (
            <NavLink key={p.to} to={p.to} end={p.to === "/"}
              className={({ isActive }) =>
                `px-2 h-8 flex items-center shrink-0 whitespace-nowrap rounded-lg
                 text-[14px] font-semibold transition-colors ${
                  isActive ? "bg-elevated text-ink" : "text-ink-mute hover:text-ink"
                }`}>
              {p.label}
            </NavLink>
          ))}
        </nav>

        {children ? (
          // A floor, so on a narrow desktop the nav row scrolls rather than
          // squeezing the page's own control (the stock search) to nothing.
          <div className="flex-1 min-w-0 lg:min-w-[220px] flex items-center">{children}</div>
        ) : (
          // With the links folded away, say where you are.
          <span className="lg:hidden flex-1 min-w-0 truncate text-[15px] font-semibold">
            {here?.label}
          </span>
        )}

        <div className="ml-auto flex items-center gap-1 lg:gap-3 shrink-0 -mr-1 lg:mr-0">
          <TodayBell />
          {dev && <span className="hidden lg:inline text-[12px] text-brand-ink">本地开发模式</span>}
          {!dev && (
            <button onClick={() => supabase.auth.signOut()}
              className="hidden lg:inline text-[13px] text-ink-mute hover:text-ink">
              退出
            </button>
          )}
        </div>
      </div>

      <Drawer open={open} onClose={shut} pages={pages} dev={dev} />
    </header>
  );
}

function Drawer({ open, onClose, pages, dev }: {
  open: boolean; onClose: () => void; pages: Page[]; dev: boolean;
}) {
  const close = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    // The page behind should not scroll under a finger dragging the menu.
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    close.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  // Always mounted so it can slide; `invisible` takes it out of the tab order
  // and the accessibility tree while shut. Portalled to <body>: the header's
  // backdrop-blur makes it the containing block for anything `fixed` inside
  // it, which shrank the menu to the header's 56px strip.
  return createPortal(
    <div className={`lg:hidden fixed inset-0 z-50 transition-[visibility] duration-200 ${
      open ? "visible" : "invisible"}`}
      role="dialog" aria-modal="true" aria-label="页面菜单">
      <div onClick={onClose}
        className={`absolute inset-0 bg-black/40 transition-opacity duration-200 ${
          open ? "opacity-100" : "opacity-0"}`} />
      <div className={`absolute inset-y-0 left-0 w-[min(82vw,300px)] bg-panel shadow-xl
        flex flex-col transition-transform duration-200 ease-out
        pt-[env(safe-area-inset-top)] pb-[env(safe-area-inset-bottom)] ${
          open ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="h-14 px-3 flex items-center justify-between border-b border-line shrink-0">
          <span className="text-[15px] font-semibold">ASSRS</span>
          <button ref={close} onClick={onClose} aria-label="关闭菜单"
            className="w-10 h-10 -mr-1 grid place-items-center rounded-lg text-ink-mute
              hover:bg-elevated active:bg-elevated text-[20px] leading-none">
            ×
          </button>
        </div>

        <nav className="flex-1 overflow-y-auto overscroll-contain p-2 flex flex-col gap-0.5">
          {pages.map((p) => (
            <NavLink key={p.to} to={p.to} end={p.to === "/"} onClick={onClose}
              className={({ isActive }) =>
                `px-3 h-12 flex items-center rounded-xl text-[16px] font-semibold
                 transition-colors ${
                  isActive ? "bg-elevated text-ink" : "text-ink-dim active:bg-elevated"
                }`}>
              {p.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-3 py-3 border-t border-line shrink-0 flex items-center">
          {dev ? (
            <span className="text-[13px] text-brand-ink">本地开发模式</span>
          ) : (
            <button onClick={() => supabase.auth.signOut()}
              className="h-10 px-3 -ml-3 rounded-lg text-[15px] text-ink-mute active:bg-elevated">
              退出登录
            </button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}

function Burger() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M3 5.5h14M3 10h14M3 14.5h14" />
    </svg>
  );
}
