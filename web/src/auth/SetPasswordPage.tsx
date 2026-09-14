/**
 * Landing page for Supabase invite and password-reset emails.
 *
 * The email link signs the user in via tokens in the URL, which supabase-js
 * picks up on load (detectSessionInUrl). This page then lets them choose a
 * password. Set Supabase → Authentication → URL Configuration so the redirect
 * targets <site>/set-password.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { useAuth } from "./AuthProvider";

export function SetPasswordPage() {
  const { ready, session } = useAuth();
  const nav = useNavigate();
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (pw.length < 8) return setError("密码至少 8 位");
    if (pw !== pw2) return setError("两次输入不一致");
    setBusy(true);
    const { error } = await supabase.auth.updateUser({ password: pw });
    setBusy(false);
    if (error) return setError(error.message);
    nav("/", { replace: true });
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <form onSubmit={submit} className="card w-full max-w-sm p-6 flex flex-col gap-3">
        <h1 className="text-[22px] font-semibold">设置密码</h1>
        {!ready ? (
          <p className="label">验证链接中…</p>
        ) : !session ? (
          <p className="text-[13px] text-up">链接无效或已过期。请回到登录页点“忘记密码”重新发送。</p>
        ) : (
          <>
            <input type="password" autoComplete="new-password" placeholder="新密码（至少 8 位）" value={pw}
              onChange={(e) => setPw(e.target.value)}
              className="bg-sunken rounded-[10px] h-11 px-3 outline-none focus:ring-2 focus:ring-cyan/40" />
            <input type="password" autoComplete="new-password" placeholder="再输入一次" value={pw2}
              onChange={(e) => setPw2(e.target.value)}
              className="bg-sunken rounded-[10px] h-11 px-3 outline-none focus:ring-2 focus:ring-cyan/40" />
            {error && <p className="text-[13px] text-up">{error}</p>}
            <button disabled={busy} className="h-11 rounded-[12px] bg-cyan text-white font-semibold disabled:opacity-60">
              {busy ? "保存中…" : "保存并进入"}
            </button>
          </>
        )}
      </form>
    </div>
  );
}
