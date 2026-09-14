import { useState } from "react";
import { Navigate } from "react-router-dom";
import { supabase, supabaseConfigured } from "../lib/supabase";
import { useAuth } from "./AuthProvider";

export function LoginPage() {
  const { session, dev } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  if (session || dev) return <Navigate to="/" replace />;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.signInWithPassword({ email: email.trim(), password });
    setBusy(false);
    if (error) setError(error.message === "Invalid login credentials" ? "邮箱或密码不正确" : error.message);
  }

  async function forgot() {
    if (!email.trim()) {
      setError("先填写邮箱，再点“忘记密码”");
      return;
    }
    const { error } = await supabase.auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/set-password`,
    });
    setNotice(error ? null : "重置邮件已发送，请查收");
    if (error) setError(error.message);
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <form onSubmit={submit} className="card w-full max-w-sm p-6 flex flex-col gap-3">
        <h1 className="text-[22px] font-semibold">ASSRS 登录</h1>
        {!supabaseConfigured && (
          <p className="text-[13px] text-brand-ink">未配置 VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY。</p>
        )}
        <input type="email" required autoComplete="email" placeholder="邮箱" value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="bg-sunken rounded-[10px] h-11 px-3 outline-none focus:ring-2 focus:ring-cyan/40" />
        <input type="password" required autoComplete="current-password" placeholder="密码" value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="bg-sunken rounded-[10px] h-11 px-3 outline-none focus:ring-2 focus:ring-cyan/40" />
        {error && <p className="text-[13px] text-up">{error}</p>}
        {notice && <p className="text-[13px] text-down">{notice}</p>}
        <button disabled={busy} className="h-11 rounded-[12px] bg-cyan text-white font-semibold disabled:opacity-60">
          {busy ? "登录中…" : "登录"}
        </button>
        <button type="button" onClick={forgot} className="text-[13px] text-cyan self-start">忘记密码</button>
      </form>
    </div>
  );
}
