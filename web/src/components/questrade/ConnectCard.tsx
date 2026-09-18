/**
 * Connecting a Questrade account — the instructions, and the one box.
 *
 * The steps are on the page rather than in a doc because getting this wrong is
 * expensive: Questrade refresh tokens are single-use, so a token pasted into
 * the wrong field, or generated twice, or copied with a trailing space, is
 * simply gone and has to be regenerated.
 *
 * Manual authorisation, not the OAuth redirect. A personal app can hand you a
 * refresh token directly, which means no callback URL has to be exposed, no
 * browser round trip, and nothing to misconfigure between here and Questrade.
 * The redirect field still has to be filled in at registration; it is never
 * exercised.
 *
 * The token is typed into a password field, POSTed once, and exchanged
 * server-side. It is never stored in the browser, never put in a URL, and the
 * API never sends any part of it back.
 */

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../../lib/api";

const STEPS = [
  {
    title: "登录 Questrade App Hub",
    body: <>打开 <Ext href="https://apphub.questrade.com">apphub.questrade.com</Ext>，
      用你平时的 Questrade 账号登录。也可以从网页版 Questrade 的
      <b> My Profile → API centre </b>进去。</>,
  },
  {
    title: "注册一个 personal app",
    body: <>选 <b>Register a personal app</b>。名字随便填（例如 <code>assrs-dashboard</code>），
      描述也随意。</>,
  },
  {
    title: "权限只勾选“读”",
    body: <>勾 <b>Read Accounts</b> 和 <b>Read Market Data</b>，
      <b className="text-up">不要勾 Trade / Submit Orders</b>。
      这样即使令牌泄露，也只能看，不能下单。</>,
  },
  {
    title: "Callback URL 填什么？",
    body: <>填 <code>https://localhost</code> 就行 —— 这个字段是必填的，但我们用的是
      <b> 手动授权</b>，整个流程不会跳转到它，所以它指向哪里都无所谓。
      （将来如果改成标准 OAuth 跳转，再把它换成 API 的 <code>/questrade/callback</code>。）</>,
  },
  {
    title: "生成手动刷新令牌",
    body: <>保存后回到这个 app 的页面，点 <b>Generate new token</b>，
      授权方式选 <b>Manual</b>。屏幕上会出现一串 refresh token。</>,
  },
  {
    title: "立刻粘贴到下面",
    body: <>复制那串令牌，贴进下面的框里点连接。
      <b className="text-up"> 这串令牌只能用一次</b> —— 用掉之后 Questrade 会发一个新的，
      由服务器保存并自动轮换；页面上不会再显示它。
      如果贴错了或者中途关掉，回 App Hub 重新生成一个即可。</>,
  },
];

function Ext({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer"
      className="text-cyan underline underline-offset-2">{children}</a>
  );
}

export function ConnectCard({ reason }: { reason?: string | null }) {
  const qc = useQueryClient();
  const [token, setToken] = useState("");

  const connect = useMutation({
    mutationFn: () => api.qtConnect(token.trim()),
    onSuccess: () => {
      setToken("");
      qc.invalidateQueries({ queryKey: ["qt"] });
    },
  });

  return (
    <div className="card p-4 flex flex-col gap-3 max-w-[760px]">
      <div>
        <h2 className="text-[15px] font-semibold">连接 Questrade</h2>
        <p className="label mt-0.5">
          只读连接。持仓、成本、账户余额来自 Questrade；历史行情仍走 Yahoo（已复权）。
        </p>
      </div>

      {reason && (
        <div className="rounded-lg bg-sunken px-3 py-2 text-[12.5px] text-up">
          {reason}
        </div>
      )}

      <ol className="flex flex-col gap-2">
        {STEPS.map((s, i) => (
          <li key={s.title} className="flex gap-2.5">
            <span className="shrink-0 w-5 h-5 rounded-full bg-sunken text-ink-dim
                             text-[11px] font-semibold flex items-center justify-center">
              {i + 1}
            </span>
            <div className="min-w-0">
              <div className="text-[13px] font-medium">{s.title}</div>
              <div className="text-[12.5px] text-ink-dim leading-snug">{s.body}</div>
            </div>
          </li>
        ))}
      </ol>

      <div className="flex flex-col gap-1.5 pt-1">
        <label className="label" htmlFor="qt-token">Refresh token</label>
        <div className="flex gap-2">
          <input id="qt-token" type="password" value={token} autoComplete="off"
            spellCheck={false}
            onChange={(e) => setToken(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && token.trim()) connect.mutate(); }}
            placeholder="从 App Hub 复制的那一串"
            className="flex-1 h-9 px-2.5 rounded-lg bg-sunken text-[13px] font-mono
                       outline-none focus:ring-2 focus:ring-cyan/40" />
          <button onClick={() => connect.mutate()}
            disabled={connect.isPending || token.trim().length < 8}
            className="h-9 px-4 rounded-lg bg-cyan text-white text-[13px] font-semibold
                       disabled:opacity-60">
            {connect.isPending ? "连接中…" : "连接"}
          </button>
        </div>
        {connect.isError && (
          <span className="text-[12.5px] text-up">
            {(connect.error as ApiError).message}
          </span>
        )}
        <p className="text-[11.5px] text-ink-mute leading-snug">
          令牌只发往服务器并立即与 Questrade 兑换，不会保存在浏览器里，也不会出现在任何链接中。
          兑换后得到的新令牌存放在数据库中仅服务端可读的表里，之后由服务器自动轮换。
        </p>
      </div>
    </div>
  );
}
