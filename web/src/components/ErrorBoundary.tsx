/**
 * The thing standing between one bad field and a white screen.
 *
 * React unmounts the entire tree when a render throws. Without a boundary
 * anywhere in the app, a single component reading a property off a field the
 * API did not send takes down the whole page — no message, no navigation, no
 * way back except a manual reload. That is exactly what happened when the
 * frontend rolled out a few minutes ahead of the API and the pair table asked
 * a response for thresholds it did not yet carry.
 *
 * One per route rather than one at the root: a broken panel should cost you
 * that panel, not the navigation you need to get away from it.
 *
 * A class component because this is the one thing hooks still cannot do.
 */

import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode; label?: string };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The stack is the only way to find this after the fact — a boundary that
    // swallows the error silently trades a visible failure for an invisible one.
    console.error("[ErrorBoundary]", this.props.label ?? "", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="card p-6 flex flex-col items-center gap-2 text-center">
        <div className="text-[13px] font-semibold">这一部分没能显示出来</div>
        <p className="label max-w-[52ch] leading-snug">
          {this.props.label && <>「{this.props.label}」</>}
          出错了，页面其余部分仍然可用。如果刚刚发布过新版本，刷新一次通常就好了。
        </p>
        <code className="text-[11px] text-ink-mute font-mono max-w-full truncate">
          {error.message}
        </code>
        <div className="flex gap-2 mt-1">
          <button onClick={() => this.setState({ error: null })}
            className="h-8 px-3 rounded-lg bg-sunken text-[12.5px]">重试</button>
          <button onClick={() => globalThis.location.reload()}
            className="h-8 px-3 rounded-lg bg-cyan text-white text-[12.5px] font-semibold">
            刷新页面
          </button>
        </div>
      </div>
    );
  }
}
