"""
scan_watchlists.py — nightly Today's-Alerts scan, run by GitHub Actions.

Why this exists
---------------
Scanning a watchlist inside Streamlit Community Cloud runs every stock's
walk-forward HMM regime detector in a single script run — ~7s of CPU per stock
on a fast desktop, far more on Community Cloud's shared CPU — and the free
tier throttles exactly that shape of workload. The HMM is slow on purpose (it
refits only on past data, which is what keeps it free of lookahead, and it
selects the MACD gear the signals depend on), so the fix is to run it
somewhere that is allowed to be slow. A GitHub Actions runner is; the page
then just reads the cache.

What it does
------------
1. Reads every user's watchlist.
2. Scans each DISTINCT ticker once. Two users watching 600519 cost one scan,
   because a stock's signals do not depend on who is watching it.
3. Retries anything that failed, once more, after the main pass.
4. Writes each user's snapshot through the same cache path the page uses,
   keyed by the TRADING SESSION the data belongs to, not the wall clock.

Why serial by default
---------------------
It was built parallel and measured before shipping. On 8 stocks, 4 workers
took 45s against 44s serial — worker start-up re-imports the whole stack and
ate the gain — and the parallel run lost two stocks to transient fetch failures
that the serial run did not hit, because four processes hitting Tushare at once
is a burst. This job runs unattended at 20:00 Beijing; finishing sooner buys
nothing, and a snapshot that silently omits stocks costs a great deal. So it
runs one stock at a time. --workers is still there if a watchlist ever grows
enough to need it — measure again first.

Run locally:   python scan_watchlists.py [--only TICKER ...] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter


def _init_worker():
    # CPU-bound numpy/sklearn work: one BLAS thread per process, or N workers
    # each spawning N threads oversubscribe the runner and run slower.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _scan_one(ticker: str) -> dict:
    """
    Signals AND chip structure for one stock, from ONE price fetch. The chip
    result rides along under `chips` so both scans share the retry pass.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import watchlist_scan
    t0 = time.perf_counter()
    frame = watchlist_scan.fetch_frame(ticker)
    out = watchlist_scan.scan_ticker(ticker, stock_df=frame)
    if frame is None:
        out["chips"] = {"ticker": ticker, "status": "error",
                        "error": "price fetch failed"}
    else:
        out["chips"] = watchlist_scan.scan_chips(ticker, stock_df=frame)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _chip_failed(r: dict) -> bool:
    return r.get("chips", {}).get("status") == "error"


def _run(tickers: list[str], workers: int, label: str) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if workers <= 1:
        for n, t in enumerate(tickers, 1):
            r = _scan_one(t)
            results[t] = r
            _log(n, len(tickers), r, label)
        return results
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futs = {pool.submit(_scan_one, t): t for t in tickers}
        for n, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as e:          # a crashed worker, not a bad stock
                r = {"ticker": t, "status": "error", "error": repr(e)[:300]}
            results[t] = r
            _log(n, len(tickers), r, label)
    return results


def _in_market(symbol: str, want: str) -> bool:
    """Whether a watchlist symbol belongs to the market this run is for."""
    if want == "ALL":
        return True
    import markets
    try:
        code = markets.split(symbol)[0]
    except LookupError:
        return False              # unparseable: not this run's problem either
    return code == "CN" if want == "CN" else code in ("US", "CA")


def _log(n, total, r, label):
    c = r.get("chips", {})
    print(f"  {label}[{n:>3}/{total}] {r['ticker']} {r['status']:<8}"
          f"chips {c.get('status', '-'):<8}"
          f"{r.get('seconds', 0):>6}s"
          + (f"  {r.get('error', '')[:90]}" if r['status'] == 'error' else "")
          + (f"  chips: {c.get('error', '')[:90]}" if c.get('status') == 'error' else ""),
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="scan just these tickers (no DB write)")
    ap.add_argument("--market", default="CN", choices=["CN", "NA", "ALL"],
                    help="CN = A-shares (Tushare, 20:00 Beijing); "
                         "NA = US/Canada (Yahoo, after the 16:00 ET close)")
    args = ap.parse_args()

    _init_worker()
    import data_manager
    import watchlist_scan

    if args.only:
        watchlists = {"__dry_run__": sorted(set(args.only))}
    else:
        watchlists = data_manager.get_all_watchlists()
    if not watchlists:
        print("No watchlists found — nothing to scan.")
        return 0

    tickers = sorted({t for ts in watchlists.values() for t in ts})
    # One table holds every market's watchlist, but the two markets close
    # nine hours apart and come from different data sources, so each run takes
    # only its own. Without this the A-share job at 20:00 Beijing would fetch
    # US symbols from Tushare and log an error a night, per symbol, forever.
    before = len(tickers)
    tickers = [t for t in tickers if _in_market(t, args.market)]
    if before != len(tickers):
        print(f"{before - len(tickers)} ticker(s) belong to another market "
              f"— skipped (--market {args.market})", flush=True)
    if not tickers:
        print(f"No {args.market} tickers to scan.")
        return 0
    total_refs = sum(len(v) for v in watchlists.values())
    print(f"{len(watchlists)} user(s) · {total_refs} watchlist entries · "
          f"{len(tickers)} distinct tickers · workers={args.workers}", flush=True)

    t_start = time.perf_counter()
    results = _run(tickers, args.workers, "")

    # Second chance for anything that errored. Transient failures cluster in
    # time, so the retry waits and goes one at a time regardless of --workers.
    failed = [t for t, r in results.items()
              if r["status"] == "error" or _chip_failed(r)]
    if failed:
        print(f"\nretrying {len(failed)} failed ticker(s) after a pause…", flush=True)
        time.sleep(15)
        results.update(_run(failed, 1, "retry "))
    wall = time.perf_counter() - t_start

    status = Counter(r["status"] for r in results.values())
    still_failed = sorted(t for t, r in results.items() if r["status"] == "error")
    chip_status = Counter(r.get("chips", {}).get("status", "-") for r in results.values())
    chips_failed = sorted(t for t, r in results.items() if _chip_failed(r))
    chip_rows = [row for r in results.values()
                 if r.get("chips", {}).get("status") == "ok"
                 for row in r["chips"]["rows"]]
    print(f"\nscanned {len(results)} in {wall:.0f}s — " +
          ", ".join(f"{k} {v}" for k, v in sorted(status.items())) +
          " · chips " + ", ".join(f"{k} {v}" for k, v in sorted(chip_status.items())),
          flush=True)

    # Refuse to publish a snapshot built mostly from failures: a run where the
    # data source was down would otherwise overwrite yesterday's good cache
    # with an almost-empty "no signals", which reads as a quiet market.
    if results and len(still_failed) / len(results) > 0.5:
        print(f"❌ {len(still_failed)}/{len(results)} tickers failed — NOT writing snapshots.")
        return 1

    if args.only:
        rows = [r["row"] for r in results.values() if r["status"] == "ok"]
        print(watchlist_scan.rank_rows(rows).to_string(index=False)[:4000])
        if chip_rows:
            import pandas as pd
            print("\n筹码结构 (decay 1.0):")
            print(pd.DataFrame([c for c in chip_rows if c["decay"] == 1.0])
                  .sort_values("setup_score", ascending=False)
                  .drop(columns=["decay"]).to_string(index=False)[:4000])
        return 1 if (still_failed or chips_failed) else 0

    # Keyed by the trading session the data belongs to. The most common
    # last-bar date, so one suspended ticker cannot drag the snapshot onto
    # the wrong day.
    dates = Counter(r["data_date"] for r in results.values() if r.get("data_date"))
    if not dates:
        print("❌ no usable data dates — NOT writing snapshots.")
        return 1
    scan_date = dates.most_common(1)[0][0]
    per_ticker_s = wall / max(len(results), 1)

    wrote = 0
    for uid, tks in watchlists.items():
        rows = [results[t]["row"] for t in tks
                if results.get(t, {}).get("status") == "ok"]
        df = watchlist_scan.rank_rows(rows)
        missing = [t for t in tks if t in still_failed]
        if df.empty:
            print(f"  user {str(uid)[:8]}… {len(tks)} stocks, no signals")
            continue
        ok = data_manager.save_signals_to_cache_for_user(
            df, scan_date, round(per_ticker_s * len(tks), 1), uid)
        wrote += bool(ok)
        print(f"  user {str(uid)[:8]}… {len(tks)} stocks → {len(df)} rows "
              f"{'saved' if ok else 'SAVE FAILED'}"
              + (f"  (missing {len(missing)}: {', '.join(missing)})" if missing else ""))

    print(f"\n✅ wrote {wrote} snapshot(s) for session {scan_date}")

    # Chip structure: one shared set of per-ticker rows, not per user. Written
    # independently of the signal snapshots — a quiet market (no signal rows)
    # still has a chip structure worth showing. The >50% guard above already
    # stopped a run where the data source was down.
    chips_ok = True
    if chip_rows:
        chips_ok = data_manager.save_chip_scan(chip_rows, scan_date)
        n_tk = len({c["ticker"] for c in chip_rows})
        print(f"{'✅' if chips_ok else '❌'} chip structure: {n_tk} tickers × "
              f"{len(watchlist_scan.CHIP_DECAYS)} decays "
              f"{'saved' if chips_ok else 'SAVE FAILED'} for session {scan_date}")
    else:
        print("⚠️ chip structure: no rows computed")

    # A partial snapshot is still better than a stale one, so it is written —
    # but the run exits non-zero so GitHub marks it failed and emails, rather
    # than a stock quietly dropping out of someone's alerts.
    rc = 0
    if still_failed:
        print(f"⚠️ {len(still_failed)} ticker(s) failed even after retry: "
              f"{', '.join(still_failed)}")
        rc = 1
    if chips_failed:
        print(f"⚠️ chip structure failed for {len(chips_failed)} ticker(s) after retry: "
              f"{', '.join(chips_failed)}")
        rc = 1
    if not chips_ok:
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
