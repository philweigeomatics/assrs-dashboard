"""
risk_sources.py — where 🧭 风险雷达's evidence comes from, one function per feed.

Every function returns a list of items in one shape — {eid, d, source, kind,
title, url} plus, for the feeds that can be read by rule, {risk, dir,
strength, fact, note, tagged_by: "rule"} — and RAISES when its source is
broken. collect() turns a raise into a reported failure, so "this feed is
down" is never shown as "nothing happened".

A-SHARES (free only: this account has no Tushare news, announcement or
互动易 permission — each returned "没有接口访问权限" when probed)
  巨潮资讯 announcements, 互动易 answered questions, 东方财富 research reports,
  and four market-wide news feeds filtered by the company's names and its
  macro keywords — all via AKShare, which scrapes and so does break; plus
  新闻联播 for policy. Tushare, where this account does have access:
  增减持 (stk_holdertrade), 限售解禁 (share_float), 股权质押 (pledge_stat),
  filed by rule.

US / CANADA
  SEC EDGAR (free, no key; asks for a contact in the User-Agent and ≤10
  requests a second): 8-K items whose meaning is fixed by the form, Form 144
  intended insider sales (one item per window), registration statements, and
  Form 4 open-market buys and discretionary sells, all filed by rule. Sales
  under a 10b5-1 plan are skipped: executives of every large company sell on
  a schedule, and counted they made 股东行为 read 高 for all of them. Yahoo company news when it answers, and
  the Guardian for the company and its macro keywords.

The market-wide feeds only serve their newest items — 东方财富's 200, 新浪's 20
— so a stock's news record is built up by refreshing, not fetched whole.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date, timedelta

import pandas as pd

from risk_radar import evidence_id

LOOKBACK_DAYS = 90
SEC_UA = {"User-Agent": "ASSRS risk radar philweigeomatics@gmail.com"}


def _item(source: str, kind: str, d, title: str, url: str | None = None,
          key: str | None = None, **tag) -> dict:
    d = pd.Timestamp(d).date().isoformat()
    title = re.sub(r"\s+", " ", str(title)).strip()[:300]
    it = {"eid": evidence_id(source, key or title, d if key is None else key),
          "d": d, "source": source, "kind": kind, "title": title, "url": url or None,
          "risk": None, "dir": None, "strength": None, "fact": None, "note": None,
          "tagged_by": None}
    if tag:
        it.update({"fact": True, "tagged_by": "rule", **tag})
    return it


def _since(days: int = LOOKBACK_DAYS) -> date:
    return date.today() - timedelta(days=days)


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() in ("", "None", "nan")


# ── A-shares: AKShare ────────────────────────────────────────────────────────
def cn_announcements(code: str, **_) -> list[dict]:
    import akshare as ak
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=code, market="沪深京", start_date=_since().strftime("%Y%m%d"),
        end_date=date.today().strftime("%Y%m%d"))
    return [_item("cninfo", "filing", r["公告时间"], f"公告：{r['公告标题']}", r.get("公告链接"))
            for _, r in df.iterrows() if not _blank(r.get("公告标题"))]


def cn_investor_qa(code: str, **_) -> list[dict]:
    """Answered questions only: an unanswered one is an investor's guess."""
    import akshare as ak
    df = ak.stock_irm_cninfo(symbol=code)
    out = []
    for _, r in df.iterrows():
        if _blank(r.get("回答内容")):
            continue
        d = pd.Timestamp(r.get("更新时间") if not _blank(r.get("更新时间")) else r["提问时间"])
        if d.date() < _since():
            continue
        q, a = str(r["问题"])[:90], str(r["回答内容"])[:170]
        out.append(_item("irm", "qa", d, f"问：{q} ｜ 答：{a}", key=str(r.get("问题编号") or q)))
    return out


def cn_research(code: str, **_) -> list[dict]:
    import akshare as ak
    df = ak.stock_research_report_em(symbol=code)
    out = []
    for _, r in df.iterrows():
        if _blank(r.get("日期")) or pd.Timestamp(r["日期"]).date() < _since():
            continue
        rating = "" if _blank(r.get("东财评级")) else f"（{r['东财评级']}）"
        out.append(_item("research_em", "research", r["日期"],
                         f"{r['机构']}{rating}：{r['报告名称']}", r.get("报告PDF链接")))
    return out


#: Market-wide feeds: (AKShare function, title column, text column, time column).
_CN_FEEDS = {
    "news_em": ("stock_info_global_em", "标题", "摘要", "发布时间"),
    "news_ths": ("stock_info_global_ths", "标题", "内容", "发布时间"),
    "news_futu": ("stock_info_global_futu", "标题", "内容", "发布时间"),
    "news_sina": ("stock_info_global_sina", None, "内容", "时间"),
}
#: Macro-keyword matches kept per feed. Keywords like 电池 match a lot.
MACRO_PER_FEED = 15


def _matches(text: str, words: list[str]) -> bool:
    return any(w and w in text for w in words)


def cn_feed(feed: str):
    fn, tcol, xcol, dcol = _CN_FEEDS[feed]

    def run(code: str, aliases: list[str], keywords: list[str], **_) -> list[dict]:
        import akshare as ak
        df = getattr(ak, fn)()
        own, macro = [], []
        for _, r in df.iterrows():
            title = "" if tcol is None or _blank(r.get(tcol)) else str(r[tcol])
            body = "" if _blank(r.get(xcol)) else str(r[xcol])
            text = f"{title} {body}"
            head = (title or body)[:160]
            if _matches(text, aliases):
                own.append(_item(feed, "news", r[dcol], head, r.get("链接")))
            elif _matches(text, keywords) and len(macro) < MACRO_PER_FEED:
                macro.append(_item(feed, "news", r[dcol], head, r.get("链接")))
        return own + macro
    return run


def cn_stock_news(code: str, **_) -> list[dict]:
    """东方财富's per-stock news. Broken on AKShare 1.17.80; kept so an upgrade revives it."""
    import akshare as ak
    df = ak.stock_news_em(symbol=code)
    return [_item("news_em_stock", "news", r["发布时间"], r["新闻标题"], r.get("新闻链接"))
            for _, r in df.iterrows()
            if not _blank(r.get("新闻标题")) and pd.Timestamp(r["发布时间"]).date() >= _since()]


#: Days of 新闻联播 read per refresh. AKShare fetches each day's articles one
#: page at a time; five days ran past the 75s feed deadline.
CCTV_DAYS = 2


def cn_cctv(code: str, keywords: list[str], **_) -> list[dict]:
    import akshare as ak
    out = []
    for k in range(1, CCTV_DAYS + 1):
        day = date.today() - timedelta(days=k)
        try:
            df = ak.news_cctv(date=day.strftime("%Y%m%d"))
        except Exception:
            continue
        for _, r in df.iterrows():
            text = f"{r['title']} {r['content']}"
            if _matches(text, keywords):
                out.append(_item("cctv", "macro", day, f"新闻联播：{r['title']}"))
    return out


# ── A-shares: Tushare, filed by rule ─────────────────────────────────────────
_PRO = []
_PRO_LOCK = threading.Lock()


def _pro():
    """
    One client, made once. data_manager.init_tushare() rebuilds it on every
    call and ts.set_token rewrites a token file as it goes; three feeds
    calling it at once read that file half-written ("No columns to parse
    from file") and one of them failed on every refresh.
    """
    with _PRO_LOCK:
        if not _PRO:
            import macro
            _PRO.append(macro._api())
        return _PRO[0]


def _ts_code(code: str) -> str:
    import data_manager
    return data_manager.get_tushare_ticker(code)


#: % of total shares below which a day's 增减持 is immaterial. On 300750,
#: 9万股 sales showed as 0.00% and, one per day, still pinned 股东行为 at 96.
MIN_HOLDER_RATIO = 0.05


def cn_holder_trades(code: str, **_) -> list[dict]:
    """
    One item per announcement day and direction, not per holder row: one
    announcement listing twenty executives' small sales is one event, and
    counted row by row it read as twenty and pinned 股东行为 at 高.
    """
    df = _pro().stk_holdertrade(ts_code=_ts_code(code),
                                start_date=_since(180).strftime("%Y%m%d"))
    if df is None or df.empty:
        return []
    df = df.assign(sell=df["in_de"].astype(str).str.upper() == "DE",
                   change_ratio=pd.to_numeric(df["change_ratio"], errors="coerce").fillna(0),
                   change_vol=pd.to_numeric(df["change_vol"], errors="coerce").fillna(0))
    out = []
    for (ann, sell), g in df.groupby(["ann_date", "sell"]):
        ratio, vol = float(g["change_ratio"].sum()), float(g["change_vol"].sum())
        if ratio < MIN_HOLDER_RATIO:
            continue
        who = "、".join(dict.fromkeys(str(h) for h in g["holder_name"]))[:40]
        title = (f"{who}{'减持' if sell else '增持'} {vol / 1e4:,.0f} 万股"
                 f"（合计占总股本 {ratio:.2f}%，{len(g)} 条）")
        out.append(_item("ts_holdertrade", "filing", ann, title, key=f"{ann}|{sell}",
                         risk="holders", dir="down" if sell else "up",
                         strength=3 if ratio >= 1 else 2 if ratio >= 0.2 else 1,
                         note="股东增减持（交易所披露）"))
    return out


def cn_unlocks(code: str, **_) -> list[dict]:
    """
    Unlocks from two weeks ago to three months ahead, one item per date.

    Dated TODAY while still ahead, under an id fixed by the unlock date, so
    the item stays at full weight until it happens instead of decaying from
    the day it was announced.
    """
    df = _pro().share_float(ts_code=_ts_code(code))
    if df is None or df.empty:
        return []
    df["float_date"] = pd.to_datetime(df["float_date"].astype(str), errors="coerce")
    lo, hi = pd.Timestamp(_since(15)), pd.Timestamp(date.today() + timedelta(days=90))
    df = df[(df["float_date"] >= lo) & (df["float_date"] <= hi)]
    out = []
    for fd, g in df.groupby("float_date"):
        ratio = float(pd.to_numeric(g["float_ratio"], errors="coerce").fillna(0).sum())
        shares = float(pd.to_numeric(g["float_share"], errors="coerce").fillna(0).sum())
        ahead = fd.date() >= date.today()
        title = (f"{fd.date().isoformat()} {'将' if ahead else '已'}解禁限售股 "
                 f"{shares / 1e8:.2f} 亿股，占总股本 {ratio:.2f}%")
        out.append(_item("ts_share_float", "data", date.today() if ahead else fd, title,
                         key=f"unlock|{fd.date().isoformat()}",
                         risk="holders", dir="down",
                         strength=3 if ratio >= 5 else 2 if ratio >= 1 else 1,
                         note="限售股解禁（交易所数据）"))
    return out


def cn_pledges(code: str, **_) -> list[dict]:
    df = _pro().pledge_stat(ts_code=_ts_code(code))
    if df is None or df.empty:
        return []
    r = df.sort_values("end_date").iloc[-1]
    ratio = float(r.get("pledge_ratio") or 0)
    if ratio < 5:
        return []
    return [_item("ts_pledge", "data", r["end_date"],
                  f"股权质押比例 {ratio:.1f}%（截至 {r['end_date']}）",
                  risk="holders", dir="down",
                  strength=3 if ratio >= 30 else 2 if ratio >= 15 else 1,
                  note="质押比例高时，股价下跌可能触发平仓")]


# ── US: SEC EDGAR ────────────────────────────────────────────────────────────
_CIK: dict[str, int] = {}
_CIK_LOCK = threading.Lock()

#: 8-K items whose meaning the form itself fixes. The rest (results, material
#: agreements, Reg FD) say THAT something happened, not which way.
EIGHT_K = {
    "1.03": ("legal", "down", 3, "破产或接管"),
    "1.05": ("governance", "down", 2, "重大网络安全事件"),
    "2.04": ("balance", "down", 2, "触发债务加速到期"),
    "2.06": ("balance", "down", 2, "重大资产减值"),
    "3.01": ("legal", "down", 3, "退市或不再符合上市标准通知"),
    "3.02": ("capital", "down", 1, "未注册发行股份"),
    "4.01": ("governance", "down", 1, "更换审计师"),
    "4.02": ("governance", "down", 3, "此前财报不可再依赖"),
    "5.02": ("governance", "down", 1, "董事或高管变动"),
}
REGISTRATIONS = {"S-1", "S-3", "S-3ASR", "424B4", "S-1/A"}


def _sec_get(url: str):
    import requests
    r = requests.get(url, headers=SEC_UA, timeout=20)
    r.raise_for_status()
    return r


def cik_for(ticker: str) -> int | None:
    with _CIK_LOCK:
        if not _CIK:
            for v in _sec_get("https://www.sec.gov/files/company_tickers.json").json().values():
                _CIK[str(v["ticker"]).upper()] = int(v["cik_str"])
    return _CIK.get(ticker.upper().replace(".", "-"))


def _form4(cik: int, acc: str, doc: str) -> dict | None:
    """Open-market buys (P) and sells (S) in one Form 4; grants and exercises ignored."""
    import xml.etree.ElementTree as ET
    raw = doc.split("/")[-1]                          # drop the xslF345X05/ styling path
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{raw}"
    root = ET.fromstring(_sec_get(url).content)
    who = (root.findtext(".//reportingOwner/reportingOwnerId/rptOwnerName") or "").strip()
    buy = sell = 0.0
    for t in root.findall(".//nonDerivativeTransaction"):
        codev = (t.findtext("./transactionCoding/transactionCode") or "").strip()
        try:
            sh = float(t.findtext("./transactionAmounts/transactionShares/value") or 0)
            px = float(t.findtext("./transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        if codev == "P":
            buy += sh * px
        elif codev == "S":
            sell += sh * px
    if buy == 0 and sell == 0:
        return None
    # Since 2023 the form says whether the trade ran under a 10b5-1 plan
    # set up months earlier. A planned sale is a schedule, not a view.
    planned = (root.findtext(".//aff10b5One") or "").strip().lower() in ("1", "true")
    return {"who": who, "buy": buy, "sell": sell, "planned": planned,
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"}


def us_edgar(code: str, **_) -> list[dict]:
    cik = cik_for(code)
    if cik is None:
        return []
    rec = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
    rows = pd.DataFrame({k: rec[k] for k in ("form", "filingDate", "items", "accessionNumber",
                                             "primaryDocument")})
    rows = rows[pd.to_datetime(rows["filingDate"]).dt.date >= _since(120)]
    index = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
    out, form4s, f144 = [], [], []
    for _, r in rows.iterrows():
        form, d = r["form"], r["filingDate"]
        if form == "8-K":
            for item in [i.strip() for i in str(r["items"]).split(",") if i.strip()]:
                if item in EIGHT_K:
                    risk, dirn, strength, what = EIGHT_K[item]
                    out.append(_item("sec_8k", "filing", d, f"8-K Item {item}：{what}", index,
                                     key=f"{r['accessionNumber']}|{item}", risk=risk,
                                     dir=dirn, strength=strength, note="8-K 条目（SEC）"))
        elif form == "144":
            f144.append(d)
        elif form in REGISTRATIONS:
            out.append(_item("sec_reg", "filing", d, f"{form}：注册发行证券", index,
                             key=r["accessionNumber"], risk="capital", dir="down",
                             strength=1, note="可能带来股本稀释"))
        elif form == "4" and len(form4s) < 12:
            form4s.append(r)
    if f144:
        # One item for the window, not one per notice: large companies file
        # these weekly, and each is an intention, not a sale.
        out.append(_item("sec_144", "filing", max(f144),
                         f"近 120 天 {len(f144)} 份 Form 144（内部人申报拟出售股份）", index,
                         key=f"144|{max(f144)}", risk="holders", dir="down", strength=1,
                         note="拟出售申报，不等于已卖出"))
    planned = 0
    for r in form4s:
        try:
            f = _form4(cik, r["accessionNumber"], r["primaryDocument"])
        except Exception:
            continue
        if not f:
            continue
        net = f["buy"] - f["sell"]
        if net <= 0 and f["planned"]:
            planned += 1
            continue
        if net > 0:
            out.append(_item("sec_form4", "filing", r["filingDate"],
                             f"{f['who']} 公开市场买入 ${f['buy'] / 1e6:,.2f}M（Form 4）", f["url"],
                             key=r["accessionNumber"], risk="holders", dir="up",
                             strength=3 if f["buy"] >= 1e6 else 2, note="内部人自掏腰包买入"))
        else:
            out.append(_item("sec_form4", "filing", r["filingDate"],
                             f"{f['who']} 公开市场卖出 ${f['sell'] / 1e6:,.2f}M（Form 4）", f["url"],
                             key=r["accessionNumber"], risk="holders", dir="down",
                             strength=2 if f["sell"] >= 1e7 else 1,
                             note="非 10b5-1 计划内的主动卖出"))
    return out


# ── US: news ─────────────────────────────────────────────────────────────────
def us_yahoo_news(code: str, **_) -> list[dict]:
    import yfinance as yf
    out = []
    for n in yf.Ticker(code).news or []:
        c = n.get("content") or n
        title = c.get("title")
        when = c.get("pubDate") or c.get("displayTime")
        if not title or not when:
            continue
        url = ((c.get("canonicalUrl") or {}).get("url")
               or (c.get("clickThroughUrl") or {}).get("url"))
        summary = str(c.get("summary") or "")[:160]
        out.append(_item("yahoo", "news", when, f"{title} — {summary}" if summary else title, url))
    return out


def _guardian(query: str, window: int, n: int, source: str) -> list[dict]:
    import game_news
    res = game_news.fetch_guardian(query, date.today().isoformat(),
                                   window_days=window, max_records=n)
    if not res["ok"]:
        raise RuntimeError(res["reason"] or "Guardian 不可用")
    return [_item(source, "news", a["seendate"], a["title"], a["url"]) for a in res["articles"]]


def us_guardian(code: str, aliases: list[str], keywords: list[str], **_) -> list[dict]:
    names = [a for a in aliases if a and not a.isupper()][:1] or aliases[:1]
    out = []
    if names:
        out += _guardian(f'"{names[0]}"', 30, 20, "guardian")
    for kw in keywords[:3]:
        out += _guardian(kw, 14, 6, "guardian_macro")
    return out


# ── running them ─────────────────────────────────────────────────────────────
SOURCES = {
    "CN": [
        ("cninfo", "巨潮资讯 公告", cn_announcements),
        ("irm", "互动易 问答", cn_investor_qa),
        ("research_em", "东方财富 研报", cn_research),
        ("news_em_stock", "东方财富 个股新闻", cn_stock_news),
        ("news_em", "东方财富 快讯", cn_feed("news_em")),
        ("news_ths", "同花顺 快讯", cn_feed("news_ths")),
        ("news_futu", "富途 快讯", cn_feed("news_futu")),
        ("news_sina", "新浪 快讯", cn_feed("news_sina")),
        ("cctv", "新闻联播", cn_cctv),
        ("ts_holdertrade", "Tushare 增减持", cn_holder_trades),
        ("ts_share_float", "Tushare 限售解禁", cn_unlocks),
        ("ts_pledge", "Tushare 股权质押", cn_pledges),
    ],
    "US": [
        ("sec", "SEC EDGAR", us_edgar),
        ("yahoo", "Yahoo 新闻", us_yahoo_news),
        ("guardian", "Guardian", us_guardian),
    ],
}
SOURCES["CA"] = [s for s in SOURCES["US"] if s[0] != "sec"]   # EDGAR covers US filers only


def collect(symbol: str, market_code: str, name: str, profile: dict | None,
            deadline: int = 75) -> tuple[list[dict], list[dict]]:
    """Every feed for one stock, in parallel. (items, one status per feed)."""
    import markets
    _, code = markets.split(symbol)
    profile = profile or {}
    aliases = [a for a in dict.fromkeys([name, *(profile.get("aliases") or [])]) if a and len(a) >= 2]
    keywords = [k for k in (profile.get("keywords") or []) if k and len(k) >= 2]

    def run(spec):
        key, label, fn = spec
        t0 = time.time()
        try:
            got = fn(code, aliases=aliases, keywords=keywords)
            return got, {"source": key, "label": label, "ok": True, "n": len(got),
                         "reason": None, "secs": round(time.time() - t0, 1)}
        except Exception as exc:
            return [], {"source": key, "label": label, "ok": False, "n": 0,
                        "reason": f"{type(exc).__name__}: {exc}"[:160],
                        "secs": round(time.time() - t0, 1)}

    specs = SOURCES.get(market_code, [])
    # A scraper that hangs must not take the request past Cloud Run's limit.
    # Its thread cannot be killed, so it finishes in the background and is
    # simply not waited for; the feed is reported as timed out.
    pool = ThreadPoolExecutor(max_workers=6)
    futures = [pool.submit(run, s) for s in specs]
    wait(futures, timeout=deadline)
    pool.shutdown(wait=False, cancel_futures=True)
    results = []
    for (key, label, _), f in zip(specs, futures):
        if f.done():
            results.append(f.result())
        else:
            results.append(([], {"source": key, "label": label, "ok": False, "n": 0,
                                 "reason": f"超过 {deadline}s 未返回", "secs": deadline}))
    items, seen = [], set()
    for got, _ in results:
        for it in got:
            if it["eid"] not in seen:
                seen.add(it["eid"])
                items.append(it)
    return items, [st for _, st in results]
