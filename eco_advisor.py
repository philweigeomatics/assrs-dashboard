"""
eco_advisor.py — read a BATCH of economic releases together, not one by one.

A single macro print means very little on its own. India's CPI at 5.54% is a
number; India's CPI at 5.54% against a 5.40% forecast and a 4.83% previous,
in the same week that Japanese producer prices undershot by 0.8pp, is the
beginning of an argument. So the unit of interpretation here is the
selection, not the event — one call for everything ticked, which is also the
cheap way to ask.

Division of labour (same as whatif_advisor and trade_review)
------------------------------------------------------------
Everything arithmetic is computed here and handed over as fact: the surprise
against forecast, the change against the previous reading, which releases
have already landed and which are still ahead. A model asked to subtract two
strings with percent signs in them will get some of them wrong with complete
confidence, and those differences are the whole substance of the exercise.

The model's job is to weigh them against each other, say which ones matter,
and say so plainly when a selection does not add up to anything — a handful
of unrelated bond auctions is not a theme, and inventing one is the failure
mode worth guarding against.

This is macro commentary for a technical workflow, not investment advice, and
the prompt says so to the model too.
"""

from __future__ import annotations

import re

#: Picked by hand, so a selection is a deliberate act rather than "everything".
#: Forty events is roughly a busy day, about 1,200 tokens of input.
MAX_EVENTS = 40
#: Below this many, a "holistic" read is just a restatement.
MIN_EVENTS = 1

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _num(text: str | None) -> float | None:
    """
    The number out of "4.9%", "6.58%", "-0.3万亿".

    Tushare bakes the unit into the string and the unit is not consistent
    even within one country, so the figure is only ever compared with another
    figure from the SAME event — forecast and previous for that release.
    """
    if not text:
        return None
    hit = _NUM.search(str(text).replace(",", ""))
    if not hit:
        return None
    try:
        return float(hit.group())
    except ValueError:
        return None


def _unit(text: str | None) -> str:
    """Whatever trails the number, so a delta can be written in the same unit."""
    if not text:
        return ""
    hit = _NUM.search(str(text).replace(",", ""))
    if not hit:
        return ""
    return str(text).replace(",", "")[hit.end():].strip()


def measure(event: dict) -> dict:
    """
    One event with its arithmetic done.

    `surprise` is actual − forecast and only exists once a release has landed
    with a forecast to miss; `change` is actual − previous; `expected_change`
    is forecast − previous, which is the only one an upcoming event has.
    """
    actual = _num(event.get("value"))
    forecast = _num(event.get("forecast"))
    prev = _num(event.get("prev"))
    unit = _unit(event.get("value") or event.get("forecast") or event.get("prev"))

    def diff(a, b):
        return None if a is None or b is None else round(a - b, 4)

    return {
        "date": event.get("date"),
        "time": event.get("time"),
        "country": event.get("country"),
        "currency": event.get("currency"),
        "event": event.get("event"),
        "unit": unit,
        "actual": actual, "forecast": forecast, "prev": prev,
        "surprise": diff(actual, forecast),
        "change": diff(actual, prev),
        "expected_change": diff(forecast, prev),
        "released": actual is not None,
    }


def build_brief(events: list[dict]) -> dict:
    """The selection, measured and sorted, ready to be argued about."""
    if not events:
        return {"ok": False, "reason": "没有选中任何事件"}
    if len(events) > MAX_EVENTS:
        return {"ok": False,
                "reason": f"一次最多 {MAX_EVENTS} 条，选中了 {len(events)} 条"}

    rows = [measure(e) for e in events]
    rows.sort(key=lambda r: (r["date"] or "", r["time"] or ""))

    released = [r for r in rows if r["released"]]
    upcoming = [r for r in rows if not r["released"]]
    countries: dict[str, int] = {}
    for r in rows:
        countries[r["country"] or "—"] = countries.get(r["country"] or "—", 0) + 1

    return {
        "ok": True,
        "count": len(rows),
        "rows": rows,
        "released": len(released),
        "upcoming": len(upcoming),
        "with_surprise": sum(1 for r in released if r["surprise"] is not None),
        "countries": sorted(countries.items(), key=lambda kv: -kv[1]),
        "span": [rows[0]["date"], rows[-1]["date"]],
    }


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:g}{unit}"


def _signed(v, unit: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:+g}{unit}"


def _lines(brief: dict) -> str:
    """
    One line per event, with the differences already taken.

    Compact on purpose: the selection is the payload and every token spent
    restating a label is one not spent on the reasoning.
    """
    out = []
    for r in brief["rows"]:
        u = r["unit"]
        bits = [f"{r['date']} {r['time'] or ''}".strip(),
                f"[{r['country']}]", r["event"]]
        if r["released"]:
            bits.append(f"实{_fmt(r['actual'], u)}")
            if r["forecast"] is not None:
                bits.append(f"预{_fmt(r['forecast'], u)}")
                bits.append(f"意外{_signed(r['surprise'], u)}")
            if r["prev"] is not None:
                bits.append(f"前{_fmt(r['prev'], u)}")
                bits.append(f"环比{_signed(r['change'], u)}")
        else:
            bits.append("未公布")
            if r["forecast"] is not None:
                bits.append(f"预{_fmt(r['forecast'], u)}")
            if r["prev"] is not None:
                bits.append(f"前{_fmt(r['prev'], u)}")
            if r["expected_change"] is not None:
                bits.append(f"预期变化{_signed(r['expected_change'], u)}")
        out.append("  " + " · ".join(bits))
    return "\n".join(out)


_SYSTEM = """\
你是一位宏观策略分析师。用户从经济日历里**勾选了一组事件**，要的是把这组
事件**放在一起**读出来的结论，而不是逐条解释。

几条硬性要求：

1. **差值已经算好了**（意外 = 实际−预期，环比 = 实际−前值）。直接引用，
   不要自己重算，也不要把"实际高于预期"说成相反。
2. **先说这组事件合在一起讲了什么**。如果它们本来就凑不成一个主题
   （比如全是国债拍卖），就**直说凑不成**，不要硬编一个叙事出来。
   这一点比给出结论更重要。
3. **区分已公布和未公布**。已公布的是事实，未公布的只是预期，
   对未公布的事件不要用"显示""表明"这类已然的说法。
4. 落到**可操作的观察点**：接下来看什么数据、什么情况会推翻这个判断。
5. 只做宏观与技术面的解读，**不是投资建议**。

**长度控制**：themes 最多 4 条，most_important 最多 3 条，
watch_next 最多 3 条。每段话不超过 3 句。宁可短而具体，不要长而泛。

输出严格的 JSON，不要加 markdown 代码块：
{
  "headline": "一句话总结这组事件（20字以内）",
  "coherent": true/false,
  "coherence_note": "如果 coherent=false，说明为什么这组事件读不出统一主题",
  "overall": "整体解读，2-4句",
  "themes": [
    {"theme": "通胀 / 增长 / 货币政策 / 流动性 / 其他",
     "read": "这个主题下发生了什么，引用具体数字",
     "events": ["对应的事件名，1-3个"]}
  ],
  "most_important": [
    {"event": "事件名", "why": "为什么这条最值得注意，引用数字"}
  ],
  "watch_next": ["接下来该盯什么，1-3条"],
  "what_would_change_it": "什么情况会推翻上面的判断"
}"""


def explain(brief: dict, *, timeout: int = 90) -> dict:
    """One DeepSeek call for the whole selection."""
    if not brief.get("ok"):
        raise LookupError(brief.get("reason") or "无法解读")

    import ai_client

    span = brief["span"]
    user = f"""\
勾选了 {brief['count']} 条经济数据事件（{span[0]} ~ {span[1]}）
已公布 {brief['released']} 条（其中 {brief['with_surprise']} 条有预期可比），\
未公布 {brief['upcoming']} 条
涉及：{"、".join(f"{c}×{n}" for c, n in brief['countries'])}

【事件明细 —— 差值已算好，直接引用】
{_lines(brief)}

请按系统提示里的 JSON 结构输出。"""

    # 3,000 truncated a real answer mid-string on a 12-event selection, and a
    # truncated JSON is a failed call rather than a short one.
    out = ai_client.call_json(_SYSTEM, user, max_tokens=6000,
                              temperature=0.3, timeout=timeout)
    return {
        "read": out,
        "count": brief["count"],
        "released": brief["released"],
        "upcoming": brief["upcoming"],
        "span": span,
    }
