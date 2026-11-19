"""
个股研报 — overview, Porter's and PESTEL from one AI call.

They were three calls with the same two-line input about the same company.
Measured side by side on 华勤技术 and 光迅科技, the single call used 16-41%
fewer output tokens (most of the saving is reasoning the model no longer
repeats three times) and took one 25-second wait instead of three.

The contract these tests hold:
  - a first generation is ONE call, whichever section asked for it,
  - each section is still cached in its own row, so per-section readers
    (the Streamlit page, `cached_ai`) do not change,
  - redoing one section redoes only that one,
  - a section that comes back mangled is not saved, and does not take the
    good ones down with it.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

GOOD = {
    "overview": {"tagline": "ODM", "summary": "Makes phones for brands."},
    "pestel": {k: ["x"] for k in ("political", "economic", "social",
                                  "technological", "environmental", "legal")},
    "porters": {k: {"score": 3, "summary": "x"} for k in
                ("rivalry", "suppliers", "buyers", "substitutes", "new_entrants")},
}


class FakeDB:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}

    def read_table(self, table, filters=None, **_):
        f = filters or {}
        hits = [r for r in self.rows.values()
                if all(r.get(k) == v for k, v in f.items())]
        return pd.DataFrame(hits)

    def delete_records(self, table, filters=None, **_):
        self.rows.pop((filters["ticker"], filters["section"]), None)

    def insert_records(self, table, rows):
        for r in rows:
            self.rows[(r["ticker"], r["section"])] = r


@pytest.fixture
def env(monkeypatch):
    """equity_brief with a dict for a database and a counting fake DeepSeek."""
    db = FakeDB()
    dm = types.ModuleType("data_manager")
    dm.db = db
    dm.get_all_stock_basic = lambda: []
    calls: list[dict] = []
    reply: dict = {"value": GOOD, "fail": []}

    def call_json(system, user, **kw):
        calls.append({"system": system, "user": user, **kw})
        if reply["fail"]:
            raise reply["fail"].pop(0)
        asked = asked_for(system)
        if len(asked) == 1:
            # Like the real model: one section comes back in its own shape.
            return reply["value"].get(asked[0])
        return {s: reply["value"].get(s) for s in asked}

    ai = types.ModuleType("ai_client")
    ai.call_json = call_json
    monkeypatch.setitem(sys.modules, "data_manager", dm)
    monkeypatch.setitem(sys.modules, "ai_client", ai)
    monkeypatch.delitem(sys.modules, "equity_brief", raising=False)
    import equity_brief as eb
    monkeypatch.setattr(eb, "ensure_equity_brief_cache_table", lambda: None)
    return types.SimpleNamespace(eb=eb, db=db, calls=calls, reply=reply)


def asked_for(system: str) -> list[str]:
    """Which sections a prompt asks for, whether combined or single."""
    combined = [s for s in GOOD if f"## {s}\n" in system]
    if combined:
        return combined
    marks = {"overview": "In 3-4 sentences", "pestel": "PESTEL",
             "porters": "Porter's Five Forces"}
    return [s for s, m in marks.items() if m in system]


def _seed(db, ticker, section, payload, lang="zh"):
    """A cached section; Chinese, the default, unless `lang` says otherwise."""
    key = section if lang == "en" else f"{section}@{lang}"
    db.insert_records("t", [{"ticker": ticker, "section": key,
                             "payload": json.dumps(payload),
                             "generated_at": "2026-01-01"}])


# ── one call ─────────────────────────────────────────────────────────────────
def test_a_first_visit_is_one_call_for_all_three(env):
    """The Streamlit page reads overview, then PESTEL, then Porter's."""
    eb = env.eb
    eb.get_company_overview("603296", "华勤技术", "消费电子")
    eb.get_pestel("603296", "华勤技术", "消费电子")
    eb.get_porters("603296", "华勤技术", "消费电子")
    assert len(env.calls) == 1


def test_each_section_still_gets_its_own_row(env):
    env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert set(env.db.rows) == {("603296", f"{s}@zh") for s in ("overview", "pestel", "porters")}
    assert json.loads(env.db.rows[("603296", "porters@zh")]["payload"]) == GOOD["porters"]


def test_only_missing_siblings_are_asked_for(env):
    _seed(env.db, "603296", "pestel", GOOD["pestel"])
    env.eb.get_company_overview("603296", "华勤技术", "消费电子")
    assert asked_for(env.calls[0]["system"]) == ["overview", "porters"]


def test_a_cached_section_costs_nothing(env):
    _seed(env.db, "603296", "overview", GOOD["overview"])
    env.eb.get_company_overview("603296", "华勤技术", "消费电子")
    assert env.calls == []


# ── redoing one ──────────────────────────────────────────────────────────────
def test_regenerating_one_section_redoes_only_that_one(env):
    for s in GOOD:
        _seed(env.db, "603296", s, {"old": True} if s != "porters" else GOOD[s])
    env.eb.get_pestel("603296", "华勤技术", "消费电子", force_refresh=True)
    assert asked_for(env.calls[0]["system"]) == ["pestel"]
    assert json.loads(env.db.rows[("603296", "overview@zh")]["payload"]) == {"old": True}


def test_the_budget_scales_with_what_is_asked(env):
    """Three sections share one reasoning pass but need room for three outputs."""
    env.eb.generate_profile("603296", "华勤技术", "消费电子", ["pestel"])
    env.eb.generate_profile("603296", "华勤技术", "消费电子")
    one, three = env.calls
    assert three["max_tokens"] > one["max_tokens"]
    assert three["timeout"] > one["timeout"]


def test_the_combined_call_uses_json_mode(env):
    """One stray quote must not cost three sections — seen live on 中兴通讯."""
    env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert env.calls[0].get("json_mode") is True


def test_one_section_is_asked_for_unwrapped(env):
    """
    {"pestel": {...}} adds a level of nesting, and a live PESTEL-only regen
    for 中兴通讯 dropped a closing bracket in exactly that spot. One section
    is asked for in its own shape, as the original prompt did, and wrapped
    here instead.
    """
    out = env.eb.generate_profile("603296", "华勤技术", "消费电子", ["pestel"])
    assert '"pestel":' not in env.calls[0]["system"]
    assert out["written"]["pestel"]["payload"] == GOOD["pestel"]


def test_broken_json_is_asked_once_more(env):
    env.reply["fail"] = [RuntimeError("AI returned invalid JSON (Expecting ',')")]
    out = env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert len(env.calls) == 2
    assert set(out["written"]) == {"overview", "pestel", "porters"}


def test_broken_json_twice_gives_up(env):
    env.reply["fail"] = [RuntimeError("AI returned invalid JSON")] * 2
    with pytest.raises(RuntimeError):
        env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert len(env.calls) == 2


def test_a_timeout_is_not_retried(env):
    """A second 90-second wait on top of the first is worse than an error."""
    env.reply["fail"] = [RuntimeError("DeepSeek API timed out after 150 s.")]
    with pytest.raises(RuntimeError):
        env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert len(env.calls) == 1


# ── a bad section ────────────────────────────────────────────────────────────
def test_a_mangled_section_is_not_saved_and_the_rest_are(env):
    env.reply["value"] = {**GOOD, "porters": {"rivalry": {"score": 3}}}  # 4 forces missing
    out = env.eb.generate_profile("603296", "华勤技术", "消费电子")
    assert out["failed"] == ["porters"]
    assert set(out["written"]) == {"overview", "pestel"}
    assert ("603296", "porters@zh") not in env.db.rows


def test_asking_for_a_mangled_section_raises(env):
    env.reply["value"] = {**GOOD, "overview": "not a dict"}
    with pytest.raises(RuntimeError):
        env.eb.get_company_overview("603296", "华勤技术", "消费电子")


# ── the API route's section ─────────────────────────────────────────────────
@pytest.fixture
def ea(env):
    from api import equity_api
    return equity_api


def test_profile_generates_only_what_is_missing(env, ea):
    _seed(env.db, "603296", "overview", GOOD["overview"])
    out = ea.generate("603296", "华勤技术", "消费电子", "profile")
    assert sorted(out["written"]) == ["pestel", "porters"]
    assert asked_for(env.calls[0]["system"]) == ["pestel", "porters"]


def test_profile_with_nothing_missing_makes_no_call(env, ea):
    for s in GOOD:
        _seed(env.db, "603296", s, GOOD[s])
    assert ea.generate("603296", "华勤技术", "消费电子", "profile")["written"] == []
    assert env.calls == []


def test_forced_profile_redoes_all_three_in_one_call(env, ea):
    for s in GOOD:
        _seed(env.db, "603296", s, {"old": True})
    out = ea.generate("603296", "华勤技术", "消费电子", "profile", force=True)
    assert len(env.calls) == 1
    assert sorted(out["written"]) == ["overview", "pestel", "porters"]


def test_peers_are_not_part_of_the_profile(env, ea):
    """Competitors need the product list and their own prompt — still separate."""
    ea.generate("603296", "华勤技术", "消费电子", "profile")
    assert "competitors" not in env.calls[0]["system"]


# ── language ─────────────────────────────────────────────────────────────────
def test_the_prompt_asks_for_the_language_and_keeps_keys_english(env):
    env.eb.generate_profile("603296", "华勤技术", "消费电子")
    env.eb.generate_profile("603296", "华勤技术", "消费电子", lang="en")
    zh, en = (c["system"] for c in env.calls)
    assert "Simplified Chinese" in zh and "JSON keys stay" in zh
    assert "in English" in en and "Simplified Chinese" not in en
    # The old prompt asked for the overview "in English" whatever the setting.
    assert "plain English" not in zh


def test_english_is_stored_under_the_bare_name_other_languages_suffixed(env):
    env.eb.generate_profile("603296", "华勤技术", "消费电子", ["pestel"], lang="en")
    env.eb.generate_profile("603296", "华勤技术", "消费电子", ["pestel"], lang="zh")
    assert {k for _, k in env.db.rows} == {"pestel", "pestel@zh"}


def test_an_older_english_section_stands_in_marked_until_regenerated(env, ea):
    _seed(env.db, "603296", "pestel", GOOD["pestel"], lang="en")
    got = ea.cached_ai("603296", "zh")
    assert got["pestel"]["payload"] == GOOD["pestel"] and got["pestel"]["lang"] == "en"
    half = ea.ai_half("603296", "华勤技术", "zh")
    assert half["other_lang"] == {"pestel": "en"} and half["lang"] == "zh"
    # ...and asking for the missing ones in Chinese redoes it too.
    ea.generate("603296", "华勤技术", "消费电子", "profile", lang="zh")
    assert asked_for(env.calls[0]["system"]) == ["overview", "pestel", "porters"]


def test_english_reads_its_own_rows_without_a_marker(env, ea):
    _seed(env.db, "603296", "pestel", GOOD["pestel"], lang="en")
    assert "lang" not in ea.cached_ai("603296", "en")["pestel"]


def test_unknown_languages_fall_back_to_the_default():
    import i18n
    assert i18n.norm("zh-CN") == "zh" and i18n.norm("en-US") == "en"
    assert i18n.norm(None) == i18n.DEFAULT == "zh" and i18n.norm("fr") == "zh"


def test_an_older_row_already_in_chinese_counts_as_chinese(env, ea):
    """The old prompt asked for English; the model often wrote Chinese anyway."""
    zh = {"rivalry": {"score": 3, "summary": "电力二次设备行业竞争格局相对稳定，龙头地位稳固。"}}
    _seed(env.db, "603296", "porters", zh, lang="en")       # the bare, pre-language key
    assert "lang" not in ea.cached_ai("603296", "zh")["porters"]
    assert ea.cached_ai("603296", "en")["porters"]["lang"] == "zh"


def test_detect_reads_chinese_with_latin_acronyms_as_chinese():
    import i18n
    assert i18n.detect({"a": ["受益于 AI 服务器和 PCB 需求增长，订单饱满。"]}) == "zh"
    assert i18n.detect({"a": "Rivalry is intense among ODMs such as 闻泰科技."}) == "en"
