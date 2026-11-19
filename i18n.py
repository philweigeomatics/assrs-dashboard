"""
i18n.py — the language the app writes in, in one place.

The web client sends its language on every request (`X-Lang`, see
web/src/lib/lang.ts); anything that produces prose — the AI sections today,
the interface once it is translated — takes it from here. Simplified Chinese
is the default, because that is the app as it stands.

Adding a language is adding it to LANGS and AI_RULE; nothing else should need
to know which languages exist.
"""

from __future__ import annotations

#: Supported languages, default first.
LANGS: tuple[str, ...] = ("zh", "en")
DEFAULT = "zh"

#: Appended to every AI prompt that writes prose. Keys stay in English so the
#: schema, the cache and the client agree whatever the language; only the
#: values a person reads are translated.
AI_RULE: dict[str, str] = {
    "zh": ("LANGUAGE: Write every value a reader sees in Simplified Chinese "
           "(简体中文). JSON keys stay exactly as in the schema, in English. "
           "Company, product and brand names keep the form they are commonly "
           "known by in China; numbers keep their units (亿元, %)."),
    "en": ("LANGUAGE: Write every value a reader sees in English. JSON keys "
           "stay exactly as in the schema."),
}


def norm(value: str | None) -> str:
    """A supported language code from whatever the client sent; DEFAULT otherwise."""
    v = (value or "").strip().lower().replace("_", "-")
    if v.startswith("zh"):
        return "zh"
    if v.startswith("en"):
        return "en"
    return DEFAULT


def ai_rule(lang: str | None) -> str:
    return AI_RULE[norm(lang)]


def detect(value) -> str:
    """
    Which language a piece of stored prose is in, for rows written before
    the language was recorded. Chinese when at least a fifth of its letters
    are CJK: a Chinese sentence carries Latin acronyms and tickers, an
    English one hardly ever a Chinese character beyond a company name.
    """
    def strings(v):
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for x in v.values():
                yield from strings(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from strings(x)

    cjk = latin = 0
    for s in strings(value):
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                cjk += 1
            elif ch.isascii() and ch.isalpha():
                latin += 1
    if cjk + latin == 0:
        return DEFAULT
    return "zh" if cjk / (cjk + latin) >= 0.2 else "en"


__all__ = ["AI_RULE", "DEFAULT", "LANGS", "ai_rule", "detect", "norm"]
