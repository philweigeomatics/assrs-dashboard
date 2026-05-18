"""
Macro Sector Theme — DeepSeek API caller.

Decoupled from the UI so the same generator can be reused from any page.
All DB interactions go through data_manager (not this module).
"""

import json

import streamlit as st

import ai_client

# Hardened version of the user's prompt — same intent, stricter output rules.
_SYSTEM_PROMPT = """\
You are an elite quantitative analyst mapping Chinese A-Share supply chains.
The user will provide a rough industry theme (e.g., "data centre", "ev").

1. Standardize the theme name into a formal Bilingual string (English / 中文).
2. Map the industry into exactly 3 to 5 chronological layers, from upstream raw \
materials/components to downstream operators/consumers.
3. For each layer, list 3 to 6 core tangible products or sub-sectors involved. \
Every item MUST be bilingual.

CRITICAL OUTPUT RULES (enforced programmatically):
- Return ONLY raw JSON. No conversational text. No markdown code fences \
(do not use ```json).
- Start your response directly with { and end with }.
- "name" MUST follow the format 'English / 中文' (English first, then ' / ', then Chinese).
- Each entry in every "items" array MUST follow the same 'English / 中文' format.
- Each "layer_name" MUST also follow the same 'English / 中文' format.
- "layers" must contain between 3 and 5 layer objects, ordered upstream → \
downstream by 1-based "layer_index".
- Items must be specific, tangible products / sub-sectors — not company names.

Schema:
{
  "name": "String (e.g., 'Data Center / 数据中心')",
  "layers": [
    {
      "layer_index": 1,
      "layer_name": "String (e.g., 'Upstream Components / 上游核心部件')",
      "items": ["Item 1 / 项目1", "Item 2 / 项目2"]
    }
  ]
}
"""


_MATCH_PROMPT = """\
Match an industry sector label to the closest entry in a list of stored themes.

Input JSON: {"sector": "...", "themes": [{"id": 1, "formal_name": "...", "raw_input": "..."}, ...]}

Rules:
- Return ONLY raw JSON: {"matched_id": <integer> or null}
- matched_id is the id of the best-matching theme, or null if none fits well.
- A match is good if the theme clearly covers the same industry, even with
  different wording (e.g. "Industrial Robot" matches "Robotics / 机器人").
- Do NOT force a match when nothing is close enough.
"""


def generate_sector_theme(raw_input: str) -> dict:
    """
    Expand a rough industry theme into chronological supply-chain layers.

    Returns the parsed dict matching the schema above.
    Raises RuntimeError on any failure (network, JSON, validation).
    """
    if not raw_input or not raw_input.strip():
        raise RuntimeError("Empty theme — please enter something to expand.")

    data = ai_client.call_json(
        _SYSTEM_PROMPT,
        f"Industry theme: {raw_input.strip()}",
        max_tokens=3000,
        temperature=0.2,
    )

    # ── Light validation so the UI never receives malformed data ──
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object, got {type(data).__name__}.")
    if "name" not in data or "layers" not in data:
        raise RuntimeError(f"Missing 'name' or 'layers'. Got keys: {list(data.keys())}")
    if not isinstance(data["layers"], list) or not data["layers"]:
        raise RuntimeError("'layers' must be a non-empty list.")

    for i, layer in enumerate(data["layers"]):
        if not isinstance(layer, dict):
            raise RuntimeError(f"Layer #{i} is not an object.")
        if "items" not in layer or not isinstance(layer["items"], list):
            raise RuntimeError(f"Layer #{i} missing 'items' list.")
        if "layer_index" not in layer:
            layer["layer_index"] = i + 1
        if "layer_name" not in layer:
            layer["layer_name"] = f"Layer {i + 1}"

    return data


def match_sector_theme(sector_name: str, all_themes: list) -> dict | None:
    """
    Find the best-matching stored theme for `sector_name`.

    Strategy:
      1. Simple case-insensitive substring match against formal_name / raw_input.
         If exactly one candidate, return it immediately (no API call).
      2. Otherwise ask DeepSeek to pick the best match (or return null).

    Returns a theme dict {id, formal_name, raw_input} or None.
    """
    if not all_themes or not sector_name.strip():
        return None

    low = sector_name.strip().lower()

    candidates = [
        t for t in all_themes
        if low in t["formal_name"].lower()
        or low in t["raw_input"].lower()
        or t["formal_name"].lower() in low
        or t["raw_input"].lower() in low
    ]

    if len(candidates) == 1:
        return candidates[0]

    # Multiple or zero candidates — ask DeepSeek
    pool = candidates if candidates else all_themes
    try:
        result = ai_client.call_json(
            _MATCH_PROMPT,
            json.dumps({
                "sector": sector_name,
                "themes": [
                    {"id": t["id"], "formal_name": t["formal_name"],
                     "raw_input": t["raw_input"]}
                    for t in pool
                ],
            }),
            max_tokens=4000,
            temperature=0.0,
        )
        matched_id = result.get("matched_id")
        if matched_id is not None:
            for t in pool:
                if t["id"] == int(matched_id):
                    return t
    except Exception:
        pass

    return candidates[0] if candidates else None


# ── Batch sector → theme matcher ──────────────────────────────────────────────

_BATCH_MATCH_PROMPT = """\
Match multiple industry sector labels to the closest entries in a list of stored themes.

Input JSON:
{"sectors": ["sector1", "sector2", ...],
 "themes": [{"id": 1, "formal_name": "...", "raw_input": "..."}, ...]}

Rules:
- Return ONLY raw JSON (start { end }). No markdown.
- Return {"matches": {"sector1": <integer or null>, ...}} — one key per sector.
- Each value is the id of the best-matching theme, or null if none fits well.
- A match is good if the theme clearly covers the same industry (different wording is fine).
- Do NOT force a match when nothing is close enough.
"""


def match_sectors_to_themes_batch(sectors: list, all_themes: list) -> dict:
    """
    Batch-match a list of sector names to stored themes with at most ONE AI call.

    Strategy:
      1. Substring match (free) per sector — unique candidate → resolved immediately.
      2. Collect all ambiguous (zero or multi-candidate) sectors.
      3. ONE AI call resolves all ambiguous sectors together.

    Returns {sector_name: theme_dict_or_None} for every sector in `sectors`.
    """
    if not all_themes or not sectors:
        return {s: None for s in sectors}

    resolved: dict = {}
    ambiguous: list = []  # [(sector_name, candidate_list), ...]

    for sector in sectors:
        low = sector.strip().lower()
        candidates = [
            t for t in all_themes
            if low in t["formal_name"].lower()
            or low in t["raw_input"].lower()
            or t["formal_name"].lower() in low
            or t["raw_input"].lower() in low
        ]
        if len(candidates) == 1:
            resolved[sector] = candidates[0]
        else:
            # 0 candidates → send all_themes to AI; multiple → send just candidates
            ambiguous.append((sector, candidates if candidates else []))

    if not ambiguous:
        return resolved

    # ONE AI call for all ambiguous sectors
    # Build a union pool of relevant themes
    seen_ids: set = set()
    pool_union: list = []
    has_zero_match = any(len(c) == 0 for _, c in ambiguous)

    for _, candidates in ambiguous:
        for t in candidates:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                pool_union.append(t)

    # Zero-match sectors need the full theme list sent to the AI
    if has_zero_match:
        for t in all_themes:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                pool_union.append(t)

    try:
        result = ai_client.call_json(
            _BATCH_MATCH_PROMPT,
            json.dumps({
                "sectors": [s for s, _ in ambiguous],
                "themes": [
                    {"id": t["id"], "formal_name": t["formal_name"],
                     "raw_input": t["raw_input"]}
                    for t in pool_union
                ],
            }),
            max_tokens=4000,
            temperature=0.0,
        )
        matches = result.get("matches", {})
        theme_by_id = {t["id"]: t for t in pool_union}
        for sector, candidates in ambiguous:
            mid = matches.get(sector)
            if mid is not None:
                try:
                    resolved[sector] = theme_by_id.get(int(mid))
                except (ValueError, TypeError):
                    resolved[sector] = None
            else:
                resolved[sector] = None
    except Exception:
        for sector, candidates in ambiguous:
            resolved[sector] = candidates[0] if candidates else None

    return resolved


# ── Batch theme classifier ────────────────────────────────────────────────────

_CLASSIFY_BATCH_PROMPT = """\
You are an expert Chinese A-share supply chain analyst.

A company's details and MULTIPLE sector supply chains are provided.
The company has ALREADY been verified to operate within EACH of these sectors —
your sole task is to identify the SPECIFIC LAYER inside each theme where the
company primarily operates.

Focus on the company's main revenue-generating activities, not aspirational ones.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- Return {"results": [...]} — MUST contain exactly one entry per theme_id provided.
  Never skip a theme. Never truncate the list.
- layer_index: integer matching one of the provided layer_index values.
  Strongly prefer a layer over returning null — since the company is known to
  operate in this sector, at least ONE layer should fit. Return null only as a
  last resort when the company's products are entirely unrelated to every layer's
  items, AND you have considered all of them.
- matched_items: 1–3 items FROM that layer that best match the company's products.
  If the products don't exactly match named items, pick the items most analogous
  to what the company produces. Empty list only if layer_index is null.

Schema:
{
  "results": [
    {
      "theme_id": <integer>,
      "layer_index": <integer or null>,
      "matched_items": ["item / 项目", ...]
    }
  ]
}
"""


def classify_ticker_across_themes(
    ticker: str,
    company_name: str,
    products: list,
    themes_list: list,
) -> list:
    """
    ONE AI call to classify the company across ALL provided themes simultaneously.

    `themes_list` — list of full theme dicts (each with "id", "formal_name", "layers").

    Returns [{"theme_id": int, "layer_index": int|None, "matched_items": [...]}, ...].
    Raises RuntimeError on failure.
    """
    if not themes_list:
        return []

    themes_blocks = []
    for theme in themes_list:
        layers_text = "\n".join(
            f"    Layer {l['layer_index']} — {l.get('layer_name', '')}: "
            + ", ".join(l.get("items", []))
            for l in sorted(
                theme.get("layers", []), key=lambda x: x.get("layer_index", 0)
            )
        )
        themes_blocks.append(
            f"Theme id={theme['id']}: {theme.get('formal_name') or theme.get('name', '')}\n"
            f"{layers_text}"
        )

    user_msg = (
        f"Company: {company_name} ({ticker})\n"
        f"Products/Services: {', '.join(products) if products else 'Unknown'}\n\n"
        + "\n\n".join(themes_blocks)
    )

    result = ai_client.call_json(
        _CLASSIFY_BATCH_PROMPT, user_msg,
        max_tokens=5000,
        temperature=0.1,
    )

    return [
        {
            "theme_id":      entry.get("theme_id"),
            "layer_index":   entry.get("layer_index"),
            "matched_items": entry.get("matched_items", []),
        }
        for entry in result.get("results", [])
    ]


# ── Layer classifier ──────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
You are an expert Chinese A-share supply chain analyst.

Given a company's known products/services and a sector supply chain with
numbered layers (upstream → downstream), determine which single layer best
describes where this company PRIMARILY operates.

Focus on the company's main revenue-generating activities, not aspirational
or minor activities.

OUTPUT RULES:
- Return ONLY raw JSON (start { end }). No markdown.
- layer_index must be an integer matching one of the provided layer_index
  values, or null if the company does not fit any layer.
- matched_items is a list of 1–3 items FROM that layer that best match the
  company's products. Empty list if layer_index is null.

Schema:
{
  "layer_index": <integer or null>,
  "matched_items": ["item / 项目", ...]
}
"""


def classify_ticker_in_theme(
    ticker: str,
    company_name: str,
    products: list,
    theme: dict,
) -> dict:
    """
    Ask DeepSeek which layer of `theme` the company primarily operates in.

    Returns {"layer_index": int|None, "matched_items": list[str]}.
    Raises RuntimeError on failure.
    """
    layers_text = "\n".join(
        f"  Layer {l['layer_index']} — {l.get('layer_name', '')}: "
        + ", ".join(l.get("items", []))
        for l in sorted(theme.get("layers", []), key=lambda x: x.get("layer_index", 0))
    )
    user_msg = (
        f"Company: {company_name} ({ticker})\n"
        f"Products/Services: {', '.join(products) if products else 'Unknown'}\n\n"
        f"Theme: {theme.get('name') or theme.get('formal_name', '')}\n"
        f"Layers:\n{layers_text}"
    )

    result = ai_client.call_json(
        _CLASSIFY_PROMPT, user_msg,
        max_tokens=4000,
        temperature=0.1,
    )

    return {
        "layer_index":   result.get("layer_index"),
        "matched_items": result.get("matched_items", []),
    }


def render_sector_layers(theme: dict, key_prefix: str = "sl") -> None:
    """
    Render a sector theme as horizontal upstream→downstream columns.
    Identical layout to sector_explorer.py; key_prefix namespaces session state
    so multiple callers on the same page don't collide.
    """
    layers = theme.get("layers") or []
    if not layers:
        st.warning("This theme has no layers stored.")
        return

    layers = sorted(layers, key=lambda l: l.get("layer_index", 0))

    st.subheader(f"🔗 {theme['formal_name']}")
    st.caption("Upstream → Downstream  ·  上游 → 下游")

    cols = st.columns(len(layers), gap="small")
    for layer_pos, (col, layer) in enumerate(zip(cols, layers)):
        with col:
            idx        = layer.get("layer_index", layer_pos + 1)
            layer_name = layer.get("layer_name", f"Layer {idx}")
            items      = layer.get("items", [])

            st.markdown(
                f"<div style='text-align:center;color:#9ca3af;font-size:11px;"
                f"font-weight:600;'>LAYER {idx}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='text-align:center;color:#e5e7eb;font-size:13px;"
                f"font-weight:700;margin-bottom:6px;line-height:1.2;'>"
                f"{layer_name}</div>",
                unsafe_allow_html=True,
            )
            st.divider()

            sel_key = f"{key_prefix}_selected"
            for item_idx, item in enumerate(items):
                highlighted = st.session_state.get(sel_key) == item
                btn_key = f"{key_prefix}_{idx}_{item_idx}"
                if st.button(
                    item,
                    key=btn_key,
                    use_container_width=True,
                    type=("primary" if highlighted else "secondary"),
                ):
                    st.session_state[sel_key] = item
                    st.rerun()

    selected = st.session_state.get(f"{key_prefix}_selected")
    if selected:
        st.markdown("---")
        c1, c2 = st.columns([5, 1])
        c1.info(f"🔎 **Selected:** {selected}")
        if c2.button("✖ Clear", key=f"{key_prefix}_clear", use_container_width=True):
            st.session_state.pop(f"{key_prefix}_selected", None)
            st.rerun()
