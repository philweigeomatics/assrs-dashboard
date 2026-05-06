"""
Macro Sector Theme — DeepSeek API caller.

Decoupled from the UI so the same generator can be reused from any page.
All DB interactions go through data_manager (not this module).
"""

import json

import requests
import streamlit as st

# ── DeepSeek config ───────────────────────────────────────────────────────────

_ENDPOINT = "https://api.deepseek.com/chat/completions"
_MODEL    = "deepseek-chat"

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


def _api_key() -> str:
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


def generate_sector_theme(raw_input: str) -> dict:
    """
    Expand a rough industry theme into chronological supply-chain layers.

    Returns the parsed dict matching the schema above.
    Raises RuntimeError on any failure (network, JSON, validation).
    """
    if not raw_input or not raw_input.strip():
        raise RuntimeError("Empty theme — please enter something to expand.")

    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Industry theme: {raw_input.strip()}"},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(
            _ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=45,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError("DeepSeek API timed out after 45 s.")
    except requests.RequestException as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences if the model adds them anyway
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"DeepSeek returned invalid JSON ({exc}). Preview: {raw[:200]}"
        ) from exc

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
    try:
        api_key = _api_key()
    except ValueError:
        return candidates[0] if candidates else None

    pool = candidates if candidates else all_themes
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _MATCH_PROMPT},
            {"role": "user", "content": json.dumps({
                "sector": sector_name,
                "themes": [
                    {"id": t["id"], "formal_name": t["formal_name"],
                     "raw_input": t["raw_input"]}
                    for t in pool
                ],
            })},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    try:
        resp = requests.post(
            _ENDPOINT, json=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw
        result = json.loads(raw)
        matched_id = result.get("matched_id")
        if matched_id is not None:
            for t in pool:
                if t["id"] == int(matched_id):
                    return t
    except Exception:
        pass

    return candidates[0] if candidates else None


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
