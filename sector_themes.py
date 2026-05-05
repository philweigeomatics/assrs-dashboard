"""
Macro Sector Theme — DeepSeek API caller.

Decoupled from the UI so the same generator can be reused from any page.
All DB interactions go through data_manager (not this module).
"""

import json

import requests

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
