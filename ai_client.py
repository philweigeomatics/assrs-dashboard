"""
ai_client.py — Centralized DeepSeek API client for ASSRS.

Single source of truth for:
  - Model name and endpoint
  - API key retrieval
  - HTTP call + error handling
  - Reasoning-model quirks (empty content when max_tokens exhausted by thinking trace)
  - Markdown fence stripping
  - JSON parsing

Usage:
    import ai_client

    # Returns a parsed dict — raises RuntimeError on any failure
    data = ai_client.call_json(system_prompt, user_msg, max_tokens=2000)

    # Returns raw text string (for non-JSON responses)
    text = ai_client.call_text(system_prompt, user_msg, max_tokens=2000)

Changing the model or endpoint in this one file affects the entire app.
"""

import json

import requests

# ── Config ────────────────────────────────────────────────────────────────────

MODEL    = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"


def _api_key() -> str:
    from api_config import _get_secret
    return _get_secret("DEEPSEEK_API_KEY")


# ── Core HTTP layer ───────────────────────────────────────────────────────────

def _raw_call(
    system_prompt: str,
    user_msg: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    """
    Make one DeepSeek chat completion request and return the raw content string.

    Handles:
      - API key retrieval errors
      - HTTP transport errors and timeouts
      - API-level error objects (wrong model name, quota, etc.)
      - Unexpected response shapes
      - Reasoning-model token exhaustion (finish_reason='length' + reasoning_tokens > 0)
      - Empty content

    Raises RuntimeError with a human-readable message on any failure.
    """
    try:
        api_key = _api_key()
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    try:
        resp = requests.post(
            ENDPOINT, json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.Timeout:
        raise RuntimeError(f"DeepSeek API timed out after {timeout} s.")
    except requests.RequestException as exc:
        raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

    try:
        resp_json = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"DeepSeek response was not JSON. "
            f"Status {resp.status_code}. Body: {resp.text[:400]}"
        ) from exc

    # Surface API-level errors (wrong model name, quota exceeded, etc.)
    if "error" in resp_json:
        err = resp_json["error"]
        raise RuntimeError(
            f"DeepSeek API error — {err.get('type', 'unknown')}: "
            f"{err.get('message', err)}"
        )

    try:
        choice  = resp_json["choices"][0]
        message = choice["message"]
        raw     = message.get("content", "").strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected DeepSeek response shape. Full response: {resp_json}"
        ) from exc

    if not raw:
        finish_reason    = choice.get("finish_reason", "unknown")
        reasoning_tokens = (
            resp_json.get("usage", {})
                     .get("completion_tokens_details", {})
                     .get("reasoning_tokens", 0)
        )
        # Reasoning model (e.g. deepseek-v4-flash) exhausted max_tokens on its
        # thinking trace before producing any output.
        if finish_reason == "length" and reasoning_tokens > 0:
            raise RuntimeError(
                f"Model exhausted {max_tokens} tokens on reasoning trace "
                f"({reasoning_tokens} reasoning tokens) before writing output. "
                f"Increase max_tokens and try again."
            )
        raise RuntimeError(
            f"DeepSeek returned empty content "
            f"(finish_reason={finish_reason!r}). Full response: {resp_json}"
        )

    return raw


# ── Public API ────────────────────────────────────────────────────────────────

def call_text(
    system_prompt: str,
    user_msg: str,
    *,
    max_tokens: int   = 4000,
    temperature: float = 0.2,
    timeout: int       = 60,
) -> str:
    """
    Call DeepSeek and return the raw text content string.

    Use this when the response is free-form prose rather than JSON.
    Raises RuntimeError on any failure.
    """
    return _raw_call(
        system_prompt, user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


def call_json(
    system_prompt: str,
    user_msg: str,
    *,
    max_tokens: int    = 4000,
    temperature: float = 0.2,
    timeout: int       = 60,
) -> dict:
    """
    Call DeepSeek and return the response parsed as a JSON dict.

    Automatically strips accidental markdown fences (```json ... ```) before
    parsing. Raises RuntimeError on transport, API, or JSON parse failure.
    """
    raw = _raw_call(
        system_prompt, user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AI returned invalid JSON ({exc}).\n"
            f"Full raw response:\n{raw[:600]}"
        ) from exc
