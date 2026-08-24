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
    data = ai_client.call_json(system_prompt, user_msg, max_tokens=4000)

    # Returns raw text string (for non-JSON responses)
    text = ai_client.call_text(system_prompt, user_msg, max_tokens=4000)

Changing the model or endpoint in this one file affects the entire app.

Two DeepSeek-specific behaviors worth knowing before touching call sites:

  - Thinking/reasoning mode defaults to ON, and reasoning tokens are drawn
    from the SAME max_tokens budget as the output — not a separate pool.
    A knowledge-heavy prompt (e.g. "recall this company's supply chain from
    memory") can burn most of max_tokens on the reasoning trace before
    writing any JSON, which surfaces here as the "exhausted max_tokens on
    reasoning trace" error below. Pass reasoning_effort="low" for tasks
    that are lookup/formatting rather than multi-step reasoning — verified
    against DeepSeek's docs that "low" actually reduces effort on
    deepseek-v4-flash (unlike deepseek-v4-pro, where "low" is currently
    silently remapped to "high").

  - `temperature` (and top_p / presence_penalty / frequency_penalty) is a
    silent no-op whenever thinking mode is on — DeepSeek accepts the
    parameter without error for backward compatibility, but it has no
    effect. If a call site actually needs deterministic/low-randomness
    output, that requires reasoning_effort tuning or thinking disabled
    entirely, not a lower temperature.
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
    reasoning_effort: str | None = None,
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

    reasoning_effort: "low" | "high" | "xhigh" | "max", or None to leave the
    API default ("high") in place. Pass "low" for lookup/formatting tasks
    that don't need multi-step reasoning — it directly reduces how many of
    max_tokens the reasoning trace consumes, which is what actually causes
    the exhaustion error below (see the module docstring).

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
    if reasoning_effort is not None:
        payload["thinking"] = {"type": "enabled", "reasoning_effort": reasoning_effort}

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
        # Occasionally the model writes the FINISHED answer into the reasoning
        # channel and emits nothing on `content`, while still reporting
        # finish_reason='stop' — it believes it is done, it just used the wrong
        # pipe. Measured on deepseek-v4-flash at roughly 1 call in 6 for a
        # JSON-output prompt; the salvaged text was complete and correct.
        # Treat that as a delivery accident rather than a failure: raising here
        # discards a paid-for, complete answer. Only 'stop' qualifies — under
        # 'length' the trace is genuinely truncated mid-thought.
        reasoning = (message.get("reasoning_content") or "").strip()
        if finish_reason == "stop" and reasoning:
            return reasoning
        # Reasoning model (e.g. deepseek-v4-flash) exhausted max_tokens on its
        # thinking trace before producing any output.
        if finish_reason == "length" and reasoning_tokens > 0:
            raise RuntimeError(
                f"Model exhausted {max_tokens} tokens on reasoning trace "
                f"({reasoning_tokens} reasoning tokens) before writing output. "
                f"Increase max_tokens, or pass reasoning_effort='low' if this "
                f"call is lookup/formatting rather than multi-step reasoning."
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
    max_tokens: int   = 8000,
    temperature: float = 0.2,
    timeout: int       = 60,
    reasoning_effort: str | None = None,
) -> str:
    """
    Call DeepSeek and return the raw text content string.

    Use this when the response is free-form prose rather than JSON.
    Raises RuntimeError on any failure. See module docstring for what
    reasoning_effort does and why temperature has no effect under thinking mode.
    """
    return _raw_call(
        system_prompt, user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def call_json(
    system_prompt: str,
    user_msg: str,
    *,
    max_tokens: int    = 8000,
    temperature: float = 0.2,
    timeout: int       = 60,
    reasoning_effort: str | None = None,
) -> dict:
    """
    Call DeepSeek and return the response parsed as a JSON dict.

    Automatically strips accidental markdown fences (```json ... ```) before
    parsing. Raises RuntimeError on transport, API, or JSON parse failure.
    See module docstring for what reasoning_effort does and why temperature
    has no effect under thinking mode.
    """
    raw = _raw_call(
        system_prompt, user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )

    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) >= 2 else raw

    # Take the outermost {...}. Needed because _raw_call may have salvaged the
    # answer from the reasoning channel, where the JSON can sit behind a line
    # or two of thinking-out-loud rather than starting at character zero.
    if not raw.startswith("{"):
        i, j = raw.find("{"), raw.rfind("}")
        if i >= 0 and j > i:
            raw = raw[i:j + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"AI returned invalid JSON ({exc}).\n"
            f"Full raw response:\n{raw[:600]}"
        ) from exc
