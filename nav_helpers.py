"""Cross-page anchor-style navigation helpers.

Streamlit's st.button + st.switch_page can't be middle-clicked or right-
clicked → "Open in new tab" because they're buttons, not links. These
helpers render visually-identical button-styled <a> elements so users get
native browser new-tab behaviour while left-click still navigates in place.

Each target page that receives params via these helpers should read
st.query_params at the top of its script and seed its own session_state,
then call st.query_params.clear() so the URL stays clean after restore.
"""
from __future__ import annotations

from typing import Iterable, Mapping
from urllib.parse import urlencode

import streamlit as st


def _button_anchor_html(
    *,
    href: str,
    label: str,
    icon: str,
    style: str,
    disabled: bool,
    full_width: bool,
    title: str,
) -> str:
    """Build the HTML for a button-styled anchor element."""
    if style == "primary":
        bg, color, border = "rgb(255, 75, 75)", "#ffffff", "rgb(255, 75, 75)"
        hover_bg, hover_border = "rgb(255, 43, 43)", "rgb(255, 43, 43)"
    else:
        bg, color, border = "transparent", "rgb(38, 39, 48)", "rgba(49, 51, 63, 0.2)"
        hover_bg, hover_border = "rgba(255, 75, 75, 0.05)", "rgb(255, 75, 75)"

    width = "100%" if full_width else "auto"
    icon_html = f"{icon} " if icon else ""
    title_attr = f' title="{title}"' if title else ""

    if disabled:
        # Render a non-clickable span styled like a disabled Streamlit button
        return (
            f'<span style="'
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"width:{width};padding:0.375rem 0.75rem;box-sizing:border-box;"
            f"background:rgba(240,242,246,0.5);color:rgba(49,51,63,0.4);"
            f"border:1px solid rgba(49,51,63,0.1);border-radius:0.5rem;"
            f"font-weight:400;line-height:1.6;cursor:not-allowed;"
            f"font-size:0.875rem;font-family:'Source Sans Pro',sans-serif;"
            f'"{title_attr}>{icon_html}{label}</span>'
        )

    return (
        f'<a href="{href}" target="_self" style="'
        f"display:inline-flex;align-items:center;justify-content:center;"
        f"width:{width};padding:0.375rem 0.75rem;box-sizing:border-box;"
        f"background:{bg};color:{color};border:1px solid {border};"
        f"border-radius:0.5rem;text-decoration:none;font-weight:400;"
        f"line-height:1.6;cursor:pointer;"
        f"font-size:0.875rem;font-family:'Source Sans Pro',sans-serif;"
        f"transition:background-color 0.15s,border-color 0.15s;"
        f'" onmouseover="this.style.background=\'{hover_bg}\';this.style.borderColor=\'{hover_border}\';"'
        f' onmouseout="this.style.background=\'{bg}\';this.style.borderColor=\'{border}\';"'
        f'{title_attr}>{icon_html}{label}</a>'
    )


def page_link_button(
    url_path: str,
    label: str,
    *,
    params: Mapping[str, str | int | None] | None = None,
    icon: str = "",
    style: str = "secondary",
    disabled: bool = False,
    use_container_width: bool = False,
    help: str = "",
) -> None:
    """
    Render a button-styled anchor link to another Streamlit page.

    Left-click navigates in the current tab (matches the old st.button →
    st.switch_page behaviour). Middle-click or right-click → "Open in new
    tab" opens the destination in a new tab — the target page reads
    query params from the URL and seeds its session_state.

    Args:
        url_path: The destination page's url_path as set in streamlit_app.py
                  (e.g. "single-stock-analysis", "pair-trader").
        label: Visible text.
        params: Query params for the target page (e.g. {"ticker": "600036"}).
                None/empty values are dropped.
        icon: Optional emoji/icon glyph prepended to the label.
        style: "primary" (filled red) or "secondary" (bordered).
        disabled: Render a non-clickable disabled-style chip.
        use_container_width: Stretch to the parent column's width.
        help: Tooltip text (rendered as the anchor's title attribute).
    """
    href = f"/{url_path}"
    if params:
        clean = {k: str(v) for k, v in params.items() if v not in (None, "")}
        if clean:
            href = f"{href}?{urlencode(clean)}"

    st.html(_button_anchor_html(
        href=href,
        label=label,
        icon=icon,
        style=style,
        disabled=disabled,
        full_width=use_container_width,
        title=help,
    ))


def consume_query_params(*keys: str) -> dict[str, str]:
    """
    Read selected keys from st.query_params, clear them from the URL, and
    return them as a dict. Skips empty values.

    Use at the top of a target page that accepts cross-page navigation
    params, then seed session_state from the returned dict.
    """
    out: dict[str, str] = {}
    qp = st.query_params
    for k in keys:
        v = qp.get(k, "")
        if isinstance(v, list):
            v = v[0] if v else ""
        v = (v or "").strip()
        if v:
            out[k] = v
    if out:
        # Clear ALL query params so the URL stays clean after deep-link restore
        for k in list(qp.keys()):
            del qp[k]
    return out


def join_tickers(tickers: Iterable[str]) -> str:
    """Pack a list of tickers into a comma-separated string for URL params."""
    return ",".join(t.strip() for t in tickers if t and t.strip())


def split_tickers(packed: str) -> list[str]:
    """Unpack the comma-separated ticker string written by join_tickers."""
    if not packed:
        return []
    return [t.strip() for t in packed.split(",") if t.strip()]
