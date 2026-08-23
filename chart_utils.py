"""
chart_utils.py — shared Plotly helpers for the multi-panel indicator charts.

Extracted from the Technical Analysis page so the Blind Replay chart can use
the same treatment instead of carrying a second copy that drifts.
"""

from __future__ import annotations


def _axis_name(ref: str) -> str:
    """'y' -> 'yaxis', 'y3' -> 'yaxis3' (trace refs to layout keys)."""
    ref = ref or "y"
    return "yaxis" + ref[1:]


def _row_map(fig, n_rows: int) -> dict:
    """
    layout axis name -> subplot row.

    Built from fig.get_subplot() rather than by arithmetic on axis numbers,
    because the numbering depends on whether any row declares secondary_y —
    the Technical Analysis chart does (row 1) and the Blind Replay chart does
    not, so a hardcoded offset is right for exactly one of them.

    Secondary axes never appear in get_subplot's primary refs, so they are
    resolved afterwards through their `overlaying` target.
    """
    out = {}
    for r in range(1, n_rows + 1):
        try:
            sp = fig.get_subplot(r, 1)
        except Exception:
            continue
        ax = getattr(sp, "yaxis", None)
        name = getattr(ax, "plotly_name", None)
        if name:
            out[name] = r

    for key in fig.layout:
        if not key.startswith("yaxis") or key in out:
            continue
        over = getattr(fig.layout[key], "overlaying", None)
        if over:
            host = _axis_name(over)
            if host in out:
                out[key] = out[host]
    return out


def split_legends_by_panel(fig, n_rows: int, panel_titles=None,
                           font_size: int = 10, side: str = "right"):
    """
    Give every subplot its OWN legend, parked beside that subplot.

    One shared legend lists every series in a single column, so a toggle in the
    middle of it gives no clue which panel it drives. Plotly supports multiple
    legends (plotly.js 2.24+ / plotly.py 5.15+) via trace.legend='legend2' plus
    a matching layout.legend2.

    Each legend is anchored at the TOP of its panel and grows downward into the
    gap above the next one, so the room available is the panel's own height
    plus the vertical spacing beneath it. Font shrinks automatically when a
    panel has more entries than fit; the floor is 8pt, below which it would
    stop being readable, so a very crowded panel overflows visibly rather than
    silently becoming illegible.

    `side="left"` parks them outside the y-axis labels instead, for charts
    where the right margin is already spoken for.
    """
    rmap = _row_map(fig, n_rows)
    if not rmap:
        return fig

    counts, rows = {}, set()
    for t in fig.data:
        r = rmap.get(_axis_name(getattr(t, "yaxis", None) or "y"))
        if r is None:
            continue
        rows.add(r)
        t.legend = "legend" if r == 1 else f"legend{r}"
        if getattr(t, "showlegend", None) is not False:
            counts[r] = counts.get(r, 0) + 1

    height = getattr(fig.layout, "height", None) or 900
    spacing_px = 0.03 * height
    x, xanchor = (1.01, "left") if side == "right" else (-0.06, "right")

    for r in sorted(rows):
        # Find this row's primary axis to read its vertical domain.
        prim = next((k for k, v in rmap.items() if v == r), None)
        dom = getattr(getattr(fig.layout, prim, None), "domain", None) if prim else None
        if not dom:
            continue

        title = None
        if panel_titles and r <= len(panel_titles):
            title = str(panel_titles[r - 1])

        room = (dom[1] - dom[0]) * height + spacing_px
        n = max(counts.get(r, 0), 1)
        title_px = (font_size + 1) * 1.7 if title else 0
        usable = max(room - 14 - title_px, 20)
        size = font_size
        while size > 8 and n * size * 1.45 > usable:
            size -= 1

        cfg = dict(
            x=x, xanchor=xanchor, y=dom[1], yanchor="top",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(120,120,120,0.5)", borderwidth=1,
            font=dict(size=size), itemsizing="constant",
        )
        if title:
            cfg["title"] = dict(text=f"<b>{title}</b>", font=dict(size=size + 1))
        fig.layout[("legend" if r == 1 else f"legend{r}")] = cfg
    return fig
