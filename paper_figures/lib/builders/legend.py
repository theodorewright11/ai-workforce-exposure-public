"""Shared one-row legend helper for the paper figures.

Plotly clamps a legend symbol to ~15 px however large the trace's marker is
(measured: sizes 16, 20, 24, 30 and 40 all render 15 px), so a swatch drawn
through the built-in legend cannot scale with the canvas — on the 2200 px
major chart it prints at a third the share of the canvas it takes on the
1180 px job-zone chart, which is why those keys read as noticeably smaller.

These charts therefore draw the key as a single centred annotation built from
the ■ glyph, which takes the annotation's font size and so holds a constant
share of the canvas at any width. It is the same device the bar-shade ramps
in adoption.py and wkrswages.py already use, and being one annotation it is
one row by construction.
"""
from __future__ import annotations

SQUARE = "■"

# Four non-breaking spaces read as a clear item break at legend size without
# needing per-canvas tuning; a single space closes up at 8 pt print.
ITEM_GAP = "&nbsp;&nbsp;&nbsp;&nbsp;"


def square_legend_html(
    items: list[tuple[str, str]],
    gap: str = ITEM_GAP,
) -> str:
    """One-row legend markup: colour square + label per (label, color) item."""
    assert items, "square_legend_html needs at least one item"
    return gap.join(
        f"<span style='color:{color}'>{SQUARE}</span>&nbsp;{label}"
        for label, color in items
    )


def paper_x_center(width: int, left_margin: int, right_margin: int) -> float:
    """Canvas midpoint expressed in paper (plot-area) x coordinates.

    A paper-referenced annotation at x=0.5 is centred on the PLOT AREA, which
    a large left label margin pushes well right of the canvas centre.
    """
    span = width - left_margin - right_margin
    assert span > 0, f"margins {left_margin}+{right_margin} exceed width {width}"
    return (width / 2 - left_margin) / span
