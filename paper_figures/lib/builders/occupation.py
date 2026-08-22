"""Occupation-structure charts.

1. build_major_stacked  — major categories: phys/non-phys composition with
   the exposed share nested inside each, one bar per major, ranked by the
   headline % tasks exposed (the two dark segments sum to it exactly).
2. build_gwa_stacked    — the same construction at GWA level.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import figure_data
from lib.builders.legend import (
    paper_x_center,
    square_legend_html,
)
from lib.paper_config import FONT_FAMILY, PAPER_PALETTE, paper_fonts, style_paper_figure
from lib.utils import save_csv, save_figure

# Segment palette — non-phys stays on the paper's tasks blue, physical on
# the workers gold (same association as the job-zone Phys Mix panel);
# exposed = saturated, unexposed = pale tint of the same hue.
SEG_COLORS: dict[str, str] = {
    "nonphys_exposed": "#3d6a85",
    "nonphys_unexposed": "#c9d9e4",
    "phys_exposed": "#b3893a",
    "phys_unexposed": "#ead9b5",
}
SEG_LABELS: dict[str, str] = {
    "nonphys_exposed": "Non-Phys Exposed",
    "nonphys_unexposed": "Non-Phys Not Exposed",
    "phys_exposed": "Phys Exposed",
    "phys_unexposed": "Phys Not Exposed",
}
SEG_ORDER = ["nonphys_exposed", "nonphys_unexposed", "phys_exposed", "phys_unexposed"]
_DARK_SEGS = {"nonphys_exposed", "phys_exposed"}


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


# ─────────────────────────────────────────────────────────────────────────
# Shared computation
# ─────────────────────────────────────────────────────────────────────────

def _stacked_frame(
    rows: pd.DataFrame,
    group_col: str,
    hours_col: str,
    exposed_col: str,
) -> pd.DataFrame:
    """Per group: the four segment widths (each in % of the group's total
    work time) plus the headline exposed %.

    Denominator = Σ (time_per_day × emp) over every eco row of the group, so
    the bar is a decomposition of the hours actually worked. The two dark
    segments sum to the group's work-time-exposed percentage exactly.
    """
    def _agg(sub: pd.DataFrame) -> pd.Series:
        den = sub[hours_col].sum()
        phys = sub["physical"]
        return pd.Series({
            "den": den,
            "nonphys_share": sub.loc[~phys, hours_col].sum() / den * 100.0,
            "phys_share": sub.loc[phys, hours_col].sum() / den * 100.0,
            "nonphys_exposed": sub.loc[~phys, exposed_col].sum() / den * 100.0,
            "phys_exposed": sub.loc[phys, exposed_col].sum() / den * 100.0,
        })

    out = rows.groupby(group_col).apply(_agg, include_groups=False).reset_index()
    out["nonphys_unexposed"] = (out["nonphys_share"] - out["nonphys_exposed"]).clip(lower=0.0)
    out["phys_unexposed"] = (out["phys_share"] - out["phys_exposed"]).clip(lower=0.0)
    out["pct_exposed"] = out["nonphys_exposed"] + out["phys_exposed"]
    return out


# ── Number columns ────────────────────────────────────────────────────────
# Three of them ride in each panel's axis tail, past the end of the 0–100%
# bars: the headline work-time-exposed percentage, then the magnitudes that
# used to be their own figure (wkrs_wages.png, now carried rather than in
# the main body). Share and size belong on one row — the group percentage IS
# workers exposed ÷ employment, so they are one statement, not two.
#
# Every column is RIGHT-anchored on its own x, header included, so a column's
# width is the wider of its header and its values and the gap between columns
# is a gap between their right edges. Annotations beyond the axis range get
# clipped, and paper-coordinate placement doesn't survive two subplots, so
# these are data-x annotations on an axis that runs past 100.
#
# Magnitudes are rounded to whole units on purpose. A decimal place on an FTE
# headcount or a wage bill implies a precision the estimate does not have, and
# every character it costs comes straight out of the bars.

def _fmt_workers_round(v: float) -> str:
    """FTE workers exposed, whole units: 11M / 640K / 84."""
    av = abs(float(v))
    if av >= 1e6:
        return f"{av / 1e6:.0f}M"
    if av >= 1e3:
        return f"{av / 1e3:.0f}K"
    return f"{av:.0f}"


def _fmt_wages_round(v: float) -> str:
    """Wages exposed, whole units: $562B / $700M / $84K."""
    av = abs(float(v))
    if av >= 1e12:
        return f"${av / 1e12:.0f}T"
    if av >= 1e9:
        return f"${av / 1e9:.0f}B"
    if av >= 1e6:
        return f"${av / 1e6:.0f}M"
    if av >= 1e3:
        return f"${av / 1e3:.0f}K"
    return f"${av:.0f}"


# (df column, header, formatter, uses the pct font role)
_VALUE_COLS: tuple[tuple[str, str, object, bool], ...] = (
    ("pct_exposed", "Exposed", lambda v: f"<b>{v:.0f}%</b>", True),
    ("workers_fte", "Workers", _fmt_workers_round, False),
    ("wages_exposed", "Wages", _fmt_wages_round, False),
)

# Right edges of those columns, per chart, in the panel's own x units, with
# the axis top that has to clear the last of them. Module constants rather
# than call-site literals so the geometry can be asserted (see the tests):
# columns strictly increasing, the first clear of the 100% bar end, the last
# inside the axis.
_MAJOR_COL_X: tuple[float, float, float] = (128.0, 179.0, 222.0)
_MAJOR_X_TOP = 225.0
_GWA_COL_X: tuple[float, float, float] = (131.0, 187.0, 234.0)
_GWA_X_TOP = 236.0


def _render_stacked_split(
    df: pd.DataFrame,
    labels: list[str],
    title: str,
    width: int,
    row_height: int,
    hspacing: float,
    left_margin: int,
    bargap: float,
    out_png: Path,
    *,
    x_range_top: float,
    col_x: tuple[float, float, float],
    x_tickvals: tuple[int, ...] = (0, 25, 50, 75, 100),
    seg_text_min: float = 9.0,
    pct_font_role: str = "tick",
    right_margin: int = 90,
    legend_y_px: int = 128,
) -> go.Figure:
    """Horizontal 100%-stacked bars split across two columns: the top half of
    the ranking on the left, the bottom half on the right.

    Halves the figure height versus one tall column. Both halves carry their
    own y labels, x axis and three number columns; the legend is shared.
    """
    px = paper_fonts(width)
    n = len(df)
    half = (n + 1) // 2          # odd counts put the extra row on the left
    splits = [(0, half), (half, n)]
    # Plotly shrinks in-bar text that doesn't fit the bar height, which would
    # push it under the 8 pt print floor. Keep the bar at least as tall as the
    # floor font instead of letting the renderer silently rescale it.
    assert row_height * (1.0 - bargap) >= px["in_chart_floor"], (
        f"row_height {row_height} at bargap {bargap} gives a "
        f"{row_height * (1.0 - bargap):.0f}px bar, under the "
        f"{px['in_chart_floor']}px in-chart floor"
    )

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False, horizontal_spacing=hspacing,
    )

    for ci, (lo, hi) in enumerate(splits, start=1):
        part = df.iloc[lo:hi]
        part_labels = labels[lo:hi]
        for seg in SEG_ORDER:
            vals = part[seg].tolist()
            dark = seg in _DARK_SEGS
            fig.add_trace(go.Bar(
                y=part_labels,
                x=vals,
                orientation="h",
                name=SEG_LABELS[seg],
                legendgroup=seg,
                showlegend=False,
                marker=dict(color=SEG_COLORS[seg], line=dict(width=0)),
                text=[f"{v:.0f}%" if v >= seg_text_min else "" for v in vals],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    size=px["in_chart_floor"],
                    family=FONT_FAMILY,
                    color="#ffffff" if dark else PAPER_PALETTE["text"],
                ),
                cliponaxis=False,
            ), row=1, col=ci)

        axis_suffix = "" if ci == 1 else str(ci)
        # Mixed refs (data x, paper y) keep each header locked to this panel's
        # own column while riding at the top of the canvas. Header and values
        # share one right edge, so the header reads as the column's label and
        # three of them can sit side by side without the widest one shunting
        # its neighbour — which a centred header does the moment a second
        # column exists.
        for (key, header, fmt, is_pct), cx in zip(_VALUE_COLS, col_x):
            role = pct_font_role if is_pct else "in_chart_floor"
            for lab, v in zip(part_labels, part[key]):
                fig.add_annotation(
                    xref=f"x{axis_suffix}", yref=f"y{axis_suffix}",
                    x=cx, y=lab, xanchor="right", yanchor="middle",
                    text=fmt(v), showarrow=False,
                    font=dict(size=px[role], family=FONT_FAMILY,
                              color=PAPER_PALETTE["text"]),
                )
            fig.add_annotation(
                xref=f"x{axis_suffix}", yref="paper",
                x=cx, y=1.004, xanchor="right", yanchor="bottom",
                showarrow=False, text=header, align="right",
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["muted"]),
            )


    # Top margin holds the title AND the number-column headers in separate
    # lanes. It is sized to the two of them plus a single gap: the headers
    # ride the top of the plot area, so every pixel of top margin beyond what
    # the two lines need shows up as dead space between them. All three
    # headers are one word for that reason — a wrapped one would cost a lane.
    # Bottom margin has to clear the axis title, the legend's own offset and
    # its line height — the legend is top-anchored, so it grows downward out
    # of the canvas if the margin doesn't cover it.
    margin_t = 128
    margin_b = legend_y_px + px["legend"] + 16
    height = half * row_height + margin_t + margin_b
    fig.update_layout(barmode="stack", bargap=bargap)
    for ci in (1, 2):
        # dtick=1 — without it the renderer thins tightly-pitched category
        # labels and silently drops every other row's tick.
        fig.update_yaxes(autorange="reversed", dtick=1,
                         tickfont=dict(size=px["in_chart_floor"],
                                       family=FONT_FAMILY),
                         automargin=False, row=1, col=ci)
        fig.update_xaxes(
            range=[0, x_range_top],
            tickvals=list(x_tickvals),
            ticktext=[f"{v}%" for v in x_tickvals],
            tickangle=0,
            title="Share of Group's Work Time",
            row=1, col=ci,
        )

    style_paper_figure(fig, title, width=width, height=height,
                       margin=dict(l=left_margin, r=right_margin,
                                   t=margin_t, b=margin_b))
    fig.update_yaxes(tickfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY))
    fig.update_layout(title=dict(y=0.985, yanchor="top"), showlegend=False)
    # Glyph legend, one row, centred on the canvas. The built-in legend can't
    # carry a swatch this size (see builders/legend.py), and at four
    # entries it wrapped to two rows on both charts. Anchored in pixels
    # because a paper-referenced y drifts with the row count.
    plot_h = height - margin_t - margin_b
    fig.add_annotation(
        xref="paper", yref="paper",
        x=paper_x_center(width, left_margin, right_margin),
        y=-legend_y_px / plot_h,
        xanchor="center", yanchor="top", showarrow=False,
        text=square_legend_html([(SEG_LABELS[s], SEG_COLORS[s]) for s in SEG_ORDER]),
        font=dict(size=px["legend"], family=FONT_FAMILY,
                  color=PAPER_PALETTE["text"]),
    )
    save_figure(fig, out_png, scale=2)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Chart 1 — major categories
# ─────────────────────────────────────────────────────────────────────────

def build_major_stacked(results: Path, figures: Path) -> None:
    dataset = figure_data.PRIMARY_DATASET
    rows = figure_data.pair_exposure_rows(dataset)
    df = _stacked_frame(rows, "major_occ_category", "hours", "hours_exposed")

    # Cross-check the decomposition against the canonical major numbers, and
    # take the magnitudes off the same frame. They are built per occupation
    # and summed there, never a category percentage multiplied back out —
    # wages would be wrong, because exposure and pay correlate across occs.
    ref = figure_data.major_exposure(dataset).set_index("category")
    for _, row in df.iterrows():
        assert row["major_occ_category"] in ref.index, (
            f"Missing reference for {row['major_occ_category']}"
        )
        expect = float(ref.loc[row["major_occ_category"], "pct"])
        assert abs(row["pct_exposed"] - expect) < 0.01, (
            f"{row['major_occ_category']}: stacked {row['pct_exposed']:.2f} "
            f"vs reference {expect:.2f}"
        )
    for col in ("workers_fte", "wages_exposed"):
        df[col] = df["major_occ_category"].map(ref[col])
        assert df[col].notna().all(), f"Missing {col} for a major"

    df = df.sort_values("pct_exposed", ascending=False).reset_index(drop=True)
    labels = [figure_data.short_major_label(c) for c in df["major_occ_category"]]

    name = "major_categories_stacked.png"
    _render_stacked_split(
        df, labels,
        title="Work Time Exposed by Major Occupational Category and Physical Makeup",
        # Left margin and gutter are both sized to the 17-char label; the
        # right margin is nearly nothing, because the last number column ends
        # at the panel's own edge. That is where the room for three columns
        # came from — not from the bars, which are within 10% of the length
        # they had when the panel carried one number.
        width=2200, row_height=62, hspacing=0.188, left_margin=336,
        # Column pitch is set by the HEADERS, not the values: "Workers" is 7
        # characters where "935K" is 4, and a character is ~0.5 of the floor
        # font. So the gap between the % and Workers columns is wider than
        # the numbers in them suggest it needs to be.
        bargap=0.32, right_margin=34, x_range_top=_MAJOR_X_TOP,
        col_x=_MAJOR_COL_X,
        x_tickvals=(0, 50, 100),
        # A 3-character label measures 73 px at the floor size, which is 22%
        # of a 331 px bar. Below that plotly silently SHRINKS the label —
        # under the 8 pt print floor, with no error — rather than dropping
        # it, which is what the old 13% threshold was quietly doing to every
        # sub-20% segment on this chart.
        seg_text_min=22.0, out_png=results / "figures" / name,
    )
    save_csv(df, results / "major_categories_stacked.csv")
    _copy_fig(results, figures, name)


# ─────────────────────────────────────────────────────────────────────────
# Chart 2 — general work activities
# ─────────────────────────────────────────────────────────────────────────

def build_gwa_stacked(results: Path, figures: Path) -> None:
    dataset = figure_data.PRIMARY_DATASET
    rows = figure_data.act_exposure_rows(dataset, "gwa_title")
    df = _stacked_frame(rows, "gwa_title", "hours_split", "hours_exposed_split")

    ref = figure_data.act_exposure(dataset, "gwa_title").set_index("category")
    for _, row in df.iterrows():
        assert row["gwa_title"] in ref.index, (
            f"Missing reference for {row['gwa_title']}"
        )
        expect = float(ref.loc[row["gwa_title"], "pct"])
        assert abs(row["pct_exposed"] - expect) < 0.01, (
            f"{row['gwa_title']}: stacked {row['pct_exposed']:.2f} "
            f"vs reference {expect:.2f}"
        )
    # The activity-grain magnitudes are the /n-split ones, so a task sitting
    # in three GWAs gives each a third of its time rather than all of it
    # three times — they sum back to the economy, same as the hours do.
    for col in ("workers_fte", "wages_exposed"):
        df[col] = df["gwa_title"].map(ref[col])
        assert df[col].notna().all(), f"Missing {col} for a GWA"

    df = df.sort_values("pct_exposed", ascending=False).reset_index(drop=True)
    labels = [figure_data.short_gwa_label(g) for g in df["gwa_title"]]

    name = "gwa_stacked.png"
    _render_stacked_split(
        df, labels,
        title="Work Time Exposed by General Work Activity and Physical Makeup",
        width=2400, row_height=50, hspacing=0.219, left_margin=418,
        bargap=0.14, right_margin=34, out_png=results / "figures" / name,
        # 37 labels across two columns, each column charged to the canvas
        # twice, plus three number columns — so the bars get less room here
        # than on the major chart and only the wider segments can carry their
        # own label. Column geometry is the same construction as there.
        x_range_top=_GWA_X_TOP, col_x=_GWA_COL_X,
        x_tickvals=(0, 50, 100), seg_text_min=25.0,
        pct_font_role="in_chart_floor",
    )
    save_csv(df, results / "gwa_stacked.csv")
    _copy_fig(results, figures, name)
