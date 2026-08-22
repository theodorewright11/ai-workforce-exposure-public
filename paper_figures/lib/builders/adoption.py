"""Adoption charts — actual AI usage relative to the median category.

1. build_major_adoption — usage per major occupational category.
2. build_gwa_adoption  — the same chart at GWA level, split across two
   columns because 37 activities in one stack is twice as tall as it needs
   to be.

Usage everywhere is Σ debiased usage pct ÷ Σ employment, anchored so the
lower-middle category reads exactly 1.00x. The denominator is employment
alone — see figure_data.intensity_pairs for why it is not re-weighted by time.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import figure_data
from lib.builders.legend import (
    paper_x_center,
)
from lib.paper_config import FONT_FAMILY, PAPER_PALETTE, paper_fonts, style_paper_figure
from lib.utils import save_csv, save_figure

TASKS_LIGHT, TASKS_DARK = "#cfe0ec", "#2c4f6b"


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _ramp(t: float, light: str = TASKS_LIGHT, dark: str = TASKS_DARK) -> str:
    def _rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    lo, hi = _rgb(light), _rgb(dark)
    r, g, b = (round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return f"rgb({r},{g},{b})"


# ─────────────────────────────────────────────────────────────────────────
# Shared frame + renderer
# ─────────────────────────────────────────────────────────────────────────

def _usage_frame(
    num: pd.Series,
    den: pd.Series,
    exposure: pd.Series,
) -> pd.DataFrame:
    """Assemble a ranked usage frame from a numerator, an employment
    denominator and a work-time-exposed series (all indexed by category)."""
    raw = num.rename("num").to_frame()
    raw["den"] = den.reindex(raw.index)
    df = raw.reset_index().rename(columns={raw.index.name or "index": "category"})
    df["ratio"] = np.where(df["den"] > 0, df["num"] / df["den"], 0.0)
    df["lift"] = figure_data.anchor_lower_median(df["ratio"])
    df["debias_share"] = df["num"] / df["num"].sum() * 100.0
    df["pct_exposed"] = df["category"].map(exposure).fillna(0.0)
    return df.sort_values("lift", ascending=False).reset_index(drop=True)


def _render_adoption(
    df: pd.DataFrame,
    labels: list[str],
    title: str,
    width: int,
    row_height: int,
    left_margin: int,
    out_png: Path,
    *,
    split: bool = False,
    hspacing: float = 0.42,
    right_margin: int = 210,
    bargap: float = 0.20,
) -> None:
    """Horizontal ×median bars shaded by work time exposed, with a Σ raw
    usage share column on the right.

    `split=True` breaks the ranking across two columns (top half left, bottom
    half right), which is what keeps a 37-row activity chart to the height of
    a 19-row one. Row labels and bar labels both sit at the 8 pt print floor
    so the row pitch can come down to the floor's own height.
    """
    px = paper_fonts(width)
    # The number column is the RAW usage share, not the debiased one. The
    # debias is a GWA-level prior on the numerator and belongs there; a
    # column of debiased shares reads as an observation when it is a
    # correction. `debias_share` is still written to the CSV. Same call the
    # job-zone chart made when it dropped its own debiased-share column.
    assert "raw_share" in df.columns, "caller must attach raw_share"
    # Plotly silently shrinks in-bar text that will not fit the bar height,
    # which would push it under the print floor with no error.
    assert row_height * (1.0 - bargap) >= px["in_chart_floor"], (
        f"row_height {row_height} at bargap {bargap} leaves a "
        f"{row_height * (1.0 - bargap):.0f}px bar, under the "
        f"{px['in_chart_floor']}px floor"
    )

    n = len(df)
    half = (n + 1) // 2 if split else n
    splits = [(0, half), (half, n)] if split else [(0, n)]
    n_cols = len(splits)

    lo, hi = float(df["pct_exposed"].min()), float(df["pct_exposed"].max())
    shade = (df["pct_exposed"] - lo) / max(hi - lo, 1e-9)

    # The bars run to the longest lift; the axis runs further so the Σ column
    # has clean whitespace to sit in rather than crowding the longest bar.
    x_top = float(df["lift"].max()) * 1.04
    x_axis_max = x_top * 1.34
    col_x = x_top * 1.11

    fig = make_subplots(rows=1, cols=n_cols, shared_yaxes=False,
                        horizontal_spacing=hspacing if split else 0.0)

    for ci, (a, b) in enumerate(splits, start=1):
        part = df.iloc[a:b]
        part_labels = labels[a:b]
        fig.add_trace(go.Bar(
            y=part_labels, x=part["lift"], orientation="h",
            marker=dict(color=[_ramp(t) for t in shade.iloc[a:b]],
                        line=dict(width=0)),
            text=[f"{v:.2f}x" for v in part["lift"]],
            textposition=["inside" if v >= 0.34 * x_top else "outside"
                          for v in part["lift"]],
            insidetextanchor="end",
            insidetextfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                                color="#ffffff"),
            outsidetextfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                                 color=PAPER_PALETTE["text"]),
            constraintext="none", cliponaxis=False,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=ci)

        suffix = "" if ci == 1 else str(ci)
        fig.add_shape(type="line", xref=f"x{suffix}", yref=f"y{suffix} domain",
                      x0=1.0, x1=1.0, y0=0, y1=1,
                      line=dict(color=PAPER_PALETTE["negative"], width=1.4,
                                dash="dash"))
        fig.add_annotation(
            xref=f"x{suffix}", yref="paper", x=1.0, y=1.004,
            xanchor="center", yanchor="bottom", showarrow=False, text="median",
            font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["negative"]),
        )
        for lab, v in zip(part_labels, part["raw_share"]):
            fig.add_annotation(
                xref=f"x{suffix}", yref=f"y{suffix}", x=col_x, y=lab,
                xanchor="left", yanchor="middle", showarrow=False,
                text=f"{v:.1f}%",
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
        fig.add_annotation(
            xref=f"x{suffix}", yref="paper", x=col_x, y=1.004,
            xanchor="left", yanchor="bottom", showarrow=False,
            text="Σ raw",
            font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["muted"]),
        )

    # Bottom margin holds the x-axis title, then the bar-shade key. The key
    # is one line now (the Σ-raw explainer moved into the methods text), so
    # it sits far closer to the axis than the two-line version could.
    MARGIN_T, SHADE_BELOW = 110, 125
    MARGIN_B = SHADE_BELOW + px["legend"] + 16
    height = half * row_height + MARGIN_T + MARGIN_B
    fig.update_layout(bargap=bargap)
    for ci in range(1, n_cols + 1):
        fig.update_yaxes(autorange="reversed", dtick=1, automargin=False,
                         showgrid=False, showline=False,
                         tickfont=dict(size=px["in_chart_floor"],
                                       family=FONT_FAMILY),
                         row=1, col=ci)
        fig.update_xaxes(range=[0, x_axis_max], showgrid=True,
                         gridcolor=PAPER_PALETTE["grid"],
                         title="Usage Relative to Median (×)", row=1, col=ci)
    # No y-axis title: the rows are named one per line and the chart title
    # already says what they are. Dropping it lets the left margin come down
    # to the width of the labels, which re-centres the two panels.

    style_paper_figure(fig, title, width=width, height=height,
                       margin=dict(l=left_margin, r=right_margin,
                                   t=MARGIN_T, b=MARGIN_B))
    fig.update_yaxes(tickfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY))
    fig.update_layout(title=dict(y=0.985, yanchor="top"))

    steps = "".join(f"<span style='color:{_ramp(v / 6)}'>█</span>" for v in range(7))
    plot_h = height - MARGIN_T - MARGIN_B
    fig.add_annotation(
        xref="paper", yref="paper",
        x=paper_x_center(width, left_margin, right_margin),
        y=-SHADE_BELOW / plot_h,
        xanchor="center", yanchor="top", showarrow=False,
        text=f"Bar shade: Work Time Exposed  {lo:.0f}% {steps} {hi:.0f}%",
        font=dict(size=px["legend"], family=FONT_FAMILY,
                  color=PAPER_PALETTE["text"]),
    )
    save_figure(fig, out_png, scale=2)


# ─────────────────────────────────────────────────────────────────────────
# Chart 1 — major-category adoption
# ─────────────────────────────────────────────────────────────────────────

def build_major_adoption(results: Path, figures: Path) -> None:
    pairs = figure_data.intensity_pairs()
    num = pairs.groupby("major_occ_category")["adj_pct"].sum()
    den = figure_data.pair_level_emp("major_occ_category")
    exposure = (
        figure_data.major_exposure(figure_data.PRIMARY_DATASET)
        .set_index("category")["pct"]
    )
    df = _usage_frame(num, den, exposure)
    assert len(df) == 22, f"Expected 22 majors, got {len(df)}"

    raw = pairs.groupby("major_occ_category")["pct_normalized"].sum()
    df["raw_share"] = df["category"].map(raw / raw.sum() * 100.0).fillna(0.0)
    save_csv(df, results / "major_adoption.csv", float_format="%.4f")

    labels = [figure_data.medium_major_label(c) for c in df["category"]]
    name = "major_adoption.png"
    _render_adoption(
        df, labels,
        title="Actual AI Usage by Major Occupational Category",
        # Split across two columns like the GWA chart: 22 majors in one stack
        # is twice as tall as it needs to be.
        #
        # Fuller labels than the stacked charts carry (24 chars, not 17): this
        # chart has one number column instead of three, so the gutter only has
        # to clear the right panel's labels and was sitting on ~290 px of
        # empty space. The gutter comes down as the labels grow, which leaves
        # the bars slightly WIDER than they were with the short ones.
        width=1900, row_height=44, left_margin=430, right_margin=90,
        split=True, hspacing=0.306,
        out_png=results / "figures" / name,
    )
    _copy_fig(results, figures, name)


# ─────────────────────────────────────────────────────────────────────────
# Chart 2 — GWA adoption
# ─────────────────────────────────────────────────────────────────────────

def build_gwa_adoption(results: Path, figures: Path) -> None:
    gwa_rows = figure_data.intensity_gwa_rows()
    num = gwa_rows.groupby("gwa_title")["adj_pct_split"].sum()
    den = figure_data.eco_gwa_weight_split()
    exposure = (
        figure_data.act_exposure(figure_data.PRIMARY_DATASET, "gwa_title")
        .set_index("category")["pct"]
    )
    df = _usage_frame(num.reindex(den.index).fillna(0.0), den, exposure)
    assert len(df) >= 35, f"Expected ~37 GWAs, got {len(df)}"

    raw = gwa_rows.groupby("gwa_title")["raw_pct_split"].sum()
    df["raw_share"] = df["category"].map(raw / raw.sum() * 100.0).fillna(0.0)
    save_csv(df, results / "gwa_adoption.csv", float_format="%.4f")

    labels = [figure_data.medium_gwa_label(g) for g in df["category"]]
    name = "gwa_adoption.png"
    _render_adoption(
        df, labels,
        title="Actual AI Usage by General Work Activity",
        # Left margin holds the longest 26-char label and nothing else; the
        # gutter holds the right panel's copy of it. Same trade as the major
        # chart — the space came out of an empty gutter, not out of the bars.
        width=1900, row_height=42, left_margin=444, right_margin=90,
        split=True, hspacing=0.337,
        out_png=results / "figures" / name,
    )
    _copy_fig(results, figures, name)
