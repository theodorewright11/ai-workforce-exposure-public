"""Focused-set chart — occupations that are both heavily AI-exposed and
projected to shed employment.

Membership is now exactly what the title says: work time exposed at or above
EXPOSURE_MIN, and a negative BLS 2024–34 employment projection. The paper's
canonical focused set (audit_risk_score) layered an SKA gate and an
exposure-trend gate on top of those two, which made the set hard to state in
a sentence and impossible to reproduce from the chart. Dropping them costs
little — the set moves from 40 occupations to 31 — and buys a definition a
reader can check.

`audit_risk_score._load_flag_df` is still the source of the BLS projection,
the job zone and the major label; only the membership rule is local.
Everything plotted is derived on the 05-31 files at the hours weighting.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import figure_data
from lib.paper_config import (
    EMP_DARK,
    EMP_LIGHT,
    FONT_FAMILY,
    PAPER_PALETTE,
    PAPER_W,
    paper_fonts,
    style_paper_figure,
)
from lib.utils import save_csv, save_figure


# Width of the workers panel in canvas pixels, MEASURED off the rendered
# figure (259 px for the 2.04M longest bar) rather than derived from W and
# the margins. The y axis keeps automargin and carries 40-character
# occupation names, so plotly grows the left margin well past MARGIN_L and
# the panel comes out ~30% narrower than the nominal geometry implies.
WORKERS_PANEL_PX = 531.0


def worker_label_inside(value: float, text: str, x_top: float,
                        floor_px: float) -> bool:
    """Does a workers label fit inside its own bar?

    Geometry, not a share of the longest bar: a flat 20%-of-max cut-off put
    Computer User Support Specialists (516K) inside a 66 px bar under a 79 px
    label, which overran to the left into the occupation names. A numeral is
    ~0.62 of the font size in this face (measured), and 14 px covers the
    inset plus a hair of breathing room.
    """
    bar_px = value * WORKERS_PANEL_PX / x_top
    return bar_px >= len(text) * 0.62 * floor_px + 14.0


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def occ_usage_lift() -> pd.Series:
    """Per-occupation AI usage per worker, as a multiple of the median
    occupation economy-wide.

    Σ debiased usage pct ÷ Σ employment over the occupation's (task,
    occupation) pairs — the same denominator rule as every other usage
    number.

    Two anchoring choices are baked in here.

    The median is taken over the ECONOMY, not over the focused set. Every
    other ×Med column in the folder anchors on the whole of the thing it
    decomposes (all 22 majors, all 5 zones, all 37 GWAs), and the focused set
    is a filtered subset rather than a decomposition — anchoring inside it
    would make 1.00× mean something different from the same-looking number on
    every other figure, and would answer "is this occupation typical of the
    set?", which is not the question. Against the economy the column says the
    set's median occupation already uses AI at 4.5× the median occupation,
    and that 7 of the 31 sit BELOW it.

    The population is the 815 occupations with at least one rated (task, occ)
    pair, not all 923. The other 108 have no observed usage at all; entering
    them as zeros would drag the median down and multiply every value on the
    chart by 1.41. Excluding them keeps the anchor "the median occupation AI
    has been seen used in", which is the conservative reading.
    """
    pairs = figure_data.intensity_pairs()
    num = pairs.groupby("title_current")["adj_pct"].sum()
    emp = figure_data.pair_level_emp("title_current")
    rate = (num / emp.reindex(num.index)).replace([np.inf, -np.inf], np.nan).dropna()
    assert not rate.empty, "No rated usage pairs"
    return figure_data.anchor_lower_median(rate)


EXPOSURE_MIN = 67.0


def build_focused_set_usage(results: Path, figures: Path) -> None:
    from lib.exploratory.risk_score import _load_flag_df

    flags_df = _load_flag_df()

    # Membership and every plotted value are derived at the hours weighting
    # on the 05-31 files, so the gate the title states is the gate the bars
    # show — the paper's set was gated on the Feb-vintage freq percentages.
    exposure = figure_data.occ_exposure(figure_data.PRIMARY_DATASET).set_index("title_current")
    s5f = flags_df.copy()
    s5f["major_short"] = s5f["major"].str.replace(" Occupations", "", regex=False)
    s5f["pct"] = s5f["title_current"].map(exposure["p"] * 100.0)
    s5f = s5f[(s5f["pct"] >= EXPOSURE_MIN) & (s5f["emp_proj_pct"] < 0)].copy()
    assert not s5f.empty, f"No occupations clear {EXPOSURE_MIN}% + negative projection"

    s5f["workers_affected"] = s5f["title_current"].map(
        exposure["p"] * exposure["emp"]
    ).fillna(0.0)
    s5f["wages_affected"] = s5f["title_current"].map(
        exposure["p"] * exposure["emp"] * exposure["wage"]
    ).fillna(0.0)
    s5f["usage_lift"] = s5f["title_current"].map(occ_usage_lift()).fillna(0.0)

    s = s5f.sort_values("pct", ascending=True).reset_index(drop=True)

    save_csv(
        s5f.sort_values("pct", ascending=False)[
            ["title_current", "major_short", "job_zone", "emp_proj_pct",
             "usage_lift", "pct", "workers_affected", "wages_affected"]
        ],
        results / "focused_set_usage.csv",
        float_format="%.4f",
    )

    # One decimal normally; two if that would collapse a lot of the column to
    # "0.0" — most of this set sits well under the median occupation.
    two_dp = (s5f["usage_lift"].round(1) == 0.0).mean() > 0.25
    fmt_lift = (lambda v: f"{v:.2f}x") if two_dp else (lambda v: f"{v:.1f}x")

    def _truncate_title(t: str, max_len: int = 50) -> str:
        if len(t) <= max_len:
            return t
        breakers = [i for i in range(max_len) if t[i] in ", "]
        cut = max(breakers) if breakers else max_len - 1
        return t[:cut].rstrip(" ,") + "…"

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    def _format_workers(w: float) -> str:
        if w >= 1_000_000:
            return f"{w/1_000_000:.1f}M"
        if w >= 1_000:
            return f"{w/1_000:.0f}K"
        return f"{w:.0f}"

    y_labels = [_truncate_title(t) for t in s["title_current"]]

    abs_emp = s["emp_proj_pct"].abs()
    cmin, cmax = float(abs_emp.min()), float(abs_emp.max())
    abs_mid = (cmin + cmax) / 2

    W = PAPER_W + 460
    px = paper_fonts(W)
    floor_px, tick_px = px["in_chart_floor"], px["tick"]
    axis_px, panel_px = px["axis_title"], px["panel_title"]

    # Panel titles are drawn below as data-anchored annotations at x=0 so
    # they sit over the start of the bars rather than centred on a panel
    # whose right half is number columns. Two lines apiece so they fit the
    # bar region, which lets them share the column-header lane and takes a
    # whole lane out of the top margin.
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["", ""],
        shared_yaxes=True, horizontal_spacing=0.04,
        column_widths=[0.4, 0.6],
    )

    common_marker = dict(
        color=abs_emp.values,
        colorscale=[[0, EMP_LIGHT], [1, EMP_DARK]],
        cmin=cmin, cmax=cmax, showscale=False, line=dict(width=0),
    )

    fig.add_trace(go.Bar(
        y=y_labels, x=s["pct"], orientation="h",
        marker=common_marker, showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=y_labels, x=s["workers_affected"], orientation="h",
        marker=common_marker, showlegend=False, hoverinfo="skip",
    ), row=1, col=2)

    MARGIN_L, MARGIN_R = 380, 110
    # Top margin holds three lanes: title, panel titles, column headers.
    # At 300 the block above the first bar was taller than four chart rows.
    MARGIN_T, MARGIN_B = 175, 165
    pct_max = float(s["pct"].max())
    wrk_max = float(s["workers_affected"].max())
    inside_threshold_pct = 0.20 * pct_max
    # Three number columns ride in the headroom past the workers bars. The
    # gaps have to clear the widest cell ("-22.1%") at floor size, so they
    # are set in units of wrk_max rather than eyeballed.
    proj_col_x = wrk_max * 1.10
    usage_col_x = wrk_max * 1.58
    x_top_b = wrk_max * 2.05


    for i, row in s.iterrows():
        is_dark = abs(row["emp_proj_pct"]) >= abs_mid
        text_color_inside = "white" if is_dark else PAPER_PALETTE["text_dark"]

        pct_text = f"{row['pct']:.0f}%"
        if row["pct"] >= inside_threshold_pct:
            fig.add_annotation(
                x=row["pct"], y=y_labels[i], xref="x1", yref="y1",
                text=pct_text, showarrow=False,
                xanchor="right", yanchor="middle", xshift=-6,
                font=dict(size=floor_px, color=text_color_inside,
                          family=FONT_FAMILY),
            )
        else:
            fig.add_annotation(
                x=row["pct"], y=y_labels[i], xref="x1", yref="y1",
                text=pct_text, showarrow=False,
                xanchor="left", yanchor="middle", xshift=4,
                font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                          family=FONT_FAMILY),
            )

        w = row["workers_affected"]
        w_text = _format_workers(w)
        if worker_label_inside(w, w_text, x_top_b, floor_px):
            fig.add_annotation(
                x=w, y=y_labels[i], xref="x2", yref="y2",
                text=w_text, showarrow=False,
                xanchor="right", yanchor="middle", xshift=-6,
                font=dict(size=floor_px, color=text_color_inside,
                          family=FONT_FAMILY),
            )
        else:
            fig.add_annotation(
                x=w, y=y_labels[i], xref="x2", yref="y2",
                text=w_text, showarrow=False,
                xanchor="left", yanchor="middle", xshift=4,
                font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                          family=FONT_FAMILY),
            )

        fig.add_annotation(
            x=proj_col_x, y=y_labels[i], xref="x2", yref="y2",
            text=f"{row['emp_proj_pct']:+.1f}%", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        )
        fig.add_annotation(
            x=usage_col_x, y=y_labels[i], xref="x2", yref="y2",
            text=fmt_lift(row["usage_lift"]), showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=floor_px, color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        )

    for x_pos, header in [(proj_col_x, "Emp<br>Proj"),
                          (usage_col_x, "Usage<br>×Med")]:
        fig.add_annotation(
            x=x_pos, y=1.0, xref="x2", yref="y2 domain",
            text=header, showarrow=False, align="left",
            xanchor="left", yanchor="bottom", yshift=4,
            font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        )

    # "y domain", not "y1 domain" — plotly only accepts the bare form for the
    # first axis.
    for xref, yref, panel_title in [("x1", "y domain", "% Work Time<br>Exposed"),
                                    ("x2", "y2 domain", "Workers<br>Exposed")]:
        fig.add_annotation(
            x=0, y=1.0, xref=xref, yref=yref,
            text=panel_title, showarrow=False, align="left",
            xanchor="left", yanchor="bottom", yshift=4,
            font=dict(size=panel_px, color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        )

    # Ticks stop at the data, not at the number columns' headroom.
    wrk_ticks = [0.0, wrk_max * 0.5, wrk_max]

    n = len(s)
    height = max(620, n * 32 + MARGIN_T + MARGIN_B)

    style_paper_figure(
        fig,
        "High AI Work Time Exposed × Negative Employment Projection",
        height=height, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )

    # Two lanes in the top margin now: the chart title pinned to the
    # container top, then a single lane holding the panel titles (left, over
    # the bars) and the number-column headers (right) side by side.
    fig.update_layout(title=dict(y=0.985, yanchor="top"))

    fig.update_xaxes(
        title=dict(text="% Work Time Exposed",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"], ticksuffix="%",
        tickfont=dict(size=tick_px, family=FONT_FAMILY),
        # Drop the boundary tick — panel 2's leading "0" sits right next to it.
        tickmode="array", tickvals=[0, 20, 40, 60],
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text="Workers Exposed",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=tick_px, family=FONT_FAMILY),
        range=[0, x_top_b],
        tickmode="array", tickvals=wrk_ticks,
        ticktext=[_format_workers(v) for v in wrk_ticks],
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text="Occupation",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickfont=dict(size=floor_px, family=FONT_FAMILY),
        tickmode="array", tickvals=y_labels, ticktext=y_labels,
        row=1, col=1,
    )
    fig.update_yaxes(showgrid=False, showline=False, showticklabels=False,
                     row=1, col=2)

    rgb_l, rgb_d = _hex_to_rgb(EMP_LIGHT), _hex_to_rgb(EMP_DARK)
    swatch_html = ""
    for i in range(7):
        t = i / 6
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    # Single-line legend: the Σ-usage explainer went with the two columns it
    # described, and the remaining line drops clear of the axis titles.
    legend_text = (
        f"BLS Emp Proj 2024–2034 (more negative → darker)&nbsp;&nbsp;"
        f"-{cmin:.0f}%&nbsp;{swatch_html}&nbsp;-{cmax:.0f}%"
    )
    fig.add_annotation(
        x=0.44, y=-92 / (height - MARGIN_T - MARGIN_B), xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="top",
        font=dict(size=floor_px, color=PAPER_PALETTE["text"],
                  family=FONT_FAMILY),
    )

    fig.update_layout(bargap=0.15)

    name = "focused_set_usage.png"
    save_figure(fig, results / "figures" / name, scale=2)
    _copy_fig(results, figures, name)
