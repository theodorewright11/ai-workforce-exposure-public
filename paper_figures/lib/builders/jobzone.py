"""Job-zone chart — the paper's three-panel violin chart plus per-zone
actual-usage numbers on the right:

  ×Med    — Σ debiased pct ÷ Σ (freq×emp over ALL eco pairs in the zone),
            divided by the median of the five zone ratios (ratio of sums,
            same construction as the major-level intensity anchor).
  Σ raw   — the zone's share of total raw usage pct.

The debiased-share column was dropped from the figure — ×Med already carries
the debiased signal, and three number columns crowded the right margin. The
value is still written to the CSV as `debias_share`.

Violin panels follow the earlier three-panel job-zone violin chart
(unchanged construction) with the right margin widened for the columns.
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
    FONT_FAMILY,
    GROUP_COLORS,
    PAPER_PALETTE,
    paper_fonts,
    style_paper_figure,
)
from lib.utils import save_csv, save_figure
from lib.figure_data import _load_occ_structural

ZONE_COLORS = {1: "#b8cfe0", 2: "#8cafc5", 3: "#6090aa", 4: "#3a6f8f", 5: "#1a4f73"}


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _zone_usage() -> pd.DataFrame:
    """Per-zone usage numbers: Σ debiased usage pct ÷ Σ employment in the
    zone, anchored on the median zone; plus debiased and raw pct shares.

    The denominator sums employment over (task, occupation) pairs, the same
    rule the activity-level charts use (see figure_data.pair_level_emp).
    """
    pairs = figure_data.intensity_pairs()
    rated = pairs.dropna(subset=["job_zone"]).copy()
    rated["job_zone"] = rated["job_zone"].astype(int)
    num = rated.groupby("job_zone").agg(
        num=("adj_pct", "sum"), raw=("pct_normalized", "sum")
    )

    den = figure_data.pair_level_emp("job_zone", ("job_zone",)).rename("den")
    den.index = den.index.astype(int)

    z = num.join(den, how="right").fillna(0.0)
    z = z.reindex([1, 2, 3, 4, 5]).fillna(0.0)
    z["ratio"] = np.where(z["den"] > 0, z["num"] / z["den"], 0.0)
    med = float(z["ratio"].median())
    assert med > 0, "Median zone usage ratio is zero"
    z["lift"] = z["ratio"] / med
    z["debias_share"] = z["num"] / z["num"].sum() * 100.0
    z["raw_share"] = z["raw"] / z["raw"].sum() * 100.0
    return z.reset_index()


def build_job_zone_usage(results: Path, figures: Path) -> None:
    occ_all = _load_occ_structural()
    # Reclassify on WORK TIME, not task count. `_load_occ_structural` buckets
    # an occupation by the share of its tasks flagged physical; every other
    # number on this figure is hours, and an occupation with three physical
    # tasks that fill half its day is not the same thing as one with three it
    # touches for ten minutes. Same 33 / 67 cuts, different denominator — and
    # it moves both the Phys Mix panel and which occupations the non-physical
    # panel contains.
    phys_share = figure_data.occ_phys_hours_share()
    occ_all["pct_physical"] = occ_all["title_current"].map(phys_share)
    assert occ_all["pct_physical"].notna().all(), (
        "occupations missing from the eco pair universe"
    )
    occ_all["occ_group"] = figure_data.phys_tier(occ_all["pct_physical"]).astype(str)

    pct = (
        figure_data.occ_exposure(figure_data.PRIMARY_DATASET)
        .set_index("title_current")["p"]
        .mul(100.0)
    )
    occ_all["pct_tasks_affected"] = occ_all["title_current"].map(pct)
    occ_all = occ_all.dropna(subset=["pct_tasks_affected", "job_zone"])
    occ_all["job_zone"] = occ_all["job_zone"].astype(int)
    occ_nonphys = occ_all[occ_all["occ_group"] == "Non-physical"].copy()

    n_all, n_nonphys = len(occ_all), len(occ_nonphys)
    zones = [1, 2, 3, 4, 5]
    # Short zone ticks. The full ZONE_LABELS text claims ~a quarter of the
    # canvas in the left margin, which squeezes the three panels until
    # their axis titles overlap. The prep levels move to the footnote.
    zl = {z: f"Zone {z}" for z in zones}

    zone_stats = []
    for z in zones:
        sub = occ_all[occ_all["job_zone"] == z]
        sub_np = occ_nonphys[occ_nonphys["job_zone"] == z]
        n_total = len(sub)
        zone_stats.append({
            "job_zone": z,
            "n_occs_all": n_total,
            "median_pct_all": round(float(sub["pct_tasks_affected"].median()), 1) if n_total else None,
            "n_occs_nonphys": len(sub_np),
            "median_pct_nonphys": round(float(sub_np["pct_tasks_affected"].median()), 1) if len(sub_np) else None,
            "pct_physical": round((sub["occ_group"] == "Physical").mean() * 100, 1) if n_total else 0.0,
            "pct_mixed": round((sub["occ_group"] == "Mixed").mean() * 100, 1) if n_total else 0.0,
            "pct_non_physical": round((sub["occ_group"] == "Non-physical").mean() * 100, 1) if n_total else 0.0,
        })

    usage = _zone_usage()
    stats_df = pd.DataFrame(zone_stats).merge(usage, on="job_zone")
    save_csv(stats_df, results / "job_zone_usage.csv")

    y_labels = [zl[z] for z in zones]
    y_order_top_down = list(reversed(y_labels))

    # Narrower than PAPER_W: at a fixed print pt a narrower canvas means less
    # dead margin around the same three panels, which is what the extra
    # whitespace on both sides was.
    W = 1180
    # The violins are one-sided (side="positive", width 0.7), so every row's
    # slack sits ABOVE its curve and a flat zone like Zone 1 leaves most of
    # its band empty. Height is therefore the only lever on that whitespace —
    # fonts scale off W, not H, so shrinking H compresses the dead space and
    # leaves every label at its print size. 565 gives ~62 px a row, which
    # still clears the floor-size "n= · med" line under each violin.
    H = 595
    # Right margin holds two number columns now that Σ debias is gone, and
    # the bottom one row of legend instead of two.
    # Top margin holds the title and the section headers in two lanes. The
    # extra 30 px over the tightest fit is the gap under the title; it is
    # added to H as well, so opening it does not eat back into the violins.
    # Left margin is now just the "Zone n" ticks plus a hair — the axis title
    # that used to share it is gone. Right margin holds the two number
    # columns, which run past the plot area, so the canvas reads centred with
    # a much smaller left margin than a symmetric one would give.
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 106, 202, 142, 118
    px = paper_fonts(W)

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        column_widths=[0.44, 0.16, 0.40], horizontal_spacing=0.055,
        subplot_titles=["", "", ""],
    )
    # Panel geometry, read back from plotly rather than hardcoded: the
    # section headers, the two dividers and the three x-axis titles all
    # anchor off these so they stay put if the column widths change.
    domains = [tuple(fig.layout[f"xaxis{i or ''}"].domain) for i in (0, 2, 3)]
    centers = [(lo + hi) / 2 for lo, hi in domains]
    dividers = [
        (domains[0][1] + domains[1][0]) / 2,
        (domains[1][1] + domains[2][0]) / 2,
    ]

    # Panel 1 — all-occupation violins.
    for z in zones:
        sub = occ_all[occ_all["job_zone"] == z]
        # A single occupation renders a degenerate sliver, not a
        # distribution — the "n= · med" line still reports it.
        if len(sub) < 2:
            continue
        fig.add_trace(go.Violin(
            x=sub["pct_tasks_affected"], y=[zl[z]] * len(sub), name=zl[z],
            marker_color=ZONE_COLORS[z], line_color=ZONE_COLORS[z],
            fillcolor=ZONE_COLORS[z], opacity=0.75,
            box_visible=False, meanline_visible=False,
            orientation="h", side="positive", width=0.7, points=False,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    for r in zone_stats:
        if not r["n_occs_all"]:
            continue
        fig.add_annotation(
            x=99, y=zl[r["job_zone"]], xref="x", yref="y",
            text=f"n={r['n_occs_all']} · med {r['median_pct_all']:.0f}%",
            showarrow=False, xanchor="right", yanchor="top", yshift=-2,
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

    # Panel 2 — phys-mix stacked bar.
    bar_y = [zl[r["job_zone"]] for r in zone_stats]
    for key, tier in [("pct_physical", "Physical"), ("pct_mixed", "Mixed"),
                      ("pct_non_physical", "Non-physical")]:
        fig.add_trace(go.Bar(
            x=[r[key] for r in zone_stats], y=bar_y, orientation="h",
            marker=dict(color=GROUP_COLORS[tier], line=dict(width=0)),
            name=f"% {tier} occs", showlegend=False, hoverinfo="skip",
        ), row=1, col=2)

    # Legend dummies (uniform swatch size). Labels are trimmed of their
    # "% … occs" wrapper so all four entries fit one row — the panel they
    # describe is already titled "Phys Mix" over a "% of Zone" axis.
    for tier_name, tier_color in (
        ("Non-physical occs", GROUP_COLORS["Non-physical"]),
        ("Mixed occs", GROUP_COLORS["Mixed"]),
        ("Physical occs", GROUP_COLORS["Physical"]),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", color=tier_color, size=16,
                        line=dict(width=0)),
            name=tier_name, showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="line-ns", color="#1a1a1a", size=22,
                    line=dict(color="#1a1a1a", width=3)),
        name="Median", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    # Panel 3 — non-physical violins (Zone 1 kept blank via dummy scatter).
    fig.add_trace(go.Scatter(
        x=[None] * len(zones), y=y_labels, mode="markers",
        marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=3)
    for z in zones:
        sub = occ_nonphys[occ_nonphys["job_zone"] == z]
        # A single occupation renders a degenerate sliver, not a
        # distribution — the "n= · med" line still reports it.
        if len(sub) < 2:
            continue
        fig.add_trace(go.Violin(
            x=sub["pct_tasks_affected"], y=[zl[z]] * len(sub), name=zl[z],
            marker_color=ZONE_COLORS[z], line_color=ZONE_COLORS[z],
            fillcolor=ZONE_COLORS[z], opacity=0.75,
            box_visible=False, meanline_visible=False,
            orientation="h", side="positive", width=0.7, points=False,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=3)
    for r in zone_stats:
        if not r["n_occs_nonphys"]:
            continue
        fig.add_annotation(
            x=99, y=zl[r["job_zone"]], xref="x3", yref="y3",
            text=f"n={r['n_occs_nonphys']} · med {r['median_pct_nonphys']:.0f}%",
            showarrow=False, xanchor="right", yanchor="top", yshift=-2,
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

    # Median lines. Only where a violin was actually drawn: a one-occupation
    # cell renders no curve, so a tick there reads as a stray mark rather
    # than as the median of anything. The "n= · med" annotation still
    # reports it. (Zone 1 non-physical is n=1 under the work-time cuts.)
    def _add_median_lines(stat_key: str, xref: str, yref: str,
                          count_key: str) -> None:
        for r in zone_stats:
            med = r[stat_key]
            if med is None or r[count_key] < 2:
                continue
            pos = y_order_top_down.index(zl[r["job_zone"]])
            fig.add_shape(
                type="line", xref=xref, yref=yref, x0=med, x1=med,
                y0=pos - 0.02, y1=pos + 0.40,
                line=dict(color="#1a1a1a", width=2), layer="above",
            )

    _add_median_lines("median_pct_all", "x", "y", "n_occs_all")
    _add_median_lines("median_pct_nonphys", "x3", "y3", "n_occs_nonphys")

    fig.update_traces(width=0.55, selector=dict(type="bar"))

    # Axes.
    # No y-axis title: "Zone 1…5" is already the only thing the rows could be,
    # and a rotated title claims a fifth of the left margin to say it again.
    # Dropping it is what lets MARGIN_L come down to the width of the ticks,
    # which re-centres the three panels on the canvas.
    fig.update_yaxes(
        categoryorder="array", categoryarray=y_order_top_down,
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        row=1, col=1,
    )
    for c in (2, 3):
        fig.update_yaxes(
            categoryorder="array", categoryarray=y_order_top_down,
            showgrid=False, showline=False, showticklabels=False,
            row=1, col=c,
        )
    # Each panel drops the tick that would land on a shared boundary — a
    # right-edge "100%" and its neighbour's left-edge "0%" merge across the
    # gap into "100%0%". Gridlines still mark every 50.
    # The three x-axis titles are drawn as paper-coordinate annotations
    # further down instead of per-axis titles. Plotly offsets an axis title
    # by that axis's own tick-label height, so the phys-mix panel (floor-size
    # ticks) would sit on a different baseline from its neighbours — and two
    # adjacent auto-centered titles collide. Explicit annotations put all
    # three on one line at positions we control.
    fig.update_xaxes(
        range=[0, 100], tickmode="array",
        tickvals=[0, 50],
        ticktext=["0%", "50%"],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY), tickangle=0,
        row=1, col=1,
    )
    fig.update_xaxes(
        range=[0, 100], tickmode="array", tickvals=[0, 100],
        ticktext=["0%", "100%"],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"], ticksuffix="%",
        tickfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY),
        tickangle=0,
        row=1, col=2,
    )
    fig.update_xaxes(
        range=[0, 100], tickmode="array",
        tickvals=[50, 100],
        ticktext=["50%", "100%"],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY), tickangle=0,
        row=1, col=3,
    )

    style_paper_figure(
        fig,
        "Work Time Exposed by Job Zone with Phys Mix and Actual AI Usage",
        height=H, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    fig.update_layout(title=dict(y=0.965, yanchor="top"))

    # Section headers — n= on its own line: the widened right margin narrows
    # the panels, so single-line "Header (n=…)" text would collide. The
    # phys-mix panel has no n, but it still gets a blank second line: the
    # block is bottom-anchored, so a one-line header would drop its bold
    # text a full line below the other two.
    for title_text, sub_n, x_paper in [
        ("All Occupations", f"n={n_all}", centers[0]),
        ("Phys Mix", "", centers[1]),
        ("Non-Physical", f"n={n_nonphys}", centers[2]),
    ]:
        sub_text = f"({sub_n})" if sub_n else "&nbsp;"
        text = (
            f"<b>{title_text}</b>"
            f"<br><span style='font-size:{px['in_chart_floor']}px;"
            f"color:{PAPER_PALETTE['neutral']}'>{sub_text}</span>"
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=x_paper, y=1.02, text=text,
            showarrow=False, xanchor="center", yanchor="bottom",
            font=dict(size=px["panel_title"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
            align="center",
        )

    # Panel separators — matched pair bracketing the phys-mix panel.
    for x_div in dividers:
        fig.add_shape(
            type="line", xref="paper", yref="paper",
            x0=x_div, x1=x_div, y0=0.0, y1=1.11,
            line=dict(color="#1a1a1a", width=2),
        )

    # x-axis titles, all on one baseline (see the update_xaxes note above).
    for x_paper, label in [
        (centers[0], "% Work Time Exposed"),
        (centers[1], "% of Zone"),
        (centers[2], "% Work Time Exposed"),
    ]:
        fig.add_annotation(
            xref="paper", yref="paper", x=x_paper, y=-0.135,
            xanchor="center", yanchor="top", showarrow=False, text=label,
            font=dict(size=px["tick"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )

    # ── NEW: per-zone usage number columns in the right margin ──────────
    usage_by_zone = usage.set_index("job_zone")
    col_specs = [
        (1.025, "lift", "×Med", "{:.2f}x"),
        (1.130, "raw_share", "Σ raw", "{:.1f}%"),
    ]
    for x_pos, col, header, fmt in col_specs:
        fig.add_annotation(
            xref="paper", yref="paper", x=x_pos, y=1.02,
            xanchor="left", yanchor="bottom", showarrow=False,
            text=f"<b>{header}</b>",
            font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )
        for z in zones:
            fig.add_annotation(
                xref="paper", x=x_pos, y=zl[z], yref="y",
                xanchor="left", yanchor="middle", yshift=8, showarrow=False,
                text=fmt.format(float(usage_by_zone.loc[z, col])),
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
    # No footnote: the job-zone axis title already names the preparation
    # scale, and the Σ columns are self-evidently shares.
    # One row. Plotly wraps a horizontal legend against the PLOT width, not
    # the container, so the four entries have to fit W − margins; that is why
    # the labels are trimmed rather than the entrywidth pinned.
    fig.update_layout(
        barmode="stack",
        legend=dict(
            orientation="h", xref="container", yref="container",
            x=0.5, xanchor="center", y=0.015, yanchor="bottom",
            font=dict(size=px["legend"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
            itemsizing="trace", traceorder="normal",
        ),
    )

    name = "job_zone_usage.png"
    save_figure(fig, results / "figures" / name)
    _copy_fig(results, figures, name)
