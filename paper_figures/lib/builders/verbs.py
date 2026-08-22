"""Verb-family (DWA bucket) charts.

1. build_verb_family_overview — one figure per (config × task universe):
   exposure distribution over the family's DWAs (violin) · mean auto-aug
   across all vs exposed task rows · phys/mixed/non-phys DWA composition
   (all-tasks universe only) · family usage intensity (×median family).
2. build_verb_family_exemplars — one row per major occupational category: the
   verb family it treats most unlike the economy, and the O*NET detailed work
   activity most responsible for that. It answers why the eight families do
   not rank the same way inside every major, which neither a rank heatmap nor
   a matrix of levels can do — the reason is always which specific work sits
   under the family label, so every row has to name it.

DWA exposure = Σ freq×(auto-aug/5) over exposed task rows ÷ Σ freq over all
its task rows (the dashboard ratio-of-totals; auto-aug on the numerator
only).

Usage is a ratio of sums — Σ debiased pct ÷ Σ (freq×emp) — with each rated
(task, occ) pair's usage AND economic weight split /n across its distinct
DWAs, so a task counted in three DWAs isn't counted three times. Family
values are anchored so the lower-middle family reads exactly 1.00×. The
usage numerator is the paper's AEI-only intensity file in every variant:
the equal 3-source GWA debias assumes a Claude-only numerator, so it can't
be re-pointed at the MCP-bearing ceiling datasets. Only the exposure lens
changes between variants.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import figure_data
from lib.figure_data import (
    FAMILY_COLORS, FAMILY_LABELS,
)
from lib.builders.legend import (
    paper_x_center,
)
from lib.paper_config import (
    FONT_FAMILY,
    GROUP_COLORS,
    PAPER_PALETTE,
    paper_fonts,
    style_paper_figure,
)
from lib.utils import save_csv, save_figure

# One definition of the physical cuts across the folder — the job-zone chart
# buckets occupations on the same thresholds and the same time denominator.
PHYS_LOWER, PHYS_UPPER = figure_data.PHYS_LOWER, figure_data.PHYS_UPPER

# Nested auto-aug bars: pale outer = exposed rows only, saturated inner =
# all rows. The inner bar can never exceed the outer one (unrated rows enter
# the all-rows mean as 0), so the gap between them IS the family's reach.
_OUTER_ALPHA = 0.34
_OUTER_WIDTH, _INNER_WIDTH = 0.74, 0.38

# Where the family-total column sits on panel 1's x axis (which runs past
# 100 to make room for it), and the midpoint of the whole panel — the
# spanning header centres on the latter, the "per DWA" sub-label on the
# violins alone.
_FAM_COL_X = 110.0
_PANEL1_MID = 68.0


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _tint(hex_color: str, alpha: float) -> str:
    """Blend a hex color toward white — a flat tint, so it stays opaque and
    doesn't darken where the inner bar overlaps it."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * (1.0 - alpha))  # noqa: E731
    return f"rgb({mix(r)},{mix(g)},{mix(b)})"


# ─────────────────────────────────────────────────────────────────────────
# Shared data prep
# ─────────────────────────────────────────────────────────────────────────

def _dwa_task_rows(dataset: str, nonphys: bool = False) -> pd.DataFrame:
    """Eco (title_current, task_normalized, dwa_title) rows with family, the
    /n-split hours weight, and the dataset's exposed hours (0 where the
    dataset didn't rate the pair).

    Hours are /n-split across the pair's DWAs, so summing them by family is a
    decomposition of the workday rather than a triple count of tasks that sit
    in several DWAs.
    """
    rows = figure_data.act_exposure_rows(
        dataset, "dwa_title", physical_mode="exclude" if nonphys else "all"
    )
    rows["family"] = figure_data.assign_family(rows["dwa_title"])
    rows["exposed"] = rows["auto_aug_mean"].notna()
    return rows


def _dwa_units(rows: pd.DataFrame, extra_group: list[str] | None = None) -> pd.DataFrame:
    """Aggregate task rows to DWA units: work time exposed as a ratio of
    totals over the DWA's hours, plus the activity's physical share.

    `phys_share` is WORK TIME on physical tasks, not the fraction of task
    rows that carry the flag — the same denominator the exposure number in
    the same row uses, and the same one the stacked charts use for a major's
    or a GWA's physical share. Counting rows would let a handful of
    ten-minute physical tasks outvote the activity's actual day.
    """
    grp_cols = (extra_group or []) + ["dwa_title", "family"]
    rows = rows.assign(
        hours_phys_split=np.where(rows["physical"], rows["hours_split"], 0.0)
    )
    units = (
        rows.groupby(grp_cols)
        .agg(
            hours=("hours_split", "sum"),
            hours_exposed=("hours_exposed_split", "sum"),
            hours_phys=("hours_phys_split", "sum"),
            n_rows=("hours_split", "size"),
        )
        .reset_index()
    )
    units["phys_share"] = np.where(
        units["hours"] > 0, units["hours_phys"] / units["hours"], 0.0
    )
    units["exposure"] = np.where(
        units["hours"] > 0, units["hours_exposed"] / units["hours"] * 100.0, 0.0
    ).clip(0.0, 100.0)
    return units


def _usage_rows(nonphys: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(numerator rows, denominator rows) at DWA grain, family-tagged."""
    num = figure_data.intensity_act_rows("dwa_title")
    den = figure_data.eco_act_split_rows("dwa_title")
    if nonphys:
        num = num[~num["physical"]].copy()
        den = den[~den["physical"]].copy()
    num["family"] = figure_data.assign_family(num["dwa_title"])
    den["family"] = figure_data.assign_family(den["dwa_title"])
    return num, den


def _family_usage(nonphys: bool) -> pd.Series:
    """Per-family usage rate, anchored so the lower-middle family = 1.00×."""
    num, den = _usage_rows(nonphys)
    return figure_data.anchor_lower_median(figure_data.usage_rate(num, den, ["family"]))


def _family_autoaug(rows: pd.DataFrame) -> pd.DataFrame:
    """Per family: the plain mean auto-aug over the family's task rows, and
    over its rated task rows only, plus the family's work time exposed.

    The two automation-level bars are unweighted means of the 0–5 score —
    what a reader expects "automation level" to be. `aug_all` counts unrated
    rows as 0, so the gap to `aug_exposed` is the family's reach.

    `pct_exposed` is a different quantity and stays hours-weighted: it is the
    ratio of totals the major and GWA charts report, cut by verb family, and
    is what the chart ranks on. The two used to be arithmetically linked
    (aug_all was 5 × the exposed share); they are now computed separately.
    """
    grouped = rows.groupby("family")
    out = grouped.agg(
        hours=("hours_split", "sum"),
        hours_exposed=("hours_exposed_split", "sum"),
        n_rows=("hours_split", "size"),
        aug_all=("auto_aug_mean", lambda s: float(s.fillna(0.0).mean())),
    )
    rated = rows[rows["exposed"]].groupby("family").agg(
        hours_rated=("hours_split", "sum"),
        aug_exposed=("auto_aug_mean", "mean"),
    )
    out = out.join(rated).fillna(0.0)
    # Reach = the share of the family's work time the dataset rated at all.
    out["reach_pct"] = np.where(
        out["hours"] > 0, out["hours_rated"] / out["hours"] * 100.0, 0.0
    )
    out["pct_exposed"] = np.where(
        out["hours"] > 0, out["hours_exposed"] / out["hours"] * 100.0, 0.0
    )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Chart 1 — verb family overview
# ─────────────────────────────────────────────────────────────────────────

def _build_verb_family_overview(
    results: Path,
    figures: Path,
    config_key: str,
    nonphys: bool,
) -> None:
    dataset = figure_data.CONFIG_DATASETS[config_key]
    rows = _dwa_task_rows(dataset, nonphys=nonphys)
    units = _dwa_units(rows)
    min_units = 1400 if nonphys else 2000
    assert units["dwa_title"].nunique() >= min_units, (
        f"Only {units['dwa_title'].nunique()} DWA units "
        f"(nonphys={nonphys}) — expected ≥ {min_units}"
    )

    aug = _family_autoaug(rows)
    fam_usage = _family_usage(nonphys)
    units["phys_tier"] = figure_data.phys_tier(units["phys_share"] * 100.0)
    tier_mix = (
        units.groupby(["family", "phys_tier"], observed=False)["dwa_title"].count()
        .unstack(fill_value=0)
    )
    tier_mix = tier_mix.div(tier_mix.sum(axis=1), axis=0) * 100.0

    # Ranked by the family's work time exposed — the number printed beside
    # the violin panel, and the same statistic the major and GWA charts rank
    # on. (It is aug_all rescaled, so the ordering matches what this chart
    # used before; the difference is that the ranking key is now on the page.)
    fam_order = aug["pct_exposed"].sort_values(ascending=False).index.tolist()
    labels = [FAMILY_LABELS[f] for f in fam_order]
    # Top-down display order via categoryarray (NOT autorange="reversed" —
    # a reversed axis flips side="positive" violins downward).
    y_order_bottom_up = list(reversed(labels))

    show_phys = not nonphys
    n_cols = 4 if show_phys else 3
    # Panel 1 carries the family-total number column in its axis tail, so it
    # takes a wider share than it did when it was violins alone.
    col_widths = [0.38, 0.21, 0.21, 0.20] if show_phys else [0.46, 0.27, 0.27]
    # Panel titles are centred on their own column, so they have to fit it —
    # the first column is the narrowest once the family labels are drawn.
    # Panel 1's is blank here and drawn as a data-anchored annotation instead:
    # centring it on the column would put it over the family-total number
    # column that now shares the panel.
    titles = ["", "Automation Level<br>(out of 5)"]
    if show_phys:
        titles.append("Phys Makeup")
    titles.append("Usage (×Med)")

    W = 2000 if show_phys else 1700
    px = paper_fonts(W)
    fig = make_subplots(
        rows=1, cols=n_cols, shared_yaxes=True,
        column_widths=col_widths, horizontal_spacing=0.05,
        subplot_titles=titles,
    )
    col_aug = 2
    col_phys = 3 if show_phys else None
    col_use = n_cols

    # Panel 1 — one-sided violins of DWA exposure per family.
    for f in fam_order:
        vals = units.loc[units["family"] == f, "exposure"]
        fig.add_trace(go.Violin(
            x=vals, y=[FAMILY_LABELS[f]] * len(vals),
            orientation="h", side="positive", width=1.55,
            points=False, hoverinfo="skip",
            line=dict(color=FAMILY_COLORS[f], width=1.2),
            fillcolor=FAMILY_COLORS[f], opacity=0.65,
            meanline_visible=False, showlegend=False,
        ), row=1, col=1)
        med = float(vals.median())
        pos = y_order_bottom_up.index(FAMILY_LABELS[f])
        fig.add_shape(
            type="line", x0=med, x1=med, y0=pos - 0.02, y1=pos + 0.42,
            line=dict(color="#222222", width=2), row=1, col=1,
        )

    # Family totals in panel 1's axis tail. The violins are a distribution
    # over DWAs; this is the one number for the family — every task row under
    # its DWAs pooled — so the reader can see that a family of individually
    # middling DWAs can still hold a large share of exposed work, and vice
    # versa. It is also the ranking key for the whole chart.
    for f in fam_order:
        fig.add_annotation(
            xref="x", yref="y", x=_FAM_COL_X, y=FAMILY_LABELS[f],
            xanchor="left", yanchor="middle", showarrow=False,
            text=f"<b>{aug.loc[f, 'pct_exposed']:.0f}%</b>",
            font=dict(size=px["tick"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )
    # Panel 1 gets a two-lane header instead of one title: the violins and
    # the number column are the SAME quantity at two grains, so one title
    # spans the panel and a sub-label names each half. Labelling the number
    # column on its own read as a fourth metric rather than the total of the
    # third.
    fig.add_annotation(
        xref="x", yref="paper", x=_PANEL1_MID, y=1.0, yshift=17 + px["panel_title"],
        xanchor="center", yanchor="bottom", showarrow=False,
        text="Work Time Exposed",
        font=dict(size=px["panel_title"], family=FONT_FAMILY,
                  color=PAPER_PALETTE["text"]),
    )
    # "family total" is two stacked words rather than one: on one line it ran
    # into the neighbouring "Automation Level" panel title. The first word
    # stays on the sub-label line shared with "per DWA" and the second drops
    # below it, into the empty band above the first violin — a second line
    # placed the usual way (a <br>) would instead push "family" UP off that
    # shared line, which is the thing that has to stay put.
    line_h = int(px["in_chart_floor"] * 1.25)
    for x_pos, anchor, shift, sub in [(42.0, "center", 17, "per DWA"),
                                      (_FAM_COL_X, "left", 17, "family"),
                                      (_FAM_COL_X, "left", 17 - line_h, "total")]:
        fig.add_annotation(
            xref="x", yref="paper", x=x_pos, y=1.0, yshift=shift,
            xanchor=anchor, yanchor="bottom", showarrow=False, text=sub,
            font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["muted"]),
        )

    # Panel 2 — nested auto-aug bars. Outer (pale) = exposed rows only,
    # inner (saturated) = all rows. Drawn as overlay, not stack.
    fig.add_trace(go.Bar(
        x=[aug.loc[f, "aug_exposed"] for f in fam_order], y=labels,
        orientation="h", width=_OUTER_WIDTH,
        marker=dict(color=[_tint(FAMILY_COLORS[f], _OUTER_ALPHA) for f in fam_order]),
        text=[f"{aug.loc[f, 'aug_exposed']:.1f}" for f in fam_order],
        textposition="outside",
        textfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["neutral"]),
        cliponaxis=False, showlegend=False, hoverinfo="skip",
    ), row=1, col=col_aug)
    # A bar shorter than its own label makes plotly rotate the text vertical,
    # which drops it under the print floor. Values below ~1.0 sit outside the
    # inner bar instead, on the pale bar behind it.
    inner_vals = [float(aug.loc[f, "aug_all"]) for f in fam_order]
    fig.add_trace(go.Bar(
        x=inner_vals, y=labels,
        orientation="h", width=_INNER_WIDTH,
        marker=dict(color=[FAMILY_COLORS[f] for f in fam_order]),
        text=[f"<b>{v:.1f}</b>" for v in inner_vals],
        textposition=["inside" if v >= 1.0 else "outside" for v in inner_vals],
        insidetextanchor="end",
        insidetextfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                            color="#ffffff"),
        outsidetextfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                             color=PAPER_PALETTE["text"]),
        constraintext="none",
        cliponaxis=False, showlegend=False, hoverinfo="skip",
    ), row=1, col=col_aug)

    # Panel 3 — phys tier composition. Stacked by explicit base so the
    # figure's global barmode can stay "overlay" for the nested bars.
    if show_phys:
        base = np.zeros(len(fam_order))
        for tier in ["Non-physical", "Mixed", "Physical"]:
            vals = np.array([
                float(tier_mix.loc[f, tier]) if f in tier_mix.index else 0.0
                for f in fam_order
            ])
            fig.add_trace(go.Bar(
                x=vals, y=labels, base=base.tolist(), orientation="h",
                width=_OUTER_WIDTH,
                name=f"{tier} DWAs",
                marker=dict(color=GROUP_COLORS[tier]),
                text=[f"{v:.0f}%" if v >= 12 else "" for v in vals],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                              color="#ffffff"),
                showlegend=True, hoverinfo="skip",
            ), row=1, col=col_phys)
            base = base + vals

        # The violin panel's median tick has no trace of its own, so it needs
        # a dummy to appear in the legend alongside the phys tiers.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="line-ns", color="#222222", size=16,
                        line=dict(color="#222222", width=2)),
            name="Median DWA", showlegend=True, hoverinfo="skip",
        ), row=1, col=1)

    # Panel 4 — family usage intensity.
    use_vals = [float(fam_usage[f]) for f in fam_order]
    fig.add_trace(go.Bar(
        x=use_vals, y=labels, orientation="h", width=_OUTER_WIDTH,
        marker=dict(color=[FAMILY_COLORS[f] for f in fam_order]),
        text=[f"{v:.2f}x" for v in use_vals],
        textposition="outside",
        textfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY),
        cliponaxis=False, showlegend=False, hoverinfo="skip",
    ), row=1, col=col_use)

    for c in range(1, n_cols + 1):
        fig.update_yaxes(categoryorder="array", categoryarray=y_order_bottom_up,
                         row=1, col=c)
    # Range runs past 100 so the family-total column has somewhere to sit;
    # the violins still occupy the 0–100 stretch the ticks label. Three ticks,
    # unrotated: at five ticks plotly turned them vertical, which then fired
    # automargin and shifted the whole paper frame up under the footnote.
    fig.update_xaxes(range=[0, 148], tickvals=[0, 50, 100],
                     ticktext=["0%", "50%", "100%"], tickangle=0,
                     row=1, col=1)
    fig.update_xaxes(range=[0, 5.6], tickvals=[0, 1, 2, 3, 4, 5],
                     row=1, col=col_aug)
    if show_phys:
        # Drop the boundary "100" tick so adjacent panels' labels don't merge.
        fig.update_xaxes(range=[0, 100.2], tickvals=[0, 50],
                         ticktext=["0%", "50%"], row=1, col=col_phys)
    fig.update_xaxes(range=[0, max(use_vals) * 1.28], row=1, col=col_use)
    # Deterministic paper frame: with automargin on, a tall tick label grows
    # the bottom margin and every paper-referenced offset below the plot
    # silently moves with it.
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False)
    fig.update_layout(barmode="overlay", bargap=0.30)

    # Top margin carries three lanes: the two-line title, then the panel
    # titles, then the sub-labels. At 235 the panel titles sat hard against
    # the title's second line; the extra 60 px is the gap between them.
    MARGIN_T, MARGIN_B = 295, 190
    MARGIN_L, MARGIN_R = 610, 70
    H = 8 * 92 + MARGIN_T + MARGIN_B
    scope = "Non-Physical Tasks" if nonphys else "All Tasks"
    # Two lines: at a fixed print pt a title's share of the canvas is
    # width-invariant, so a 100-character title cannot be rescued by widening
    # the figure — it has to wrap or lose words. The break sits as late as it
    # can: measured at the title size, line 1 is 1898 px of the 1980 px the
    # canvas gives it, so "Phys Makeup" is the last phrase that fits.
    style_paper_figure(
        fig,
        "AI Exposure by Verb Family with Work Time Exposed, Automation Level,"
        f" Phys Makeup<br>and Actual AI Usage, {scope}",
        width=W, height=H,
        # Left margin set explicitly rather than left to automargin: the
        # one-line footnote below is centred on the CANVAS, which needs the
        # plot-area origin to be a known number.
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    # Only the subplot titles — the family-total header and column values are
    # annotations too, and must keep their own (smaller) sizes.
    for ann in fig.layout.annotations[:n_cols]:
        ann.font = dict(size=px["panel_title"], family=FONT_FAMILY,
                        color=PAPER_PALETTE["text"])
    # A two-line title anchored "top" still hangs its first line ABOVE the
    # anchor by about 0.78 of the font size — at the 0.985 the one-line
    # charts use, line 1 runs off the canvas. Solve for the y that puts the
    # block's top edge 18 px inside the container.
    fig.update_layout(title=dict(
        y=1 - (18 + 0.78 * px["title"]) / H, yanchor="top",
    ))
    # Three stacked lanes below the plot, measured in px from the plot's
    # bottom edge: tick labels (~48), then the phys-tier legend, then the
    # nested-bar key. Both keys used to be positioned independently against
    # different frames and landed on top of each other.
    plot_h = H - MARGIN_T - MARGIN_B
    LEGEND_BELOW, FOOTNOTE_BELOW = 105, 125
    fig.update_layout(legend=dict(
        orientation="h", xref="container", yref="container",
        x=0.5, xanchor="center", y=(MARGIN_B - LEGEND_BELOW) / H,
        yanchor="bottom", traceorder="normal",
    ))
    # Key for the nested bars — without it the two numbers read as one metric.
    fig.add_annotation(
        xref="paper", yref="paper",
        x=paper_x_center(W, MARGIN_L, MARGIN_R),
        y=-FOOTNOTE_BELOW / plot_h,
        xanchor="center", yanchor="top", showarrow=False,
        text=("Automation level: pale = exposed tasks average, "
              "solid = all tasks average (unreached tasks = 0)"),
        font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                  color=PAPER_PALETTE["neutral"]),
    )

    suffix = "_nonphys" if nonphys else ""
    name = f"verb_family_{config_key}{suffix}.png"
    save_figure(fig, results / "figures" / name, scale=2)

    stats = pd.DataFrame({
        "family": fam_order,
        "label": labels,
        "pct_work_time_exposed": [aug.loc[f, "pct_exposed"] for f in fam_order],
        "mean_autoaug_all": [aug.loc[f, "aug_all"] for f in fam_order],
        "mean_autoaug_exposed": [aug.loc[f, "aug_exposed"] for f in fam_order],
        "reach_pct": [aug.loc[f, "reach_pct"] for f in fam_order],
        "usage_x": use_vals,
        "median_dwa_exposure": [
            float(units.loc[units["family"] == f, "exposure"].median())
            for f in fam_order
        ],
        **{
            f"tier_{t.lower().replace('-', '')}": [
                float(tier_mix.loc[f, t]) if f in tier_mix.index else 0.0
                for f in fam_order
            ]
            for t in ["Non-physical", "Mixed", "Physical"]
        },
    })
    save_csv(stats, results / f"verb_family_{config_key}{suffix}.csv")
    _copy_fig(results, figures, name)


def build_verb_family_overview(results: Path, figures: Path) -> None:
    """All Confirmed on the full task universe — the main-body figure.

    The other four configs used to be rendered here as variants. They said
    the same thing four more times (only the exposure lens changed; the usage
    panel is identical across them by construction), so the set is cut to the
    primary one.
    """
    _build_verb_family_overview(results, figures, "all_confirmed", nonphys=False)


def build_verb_family_overview_nonphys(results: Path, figures: Path) -> None:
    """All Confirmed restricted to non-physical tasks on both sides of every
    ratio — the supplemental counterpart. The phys-makeup panel is dropped;
    it would be constant."""
    _build_verb_family_overview(results, figures, "all_confirmed", nonphys=True)


# ─────────────────────────────────────────────────────────────────────────
# Chart 2 — what the work actually is: an exemplar DWA per family
# ─────────────────────────────────────────────────────────────────────────

# Exemplar majors are drawn from the ten most exposed. Unrestricted, the
# "least" side is a physical major in all eight families and the exemplar is
# a physical activity at 0% — eight rows re-telling the phys/non-phys split
# the major stacked chart already carries. Inside the top ten, both sides are
# majors that genuinely do the work, so the row is a real contrast.
TOP_MAJORS_POOL = 10
# The activity column is fitted by WIDTH, not by character count. The font is
# proportional, so a fixed character cap leaves every truncated row ending
# somewhere different — 44 narrow letters are visibly shorter than 44 wide
# ones and the ellipses land all over the column. These per-character em
# widths are approximate but consistent, which is all a flush right edge
# needs: each truncated row is grown until one more character would overrun,
# so they all stop at the same place.
_EM_NARROW = frozenset("iljItfr|.,:;'!`()[]{}/" + chr(92) + "- ")
_EM_WIDE = frozenset("mwMW%@")
_EM_UPPER = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _em(text: str) -> float:
    """Approximate rendered width of `text`, in em."""
    total = 0.0
    for ch in text:
        if ch in _EM_NARROW:
            total += 0.30
        elif ch in _EM_WIDE:
            total += 0.86
        elif ch in _EM_UPPER:
            total += 0.68
        else:
            total += 0.55
    return total


def _fit(name: str, suffix: str, budget_em: float) -> str:
    """Longest prefix of `name` whose width plus `suffix` fits `budget_em`,
    ellipsed if anything was dropped."""
    name = str(name).rstrip(".")
    if _em(name + suffix) <= budget_em:
        return name
    for i in range(len(name), 0, -1):
        candidate = name[:i].rstrip(" ,") + "…"
        if _em(candidate + suffix) <= budget_em:
            return candidate
    return "…"


def _family_major_cells(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame]:
    """Task rows, (major, family) cells, and (major, family, DWA) units — all
    plain ratios of totals over hours, the same statistic every other chart in
    the set reports."""
    rows = _dwa_task_rows(dataset)
    pool = (
        figure_data.major_exposure(dataset)
        .sort_values("pct", ascending=False)["category"].tolist()[:TOP_MAJORS_POOL]
    )

    def _agg(keys: list[str], src: pd.DataFrame) -> pd.DataFrame:
        g = src.groupby(keys).agg(
            hours=("hours_split", "sum"), exposed=("hours_exposed_split", "sum"),
        ).reset_index()
        g["pct"] = g["exposed"] / g["hours"].replace(0.0, np.nan) * 100.0
        return g

    cells = _agg(["major_occ_category", "family"],
                 rows[rows["major_occ_category"].isin(pool)])
    units = _agg(["major_occ_category", "family", "dwa_title"], rows)
    return rows, cells, units


def _exemplar(units: pd.DataFrame, major: str, family: str, cell_pct: float,
              most: bool) -> pd.Series:
    """The DWA that most drives this cell up (most=True) or down.

    A cell's percentage is the hours-weighted mean of its DWAs, so every DWA
    has a contribution `share × (pct − cell)` in percentage points, and those
    contributions sum to exactly zero. The largest positive one is, by
    definition, the activity most responsible for the cell reading high.

    Selecting on contribution rather than on the DWA's own percentage is what
    removes the need for a minimum-size floor: a sliver holding 0.1% of the
    cell contributes 0.03 pp however extreme its rate, so it can never win.
    Ranking on the rate alone surfaced exactly those slivers; ranking on
    exposed hours degenerated on the "least" side, where the smallest activity
    always has the fewest exposed hours; ranking on total hours picked
    activities that contradicted their own row.
    """
    c = units[(units["major_occ_category"] == major)
              & (units["family"] == family)].copy()
    assert not c.empty, f"No DWA units for {major} / {family}"
    c["share"] = c["hours"] / c["hours"].sum()
    c["contrib"] = c["share"] * (c["pct"] - cell_pct)
    assert abs(float(c["contrib"].sum())) < 1e-6, (
        f"Contributions must sum to zero, got {float(c['contrib'].sum()):.3g}"
    )
    return c.sort_values("contrib", ascending=not most).iloc[0]


def _major_standouts(cells: pd.DataFrame, eco: pd.Series,
                     units: pd.DataFrame) -> pd.DataFrame:
    """One row per major: the verb family it is furthest from the economy on,
    among the families that make up the larger half of its working day.

    The eligibility rule exists because a gap in percentage points carries no
    size with it. Legal's Operate family is 0.2% of Legal's working time, and
    that scrap reads 47% against the economy's 11% — a 37-point gap that won
    the row, while Legal's Communicate family, a quarter of the actual job,
    never appeared. Restricting to the families above the major's own median
    share fixes it without a tuned number: the eligible set covers 67-91% of
    each major's day (median 79%).

    Weighting the gap by share instead — the contribution rule used to pick
    the activity inside a cell — does not work at this level. Create is
    27-55% of every major's day, so the size term swamps the gap term and the
    same family wins nine rows out of ten. That rule needs the candidates to
    be of comparable size, which DWAs inside one cell are and verb families
    across a major are not.
    """
    out: list[dict] = []
    for major, grp in cells.groupby("major_occ_category"):
        eligible = grp[grp["share"] >= grp["share"].median()]
        assert not eligible.empty, f"No eligible family for {major}"
        row = eligible.loc[eligible["gap"].abs().idxmax()]
        ex = _exemplar(units, major, row["family"], row["pct"], row["gap"] > 0)
        out.append({
            "major": major,
            "major_short": figure_data.short_major_label(major),
            "family": row["family"],
            "family_short": FAMILY_LABELS[row["family"]].split(" /")[0],
            "pct": float(row["pct"]),
            "eco_pct": float(eco[row["family"]]),
            "gap": float(row["gap"]),
            "family_share": float(row["share"]),
            "eligible_share": float(eligible["share"].sum()),
            "dwa_title": str(ex["dwa_title"]),
            "dwa_pct": float(ex["pct"]),
            "dwa_share_of_cell": float(ex["share"]) * 100.0,
            "contrib_pp": float(ex["contrib"]),
        })
    # Signed, treating the gap as the number it is: the largest positive gap
    # first, down through the smaller ones, then the negatives at the bottom.
    return (pd.DataFrame(out)
            .sort_values("gap", ascending=False)
            .reset_index(drop=True))


def build_verb_family_exemplars(results: Path, figures: Path) -> None:
    """One row per major: the verb family it treats most unlike the economy,
    and the O*NET activity most responsible for that.

    The figure answers why the eight families do not rank the same way inside
    every major. A matrix of levels or of ranks can show THAT they reorder but
    never why, because the reason is always which specific work sits under the
    family label — Legal spends a quarter of its day communicating and the
    largest piece of that, representing clients in proceedings, is 9% reached.
    So every row names the activity.
    """
    dataset = figure_data.PRIMARY_DATASET
    rows, cells, units = _family_major_cells(dataset)
    eco = _family_autoaug(rows)["pct_exposed"]
    cells = cells.copy()
    cells["share"] = (
        cells["hours"]
        / cells.groupby("major_occ_category")["hours"].transform("sum") * 100.0
    )
    cells["gap"] = cells["pct"] - cells["family"].map(eco)
    df = _major_standouts(cells, eco, units)
    save_csv(df, results / "verb_family_exemplars.csv")

    # ── layout ──────────────────────────────────────────────────────────
    # A table, not a chart: the row's payload is two percentages and a named
    # activity, and what has to be readable is the activity names against each
    # other. Column x are paper fractions budgeted in CHARACTERS — at 8 pt
    # this canvas holds ~91 across the plot however wide it is drawn, so the
    # activity column gets what the fixed columns leave it.
    W = 2400
    px = paper_fonts(W)
    # Every number on the figure is work time exposed. The title says so and
    # the header block says it again over the two number columns, because a
    # bare "major / economy" pair reads as a share of something unnamed.
    # Margins are trimmed to the header and the last rule — the columns are
    # packed left so the activity gets the rest.
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 70, 70, 150, 44
    # Two header lines: the spanning "Work Time Exposed" over the number
    # columns, then the per-column labels.
    COLHEAD_PX, ROW_PX = 120, 58
    X_MAJOR, X_FAMILY = 0.004, 0.235
    # The number columns are packed as far left as the family column allows.
    # What limits it is not the family NAME but the colour chip in front of
    # it: "■&nbsp; " costs about eight characters' width, far more than the
    # glyph looks, so "Communicate" plus a chip reaches ~0.415 and the values
    # cannot start before that. Everything saved goes to the activity column,
    # which is the one that truncates.
    X_LVL, X_ECO, X_ACT = 0.428, 0.512, 0.528
    # Width budget for the activity column, in em (see _fit). Calibrated off a
    # render: "Evaluate student work (80%)" is ~13.1 em and measures 0.248
    # paper, so an em is ~0.0189 paper and the 0.472 from X_ACT to the canvas
    # edge holds ~25.
    _ACT_BUDGET_EM = 25.2

    body = dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                color=PAPER_PALETTE["text"])
    muted = dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                 color=PAPER_PALETTE["neutral"])
    head = dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                color=PAPER_PALETTE["muted"])

    fig = go.Figure()
    anns: list[dict] = []

    def _cell(x: float, y: float, text: str, font: dict,
              anchor: str = "left") -> None:
        anns.append(dict(xref="paper", yref="y", x=x, y=y, text=text,
                         xanchor=anchor, yanchor="middle", showarrow=False,
                         font=font, align="left"))

    # Line 1 spans the two number columns; line 2 names every column. The
    # span sits over the midpoint of the pair, not of the gap between their
    # anchors, because both are right-anchored and the values are ~0.03 wide.
    _cell((X_LVL - 0.036 + X_ECO) / 2, COLHEAD_PX * 0.28,
          "<b>Work Time Exposed</b>", head, "center")
    for x, t, a in ((X_MAJOR, "Major category", "left"),
                    (X_FAMILY, "Family", "left"),
                    (X_LVL, "major", "right"),
                    (X_ECO, "economy", "right"),
                    (X_ACT, "DWA most responsible (work time exposed)",
                     "left")):
        _cell(x, COLHEAD_PX * 0.74, f"<b>{t}</b>", head, a)

    cursor = COLHEAD_PX
    for _i, r in df.iterrows():
        y = cursor + ROW_PX / 2
        cursor += ROW_PX
        _cell(X_MAJOR, y, r["major_short"], body)
        _cell(X_FAMILY, y,
              f"<span style='color:{FAMILY_COLORS[r['family']]}'>■</span> "
              f"{r['family_short']}", body)
        # Bold whichever side is higher, so the direction of the gap reads
        # off the row without a signed column.
        hi_major = r["pct"] >= r["eco_pct"]
        _cell(X_LVL, y,
              f"<b>{r['pct']:.0f}%</b>" if hi_major else f"{r['pct']:.0f}%",
              body if hi_major else muted, "right")
        _cell(X_ECO, y,
              f"{r['eco_pct']:.0f}%" if hi_major else f"<b>{r['eco_pct']:.0f}%</b>",
              muted if hi_major else body, "right")
        suffix = f" ({r['dwa_pct']:.0f}%)"
        _cell(X_ACT, y,
              f"{_fit(r['dwa_title'], suffix, _ACT_BUDGET_EM)} "
              f"<span style='color:{PAPER_PALETTE['neutral']}'>"
              f"({r['dwa_pct']:.0f}%)</span>", body)
    total_px = cursor
    height = total_px + MARGIN_T + MARGIN_B

    for ann in anns:
        fig.add_annotation(**ann)
    head_y = COLHEAD_PX
    fig.add_shape(type="line", xref="paper", yref="y", x0=0.0, x1=1.0,
                  y0=head_y, y1=head_y, layer="below",
                  line=dict(color=PAPER_PALETTE["text"], width=2))
    for i in range(1, len(df) + 1):
        y = head_y + i * ROW_PX
        fig.add_shape(type="line", xref="paper", yref="y", x0=0.0, x1=1.0,
                      y0=y, y1=y, layer="below",
                      line=dict(color=PAPER_PALETTE["grid"], width=1))
    # An empty trace so the y axis exists to anchor the rows against.
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(range=[0, 1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[total_px, 0], visible=False, fixedrange=True)

    # Two lines — the whole sentence is 3010 px against the 2376 a one-line
    # title gets — broken at the last word that fits rather than at the
    # clause, so line 1 runs to 1964 px instead of stopping at 1611.
    style_paper_figure(
        fig,
        "Top 10 Most Work Time Exposed Majors with the Verb Family Each Is Most"
        "<br>Differently Exposed Than the Economy",
        width=W, height=height,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    # A "top"-anchored title hangs its first line above the anchor by ~0.78
    # of the font size, so the one-line charts' y=0.985 clips line 1.
    fig.update_layout(
        showlegend=False,
        title=dict(y=1 - (18 + 0.78 * px["title"]) / height, yanchor="top"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    name = "verb_family_exemplars.png"
    save_figure(fig, results / "figures" / name, scale=2)
    _copy_fig(results, figures, name)
