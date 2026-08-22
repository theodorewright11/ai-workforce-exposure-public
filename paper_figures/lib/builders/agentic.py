"""Agentic-AI figure — the fifteen intermediate work activities where MCP
tooling coverage runs furthest ahead of confirmed agentic use, split across
two columns.

Per row: how much of the activity's work time agentic AI already reaches
(AEI API usage), how much of it the MCP catalogue covers, and how much of the
many server ratings stand behind that coverage, and how much of the
catalogue's own attention the activity holds.

Both bars are shares of the same work-time denominator, so they sit on one
axis untouched. The size gap between them is the finding.

Only IWAs. The occupation side was dropped — at broad grain its top ten led
with occupations holding ~0.00-0.04% of economy work hours, and at minor
grain it restated the majors the other charts already carry. The rating-count
column went with it; MCP Share is the evidence column that survives, and it
is where the catalogue POINTS, not observed usage. Counts stay in the CSV.

The tooling side is the standalone MCP Cumul. v4 snapshot rather than the
MCP + API composite. The composite AVERAGES the two sources per task, and
MCP scores below API on 3,561 of their 3,685 shared (task, occ) pairs, so a
composite "ceiling" came out BELOW confirmed usage for five majors and any
headroom segment clipped to zero — which read as "no room to grow" when what
had actually happened was that the average pulled the number down. The
standalone catalogue says the true thing: MCP reaches 5,998 pairs the API
data never touched, and on the pairs they share it is the more conservative
rater.
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
    square_legend_html,
)
from lib.paper_config import (
    CONFIG_COLORS,
    FONT_FAMILY,
    PAPER_PALETTE,
    paper_fonts,
    style_paper_figure,
)
from lib.utils import save_csv, save_figure

CONF_COLOR = CONFIG_COLORS["agentic_confirmed"]
MCP_COLOR = "#8fb4c9"

# Rating columns in the MCP pipeline output: how many matched servers scored
# the task at each level. Level 1 means "not automatable", so the count of
# ratings that say anything at all is 2–5.
RATING_COLS = ["n_rating_2", "n_rating_3", "n_rating_4", "n_rating_5"]


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _fmt_count(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


# ─────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────

def _mcp_rating_counts(group_col: str) -> pd.DataFrame:
    """Per group: total server ratings of 2–5, and the counts at 4 and 5.

    Summed at (task, occupation) pair grain — the same grain the exposure
    numbers run on — so a task that sits in twenty occupations contributes
    its ratings to each of them, exactly as it contributes its hours.
    """
    meta = figure_data.backend_config.DATASETS[figure_data.MCP_DATASET]
    usecols = ["title_current", "task_normalized", group_col, *RATING_COLS]
    df = pd.read_csv(meta["file"], usecols=usecols, low_memory=False)
    pairs = (
        df.dropna(subset=[group_col])
        .groupby(["title_current", "task_normalized", group_col], sort=False)
        .first()
        .reset_index()
    )
    for c in RATING_COLS:
        pairs[c] = pd.to_numeric(pairs[c], errors="coerce").fillna(0.0)
    out = pairs.groupby(group_col)[RATING_COLS].sum()
    out["n_rated"] = out[RATING_COLS].sum(axis=1)
    return out.rename(columns={"n_rating_4": "n_4", "n_rating_5": "n_5"})


def _mcp_pct_share(group_col: str, is_activity: bool) -> pd.Series:
    """Each category's share of the MCP catalogue's own attention, in percent.

    `pct_normalized` in the MCP file is already a proper share — it sums to
    exactly 100 over the 9,683 unique (task, occupation) pairs the catalogue
    matched — so rolling it up needs no renormalising, only the right grain.
    An occupation pair belongs to exactly one occupation group, so those sum
    to 100 directly. A pair sits in several activities, so at activity grain
    the share is `/n`-split across the pair's distinct activities first, the
    same split every other activity-grain number in this folder uses; without
    it the same attention is counted two or three times.
    """
    mcp = figure_data.mcp_pairs()[["title_current", "task_normalized", "pct_normalized"]]
    if is_activity:
        key = (figure_data.eco_pairs((group_col,))
               [["title_current", "task_normalized", group_col]].dropna())
        n = key.groupby(["title_current", "task_normalized"])[group_col].transform(
            "nunique")
        joined = key.assign(w=1.0 / n).merge(
            mcp, on=["title_current", "task_normalized"], how="inner")
        share = (joined["pct_normalized"] * joined["w"]).groupby(
            joined[group_col]).sum()
    else:
        key = (figure_data.eco_pairs((group_col,))[["title_current", group_col]]
               .drop_duplicates().dropna())
        share = (mcp.merge(key, on="title_current", how="inner")
                 .groupby(group_col)["pct_normalized"].sum())
    total = float(share.sum())
    assert total > 0, f"No MCP attention mapped to {group_col}"
    return (share / total * 100.0).rename("mcp_pct_share")


def _occ_group_exposure(dataset: str, group_col: str) -> pd.Series:
    """Work time exposed per occupation group, employment-weighted.

    The same rollup figure_data.major_exposure does, for any occupation-level
    grouping: the dashboard pipeline is asked only for per-occupation
    percentages and the weighting is done here, because under time_day a
    plain group ratio-of-totals is an unweighted mean over occupations.
    """
    occ = figure_data.occ_exposure(dataset)
    hier = (
        figure_data.eco_pairs((group_col,))
        .drop_duplicates("title_current")
        .set_index("title_current")[group_col]
    )
    occ = occ.assign(grp=occ["title_current"].map(hier)).dropna(subset=["grp"])
    g = (
        occ.assign(
            hours=lambda d: d["emp"] * figure_data.OCC_DAY_HOURS,
            hours_exposed=lambda d: d["p"] * d["emp"] * figure_data.OCC_DAY_HOURS,
        )
        .groupby("grp")
        .agg(hours=("hours", "sum"), hours_exposed=("hours_exposed", "sum"))
    )
    return (g["hours_exposed"] / g["hours"].replace(0.0, np.nan) * 100.0).rename("pct")


def _level_frame(group_col: str, is_activity: bool) -> pd.DataFrame:
    """Per category at one level: confirmed agentic and MCP tooling work time
    exposed, the MCP rating counts, and the category's share of the MCP
    catalogue's attention. Ranked by MCP coverage."""
    if is_activity:
        conf = figure_data.act_exposure(
            figure_data.AGENTIC_DATASET, group_col).set_index("category")["pct"]
        tool = figure_data.act_exposure(
            figure_data.MCP_DATASET, group_col).set_index("category")["pct"]
    else:
        conf = _occ_group_exposure(figure_data.AGENTIC_DATASET, group_col)
        tool = _occ_group_exposure(figure_data.MCP_DATASET, group_col)

    df = pd.DataFrame({"pct_conf": conf, "pct_mcp": tool}).dropna(subset=["pct_mcp"])
    counts = _mcp_rating_counts(group_col)
    df = df.join(counts[["n_4", "n_5", "n_rated"]], how="left").fillna(0.0)
    df["n_high"] = df["n_4"] + df["n_5"]
    df = df.join(_mcp_pct_share(group_col, is_activity), how="left").fillna(
        {"mcp_pct_share": 0.0})
    df.index.name = "category"
    # Ranked by how far tooling coverage runs AHEAD of confirmed use, not by
    # coverage alone. Coverage alone puts activities at the top that agentic
    # AI has already reached — a 97% covered / 79% used row is mostly a
    # statement that both are high. The gap is the headroom the figure is
    # about, so it is what the ranking should be on.
    df["gap"] = df["pct_mcp"] - df["pct_conf"]
    return (
        df.reset_index()
        .sort_values("gap", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────
# The agentic figure — top 15 intermediate work activities, split two-column
# ─────────────────────────────────────────────────────────────────────────

# One level, not two. The occupation side was dropped: at broad grain its top
# ten led with occupations holding ~0.00-0.04% of economy work hours, and at
# minor grain it restated the majors the other charts already carry. IWA is
# the level that earns its place — 331 categories, and its unguarded top 15
# all rest on thousands of server ratings, so the ranking needs no gate.
LEVEL_COL, LEVEL_IS_ACTIVITY = "iwa_title", True
TOP_N = 15
# 8 then 7: an odd count puts the extra row on the left, matching the split
# stacked charts in occupation.py.
_SPLITS = ((0, 8), (8, TOP_N))

SERIES = [
    ("pct_conf", "Confirmed agentic use", CONF_COLOR),
    ("pct_mcp", "MCP agentic tooling", MCP_COLOR),
]

# Nested bars, not grouped: the MCP bar is the full width and the confirmed
# bar sits inside it, so each activity is ONE bar and the pale remainder IS
# the gap the figure ranks on. Every row here has MCP above confirmed by
# construction, so the inner bar can never overrun the outer one.
_OUTER_W, _INNER_W = 0.74, 0.40

# Both halves share one 0-100% axis so the numbers are comparable across the
# figure; the tail past 100 holds the evidence column.
#
# Only the CONFIRMED number prints outside its own bar. The tooling number
# rides a fixed right-anchored column past the longest tooling bar instead:
# a label is ~41 axis units wide here, so a confirmed bar at 28% ends its
# label at 69 — right where a tooling label at 72% would start. Printing
# both outside their own bars put those two on top of each other on five of
# the fifteen rows. A column also lines the tooling numbers up with each
# other, which is the comparison the figure is ranked on.
_MCP_COL_X = 128.0
_SHARE_X = 134.0
_X_AXIS_MAX = 180.0
_LABEL_LINE = 24          # chars per line of a wrapped two-line y label


def _wrap(label: str) -> str:
    """Two-line y label. Every row carries two bars, so a second line of text
    costs nothing vertically — which is why these are wrapped rather than
    truncated."""
    label = str(label).replace(" Occupations", "").rstrip(".")
    if len(label) <= _LABEL_LINE:
        return label
    cut = label.rfind(" ", 0, _LABEL_LINE + 6)
    if cut <= 12:
        return label[:_LABEL_LINE * 2 - 1] + "…"
    head, tail = label[:cut], label[cut + 1:]
    if len(tail) > _LABEL_LINE + 6:
        tail = tail[:_LABEL_LINE + 5].rstrip(" ,") + "…"
    return f"{head}<br>{tail}"


def build_agentic_tooling(results: Path, figures: Path) -> None:
    """Confirmed agentic use against the MCP tool catalogue, over the fifteen
    intermediate work activities where the catalogue runs furthest ahead of
    confirmed use.

    Both bars are shares of the same work-time denominator, so the two halves
    sit on one axis and every number is comparable with every other. MCP Share
    is the one evidence column: where the catalogue's own attention points.
    """
    frame = _level_frame(LEVEL_COL, LEVEL_IS_ACTIVITY)
    save_csv(frame.assign(level=LEVEL_COL), results / "agentic_tooling.csv")
    top = frame.head(TOP_N).reset_index(drop=True)

    W = 2400
    px = paper_fonts(W)
    # Right margin carries the right half's evidence column, which sits
    # past the panel's own edge — at 90 the column header clipped.
    MARGIN_L, MARGIN_R = 620, 130
    MARGIN_T, LEGEND_BELOW = 190, 128
    MARGIN_B = LEGEND_BELOW + px["legend"] + 20
    # Two bars to a row means the pitch is ROW_H × (1 − bargap) / 2, and each
    # bar carries its own label — so the pitch has to clear the in-chart floor
    # or plotly silently shrinks both labels under the print minimum.
    # One bar to a row now, so the row height is set by the two-line y label
    # rather than by stacking two bars — which is where the height saving
    # comes from. The outer bar still has to clear the in-chart floor.
    #
    # ROW_H is then set by a spacing rule, not by the bar: a two-line label is
    # ~2 x floor x 1.15 tall with ~0.3 x floor between its own lines, so at
    # ROW_H = 96 adjacent rows sat closer together than the two lines of one
    # label — which reads as the wrong grouping. The row pitch has to leave
    # MORE space between labels than a label has inside itself.
    #
    # The gutter has to clear the left half's evidence column AND the right
    # half's two-line labels, which grow toward each other inside it.
    ROW_H, BARGAP, GUTTER = 118, 0.20, 0.40
    assert ROW_H * _OUTER_W >= px["in_chart_floor"], (
        f"{ROW_H * _OUTER_W:.0f}px bar under the {px['in_chart_floor']}px floor"
    )
    _label_h = 2 * px["in_chart_floor"] * 1.15
    _line_gap = px["in_chart_floor"] * 0.3
    assert ROW_H - _label_h > _line_gap, (
        f"row {ROW_H}px leaves {ROW_H - _label_h:.0f}px between labels, under "
        f"the {_line_gap:.0f}px inside one — rows will read as mis-grouped"
    )
    rows_per_half = max(hi - lo for lo, hi in _SPLITS)
    height = rows_per_half * ROW_H + MARGIN_T + MARGIN_B

    # make_subplots only does uniform spacing; the gutter has to clear the
    # right half's two-line labels, which are as wide as the left margin.
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.01)
    panel_w = (1.0 - GUTTER) / 2
    for ci in (1, 2):
        suffix = "" if ci == 1 else str(ci)
        start = 0.0 if ci == 1 else panel_w + GUTTER
        fig.layout[f"xaxis{suffix}"].domain = (start, start + panel_w)

    for ci, (lo, hi) in enumerate(_SPLITS, start=1):
        suffix = "" if ci == 1 else str(ci)
        part = top.iloc[lo:hi]
        labels = [_wrap(v) for v in part["category"]]

        # Outer = tooling, inner = confirmed use, drawn in that order so the
        # inner one lands on top. The tooling number prints outside the outer
        # bar; the confirmed number inside its own where it fits.
        conf, mcp = part["pct_conf"].tolist(), part["pct_mcp"].tolist()
        fig.add_trace(go.Bar(
            y=labels, x=mcp, orientation="h", width=_OUTER_W,
            marker=dict(color=MCP_COLOR, line=dict(width=0)),
            cliponaxis=False, showlegend=False, hoverinfo="skip",
        ), row=1, col=ci)
        for lab, v in zip(labels, mcp):
            fig.add_annotation(
                xref=f"x{suffix}", yref=f"y{suffix}", x=_MCP_COL_X, y=lab,
                xanchor="right", yanchor="middle", showarrow=False,
                text=f"{v:.1f}%",
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
        fig.add_trace(go.Bar(
            y=labels, x=conf, orientation="h", width=_INNER_W,
            marker=dict(color=CONF_COLOR, line=dict(width=0)),
            # Always outside — it lands on the pale tooling bar, which takes
            # dark text fine, and an inside label would vanish on the short
            # bars this ranking is full of.
            text=[f"{v:.1f}%" for v in conf], textposition="outside",
            outsidetextfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                                 color=PAPER_PALETTE["text"]),
            constraintext="none", cliponaxis=False,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=ci)

        for lab, n, v in zip(labels, part["n_rated"], part["mcp_pct_share"]):
            fig.add_annotation(
                xref=f"x{suffix}", yref=f"y{suffix}", x=_SHARE_X, y=lab,
                xanchor="left", yanchor="middle", showarrow=False,
                text=f"{_fmt_count(float(n))} "
                     f"<span style='color:{PAPER_PALETTE['neutral']}'>"
                     f"({float(v):.2f}%)</span>",
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
        fig.add_annotation(
            xref=f"x{suffix}", yref="paper", x=_SHARE_X, y=1.004,
            xanchor="left", yanchor="bottom", showarrow=False,
            text="MCP ratings",
            font=dict(size=px["in_chart_floor"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["muted"]),
        )

        fig.update_xaxes(
            range=[0, _X_AXIS_MAX], tickvals=[0, 50, 100],
            ticktext=["0%", "50%", "100%"], tickangle=0, automargin=False,
            title=dict(text="Work Time Exposed",
                       font=dict(size=px["axis_title"])),
            row=1, col=ci,
        )
        fig.update_yaxes(autorange="reversed", dtick=1, automargin=False,
                         showgrid=False, showline=False,
                         tickfont=dict(size=px["in_chart_floor"],
                                       family=FONT_FAMILY),
                         row=1, col=ci)

    fig.update_layout(barmode="overlay", bargap=BARGAP)
    # The rows are IWAs and nothing else on the figure says so. Naming the
    # grain in the title rather than in a y-axis title keeps it consistent
    # with the other charts, which all dropped theirs — and the left margin
    # here is already two lines of activity name. "Agentic" comes out of the
    # last clause because the legend carries it twice; the full phrasing
    # measures 2542 px against the 2376 the canvas gives a one-line title.
    style_paper_figure(
        fig,
        "Intermediate Work Activities Where MCP Tooling Runs Furthest Ahead "
        "of Confirmed Use",
        width=W, height=height,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    fig.update_layout(showlegend=False, title=dict(y=0.985, yanchor="top"))
    fig.update_yaxes(tickfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY))

    # Legend only — every number on the figure is defined in the paper text,
    # so no explanatory footnote rides along with it.
    plot_h = height - MARGIN_T - MARGIN_B
    fig.add_annotation(
        xref="paper", yref="paper",
        x=paper_x_center(W, MARGIN_L, MARGIN_R), y=-LEGEND_BELOW / plot_h,
        xanchor="center", yanchor="top", showarrow=False,
        text=square_legend_html([(name, color) for _c, name, color in SERIES]),
        font=dict(size=px["legend"], family=FONT_FAMILY,
                  color=PAPER_PALETTE["text"]),
    )

    name = "agentic_tooling.png"
    save_figure(fig, results / "figures" / name, scale=2)
    _copy_fig(results, figures, name)
