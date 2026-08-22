"""Trend chart — work time exposed over time, three All Confirmed series,
all with a 2-year linear OLS projection:

  1. non-physical tasks only
  2. physical tasks only
  3. all tasks (aggregate)

Ratio-of-totals against the physical-mode-matched eco denominator, same
construction as the v1 supplemental non-phys trend.

A fourth line — Agentic Confirmed on the cumulative eco2025 AEI API files —
was dropped: it put a second dataset on the same axes, and what this figure
is for is the physical / non-physical divergence inside one of them. The
agentic series is still in `figure_data.AGENTIC_ECO2025_SERIES` if it is ever
wanted back.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import figure_data
from lib.paper_config import (
    FONT_FAMILY,
    METRIC_COLORS,
    PAPER_PALETTE,
    PAPER_W,
    paper_fonts,
    style_paper_figure,
)
from lib.utils import save_csv, save_figure


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _min_gap_positions(ys: list[float], gap: float) -> list[float]:
    """Nudge a sorted-by-value set of label centres apart to `gap` units.

    Sweeps upward from the lowest, then re-centres the block on its original
    midpoint so the whole group does not drift off its own lines.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    prev: float | None = None
    for i in order:
        v = ys[i] if prev is None else max(ys[i], prev + gap)
        out[i] = v
        prev = v
    shift = (sum(ys) - sum(out)) / len(ys)
    return [v + shift for v in out]


def _assign_above_below(
    ys: list[float], label_h: float, offset: float,
) -> list[float]:
    """Choose an above-or-below side per line so no label box crosses another
    series' line, and no two label boxes overlap.

    Everything is in axis units. With four series there are only 16 side
    combinations, so this is brute-forced and scored: a crossing costs a lot,
    and among clean layouts the one hugging its own line wins. Sides rather
    than free placement is what keeps this working at the chart's actual
    height — the lines sit close enough together that most of the gaps
    between them cannot hold a label at all.
    """
    from itertools import product

    n = len(ys)
    half = label_h * 0.58          # half box height, with a little padding
    best, best_cost = None, float("inf")
    for sides in product((1.0, -1.0), repeat=n):
        centres = [ys[i] + sides[i] * offset for i in range(n)]
        crossings = 0
        for i in range(n):
            lo, hi = centres[i] - half, centres[i] + half
            crossings += sum(1 for y in ys if lo < y < hi)
            for j in range(i + 1, n):
                if abs(centres[i] - centres[j]) < label_h * 1.15:
                    crossings += 1
        # Tie-break on "below", which reads as attached to its line.
        cost = crossings * 1e4 + sum(1 for s in sides if s > 0)
        if cost < best_cost:
            best, best_cost = centres, cost
    assert best is not None
    return best


def _run_config_phys_mode_time(dataset_name: str, physical_mode: str,
                               agg_level: str = "occupation") -> pd.DataFrame:
    """Exposure for one dataset under one physical mode, on the time_day weight.

    Local rather than imported because the paper helper hardcodes
    method="freq" and the paper figures must keep it.
    """
    from backend.compute import get_group_data

    assert physical_mode in {"all", "exclude", "only"}, physical_mode
    data = get_group_data({
        "selected_datasets": [dataset_name], "combine_method": "Average",
        "method": "time_day", "use_auto_aug": True,
        "physical_mode": physical_mode, "geo": "nat",
        "agg_level": agg_level, "sort_by": "% Tasks Affected",
        "top_n": 9999, "search_query": "", "context_size": 3,
    })
    assert data is not None, f"No data for {dataset_name}"
    df: pd.DataFrame = data["df"]
    return df.rename(columns={data["group_col"]: "category"})


# (series_key, label, dataset list, physical_mode, color, dash)
def _series_spec() -> list[tuple[str, str, list[str], str, str, str]]:
    # Three series, all All Confirmed — so the dataset is named once in the
    # title and the legend carries only what separates the lines. That is
    # what lets the legend be one row, which is what lets the bottom margin
    # come down. The agentic line was dropped: it is a different dataset on
    # the same axes, and the phys/non-phys divergence is the finding here.
    return [
        ("confirmed_nonphys", "Non-physical", figure_data.AC_SERIES, "exclude",
         METRIC_COLORS["tasks"], "solid"),
        ("confirmed_phys", "Physical", figure_data.AC_SERIES, "only",
         METRIC_COLORS["workers"], "solid"),
        ("confirmed_all", "All Tasks", figure_data.AC_SERIES, "all",
         "#233f57", "solid"),
    ]


def build_trend_phys(results: Path, figures: Path) -> None:
    from backend.compute import load_eco_baseline

    spec = _series_spec()

    # Hours per occupation under each physical filter: time_per_day x emp,
    # the same denominator every other exposure number uses. Employment
    # weighting is what makes the aggregate line "of all hours worked, this
    # share is exposed" rather than an unweighted average over occupations.
    emp_by_occ = (
        figure_data.eco_pairs()
        .drop_duplicates("title_current")
        .set_index("title_current")[figure_data.EMP_COL]
    )
    eco_tc_by_mode: dict[str, pd.Series] = {}
    for mode in {s[3] for s in spec}:
        eco_phys = load_eco_baseline(method="time_day", physical_mode=mode,
                                     geo="nat")
        hours = eco_phys.groupby("title_current")["task_comp"].sum()
        eco_tc_by_mode[mode] = hours * emp_by_occ.reindex(hours.index).fillna(0.0)

    trend_rows: list[dict] = []
    for series_key, label, datasets, mode, _c, _d in spec:
        eco_tc_by_occ = eco_tc_by_mode[mode]
        eco_tc_total = float(eco_tc_by_occ.sum())
        for ds_name in datasets:
            date_str = ds_name.rsplit(" ", 1)[-1]
            df = _run_config_phys_mode_time(ds_name, mode, "occupation")
            eco_tc_aligned = df["category"].map(eco_tc_by_occ).fillna(0.0)
            ai_total = float(((df["pct_tasks_affected"] / 100.0) * eco_tc_aligned).sum())
            pct = ai_total / eco_tc_total * 100.0 if eco_tc_total > 0 else 0.0
            trend_rows.append({
                "series": series_key, "label": label, "date": date_str,
                "dataset": ds_name, "physical_mode": mode,
                "pct_tasks_affected": round(pct, 1),
            })
    trend_df = pd.DataFrame(trend_rows)
    save_csv(trend_df, results / "trend_phys.csv")

    # The series share a narrow band, so vertical space was carrying no
    # information — nearly half the old canvas was margin. The labels are
    # kept off the lines by placing the first and last groups OUTSIDE the
    # data entirely and solving sides for the middle group, not by growing
    # the canvas. Printed at the 6.5" column this is 2.9" tall against the
    # 3.6" the four-line version took.
    W, H = PAPER_W, 620
    px = paper_fonts(W)
    # Bottom margin = the rotated ticks, the axis title, then ONE legend row.
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 110, 215, 92, 208

    def _project(dates: list[str], ys: list[float]) -> tuple[list[str], list[float]]:
        ts = [pd.Timestamp(d) for d in dates]
        x = np.array([(t - ts[0]).days for t in ts], dtype=float)
        b, a = np.polyfit(x, np.array(ys, dtype=float), deg=1)
        fx = [x[-1] + h for h in (183, 365, 730)]
        f_dates = [(ts[0] + pd.Timedelta(days=int(v))).strftime("%Y-%m-%d") for v in fx]
        f_ys = [float(a + b * v) for v in fx]
        return f_dates, f_ys

    fig = go.Figure()
    panel_vals: list[float] = []
    series_pts: list[dict] = []
    for series_key, label, _datasets, _mode, color, dash in spec:
        sub = trend_df[trend_df["series"] == series_key].sort_values("date")
        xvals = list(sub["date"])
        yvals = [float(v) for v in sub["pct_tasks_affected"]]
        panel_vals.extend(yvals)

        fig.add_trace(go.Scatter(
            x=xvals, y=yvals, name=label, showlegend=False,
            mode="lines+markers",
            line=dict(color=color, width=3, dash=dash),
            marker=dict(size=8, color=color),
            hoverinfo="skip", cliponaxis=False,
        ))

        f_dates, f_ys = _project(xvals, yvals)
        f_ys = [max(v, 0.0) for v in f_ys]
        fig.add_trace(go.Scatter(
            x=[xvals[-1]] + f_dates, y=[yvals[-1]] + f_ys,
            mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=7, color=color, symbol="x"),
            showlegend=False, hoverinfo="skip", cliponaxis=False, opacity=0.7,
        ))
        panel_vals.extend(f_ys)
        series_pts.append({
            "color": color,
            "first_x": xvals[0], "first_y": yvals[0],
            "last_x": xvals[-1], "last_y": yvals[-1],
            "proj_x": f_dates[-1], "proj_y": f_ys[-1],
        })

    v_lo, v_hi = min(panel_vals), max(panel_vals)
    spread = max(v_hi - v_lo, 1.0)
    y_lo = max(0.0, v_lo - spread * 0.16)
    y_hi = v_hi + spread * 0.16
    fig.update_yaxes(
        ticksuffix="%", range=[y_lo, y_hi],
        title=dict(text="Work Time Exposed", font=dict(size=px["axis_title"])),
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )

    # ── Data labels, placed so none of them sits on a line ────────────────
    # Three groups, each solved in its own way. The first and last points get
    # their labels OUTSIDE the data — left of the first snapshot, right of
    # the projection endpoint — where no line runs at all; only the middle
    # group (the last observed snapshot) has to be threaded between lines.
    plot_h = H - MARGIN_T - MARGIN_B
    units_per_px = (y_hi - y_lo) / plot_h
    label_h = px["in_chart_floor"] * units_per_px          # label height in axis units
    min_gap = label_h * 1.35

    first_ys = _min_gap_positions([s["first_y"] for s in series_pts], min_gap)
    proj_ys = _min_gap_positions([s["proj_y"] for s in series_pts], min_gap)
    last_ys = _assign_above_below(
        [s["last_y"] for s in series_pts], label_h, offset=label_h * 1.15,
    )
    for s, fy, ly, py in zip(series_pts, first_ys, last_ys, proj_ys):
        font = dict(size=px["in_chart_floor"], color=s["color"],
                    family=FONT_FAMILY)
        fig.add_annotation(
            x=s["first_x"], y=fy, text=f"{s['first_y']:.1f}%", showarrow=False,
            xanchor="right", xshift=-10, font=font,
        )
        fig.add_annotation(
            x=s["last_x"], y=ly, text=f"{s['last_y']:.1f}%", showarrow=False,
            xanchor="center", font=font,
        )
        fig.add_annotation(
            x=s["proj_x"], y=py, text=f"2yr: {s['proj_y']:.1f}%",
            showarrow=False, xanchor="left", xshift=12, font=font,
        )
    # Pad the left so the first-point labels, which sit to the LEFT of their
    # markers, land on empty plot area rather than on the y-axis ticks.
    x_lo = (pd.Timestamp(series_pts[0]["first_x"])
            - pd.Timedelta(days=105)).strftime("%Y-%m-%d")
    x_hi = (pd.Timestamp(series_pts[0]["proj_x"])
            + pd.Timedelta(days=55)).strftime("%Y-%m-%d")
    fig.update_xaxes(
        title=dict(text="Snapshot Date", font=dict(size=px["axis_title"])),
        tickangle=-30, range=[x_lo, x_hi],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )

    style_paper_figure(
        fig,
        "All Confirmed Work Time Exposed Over Time: Physical, Non-Physical, All Tasks",
        height=H, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    fig.update_layout(showlegend=False)

    # Manual centered legend, one row: three series plus the projection key.
    LEG_LINE = 0.044
    LEG_GAP = 0.010
    LEG_SP = 0.055
    char_w = px["legend"] * 0.55 / W

    def _w(lbl: str) -> float:
        return LEG_LINE + LEG_GAP + len(lbl) * char_w

    PROJ = "2-yr OLS Projection"
    rows: list[list[tuple[str, str, str, bool]]] = [
        [(spec[0][1], spec[0][4], spec[0][5], False),
         (spec[1][1], spec[1][4], spec[1][5], False),
         (spec[2][1], spec[2][4], spec[2][5], False),
         (PROJ, PAPER_PALETTE["text"], "dot", True)],
    ]
    # Anchored in px below the plot, not as a fraction of it: a paper-y
    # fraction moves with the plot height.
    row_ys = [-165 / plot_h]
    for row_items, leg_y in zip(rows, row_ys):
        total = sum(_w(l) for l, _, _, _ in row_items) + LEG_SP * (len(row_items) - 1)
        cursor = 0.5 - total / 2
        for label, color, dash_style, is_proj in row_items:
            x0, x1 = cursor, cursor + LEG_LINE
            if is_proj:
                mid, gh = (x0 + x1) / 2, 0.006
                for a, b in [(x0, mid - gh), (mid + gh, x1)]:
                    fig.add_shape(type="line", xref="paper", yref="paper",
                                  x0=a, x1=b, y0=leg_y, y1=leg_y,
                                  line=dict(color=color, width=2, dash="dot"))
                fig.add_annotation(xref="paper", yref="paper", x=mid, y=leg_y,
                                   text="×", showarrow=False, xanchor="center",
                                   yanchor="middle",
                                   font=dict(size=px["legend"],
                                             family=FONT_FAMILY, color=color))
            else:
                fig.add_shape(type="line", xref="paper", yref="paper",
                              x0=x0, x1=x1, y0=leg_y, y1=leg_y,
                              line=dict(color=color, width=3, dash=dash_style))
            fig.add_annotation(
                xref="paper", yref="paper", x=x1 + LEG_GAP, y=leg_y,
                text=label, showarrow=False, xanchor="left", yanchor="middle",
                font=dict(size=px["legend"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
            cursor += _w(label) + LEG_SP

    name = "trend_phys.png"
    save_figure(fig, results / "figures" / name)
    _copy_fig(results, figures, name)
