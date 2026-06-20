"""
Part 2 — Characterization: Where AI Exposure Falls

Five chart groups characterizing the structural distribution of AI exposure:
1. Physical/Informational Divide: box plots of % tasks affected by occ group
2. Job Zone: violin plots of % tasks affected by job zone (1–5)
3. SKA Levels: AI max bar + workforce markers for every SKA element (3 subplots)
4. Work Activities: all GWAs ranked by % tasks affected, colored by workers
5. Major Categories: all 22 majors, three side-by-side panels (pct/workers/wages)

Run from project root:
    venv/Scripts/python -m lib.builders.part2
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.config import (
    REFERENCE_DIR,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ROOT,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.compute_ska import load_ska_data
from lib.utils import FONT_FAMILY, save_figure, save_csv
from lib.paper_config import (
    PAPER_W, PAPER_H,
    TITLE_FS, SUBTITLE_FS, INSIDE_FS, OUTSIDE_FS, TICK_FS, LABEL_FS,
    LEGEND_FS, ANNOT_FS,
    METRIC_COLORS, PAPER_PALETTE,
    paper_fonts,
    style_paper_figure, fmt_wages, fmt_workers,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PRIMARY_KEY = "all_confirmed"
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]
PRIMARY_LABEL = ANALYSIS_CONFIG_LABELS[PRIMARY_KEY]

# Physical/informational thresholds (matches exploratory).
# Display order is Physical → Mixed → Non-physical, top to bottom.
PHYS_LOWER = 33.0
PHYS_UPPER = 67.0
OCC_GROUPS = ["Physical", "Mixed", "Non-physical"]
GROUP_COLORS = {
    "Non-physical": METRIC_COLORS["tasks"],     # Slate blue
    "Mixed":        METRIC_COLORS["wages"],     # Sage green
    "Physical":     METRIC_COLORS["workers"],   # Gold / yellow
}

# Job zone labels — prep level in parentheses so y-axis ticks don't carry
# em-dashes (cleaner read at paper print size).
ZONE_LABELS = {
    1: "Zone 1 (Little/No Prep)",
    2: "Zone 2 (Some Prep)",
    3: "Zone 3 (Medium Prep)",
    4: "Zone 4 (Considerable Prep)",
    5: "Zone 5 (Extensive Prep)",
}

# SKA constants
IMPORTANCE_THRESHOLD = 3.0
TOP_N_FOR_AVERAGE = 10
SKA_LABEL_MAX = 45


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _run_config(
    dataset_name: str,
    agg_level: str = "occupation",
    physical_mode: str = "all",
) -> pd.DataFrame:
    """Run the dashboard pipeline. `physical_mode='exclude'` strips physical
    tasks from both numerator and denominator (used by variant B charts)."""
    from backend.compute import get_group_data
    config = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": physical_mode,
        "geo": "nat",
        "agg_level": agg_level,
        "sort_by": "% Tasks Affected",
        "top_n": 9999,
        "search_query": "",
        "context_size": 3,
    }
    data = get_group_data(config)
    assert data is not None, f"No data for {dataset_name} ({physical_mode}, {agg_level})"
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    df = df.rename(columns={group_col: "category"})
    return df


# ─────────────────────────────────────────────────────────────────────────
# Structural variants: Variant A (eco-only non-phys task share, ratio of
# totals) and Variant B (dashboard pipeline restricted to non-phys tasks).
# Both serve the major-cat trio at the top of Part 2 and the GWA quintet.
# ─────────────────────────────────────────────────────────────────────────

LEVEL_COL: dict[str, str] = {
    "major":      "major_occ_category",
    "minor":      "minor_occ_category",
    "broad":      "broad_occ",
    "occupation": "title_current",
}


def _coerce_phys_bool(val) -> bool:
    """Mirror backend.compute._phys_bool. eco rows store physical as
    1/0/True/False/'True'/'False' depending on import path."""
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "y"}
    return bool(val)


def compute_variant_a(agg_level: str = "major") -> pd.DataFrame:
    """Variant A — naive non-physical task share by group, freq-weighted.

    For each task-occ pair: w_all = freq_mean, w_nonphys = freq_mean if
    non-physical else 0. Per group (occupation / broad / minor / major):
    pct_A = Σ w_nonphys / Σ w_all × 100  (ratio of totals).

    Returns DataFrame with columns: category, pct_tasks_affected.
    """
    cols = [
        "title_current", "task_normalized",
        "broad_occ", "minor_occ_category", "major_occ_category",
        "physical", "freq_mean",
    ]
    df = pd.read_csv(DATA_DIR / "final_eco_2025.csv", usecols=cols)
    df = df.groupby(["title_current", "task_normalized"], sort=False, as_index=False).first()
    df["physical_bool"] = df["physical"].apply(_coerce_phys_bool)
    df["freq_mean"] = df["freq_mean"].fillna(0.0).astype(float)
    df["w_all"]      = df["freq_mean"]
    df["w_nonphys"]  = np.where(df["physical_bool"], 0.0, df["freq_mean"])

    gc = LEVEL_COL[agg_level]
    agg = df.groupby(gc, sort=False, as_index=False).agg(
        num=("w_nonphys", "sum"),
        den=("w_all", "sum"),
    )
    agg["pct_tasks_affected"] = (
        agg["num"] / agg["den"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    return agg.rename(columns={gc: "category"})[["category", "pct_tasks_affected"]]


def compute_variant_a_gwa() -> pd.DataFrame:
    """Variant A at GWA level: pct_A per GWA = Σ freq_mean[non-phys] /
    Σ freq_mean[all] within the GWA's task pool. eco_2025 expands tasks by
    work-activity so we group on (task_normalized, gwa_title) rather than
    deduping by task alone."""
    cols = ["task_normalized", "gwa_title", "physical", "freq_mean"]
    df = pd.read_csv(DATA_DIR / "final_eco_2025.csv", usecols=cols)
    df = df.dropna(subset=["gwa_title"])
    df = df.groupby(["task_normalized", "gwa_title"], sort=False, as_index=False).first()
    df["physical_bool"] = df["physical"].apply(_coerce_phys_bool)
    df["freq_mean"] = df["freq_mean"].fillna(0.0).astype(float)
    df["w_all"]     = df["freq_mean"]
    df["w_nonphys"] = np.where(df["physical_bool"], 0.0, df["freq_mean"])
    agg = df.groupby("gwa_title", sort=False, as_index=False).agg(
        num=("w_nonphys", "sum"),
        den=("w_all", "sum"),
    )
    agg["pct_tasks_affected"] = (
        agg["num"] / agg["den"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    return agg.rename(columns={"gwa_title": "category"})[["category", "pct_tasks_affected"]]


# ─────────────────────────────────────────────────────────────────────────
# Trend helpers — linear OLS projection (mirrors Part 1's extrapolation)
# ─────────────────────────────────────────────────────────────────────────

def _linear_project(dates: list[pd.Timestamp], yvals: list[float],
                    horizon_days: int) -> tuple[float, float, float]:
    """OLS y = a + b·t. Returns (slope_b_per_day, projected_y, r_squared).

    Linear is the simplest defensible "if recent rate continues" model
    given 4-snapshot input series; longer horizons need richer models."""
    if len(dates) < 2:
        return 0.0, float(yvals[-1] if yvals else 0.0), 0.0
    t0 = dates[0]
    x = np.array([(t - t0).days for t in dates], dtype=float)
    y = np.array(yvals, dtype=float)
    b, a = np.polyfit(x, y, deg=1)
    last_x = x[-1]
    projected = float(a + b * (last_x + horizon_days))
    # r²
    y_pred = a + b * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(b), projected, r2


def _get_national_totals() -> tuple[float, float]:
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    occ = eco.drop_duplicates(subset=["title_current"])
    total_emp = float(occ["emp_tot_nat_2025"].sum())
    total_wages = float((occ["emp_tot_nat_2025"] * occ["a_med_nat_2025"]).sum())
    return total_emp, total_wages


def _load_occ_structural() -> pd.DataFrame:
    """Load eco_2025 and compute per-occupation structural data:
    pct_physical, occ_group, job_zone.

    pct_physical is computed over UNIQUE (occ, task) pairs — eco_2025 expands
    each task across its GWA/IWA/DWA classifications, and that expansion is
    not proportional between physical and non-physical tasks. Counting raw
    rows weights tasks by their WA-expansion factor and produces the wrong
    per-occ physical share. This matches the dashboard backend pipeline.
    """
    eco = pd.read_csv(DATA_DIR / "final_eco_2025.csv")
    assert "title_current" in eco.columns
    assert "physical" in eco.columns
    assert "job_zone" in eco.columns
    assert "task_normalized" in eco.columns

    # Dedup on (occ, task) before counting. job_zone, emp, wage are occ-level
    # constants so the dedup leaves them untouched.
    eco_unique = eco.drop_duplicates(["title_current", "task_normalized"])

    occ = (
        eco_unique.groupby("title_current")
        .agg(
            n_tasks=("physical", "count"),
            n_physical=("physical", "sum"),
            job_zone=("job_zone", "first"),
            emp=("emp_tot_nat_2025", "first"),
            wage=("a_med_nat_2025", "first"),
        )
        .reset_index()
    )
    occ["pct_physical"] = occ["n_physical"] / occ["n_tasks"] * 100

    occ["occ_group"] = "Mixed"
    occ.loc[occ["pct_physical"] < PHYS_LOWER, "occ_group"] = "Non-physical"
    occ.loc[occ["pct_physical"] > PHYS_UPPER, "occ_group"] = "Physical"

    return occ


def _get_wa_data(dataset_name: str, level: str = "gwa") -> pd.DataFrame:
    """Get work activity exposure for one pre-combined dataset."""
    from backend.compute import compute_work_activities
    settings = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": "all",
        "geo": "nat",
        "sort_by": "workers_affected",
        "top_n": 9999,
    }
    result = compute_work_activities(settings)
    group = result.get("mcp_group") or result.get("aei_group")
    if group is None:
        return pd.DataFrame()
    rows = group.get(level, [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
# Chart 1: Physical / Informational Divide
# ─────────────────────────────────────────────────────────────────────────

def build_phys_info_divide(results: Path, figures: Path) -> None:
    occ = _load_occ_structural()
    pct = get_pct_tasks_affected(PRIMARY_DATASET)

    occ["pct_tasks_affected"] = occ["title_current"].map(pct)
    occ = occ.dropna(subset=["pct_tasks_affected"])

    # Summary stats for CSV
    summary_rows = []
    for grp in OCC_GROUPS:
        sub = occ[occ["occ_group"] == grp]
        summary_rows.append({
            "occ_group": grp,
            "n_occs": len(sub),
            "median_pct": round(float(sub["pct_tasks_affected"].median()), 1),
            "mean_pct": round(float(sub["pct_tasks_affected"].mean()), 1),
            "q25": round(float(sub["pct_tasks_affected"].quantile(0.25)), 1),
            "q75": round(float(sub["pct_tasks_affected"].quantile(0.75)), 1),
        })
    save_csv(pd.DataFrame(summary_rows), results / "phys_info_summary.csv")

    # Box plot
    fig = go.Figure()

    for grp in OCC_GROUPS:
        subset = occ[occ["occ_group"] == grp]
        fig.add_trace(go.Box(
            x=subset["pct_tasks_affected"],
            name=f"{grp}  (n={len(subset)})",
            marker_color=GROUP_COLORS[grp],
            line_color=GROUP_COLORS[grp],
            fillcolor=GROUP_COLORS[grp],
            opacity=0.7,
            boxmean=True,
            orientation="h",
        ))

    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=[
                f"{g}  (n={int(occ[occ['occ_group'] == g].shape[0])})"
                for g in reversed(OCC_GROUPS)
            ],
        ),
    )

    style_paper_figure(
        fig,
        "Task Exposure by Physical, Mixed, or Non-Physical Occupations",
        subtitle=(
            f"Distribution of % tasks exposed across {len(occ)} occupations "
            "(Physical = >67% tasks physical · Mixed = 33–67% · Non-physical = <33%)"
        ),
        height=460,
        width=PAPER_W,
        margin=dict(l=80, r=60, t=100, b=80),
    )

    fig.update_layout(showlegend=False)

    fig.update_xaxes(
        title=dict(text="% Tasks Exposed", font=dict(size=LABEL_FS)),
        range=[0, 100], dtick=10,
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
    )
    fig.update_yaxes(
        title=dict(text="Occupation Group", font=dict(size=LABEL_FS - 2)),
        showgrid=False, showline=False,
        tickfont=dict(size=TICK_FS, family=FONT_FAMILY),
    )

    save_figure(fig, results / "figures" / "phys_info_divide.png")
    _copy_fig(results, figures, "phys_info_divide.png")
    print("  -> phys_info_divide.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 2: Job Zone Violin
# ─────────────────────────────────────────────────────────────────────────

def build_job_zone_violin(results: Path, figures: Path) -> None:
    """Three-panel job-zone chart:
      1. All Occupations — violins for Zones 1–5
      2. Phys Mix       — narrow stacked bar of phys/mixed/non-phys share
      3. Non-Physical Only — violins for occs with pct_physical < 33%
                             (Zone 1 row kept blank to preserve alignment)

    Section headers ("All Occupations" / "Phys Mix" / "Non-Physical Only")
    sit above each panel — same labeled-section pattern as the convergence
    matrix's Internal / External headers.
    """
    occ_all = _load_occ_structural()
    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    occ_all["pct_tasks_affected"] = occ_all["title_current"].map(pct)
    occ_all = occ_all.dropna(subset=["pct_tasks_affected", "job_zone"])
    occ_all["job_zone"] = occ_all["job_zone"].astype(int)
    occ_nonphys = occ_all[occ_all["occ_group"] == "Non-physical"].copy()

    n_all = len(occ_all)
    n_nonphys = len(occ_nonphys)

    # Anchor the full 1–5 zone ladder so the non-phys side keeps a blank
    # Zone 1 row aligned with the All side.
    zones = [1, 2, 3, 4, 5]
    zone_labels_short = {z: ZONE_LABELS.get(z, f"Zone {z}") for z in zones}

    # Summary stats — including phys-mix breakdown per zone (used for the
    # middle stacked-bar panel and the CSV export).
    zone_stats = []
    for z in zones:
        sub = occ_all[occ_all["job_zone"] == z]
        n_total = len(sub)
        n_phys = int((sub["occ_group"] == "Physical").sum())
        n_mix  = int((sub["occ_group"] == "Mixed").sum())
        n_non  = int((sub["occ_group"] == "Non-physical").sum())
        sub_np = occ_nonphys[occ_nonphys["job_zone"] == z]
        zone_stats.append({
            "job_zone": z,
            "n_occs_all": n_total,
            "median_pct_all": round(float(sub["pct_tasks_affected"].median()), 1) if n_total else None,
            "mean_pct_all":   round(float(sub["pct_tasks_affected"].mean()),   1) if n_total else None,
            "n_occs_nonphys": len(sub_np),
            "median_pct_nonphys": round(float(sub_np["pct_tasks_affected"].median()), 1) if len(sub_np) else None,
            "mean_pct_nonphys":   round(float(sub_np["pct_tasks_affected"].mean()),   1) if len(sub_np) else None,
            "n_physical":     n_phys,
            "n_mixed":        n_mix,
            "n_non_physical": n_non,
            "pct_physical":     round(n_phys / n_total * 100, 1) if n_total else 0.0,
            "pct_mixed":        round(n_mix / n_total * 100, 1) if n_total else 0.0,
            "pct_non_physical": round(n_non / n_total * 100, 1) if n_total else 0.0,
        })
    save_csv(pd.DataFrame(zone_stats), results / "job_zone_summary.csv")

    # Color gradient: Zone 1 lightest, Zone 5 darkest
    zone_colors = {
        1: "#b8cfe0",
        2: "#8cafc5",
        3: "#6090aa",
        4: "#3a6f8f",
        5: "#1a4f73",
    }

    # Use plain zone labels on the shared y-axis (no n) — n's belong in the
    # section headers since the all and non-phys sides have different counts.
    y_labels = [zone_labels_short[z] for z in zones]
    y_order_top_down = list(reversed(y_labels))

    # Canvas sized for three panels at 6.5" column width. Font sizes resolved
    # from paper_fonts(W) so the ladder (11/10/10/9/9/8 pt) lands at print pt.
    W = PAPER_W
    H = 720
    px = paper_fonts(W)

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        column_widths=[0.46, 0.16, 0.38],
        horizontal_spacing=0.04,
        subplot_titles=["", "", ""],  # headers added manually below
    )

    # ── Panel 1: All Occupations violins ────────────────────────────────
    # Box + meanline turned off — replaced by a custom black median line per
    # zone (added below as shapes) so the central tendency reads cleanly at
    # paper print size.
    for z in zones:
        sub = occ_all[occ_all["job_zone"] == z]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Violin(
            x=sub["pct_tasks_affected"],
            y=[zone_labels_short[z]] * len(sub),
            name=zone_labels_short[z],
            marker_color=zone_colors[z],
            line_color=zone_colors[z],
            fillcolor=zone_colors[z],
            opacity=0.75,
            box_visible=False,
            meanline_visible=False,
            orientation="h",
            side="positive",
            width=0.7,
            points=False,
            showlegend=False,
            hovertemplate=(
                f"{zone_labels_short[z]} (All)<br>"
                "%{x:.1f}% tasks exposed<extra></extra>"
            ),
        ), row=1, col=1)

    # Per-zone inline n + median label, right-justified inside the panel.
    for r in zone_stats:
        n_occs = r["n_occs_all"]
        if n_occs == 0:
            continue
        med = r["median_pct_all"]
        med_str = f" · med {med:.0f}%" if med is not None else ""
        fig.add_annotation(
            x=99, y=zone_labels_short[r["job_zone"]],
            xref="x", yref="y",
            text=f"n={n_occs}{med_str}",
            showarrow=False,
            xanchor="right", yanchor="top",
            yshift=-2,
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

    # ── Phys Mix stacked bars ───────────────────────────────────────────
    # Bars themselves carry no legend — we use dummy scatter traces below
    # so we can control marker size (default bar legend swatches are tiny).
    pct_phys_arr = [r["pct_physical"]     for r in zone_stats]
    pct_mix_arr  = [r["pct_mixed"]        for r in zone_stats]
    pct_non_arr  = [r["pct_non_physical"] for r in zone_stats]
    bar_y = [zone_labels_short[r["job_zone"]] for r in zone_stats]

    fig.add_trace(go.Bar(
        x=pct_phys_arr, y=bar_y, orientation="h",
        marker=dict(color=GROUP_COLORS["Physical"], line=dict(width=0)),
        name="% Physical occs",
        showlegend=False,
        hovertemplate="Physical: %{x:.0f}%<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=pct_mix_arr, y=bar_y, orientation="h",
        marker=dict(color=GROUP_COLORS["Mixed"], line=dict(width=0)),
        name="% Mixed occs",
        showlegend=False,
        hovertemplate="Mixed: %{x:.0f}%<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=pct_non_arr, y=bar_y, orientation="h",
        marker=dict(color=GROUP_COLORS["Non-physical"], line=dict(width=0)),
        name="% Non-physical occs",
        showlegend=False,
        hovertemplate="Non-physical: %{x:.0f}%<extra></extra>",
    ), row=1, col=2)

    # ── Legend dummy traces ─────────────────────────────────────────────
    # All legend entries are dummy Scatter traces so we can size the
    # swatches uniformly (bar-trace legend markers are tiny and not
    # controllable). Anchored to col=1 (violin panel) with None coords so
    # they don't render anywhere on the chart. Order: Non-physical →
    # Mixed → Physical → Median (left to right in legend).
    legend_marker_size = 16
    median_line_size = 22  # matches text cap height — fully vertical span
    for tier_name, tier_color in (
        ("% Non-physical occs", GROUP_COLORS["Non-physical"]),
        ("% Mixed occs",        GROUP_COLORS["Mixed"]),
        ("% Physical occs",     GROUP_COLORS["Physical"]),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                color=tier_color,
                size=legend_marker_size,
                line=dict(width=0),
            ),
            name=tier_name,
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            symbol="line-ns",
            color="#1a1a1a",
            size=median_line_size,
            line=dict(color="#1a1a1a", width=3),
        ),
        name="Median",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # ── Panel 3: Non-Physical Only violins ──────────────────────────────
    # Force the full 1–5 ladder on this axis with a dummy invisible scatter
    # so Zone 1 stays as a blank row.
    fig.add_trace(go.Scatter(
        x=[None] * len(zones),
        y=y_labels,
        mode="markers",
        marker=dict(opacity=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=3)

    for z in zones:
        sub = occ_nonphys[occ_nonphys["job_zone"] == z]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Violin(
            x=sub["pct_tasks_affected"],
            y=[zone_labels_short[z]] * len(sub),
            name=zone_labels_short[z],
            marker_color=zone_colors[z],
            line_color=zone_colors[z],
            fillcolor=zone_colors[z],
            opacity=0.75,
            box_visible=False,
            meanline_visible=False,
            orientation="h",
            side="positive",
            width=0.7,
            points=False,
            showlegend=False,
            hovertemplate=(
                f"{zone_labels_short[z]} (Non-physical)<br>"
                "%{x:.1f}% tasks exposed<extra></extra>"
            ),
        ), row=1, col=3)

    for r in zone_stats:
        z = r["job_zone"]
        n_np = r["n_occs_nonphys"]
        if n_np == 0:
            continue
        med = r["median_pct_nonphys"]
        med_str = f" · med {med:.0f}%" if med is not None else ""
        fig.add_annotation(
            x=99, y=zone_labels_short[z],
            xref="x3", yref="y3",
            text=f"n={n_np}{med_str}",
            showarrow=False,
            xanchor="right", yanchor="top",
            yshift=-2,
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

    # ── Black median lines per zone ─────────────────────────────────────
    # Each violin gets a thick vertical black line at the median. Median
    # over mean — robust to skew in the long-tail zones. Y extent spans
    # the violin's positive-side band: from baseline (category index) up
    # to baseline + ~0.45 (matches the 0.7 violin width with a small lift
    # off the baseline so the line reads as a separate mark).
    def _add_median_lines(stat_key: str, xref: str, yref: str) -> None:
        for r in zone_stats:
            med = r[stat_key]
            if med is None:
                continue
            pos = y_order_top_down.index(zone_labels_short[r["job_zone"]])
            fig.add_shape(
                type="line",
                xref=xref, yref=yref,
                x0=med, x1=med,
                y0=pos - 0.02, y1=pos + 0.40,
                line=dict(color="#1a1a1a", width=2),
                layer="above",
            )

    _add_median_lines("median_pct_all",     "x",  "y")
    _add_median_lines("median_pct_nonphys", "x3", "y3")

    # Narrow the phys-mix bar height to match the slimmer violins.
    fig.update_traces(width=0.55, selector=dict(type="bar"))

    # ── Y / X axes ──────────────────────────────────────────────────────
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y_order_top_down,
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title=dict(text="Job Zone (O*NET Preparation Level)",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        row=1, col=1,
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=y_order_top_down,
        showgrid=False, showline=False, showticklabels=False,
        row=1, col=2,
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=y_order_top_down,
        showgrid=False, showline=False, showticklabels=False,
        row=1, col=3,
    )

    # All Occs x-axis: drop the "100" tick — it sits right next to the
    # Phys Mix panel and reads as visual noise. Same dtick=20 spacing.
    fig.update_xaxes(
        title=dict(text="% Tasks Exposed",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        range=[0, 100],
        tickmode="array",
        tickvals=[0, 20, 40, 60, 80],
        ticktext=["0%", "20%", "40%", "60%", "80%"],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        row=1, col=1,
    )
    fig.update_xaxes(
        range=[0, 100], dtick=50,
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        ticksuffix="%",
        tickfont=dict(size=px["in_chart_floor"], family=FONT_FAMILY),
        title=dict(text="% of zone",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        row=1, col=2,
    )
    # Non-Phys x-axis: drop the "0" tick — sits hard against the black
    # divider from the Phys Mix panel and reads as visual noise.
    fig.update_xaxes(
        title=dict(text="% Tasks Exposed",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        range=[0, 100],
        tickmode="array",
        tickvals=[20, 40, 60, 80, 100],
        ticktext=["20%", "40%", "60%", "80%", "100%"],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        row=1, col=3,
    )

    style_paper_figure(
        fig,
        "Task Exposure by Job Zone — Full Economy vs Non-Physical Occupations Only",
        subtitle="",
        height=H, width=W,
        margin=dict(l=240, r=50, t=180, b=210),
    )

    # Pin the figure title near the top of the figure so the section
    # headers below it have a clean vertical lane to live in. With
    # t=180 margin and title y=0.965, the title sits in the top ~25 px
    # and headers at y=1.02 (just above plot area) land below it.
    fig.update_layout(title=dict(y=0.965, yanchor="top"))

    # Section headers above each panel — bold, panel_title size, paper-coord
    # x positioned at the midpoint of each column. Mirrors the Internal /
    # External pattern from the convergence chart. Column centers derived
    # from column_widths=[0.46, 0.16, 0.38] with horizontal_spacing=0.04:
    # col 1 center ≈ 0.212, col 2 center ≈ 0.537, col 3 center ≈ 0.825.
    header_specs = [
        ("All Occupations",    f"n={n_all}",       0.212),
        ("Phys Mix",           "",                  0.537),
        ("Non-Physical",       f"n={n_nonphys}",   0.825),
    ]
    for title_text, sub_n, x_paper in header_specs:
        text = f"<b>{title_text}</b>"
        if sub_n:
            text += (f"  <span style='font-size:{px['in_chart_floor']}px;"
                     f"color:{PAPER_PALETTE['neutral']}'>({sub_n})</span>")
        fig.add_annotation(
            xref="paper", yref="paper",
            x=x_paper, y=1.02,
            text=text,
            showarrow=False,
            xanchor="center", yanchor="bottom",
            font=dict(size=px["panel_title"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
            align="center",
        )

    # Vertical separators in the gaps between panels. The Phys Mix |
    # Non-Physical boundary gets a clear black bar to signal that the
    # right panel is a different population (non-physical only), not just
    # a continuation. The All | Phys Mix boundary stays faint since the
    # Phys Mix panel is a structural readout of the same population.
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0.4432, x1=0.4432, y0=0.0, y1=1.02,
        line=dict(color=PAPER_PALETTE["grid"], width=1),
    )
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0.6304, x1=0.6304, y0=0.0, y1=1.11,
        line=dict(color="#1a1a1a", width=2),
    )

    # Legend with xref="container" / yref="container": x/y are in figure
    # container coords (0–1 of full figure width/height), not plot-area
    # paper coords. x=0.5 with xanchor="center" centers under the whole
    # figure regardless of asymmetric margins. traceorder="normal" disables
    # plotly's auto-reverse (triggered by barmode="stack") so the legend
    # reads in the order we add the dummy traces. itemsizing="trace" uses
    # the marker.size from each trace (default "constant" caps swatch size).
    fig.update_layout(
        barmode="stack",
        legend=dict(
            orientation="h",
            xref="container", yref="container",
            x=0.5, xanchor="center",
            y=0.04, yanchor="bottom",
            font=dict(size=px["legend"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
            itemsizing="trace",
            traceorder="normal",
        ),
    )

    save_figure(fig, results / "figures" / "job_zone_violin.png")
    _copy_fig(results, figures, "job_zone_violin.png")
    print("  -> job_zone_violin.png")


# ─────────────────────────────────────────────────────────────────────────
# Combined Phys/Info × Job Zone — Option A (stacked) and Option B (faceted)
# ─────────────────────────────────────────────────────────────────────────

ZONE_COLORS = {
    1: "#b8cfe0",
    2: "#8cafc5",
    3: "#6090aa",
    4: "#3a6f8f",
    5: "#1a4f73",
}


def _occ_with_pct() -> pd.DataFrame:
    occ = _load_occ_structural()
    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    occ["pct_tasks_affected"] = occ["title_current"].map(pct)
    occ = occ.dropna(subset=["pct_tasks_affected", "job_zone"])
    occ["job_zone"] = occ["job_zone"].astype(int)
    return occ


def build_combined_stacked(results: Path, figures: Path) -> None:
    """Option A — phys/info boxes (top) + job zone violins (bottom), shared x-axis."""
    occ = _occ_with_pct()
    zones = sorted(occ["job_zone"].unique())
    n_phys = len(OCC_GROUPS)
    n_zones = len(zones)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[n_phys / (n_phys + n_zones), n_zones / (n_phys + n_zones)],
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            "Physical / Informational Divide",
            "Job Zone (Preparation Level)",
        ],
    )

    # Top: phys/info boxes
    for grp in OCC_GROUPS:
        sub = occ[occ["occ_group"] == grp]
        fig.add_trace(go.Box(
            x=sub["pct_tasks_affected"],
            name=f"{grp}  (n={len(sub)})",
            marker_color=GROUP_COLORS[grp],
            line_color=GROUP_COLORS[grp],
            fillcolor=GROUP_COLORS[grp],
            opacity=0.7,
            boxmean=True,
            orientation="h",
            showlegend=False,
        ), row=1, col=1)

    # Bottom: zone violins
    for z in zones:
        sub = occ[occ["job_zone"] == z]
        label = ZONE_LABELS.get(z, f"Zone {z}")
        fig.add_trace(go.Violin(
            x=sub["pct_tasks_affected"],
            name=f"{label}  (n={len(sub)})",
            marker_color=ZONE_COLORS[z],
            line_color=ZONE_COLORS[z],
            fillcolor=ZONE_COLORS[z],
            opacity=0.7,
            box_visible=True,
            meanline_visible=True,
            orientation="h",
            side="positive",
            width=0.85,
            showlegend=False,
        ), row=2, col=1)

    # Y-axis ordering per row
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[
            f"{g}  (n={int(occ[occ['occ_group'] == g].shape[0])})"
            for g in reversed(OCC_GROUPS)
        ],
        showgrid=False, showline=False,
        tickfont=dict(size=TICK_FS, family=FONT_FAMILY),
        row=1, col=1,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[
            f"{ZONE_LABELS.get(z, f'Zone {z}')}  (n={int(occ[occ['job_zone'] == z].shape[0])})"
            for z in reversed(zones)
        ],
        showgrid=False, showline=False,
        tickfont=dict(size=TICK_FS - 1, family=FONT_FAMILY),
        row=2, col=1,
    )

    fig.update_xaxes(
        range=[0, 100], dtick=10,
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        row=1, col=1,
    )
    fig.update_xaxes(
        range=[0, 100], dtick=10,
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        title=dict(text="% Tasks Affected", font=dict(size=LABEL_FS)),
        row=2, col=1,
    )

    style_paper_figure(
        fig,
        "Where AI Exposure Falls — by Physical Mix and Preparation Level",
        subtitle=f"Distribution of % tasks affected across {len(occ)} occupations",
        height=820,
        width=PAPER_W,
        margin=dict(l=20, r=60, t=90, b=70),
    )

    # Style subplot titles
    panel_titles = {
        "Physical / Informational Divide",
        "Job Zone (Preparation Level)",
    }
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=LABEL_FS, family=FONT_FAMILY, color=PAPER_PALETTE["text"])

    save_figure(fig, results / "figures" / "phys_zone_stacked.png", scale=2)
    _copy_fig(results, figures, "phys_zone_stacked.png")
    print("  -> phys_zone_stacked.png")


def build_combined_faceted(results: Path, figures: Path) -> None:
    """Option B — 3 panels (Non-physical / Mixed / Physical), each containing 5
    job zone violins. Cross-tab view.
    """
    occ = _occ_with_pct()
    zones = sorted(occ["job_zone"].unique())

    # Cross-tab summary CSV
    rows_csv = []
    for grp in OCC_GROUPS:
        for z in zones:
            sub = occ[(occ["occ_group"] == grp) & (occ["job_zone"] == z)]
            rows_csv.append({
                "occ_group": grp,
                "job_zone": z,
                "n_occs": len(sub),
                "median_pct": round(float(sub["pct_tasks_affected"].median()), 1) if len(sub) else None,
                "mean_pct": round(float(sub["pct_tasks_affected"].mean()), 1) if len(sub) else None,
            })
    save_csv(pd.DataFrame(rows_csv), results / "phys_zone_crosstab.csv")

    panel_titles = [
        f"{g}  (n={int(occ[occ['occ_group'] == g].shape[0])})"
        for g in OCC_GROUPS
    ]

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=panel_titles,
    )

    y_labels = [f"Zone {z}" for z in zones]

    for col_idx, grp in enumerate(OCC_GROUPS, start=1):
        grp_df = occ[occ["occ_group"] == grp]
        for z in zones:
            sub = grp_df[grp_df["job_zone"] == z]
            label = f"Zone {z}"
            if len(sub) == 0:
                # Add an invisible placeholder so the axis row still renders
                fig.add_trace(go.Scatter(
                    x=[None], y=[label],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=1, col=col_idx)
                continue
            fig.add_trace(go.Violin(
                x=sub["pct_tasks_affected"],
                y=[label] * len(sub),
                marker_color=ZONE_COLORS[z],
                line_color=ZONE_COLORS[z],
                fillcolor=ZONE_COLORS[z],
                opacity=0.7,
                box_visible=True,
                meanline_visible=True,
                orientation="h",
                side="positive",
                width=0.9,
                points=False,
                showlegend=False,
                name=f"{grp} — {label}",
                hovertemplate=(
                    f"{grp}, {label}<br>"
                    "%{x:.1f}%<extra></extra>"
                ),
            ), row=1, col=col_idx)

        # Cell-level n + median annotations to the right of each violin
        for z in zones:
            sub = grp_df[grp_df["job_zone"] == z]
            if len(sub) == 0:
                txt = "n=0"
            else:
                med = sub["pct_tasks_affected"].median()
                txt = f"n={len(sub)}, med {med:.0f}%"
            fig.add_annotation(
                x=99, y=f"Zone {z}",
                xref=f"x{'' if col_idx == 1 else col_idx}",
                yref=f"y{'' if col_idx == 1 else col_idx}",
                text=txt,
                showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
            )

    y_order = [f"Zone {z}" for z in reversed(zones)]
    for col_idx in range(1, 4):
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=y_order,
            showgrid=False, showline=False,
            tickfont=dict(size=TICK_FS, family=FONT_FAMILY),
            row=1, col=col_idx,
        )
        x_kwargs = dict(
            range=[0, 100], dtick=20,
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
            showline=True, linecolor=PAPER_PALETTE["grid"],
            row=1, col=col_idx,
        )
        if col_idx == 2:
            fig.update_xaxes(
                title=dict(text="% Tasks Affected", font=dict(size=LABEL_FS)),
                **x_kwargs,
            )
        else:
            fig.update_xaxes(**x_kwargs)

    style_paper_figure(
        fig,
        "AI Exposure by Physical Mix × Preparation Level",
        subtitle=f"Job zone violins within each occupation group ({len(occ)} occupations)",
        height=640,
        width=PAPER_W,
        margin=dict(l=20, r=60, t=90, b=70),
    )

    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=LABEL_FS, family=FONT_FAMILY, color=PAPER_PALETTE["text"])

    save_figure(fig, results / "figures" / "phys_zone_faceted.png", scale=2)
    _copy_fig(results, figures, "phys_zone_faceted.png")
    print("  -> phys_zone_faceted.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 3: SKA Levels
# ─────────────────────────────────────────────────────────────────────────

def _compute_ska_variants(
    onet_df: pd.DataFrame,
    pct_series: pd.Series,
    type_name: str,
    phys_map: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute AI and workforce imp×lv variants per element for one SKA
    type. `phys_map` is an optional title_current → pct_physical Series;
    when present, each element record carries `phys_score` (unweighted
    mean of pct_physical across occs with imp ≥ 3 for that element) and
    `phys_tier` (Physical / Mixed / Non-physical bucket)."""
    df = onet_df.copy()
    df["pct"] = df["title"].map(pct_series)
    df = df.dropna(subset=["pct", "importance", "level"])
    df = df[df["importance"] >= IMPORTANCE_THRESHOLD].copy()
    assert len(df) > 0, f"No {type_name} rows after importance filter"

    df["occ_score"] = df["importance"] * df["level"]
    df["ai_product"] = (df["pct"] / 100.0) * df["occ_score"]
    if phys_map is not None:
        df["pct_physical_occ"] = df["title"].map(phys_map)

    records = []
    for element_name, grp in df.groupby("element_name"):
        ai_vals = grp["ai_product"].dropna()
        occ_vals = grp["occ_score"].dropna()
        n_ai = len(ai_vals)
        n_occ = len(occ_vals)
        top_n_ai = min(TOP_N_FOR_AVERAGE, n_ai)
        top_n_occ = min(TOP_N_FOR_AVERAGE, n_occ)

        rec = {
            "element_name": element_name,
            "type": type_name,
            "n_occs": n_ai,
            "ai_95th": float(ai_vals.quantile(0.95)) if n_ai >= 2 else (float(ai_vals.iloc[0]) if n_ai == 1 else float("nan")),
            "ai_max": float(ai_vals.max()) if n_ai >= 1 else float("nan"),
            "ai_top10": float(ai_vals.nlargest(top_n_ai).mean()) if n_ai >= 1 else float("nan"),
            "eco_max": float(occ_vals.max()) if n_occ >= 1 else float("nan"),
            "eco_p95": float(occ_vals.quantile(0.95)) if n_occ >= 2 else (float(occ_vals.iloc[0]) if n_occ == 1 else float("nan")),
            "eco_top10": float(occ_vals.nlargest(top_n_occ).mean()) if n_occ >= 1 else float("nan"),
            "eco_mean": float(occ_vals.mean()) if n_occ >= 1 else float("nan"),
        }
        if phys_map is not None:
            phys_vals = grp["pct_physical_occ"].dropna()
            phys_score = float(phys_vals.mean()) if len(phys_vals) else float("nan")
            rec["phys_score"] = phys_score
            rec["phys_tier"] = _phys_tier(phys_score) if pd.notna(phys_score) else "Non-physical"
        records.append(rec)

    return pd.DataFrame(records)


# O*NET subcategory maps (mirror ska_category_breakdown)
ABILITY_SUBCATEGORY: dict[str, str] = {
    "1.A.1.a": "Verbal", "1.A.1.b": "Idea Generation", "1.A.1.c": "Quantitative",
    "1.A.1.d": "Memory", "1.A.1.e": "Perceptual", "1.A.1.f": "Spatial",
    "1.A.1.g": "Attentiveness",
    "1.A.2.a": "Fine Manipulative", "1.A.2.b": "Control Movement", "1.A.2.c": "Reaction",
    "1.A.3.a": "Strength", "1.A.3.b": "Endurance",
    "1.A.3.c": "Flexibility, Balance, Coordination",
    "1.A.4.a": "Visual", "1.A.4.b": "Auditory and Speech",
}

SKILLS_SUBCATEGORY: dict[str, str] = {
    "2.A.1": "Content",
    "2.A.2": "Process",
    "2.B.1": "Social",
    "2.B.2": "Complex Problem Solving",
    "2.B.3": "Technical",
    "2.B.4": "Systems",
    "2.B.5": "Resource Management",
}

KNOWLEDGE_CATEGORY: dict[str, str] = {
    "2.C.1": "Business and Management",
    "2.C.2": "Manufacturing and Production",
    "2.C.3": "Engineering and Technology",
    "2.C.4": "Mathematics and Science",
    "2.C.5": "Health Services",
    "2.C.6": "Education and Training",
    "2.C.7": "Arts and Humanities",
    "2.C.8": "Law and Public Safety",
    "2.C.9": "Communications",
    "2.C.10": "Transportation",
}

# Muted red for AI markers — visible on blue without being aggressive
AI_MARKER_COLOR = "#a04444"
# Workforce Mean marker — dark gray, intentionally NOT pure black so it
# stays distinguishable from the black in-chart bar value text when the
# marker overlaps a digit. Opacity is applied per-trace so the digit
# shines through when they collide.
WORKFORCE_MEAN_COLOR = "#3a3a3a"
# Chart markers render at this opacity so overlapping bar value digits
# remain readable. The legend swatches mirror this with the matching rgba
# below so the legend visually matches what gets drawn on the bars.
SKA_MARKER_OPACITY = 0.75
AI_MARKER_COLOR_RGBA = f"rgba(160, 68, 68, {SKA_MARKER_OPACITY})"
WORKFORCE_MEAN_COLOR_RGBA = f"rgba(58, 58, 58, {SKA_MARKER_OPACITY})"

# SKA AI Top-10 bar color per phys-mix tier. Uses the same Physical / Mixed
# / Non-physical palette as the major trio so the structural cut tracks
# visually across Part 2.
SKA_BAR_COLOR_BY_TIER: dict[str, str] = {
    "Non-physical": METRIC_COLORS["tasks"],     # Slate blue — same as default tasks
    "Mixed":        METRIC_COLORS["wages"],     # Sage green
    "Physical":     METRIC_COLORS["workers"],   # Gold / yellow
}

# Legend ordering for the phys-mix swatches row (Physical → Mixed → Non-physical).
SKA_TIER_LEGEND_ORDER: tuple[str, ...] = ("Physical", "Mixed", "Non-physical")


# ── Text measurement for the manual legend ──────────────────────────────
# Estimating row width via `len(label) × constant` over-counts narrow
# glyphs (lowercase, digits, parens, the ↓ arrow), which makes wide rows
# render visibly left of center even when the math claims they're
# centered. Measuring against the real Inter TTF fixes that.
_INTER_TTF_CANDIDATES = (
    r"C:\Users\teddy\AppData\Local\Microsoft\Windows\Fonts\Inter-Regular.ttf",
    r"C:\Windows\Fonts\Inter-Regular.ttf",
    r"C:\Windows\Fonts\Inter.ttf",
)
_INTER_FONT_CACHE: dict[int, object] = {}


def _measure_text_px(label: str, font_size_px: int) -> float:
    """Return the rendered pixel width of `label` in Inter at `font_size_px`.

    Falls back to the rough char-factor estimate if Pillow or the Inter TTF
    isn't available — the estimate is fine for short, capital-heavy labels;
    the measurement only matters for the long mixed-case rows like
    'AI Top-10 Avg  (color = phys tier ↓)'."""
    try:
        from PIL import ImageFont
    except Exception:
        return len(label) * (font_size_px * 0.65)

    font = _INTER_FONT_CACHE.get(font_size_px)
    if font is None:
        from pathlib import Path as _P
        ttf = next(
            (p for p in _INTER_TTF_CANDIDATES if _P(p).exists()),
            None,
        )
        if ttf is None:
            return len(label) * (font_size_px * 0.65)
        font = ImageFont.truetype(ttf, font_size_px)
        _INTER_FONT_CACHE[font_size_px] = font

    # `getlength` returns the advance width in pixels for the given string
    # at the loaded font size — exactly the metric Plotly's renderer uses
    # when laying out text at the same logical size.
    return float(font.getlength(label))


def draw_ska_manual_legend(
    fig: go.Figure,
    rows: list[list[tuple[str, str, str, str]]],
    *,
    fig_width: int,
    fig_height: int,
    margin_l: int,
    margin_r: int,
    margin_t: int,
    margin_b: int,
    legend_font_px: int,
) -> None:
    """Render N rows of legend entries below the plot as shapes + annotations.

    Each entry is a (kind, color, symbol_char, label) tuple where kind is
    "bar" (renders a colored rectangle swatch) or "marker" (renders a
    unicode glyph at `color`). Used by every SKA chart so swatch/symbol
    sizing stays identical across main body and appendix."""
    # Sizes — chosen as middle ground; same across all SKA charts.
    # Pitch/swatch tightened from 50/20 so the legend block doesn't
    # dominate the bottom margin of these (already tall) charts.
    swatch_w, swatch_h = 26, 16
    text_gap = 12
    inter_gap = 50
    row_pitch = 34
    symbol_font_px = legend_font_px + 4

    plot_left_px = margin_l
    plot_right_px = fig_width - margin_r
    plot_top_px = margin_t
    plot_bottom_px = fig_height - margin_b
    plot_w_px = plot_right_px - plot_left_px
    plot_h_px = plot_bottom_px - plot_top_px
    half_h_paper = swatch_h / (2 * plot_h_px)

    def fig_x_to_paper(x):
        return (x - plot_left_px) / plot_w_px

    def fig_y_to_paper(y_from_top):
        return (plot_bottom_px - y_from_top) / plot_h_px

    # Row 1 sits ~100 px below the plot bottom — clears the x-axis ticks and
    # the axis title (which itself sits 50–60 px below the plot via standoff),
    # with a small whitespace buffer before the legend.
    row1_from_top = plot_bottom_px + 100

    # Each row is independently centered on the figure midpoint so the
    # legend is centered with respect to the whole PNG (not the plot area,
    # which can sit off-center when the left margin is much wider than the
    # right margin to accommodate y-axis labels).
    fig_center_px = fig_width / 2

    for r, entries in enumerate(rows):
        y_from_top = row1_from_top + r * row_pitch
        y_paper = fig_y_to_paper(y_from_top)

        item_widths = []
        for _, _, _, label in entries:
            text_w = _measure_text_px(label, legend_font_px)
            item_widths.append(swatch_w + text_gap + text_w)
        total_w = sum(item_widths) + inter_gap * (len(entries) - 1)
        cursor_x = fig_center_px - total_w / 2

        for (kind, color, symbol, label), iw in zip(entries, item_widths):
            sx0 = cursor_x
            x_left_p = fig_x_to_paper(sx0)
            x_right_p = fig_x_to_paper(sx0 + swatch_w)
            x_mid_p = fig_x_to_paper(sx0 + swatch_w / 2)
            x_text_p = fig_x_to_paper(sx0 + swatch_w + text_gap)
            if kind == "bar":
                fig.add_shape(
                    type="rect", xref="paper", yref="paper",
                    x0=x_left_p, x1=x_right_p,
                    y0=y_paper - half_h_paper, y1=y_paper + half_h_paper,
                    fillcolor=color, line=dict(width=0), layer="above",
                )
            else:
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=x_mid_p, y=y_paper,
                    text=symbol, showarrow=False,
                    xanchor="center", yanchor="middle",
                    font=dict(size=symbol_font_px, color=color,
                              family=FONT_FAMILY),
                )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=x_text_p, y=y_paper,
                text=label, showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=legend_font_px, color=PAPER_PALETTE["neutral"],
                          family=FONT_FAMILY),
            )
            cursor_x += iw + inter_gap


def ska_legend_rows() -> list[list[tuple[str, str, str, str]]]:
    """The three rows used by every SKA chart."""
    return [
        [
            ("bar", "#e8e8e2", "■", "Workforce Max"),
            ("bar", SKA_BAR_COLOR_BY_TIER["Mixed"], "■",
             "AI Top-10 Avg  (color = phys tier ↓)"),
        ],
        [("bar", SKA_BAR_COLOR_BY_TIER[t], "■", t)
         for t in SKA_TIER_LEGEND_ORDER],
        [
            ("marker", AI_MARKER_COLOR_RGBA, "◆", "AI Max"),
            ("marker", WORKFORCE_MEAN_COLOR_RGBA, "●", "Workforce Mean"),
        ],
    ]


def _major_phys_mix_shares(occ_struct: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-major % of occupations in each phys/mixed/non-phys bucket.

    `occ_struct` is the output of `_load_occ_structural()`. We join it to
    eco_2025's title_current → major mapping and tally occ_group shares
    per major. The result is consumed by panel 6 of the major trio."""
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    occ_to_major = (
        eco[["title_current", "major_occ_category"]]
        .drop_duplicates()
        .set_index("title_current")["major_occ_category"]
    )
    df = occ_struct[["title_current", "occ_group"]].copy()
    df["major"] = df["title_current"].map(occ_to_major)
    df = df.dropna(subset=["major"])

    out: dict[str, dict[str, float]] = {}
    for major, grp in df.groupby("major"):
        total = len(grp)
        if total == 0:
            continue
        counts = grp["occ_group"].value_counts()
        out[str(major)] = {
            "pct_physical":     float(counts.get("Physical", 0)     / total * 100.0),
            "pct_mixed":        float(counts.get("Mixed", 0)        / total * 100.0),
            "pct_non_physical": float(counts.get("Non-physical", 0) / total * 100.0),
            "n_occs":           total,
        }
    return out


def _gwa_phys_task_shares() -> dict[str, dict[str, float]]:
    """Per-GWA % of tasks that are physical vs non-physical.

    Dedupes to (task_normalized, gwa_title) before tallying, since eco_2025
    expands tasks across the work-activity hierarchy. Two-segment readout
    (no "mixed" — task physical flag is binary)."""
    cols = ["task_normalized", "gwa_title", "physical"]
    df = pd.read_csv(DATA_DIR / "final_eco_2025.csv", usecols=cols)
    df = df.dropna(subset=["gwa_title"])
    df = df.groupby(["task_normalized", "gwa_title"], sort=False, as_index=False).first()
    df["physical_bool"] = df["physical"].apply(_coerce_phys_bool)

    out: dict[str, dict[str, float]] = {}
    for gwa, grp in df.groupby("gwa_title"):
        total = len(grp)
        if total == 0:
            continue
        n_phys = int(grp["physical_bool"].sum())
        out[str(gwa)] = {
            "pct_physical":     float(n_phys / total * 100.0),
            "pct_non_physical": float((total - n_phys) / total * 100.0),
            "n_tasks":          total,
        }
    return out


def _load_occ_phys_map() -> pd.Series:
    """title_current → pct_physical (occ-level), used to color SKA element
    rows by the average physicality of their user base. Counts UNIQUE
    (occ, task) pairs (eco_2025 expands tasks across GWA/IWA/DWA, and that
    expansion is not proportional between physical and non-physical tasks),
    so dedup is required before the n_physical / n_tasks division. Matches
    `_load_occ_structural` and the dashboard backend pipeline."""
    eco = pd.read_csv(DATA_DIR / "final_eco_2025.csv",
                       usecols=["title_current", "task_normalized", "physical"])
    eco_unique = eco.drop_duplicates(["title_current", "task_normalized"])
    eco_unique["physical_bool"] = eco_unique["physical"].apply(_coerce_phys_bool)
    grouped = eco_unique.groupby("title_current")["physical_bool"].agg(["sum", "count"])
    pct_phys = (grouped["sum"] / grouped["count"] * 100.0).fillna(0.0)
    return pct_phys


def _phys_tier(pct_physical: float) -> str:
    if pct_physical > PHYS_UPPER:
        return "Physical"
    if pct_physical < PHYS_LOWER:
        return "Non-physical"
    return "Mixed"


def _ability_subcat(eid: str) -> str:
    parts = eid.split(".")
    sub_key = ".".join(parts[:4]) if len(parts) >= 4 else ".".join(parts[:3])
    return ABILITY_SUBCATEGORY.get(sub_key, "Other")


def _skills_subcat(eid: str) -> str:
    parts = eid.split(".")
    cat_key = ".".join(parts[:3]) if len(parts) >= 3 else ""
    return SKILLS_SUBCATEGORY.get(cat_key, "Other")


def _knowledge_cat(eid: str) -> str:
    parts = eid.split(".")
    cat_key_3 = ".".join(parts[:3]) if len(parts) >= 3 else ""
    if cat_key_3 in KNOWLEDGE_CATEGORY:
        return KNOWLEDGE_CATEGORY[cat_key_3]
    cat_key_2 = ".".join(parts[:2]) if len(parts) >= 2 else ""
    return KNOWLEDGE_CATEGORY.get(cat_key_2, "Other")


def _compute_subcategory_rollup(
    onet_path: Path,
    pct_series: pd.Series,
    cat_fn,
    phys_map: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Per-subcategory rollup of element-level AI capability metrics.

    For each element: imp×lv per (occ, element) row at imp ≥ 3,
    then ai_top10 = mean of top-10 ai_product across occs, ai_max =
    max ai_product, eco_max = max occ_score, eco_mean = mean occ_score.
    Element-level value rolled to subcategory by mean of (metric / eco_max × 100).
    """
    df = pd.read_csv(onet_path, dtype=str)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code", "Title": "title",
        "Element ID": "element_id", "Element Name": "element_name",
        "Scale ID": "scale_id", "Data Value": "data_value",
    })
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["scale_id"].isin(["IM", "LV"])]

    pivoted = (
        df.pivot_table(
            index=["soc_code", "title", "element_id", "element_name"],
            columns="scale_id", values="data_value", aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"}).dropna(
        subset=["importance", "level"]
    )

    pivoted["pct"] = pivoted["title"].map(pct_series)
    pivoted = pivoted.dropna(subset=["pct"])
    pivoted = pivoted[pivoted["importance"] >= IMPORTANCE_THRESHOLD].copy()

    pivoted["occ_score"] = pivoted["importance"] * pivoted["level"]
    pivoted["ai_product"] = (pivoted["pct"] / 100.0) * pivoted["occ_score"]
    if phys_map is not None:
        pivoted["pct_physical_occ"] = pivoted["title"].map(phys_map)

    elem_rows = []
    for (eid, ename), grp in pivoted.groupby(["element_id", "element_name"]):
        ai_vals = grp["ai_product"]
        occ_vals = grp["occ_score"]
        n = len(ai_vals)
        top_n = min(TOP_N_FOR_AVERAGE, n)
        rec = {
            "element_id": eid,
            "element_name": ename,
            "subcategory": cat_fn(eid),
            "ai_top10": float(ai_vals.nlargest(top_n).mean()),
            "ai_p95": float(ai_vals.quantile(0.95)) if n >= 2 else float(ai_vals.iloc[0]),
            "ai_max": float(ai_vals.max()),
            "eco_max": float(occ_vals.max()),
            "eco_mean": float(occ_vals.mean()),
        }
        if phys_map is not None:
            phys_vals = grp["pct_physical_occ"].dropna()
            rec["phys_score"] = float(phys_vals.mean()) if len(phys_vals) else float("nan")
        elem_rows.append(rec)
    elem_df = pd.DataFrame(elem_rows)
    for col in ["ai_top10", "ai_p95", "ai_max", "eco_mean"]:
        elem_df[f"{col}_pct"] = elem_df[col] / elem_df["eco_max"] * 100.0

    cat_rows = []
    for sub, grp in elem_df.groupby("subcategory"):
        row = {
            "subcategory": sub,
            "n_elements": len(grp),
            "ai_top10_pct": float(grp["ai_top10_pct"].mean()),
            "ai_p95_pct":   float(grp["ai_p95_pct"].mean()),
            "ai_max_pct":   float(grp["ai_max_pct"].mean()),
            "eco_mean_pct": float(grp["eco_mean_pct"].mean()),
        }
        if "phys_score" in grp.columns:
            phys_vals = grp["phys_score"].dropna()
            phys_score_cat = float(phys_vals.mean()) if len(phys_vals) else float("nan")
            row["phys_score"] = phys_score_cat
            row["phys_tier"]  = _phys_tier(phys_score_cat) if pd.notna(phys_score_cat) else "Non-physical"
        cat_rows.append(row)
    return (
        pd.DataFrame(cat_rows)
        .sort_values("ai_top10_pct", ascending=False)
        .reset_index(drop=True)
    )


def _build_ska_subcategory_chart(
    knowledge_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    n_know_elements: int,
    n_abil_elements: int,
    results: Path, figures: Path,
) -> None:
    """Knowledge + Abilities at subcategory level. Each cell is a mean
    across the elements in that subcategory. Bar = AI Top-10 mean (% of
    workforce max), colored by phys-mix tier. Red diamond = AI Max %.
    Black dot = workforce mean %."""
    px = paper_fonts(PAPER_W)

    n_know = len(knowledge_df)
    n_abil = len(abilities_df)
    total = n_know + n_abil
    row_heights = [n_know / total, n_abil / total]

    # Pick a left margin that fits the longest "Subcategory  (n=X)" label
    # (~7 px/char at 8 pt Inter — y-tick dropped to the floor below) plus
    # the rotated y-axis title (~30 px) and a small left-edge buffer.
    max_label = max(
        len(f"{r.subcategory}  (n={int(r.n_elements)})")
        for df in (knowledge_df, abilities_df) for r in df.itertuples()
    )
    margin_l = max(180, max_label * 11 + 90)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.10,
        subplot_titles=[
            f"Knowledge  ({n_know} subcategories | {n_know_elements} elements)",
            f"Abilities  ({n_abil} subcategories | {n_abil_elements} elements)",
        ],
    )

    for row, df in enumerate([knowledge_df, abilities_df], start=1):
        sub_labels = [
            f"{r.subcategory}  (n={int(r.n_elements)})"
            for r in df.itertuples()
        ]
        fig.add_trace(go.Bar(
            y=sub_labels, x=[100] * len(sub_labels), orientation="h",
            name="Workforce Max",
            marker=dict(color="#e8e8e2", line=dict(width=0)),
            showlegend=False,
            hovertemplate="Workforce max: 100%<extra></extra>",
        ), row=row, col=1)

        if "phys_tier" in df.columns:
            bar_colors = [SKA_BAR_COLOR_BY_TIER.get(t, METRIC_COLORS["tasks"])
                          for t in df["phys_tier"].fillna("Non-physical")]
        else:
            bar_colors = [METRIC_COLORS["tasks"]] * len(sub_labels)

        fig.add_trace(go.Bar(
            y=sub_labels, x=df["ai_top10_pct"], orientation="h",
            name="AI Top-10 Avg",
            marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
            showlegend=False,
            text=[f"{v:.0f}%" for v in df["ai_top10_pct"]],
            textposition="outside",
            textfont=dict(size=px["in_chart_floor"],
                          color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            hovertemplate="AI Top-10 avg (% of max): %{x:.1f}%<extra></extra>",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            y=sub_labels, x=df["ai_max_pct"], mode="markers",
            name="AI Max",
            marker=dict(color=AI_MARKER_COLOR, symbol="diamond", size=14,
                        opacity=0.75),
            showlegend=False,
            hovertemplate="AI Max (% of max): %{x:.1f}%<extra></extra>",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            y=sub_labels, x=df["eco_mean_pct"], mode="markers",
            name="Workforce Mean",
            marker=dict(color=WORKFORCE_MEAN_COLOR, symbol="circle", size=10,
                        opacity=0.75,
                        line=dict(width=1, color=WORKFORCE_MEAN_COLOR)),
            showlegend=False,
            hovertemplate="Workforce mean (% of max): %{x:.1f}%<extra></extra>",
        ), row=row, col=1)

        fig.update_yaxes(
            autorange="reversed", row=row, col=1,
            type="category",
            tickmode="array", tickvals=sub_labels, ticktext=sub_labels,
            tickfont=dict(size=px["in_chart_floor"], color=PAPER_PALETTE["text"],
                          family=FONT_FAMILY),
            showgrid=False, showline=False,
        )
        fig.update_xaxes(
            range=[0, 100], ticksuffix="%",
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
            tickfont=dict(size=px["tick"], color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
            showline=False, zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
            row=row, col=1,
        )

    # Y axis titles per subplot — large standoff pushes the rotated title
    # flush with the figure's left edge. Title font shrunk to 9 pt (tick
    # size) so the rotated text fits within the compressed panel height.
    yt_standoff = max(0, margin_l - 40)
    # "O*NET" prefix dropped (already present in the chart title) so the
    # rotated title fits within the compressed panel 1 height (10 rows
    # × 32 px = 320 px plot).
    fig.update_yaxes(
        title=dict(text="Knowledge Subcategory",
                   font=dict(size=px["tick"], family=FONT_FAMILY),
                   standoff=yt_standoff),
        automargin=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="Ability Subcategory",
                   font=dict(size=px["tick"], family=FONT_FAMILY),
                   standoff=yt_standoff),
        automargin=False,
        row=2, col=1,
    )
    fig.update_xaxes(
        title=dict(
            text="AI Capability as % of Workforce Max (Subcategory Average)",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
            standoff=15,
        ),
        row=2, col=1,
    )

    margin_t, margin_b = 120, 230
    margin_r = 80
    fig_height = max(700, total * 32 + margin_t + margin_b)

    draw_ska_manual_legend(
        fig, ska_legend_rows(),
        fig_width=PAPER_W, fig_height=fig_height,
        margin_l=margin_l, margin_r=margin_r,
        margin_t=margin_t, margin_b=margin_b,
        legend_font_px=px["legend"],
    )

    fig.update_layout(
        title=dict(
            text="AI Capability as % of Workforce Max — O*NET Knowledge and Abilities",
            font=dict(size=px["title"], color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left",
        ),
        height=fig_height,
        width=PAPER_W,
        font=dict(family=FONT_FAMILY, size=px["tick"], color=PAPER_PALETTE["text"]),
        plot_bgcolor=PAPER_PALETTE["surface"],
        paper_bgcolor=PAPER_PALETTE["surface"],
        barmode="overlay",
        showlegend=False,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
    )

    panel_starts = ("Knowledge  ", "Abilities  ")
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and any(ann.text.startswith(s) for s in panel_starts):
            ann.font = dict(size=px["panel_title"], family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    save_figure(fig, results / "figures" / "ska_knowledge_abilities.png", scale=2)
    _copy_fig(results, figures, "ska_knowledge_abilities.png")
    print("  -> ska_knowledge_abilities.png")


def build_ska_levels(results: Path, figures: Path) -> None:
    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    ska_data = load_ska_data()
    phys_map = _load_occ_phys_map()

    # Skills — element level (per-element bars)
    skills_df = _compute_ska_variants(ska_data.skills, pct, "skills", phys_map=phys_map)
    print(f"    skills: {len(skills_df)} elements")

    # Knowledge — subcategory rollup (10 categories from O*NET 2.C.1–2.C.10)
    know_path = REFERENCE_DIR / "knowledge_v30.1.csv"
    knowledge_cat = _compute_subcategory_rollup(know_path, pct, _knowledge_cat, phys_map=phys_map)
    n_know_elements = _count_elements(know_path, pct)
    print(f"    knowledge (subcategory): {len(knowledge_cat)} subcategories "
          f"({n_know_elements} elements)")

    # Abilities — subcategory rollup (15 subcategories under 1.A.1–1.A.4)
    abil_path = REFERENCE_DIR / "abilities_v30.1.csv"
    abilities_cat = _compute_subcategory_rollup(abil_path, pct, _ability_subcat, phys_map=phys_map)
    n_abil_elements = _count_elements(abil_path, pct)
    print(f"    abilities (subcategory): {len(abilities_cat)} subcategories "
          f"({n_abil_elements} elements)")

    save_csv(skills_df, results / "ska_skills.csv", float_format="%.4f")
    save_csv(knowledge_cat, results / "ska_knowledge_by_subcategory.csv",
             float_format="%.2f")
    save_csv(abilities_cat, results / "ska_abilities_by_subcategory.csv",
             float_format="%.2f")

    # Main-body skills chart is the full-element version (previously lived
    # in the appendix as ska_skills_full). Same bar/marker/legend framing
    # as the knowledge/abilities subcategory chart below; replaces the
    # earlier subcategory-rollup-only main-body skills chart.
    from lib.builders.appendix import (
        _build_one_ska_full_chart, _element_subcat_lookup,
    )
    skill_subcat = _element_subcat_lookup(
        REFERENCE_DIR / "skills_v30.1.csv", _skills_subcat,
    )
    _build_one_ska_full_chart(
        skills_df, "Skills", skill_subcat,
        "ska_skills.png", results, figures,
    )

    _build_ska_subcategory_chart(
        knowledge_cat, abilities_cat,
        n_know_elements, n_abil_elements,
        results, figures,
    )

    # Tidy-up: remove the old combined chart if it lingers from earlier runs
    for stale in ("ska_levels.png",):
        for d in (results / "figures", figures):
            p = d / stale
            if p.exists():
                p.unlink()


def _count_elements(onet_path: Path, pct_series: pd.Series) -> int:
    """Number of unique elements after the pct/imp filter — for label use."""
    df = pd.read_csv(onet_path, dtype=str)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code", "Title": "title",
        "Element ID": "element_id", "Scale ID": "scale_id",
        "Data Value": "data_value",
    })
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["scale_id"].isin(["IM", "LV"])]
    pivoted = (
        df.pivot_table(
            index=["soc_code", "title", "element_id"],
            columns="scale_id", values="data_value", aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"}).dropna(
        subset=["importance", "level"]
    )
    pivoted["pct"] = pivoted["title"].map(pct_series)
    pivoted = pivoted.dropna(subset=["pct"])
    pivoted = pivoted[pivoted["importance"] >= IMPORTANCE_THRESHOLD]
    return int(pivoted["element_id"].nunique())


# ─────────────────────────────────────────────────────────────────────────
# Chart 4: All GWAs by % Tasks Affected
# ─────────────────────────────────────────────────────────────────────────

def build_gwa_chart(results: Path, figures: Path) -> None:
    """Five-panel GWA quintet matching the major trio: variant A % |
    variant B % | all_confirmed % | workers | wages. All ~41 GWAs visible,
    shared y-axis ordered by all_confirmed % tasks descending."""
    base = _get_wa_data(PRIMARY_DATASET, "gwa")
    variant_a = compute_variant_a_gwa()
    variant_b_df = _get_wa_data_with_phys(PRIMARY_DATASET, "gwa", physical_mode="exclude")
    assert not base.empty,      "all_confirmed GWA data is empty"
    assert not variant_a.empty, "variant_a GWA data is empty"
    assert not variant_b_df.empty, "variant_b GWA data is empty"

    base = base.sort_values("pct_tasks_affected", ascending=False).reset_index(drop=True)
    a_map = variant_a.set_index("category")["pct_tasks_affected"]
    b_map = variant_b_df.set_index("category")["pct_tasks_affected"]
    base["pct_a"] = base["category"].map(a_map)
    base["pct_b"] = base["category"].map(b_map)
    save_csv(base, results / "gwa_exposure.csv")

    n_gwas = len(base)

    # Per-GWA phys task shares for panel 6.
    gwa_phys = _gwa_phys_task_shares()

    # Reverse for plotly: top of chart = highest-pct GWA.
    categories_r = list(reversed(base["category"].tolist()))
    pct_a_r     = list(reversed(base["pct_a"].fillna(0.0).tolist()))
    pct_b_r     = list(reversed(base["pct_b"].fillna(0.0).tolist()))
    pct_r       = list(reversed(base["pct_tasks_affected"].tolist()))
    workers_r   = list(reversed(base["workers_affected"].tolist()))
    wages_r     = list(reversed(base["wages_affected"].tolist()))
    pct_phys_r  = [gwa_phys.get(c, {}).get("pct_physical",     0.0) for c in categories_r]
    pct_non_r   = [gwa_phys.get(c, {}).get("pct_non_physical", 0.0) for c in categories_r]

    fig = make_subplots(
        rows=1, cols=6,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=[
            "Variant A: Non-Phys Task Share",
            "Variant B: % in Non-Phys Work",
            "% Tasks Exposed (All Confirmed)",
            "Workers Exposed (All Confirmed)",
            "Wages Exposed (All Confirmed)",
            "Phys Mix (Tasks)",
        ],
        column_widths=[1.0, 1.0, 1.0, 1.0, 1.0, 0.55],
    )

    VARIANT_A_COLOR = "#7a9ab8"
    VARIANT_B_COLOR = "#3a6f8f"

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_a_r, orientation="h",
        marker=dict(color=VARIANT_A_COLOR, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in pct_a_r],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=INSIDE_FS - 2, color="white", family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
        hovertemplate="Variant A: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_b_r, orientation="h",
        marker=dict(color=VARIANT_B_COLOR, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in pct_b_r],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=INSIDE_FS - 2, color="white", family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
        hovertemplate="Variant B: %{x:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_r, orientation="h",
        marker=dict(color=METRIC_COLORS["tasks"], line=dict(width=0)),
        text=[f"{v:.0f}%" for v in pct_r],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=INSIDE_FS - 2, color="white", family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
        hovertemplate="All Confirmed: %{x:.1f}%<extra></extra>",
    ), row=1, col=3)

    fig.add_trace(go.Bar(
        y=categories_r, x=workers_r, orientation="h",
        marker=dict(color=METRIC_COLORS["workers"], line=dict(width=0)),
        text=[fmt_workers(v) for v in workers_r],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=INSIDE_FS - 2, color="white", family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
    ), row=1, col=4)

    fig.add_trace(go.Bar(
        y=categories_r, x=wages_r, orientation="h",
        marker=dict(color=METRIC_COLORS["wages"], line=dict(width=0)),
        text=[fmt_wages(v) for v in wages_r],
        textposition="inside", insidetextanchor="end",
        textfont=dict(size=INSIDE_FS - 2, color="white", family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
    ), row=1, col=5)

    # Panel 6 — per-GWA task phys-mix stacked bar (2-segment, since the
    # task physical flag is binary at the task level). Uses `base` to stack
    # manually so the other panels' single-trace bars don't get squeezed.
    fig.add_trace(go.Bar(
        y=categories_r, x=pct_phys_r, base=0, orientation="h",
        marker=dict(color=GROUP_COLORS["Physical"], line=dict(width=0)),
        name="% Physical tasks", showlegend=True,
        hovertemplate="Physical: %{x:.0f}%<extra></extra>",
    ), row=1, col=6)
    fig.add_trace(go.Bar(
        y=categories_r, x=pct_non_r, base=pct_phys_r, orientation="h",
        marker=dict(color=GROUP_COLORS["Non-physical"], line=dict(width=0)),
        name="% Non-physical tasks", showlegend=True,
        hovertemplate="Non-physical: %{x:.0f}%<extra></extra>",
    ), row=1, col=6)

    # Dummy invisible trace carrying just a dashed-line glyph for the legend,
    # so readers know what the 33/67 reference lines mean.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="#1a1a1a", width=1.4, dash="dash"),
        name="33% / 67% phys-mix bucket cuts",
        showlegend=True, hoverinfo="skip",
    ), row=1, col=6)

    height = max(PAPER_H + 600, n_gwas * 38 + 320)

    style_paper_figure(
        fig,
        "Task Exposure Across All O*NET General Work Activities — Variants A, B, and All Confirmed",
        subtitle=(
            f"All {n_gwas} GWAs ranked by All Confirmed % tasks exposed. "
            "Variant A = naive non-physical task share within the GWA's task pool. "
            "Variant B = AI exposure restricted to non-physical tasks. "
            "Right panel: share of each GWA's tasks classified as Physical vs Non-Physical."
        ),
        height=height,
        width=PAPER_W + 1000,
        margin=dict(l=40, r=80, t=170, b=140),
    )

    import math

    def _nice_ticks(max_val: float, n_ticks: int = 4) -> list[float]:
        if max_val <= 0:
            return [0.0]
        raw_step = max_val / (n_ticks - 1)
        magnitude = 10 ** math.floor(math.log10(raw_step))
        step = math.ceil(raw_step / magnitude) * magnitude
        ticks = [step * i for i in range(n_ticks + 1)]
        return [t for t in ticks if t <= max_val * 1.05]

    def _strip_zero_decimal(s: str) -> str:
        for unit in ("M", "B", "K", "T"):
            s = s.replace(f".0{unit}", unit)
        return s

    workers_max = float(base["workers_affected"].max())
    wages_max   = float(base["wages_affected"].max())
    worker_ticks = _nice_ticks(workers_max)
    wage_ticks   = _nice_ticks(wages_max)

    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
    )
    fig.update_xaxes(ticksuffix="%", title=dict(text="% (Variant A)",         font=dict(size=LABEL_FS - 4)), row=1, col=1)
    fig.update_xaxes(ticksuffix="%", title=dict(text="% (Variant B)",         font=dict(size=LABEL_FS - 4)), row=1, col=2)
    fig.update_xaxes(ticksuffix="%", title=dict(text="% Tasks Exposed",       font=dict(size=LABEL_FS - 4)), row=1, col=3)
    fig.update_xaxes(
        tickvals=worker_ticks,
        ticktext=[_strip_zero_decimal(fmt_workers(v)) for v in worker_ticks],
        title=dict(text="Workers Exposed", font=dict(size=LABEL_FS - 4)),
        row=1, col=4,
    )
    fig.update_xaxes(
        tickvals=wage_ticks,
        ticktext=[_strip_zero_decimal(fmt_wages(v)) for v in wage_ticks],
        title=dict(text="Wages Exposed", font=dict(size=LABEL_FS - 4)),
        row=1, col=5,
    )
    fig.update_xaxes(
        range=[0, 100], dtick=25,
        ticksuffix="%", tickfont=dict(size=TICK_FS - 4, family=FONT_FAMILY),
        title=dict(text="% of GWA's tasks", font=dict(size=LABEL_FS - 4)),
        row=1, col=6,
    )

    # 33% and 67% reference lines on the phys-mix panel — same thresholds
    # used to bucket occupations as Non-physical / Mixed / Physical, so a
    # GWA whose phys segment crosses 67 reads as predominantly physical,
    # and one whose phys segment stays under 33 reads as predominantly
    # non-physical. Dark gray with strong opacity so the line reads
    # consistently against colored bars and the white row gaps between
    # bars; "above" layer so the line is never hidden by the bars.
    fig.add_vline(
        x=33, layer="above",
        line=dict(color="#1a1a1a", width=1.4, dash="dash"),
        row=1, col=6, opacity=0.85,
    )
    fig.add_vline(
        x=67, layer="above",
        line=dict(color="#1a1a1a", width=1.4, dash="dash"),
        row=1, col=6, opacity=0.85,
    )

    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_yaxes(
        title=dict(text="O*NET General Work Activity", font=dict(size=LABEL_FS - 2)),
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        row=1, col=1,
    )

    panel_titles = {
        "Variant A: Non-Phys Task Share",
        "Variant B: % in Non-Phys Work",
        "% Tasks Exposed (All Confirmed)",
        "Workers Exposed (All Confirmed)",
        "Wages Exposed (All Confirmed)",
        "Phys Mix (Tasks)",
    }
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=LABEL_FS - 2, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    fig.update_layout(
        bargap=0.3,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.04, xanchor="center", x=0.5,
            font=dict(size=LEGEND_FS - 2, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )

    save_figure(fig, results / "figures" / "gwa_exposure.png", scale=2)
    _copy_fig(results, figures, "gwa_exposure.png")
    print("  -> gwa_exposure.png")


def _get_wa_data_with_phys(dataset_name: str, level: str, physical_mode: str) -> pd.DataFrame:
    """Variant of _get_wa_data that lets the caller override physical_mode
    (variant B uses 'exclude')."""
    from backend.compute import compute_work_activities
    settings = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": physical_mode,
        "geo": "nat",
        "sort_by": "workers_affected",
        "top_n": 9999,
    }
    result = compute_work_activities(settings)
    group = result.get("mcp_group") or result.get("aei_group")
    if group is None:
        return pd.DataFrame()
    rows = group.get(level, [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
# Chart 5: All 22 Major Categories — 3 Side-by-Side Panels
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# Major Categories — split into two paper figures:
#   1. major_categories_pct.png         — % Tasks Exposed | Variant A | Variant B
#   2. major_categories_wkrs_wages.png  — % Tasks Exposed | Workers | Wages
#
# Both use width = PAPER_W (1400) so the 8 pt paper floor = 24 px on canvas;
# fonts are bumped on these two charts only. % Tasks Exposed is the anchor
# panel on both charts, sorted identically.
# ─────────────────────────────────────────────────────────────────────────

# Wider canvas (2000 px) so each panel is ~2x wider than at the default
# 1400, which makes the bars themselves visually ~2x longer when pasted at
# 6.5". Fonts come from paper_fonts(W) so the printed pt sizes follow the
# standardized ladder (title 11 pt, panel/axis 10 pt, tick/legend 9 pt,
# in-chart floor 8 pt) — see analysis/paper/paper_config.py.
from lib.paper_config import paper_fonts as _paper_fonts
_MAJ_CANVAS_W    = 2000
_MAJ_PX          = _paper_fonts(_MAJ_CANVAS_W)
_MAJ_TITLE_FS    = _MAJ_PX["title"]            # 11 pt
_MAJ_PANEL_FS    = _MAJ_PX["panel_title"]      # 10 pt
_MAJ_LABEL_FS    = _MAJ_PX["axis_title"]       # 10 pt
_MAJ_TICK_FS     = _MAJ_PX["tick"]             # 9 pt — x-axis ticks
_MAJ_BARTEXT_FS  = _MAJ_PX["in_chart_floor"]   # 8 pt floor for bar value text
# Y-axis tick labels drop to the 8 pt floor so the per-row pitch can be
# compressed — 22 rows otherwise dominate the page. 8 pt is the floor;
# anything smaller would violate the paper-chart sizing rules.
_MAJ_YTICK_FS    = _MAJ_PX["in_chart_floor"]   # 8 pt

VARIANT_A_COLOR = "#7a9ab8"
VARIANT_B_COLOR = "#3a6f8f"


def _axis_max_and_ticks(max_val: float) -> tuple[float, list[float]]:
    """Tight axis range + clean tick values for a panel whose longest
    bar should fill ~95% of the panel width. Returns (range_max, ticks).

    Range extends 5% past max_val. Step is picked from {1, 2, 5} × 10^k
    so ticks land at clean intervals; we aim for 3 intervals across the
    range which yields 3–4 visible ticks."""
    import math
    if max_val <= 0:
        return 1.0, [0.0]
    range_max = max_val * 1.05
    raw_step = range_max / 3.0
    magnitude = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / magnitude
    # Step bases include 2.5 (so ranges ~75 pick step 25 → 4 ticks; ranges
    # ~650 pick step 250 → 3 ticks). The norm<3.0 threshold for step 2.5
    # (was 3.5) pushes ranges ~100 into step 50 instead — 3 ticks
    # [0, 50, 100] instead of 5 [0, 25, 50, 75, 100] — which prevents
    # crowding on narrower panels (e.g. GWA, where long y-labels claim
    # more left margin via automargin).
    if norm < 1.5:
        step = 1.0 * magnitude
    elif norm < 2.25:
        step = 2.0 * magnitude
    elif norm < 3.0:
        step = 2.5 * magnitude
    elif norm < 7.0:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude
    n_max = int(range_max / step)
    ticks = [float(step * i) for i in range(n_max + 1)]
    return float(range_max), ticks


def _balanced_wrap(label: str, max_chars: int = 22) -> str:
    """Pick the 2-line break (after any comma or space) that minimizes
    the longer line's length. Balanced lines visually read as
    center-aligned even though Plotly anchors y-axis tick labels to the
    axis line, so no single line pokes far left of the others. Returns
    the original label if it fits within `max_chars` or has no break
    point."""
    if len(label) <= max_chars:
        return label
    candidates: list[tuple[int, int, str, str]] = []
    for i in range(1, len(label)):
        if label[i - 1] in (",", " "):
            line1 = label[:i].rstrip(", ").rstrip()
            line2 = label[i:].lstrip()
            if not line1 or not line2:
                continue
            # Preserve a trailing comma on line 1 if the break came right
            # after one — reads more naturally as "Arts, Design,..." than
            # "Arts, Design".
            if label[i - 1] == ",":
                line1 = line1 + ","
            candidates.append((max(len(line1), len(line2)), i, line1, line2))
    if not candidates:
        return label
    candidates.sort(key=lambda t: (t[0], t[1]))
    _, _, line1, line2 = candidates[0]
    return f"{line1}<br>{line2}"


def _wrap_major_label(label: str, max_chars: int = 22) -> str:
    """Major-cat y-axis labels — strip the redundant ' Occupations'
    suffix (every major ends in it) and then balanced-wrap."""
    return _balanced_wrap(label.replace(" Occupations", ""), max_chars)


# Single-line GWA labels for paper charts. Long O*NET names get a
# hand-picked shorter form so 41 rows fit on one page without wrapping
# (a per-row pitch tight enough for one line of 8 pt text would cause
# 2-line wraps to collide). Shortenings preserve the semantic core of
# the GWA — trimming connectives, abbreviating "Information" → "Info"
# only where needed, and using "/" for list-or-list expansions.
_GWA_SHORT_LABELS: dict[str, str] = {
    "Interpreting the Meaning of Information for Others":
        "Interpreting Information for Others",
    "Communicating with People Outside the Organization":
        "Communicating with People Outside Org",
    "Establishing and Maintaining Interpersonal Relationships":
        "Establishing Interpersonal Relationships",
    "Providing Consultation and Advice to Others":
        "Providing Consultation and Advice",
    "Organizing, Planning, and Prioritizing Work":
        "Planning and Prioritizing Work",
    "Performing for or Working Directly with the Public":
        "Performing for or with the Public",
    "Judging the Qualities of Objects, Services, or People":
        "Judging Qualities of Objects/People",
    "Evaluating Information to Determine Compliance with Standards":
        "Evaluating Compliance with Standards",
    "Resolving Conflicts and Negotiating with Others":
        "Resolving Conflicts and Negotiating",
    "Communicating with Supervisors, Peers, or Subordinates":
        "Communicating with Supervisors/Peers",
    "Estimating the Quantifiable Characteristics of Products, Events, or Information":
        "Estimating Quantifiable Characteristics",
    "Identifying Objects, Actions, and Events":
        "Identifying Objects/Actions/Events",
    "Guiding, Directing, and Motivating Subordinates":
        "Guiding and Directing Subordinates",
    "Monitoring Processes, Materials, or Surroundings":
        "Monitoring Processes/Materials",
    "Inspecting Equipment, Structures, or Materials":
        "Inspecting Equipment/Structures",
    "Repairing and Maintaining Mechanical Equipment":
        "Repairing Mechanical Equipment",
    "Repairing and Maintaining Electronic Equipment":
        "Repairing Electronic Equipment",
    "Performing General Physical Activities":
        "Performing Physical Activities",
    "Operating Vehicles, Mechanized Devices, or Equipment":
        "Operating Vehicles and Equipment",
}


def _wrap_gwa_label(label: str, max_chars: int = 32) -> str:
    """GWA y-axis labels — force single line via a hand-shortened map so
    the 41-row chart fits one page. Falls back to the original label
    (no wrap) when no short form is registered. `max_chars` retained for
    signature compatibility with `_wrap_major_label` callers but
    unused — multi-line wrap is what we're avoiding."""
    return _GWA_SHORT_LABELS.get(label, label)




def _major_base_data() -> pd.DataFrame:
    """Shared loader for both split major-cat charts. Returns one row per
    major with pct_tasks_affected (All Confirmed), pct_a, pct_b,
    workers_affected, wages_affected — sorted by All Confirmed % tasks
    desc so both figures share the same y-ordering."""
    base = _run_config(PRIMARY_DATASET, "major")
    variant_a = compute_variant_a("major")
    variant_b = _run_config(PRIMARY_DATASET, "major", physical_mode="exclude")
    assert not base.empty,      "all_confirmed major data is empty"
    assert not variant_a.empty, "variant_a major data is empty"
    assert not variant_b.empty, "variant_b major data is empty"

    base = base.sort_values("pct_tasks_affected", ascending=False).reset_index(drop=True)
    a_map = variant_a.set_index("category")["pct_tasks_affected"]
    b_map = variant_b.set_index("category")["pct_tasks_affected"]
    base["pct_a"] = base["category"].map(a_map)
    base["pct_b"] = base["category"].map(b_map)
    return base


def _nice_ticks(max_val: float, n_ticks: int = 5) -> list[float]:
    import math
    if max_val <= 0:
        return [0.0]
    raw_step = max_val / (n_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = math.ceil(raw_step / magnitude) * magnitude
    ticks = [step * i for i in range(n_ticks + 1)]
    return [t for t in ticks if t <= max_val * 1.05]


def _strip_zero_decimal(s: str) -> str:
    for unit in ("M", "B", "K", "T"):
        s = s.replace(f".0{unit}", unit)
    return s


def _style_major_split(
    fig: go.Figure,
    title: str,
    n_cats: int,
    panel_titles: set[str],
    bottom_margin: int = 180,
) -> None:
    """Shared layout for the two split major-cat charts. Width = PAPER_W,
    fonts pulled from paper_fonts(W) so print pt sizes follow the
    standardized ladder. Margins sized to leave room for the title,
    x-tick + x-axis title at bottom, and the (now ' Occupations'-stripped)
    major-cat labels on the left — tight enough to give the bars
    themselves the bulk of the canvas width.

    `bottom_margin` defaults to 180 px (room for 2-line axis titles like
    the pct chart's "Hypothetical Exposure if / All Non-Phys Automatable").
    Charts with 1-line axis titles (e.g. workers/wages) pass a smaller
    value to avoid empty space at the bottom — the figure's total height
    is recomputed to match."""
    # Wide canvas (2000 px) so each panel has ~2x the horizontal room,
    # which (combined with the abbreviated y-labels) lets the bars
    # themselves stretch out further. Per-row pitch tightened to 90
    # (down from 130) — with 8 pt y-tick font, two wrapped lines fit
    # in ~82 px so 90 leaves a small breathing gap. Top margin cut to
    # 110 (title is 11 pt — generous, but kept above the plot panels).
    TOP_MARGIN = 110
    PER_ROW = 90
    BOTTOM_DEFAULT = 180
    height = max(PAPER_H + 200, n_cats * PER_ROW + TOP_MARGIN + BOTTOM_DEFAULT)
    # Add/subtract the bottom-margin delta from total height so the
    # plot area stays constant whether the caller asks for less
    # (1-line title) or more (3-line title) bottom space.
    height += bottom_margin - BOTTOM_DEFAULT
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=_MAJ_TITLE_FS, color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left",
            y=0.99, yanchor="top",
        ),
        font=dict(family=FONT_FAMILY, color=PAPER_PALETTE["text"]),
        plot_bgcolor=PAPER_PALETTE["surface"],
        paper_bgcolor=PAPER_PALETTE["surface"],
        width=_MAJ_CANVAS_W,
        height=height,
        margin=dict(l=110, r=110, t=TOP_MARGIN, b=bottom_margin),
        bargap=0.15,
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=_MAJ_TICK_FS, family=FONT_FAMILY),
        tickangle=0,
        ticklabelstandoff=8,
    )
    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_yaxes(
        tickfont=dict(size=_MAJ_YTICK_FS, family=FONT_FAMILY),
        # Compact left-side text apparatus — pull tick labels close to
        # the axis line, and pull the y-axis title close to the labels.
        # automargin still claims enough room for the longest label,
        # but the standoffs cut the slack out of the zone.
        automargin=True,
        ticklabelstandoff=2,
        ticks="",
        ticklen=0,
        row=1, col=1,
    )
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=_MAJ_PANEL_FS, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])


def build_major_categories_pct(results: Path, figures: Path) -> None:
    """Chart A — three % panels: All Confirmed | Variant A | Variant B.

    Variant A = naive non-phys task share (eco only, no AI signal).
    Variant B = pipeline pct restricted to non-physical tasks on both
    sides. All Confirmed is the anchor first panel — both split charts
    share this column so the eye can bridge between figures."""
    base = _major_base_data()
    save_csv(base, results / "major_categories.csv")

    categories_r = [_wrap_major_label(c) for c in reversed(base["category"].tolist())]
    pct_r   = list(reversed(base["pct_tasks_affected"].tolist()))
    pct_a_r = list(reversed(base["pct_a"].fillna(0.0).tolist()))
    pct_b_r = list(reversed(base["pct_b"].fillna(0.0).tolist()))
    n_cats = len(categories_r)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["", "", ""],
        horizontal_spacing=0.09,
        shared_yaxes=True,
    )

    inside_font  = dict(size=_MAJ_BARTEXT_FS, color="white",               family=FONT_FAMILY)
    outside_font = dict(size=_MAJ_BARTEXT_FS, color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    # Per-bar inside/outside decision: only put white text inside the
    # bar if the bar is comfortably wider than the text. Below the
    # threshold the data label sits outside in dark text so it reads
    # cleanly. Threshold of 30% of the [0, 100] x-range gives ~145 px
    # of bar room on a 487 px panel — enough for "XX.X%" at the 8 pt
    # floor with comfortable padding.
    PCT_INSIDE_THRESHOLD = 30.0
    pos_pct  = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_r]
    pos_a    = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_a_r]
    pos_b    = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_b_r]

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_r, orientation="h",
        marker=dict(color=METRIC_COLORS["tasks"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_r],
        textposition=pos_pct,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="All Confirmed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_a_r, orientation="h",
        marker=dict(color=VARIANT_A_COLOR, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_a_r],
        textposition=pos_a,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="Variant A: %{x:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_b_r, orientation="h",
        marker=dict(color=VARIANT_B_COLOR, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_b_r],
        textposition=pos_b,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="Variant B: %{x:.1f}%<extra></extra>",
    ), row=1, col=3)

    _style_major_split(
        fig,
        "AI Exposure by Major Occupational Category",
        n_cats=n_cats,
        panel_titles=set(),
    )
    # Percentage axes use a uniform [0, 100] range with ticks at
    # [0, 50, 100] across all three panels — keeps the reader's sense of
    # scale consistent panel-to-panel even when the longest bar in some
    # panels falls well short of 100%.
    PCT_RANGE = [0, 100]
    PCT_TICKS = [0, 50, 100]
    PCT_TICKTEXT = [f"{v}%" for v in PCT_TICKS]
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(text="Tasks Exposed", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=1,
    )
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(
            text="Hypothetical Exposure if<br>All Non-Phys Automatable",
            font=dict(size=_MAJ_LABEL_FS),
        ),
        row=1, col=2,
    )
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(text="Exposure of<br>Non-Phys Tasks", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=3,
    )
    fig.update_yaxes(
        title=dict(
            text="Major Occupational Category",
            font=dict(size=_MAJ_LABEL_FS),
            standoff=4,
        ),
        row=1, col=1,
    )

    save_figure(fig, results / "figures" / "major_categories_pct.png", scale=2)
    _copy_fig(results, figures, "major_categories_pct.png")
    print("  -> major_categories_pct.png")


def build_major_categories_wkrs_wages(results: Path, figures: Path) -> None:
    """Chart B — three All Confirmed panels: % Tasks | Workers | Wages.

    First panel repeats All Confirmed % Tasks Exposed as the anchor
    column so the reader can bridge ordering between Chart A and Chart B."""
    base = _major_base_data()

    categories_r = [_wrap_major_label(c) for c in reversed(base["category"].tolist())]
    pct_r     = list(reversed(base["pct_tasks_affected"].tolist()))
    workers_r = list(reversed(base["workers_affected"].tolist()))
    wages_r   = list(reversed(base["wages_affected"].tolist()))
    n_cats = len(categories_r)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["", "", ""],
        horizontal_spacing=0.09,
        shared_yaxes=True,
    )

    inside_font  = dict(size=_MAJ_BARTEXT_FS, color="white",               family=FONT_FAMILY)
    outside_font = dict(size=_MAJ_BARTEXT_FS, color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    # Per-bar inside/outside decision: bars wide enough get the white
    # in-bar label; the rest fall outside in dark text. % chart uses a
    # fixed 0-100 range; workers/wages use a fraction of the panel's max
    # so the threshold scales with the data.
    workers_max_v = float(base["workers_affected"].max()) if not base.empty else 0.0
    wages_max_v   = float(base["wages_affected"].max())   if not base.empty else 0.0
    PCT_INSIDE_THRESHOLD = 30.0
    WKR_INSIDE_THRESHOLD = 0.30 * workers_max_v
    WAG_INSIDE_THRESHOLD = 0.30 * wages_max_v
    pos_pct  = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_r]
    pos_wkrs = ["inside" if v >= WKR_INSIDE_THRESHOLD else "outside" for v in workers_r]
    pos_wags = ["inside" if v >= WAG_INSIDE_THRESHOLD else "outside" for v in wages_r]

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_r, orientation="h",
        marker=dict(color=METRIC_COLORS["tasks"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_r],
        textposition=pos_pct,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="All Confirmed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=categories_r, x=workers_r, orientation="h",
        marker=dict(color=METRIC_COLORS["workers"], line=dict(width=0)),
        text=[fmt_workers(v) for v in workers_r],
        textposition=pos_wkrs,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=categories_r, x=wages_r, orientation="h",
        marker=dict(color=METRIC_COLORS["wages"], line=dict(width=0)),
        text=[fmt_wages(v) for v in wages_r],
        textposition=pos_wags,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
    ), row=1, col=3)

    _style_major_split(
        fig,
        "AI Exposure by Major Occupational Category",
        n_cats=n_cats,
        panel_titles=set(),
        bottom_margin=130,  # 1-line axis titles ("Workers Exposed" etc.)
    )
    # Per-panel tight axis ranges so the longest bar in each panel
    # reaches ~95% of panel width. % chart uses its own axis logic;
    # workers/wages share the same helper with their natural units; the
    # % axis uses the same uniform [0, 100] / [0, 50, 100] presentation
    # as the major-pct chart so the reader's scale anchor doesn't shift
    # between figures.
    r_wkr, t_wkr = _axis_max_and_ticks(float(base["workers_affected"].max()))
    r_wag, t_wag = _axis_max_and_ticks(float(base["wages_affected"].max()))
    fig.update_xaxes(
        range=[0, 100], tickvals=[0, 50, 100],
        ticktext=["0%", "50%", "100%"],
        title=dict(text="Tasks Exposed", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=1,
    )
    fig.update_xaxes(
        range=[0, r_wkr], tickvals=t_wkr,
        ticktext=[_strip_zero_decimal(fmt_workers(v)) for v in t_wkr],
        title=dict(text="Workers Exposed", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=2,
    )
    fig.update_xaxes(
        range=[0, r_wag], tickvals=t_wag,
        ticktext=[_strip_zero_decimal(fmt_wages(v)) for v in t_wag],
        title=dict(text="Wages Exposed", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=3,
    )
    fig.update_yaxes(
        title=dict(
            text="Major Occupational Category",
            font=dict(size=_MAJ_LABEL_FS),
            standoff=4,
        ),
        row=1, col=1,
    )

    save_figure(fig, results / "figures" / "major_categories_wkrs_wages.png", scale=2)
    _copy_fig(results, figures, "major_categories_wkrs_wages.png")
    print("  -> major_categories_wkrs_wages.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart: Job zone violin restricted to non-physical occupations
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# GWA split charts — mirrors of the major-cat split: pct (Tasks Exposed |
# Hypothetical Variant A | Variant B). The workers/wages version lives in
# appendix/run.py since the per-row footprint becomes very tall at 41 rows.
# ─────────────────────────────────────────────────────────────────────────

# Tighter row height than the major chart (110 px vs 130 px). 41 GWA rows
# at the major's 130 px/row would print past 18 inches; 110 keeps the
# print height under ~15" while still giving each 2-line label and bar
# enough vertical room (the 2-line label is ~80 px tall at 9 pt).
_GWA_ROW_HEIGHT = 38


def _gwa_base_data() -> pd.DataFrame:
    """Shared loader for the GWA split charts. Returns one row per GWA
    with pct_tasks_affected (All Confirmed), pct_a, pct_b,
    workers_affected, wages_affected — sorted by All Confirmed % tasks
    descending so both gwa_pct and gwa_wkrs_wages share the same
    y-ordering."""
    base = _get_wa_data(PRIMARY_DATASET, "gwa")
    variant_a = compute_variant_a_gwa()
    variant_b_df = _get_wa_data_with_phys(PRIMARY_DATASET, "gwa", physical_mode="exclude")
    assert not base.empty,         "all_confirmed GWA data is empty"
    assert not variant_a.empty,    "variant_a GWA data is empty"
    assert not variant_b_df.empty, "variant_b GWA data is empty"

    base = base.sort_values("pct_tasks_affected", ascending=False).reset_index(drop=True)
    a_map = variant_a.set_index("category")["pct_tasks_affected"]
    b_map = variant_b_df.set_index("category")["pct_tasks_affected"]
    base["pct_a"] = base["category"].map(a_map)
    base["pct_b"] = base["category"].map(b_map)
    return base


def _style_gwa_split(
    fig: go.Figure,
    title: str,
    n_cats: int,
    panel_titles: set[str],
    bottom_margin: int = 180,
) -> None:
    """Layout shared by the GWA split charts. Same canvas/font apparatus
    as `_style_major_split` but with a smaller per-row height (41 rows
    vs 22 for majors). Per-row pitch tightened to 80 — with 8 pt y-tick
    font, two wrapped lines fit in ~58 px so 80 leaves a small breathing
    gap. Top margin cut to 110 (title is 11 pt). Default bottom_margin
    180 covers a 2-line axis title; callers with longer titles pass a
    larger value (height auto-adjusts)."""
    TOP_MARGIN = 110
    BOTTOM_DEFAULT = 180
    height = max(PAPER_H + 200, n_cats * _GWA_ROW_HEIGHT + TOP_MARGIN + BOTTOM_DEFAULT)
    # Add/subtract the bottom-margin delta from total height so the
    # plot area stays constant whether the caller asks for more (4-line
    # axis title) or less (1-line title) bottom space.
    height += bottom_margin - BOTTOM_DEFAULT
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=_MAJ_TITLE_FS, color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left",
            y=0.99, yanchor="top",
        ),
        font=dict(family=FONT_FAMILY, color=PAPER_PALETTE["text"]),
        plot_bgcolor=PAPER_PALETTE["surface"],
        paper_bgcolor=PAPER_PALETTE["surface"],
        width=_MAJ_CANVAS_W,
        height=height,
        margin=dict(l=110, r=110, t=TOP_MARGIN, b=bottom_margin),
        bargap=0.15,
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=_MAJ_TICK_FS, family=FONT_FAMILY),
        tickangle=0,
        ticklabelstandoff=8,
    )
    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_yaxes(
        tickfont=dict(size=_MAJ_YTICK_FS, family=FONT_FAMILY),
        automargin=True,
        ticklabelstandoff=2,
        ticks="",
        ticklen=0,
        row=1, col=1,
    )
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=_MAJ_PANEL_FS, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])


def build_gwa_pct(results: Path, figures: Path) -> None:
    """GWA Chart A — three % panels: All Confirmed | Hypothetical Variant A | Variant B.

    Same panel framing as `build_major_categories_pct`, applied to the
    41 O*NET Generalized Work Activities."""
    base = _gwa_base_data()
    save_csv(base, results / "gwa_exposure.csv")

    categories_r = [_wrap_gwa_label(c) for c in reversed(base["category"].tolist())]
    pct_r   = list(reversed(base["pct_tasks_affected"].tolist()))
    pct_a_r = list(reversed(base["pct_a"].fillna(0.0).tolist()))
    pct_b_r = list(reversed(base["pct_b"].fillna(0.0).tolist()))
    n_cats = len(categories_r)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["", "", ""],
        horizontal_spacing=0.09,
        shared_yaxes=True,
    )

    inside_font  = dict(size=_MAJ_BARTEXT_FS, color="white",               family=FONT_FAMILY)
    outside_font = dict(size=_MAJ_BARTEXT_FS, color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    # Higher threshold than the major chart (50% of max vs 30%) — the 41
    # compressed GWA rows make any near-bar text collisions more visible,
    # so we keep inside-white text for clearly-tall bars only and let
    # everything below the midpoint fall outside in dark text.
    PCT_INSIDE_THRESHOLD = 50.0
    pos_pct = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_r]
    pos_a   = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_a_r]
    pos_b   = ["inside" if v >= PCT_INSIDE_THRESHOLD else "outside" for v in pct_b_r]

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_r, orientation="h",
        marker=dict(color=METRIC_COLORS["tasks"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_r],
        textposition=pos_pct,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="All Confirmed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_a_r, orientation="h",
        marker=dict(color=VARIANT_A_COLOR, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_a_r],
        textposition=pos_a,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="Variant A: %{x:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=categories_r, x=pct_b_r, orientation="h",
        marker=dict(color=VARIANT_B_COLOR, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in pct_b_r],
        textposition=pos_b,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        textangle=0,
        showlegend=False, cliponaxis=False, constraintext="none",
        hovertemplate="Variant B: %{x:.1f}%<extra></extra>",
    ), row=1, col=3)

    _style_gwa_split(
        fig,
        "AI Exposure by General Work Activity",
        n_cats=n_cats,
        panel_titles=set(),
        bottom_margin=260,  # middle-panel axis title runs to 4 lines
    )
    # Uniform [0, 100] range with ticks [0, 50, 100] across all three
    # percentage panels — matches the major-cat chart so the scale
    # anchor reads consistently figure-to-figure.
    PCT_RANGE = [0, 100]
    PCT_TICKS = [0, 50, 100]
    PCT_TICKTEXT = [f"{v}%" for v in PCT_TICKS]
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(text="Tasks Exposed", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=1,
    )
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(
            # 4-line wrap (vs. major chart's 2-line) — at the GWA panel
            # width even a 3-line break is crowded against the neighbor
            # panels' tick labels; 4 lines keeps each line ~12 chars wide
            # so the title stays clearly within the middle panel.
            text=(
                "Hypothetical<br>"
                "Exposure if<br>"
                "All Non-Phys<br>"
                "Automatable"
            ),
            font=dict(size=_MAJ_LABEL_FS),
        ),
        row=1, col=2,
    )
    fig.update_xaxes(
        range=PCT_RANGE, tickvals=PCT_TICKS, ticktext=PCT_TICKTEXT,
        title=dict(text="Exposure of<br>Non-Phys Tasks", font=dict(size=_MAJ_LABEL_FS)),
        row=1, col=3,
    )
    fig.update_yaxes(
        title=dict(
            text="O*NET General Work Activity",
            font=dict(size=_MAJ_LABEL_FS),
            standoff=4,
        ),
        # Force every category label — plotly auto-thins categorical ticks
        # when the plot is short; tickmode="array" pins one label per bar.
        tickmode="array", tickvals=categories_r, ticktext=categories_r,
        row=1, col=1,
    )

    save_figure(fig, results / "figures" / "gwa_pct.png", scale=2)
    _copy_fig(results, figures, "gwa_pct.png")
    print("  -> gwa_pct.png")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)

    print("=" * 60)
    print("Part 2: Characterization — Where AI Exposure Falls")
    print("=" * 60)

    print("\n[1a/6] Major Occupational Categories — % Tasks Exposed (All Confirmed | Variant A | Variant B)")
    build_major_categories_pct(results, figures)

    print("\n[1b/5] Major Occupational Categories — Workers and Wages (All Confirmed)")
    build_major_categories_wkrs_wages(results, figures)

    print("\n[2/4] Job Zone Violin — Full Economy vs Non-Physical Only")
    build_job_zone_violin(results, figures)

    print("\n[3/4] Generalized Work Activities — % Tasks Exposed (All Confirmed | Hypothetical Variant A | Variant B)")
    build_gwa_pct(results, figures)

    print("\n[4/4] SKA Levels (with phys-mix coloring)")
    build_ska_levels(results, figures)

    # Clear stale figures from previous Part 2 layouts (box plot, phys/zone
    # combined charts, the old 6-panel major_categories.png, the
    # major_categories_trend chart which moved to the appendix, the
    # standalone non-phys job-zone chart that has been folded into the main
    # job_zone_violin three-panel layout, and the old 6-panel gwa_exposure
    # chart replaced by the split gwa_pct / gwa_wkrs_wages pair).
    for stale in ("phys_info_divide.png", "phys_zone_stacked.png",
                  "phys_zone_faceted.png", "major_categories.png",
                  "major_categories_trend.png",
                  "job_zone_violin_nonphys.png",
                  "gwa_exposure.png"):
        for d in (results / "figures", figures):
            p = d / stale
            if p.exists():
                p.unlink()

    print("\n" + "=" * 60)
    print("Part 2 complete — figures in results/figures/ and figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
