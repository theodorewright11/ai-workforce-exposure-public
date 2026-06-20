"""
Part 1 — Scale, Convergence, Growth

Three chart groups for the first section of the Results chapter:
1. Overview: Five-config aggregate economic footprint (grouped horizontal bars)
2. Convergence: Spearman rank correlation across four independent sources (2x2 heatmaps)
3. Temporal: Task penetration growth over time (line chart + delta tables)

Run from project root:
    venv/Scripts/python -m lib.builders.part1
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.config import (
    REFERENCE_DIR,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ANALYSIS_CONFIG_SERIES,
    ANALYSIS_DIR,
    ROOT,
    ensure_results_dir,
)
from lib.utils import FONT_FAMILY, save_figure, save_csv
from lib.paper_config import (
    PAPER_W, PAPER_H,
    TITLE_FS, SUBTITLE_FS, INSIDE_FS, OUTSIDE_FS, TICK_FS, LABEL_FS,
    LEGEND_FS, ANNOT_FS, HEATMAP_TEXT_FS, TABLE_HEADER_FS, TABLE_CELL_FS,
    METRIC_COLORS, METRIC_COLORS_LIGHT, HEATMAP_LOW, HEATMAP_HIGH,
    TREND_COLORS, PAPER_PALETTE,
    style_paper_figure, fmt_wages, fmt_workers, fmt_date,
    paper_fonts, paper_dataset_for,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ── Config display order ─────────────────────────────────────────────────
CONFIG_ORDER: list[str] = [
    "all_confirmed",
    "human_conversation",
    "agentic_confirmed",
    "agentic_ceiling",
    "all_ceiling",
]

# ── Correlation sources ──────────────────────────────────────────────────
CORR_SOURCES: dict[str, dict[str, str]] = {
    "claude":     {"dataset": "AEI Conv 2026-02-12",  "label": "Claude Browser"},
    "claude_api": {"dataset": "AEI API 2025 2026-02-12", "label": "Claude API"},
    "copilot":    {"dataset": "Microsoft",             "label": "Copilot"},
    "mcp":        {"dataset": "MCP Cumul. v4",         "label": "MCP"},
}
CORR_ORDER: list[str] = ["claude", "claude_api", "copilot", "mcp"]
CORR_LABELS: list[str] = [CORR_SOURCES[k]["label"] for k in CORR_ORDER]

# The main-paper source convergence chart also includes the All Confirmed
# aggregate as an extra row (appended last so it lands at the top of the
# y-axis and its lower-triangle internal block correlates against each of
# the four individual sources). The appendix full-matrix chart keeps
# CORR_ORDER untouched so all_confirmed appears only once via CONFIG_ORDER.
SOURCE_CHART_EXTRA_KEY: str = "all_confirmed"
SOURCE_CHART_EXTRA_LABEL: str = "All Confirmed"
SOURCE_CHART_EXTRA_DATASET: str = ANALYSIS_CONFIGS["all_confirmed"]

AGG_LEVELS: list[str] = ["major", "minor", "broad", "occupation"]
AGG_TITLES: dict[str, str] = {
    "major": "Major level",
    "minor": "Minor level",
    "broad": "Broad level",
    "occupation": "Occ level",
}
# Display word for the level in a chart title, and short slug for filenames.
LEVEL_TITLE_WORD: dict[str, str] = {
    "major": "Major", "minor": "Minor", "broad": "Broad",
    "occupation": "Occupation",
}
LEVEL_FILE_SHORT: dict[str, str] = {
    "major": "major", "minor": "minor", "broad": "broad",
    "occupation": "occ",
}

TREND_CONFIGS: list[str] = ["all_confirmed", "all_ceiling"]

# ── External benchmarks (for convergence_external chart) ─────────────────
# Four external occupation-level AI-exposure measures from prior academic
# work. The convergence_external chart correlates our four internal sources
# against each of these benchmarks at the same four SOC aggregation levels.
EXT_SOURCES: list[tuple[str, str]] = [
    ("gpt_beta",      "Eloundou GPT-4 β"),
    ("human_beta",    "Eloundou Human β"),
    ("aioe_mean",     "AIOE Overall"),
    ("aioe_rc",       "AIOE Reading Compr."),
    ("schaal_overall", "Schaal Overall"),
    ("schaal_da",     "Schaal DA"),
    ("schaal_ag",     "Schaal AG"),
    ("tomlinson_copilot", "Tomlinson (Copilot)"),
]

# Cells to gray out as contaminated by the Copilot task-filter pipeline
# (Eloundou labels were used to filter which Copilot tasks were included,
# so any correlation between a Copilot-containing measure and an Eloundou
# benchmark double-counts that signal). Keys are (row_label, col_label)
# pairs matching the labels rendered on each chart.
ELOUNDOU_LABELS: set[str] = {"Eloundou GPT-4 β", "Eloundou Human β"}
# Copilot and All Confirmed both inherit Microsoft's Eloundou-label task
# filter, so any correlation against an Eloundou benchmark double-counts that
# signal. Gray those cells out (transparency note in the chart).
CONTAMINATED_SOURCE_ROWS: set[str] = {"Copilot", "All Confirmed"}
CONTAMINATED_CONFIG_ROWS: set[str] = {
    "All Confirmed", "All Sources (Ceiling)", "Conversational Confirmed",
}

# Long category labels get wrapped onto two lines for axis tick display
# (tick text only — the underlying category keys stay unchanged so cell
# annotations and contamination checks continue to match against the
# original strings).
TICK_LABEL_WRAPS: dict[str, str] = {
    "Conversational Confirmed": "Conversational<br>Confirmed",
    "All Sources (Ceiling)":    "All Sources<br>(Ceiling)",
    "Agentic Confirmed":        "Agentic<br>Confirmed",
    "Agentic Ceiling":          "Agentic<br>Ceiling",
    "All Confirmed":            "All<br>Confirmed",
    "Claude Browser":           "Claude<br>Browser",
    "Claude API":               "Claude<br>API",
    "Eloundou GPT-4 β":         "Eloundou<br>GPT-4 β",
    "Eloundou Human β":         "Eloundou<br>Human β",
    "AIOE Overall":             "AIOE<br>Overall",
    "AIOE Reading Compr.":      "AIOE Reading<br>Compr.",
    "Schaal Overall":           "Schaal<br>Overall",
    "Schaal DA":                "Schaal<br>DA",
    "Schaal AG":                "Schaal<br>AG",
    "Tomlinson (Copilot)":      "Tomlinson<br>(Copilot)",
}


def _wrap_tick_labels(labels: list[str]) -> list[str]:
    return [TICK_LABEL_WRAPS.get(lbl, lbl) for lbl in labels]

GPTS_CSV = REFERENCE_DIR / "gpts_are_gpts_occ_data.csv"
AIOE_MATRIX_PATH = REFERENCE_DIR / "aioe_ability_matrix.csv"
ABILITIES_PATH = REFERENCE_DIR / "abilities_v30.1.csv"
SCHAAL_INDICES_CSV = REFERENCE_DIR / "Comparison of Indices.csv"
TOMLINSON_CSV = REFERENCE_DIR / "ai_applicability_scores.csv"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _get_national_totals() -> tuple[float, float]:
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    occ = eco.drop_duplicates(subset=["title_current"])
    total_emp = float(occ["emp_tot_nat_2025"].sum())
    total_wages = float((occ["emp_tot_nat_2025"] * occ["a_med_nat_2025"]).sum())
    return total_emp, total_wages


def _eco_task_comp_by_occ() -> pd.Series:
    """Per-occupation eco baseline task_comp sum (freq, no auto-aug, all phys).
    Used as the economy-wide denominator for the overview chart's ratio-of-
    totals % tasks number."""
    from backend.compute import load_eco_baseline
    eco = load_eco_baseline(method="freq", physical_mode="all", geo="nat")
    return eco.groupby("title_current")["task_comp"].sum()


def _get_eco_task_count() -> int:
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    return int(eco["task_normalized"].nunique())


def _run_config(dataset_name: str, agg_level: str = "occupation") -> pd.DataFrame:
    from backend.compute import get_group_data
    config = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": "all",
        "geo": "nat",
        "agg_level": agg_level,
        "sort_by": "% Tasks Affected",
        "top_n": 9999,
        "search_query": "",
        "context_size": 3,
    }
    data = get_group_data(config)
    assert data is not None, f"No data for {dataset_name}"
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    df = df.rename(columns={group_col: "category"})
    return df


def _count_tasks(dataset_name: str) -> int:
    from backend.config import DATASETS
    meta = DATASETS.get(dataset_name)
    if meta is None:
        return 0
    fpath = Path(meta["file"])
    if not fpath.exists():
        return 0
    df = pd.read_csv(fpath, usecols=["task_normalized"])
    return int(df["task_normalized"].nunique())


def _avg_auto_aug(dataset_name: str) -> float:
    """Average auto_aug_mean across unique tasks that have a value."""
    from backend.config import DATASETS
    meta = DATASETS.get(dataset_name)
    if meta is None:
        return 0.0
    fpath = Path(meta["file"])
    if not fpath.exists():
        return 0.0
    df = pd.read_csv(fpath, usecols=["task_normalized", "auto_aug_mean"])
    task_avg = df.groupby("task_normalized")["auto_aug_mean"].mean()
    return float(task_avg.mean())


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _stars(p: float) -> str:
    """Standard significance asterisks for two-tailed correlation p-values."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


SIG_NOTE: str = "All correlations significant at p < .001 (two-tailed Spearman)."


# ─────────────────────────────────────────────────────────────────────────
# Chart 1: Overview
# ─────────────────────────────────────────────────────────────────────────

def build_overview(results: Path, figures: Path) -> None:
    total_emp, total_wages = _get_national_totals()
    eco_tc_by_occ = _eco_task_comp_by_occ()
    eco_tc_total = float(eco_tc_by_occ.sum())

    rows: list[dict] = []
    for key in CONFIG_ORDER:
        # paper_dataset_for() applies paper-internal overrides — e.g.
        # agentic_confirmed → eco_2025-rebased file — without forcing
        # exploratory/claude_lab scripts that share ANALYSIS_CONFIGS
        # onto the same paper-specific dataset.
        ds = paper_dataset_for(key)
        label = ANALYSIS_CONFIG_LABELS[key]
        df = _run_config(ds, "occupation")

        workers = float(df["workers_affected"].sum())
        wages = float(df["wages_affected"].sum())
        # Ratio-of-totals across all (task, occ) pairs in the economy:
        # numerator is each occ's ai_task_comp (= pct[occ]/100 × eco_tc[occ]);
        # denominator is the economy-wide eco_tc sum.
        eco_tc_aligned = df["category"].map(eco_tc_by_occ).fillna(0.0)
        ai_tc_total = float(((df["pct_tasks_affected"] / 100.0) * eco_tc_aligned).sum())
        pct_tasks = (ai_tc_total / eco_tc_total * 100.0) if eco_tc_total > 0 else 0.0
        pct_workers = workers / total_emp * 100
        pct_wages = wages / total_wages * 100

        rows.append({
            "config": key, "label": label,
            "pct_tasks": round(pct_tasks, 1),
            "pct_workers": round(pct_workers, 1),
            "pct_wages": round(pct_wages, 1),
            "workers": workers, "wages": wages,
        })
        print(f"  {label}: {pct_tasks:.1f}% tasks, "
              f"{fmt_workers(workers)} ({pct_workers:.1f}%), "
              f"{fmt_wages(wages)} ({pct_wages:.1f}%)")

    save_csv(pd.DataFrame(rows), results / "overview_totals.csv")

    fig = go.Figure()
    plot_rows = list(reversed(rows))
    labels = [r["label"] for r in plot_rows]

    # Bar order within each config: tasks → workers → wages (top to bottom
    # within each grouped cluster). Plotly grouped bars stack first-trace
    # at the bottom of the cluster, so add them in reverse.
    metrics = [
        ("pct_tasks",   "Tasks Exposed",
         METRIC_COLORS["tasks"],
         lambda r: f"{r['pct_tasks']:.1f}% tasks"),
        ("pct_workers", "Workers Exposed",
         METRIC_COLORS["workers"],
         lambda r: f"{fmt_workers(r['workers'])} ({r['pct_workers']:.1f}%) workers"),
        ("pct_wages",   "Wages Exposed",
         METRIC_COLORS["wages"],
         lambda r: f"{fmt_wages(r['wages'])} ({r['pct_wages']:.1f}%) wages"),
    ]

    # All font sizes resolved from the standardized pt ladder (see
    # ANALYSIS_CLAUDE.md → Paper Chart Formatting). Inside-bar text sits at
    # the tick size — above the 8 pt in-chart floor.
    px = paper_fonts(PAPER_W)

    for pct_key, name, color, fmt_fn in reversed(metrics):
        fig.add_trace(go.Bar(
            y=labels,
            x=[r[pct_key] for r in plot_rows],
            name=name,
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            text=[fmt_fn(r) for r in plot_rows],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=px["tick"], color="white", family=FONT_FAMILY),
            showlegend=False,
        ))

    # Bar traces' default legend swatches are tiny and not sizable; emit
    # dummy scatter markers for the legend instead so we can scale them up.
    for pct_key, name, color, _ in metrics:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=22, color=color),
            name=name,
            showlegend=True,
            hoverinfo="skip",
        ))

    fig.update_layout(
        barmode="group",
        bargap=0.18,
        bargroupgap=0.04,
        legend=dict(traceorder="normal"),
        xaxis=dict(
            title=dict(text="% of National Total",
                       font=dict(size=px["axis_title"], family=FONT_FAMILY)),
            range=[0, 65],
            ticksuffix="%",
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=dict(text="Data Configuration",
                       font=dict(size=px["axis_title"], family=FONT_FAMILY)),
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        ),
    )

    style_paper_figure(
        fig,
        "AI Economic Exposure Across Data Configurations",
        subtitle="",
        height=PAPER_H + 270,
        margin=dict(l=20, r=60, t=90, b=180),
    )

    # Legend in container coords (0-1 of full figure width/height) so the
    # asymmetric l/r margins don't shift it off the figure center.
    # itemsizing="trace" lets each dummy scatter's marker.size drive the
    # legend swatch size (the bar traces themselves are not in the legend).
    fig.update_layout(
        legend=dict(
            orientation="h",
            xref="container", yref="container",
            x=0.5, xanchor="center",
            y=0.02, yanchor="bottom",
            font=dict(size=px["legend"], family=FONT_FAMILY),
            itemsizing="trace",
        ),
    )
    # style_paper_figure resets axis tick/title fonts — re-apply ours.
    fig.update_xaxes(
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title_font=dict(size=px["axis_title"], family=FONT_FAMILY),
    )
    fig.update_yaxes(
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title_font=dict(size=px["axis_title"], family=FONT_FAMILY),
    )

    save_figure(fig, results / "figures" / "overview.png")
    _copy_fig(results, figures, "overview.png")
    print("  -> overview.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 2: Convergence — internal sources + external benchmarks combined
# ─────────────────────────────────────────────────────────────────────────

def _load_eloundou_occ() -> pd.DataFrame:
    """Eloundou et al. (2023) per-occupation ratings, scaled ×100 to match
    our pct_tasks_affected units. Returns title_current, gpt_beta, human_beta."""
    df = pd.read_csv(GPTS_CSV)
    assert "Title" in df.columns, f"Title column missing in {GPTS_CSV}"
    for c in ("dv_rating_beta", "human_rating_beta"):
        assert c in df.columns, f"{c} column missing in {GPTS_CSV}"
    out = pd.DataFrame({
        "title_current": df["Title"].astype(str),
        "gpt_beta":      pd.to_numeric(df["dv_rating_beta"], errors="coerce") * 100.0,
        "human_beta":    pd.to_numeric(df["human_rating_beta"], errors="coerce") * 100.0,
    })
    assert out["gpt_beta"].notna().any(), "Eloundou gpt_beta is all NaN after load"
    return out


def _load_schaal_occ() -> pd.DataFrame:
    """Schaal 2025 occupation-level scores from `Comparison of Indices.csv`.
    Title joins exactly to title_current. Returns title_current,
    schaal_overall (auto_w), schaal_da (da_w), schaal_ag (ag_w)."""
    df = pd.read_csv(SCHAAL_INDICES_CSV)
    assert "title" in df.columns, f"title column missing in {SCHAAL_INDICES_CSV}"
    for c in ("auto_w", "da_w", "ag_w"):
        assert c in df.columns, f"{c} column missing in {SCHAAL_INDICES_CSV}"
    out = pd.DataFrame({
        "title_current":   df["title"].astype(str),
        "schaal_overall":  pd.to_numeric(df["auto_w"], errors="coerce"),
        "schaal_da":       pd.to_numeric(df["da_w"],   errors="coerce"),
        "schaal_ag":       pd.to_numeric(df["ag_w"],   errors="coerce"),
    })
    assert out["schaal_overall"].notna().any(), "Schaal auto_w is all NaN after load"
    assert out["schaal_da"].notna().any(),      "Schaal da_w is all NaN after load"
    assert out["schaal_ag"].notna().any(),      "Schaal ag_w is all NaN after load"
    return out


def _load_tomlinson_occ() -> pd.DataFrame:
    """Tomlinson, Jaffe, Wang, Counts & Suri (2025) AI applicability score per
    SOC, derived from ~100k Bing Copilot conversations × O*NET IWA weights
    × LLM completion + scope. Title joins exactly to title_current. Returns
    title_current, tomlinson_copilot."""
    df = pd.read_csv(TOMLINSON_CSV)
    assert "title" in df.columns, f"title column missing in {TOMLINSON_CSV}"
    assert "ai_applicability_score" in df.columns, \
        f"ai_applicability_score column missing in {TOMLINSON_CSV}"
    out = pd.DataFrame({
        "title_current":      df["title"].astype(str),
        "tomlinson_copilot":  pd.to_numeric(df["ai_applicability_score"], errors="coerce"),
    }).dropna(subset=["tomlinson_copilot"])
    assert not out.empty, "Tomlinson scores are all NaN after load"
    return out


def _compute_aioe_occ() -> pd.DataFrame:
    """Per-occupation AIOE scores computed as ratio-of-sums of imp×lv×ability_cap
    over imp≥3 ability rows (per Felten/Raj/Seamans framing). Two variants:
    mean of the 10 AI-application columns, and Reading Comprehension only.
    Values are ×100 to match pct_tasks_affected. Returns title_current,
    aioe_mean, aioe_rc."""
    matrix = pd.read_csv(AIOE_MATRIX_PATH, index_col=0)
    assert matrix.shape == (52, 10), f"AIOE matrix shape {matrix.shape} — expected (52, 10)"
    # AIOE labels this "Visual Color Determination"; O*NET v30.1 uses
    # "Visual Color Discrimination". Same element.
    matrix = matrix.rename(index={
        "Visual Color Determination": "Visual Color Discrimination",
    })
    per_ability = pd.DataFrame({
        "ability_name": matrix.index,
        "aioe_mean":    matrix.mean(axis=1).values,
        "aioe_rc":      matrix["Reading Comprehension"].values,
    })

    abilities = pd.read_csv(ABILITIES_PATH, dtype=str)
    abilities = abilities.rename(columns={
        "O*NET-SOC Code": "soc_code",
        "Title":          "title_current",
        "Element Name":   "ability_name",
        "Scale ID":       "scale_id",
        "Data Value":     "data_value",
    })
    abilities["data_value"] = pd.to_numeric(abilities["data_value"], errors="coerce")
    abilities = abilities[abilities["scale_id"].isin(["IM", "LV"])]
    pivoted = (
        abilities.pivot_table(
            index=["title_current", "ability_name"],
            columns="scale_id", values="data_value", aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"})
    pivoted = pivoted.dropna(subset=["importance", "level"])

    joined = pivoted.merge(per_ability, on="ability_name", how="inner")
    # imp ≥ 3 filter is applied per (occ, ability) row
    filt = joined[joined["importance"] >= 3].copy()
    assert not filt.empty, "AIOE: no rows after imp>=3 filter"
    filt["weight"] = filt["importance"] * filt["level"]

    grouped = filt.groupby("title_current")
    rows: list[dict] = []
    for title, g in grouped:
        w_sum = float(g["weight"].sum())
        if w_sum == 0:
            continue
        rows.append({
            "title_current": title,
            "aioe_mean": float((g["weight"] * g["aioe_mean"]).sum() / w_sum) * 100.0,
            "aioe_rc":   float((g["weight"] * g["aioe_rc"]).sum()   / w_sum) * 100.0,
        })
    out = pd.DataFrame(rows)
    assert not out.empty, "AIOE per-occ scores are empty"
    return out


def _ext_at_level(ext_df: pd.DataFrame, col: str, agg_level: str) -> pd.Series:
    """Roll an external benchmark from occupation level to SOC group level
    using an unweighted mean across matched occupations (each occupation
    contributes equally to its group). Matches the rollup method used in
    the exploratory gpts_are_gpts and aioe_comparison charts 14/18."""
    work = ext_df[["title_current", col]].dropna().copy()
    if agg_level == "occupation":
        return work.set_index("title_current")[col]

    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    level_col = {
        "major": "major_occ_category",
        "minor": "minor_occ_category",
        "broad": "broad_occ",
    }[agg_level]
    occ_to_group = (
        eco[["title_current", level_col]].drop_duplicates()
           .set_index("title_current")[level_col]
    )
    work["group"] = work["title_current"].map(occ_to_group)
    work = work.dropna(subset=["group"])
    return work.groupby("group")[col].mean()


def _build_convergence_chart(
    rows_keys: list[str],
    rows_labels: list[str],
    rows_data: dict[str, dict[str, pd.Series]],
    title: str,
    subtitle: str,
    out_name: str,
    csv_name: str,
    results: Path,
    figures: Path,
    y_axis_title: str,
    levels: list[str] | None = None,
    contaminated_rows: set[str] | None = None,
) -> None:
    """Build one combined heatmap (lower-tri internal + external block).

    `rows_keys` and `rows_labels` define the y-axis. `rows_data` is a
    nested dict {key → {level → pd.Series}} of pct_tasks_affected at
    each SOC level. `levels` selects which SOC levels become panels —
    panels are stacked vertically (one column) so each one renders at
    the full figure width, keeping fonts legible when the figure is
    scaled into the paper. `contaminated_rows` is the set of row labels
    whose correlations against ELOUNDOU_LABELS columns should be visually
    grayed out (the Eloundou-filter contamination on Copilot-containing
    measures).
    """
    levels = levels or ["major", "occupation"]
    n_levels = len(levels)
    contaminated_rows = contaminated_rows or set()
    eloundou = _load_eloundou_occ()
    aioe = _compute_aioe_occ()
    schaal = _load_schaal_occ()
    tomlinson = _load_tomlinson_occ()
    ext_df = (
        eloundou.merge(aioe,      on="title_current", how="outer")
                .merge(schaal,    on="title_current", how="outer")
                .merge(tomlinson, on="title_current", how="outer")
    )

    ext_keys = [k for k, _ in EXT_SOURCES]
    ext_labels = [lbl for _, lbl in EXT_SOURCES]
    n = len(rows_keys)
    n_ext = len(EXT_SOURCES)

    # Insert one blank column between the internal block and the external
    # block to visually separate the two groups. The gap column sits at
    # position `n` (index n in the matrix, label "" so no x-tick renders).
    GAP_LABEL = " "
    x_labels = list(rows_labels) + [GAP_LABEL] + list(ext_labels)
    n_cols = len(x_labels)
    EXT_OFFSET = n + 1   # column index where external block starts

    # Single-line tick labels on both axes (paper rule: x-tick labels in
    # the main-body charts never wrap to two lines). Rotation is steep
    # enough below that long single-line names still fit their column
    # slot. Cell annotations and contamination checks key off the
    # original strings; plotly uses these display labels for ticks.
    x_labels_disp = list(x_labels)
    rows_labels_disp = list(rows_labels)

    corr_records: list[dict] = []
    matrices: dict[str, np.ndarray] = {}
    pmatrices: dict[str, np.ndarray] = {}

    for level in levels:
        mat = np.full((n, n_cols), np.nan)
        pmat = np.full((n, n_cols), np.nan)

        # Internal block (lower triangle)
        for i in range(n):
            for j in range(i):
                si = rows_data[rows_keys[i]][level]
                sj = rows_data[rows_keys[j]][level]
                merged = pd.concat([si, sj], axis=1, join="inner").dropna()
                if len(merged) < 3:
                    continue
                rho, pval = stats.spearmanr(merged.iloc[:, 0], merged.iloc[:, 1])
                mat[i, j] = rho
                pmat[i, j] = pval
                corr_records.append({
                    "level": level, "kind": "internal",
                    "source_a": rows_labels[i], "source_b": rows_labels[j],
                    "rho": round(float(rho), 3),
                    "p_value": round(float(pval), 6),
                    "n": len(merged), "stars": _stars(pval),
                })

        # External block (offset by 1 to skip the gap column)
        for i, skey in enumerate(rows_keys):
            ours = rows_data[skey][level]
            for k, ext_key in enumerate(ext_keys):
                theirs = _ext_at_level(ext_df, ext_key, level)
                merged = pd.concat(
                    [ours.rename("x"), theirs.rename("y")],
                    axis=1, join="inner",
                ).dropna()
                if len(merged) < 3:
                    continue
                rho, pval = stats.spearmanr(merged["x"], merged["y"])
                mat[i, EXT_OFFSET + k] = rho
                pmat[i, EXT_OFFSET + k] = pval
                corr_records.append({
                    "level": level, "kind": "external",
                    "source_a": rows_labels[i], "source_b": ext_labels[k],
                    "rho": round(float(rho), 3),
                    "p_value": round(float(pval), 6),
                    "n": len(merged), "stars": _stars(pval),
                })

        matrices[level] = mat
        pmatrices[level] = pmat

    save_csv(pd.DataFrame(corr_records), results / csv_name)

    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices.values()])
    z_min = float(np.floor(all_vals.min() * 20) / 20)
    z_max = 1.0

    # One panel per SOC level (stacked if more than one) so each panel
    # spans the full figure width — this keeps the cell numbers and tick
    # labels legible once the figure is scaled into the paper. For a
    # single-level chart the SOC level is already in the figure title, so
    # the per-panel subplot title is suppressed.
    fig = make_subplots(
        rows=n_levels, cols=1,
        subplot_titles=([AGG_TITLES[l] for l in levels]
                        if n_levels > 1 else [""]),
        vertical_spacing=0.13,
    )

    # All font sizes are resolved from the standardized pt ladder via
    # paper_fonts(fig_width) once fig_width is fixed below. We compute
    # fig_width first so chrome and in-chart text scale together. When
    # the contamination legend is present, we widen the canvas so the
    # two-line legend banner can span ~the full canvas at the legend pt
    # size without clipping on the right edge.
    base_w = PAPER_W + max(0, (n_cols - 8) * 100)
    fig_width = base_w + (300 if contaminated_rows else 0)
    fig_height = 740 + n_levels * 780
    px = paper_fonts(fig_width)
    # Cell text sits at the in-chart floor (8 pt @ 6.5"). The full ladder:
    # title 11 / panel 10 / axis 10 / tick 9 / legend 9 / floor 8.
    cell_fs = px["in_chart_floor"]

    contam_color = "rgba(200, 200, 200, 0.92)"
    contam_text  = "#777777"

    for idx, level in enumerate(levels):
        row_pos = idx + 1
        col_pos = 1
        mat = matrices[level]
        pmat = pmatrices[level]

        fig.add_trace(
            go.Heatmap(
                z=mat.tolist(),
                x=x_labels_disp,
                y=rows_labels_disp,
                colorscale=[[0, HEATMAP_LOW], [1, HEATMAP_HIGH]],
                zmin=z_min, zmax=z_max,
                showscale=(idx == 0),
                hoverinfo="z",
                colorbar=dict(
                    title=dict(text="Spearman ρ",
                               font=dict(size=px["axis_title"], family=FONT_FAMILY),
                               side="right"),
                    len=0.55, y=0.5,
                    tickfont=dict(size=px["tick"], family=FONT_FAMILY),
                    dtick=0.1,
                ),
            ),
            row=row_pos, col=col_pos,
        )

        x_axis = f"x{idx + 1}" if idx > 0 else "x"
        y_axis = f"y{idx + 1}" if idx > 0 else "y"

        for i in range(n):
            for j in range(n_cols):
                val = mat[i, j]
                if np.isnan(val):
                    continue

                # Contamination check: row label is in contaminated set AND
                # this column is one of the Eloundou columns. Apply gray
                # overlay first (so annotation sits on top), then render
                # the value in muted text.
                row_label = rows_labels[i]
                col_label = x_labels[j]
                is_contam = (row_label in contaminated_rows
                             and col_label in ELOUNDOU_LABELS)

                if is_contam:
                    fig.add_shape(
                        type="rect",
                        x0=j - 0.5, x1=j + 0.5,
                        y0=i - 0.5, y1=i + 0.5,
                        xref=x_axis, yref=y_axis,
                        fillcolor=contam_color,
                        line=dict(width=0),
                        layer="above",
                    )
                    txt_color = contam_text
                else:
                    norm = (val - z_min) / max(z_max - z_min, 1e-9)
                    txt_color = "white" if norm >= 0.55 else PAPER_PALETTE["text_dark"]

                fig.add_annotation(
                    x=x_labels_disp[j], y=rows_labels_disp[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(size=cell_fs, family=FONT_FAMILY, color=txt_color),
                    xref=x_axis, yref=y_axis,
                )

        # Group header annotations centered above each block. Positioned
        # in axis coordinates (x = block midpoint), with a pixel yshift
        # so they sit just above the top edge of the heatmap regardless
        # of zoom level.
        internal_mid = (n - 1) / 2.0
        external_mid = EXT_OFFSET + (n_ext - 1) / 2.0
        for header_text, header_x in [("Internal", internal_mid),
                                       ("External", external_mid)]:
            fig.add_annotation(
                x=header_x, y=n - 0.5,
                text=f"<b>{header_text}</b>",
                showarrow=False,
                xanchor="center", yanchor="bottom",
                yshift=24,
                font=dict(size=px["panel_title"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
                xref=x_axis, yref=y_axis,
            )

        # Vertical divider between internal and external blocks
        fig.add_shape(
            type="line",
            x0=n - 0.5 + 0.5, x1=n - 0.5 + 0.5,  # midpoint of the gap col
            y0=-0.5, y1=n - 0.5,
            xref=x_axis, yref=y_axis,
            line=dict(color=PAPER_PALETTE["text"], width=5),
        )

    # Each panel spans the full width; figure height scales with the
    # number of stacked panels. fig_width / fig_height were set above
    # so paper_fonts(fig_width) drives all chrome. Subtitle is dropped
    # entirely — its content moves to the figure caption in the paper.
    # Bottom margin is sized to fit single-line x-tick labels at -75°
    # plus the spread-out contamination legend underneath.
    style_paper_figure(
        fig,
        title,
        subtitle="",
        width=fig_width,
        height=fig_height,
        margin=dict(l=180, r=180, t=220, b=620),
    )

    # Bump subplot titles a bit so they sit clear of the "Internal" /
    # "External" group headers placed below them. Size = ladder
    # panel_title; only present when n_levels > 1 (single-level charts
    # suppress the subplot label).
    agg_title_set = set(AGG_TITLES.values())
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in agg_title_set:
            ann.font = dict(
                size=px["panel_title"], family=FONT_FAMILY,
                color=PAPER_PALETTE["text"],
            )
            ann.yshift = 56

    # Contamination legend — two lines spanning the chart width. Placed
    # in the bottom margin, below the angled x-tick labels. Single-line
    # would be ~190 chars wide and clip; two lines at the in-chart floor
    # (8 pt — legitimate per the paper ladder) read as one horizontal
    # banner that uses ~the full canvas. Swatch is a real paper-coordinate
    # rectangle (Plotly's PNG export ignores HTML background-color on
    # spans).
    if contaminated_rows:
        # Swatch is sized to match the legend font height so it visually
        # pairs with one line of legend text. Paper coords are fractions
        # of the plot domain, so we convert font px → paper coords via
        # plot_width / plot_height. The plot domain is fig_width minus
        # left+right margins; plot height is fig_height minus top+bottom.
        plot_w = fig_width - 180 - 180
        plot_h = fig_height - 220 - 620
        legend_fs = px["in_chart_floor"]
        swatch_paper_w = legend_fs / plot_w
        swatch_paper_h = legend_fs / plot_h
        sx0 = 0.005
        sx1 = sx0 + swatch_paper_w
        sy_center = -0.760
        sy0 = sy_center - swatch_paper_h / 2
        sy1 = sy_center + swatch_paper_h / 2
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=sx0, x1=sx1, y0=sy0, y1=sy1,
            fillcolor=contam_color,
            line=dict(color=contam_text, width=1),
            layer="above",
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=sx1 + 0.008, y=sy_center,
            xanchor="left", yanchor="middle",
            text=("<b>Eloundou-contaminated cell</b> — Eloundou's task labels were used to filter Copilot tasks,<br>"
                  "so any correlation against a Copilot-containing measure double-counts that signal."),
            showarrow=False,
            align="left",
            font=dict(size=legend_fs, family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )

    # Tick fonts pulled from the ladder; tick angle -75° lets single-line
    # x-axis labels fit inside their column slot without bleeding into
    # neighbors (only ~26% of the label's pixel width sits horizontally
    # at -75°).
    for i in range(1, n_levels + 1):
        xkey = f"xaxis{i}" if i > 1 else "xaxis"
        ykey = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xkey].tickfont = dict(size=px["tick"], family=FONT_FAMILY)
        fig.layout[ykey].tickfont = dict(size=px["tick"], family=FONT_FAMILY)
        fig.layout[xkey].tickangle = -75
        if y_axis_title:
            fig.layout[ykey].title = dict(
                text=y_axis_title,
                font=dict(size=px["axis_title"], family=FONT_FAMILY),
            )

    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


# SOC levels shown in the main-paper convergence charts. Minor and Broad
# move to the appendix (full-matrix style) so the main charts can stay at
# two stacked panels — wide enough to keep every font ≥ 8pt once scaled
# into the paper.
MAIN_CONVERGENCE_LEVELS: list[str] = ["major", "occupation"]


def build_convergence(results: Path, figures: Path) -> None:
    """Source-level external benchmark comparison: 4 individual AI sources
    plus the All Confirmed aggregate on the y-axis; same set on x (lower-
    triangle) followed by the external academic benchmarks. One single-
    panel chart per SOC level (Major, Occupation).
    """
    chart_keys = list(CORR_ORDER) + [SOURCE_CHART_EXTRA_KEY]
    chart_labels = list(CORR_LABELS) + [SOURCE_CHART_EXTRA_LABEL]

    source_data: dict[str, dict[str, pd.Series]] = {}
    for skey in CORR_ORDER:
        ds = CORR_SOURCES[skey]["dataset"]
        source_data[skey] = {}
        for level in MAIN_CONVERGENCE_LEVELS:
            df = _run_config(ds, level)
            source_data[skey][level] = df.set_index("category")["pct_tasks_affected"]
        print(f"  {CORR_SOURCES[skey]['label']}: loaded {MAIN_CONVERGENCE_LEVELS}")

    # All Confirmed (aggregate config) — loaded the same way and added as
    # an extra row only on this main-paper chart.
    source_data[SOURCE_CHART_EXTRA_KEY] = {}
    for level in MAIN_CONVERGENCE_LEVELS:
        df = _run_config(SOURCE_CHART_EXTRA_DATASET, level)
        source_data[SOURCE_CHART_EXTRA_KEY][level] = (
            df.set_index("category")["pct_tasks_affected"]
        )
    print(f"  {SOURCE_CHART_EXTRA_LABEL}: loaded {MAIN_CONVERGENCE_LEVELS}")

    for level in MAIN_CONVERGENCE_LEVELS:
        short = LEVEL_FILE_SHORT[level]
        _build_convergence_chart(
            rows_keys=chart_keys,
            rows_labels=chart_labels,
            rows_data=source_data,
            title=("Benchmark Comparison by AI Source "
                   f"({LEVEL_TITLE_WORD[level]} Level)"),
            subtitle="Spearman ρ across our internal sources and academic benchmarks",
            out_name=f"convergence_{short}.png",
            csv_name=f"spearman_combined_{short}.csv",
            results=results, figures=figures,
            y_axis_title="",
            levels=[level],
            contaminated_rows=CONTAMINATED_SOURCE_ROWS,
        )


# ─────────────────────────────────────────────────────────────────────────
# Chart 3: Temporal
# ─────────────────────────────────────────────────────────────────────────

# Earlier dates added to the table (cream rows). AI Capability is barred
# because the all_confirmed / all_ceiling combined series doesn't have
# enough source coverage on these dates to compute a stable score, but the
# combined "tasks rated" count is still meaningful — we draw it from the
# date-matched all_confirmed / all_ceiling files (`AEI Both + Micro` /
# `All` respectively), which mirror the line-chart series.
HISTORICAL_DATES: list[str] = ["2024-09-30", "2024-12-23"]
# Per-config dataset names for the historical task counts.
HISTORICAL_DATASETS: dict[str, dict[str, str]] = {
    "all_confirmed": {
        "2024-09-30": "AEI Both + Micro 2024-09-30",
        "2024-12-23": "AEI Both + Micro 2024-12-23",
    },
    "all_ceiling": {
        "2024-09-30": "All 2024-09-30",
        "2024-12-23": "All 2024-12-23",
    },
}

# Per-snapshot "Source Release" labels — what was newly released on each
# snapshot date. Shown as a column in the temporal_tables figure to let
# readers see at a glance which source family's update is driving each
# row's change in tasks rated / auto-aug score.
SOURCE_RELEASE_LABELS: dict[str, str] = {
    "2024-09-30": "Microsoft",
    "2024-12-23": "AEI Browser v1",
    "2025-03-06": "AEI Browser v2",
    "2025-04-24": "MCP v1",
    "2025-05-24": "MCP v2",
    "2025-07-23": "MCP v3",
    "2025-08-11": "AEI Browser v3 + AEI API v3",
    "2025-11-13": "AEI Browser v4 + AEI API v4",
    "2026-02-12": "AEI Browser v5 + AEI API v5",
    "2026-02-18": "MCP v4",
}

# Light → deep green gradient for the two Δ columns. Lightest end is
# nearly white so small positive deltas stay legible; darkest end is a
# muted sage so the biggest additions visually pop without overwhelming.
DELTA_GRADIENT_LO: tuple[int, int, int] = (245, 250, 247)
DELTA_GRADIENT_HI: tuple[int, int, int] = (122, 178, 150)
DELTA_NEUTRAL: str = "#ffffff"


def _delta_gradient(values: list[float | None], historical_fill: str) -> list[str]:
    """Map a column of delta values to a light→deep green gradient by
    magnitude. ``None`` entries (e.g. "—" or historical placeholders) get
    the historical fill; non-positive entries stay white."""
    pos_vals = [v for v in values if v is not None and v > 0]
    if not pos_vals:
        return [historical_fill if v is None else DELTA_NEUTRAL for v in values]
    max_v = max(pos_vals)
    out: list[str] = []
    for v in values:
        if v is None:
            out.append(historical_fill)
            continue
        if v <= 0 or max_v == 0:
            out.append(DELTA_NEUTRAL)
            continue
        t = v / max_v
        r = int(DELTA_GRADIENT_LO[0] + (DELTA_GRADIENT_HI[0] - DELTA_GRADIENT_LO[0]) * t)
        g = int(DELTA_GRADIENT_LO[1] + (DELTA_GRADIENT_HI[1] - DELTA_GRADIENT_LO[1]) * t)
        b = int(DELTA_GRADIENT_LO[2] + (DELTA_GRADIENT_HI[2] - DELTA_GRADIENT_LO[2]) * t)
        out.append(f"rgb({r},{g},{b})")
    return out


def _build_trend_data() -> pd.DataFrame:
    total_emp, total_wages = _get_national_totals()
    eco_tasks = _get_eco_task_count()
    eco_tc_by_occ = _eco_task_comp_by_occ()
    eco_tc_total = float(eco_tc_by_occ.sum())

    trend_rows: list[dict] = []
    for config_key in TREND_CONFIGS:
        series = ANALYSIS_CONFIG_SERIES[config_key]
        label = ANALYSIS_CONFIG_LABELS[config_key]
        for ds_name in series:
            date_str = ds_name.rsplit(" ", 1)[-1]
            df = _run_config(ds_name, "occupation")

            workers = float(df["workers_affected"].sum())
            wages = float(df["wages_affected"].sum())
            pct_emp = workers / total_emp * 100
            # Ratio-of-totals across all (task, occ) pairs in the economy
            # (matches the overview chart's % tasks number).
            eco_tc_aligned = df["category"].map(eco_tc_by_occ).fillna(0.0)
            ai_tc_total = float(((df["pct_tasks_affected"] / 100.0) * eco_tc_aligned).sum())
            pct_tasks = (ai_tc_total / eco_tc_total * 100.0) if eco_tc_total > 0 else 0.0
            n_tasks = _count_tasks(ds_name)
            auto_aug = _avg_auto_aug(ds_name)

            trend_rows.append({
                "config": config_key,
                "label": label,
                "date": date_str,
                "dataset": ds_name,
                "pct_of_employment": round(pct_emp, 1),
                "pct_tasks_affected": round(pct_tasks, 1),
                "workers": workers,
                "wages": wages,
                "n_tasks": n_tasks,
                "avg_auto_aug": round(auto_aug, 2),
                "eco_tasks": eco_tasks,
                "total_emp": total_emp,
                "total_wages": total_wages,
            })
            print(f"  {label} {date_str}: {pct_tasks:.1f}% tasks aff, "
                  f"{fmt_workers(workers)}, {n_tasks} tasks, "
                  f"auto-aug {auto_aug:.2f}")

    return pd.DataFrame(trend_rows)


def _build_historical_rows(config_key: str) -> list[dict]:
    """Cream-row task counts for the dates that pre-date the line chart.

    The All Confirmed and All Ceiling tables each pull their historical
    rows from the same combined-source dataset family that the line chart
    uses (AEI Both + Micro for confirmed, All for ceiling). Sep 2024 only
    has Microsoft contributing, so the count there is Microsoft's rated
    set; Dec 2024 has Microsoft + AEI Conv v1, and the AEI Both + Micro /
    All files contain the union."""
    rows: list[dict] = []
    for date_str in HISTORICAL_DATES:
        ds_name = HISTORICAL_DATASETS[config_key][date_str]
        n_tasks = _count_tasks(ds_name)
        rows.append({"date": date_str, "n_tasks": n_tasks})
        print(f"  historical {date_str} ({config_key}, {ds_name}): {n_tasks} tasks rated")
    return rows


def _build_combined_table(trend_df: pd.DataFrame, results: Path, figures: Path) -> None:
    """One PNG per config — split into separate files so the paper can
    place each table independently without the wasted inter-table
    whitespace of a stacked subplot figure.

    Each table includes Sep 2024 and Dec 2024 historical rows pulled from
    that config's own dataset family (AEI Both + Micro for confirmed, All
    for ceiling). AI Capability cell is barred for those rows because the
    confirmed/ceiling AI-capability metric isn't well-defined that early
    in the series (only one or two sources contributing)."""
    highlight = PAPER_PALETTE["row_highlight"]
    white = PAPER_PALETTE["surface"]
    historical_fill = "#f5f0e8"  # subtle cream to mark historical rows
    total_eco_tasks = _get_eco_task_count()

    def _fmt_short(iso: str) -> str:
        """Three-letter month variant of fmt_date (e.g. 'Sep 30, 2024').
        Used here so the date column fits each plain date on one line."""
        from datetime import datetime
        dt = datetime.strptime(iso, "%Y-%m-%d")
        return dt.strftime("%b %d, %Y").replace(" 0", " ")

    # Canvas width drives the font ladder via paper_fonts(). We apply a
    # local "one-step-smaller" ladder here so the inlined-parenthetical
    # headers ("Tasks Rated (of 17,507)" / "Auto-Aug Score (0–5)") fit
    # on one line — header drops 9 → 8 pt (the floor, still
    # spec-compliant) which gives ~10% horizontal slack. Cells stay at
    # the 8 pt floor; chrome (title) drops one step too for a coherent
    # ladder.
    TABLE_W = PAPER_W + 500
    def _local_table_px(pt: float) -> int:
        return max(1, round(pt * TABLE_W / (6.5 * 72)))
    px = {
        "title":          _local_table_px(10),
        "panel_title":    _local_table_px(9),
        "axis_title":     _local_table_px(9),
        "tick":           _local_table_px(8),
        "legend":         _local_table_px(8),
        "in_chart_floor": _local_table_px(8),
    }
    # Heights generous enough to leave whitespace around the text but
    # not so much that the row feels half-empty. Combined with a
    # leading "<br>" on every header and cell value (see below), the
    # extra space lands above the text instead of below — plotly Tables
    # top-align, so without the leading break the slack would end up at
    # the bottom. (line-height ≈ 1.4 × font_px.)
    cell_row_h = round(px["in_chart_floor"] * 2.6)
    header_h   = round(px["tick"] * 2.6)

    # Map config key → filename suffix.
    config_suffix = {"all_confirmed": "confirmed", "all_ceiling": "ceiling"}

    for config_key in TREND_CONFIGS:
        sub = trend_df[trend_df["config"] == config_key].sort_values("date").reset_index(drop=True)
        if sub.empty:
            continue

        historical_rows = _build_historical_rows(config_key)

        col_date: list[str] = []
        col_source: list[str] = []
        col_tasks: list[str] = []
        col_dtasks: list[str] = []
        col_autoaug: list[str] = []
        col_dautoaug: list[str] = []
        date_fills: list[str] = []
        # Numeric copies of the delta columns drive the gradient coloring.
        dtasks_vals: list[float | None] = []
        dautoaug_vals: list[float | None] = []

        # Historical (cream) rows first — combined-dataset task counts.
        prev_n_tasks: int | None = None
        for hr in historical_rows:
            col_date.append(_fmt_short(hr["date"]))
            col_source.append(SOURCE_RELEASE_LABELS.get(hr["date"], "—"))
            col_tasks.append(f"{int(hr['n_tasks']):,}")
            if prev_n_tasks is None:
                col_dtasks.append("—")
                dtasks_vals.append(None)
            else:
                dt = int(hr["n_tasks"]) - prev_n_tasks
                col_dtasks.append(f"{'+' if dt >= 0 else ''}{dt:,}")
                dtasks_vals.append(float(dt))
            col_autoaug.append("—")
            col_dautoaug.append("—")
            dautoaug_vals.append(None)
            date_fills.append(historical_fill)
            prev_n_tasks = int(hr["n_tasks"])

        # Series rows (the line-chart range).
        for i, (_, r) in enumerate(sub.iterrows()):
            is_start_combined = (i == 0)
            is_end = (i == len(sub) - 1)

            # No "Series start:" / "End:" prefix — the light-blue
            # row-highlight fill already marks those rows, and dropping
            # the prefix keeps every Date cell on a single line.
            col_date.append(_fmt_short(r["date"]))

            col_source.append(SOURCE_RELEASE_LABELS.get(r["date"], "—"))
            col_tasks.append(f"{int(r['n_tasks']):,}")
            col_autoaug.append(f"{r['avg_auto_aug']:.2f}")

            curr_n_tasks = int(r["n_tasks"])
            if prev_n_tasks is None:
                col_dtasks.append("—")
                dtasks_vals.append(None)
            else:
                dt = curr_n_tasks - prev_n_tasks
                col_dtasks.append(f"{'+' if dt >= 0 else ''}{dt:,}")
                dtasks_vals.append(float(dt))
            prev_n_tasks = curr_n_tasks

            if is_start_combined:
                col_dautoaug.append("—")
                dautoaug_vals.append(None)
            else:
                prev = sub.iloc[i - 1]
                da = float(r["avg_auto_aug"] - prev["avg_auto_aug"])
                col_dautoaug.append(f"{'+' if da >= 0 else ''}{da:.2f}")
                dautoaug_vals.append(da)

            date_fills.append(highlight if (is_start_combined or is_end) else white)

        n_rows = len(col_date)
        n_hist = len(HISTORICAL_DATES)
        cell_fills = [historical_fill] * n_hist + [white] * (n_rows - n_hist)

        # Gradient fills for the two Δ columns (positive only; historical
        # rows keep the cream fill via the helper's None handling).
        dtasks_fills = _delta_gradient(dtasks_vals, historical_fill)
        dautoaug_fills = _delta_gradient(dautoaug_vals, historical_fill)

        header_color = (PAPER_PALETTE["all_confirmed"]
                        if "confirmed" in config_key
                        else PAPER_PALETTE["all_ceiling"])

        # Leading "<br>" on every header and cell value pushes the
        # text down to the second line, leaving a blank line of
        # whitespace above it. Plotly Tables top-align cells with no
        # valign flag — this is the only way to get the visual
        # whitespace on top rather than the bottom.
        pad = "<br>"
        header_values = [
            f"{pad}Date",
            f"{pad}Source Release",
            f"{pad}Tasks Rated (of {total_eco_tasks:,})",
            f"{pad}Δ Tasks",
            f"{pad}Auto-Aug Score (0–5)",
            f"{pad}Δ Auto-Aug",
        ]
        cell_columns = [col_date, col_source, col_tasks,
                        col_dtasks, col_autoaug, col_dautoaug]
        cell_values = [[f"{pad}{v}" for v in col] for col in cell_columns]

        # One figure per config — no subplot wrapper, no inter-table gap.
        fig = go.Figure()
        fig.add_trace(go.Table(
            columnwidth=[260, 460, 380, 160, 400, 220],
            header=dict(
                values=header_values,
                font=dict(size=px["tick"], family=FONT_FAMILY, color="white"),
                fill_color=header_color,
                align="center",
                height=header_h,
            ),
            cells=dict(
                values=cell_values,
                font=dict(size=px["in_chart_floor"], family=FONT_FAMILY),
                fill_color=[
                    date_fills, cell_fills, cell_fills,
                    dtasks_fills, cell_fills, dautoaug_fills,
                ],
                align="center",
                height=cell_row_h,
            ),
        ))

        # Tight height: title area + header + rows + bottom pad.
        # Plotly tables render at their natural pixel size (header_h +
        # n_rows × cell_row_h) and do not stretch to fill the
        # surrounding plot area — extra figure height shows as visible
        # whitespace below the last row. Snug margins (t=80, b=20)
        # eliminate that.
        TITLE_AREA = 80
        BOTTOM_PAD = 20
        height = TITLE_AREA + header_h + n_rows * cell_row_h + BOTTOM_PAD

        style_paper_figure(
            fig,
            ANALYSIS_CONFIG_LABELS[config_key],
            subtitle="",
            width=TABLE_W,
            height=height,
            margin=dict(l=10, r=10, t=TITLE_AREA, b=BOTTOM_PAD),
        )
        fig.update_layout(
            title=dict(font=dict(size=px["title"])),
        )

        fname = f"temporal_table_{config_suffix[config_key]}.png"
        save_figure(fig, results / "figures" / fname)
        _copy_fig(results, figures, fname)
        print(f"  -> {fname}")

    # Clean up the legacy combined PNG so it doesn't sit stale in the
    # committed figures dir.
    for legacy in (results / "figures" / "temporal_tables.png",
                   figures / "temporal_tables.png"):
        if legacy.exists():
            legacy.unlink()


def _build_three_panel_trend(trend_df: pd.DataFrame, results: Path, figures: Path) -> None:
    """Three side-by-side panels (Tasks / Workers / Wages), each plotting
    All Confirmed and All Sources (Ceiling) lines. Per-panel metric color:
    tasks=blue, workers=gold, wages=green. All Confirmed = solid line in
    primary color; All Sources (Ceiling) = dashed line in lighter shade.

    Per-point value labels are rendered as annotations with an explicit
    pixel `yshift` so they sit a fixed distance above/below the marker
    regardless of how the line curves — text mode + textposition only
    offsets ~6px which isn't enough to clear a curving line. Confirmed
    labels go below their line; ceiling labels go above theirs.

    The legend is rendered from two neutral-gray dummy traces (one solid,
    one dashed) so it conveys *line style* rather than implying any one
    panel's color is "the" color of confirmed vs. ceiling."""
    panels = [
        ("pct",     "Tasks Exposed", "Tasks Exposed",     "tasks",
         lambda v: f"{v:.1f}%",
         lambda subset: subset["pct_tasks_affected"]),
        ("workers", "Workers Exposed", "Workers Exposed",     "workers",
         lambda v: fmt_workers(v),
         lambda subset: subset["workers"]),
        ("wages",   "Wages Exposed",   "Wages Exposed (USD)", "wages",
         lambda v: fmt_wages(v),
         lambda subset: subset["wages"]),
    ]

    # Canvas width drives the font ladder via paper_fonts(). The +290
    # height bump leaves room below the per-panel "Snapshot Date" titles
    # for the manual side-by-side legend (placed in paper-y < 0 space).
    TREND_W = PAPER_W + 100
    TREND_H = PAPER_H + 290
    TREND_BOTTOM_MARGIN = 280
    px = paper_fonts(TREND_W)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.10,
    )

    # Legend is rendered manually below the subplots (shapes + annotations
    # in paper space) — plotly's auto-legend was wrapping the two entries
    # onto separate rows under orientation="h" regardless of entrywidth /
    # itemsizing settings. Doing it by hand guarantees side-by-side.
    legend_color = PAPER_PALETTE["text"]

    # Local "one-step-smaller" ladder for this chart only — the three
    # panels were visually cramped under the standard 11/10/10/9/9/8
    # paper ladder. Data labels were already pinned to the 8 pt floor,
    # so the chrome drops one step (title 10, panel 9, axis 9, tick 8,
    # legend 8) and the in-chart labels stay at 8. Hierarchy held:
    # title ≥ panel ≥ axis ≥ tick ≥ floor, legend == tick. The shared
    # paper FONT_PT_LADDER is unchanged.
    def _local_px(pt: float) -> int:
        return max(1, round(pt * TREND_W / (6.5 * 72)))

    TREND_TITLE_FS = _local_px(10)
    TREND_PANEL_FS = _local_px(9)
    TREND_AXIS_FS = _local_px(9)
    TREND_TICK_FS = _local_px(8)
    TREND_LEGEND_FS = _local_px(8)
    LABEL_FS_DATA = _local_px(8)
    LABEL_FS_HORIZON = _local_px(8)

    # Pixel offset for value labels above/below each marker. This is a
    # fixed pixel shift so labels stay clear of the line as it curves
    # between markers, regardless of zoom or aspect ratio.
    LABEL_YSHIFT_PX = 32

    # OLS extrapolation removed — chart now shows observed window only.
    EXTRAP_HORIZONS_DAYS: list[tuple[str, int]] = []

    def _linear_fit_project(dates: list[str], yvals: list[float],
                            horizon_days: list[int]) -> tuple[list[pd.Timestamp], list[float]]:
        """OLS y = a + b·t on observed (date, value) points; project values at each
        horizon past the final observed date. Returns (future_dates, future_values).

        Linear is the simplest defensible "if recent rate continues" frame for a
        2-year window. Pretends nothing past linear; saturation is out of scope."""
        if len(dates) < 2:
            return [], []
        ts = [pd.Timestamp(d) for d in dates]
        t0 = ts[0]
        x = np.array([(t - t0).days for t in ts], dtype=float)
        y = np.array(yvals, dtype=float)
        b, a = np.polyfit(x, y, deg=1)
        last_x = x[-1]
        future_xs = [last_x + h for h in horizon_days]
        future_ts = [t0 + pd.Timedelta(days=int(fx)) for fx in future_xs]
        future_ys = [float(a + b * fx) for fx in future_xs]
        return future_ts, future_ys

    def _spaced_label_indices(dates: list[str], min_days: int = 25) -> set[int]:
        """Pick which date indices get a value label drawn.

        Walks backwards from the last point and keeps a label only if it
        is at least `min_days` from the next kept label. This prevents
        labels at very close dates (e.g. Feb 12 / Feb 18 on the ceiling
        series) from stacking on top of each other horizontally.
        """
        if not dates:
            return set()
        parsed = [pd.Timestamp(d) for d in dates]
        keep = [len(dates) - 1]
        for i in range(len(dates) - 2, -1, -1):
            if (parsed[keep[-1]] - parsed[i]).days >= min_days:
                keep.append(i)
        return set(keep)

    for col_idx, (key, _panel_title, y_axis_title, metric_key, fmt_fn, getter) in enumerate(
        panels, start=1
    ):
        x_ref = "x" if col_idx == 1 else f"x{col_idx}"
        y_ref = "y" if col_idx == 1 else f"y{col_idx}"

        panel_vals: list[float] = []
        for config_key in TREND_CONFIGS:
            subset = trend_df[trend_df["config"] == config_key].sort_values("date").reset_index(drop=True)
            label = ANALYSIS_CONFIG_LABELS[config_key]
            if config_key == "all_confirmed":
                color = METRIC_COLORS[metric_key]
                dash = "solid"
                yshift = -LABEL_YSHIFT_PX  # below marker
            else:
                color = METRIC_COLORS_LIGHT[metric_key]
                dash = "dash"
                yshift = LABEL_YSHIFT_PX   # above marker

            xvals = list(subset["date"])
            yvals = list(getter(subset))
            panel_vals.extend(float(v) for v in yvals)

            # The line + markers (showlegend=False — legend uses the dummy
            # neutral traces, since each panel's color is different).
            fig.add_trace(go.Scatter(
                x=xvals, y=yvals,
                name=label,
                legendgroup=config_key,
                showlegend=False,
                mode="lines+markers",
                line=dict(color=color, width=3, dash=dash),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y}}<extra></extra>",
                cliponaxis=False,
            ), row=1, col=col_idx)

            # Linear "if-recent-rate-continued" projection extending past
            # the final observed point. Only the 2yr horizon gets a value
            # label — the 6mo / 1yr points carry the line shape but would
            # crowd the panel if labeled.
            horizon_days = [d for _, d in EXTRAP_HORIZONS_DAYS]
            twoyr_idx = next(
                (i for i, (lbl, _) in enumerate(EXTRAP_HORIZONS_DAYS) if lbl == "2yr"),
                len(EXTRAP_HORIZONS_DAYS) - 1,
            )
            future_ts, future_ys = _linear_fit_project(xvals, yvals, horizon_days)
            if future_ts:
                proj_x = [pd.Timestamp(xvals[-1])] + future_ts
                proj_y = [yvals[-1]] + future_ys
                fig.add_trace(go.Scatter(
                    x=proj_x, y=proj_y,
                    mode="lines+markers",
                    line=dict(color=color, width=2, dash="dot"),
                    marker=dict(size=7, color=color, symbol="x"),
                    showlegend=False,
                    hovertemplate=f"<b>{label} (linear proj.)</b><br>%{{x}}<br>%{{y}}<extra></extra>",
                    cliponaxis=False,
                    opacity=0.7,
                ), row=1, col=col_idx)
                panel_vals.extend(future_ys)
                hz_label, _ = EXTRAP_HORIZONS_DAYS[twoyr_idx]
                fig.add_annotation(
                    x=future_ts[twoyr_idx], y=future_ys[twoyr_idx],
                    xref=x_ref, yref=y_ref,
                    text=f"{hz_label}: {fmt_fn(future_ys[twoyr_idx])}",
                    showarrow=False,
                    yshift=yshift,
                    font=dict(size=LABEL_FS_HORIZON, color=color, family=FONT_FAMILY),
                )

            # Per-point value labels on the observed data line. Confirmed
            # (primary lens) gets every spaced point labeled; ceiling has
            # ~10 observations so we keep only the first and last to avoid
            # crowding the panel. For tightly clustered confirmed labels
            # (gap < CLOSE_GAP_DAYS), we stagger the y-position so the
            # earlier label sits one row further from the line — keeps both
            # readable without dropping either.
            CLOSE_GAP_DAYS = 120
            STAGGER_PX = 16
            if config_key == "all_confirmed":
                kept_set = _spaced_label_indices(xvals)
            elif len(xvals) >= 2:
                # Ceiling's first point coincides with confirmed's first point
                # on the same date — skip its label to avoid the doubled number.
                kept_set = {len(xvals) - 1}
            else:
                kept_set = set(range(len(xvals)))

            # Compute per-label yshift, applying stagger to a kept label
            # whose neighbour to the right is within CLOSE_GAP_DAYS.
            kept_sorted = sorted(kept_set)
            kept_ts = {i: pd.Timestamp(xvals[i]) for i in kept_sorted}
            per_label_yshift: dict[int, int] = {}
            for pos, i in enumerate(kept_sorted):
                shift = yshift
                if pos + 1 < len(kept_sorted):
                    nxt = kept_sorted[pos + 1]
                    if (kept_ts[nxt] - kept_ts[i]).days < CLOSE_GAP_DAYS:
                        # Push earlier label further from the line (same
                        # direction as yshift — away from the marker).
                        shift = yshift + (-STAGGER_PX if yshift < 0 else STAGGER_PX)
                per_label_yshift[i] = shift

            for i, (x_i, y_i) in enumerate(zip(xvals, yvals)):
                if i not in kept_set:
                    continue
                fig.add_annotation(
                    x=x_i, y=y_i,
                    xref=x_ref, yref=y_ref,
                    text=fmt_fn(y_i),
                    showarrow=False,
                    yshift=per_label_yshift[i],
                    font=dict(size=LABEL_FS_DATA, color=color, family=FONT_FAMILY),
                )

        # Tight y-range — leave enough room above and below the data band
        # for the pixel-shifted annotations to render without clipping.
        if panel_vals:
            v_lo, v_hi = min(panel_vals), max(panel_vals)
            spread = v_hi - v_lo
            pad_lo = spread * 0.22
            pad_hi = spread * 0.30  # extra room for the 2yr projection label
            y_min = max(0.0, v_lo - pad_lo)
            y_max = v_hi + pad_hi
        else:
            y_min, y_max = 0.0, 1.0

        if key == "pct":
            fig.update_yaxes(ticksuffix="%", range=[y_min, y_max], row=1, col=col_idx)
        elif key == "wages":
            fig.update_yaxes(tickprefix="$", range=[y_min, y_max], row=1, col=col_idx)
        else:
            fig.update_yaxes(range=[y_min, y_max], row=1, col=col_idx)

        fig.update_yaxes(
            title=dict(text=y_axis_title, font=dict(size=TREND_AXIS_FS)),
            tickfont=dict(size=TREND_TICK_FS, family=FONT_FAMILY),
            row=1, col=col_idx,
        )
        fig.update_xaxes(
            title=dict(text="Snapshot Date", font=dict(size=TREND_AXIS_FS)),
            tickangle=-30,
            tickfont=dict(size=TREND_TICK_FS, family=FONT_FAMILY),
            row=1, col=col_idx,
        )

    style_paper_figure(
        fig,
        "All Confirmed vs All Sources (Ceiling) Over Time",
        subtitle="",
        height=TREND_H,
        width=TREND_W,
        margin=dict(l=90, r=60, t=130, b=TREND_BOTTOM_MARGIN),
    )

    fig.update_layout(
        title=dict(font=dict(size=TREND_TITLE_FS)),
        showlegend=False,
    )

    # ── Manual legend (paper space) ─────────────────────────────────────
    # Items are laid out left-to-right with approximate text widths so
    # the whole legend block — not just the item-center symmetry — is
    # centered against the figure midpoint. Symmetric placement around
    # x=0.5 visibly skewed right because "All Sources (Ceiling)" is
    # wider than "All Confirmed".
    LEG_Y = -0.30                 # paper-y (negative = below plot area)
    LEG_LINE_LEN = 2 * 0.022      # length of each line indicator
    LEG_TEXT_GAP = 0.008          # gap between line end and text
    LEG_ITEM_SPACING = 0.05       # gap between end of item N's text and start of N+1's line
    legend_items = [
        ("All Confirmed",          "solid"),
        ("All Sources (Ceiling)",  "dash"),
    ]
    # Approximate per-character width (paper units) at the legend
    # font. Tuned empirically for Inter at TREND_LEGEND_FS px on a
    # TREND_W canvas — fine-grained enough to keep the block visibly
    # centered without a true text-metrics lookup.
    char_w = TREND_LEGEND_FS * 0.55 / TREND_W

    def _item_width(label: str) -> float:
        return LEG_LINE_LEN + LEG_TEXT_GAP + len(label) * char_w

    total_w = (
        sum(_item_width(lbl) for lbl, _ in legend_items)
        + LEG_ITEM_SPACING * (len(legend_items) - 1)
    )
    cursor_x = 0.5 - total_w / 2

    for label, dash_style in legend_items:
        line_start = cursor_x
        line_end = cursor_x + LEG_LINE_LEN
        text_x = line_end + LEG_TEXT_GAP
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=line_start, x1=line_end,
            y0=LEG_Y, y1=LEG_Y,
            line=dict(color=legend_color, width=3, dash=dash_style),
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=text_x, y=LEG_Y,
            text=label, showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=TREND_LEGEND_FS, family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )
        cursor_x += _item_width(label) + LEG_ITEM_SPACING
    # style_paper_figure resets axis tick/title fonts to TICK_FS / LABEL_FS;
    # re-apply ours so the axes don't render at 15 / 16 px.
    fig.update_xaxes(
        tickfont=dict(size=TREND_TICK_FS, family=FONT_FAMILY),
        title_font=dict(size=TREND_AXIS_FS, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        tickfont=dict(size=TREND_TICK_FS, family=FONT_FAMILY),
        title_font=dict(size=TREND_AXIS_FS, family=FONT_FAMILY),
    )

    panel_titles = {p[1] for p in panels}
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=TREND_PANEL_FS, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    save_figure(fig, results / "figures" / "temporal_trend.png")
    _copy_fig(results, figures, "temporal_trend.png")
    print("  -> temporal_trend.png")


def build_temporal(results: Path, figures: Path) -> None:
    trend_df = _build_trend_data()
    save_csv(trend_df, results / "trend_data.csv")
    _build_three_panel_trend(trend_df, results, figures)
    _build_combined_table(trend_df, results, figures)

    # Remove stale single-config tables and temporal_deltas if they exist
    for stale in (
        "temporal_table_all_confirmed.png",
        "temporal_table_all_ceiling.png",
        "temporal_deltas.png",
    ):
        for d in (results / "figures", figures):
            p = d / stale
            if p.exists():
                p.unlink()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)

    print("=" * 60)
    print("Part 1: Scale, Convergence, Growth")
    print("=" * 60)

    print("\n[1/3] External benchmark comparison: by AI Source")
    build_convergence(results, figures)

    print("\n[2/3] Overview: Six-config aggregate footprint")
    build_overview(results, figures)

    print("\n[3/3] Temporal: Growth trends + data tables")
    build_temporal(results, figures)

    print("\n" + "=" * 60)
    print("Part 1 complete — figures in results/figures/ and figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
