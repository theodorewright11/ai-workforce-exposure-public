"""External-benchmark figures — do the hours-weighted numbers still line
up with everybody else's?

Three figures:

  1-2. The full all-vs-all convergence matrix at major and at occupation
       level: nine internal measures against eight external academic
       indices, lower triangle only.
  3.   Eloundou divergence — where our reading and Eloundou's GPT-4 β
       disagree, by major occupational category.

All three share the correlation-matrix and divergence renderers defined at
the bottom of this module. Those renderers take optional injection hooks so
what goes IN can change without forking the layout:

  - Internal exposure is hours-weighted (`time_per_day`) and pinned to the
    2026-05-31 snapshots, via `figure_data.occ_exposure` / `figure_data.major_exposure`.
  - The major-level number is EMPLOYMENT-WEIGHTED. This is the reason these
    figures do not simply call the paper's `_run_config` with
    method="time_day": `time_per_day` sums to a constant 7.0 per occupation,
    so a plain ratio-of-totals at major level is arithmetically the
    unweighted mean of the member occupations' percentages, not a share of
    hours worked. `figure_data.major_exposure` puts employment back.
  - The external benchmarks are untouched. They are occupation-level
    constants with no weighting choice inside them, and they roll up to
    major by unweighted mean exactly as in the paper.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import figure_data
from lib.config import (
    REFERENCE_DIR,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ROOT,
)
from lib.utils import FONT_FAMILY, save_figure, save_csv
from lib.paper_config import (
    PAPER_W,
    PAPER_H,
    PAPER_PALETTE,
    style_paper_figure,
    paper_fonts,
    paper_dataset_for,
)

DATA_DIR = ROOT / "data"

# Internal measure key -> 05-31 dataset. The four raw sources first (the
# paper's CORR_ORDER), then the five configs (CONFIG_ORDER).
#
# Microsoft and MCP Cumul. v4 are single snapshots with no 05-31 twin, so
# they carry over unchanged — they are the same file the paper reads.
INTERNAL_DATASETS: dict[str, str] = {
    "claude": "AEI Conv eco2025 2026-05-31",
    "claude_api": "AEI API eco2025 2026-05-31",
    "copilot": "Microsoft",
    "mcp": figure_data.MCP_DATASET,
    **figure_data.CONFIG_DATASETS,
}

_LEVEL_TITLES: dict[str, str] = {
    "major": "SOC Major Group",
    "occupation": "Occupation",
}


def internal_series(key: str, level: str) -> Optional[pd.Series]:
    """Hours-weighted exposure for one internal measure at one SOC level,
    indexed the way the paper's matrix expects (category -> 0-100).

    Returns None for a key we have no pin for, which tells
    `build_convergence_full` to fall back to its own freq-weighted series —
    so an added measure degrades to the paper reading rather than vanishing.
    """
    dataset = INTERNAL_DATASETS.get(key)
    if dataset is None:
        return None
    if level == "occupation":
        occ = figure_data.occ_exposure(dataset)
        return occ.set_index("title_current")["p"] * 100.0
    if level == "major":
        return figure_data.major_exposure(dataset).set_index("category")["pct"]
    raise AssertionError(f"convergence supports major/occupation, got {level!r}")


def _build_convergence(results: Path, figures: Path, level: str) -> None:
    short = "occ" if level == "occupation" else level
    build_convergence_full(
        results, figures,
        levels=[(level, _LEVEL_TITLES[level])],
        out_name=f"convergence_full_{short}.png",
        csv_name=f"spearman_combined_full_{short}.csv",
        internal_series_fn=internal_series,
        title=(
            "Full-Matrix Benchmark Comparison — Work Time Exposed "
            f"({_LEVEL_TITLES[level]})"
        ),
    )


def build_convergence_major(results: Path, figures: Path) -> None:
    _build_convergence(results, figures, "major")


def build_convergence_occ(results: Path, figures: Path) -> None:
    _build_convergence(results, figures, "occupation")


def build_eloundou_divergence(results: Path, figures: Path) -> None:
    """Per-occupation z-scores of our work-time exposure and Eloundou's
    GPT-4 β, differenced, then averaged unweighted within each major.

    Only our side of the comparison changes. The rollup stays an unweighted
    mean of occupation-level differences — z-scores are already unitless and
    employment-weighting them would turn a statement about occupations into
    a statement about the biggest occupations in each major.
    """
    ours = (
        figure_data.occ_exposure(figure_data.PRIMARY_DATASET)
        .set_index("title_current")["p"]
        * 100.0
    )
    build_eloundou_divergence_major(
        results, figures,
        ours=ours,
        title="Where We and Eloundou Disagree by Major Occupational Category",
        # The axis title is centred on the plot area and this chart's left
        # margin is large, so a title longer than the paper's own runs off
        # the right edge. "z-score:" instead of "z-score difference:" buys
        # back exactly the room "Work Time Exposed" costs over "All
        # Confirmed".
        x_title="z-score: Work Time Exposed − Eloundou GPT-4 β",
        out_name="eloundou_divergence_major.png",
        csv_name="eloundou_divergence_major.csv",
    )


# ─────────────────────────────────────────────────────────────────────────
# Correlation sources, external academic indices, and the shared matrix /
# divergence renderers. Only these two figures use them.
# ─────────────────────────────────────────────────────────────────────────

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
    "claude":     {"dataset": "AEI Conv 2026-02-12",  "label": "Claude Browser"},  # eco_2015 AEI Conv family has no post-Feb-2026 file
    "claude_api": {"dataset": "AEI API 2025 2026-05-31", "label": "Claude API"},
    "copilot":    {"dataset": "Microsoft",             "label": "Copilot"},
    "mcp":        {"dataset": "MCP Cumul. v4",         "label": "MCP"},
}


CORR_ORDER: list[str] = ["claude", "claude_api", "copilot", "mcp"]


CORR_LABELS: list[str] = [CORR_SOURCES[k]["label"] for k in CORR_ORDER]


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


GPTS_CSV = REFERENCE_DIR / "gpts_are_gpts_occ_data.csv"


AIOE_MATRIX_PATH = REFERENCE_DIR / "aioe_ability_matrix.csv"


ABILITIES_PATH = REFERENCE_DIR / "abilities_v30.1.csv"


SCHAAL_INDICES_CSV = REFERENCE_DIR / "Comparison of Indices.csv"


TOMLINSON_CSV = REFERENCE_DIR / "ai_applicability_scores.csv"


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


# Star tiers, strongest first. Labels are HTML-escaped because they get
# rendered into Plotly annotations, where a bare "<" opens a tag and eats
# the rest of the string.
_SIG_TIERS: list[tuple[str, str]] = [("***", ".001"), ("**", ".01"), ("*", ".05")]


def _sig_note(p_values: list[float]) -> str:
    """One-line significance summary for a correlation matrix.

    Per-cell stars clutter a heatmap that already carries a number in
    every cell, so the convergence charts state the significance floor
    once underneath instead. The sentence is derived from the actual
    p-values rather than asserted: at Major level only 22 SOC categories
    back each ρ, and a handful of cells genuinely land at p < .01 / .05
    instead of p < .001. Returns "" when there's nothing to report.
    """
    vals = [float(p) for p in p_values if np.isfinite(p)]
    if not vals:
        return ""
    n = len(vals)
    counts = {stars: 0 for stars, _ in _SIG_TIERS}
    n_ns = 0
    for p in vals:
        s = _stars(p)
        if s:
            counts[s] += 1
        else:
            n_ns += 1

    suffix = "(two-tailed Spearman)"
    hit = [(stars, cut) for stars, cut in _SIG_TIERS if counts[stars]]

    if n_ns:
        if not hit:
            return f"None of the {n} correlations reach p &lt; .05 {suffix}."
        weakest = hit[-1][1]
        return (f"{n - n_ns} of {n} correlations significant at "
                f"p &lt; {weakest} {suffix}; {n_ns} not significant.")

    # Everything clears .05. One tier → state it flat; otherwise lead with
    # the floor that actually holds for every cell, then break it down.
    weakest = hit[-1][1]
    if len(hit) == 1:
        return f"All {n} correlations significant at p &lt; {weakest} {suffix}."
    breakdown = ", ".join(f"{counts[s]} at p &lt; {cut}" for s, cut in hit)
    return (f"All {n} correlations significant at p &lt; {weakest} "
            f"{suffix} — {breakdown}.")


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


# ── Renderers ───────────────────────────────────────────────────────────

PRIMARY_KEY = "all_confirmed"


PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def build_convergence_full(
    results: Path,
    figures: Path,
    levels: list[tuple[str, str]] | None = None,
    out_name: str = "convergence_full.png",
    csv_name: str = "spearman_combined_full.csv",
    internal_series_fn=None,
    title: str | None = None,
) -> None:
    """Full square correlation matrix: every internal measure (4 AI sources +
    5 ANALYSIS_CONFIGS) and every external benchmark (8 academic indices)
    on both x and y axes, lower-triangular cells only. Two panels stacked
    vertically — `levels` selects which SOC levels (default Major + Occ).

    A blank gap row + gap column separate the internal section from the
    external section on both axes. Cell rendering, gray-out, and the
    contamination legend follow the conventions of the main paper charts.

    `internal_series_fn(key, level) -> pd.Series | None` lets a caller
    supply its own exposure series for an internal measure, falling back
    to the paper's freq-weighted `_run_config` wherever it returns None.
    `internal_series` above uses it to swap in hours-weighted, 05-31-pinned
    series without forking 400 lines of heatmap layout. The external
    benchmarks are occupation-level constants and are never overridden.
    """
    from scipy import stats
    from lib.paper_config import (
        HEATMAP_TEXT_FS, HEATMAP_LOW, HEATMAP_HIGH,
    )

    from lib.config import ANALYSIS_CONFIG_LABELS

    LEVELS = levels or [("major", "Major level"), ("occupation", "Occ level")]

    # ── Internal measures ────────────────────────────────────────────
    internal_keys = list(CORR_ORDER) + list(CONFIG_ORDER)
    internal_labels = (list(CORR_LABELS)
                       + [ANALYSIS_CONFIG_LABELS[k] for k in CONFIG_ORDER])
    n_int = len(internal_keys)

    def _internal_at(key: str, dataset: str, level: str) -> pd.Series:
        if internal_series_fn is not None:
            override = internal_series_fn(key, level)
            if override is not None:
                assert not override.empty, f"empty override for {key}/{level}"
                return override
        return _run_config(dataset, level).set_index("category")["pct_tasks_affected"]

    internal_data: dict[str, dict[str, pd.Series]] = {}
    for skey in CORR_ORDER:
        ds = CORR_SOURCES[skey]["dataset"]
        internal_data[skey] = {
            lvl: _internal_at(skey, ds, lvl) for lvl, _ in LEVELS
        }
        print(f"  {CORR_SOURCES[skey]['label']}: loaded {[l for l, _ in LEVELS]}")
    for ckey in CONFIG_ORDER:
        ds = paper_dataset_for(ckey)
        internal_data[ckey] = {
            lvl: _internal_at(ckey, ds, lvl) for lvl, _ in LEVELS
        }
        print(f"  {ANALYSIS_CONFIG_LABELS[ckey]}: loaded {[l for l, _ in LEVELS]}")

    # ── External measures ────────────────────────────────────────────
    eloundou = _load_eloundou_occ()
    aioe = _compute_aioe_occ()
    schaal = _load_schaal_occ()
    tomlinson = _load_tomlinson_occ()
    ext_df = (eloundou.merge(aioe,      on="title_current", how="outer")
                       .merge(schaal,    on="title_current", how="outer")
                       .merge(tomlinson, on="title_current", how="outer"))

    ext_keys = [k for k, _ in EXT_SOURCES]
    ext_labels = [lbl for _, lbl in EXT_SOURCES]
    n_ext = len(ext_keys)

    external_data: dict[str, dict[str, pd.Series]] = {}
    for ekey in ext_keys:
        external_data[ekey] = {}
        for lvl, _ in LEVELS:
            external_data[ekey][lvl] = _ext_at_level(ext_df, ekey, lvl)
    print(f"  External benchmarks: loaded {n_ext} columns × {len(LEVELS)} levels")

    # ── Layout: gap inserted between internal and external on each axis
    all_keys = internal_keys + ext_keys
    all_labels = internal_labels + ext_labels
    all_data = {**internal_data, **external_data}
    n_meas = n_int + n_ext           # 17

    GAP_LABEL = " "
    layout_labels = list(internal_labels) + [GAP_LABEL] + list(ext_labels)
    # Both axes use single-line full-length labels. With 18 columns
    # squeezed into 6.5 inches, any wrap forces a "taller" bounding box
    # whose rotated extent overflows the column slot — so single-line
    # plus a near-vertical tick angle (-75°, set below) is the only
    # configuration that fits without label-to-label collision.
    n_layout = len(layout_labels)    # 18
    EXT_OFFSET = n_int + 1

    def m2l(m_idx: int) -> int:
        """measure index → layout index (skipping the gap row/col at n_int)"""
        return m_idx if m_idx < n_int else m_idx + 1

    contaminated_internals = CONTAMINATED_SOURCE_ROWS | CONTAMINATED_CONFIG_ROWS

    # ── Compute lower-tri correlations ───────────────────────────────
    matrices: dict[str, np.ndarray] = {}
    pmatrices: dict[str, np.ndarray] = {}
    records: list[dict] = []
    # Raw (unrounded) p-values feed the significance note — the CSV column
    # can't, since save_csv formats floats at %.2f. See the `%.3g` below.
    record_pvals: list[float] = []

    for level, _ in LEVELS:
        mat = np.full((n_layout, n_layout), np.nan)
        pmat = np.full((n_layout, n_layout), np.nan)
        for i in range(n_meas):
            for j in range(i):
                key_i, key_j = all_keys[i], all_keys[j]
                si = all_data[key_i][level]
                sj = all_data[key_j][level]
                merged = pd.concat([si, sj], axis=1, join="inner").dropna()
                if len(merged) < 3:
                    continue
                rho, pval = stats.spearmanr(merged.iloc[:, 0], merged.iloc[:, 1])
                li, lj = m2l(i), m2l(j)
                mat[li, lj] = rho
                pmat[li, lj] = pval
                record_pvals.append(float(pval))
                records.append({
                    "level": level,
                    "measure_a": all_labels[i],
                    "measure_b": all_labels[j],
                    "rho": round(float(rho), 3),
                    # String so save_csv's %.2f float_format leaves it alone;
                    # still parses back as a float via pd.read_csv.
                    "p_value": f"{float(pval):.3g}",
                    "n": len(merged),
                    "stars": _stars(pval),
                })
        matrices[level] = mat
        pmatrices[level] = pmat
        print(f"  {level}: {int(np.isfinite(mat).sum())} cells filled")

    save_csv(pd.DataFrame(records), results / csv_name)

    # ── Render: 2 panels stacked vertically ──────────────────────────
    all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices.values()])
    z_min = float(np.floor(all_vals.min() * 20) / 20)
    z_max = 1.0

    n_panels = len(LEVELS)
    # For a single-panel chart the SOC level already appears in the main
    # title — the per-panel subplot title would just repeat it and crowds
    # the title bar visually. Suppress it; only stack-panel charts (when
    # called with multiple levels) keep the subplot label.
    panel_subplot_titles = ([title for _, title in LEVELS]
                            if n_panels > 1 else [""])
    fig = make_subplots(
        rows=n_panels, cols=1,
        subplot_titles=panel_subplot_titles,
        vertical_spacing=0.10,
    )

    # Chrome (title / panel / axis / tick / legend) is resolved from the
    # standardized pt ladder via paper_fonts(fig_width). Cell text is a
    # documented exception to the 8 pt floor: with 18 columns sharing a
    # 6.5" print column, the per-cell slot is ~0.36 inches and 4-char
    # values ("0.95") at 8 pt would overflow into neighboring cells.
    # We size cell text dynamically to fit ~85% of the cell width — this
    # prints at ~7 pt, which is the same tradeoff the chart was using
    # pre-refactor (reference data labels, not primary chart text).
    import math as _math
    fig_width = PAPER_W + 1500          # 2900 px wide
    px = paper_fonts(fig_width)
    # All vertical positioning below the plot is derived from the actual
    # x-axis label extent. tick_fs and the longest layout label decide
    # how far the -75°-rotated labels reach below the plot; we then
    # anchor the legend at a fixed pixel gap below them and size
    # margin_b to contain both. This replaces the trial-and-error
    # sy_center constants that kept landing the legend either on top
    # of the labels or far past them.
    margin_l, margin_r, margin_t = 720, 180, 240
    plot_h_target = 1320                            # 18 rows × ~73 px
    cell_h_target = plot_h_target / len(layout_labels)
    tick_fs_est = min(px["tick"], int(cell_h_target / 1.6))
    longest_label_chars = max(len(s) for s in layout_labels)
    x_label_px = longest_label_chars * 0.55 * tick_fs_est
    x_label_vert_extent = x_label_px * _math.sin(_math.radians(75))
    legend_gap_px = 80                              # gap below x-labels
    legend_fs = px["in_chart_floor"]
    legend_line_h = legend_fs * 1.3
    # 2 contamination lines + 1 significance-note line, + buffer.
    legend_block_h = legend_line_h * 3 + 20
    legend_top_offset = x_label_vert_extent + legend_gap_px
    legend_bottom_offset = legend_top_offset + legend_block_h
    margin_b = int(legend_bottom_offset + 40)       # 40 px canvas buffer
    fig_height = plot_h_target + margin_t + margin_b
    plot_h = plot_h_target
    cell_w_px = (fig_width - margin_l - margin_r) / len(layout_labels)
    cell_fs = min(px["in_chart_floor"], int(cell_w_px / 3.0))
    contam_color = "rgba(200, 200, 200, 0.92)"
    contam_text  = "#777777"

    for idx, (level, _) in enumerate(LEVELS):
        row_pos = idx + 1
        mat = matrices[level]
        pmat = pmatrices[level]

        fig.add_trace(
            go.Heatmap(
                z=mat.tolist(),
                x=layout_labels,
                y=layout_labels,
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
            row=row_pos, col=1,
        )

        x_axis = f"x{idx + 1}" if idx > 0 else "x"
        y_axis = f"y{idx + 1}" if idx > 0 else "y"

        # Cell annotations + contamination overlays
        for li in range(n_layout):
            for lj in range(n_layout):
                val = mat[li, lj]
                if np.isnan(val):
                    continue
                row_label = layout_labels[li]
                col_label = layout_labels[lj]
                # Eloundou × Copilot-containing on either axis is contaminated
                contam_pair = (
                    (row_label in ELOUNDOU_LABELS and col_label in contaminated_internals)
                    or (col_label in ELOUNDOU_LABELS and row_label in contaminated_internals)
                )
                if contam_pair:
                    fig.add_shape(
                        type="rect",
                        x0=lj - 0.5, x1=lj + 0.5,
                        y0=li - 0.5, y1=li + 0.5,
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
                    x=layout_labels[lj], y=layout_labels[li],
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(size=cell_fs, family=FONT_FAMILY, color=txt_color),
                    xref=x_axis, yref=y_axis,
                )

        # X-axis group headers (above each column block)
        internal_x_mid = (n_int - 1) / 2.0
        external_x_mid = EXT_OFFSET + (n_ext - 1) / 2.0
        for header_text, header_x in [("Internal", internal_x_mid),
                                       ("External", external_x_mid)]:
            fig.add_annotation(
                x=header_x, y=n_layout - 0.5,
                text=f"<b>{header_text}</b>",
                showarrow=False,
                xanchor="center", yanchor="bottom",
                yshift=28,
                font=dict(size=px["panel_title"], family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
                xref=x_axis, yref=y_axis,
            )

        # Y-axis group headers are intentionally omitted. The matrix is
        # square, so the x-axis "Internal" / "External" headers plus the
        # horizontal divider line below row n_int already make the row
        # grouping unambiguous — and dropping them shaves enough left
        # margin to fit single-line y-axis tick labels without truncating
        # or abbreviating them.

        # Vertical + horizontal dividers between internal and external blocks
        fig.add_shape(
            type="line",
            x0=n_int, x1=n_int,
            y0=-0.5, y1=n_layout - 0.5,
            xref=x_axis, yref=y_axis,
            line=dict(color=PAPER_PALETTE["text"], width=5),
        )
        fig.add_shape(
            type="line",
            x0=-0.5, x1=n_layout - 0.5,
            y0=n_int, y1=n_int,
            xref=x_axis, yref=y_axis,
            line=dict(color=PAPER_PALETTE["text"], width=5),
        )

    # ── Figure-level styling ─────────────────────────────────────────
    # fig_width / fig_height set above so paper_fonts(fig_width) drives
    # all chrome. Y-axis labels are single-line so margin l is generous;
    # x-axis labels run diagonally at -75° (matching the main-body
    # convergence chart) so margin b accommodates their extent. Subtitle
    # is dropped — its content moves to the figure caption. Margins are
    # defined once above so cell_fs can reference plot_w in the same scope.
    level_names = " & ".join(t.replace(" level", "") for _, t in LEVELS)
    style_paper_figure(
        fig,
        title=title or f"Full-Matrix Benchmark Comparison ({level_names} Level)",
        subtitle="",
        width=fig_width,
        height=fig_height,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
    )

    # Bump subplot titles (only present when n_panels > 1).
    panel_title_set = {title for _, title in LEVELS}
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_title_set:
            ann.font = dict(size=px["panel_title"], family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])
            ann.yshift = 64

    # Tick labels for the dense 18×18 matrix run slightly below the 9 pt
    # ladder. At the ladder size, y-axis labels visually crowd against
    # neighboring rows. We scale them to ~1.6× the row height so adjacent
    # labels read as separate lines, capped at the ladder size (so this
    # never grows above the spec — only shrinks when matrices are dense).
    # Tick angle -75° matches the main-body convergence chart.
    cell_h_px = (fig_height - margin_t - margin_b) / len(layout_labels)
    tick_fs = min(px["tick"], int(cell_h_px / 1.6))
    for i in range(1, n_panels + 1):
        xkey = f"xaxis{i}" if i > 1 else "xaxis"
        ykey = f"yaxis{i}" if i > 1 else "yaxis"
        fig.layout[xkey].tickfont = dict(size=tick_fs, family=FONT_FAMILY)
        fig.layout[ykey].tickfont = dict(size=tick_fs, family=FONT_FAMILY)
        fig.layout[xkey].tickangle = -75
        # Force every row/column label to render. Without this, Plotly
        # auto-decimates whenever it thinks the labels are too dense
        # (silently dropping every other tick). With the figure height
        # above sized to actually fit them, this just prevents the
        # heuristic from kicking in conservatively.
        tickvals = list(range(len(layout_labels)))
        fig.layout[xkey].tickmode = "array"
        fig.layout[xkey].tickvals = tickvals
        fig.layout[xkey].ticktext = layout_labels
        fig.layout[ykey].tickmode = "array"
        fig.layout[ykey].tickvals = tickvals
        fig.layout[ykey].ticktext = layout_labels

    # Contamination legend — two lines, centered over the full canvas.
    # The main-body convergence chart uses xref="paper" with
    # near-symmetric margins, so its left-aligned legend lands roughly
    # at canvas center visually. This chart's margins are very
    # asymmetric (720 left for long row labels, 180 right), so anchoring
    # at the plot's left edge would put the legend hard left of canvas
    # center. We compute the canvas x where the swatch+text block needs
    # to start so the whole unit is centered on the canvas, then
    # convert that canvas x into paper coords (allowed to go negative —
    # paper coords aren't clipped, the legend just extends into the
    # generous left margin).
    plot_w = fig_width - margin_l - margin_r
    legend_text = (
        "<b>Eloundou-contaminated cell</b> — Eloundou's task labels were "
        "used to filter Copilot tasks,<br>so any correlation against a "
        "Copilot-containing measure double-counts that signal."
    )
    # Empirical char width for Inter at the legend pt: ~0.43 × font_px
    # (calibrated against the prior rendered appendix charts). Longest
    # rendered line after the <br> is the first line at ~92 chars.
    swatch_w_px = legend_fs
    gap_px = max(8, int(legend_fs * 0.3))
    longest_line_chars = 92
    char_w_ratio = 0.43
    text_w_px = longest_line_chars * char_w_ratio * legend_fs
    block_w_px = swatch_w_px + gap_px + text_w_px

    # Canvas x where the block starts so its center hits canvas center.
    block_start_canvas_px = (fig_width - block_w_px) / 2
    # Convert canvas px → paper coords (relative to plot domain).
    sx0 = (block_start_canvas_px - margin_l) / plot_w
    swatch_paper_w = swatch_w_px / plot_w
    swatch_paper_h = swatch_w_px / plot_h
    sx1 = sx0 + swatch_paper_w
    # Legend sits a fixed gap below the x-tick labels in paper coords,
    # centered on the first two of the block's three lines (the third is
    # the significance note added below).
    sy_center = -(legend_top_offset + legend_line_h) / plot_h
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
        x=sx1 + gap_px / plot_w, y=sy_center,
        xanchor="left", yanchor="middle",
        text=legend_text,
        showarrow=False,
        align="left",
        font=dict(size=legend_fs, family=FONT_FAMILY,
                  color=PAPER_PALETTE["text"]),
    )

    # Significance note — one line stating the floor for every cell in the
    # matrix, in place of per-cell stars (which would double the text in a
    # heatmap that already prints a number per cell). Centered on the
    # canvas like the contamination legend above it.
    sig_note = _sig_note(record_pvals)
    if sig_note:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=(fig_width / 2 - margin_l) / plot_w,
            y=-(legend_top_offset + legend_line_h * 2 + 8) / plot_h,
            xanchor="center", yanchor="top",
            text=sig_note,
            showarrow=False,
            align="center",
            font=dict(size=legend_fs, family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )

    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def build_eloundou_divergence_major(
    results: Path,
    figures: Path,
    ours: pd.Series | None = None,
    title: str | None = None,
    x_title: str | None = None,
    out_name: str = "eloundou_divergence_major.png",
    csv_name: str = "eloundou_divergence_major.csv",
) -> None:
    """Single-panel z-score divergence by Major Occupational Category.

    Per-occupation z-scores of our `all_confirmed` pct_tasks_affected and
    Eloundou et al. (2024) GPT-4 β are differenced, then averaged within
    each Major Occupational Category. Positive (blue) = we read more
    exposure than Eloundou; negative (orange) = Eloundou reads more.

    Mirrors the All Confirmed panel of `extcompare_eloundou_diff`'s
    `major_diverging_zscore` chart, formatted for the supplement.

    `ours` overrides our side of the comparison with a caller-supplied
    per-occupation exposure Series (`build_eloundou_divergence` passes the
    hours-weighted 05-31 one); the Eloundou side and the rollup are
    unchanged.
    """
    from scipy import stats as _stats  # noqa: F401  (kept for parity)
    from backend.compute import load_eco_raw
    from lib.utils import COLORS as _COLORS

    # ── 1. Load Eloundou GPT-4 β per occupation (x100 to match pct units)
    gpts_csv = REFERENCE_DIR / "gpts_are_gpts_occ_data.csv"
    elo_df = pd.read_csv(gpts_csv)
    assert "Title" in elo_df.columns
    assert "dv_rating_beta" in elo_df.columns
    elo = (
        pd.DataFrame({
            "title_current": elo_df["Title"].astype(str),
            "eloundou": pd.to_numeric(elo_df["dv_rating_beta"],
                                      errors="coerce") * 100.0,
        })
        .dropna(subset=["eloundou"])
        .groupby("title_current")["eloundou"]
        .mean()
    )

    # ── 2. all_confirmed pct per occupation
    pct = get_pct_tasks_affected(PRIMARY_DATASET) if ours is None else ours
    assert not pct.empty, "empty exposure series"

    # ── 3. Major occ category map
    eco = load_eco_raw()
    assert "major_occ_category" in eco.columns
    major = (
        eco[["title_current", "major_occ_category"]]
        .drop_duplicates()
        .set_index("title_current")["major_occ_category"]
    )

    df = pd.DataFrame({"eloundou": elo, "ours": pct}).dropna()
    df["major"] = df.index.map(major)
    df = df.dropna(subset=["major"])
    assert len(df) > 500, f"only {len(df)} occs matched"

    # ── 4. z-score each measure (population sd), then per-occ diff
    for col in ("eloundou", "ours"):
        sd = df[col].std(ddof=0)
        assert sd > 0
        df[f"{col}_z"] = (df[col] - df[col].mean()) / sd
    df["diff"] = df["ours_z"] - df["eloundou_z"]

    # ── 5. Roll up to Major Occupational Category by unweighted mean
    s = df.groupby("major")["diff"].mean().sort_values()
    save_csv(
        s.reset_index().rename(columns={"diff": "mean_z_diff"}),
        results / csv_name,
        float_format="%.3f",
    )
    print(f"  matched {len(df)} occs across {s.size} major categories")

    # ── 6. Bar labels — strip " Occupations" suffix; keep single-line so
    # the y-axis title sits flush against tick labels rather than between
    # wrapped rows.
    def _clean(label: str) -> str:
        return label.replace(" Occupations", "")

    labels = [_clean(m) for m in s.index]

    OURS_HIGHER = _COLORS["primary"]   # slate blue
    ELO_HIGHER  = _COLORS["accent"]    # orange
    bar_colors = [OURS_HIGHER if v >= 0 else ELO_HIGHER for v in s.values]

    px = paper_fonts(PAPER_W)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=s.values,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors),
        text=[f"{v:+.2f}" for v in s.values],
        textposition="outside",
        textfont=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["text"], family=FONT_FAMILY),
        showlegend=False,
        hovertemplate="%{y}<br>diff %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=PAPER_PALETTE["text"], width=1))

    fig_height = max(PAPER_H, len(labels) * 38 + 220)

    # Margins: left holds the rotated y-axis title (~30 px wide bbox)
    # PLUS a ~50 px breathing gap PLUS the longest tick label (~14 px/char
    # at the 9 pt tick font on a 1400 px canvas). The 12 px/char estimate
    # used previously undershot true caps-heavy label width and the tick
    # labels overlapped the rotated title.
    longest_label_chars = max(len(lab) for lab in labels)
    # 13 px/char is calibrated to clear the longest tick label off the
    # rotated y-axis title bbox without squeezing the plot too narrow.
    margin_l = max(450, int(longest_label_chars * 13 + 130))
    # Right margin trimmed from the original 180 px (visible excess
    # whitespace) but kept generous enough that "…Eloundou GPT-4 β"
    # and the "+0.74" data label both render fully.
    margin_r = 140

    # An outside bar label is drawn beyond the bar END, so the axis has to
    # reserve room for it on BOTH sides or the most-negative bar's label
    # runs left out of the plot and lands on top of its own tick label —
    # which is what "Computer and Mathematical-0.39" was. A flat 22%-of-max
    # pad cannot do this: the label's width is fixed in PIXELS while the pad
    # is in axis units, so whether it fits depends on the data range.
    #
    # Solve for the range instead. With a label of w pixels on each extreme,
    #   total_range = span + 2 x (w / plot_w) x total_range
    # so total_range = span / (1 - 2w/plot_w), and the pad is the difference.
    plot_w_px = float(PAPER_W - margin_l - margin_r)
    label_px = len("-0.00") * px["in_chart_floor"] * 0.58 + 12   # + gap
    label_frac = label_px / plot_w_px
    assert label_frac < 0.3, "labels too wide for the plot area"
    span = float(s.max() - s.min()) or 1.0
    pad = max(
        (span / (1.0 - 2.0 * label_frac) - span) / 2.0,
        float(s.abs().max() or 1) * 0.10,
    )

    style_paper_figure(
        fig,
        title=title or "Where We and Eloundou Disagree by Major Occupational Category",
        width=PAPER_W,
        height=fig_height,
        # Bottom margin holds x-tick row + axis title + ~50 px gap +
        # the one-line legend (visible breathing room from the axis title).
        margin=dict(l=margin_l, r=margin_r, t=90, b=210),
    )

    fig.update_xaxes(
        title=dict(
            text=x_title or "z-score difference: All Confirmed − Eloundou GPT-4 β",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
        ),
        range=[s.min() - pad, s.max() + pad],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        gridcolor=PAPER_PALETTE["grid"],
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(
            text="Major Occupational Category",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
            # standoff = distance (px) from axis line back to the rotated
            # title's center. Set to (margin_l - 50) so the title sits
            # ~50 px in from the canvas left edge — far enough that the
            # rotated title's bbox doesn't bump into the longest tick
            # label (which extends ~longest_label_chars × 14 px left
            # from the axis line).
            standoff=max(0, margin_l - 50),
        ),
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        automargin=False,
        showgrid=False,
    )
    # ── Manual one-row legend (two swatches + labels) ─────────────────
    # Plotly's auto-legend doesn't reliably render two items on a single
    # row in horizontal mode here — it stacks them, and `entrywidth`
    # truncates labels. Drawing the legend as (shape, annotation) pairs
    # in paper coords pins both entries on one line below the x-axis
    # title with a controlled gap.
    legend_items = [
        (OURS_HIGHER, "We read more exposure"),
        (ELO_HIGHER,  "Eloundou reads more exposure"),
    ]
    plot_w_px = PAPER_W - margin_l - margin_r
    plot_h_px = fig_height - 90 - 210      # mirrors margin_t / margin_b above
    swatch_px = px["legend"]               # square swatch matches legend font height
    item_gap_px = 28                       # gap between the two legend items
    swatch_text_gap_px = 8                 # gap between swatch and its label
    char_w_px = swatch_px * 0.55           # Inter at legend pt, rough avg width
    item_widths_px = [swatch_px + swatch_text_gap_px + len(text) * char_w_px
                      for _, text in legend_items]
    total_w_px = sum(item_widths_px) + item_gap_px * (len(legend_items) - 1)
    # Center horizontally across the FULL canvas (not the plot area —
    # margin_l is much larger than margin_r so plot-centered would sit
    # noticeably right of the PNG's center). Vertical placement: ~140 px
    # below plot bottom, which lands ~80 px below the x-axis title.
    legend_y_paper = -140 / plot_h_px
    canvas_center_px = PAPER_W / 2.0
    legend_start_canvas_px = canvas_center_px - total_w_px / 2.0
    # Convert that canvas-pixel start position into plot-area paper
    # coords (since the shapes / annotations use xref="paper").
    start_x_px_in_plot = legend_start_canvas_px - margin_l

    cursor_px = start_x_px_in_plot
    swatch_half = (swatch_px / 2) / plot_h_px
    for (color, text), item_w_px in zip(legend_items, item_widths_px):
        swatch_x0 = cursor_px / plot_w_px
        swatch_x1 = (cursor_px + swatch_px) / plot_w_px
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=swatch_x0, x1=swatch_x1,
            y0=legend_y_paper - swatch_half,
            y1=legend_y_paper + swatch_half,
            fillcolor=color, line=dict(width=0),
            layer="above",
        )
        text_x = (cursor_px + swatch_px + swatch_text_gap_px) / plot_w_px
        fig.add_annotation(
            xref="paper", yref="paper",
            x=text_x, y=legend_y_paper,
            xanchor="left", yanchor="middle",
            text=text, showarrow=False,
            font=dict(size=px["legend"], family=FONT_FAMILY,
                      color=PAPER_PALETTE["text"]),
        )
        cursor_px += item_w_px + item_gap_px

    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")
