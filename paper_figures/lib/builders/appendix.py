"""Appendix figures.

Charts produced fresh by this run.py (no copying from elsewhere), ordered
to match the paper's chart flow — correlation, aggregate economy, trend,
then SKA:

1. convergence_full — full square correlation matrix, one chart per SOC
   level (major / minor / broad / occ).
2. overview_no_autoaug — paper part_1 `overview` recomputed with
   `use_auto_aug=False`.
3. temporal_trend_nonphys — single-panel non-physical variant of Part 1's
   `temporal_trend`.
4. ska_full — three separate element-level SKA charts (one each for
   skills, knowledge, abilities). Mirrors the main-body Part 2 framing
   (bar = AI Top-10 % of workforce max, colored by phys-mix tier; dots
   = AI Max ◆ and Workforce Mean ●), expanded to the full element list
   with O*NET subcategory in parentheses on each K/A element label.
   Outputs: ska_skills_full.png, ska_knowledge_full.png, ska_abilities_full.png.
5. state_clusters_each_ranked — companion to Part 3 state_clusters_map.
   Two-panel ranked bars where each panel sorts the 51 states by its own
   metric (left by % workforce exposed, right by % in High AI Exp & <0
   Emp Proj occupations). Cluster colors carry across panels.
6. state_clusters_combined_ranked — companion to state_clusters_each_ranked.
   Sums each state's rank on the two panels and sorts ascending. Single
   bar per state colored by Ward cluster (matches the panel chart);
   end-of-bar label is the combined rank sum (bold) with the two
   component ranks in parentheses.
7. adoption_friction_scatter — Section 3 adoption-layer probe. Restrict
   to occupations that are mostly non-physical (Part 2's Non-Physical
   bucket: <33% physical tasks) and scatter each occ's mean rating of
   the friction property (r, df) against its % tasks exposed. Two
   panels, Spearman ρ + OLS fit inset per panel.
8. capability_vs_adoption_all_occs — companion 4-row chart across all
   923 occupations. Row 1 (one wide panel): pct_physical, the raw
   structural variable (ρ −0.78). Rows 2–3: capability properties
   (Schaal ag, Schaal da, our s, our d; ρ +0.55 to +0.68). Row 4:
   our adoption frictions (r, df; ρ −0.17 to −0.19). Shows why the
   friction signal needs the non-phys restriction — capability props
   ride the phys/non-phys split visible in row 1.

Run from project root:
    venv/Scripts/python -m lib.builders.appendix
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.config import (
    REFERENCE_DIR,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ANALYSIS_CONFIG_SERIES,
    ROOT,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.compute_ska import load_ska_data
from lib.utils import FONT_FAMILY, save_figure, save_csv
from lib.paper_config import (
    PAPER_W, PAPER_H,
    TITLE_FS, SUBTITLE_FS, LABEL_FS, TICK_FS, ANNOT_FS, LEGEND_FS,
    INSIDE_FS,
    METRIC_COLORS, METRIC_COLORS_LIGHT, PAPER_PALETTE,
    fmt_workers, fmt_wages,
    style_paper_figure, paper_fonts, paper_dataset_for,
)
# GWA wkrs/wages reuses the part_2 helpers (axis picker, wrap, layout
# style) so the appendix chart matches the main-text gwa_pct figure.
from lib.builders.part2 import (
    _gwa_base_data, _wrap_gwa_label, _style_gwa_split,
    _axis_max_and_ticks, _strip_zero_decimal,
    _MAJ_BARTEXT_FS, _MAJ_LABEL_FS,
)

# Match paper part_1 ordering (top → bottom in chart = first → last here)
OVERVIEW_CONFIG_ORDER: list[str] = [
    "all_confirmed",
    "human_conversation",
    "agentic_confirmed",
    "agentic_ceiling",
    "all_ceiling",
]

HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PRIMARY_KEY = "all_confirmed"
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]

# Intensity-figure dataset (used by build_intensity_drivers + build_underadoption_gap).
# AEI Conv + AEI API pooled onto eco_2025, no Microsoft. Equal 3-source debias
# (Claude/Copilot/ChatGPT GWA priors) still applies. Mirrors the constant in
# part_3/run.py — kept here to avoid a paper-internal import.
_INTENSITY_DATASET = "AEI Both 2025 2026-02-12"
_INTENSITY_V3_KEY = "aei_all_eco2025"

PHYS_LOWER = 33.0
PHYS_UPPER = 67.0

# Order panels Physical → Mixed → Non-physical
OCC_GROUPS = ["Physical", "Mixed", "Non-physical"]
GROUP_COLORS = {
    "Non-physical": METRIC_COLORS["tasks"],
    "Mixed":        METRIC_COLORS["workers"],
    "Physical":     METRIC_COLORS["wages"],
}

ZONE_LABELS = {
    1: "Zone 1 — Little/No Prep",
    2: "Zone 2 — Some Prep",
    3: "Zone 3 — Medium Prep",
    4: "Zone 4 — Considerable Prep",
    5: "Zone 5 — Extensive Prep",
}
ZONE_COLORS = {
    1: "#b8cfe0", 2: "#8cafc5", 3: "#6090aa",
    4: "#3a6f8f", 5: "#1a4f73",
}

IMPORTANCE_THRESHOLD = 3.0
TOP_N_FOR_AVERAGE = 10

# Linear projection horizon for the major-categories trend chart
PROJECTION_DAYS = 730  # 2 years


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _load_occ_structural() -> pd.DataFrame:
    """Per-occ pct_physical, computed over UNIQUE (occ, task) pairs.
    See ANALYSIS_ARCHITECTURE.md Common Pitfalls — eco_2025 expands tasks
    across GWA/IWA/DWA non-proportionally between physical and non-physical
    tasks, so dedup is required before counting."""
    eco = pd.read_csv(DATA_DIR / "final_eco_2025.csv")
    eco_unique = eco.drop_duplicates(["title_current", "task_normalized"])
    occ = (
        eco_unique.groupby("title_current")
        .agg(
            n_tasks=("physical", "count"),
            n_physical=("physical", "sum"),
            job_zone=("job_zone", "first"),
        )
        .reset_index()
    )
    occ["pct_physical"] = occ["n_physical"] / occ["n_tasks"] * 100
    occ["occ_group"] = "Mixed"
    occ.loc[occ["pct_physical"] < PHYS_LOWER, "occ_group"] = "Non-physical"
    occ.loc[occ["pct_physical"] > PHYS_UPPER, "occ_group"] = "Physical"
    return occ


# ──────────────────────────────────────────────────────────────────────────
# ska_full — element-level SKA mirroring the main-body framing.
# Bar = AI Top-10 Avg as % of workforce max, colored by phys-mix tier.
# Dots = AI Max (red diamond) + Workforce Mean (black circle).
# Knowledge / Abilities labels carry their O*NET subcategory in parentheses.
# ──────────────────────────────────────────────────────────────────────────

def _compute_ska_variants(
    onet_df: pd.DataFrame,
    pct_series: pd.Series,
    type_name: str,
    phys_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Element-level AI / workforce variants. When `phys_map` is supplied,
    each row carries a `phys_tier` bucket (Physical / Mixed / Non-physical)."""
    from lib.builders.part2 import _phys_tier
    df = onet_df.copy()
    df["pct"] = df["title"].map(pct_series)
    df = df.dropna(subset=["pct", "importance", "level"])
    df = df[df["importance"] >= IMPORTANCE_THRESHOLD].copy()
    assert len(df) > 0, f"No {type_name} rows after importance filter"

    df["occ_score"] = df["importance"] * df["level"]
    df["ai_product"] = (df["pct"] / 100.0) * df["occ_score"]
    if phys_map is not None:
        df["pct_physical_occ"] = df["title"].map(phys_map)

    records: list[dict] = []
    for element_name, grp in df.groupby("element_name"):
        ai_vals = grp["ai_product"].dropna()
        occ_vals = grp["occ_score"].dropna()
        n_ai = len(ai_vals)
        n_occ = len(occ_vals)
        top_n_ai = min(TOP_N_FOR_AVERAGE, n_ai)
        rec = {
            "element_name": element_name,
            "type": type_name,
            "n_occs": n_ai,
            "ai_max":   float(ai_vals.max()) if n_ai >= 1 else float("nan"),
            "ai_top10": float(ai_vals.nlargest(top_n_ai).mean()) if n_ai >= 1 else float("nan"),
            "eco_max":  float(occ_vals.max()) if n_occ >= 1 else float("nan"),
            "eco_mean": float(occ_vals.mean()) if n_occ >= 1 else float("nan"),
        }
        if phys_map is not None:
            phys_vals = grp["pct_physical_occ"].dropna()
            phys_score = float(phys_vals.mean()) if len(phys_vals) else float("nan")
            rec["phys_score"] = phys_score
            rec["phys_tier"] = (
                _phys_tier(phys_score) if pd.notna(phys_score) else "Non-physical"
            )
        records.append(rec)
    return pd.DataFrame(records)


def _element_subcat_lookup(onet_path: Path, cat_fn) -> dict[str, str]:
    """element_name → subcategory, built from an O*NET v30.1 CSV using the
    Element-ID-keyed mappings shared with the part_2 subcategory rollups."""
    df = pd.read_csv(onet_path, dtype=str).rename(columns={
        "Element ID": "element_id", "Element Name": "element_name",
    })
    df = df.drop_duplicates(subset=["element_id", "element_name"])
    return {row.element_name: cat_fn(row.element_id) for row in df.itertuples()}


def _build_one_ska_full_chart(
    df: pd.DataFrame,
    type_label: str,
    subcat_map: dict[str, str] | None,
    out_name: str,
    results: Path,
    figures: Path,
) -> None:
    """One element-level appendix chart: bar = AI Top-10 % colored by phys-mix
    tier, dots = AI Max ◆ + Workforce Mean ●, 3-row legend, Knowledge/Abilities
    labels carry their O*NET subcategory in parens."""
    from lib.builders.part2 import (
        SKA_BAR_COLOR_BY_TIER, SKA_TIER_LEGEND_ORDER, AI_MARKER_COLOR,
        WORKFORCE_MEAN_COLOR,
    )

    px = paper_fonts(PAPER_W)

    df = df.copy()
    df["ai_top10_pct"] = df["ai_top10"] / df["eco_max"].replace(0, float("nan")) * 100
    df["ai_max_pct"]   = df["ai_max"]   / df["eco_max"].replace(0, float("nan")) * 100
    df["eco_mean_pct"] = df["eco_mean"] / df["eco_max"].replace(0, float("nan")) * 100
    df = df.sort_values("ai_top10_pct", ascending=False).reset_index(drop=True)

    if subcat_map is not None:
        labels = [f"{e}  ({subcat_map.get(e, '—')})" for e in df["element_name"]]
    else:
        labels = df["element_name"].tolist()

    bar_vals   = df["ai_top10_pct"].fillna(0).tolist()
    max_vals   = df["ai_max_pct"].fillna(0).tolist()
    emean_vals = df["eco_mean_pct"].fillna(0).tolist()

    if "phys_tier" in df.columns:
        bar_colors = [SKA_BAR_COLOR_BY_TIER.get(t, METRIC_COLORS["tasks"])
                      for t in df["phys_tier"].fillna("Non-physical")]
    else:
        bar_colors = [METRIC_COLORS["tasks"]] * len(labels)

    margin_t, margin_b = 60, 230
    margin_r = 60
    # y-tick font drops to the 8 pt floor (set in update_yaxes below) so the
    # per-row pitch can compress from 42 → 32. Bottom margin holds axis
    # title (~50 px) + the tightened 3-row legend (~170 px) + buffer.
    fig_height = max(800, len(labels) * 32 + margin_t + margin_b)
    # Left margin = rotated y-axis title (~30 px) + longest tick label
    # (~11–12 px/char at 8 pt Inter on the appendix's wider canvas) +
    # buffer. Skills/Knowledge factor is 13 because their longest labels
    # ("Management of Personnel Resources  (Resource Management)" — 56
    # chars) would otherwise overflow into the rotated y-axis title.
    # Abilities sits at factor 12 — its longest label is similar length
    # but real glyph width is slightly tighter, and factor 13 left a
    # visible empty band between the rotated title and the labels.
    max_label_chars = max(len(lab) for lab in labels)
    if type_label == "Abilities":
        margin_l = max(220, max_label_chars * 12 + 90)
    else:
        margin_l = max(440, max_label_chars * 13 + 110)

    fig = go.Figure()

    # Plotly legend hidden — we draw a manual 3-row legend with shapes +
    # annotations below the plot. More reliable than multi-legend, which
    # wraps unpredictably for wider entries.
    fig.add_trace(go.Bar(
        y=labels, x=[100] * len(labels), orientation="h",
        name="Workforce Max",
        marker=dict(color="#e8e8e2", line=dict(width=0)),
        hovertemplate="Workforce max: 100%<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=bar_vals, orientation="h",
        name="AI Top-10 Avg  (color = phys tier ↓)",
        marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in bar_vals],
        textposition="outside",
        textfont=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["text"], family=FONT_FAMILY),
        hovertemplate="AI Top-10 avg (% of workforce max): %{x:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=max_vals, mode="markers",
        name="AI Max",
        marker=dict(color=AI_MARKER_COLOR, symbol="diamond", size=11,
                    opacity=0.75),
        hovertemplate="AI Max (% of workforce max): %{x:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=emean_vals, mode="markers",
        name="Workforce Mean",
        marker=dict(color=WORKFORCE_MEAN_COLOR, symbol="circle", size=8,
                    opacity=0.75,
                    line=dict(width=1, color=WORKFORCE_MEAN_COLOR)),
        hovertemplate="Workforce mean (% of workforce max): %{x:.1f}%<extra></extra>",
        showlegend=False,
    ))

    # Same manual legend helper as the main body Part 2 charts, so swatch
    # / symbol sizing stays identical across all five SKA figures.
    from lib.builders.part2 import (
        draw_ska_manual_legend, ska_legend_rows,
    )
    draw_ska_manual_legend(
        fig, ska_legend_rows(),
        fig_width=PAPER_W, fig_height=fig_height,
        margin_l=margin_l, margin_r=margin_r,
        margin_t=margin_t, margin_b=margin_b,
        legend_font_px=px["legend"],
    )

    fig.update_layout(
        title=dict(
            text=f"AI Capability as % of Workforce Max — Full {type_label} Elements",
            font=dict(size=px["title"], color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
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
    fig.update_yaxes(
        title=dict(
            text=f"O*NET {type_label} Element  (Subcategory)",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
            # standoff = distance from axis line to title's center. Setting
            # near margin_l pushes the rotated title flush to the figure's
            # left edge instead of leaving it floating in the margin.
            standoff=max(0, margin_l - 40),
        ),
        type="category",
        autorange=False, range=[len(labels) - 0.5, -0.3],
        automargin=False,  # honor margin_l exactly; don't auto-expand
        tickmode="array", tickvals=labels, ticktext=labels,
        tickfont=dict(size=px["in_chart_floor"], color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        showgrid=False, showline=False,
    )
    fig.update_xaxes(
        title=dict(
            text="AI Capability as % of Workforce Max",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
            standoff=15,
        ),
        range=[0, 100], ticksuffix="%",
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True,
        tickfont=dict(size=px["tick"], color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showline=False, zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
    )

    save_figure(fig, results / "figures" / out_name, scale=2)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_ska_full(results: Path, figures: Path) -> None:
    """Two appendix SKA charts — knowledge and abilities — at the element
    level. The skills element-level chart was promoted to the main body
    (part_2/ska_skills.png) so it no longer ships here."""
    from lib.builders.part2 import (
        _load_occ_phys_map,
        _ability_subcat, _knowledge_cat,
    )

    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    ska_data = load_ska_data()
    phys_map = _load_occ_phys_map()

    know_subcat = _element_subcat_lookup(
        REFERENCE_DIR / "knowledge_v30.1.csv", _knowledge_cat,
    )
    abil_subcat = _element_subcat_lookup(
        REFERENCE_DIR / "abilities_v30.1.csv", _ability_subcat,
    )

    specs = [
        ("knowledge", "Knowledge", ska_data.knowledge, know_subcat,  "ska_knowledge_full.png"),
        ("abilities", "Abilities", ska_data.abilities, abil_subcat,  "ska_abilities_full.png"),
    ]

    elements_all: list[pd.DataFrame] = []
    for type_name, type_label, onet_df, subcat_map, out_name in specs:
        df = _compute_ska_variants(onet_df, pct, type_name, phys_map=phys_map)
        elements_all.append(df)
        print(f"    {type_name}: {len(df)} elements")
        _build_one_ska_full_chart(df, type_label, subcat_map, out_name, results, figures)

    save_csv(
        pd.concat(elements_all, ignore_index=True),
        results / "ska_full.csv", float_format="%.4f",
    )

    # Remove the prior single-chart artifact if still present.
    for stale in ("ska_full.png",):
        for d in (results / "figures", figures):
            p = d / stale
            if p.exists():
                p.unlink()


def build_convergence_full(
    results: Path,
    figures: Path,
    levels: list[tuple[str, str]] | None = None,
    out_name: str = "convergence_full.png",
    csv_name: str = "spearman_combined_full.csv",
) -> None:
    """Full square correlation matrix: every internal measure (4 AI sources +
    5 ANALYSIS_CONFIGS) and every external benchmark (8 academic indices)
    on both x and y axes, lower-triangular cells only. Two panels stacked
    vertically — `levels` selects which SOC levels (default Major + Occ).

    A blank gap row + gap column separate the internal section from the
    external section on both axes. Cell rendering, gray-out, and the
    contamination legend follow the conventions of the main paper charts.
    """
    from scipy import stats
    from lib.builders.part1 import (
        CORR_SOURCES, CORR_ORDER, CORR_LABELS,
        CONFIG_ORDER,
        EXT_SOURCES,
        ELOUNDOU_LABELS,
        CONTAMINATED_SOURCE_ROWS, CONTAMINATED_CONFIG_ROWS,
        SIG_NOTE,
        _run_config,
        _load_eloundou_occ, _compute_aioe_occ,
        _load_schaal_occ, _load_tomlinson_occ,
        _ext_at_level, _stars,
        _wrap_tick_labels,
    )
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

    internal_data: dict[str, dict[str, pd.Series]] = {}
    for skey in CORR_ORDER:
        ds = CORR_SOURCES[skey]["dataset"]
        internal_data[skey] = {}
        for lvl, _ in LEVELS:
            df = _run_config(ds, lvl)
            internal_data[skey][lvl] = df.set_index("category")["pct_tasks_affected"]
        print(f"  {CORR_SOURCES[skey]['label']}: loaded {[l for l, _ in LEVELS]}")
    for ckey in CONFIG_ORDER:
        ds = paper_dataset_for(ckey)
        internal_data[ckey] = {}
        for lvl, _ in LEVELS:
            df = _run_config(ds, lvl)
            internal_data[ckey][lvl] = df.set_index("category")["pct_tasks_affected"]
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
    # plus a near-vertical tick angle (-85°, set below) is the only
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
                records.append({
                    "level": level,
                    "measure_a": all_labels[i],
                    "measure_b": all_labels[j],
                    "rho": round(float(rho), 3),
                    "p_value": round(float(pval), 6),
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
    legend_block_h = legend_line_h * 2 + 20         # 2 lines + buffer
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
        title=f"Full-Matrix Benchmark Comparison ({level_names} Level)",
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
    # Vertical anchor unchanged from before — legend sits a fixed gap
    # below the x-tick labels in paper coords.
    sy_center = -(legend_top_offset + legend_block_h / 2) / plot_h
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

    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def _run_overview_config(dataset_name: str, use_auto_aug: bool) -> pd.DataFrame:
    from backend.compute import get_group_data
    config = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": use_auto_aug,
        "physical_mode": "all",
        "geo": "nat",
        "agg_level": "occupation",
        "sort_by": "% Tasks Affected",
        "top_n": 9999,
        "search_query": "",
        "context_size": 3,
    }
    data = get_group_data(config)
    assert data is not None, f"No data for {dataset_name}"
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    return df.rename(columns={group_col: "category"})


def _national_totals_emp_wages() -> tuple[float, float]:
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    occ = eco.drop_duplicates(subset=["title_current"])
    total_emp = float(occ["emp_tot_nat_2025"].sum())
    total_wages = float((occ["emp_tot_nat_2025"] * occ["a_med_nat_2025"]).sum())
    return total_emp, total_wages


def _eco_tc_by_occ_app() -> pd.Series:
    """Per-occupation eco baseline task_comp sum (freq, no auto-aug, all phys).
    Denominator for the economy-wide ratio-of-totals % tasks number — mirrors
    `_eco_task_comp_by_occ` in part_1/run.py."""
    from backend.compute import load_eco_baseline
    eco = load_eco_baseline(method="freq", physical_mode="all", geo="nat")
    return eco.groupby("title_current")["task_comp"].sum()


def _econ_pct_tasks(df: pd.DataFrame, eco_tc_by_occ: pd.Series) -> float:
    """Ratio-of-totals % tasks across all (task, occ) pairs in the economy:
    Σ_occ (pct[occ]/100 × eco_tc[occ]) / Σ_occ eco_tc[occ] × 100."""
    eco_tc_total = float(eco_tc_by_occ.sum())
    if eco_tc_total <= 0:
        return 0.0
    eco_aligned = df["category"].map(eco_tc_by_occ).fillna(0.0)
    ai_tc_total = float(((df["pct_tasks_affected"] / 100.0) * eco_aligned).sum())
    return ai_tc_total / eco_tc_total * 100.0


def _compute_paper_overview_rows(total_emp: float, total_wages: float) -> list[dict]:
    """Reproduce the paper part_1 build_overview values (auto_aug=True, method=freq)
    so variant charts can show delta-vs-paper."""
    eco_tc_by_occ = _eco_tc_by_occ_app()
    rows: list[dict] = []
    for key in OVERVIEW_CONFIG_ORDER:
        ds = paper_dataset_for(key)
        df = _run_overview_config(ds, use_auto_aug=True)
        workers = float(df["workers_affected"].sum())
        wages = float(df["wages_affected"].sum())
        pct_tasks = _econ_pct_tasks(df, eco_tc_by_occ)
        rows.append({
            "config": key,
            "pct_tasks": round(pct_tasks, 1),
            "pct_workers": round(workers / total_emp * 100, 1),
            "pct_wages": round(wages / total_wages * 100, 1),
        })
    return rows


def _render_overview_with_deltas(
    rows: list[dict],
    paper_rows: list[dict],
    title: str,
    subtitle: str,
    out_name: str,
    results: Path,
    figures: Path,
    x_range_max: float = 75.0,
) -> None:
    """Render the overview chart with delta-vs-paper annotated inside each bar
    AND a thin vertical marker on each bar at the paper chart's value (so the
    reader can see where the original landed without flipping back)."""
    paper_lookup = {p["config"]: p for p in paper_rows}

    fig = go.Figure()
    plot_rows = list(reversed(rows))
    labels = [r["label"] for r in plot_rows]

    metrics = [
        ("pct_tasks",   "Tasks Exposed",
         METRIC_COLORS["tasks"], "pct_tasks",
         lambda r, d: f"{r['pct_tasks']:.1f}%  Δ{d:+.1f}pp"),
        ("pct_workers", "Workers Exposed",
         METRIC_COLORS["workers"], "pct_workers",
         lambda r, d: f"{fmt_workers(r['workers'])} ({r['pct_workers']:.1f}%)  Δ{d:+.1f}pp"),
        ("pct_wages",   "Wages Exposed",
         METRIC_COLORS["wages"], "pct_wages",
         lambda r, d: f"{fmt_wages(r['wages'])} ({r['pct_wages']:.1f}%)  Δ{d:+.1f}pp"),
    ]

    # All font sizes resolved from the standardized pt ladder via
    # paper_fonts(PAPER_W), matching the paper part_1 overview chart so the
    # two figures read at identical print pt sizes.
    px = paper_fonts(PAPER_W)

    # Build per-(row, metric) lookups of value strings so we can emit text
    # as manual annotations centered between the bar start (x=0) and the
    # paper-chart-value tick. Bars themselves carry no text — they only
    # render the colored fill.
    text_strings: dict[tuple[int, str], str] = {}
    for pct_key, name, color, paper_key, fmt_fn in reversed(metrics):
        for y_idx, r in enumerate(plot_rows):
            paper_val = paper_lookup[r["config"]][paper_key]
            delta = r[pct_key] - paper_val
            text_strings[(y_idx, pct_key)] = fmt_fn(r, delta)
        fig.add_trace(go.Bar(
            y=labels,
            x=[r[pct_key] for r in plot_rows],
            name=name,
            orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            showlegend=False,
        ))

    # Legend entries are emitted as dummy scatter traces in display order
    # (marker first, then tasks → workers → wages) so the legend reads
    # left-to-right in that order. Scatter markers (rather than bar
    # traces) let us set marker.size to make swatches visibly bigger.
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            symbol="line-ns",
            color="rgba(20,20,20,0.95)",
            size=22,
            line=dict(color="rgba(20,20,20,0.95)", width=2),
        ),
        name="Value with Auto-Aug weighting",
        showlegend=True,
        hoverinfo="skip",
    ))
    for pct_key, name, color, _, _ in metrics:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", size=22, color=color),
            name=name,
            showlegend=True,
            hoverinfo="skip",
        ))

    # Vertical "where the paper chart landed" markers, one per bar in each cluster.
    # bargap/bargroupgap match the paper part_1 overview chart so bar heights
    # are identical between the two figures (the appendix figure differs only
    # in its 2-row legend, not its bar geometry).
    n_per_cluster = 3
    bargap = 0.18
    bargroupgap = 0.04
    cluster_span = 1.0 - bargap
    bar_pitch = cluster_span / n_per_cluster
    bar_height = bar_pitch * (1.0 - bargroupgap)
    half_span = cluster_span / 2.0
    # Plotly grouped bars order: trace 0 at the BOTTOM of the cluster.
    # Our metrics were added in reverse so wages=trace0, workers=trace1, tasks=trace2.
    sub_centers = {
        "pct_wages":   -half_span + 0.5 * bar_pitch,
        "pct_workers": -half_span + 1.5 * bar_pitch,
        "pct_tasks":   -half_span + 2.5 * bar_pitch,
    }
    # Tick markers span the full bar height. The per-bar value labels are
    # placed as annotations: centered between x=0 and the tick when the
    # text fits there (so the tick never crosses text); otherwise placed
    # outside the bar (after the bar end) in dark text on the white
    # background.
    # Text-width estimate in axis units: at 9pt (27 px) Inter on a
    # 1400 px canvas spanning 75 x-units, one character is ≈ 1.0 axis units.
    # Buffer of 2 units on each side keeps text clear of the tick / bar
    # edge under kerning variance and avoids borderline-fit labels
    # squeezing against the tick.
    char_w_units = 1.0
    edge_buffer = 2.0
    shapes = []
    text_annotations: list[dict] = []
    for y_idx, r in enumerate(plot_rows):
        paper_r = paper_lookup[r["config"]]
        for pct_key, paper_key in [("pct_tasks", "pct_tasks"),
                                    ("pct_workers", "pct_workers"),
                                    ("pct_wages", "pct_wages")]:
            xv = paper_r[paper_key]
            bar_val = r[pct_key]
            yc = y_idx + sub_centers[paper_key]
            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=xv, x1=xv,
                y0=yc - bar_height / 2.0,
                y1=yc + bar_height / 2.0,
                line=dict(color="rgba(20,20,20,0.95)", width=2),
                layer="above",
            ))

            label = text_strings[(y_idx, pct_key)]
            text_w = len(label) * char_w_units
            available = xv - edge_buffer  # room between bar start and tick
            if text_w <= available:
                # Fits in [0, tick]: center there, white text on bar.
                text_annotations.append(dict(
                    xref="x", yref="y",
                    x=xv / 2.0, y=yc,
                    text=label, showarrow=False,
                    xanchor="center", yanchor="middle",
                    font=dict(size=px["tick"], color="white", family=FONT_FAMILY),
                ))
            else:
                # Doesn't fit; place outside the bar in dark text.
                text_annotations.append(dict(
                    xref="x", yref="y",
                    x=bar_val + 0.5, y=yc,
                    text=label, showarrow=False,
                    xanchor="left", yanchor="middle",
                    font=dict(size=px["tick"],
                              color=PAPER_PALETTE["text"], family=FONT_FAMILY),
                ))

    fig.update_layout(
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        legend=dict(traceorder="normal"),
        xaxis=dict(
            title=dict(text="% of National Total",
                       font=dict(size=px["axis_title"], family=FONT_FAMILY)),
            range=[0, x_range_max],
            ticksuffix="%",
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        ),
        yaxis=dict(
            title=dict(text="Data Configuration",
                       font=dict(size=px["axis_title"], family=FONT_FAMILY)),
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        ),
        shapes=shapes,
        annotations=text_annotations,
    )

    # Plot area sized to match the main-body overview chart exactly. The
    # bottom margin is sized to hold x-axis ticks + axis title + the
    # 2-row legend with no trailing whitespace (the legend is pinned to
    # the figure bottom via yref="container").
    style_paper_figure(
        fig, title, subtitle=subtitle,
        height=PAPER_H + 320,
        margin=dict(l=20, r=60, t=90, b=230),
    )

    # Legend in container coords (0-1 of full figure width/height) so the
    # asymmetric l/r margins don't shift it off the figure center, pinned
    # to the figure bottom edge (yanchor=bottom). itemsizing="trace" lets
    # each dummy scatter's marker.size drive the legend swatch size.
    fig.update_layout(
        legend=dict(
            orientation="h",
            xref="container", yref="container",
            x=0.5, xanchor="center",
            y=0.01, yanchor="bottom",
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

    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_overview_no_autoaug(results: Path, figures: Path) -> None:
    """Variant of paper part_1 build_overview with auto_aug weighting off.
    Every affected task contributes its full freq weight regardless of its
    0–5 automatability rating. Same five configs, same layout. Each bar
    carries the delta-vs-paper-chart in percentage points (and a small
    black tick on the bar at the paper chart's value)."""
    total_emp, total_wages = _national_totals_emp_wages()
    paper_rows = _compute_paper_overview_rows(total_emp, total_wages)
    eco_tc_by_occ = _eco_tc_by_occ_app()

    rows: list[dict] = []
    for key in OVERVIEW_CONFIG_ORDER:
        ds = paper_dataset_for(key)
        label = ANALYSIS_CONFIG_LABELS[key]
        df = _run_overview_config(ds, use_auto_aug=False)

        workers = float(df["workers_affected"].sum())
        wages = float(df["wages_affected"].sum())
        pct_tasks = _econ_pct_tasks(df, eco_tc_by_occ)
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

    save_csv(pd.DataFrame(rows), results / "overview_no_autoaug_totals.csv")

    _render_overview_with_deltas(
        rows, paper_rows,
        title="AI Economic Exposure Across Data Configurations — No Auto-Aug Weighting",
        subtitle=(
            "Each affected task contributes its full freq weight regardless of its 0–5 automatability score."
            "<br>Δ inside each bar = delta vs. the paper chart in percentage points."
        ),
        out_name="overview_no_autoaug.png",
        results=results, figures=figures,
        x_range_max=75.0,
    )


def _run_config_phys_mode(dataset_name: str, physical_mode: str,
                          agg_level: str = "occupation") -> pd.DataFrame:
    """`physical_mode` must be one of 'all' / 'exclude' / 'only' —
    `apply_physical_filter` silently ignores anything else (real foot-gun)."""
    from backend.compute import get_group_data
    assert physical_mode in {"all", "exclude", "only"}, physical_mode
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
    assert data is not None, f"No data for {dataset_name}"
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    df = df.rename(columns={group_col: "category"})
    return df


def _run_config_nonphys(dataset_name: str, agg_level: str = "occupation") -> pd.DataFrame:
    """Back-compat shim — non-physical task filter."""
    return _run_config_phys_mode(dataset_name, "exclude", agg_level)


def build_temporal_trend_nonphys(results: Path, figures: Path) -> None:
    """Single-panel % tasks exposed trend, split by physical task filter.

    Three lines:
    - All Confirmed (non-physical tasks only)
    - All Sources / Ceiling (non-physical tasks only)
    - All Confirmed (physical tasks only)

    Workers and wages panels are omitted: a phys/non-phys split of
    workers would require splitting each occupation's employment between
    its phys and non-phys task load, which is out of scope here."""

    # (config_key, physical_mode, series_key, line_label, line color, dash)
    series_spec: list[tuple[str, str, str, str, str, str]] = [
        ("all_confirmed", "exclude", "confirmed_nonphys",
         "All Confirmed — Non-physical Tasks",
         METRIC_COLORS["tasks"], "solid"),
        ("all_ceiling", "exclude", "ceiling_nonphys",
         "All Sources (Ceiling) — Non-physical Tasks",
         METRIC_COLORS_LIGHT["tasks"], "dash"),
        ("all_confirmed", "only", "confirmed_phys",
         "All Confirmed — Physical Tasks",
         METRIC_COLORS["workers"], "solid"),
    ]

    # ── 1. Build trend data ──────────────────────────────────────────
    # Pre-compute the eco_task_comp denominator for each physical mode in
    # use, so the % tasks number is a ratio-of-totals across the matching
    # eco subset (non-physical tasks only / physical tasks only).
    from backend.compute import load_eco_baseline
    eco_tc_by_mode: dict[str, pd.Series] = {}
    for phys_mode in {p for _, p, _, _, _, _ in series_spec}:
        eco_phys = load_eco_baseline(method="freq", physical_mode=phys_mode, geo="nat")
        eco_tc_by_mode[phys_mode] = (
            eco_phys.groupby("title_current")["task_comp"].sum()
        )

    trend_rows: list[dict] = []
    for config_key, phys_mode, series_key, label, _color, _dash in series_spec:
        series = ANALYSIS_CONFIG_SERIES[config_key]
        eco_tc_by_occ = eco_tc_by_mode[phys_mode]
        eco_tc_total = float(eco_tc_by_occ.sum())
        for ds_name in series:
            date_str = ds_name.rsplit(" ", 1)[-1]
            df = _run_config_phys_mode(ds_name, phys_mode, "occupation")
            # Ratio-of-totals across (task, occ) pairs in the matching eco
            # subset (phys-filtered both sides).
            eco_tc_aligned = df["category"].map(eco_tc_by_occ).fillna(0.0)
            ai_tc_total = float(((df["pct_tasks_affected"] / 100.0) * eco_tc_aligned).sum())
            pct_tasks = (ai_tc_total / eco_tc_total * 100.0) if eco_tc_total > 0 else 0.0
            trend_rows.append({
                "series": series_key,
                "config": config_key,
                "physical_mode": phys_mode,
                "label": label,
                "date": date_str,
                "dataset": ds_name,
                "pct_tasks_affected": round(pct_tasks, 1),
            })
            print(f"  {label} {date_str}: {pct_tasks:.1f}%")
    trend_df = pd.DataFrame(trend_rows)
    save_csv(trend_df, results / "temporal_trend_nonphys.csv")

    # Paper-chart font ladder (see ANALYSIS_CLAUDE.md → Paper Chart Formatting).
    # All sizes come from paper_fonts(TRENDNP_W) so printed pt at 6.5"
    # column matches the standard 11/10/10/9/9/8 ladder.
    TRENDNP_W = PAPER_W
    TRENDNP_H = PAPER_H + 320
    px = paper_fonts(TRENDNP_W)
    TRENDNP_TITLE_FS = px["title"]
    TRENDNP_AXIS_FS = px["axis_title"]
    TRENDNP_TICK_FS = px["tick"]
    TRENDNP_LEGEND_FS = px["legend"]
    LABEL_FS_DATA = px["in_chart_floor"]
    LABEL_FS_HORIZON = px["in_chart_floor"]
    LABEL_YSHIFT_PX = 32

    # Linear OLS extrapolation: extend each line at its recent rate to a
    # 2-year horizon. Labeled only at the 2yr endpoint per series.
    EXTRAP_HORIZONS_DAYS: list[tuple[str, int]] = [
        ("6mo", 183), ("1yr", 365), ("2yr", 730),
    ]

    def _linear_fit_project(dates: list[str], yvals: list[float],
                            horizon_days: list[int]) -> tuple[list[pd.Timestamp], list[float]]:
        if len(dates) < 2 or not horizon_days:
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
        if not dates:
            return set()
        parsed = [pd.Timestamp(d) for d in dates]
        keep = [len(dates) - 1]
        for i in range(len(dates) - 2, -1, -1):
            if (parsed[keep[-1]] - parsed[i]).days >= min_days:
                keep.append(i)
        return set(keep)

    # ── 3. Build chart ───────────────────────────────────────────────
    fig = go.Figure()

    # Legend will be rendered manually after the chart is laid out
    # (paper-space line shapes + annotations). Plotly's auto-legend
    # with three long items wraps to two rows but left-aligns the
    # second row, leaving the block off-center vs the figure midpoint.
    legend_color = PAPER_PALETTE["text"]

    panel_vals: list[float] = []
    for idx, (config_key, _phys, series_key, label, color, dash) in enumerate(series_spec):
        subset = trend_df[trend_df["series"] == series_key].sort_values("date").reset_index(drop=True)

        # Stagger label position: alternate below / above the marker so
        # the three series' labels don't pile up.
        yshift = -LABEL_YSHIFT_PX if idx % 2 == 0 else LABEL_YSHIFT_PX

        xvals = list(subset["date"])
        yvals = [float(v) for v in subset["pct_tasks_affected"]]
        panel_vals.extend(yvals)

        fig.add_trace(go.Scatter(
            x=xvals, y=yvals,
            name=label,
            showlegend=False,
            mode="lines+markers",
            line=dict(color=color, width=3, dash=dash),
            marker=dict(size=8, color=color),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y}}%<extra></extra>",
            cliponaxis=False,
        ))

        # Linear OLS projection to the 2yr horizon (labels only at 2yr).
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
                hovertemplate=(
                    f"<b>{label} (linear proj.)</b><br>%{{x}}<br>%{{y}}%<extra></extra>"
                ),
                cliponaxis=False,
                opacity=0.7,
            ))
            panel_vals.extend(future_ys)
            hz_label, _ = EXTRAP_HORIZONS_DAYS[twoyr_idx]
            fig.add_annotation(
                x=future_ts[twoyr_idx], y=future_ys[twoyr_idx],
                text=f"{hz_label}: {future_ys[twoyr_idx]:.1f}%",
                showarrow=False,
                yshift=yshift,
                font=dict(size=LABEL_FS_HORIZON, color=color, family=FONT_FAMILY),
            )

        # Per-point data labels. Confirmed series get spaced labels;
        # ceiling shows only its last point — its first point sits on
        # the confirmed-non-phys first point at the same value, so a
        # label there would just stack a duplicate.
        if config_key == "all_confirmed":
            kept_set = _spaced_label_indices(xvals)
        elif len(xvals) >= 2:
            kept_set = {len(xvals) - 1}
        else:
            kept_set = set(range(len(xvals)))

        CLOSE_GAP_DAYS = 120
        STAGGER_PX = 18
        kept_sorted = sorted(kept_set)
        kept_ts = {i: pd.Timestamp(xvals[i]) for i in kept_sorted}
        per_label_yshift: dict[int, int] = {}
        for pos, i in enumerate(kept_sorted):
            shift = yshift
            if pos + 1 < len(kept_sorted):
                nxt = kept_sorted[pos + 1]
                if (kept_ts[nxt] - kept_ts[i]).days < CLOSE_GAP_DAYS:
                    shift = yshift + (-STAGGER_PX if yshift < 0 else STAGGER_PX)
            per_label_yshift[i] = shift

        for i, (x_i, y_i) in enumerate(zip(xvals, yvals)):
            if i not in kept_set:
                continue
            fig.add_annotation(
                x=x_i, y=y_i,
                text=f"{y_i:.1f}%",
                showarrow=False,
                yshift=per_label_yshift[i],
                font=dict(size=LABEL_FS_DATA, color=color, family=FONT_FAMILY),
            )

    if panel_vals:
        v_lo, v_hi = min(panel_vals), max(panel_vals)
        spread = max(v_hi - v_lo, 1.0)
        y_min = max(0.0, v_lo - spread * 0.18)
        y_max = v_hi + spread * 0.18
    else:
        y_min, y_max = 0.0, 1.0

    fig.update_yaxes(
        ticksuffix="%",
        range=[y_min, y_max],
        title=dict(text="Tasks Exposed",
                   font=dict(size=TRENDNP_AXIS_FS)),
        tickfont=dict(size=TRENDNP_TICK_FS, family=FONT_FAMILY),
    )
    fig.update_xaxes(
        title=dict(text="Snapshot Date", font=dict(size=TRENDNP_AXIS_FS)),
        tickangle=-30,
        tickfont=dict(size=TRENDNP_TICK_FS, family=FONT_FAMILY),
    )

    style_paper_figure(
        fig,
        "Trend Over Time — Physical vs Non-physical Tasks",
        subtitle="",
        height=TRENDNP_H,
        width=TRENDNP_W,
        margin=dict(l=90, r=60, t=130, b=290),
    )
    fig.update_layout(
        title=dict(font=dict(size=TRENDNP_TITLE_FS)),
        showlegend=False,
    )

    # ── Manual legend (paper space) ─────────────────────────────────────
    # Two rows centered independently. Plotly's auto-legend wraps long
    # 3-item legends to two rows but left-aligns the second row, leaving
    # the block visibly off-center. Each row here uses the same
    # text-width-aware centering as the main-body trend legend.
    LEG_LINE_LEN = 0.044
    LEG_TEXT_GAP = 0.010
    LEG_ITEM_SPACING = 0.05
    LEG_ROW1_Y = -0.24
    LEG_ROW2_Y = -0.34
    char_w = TRENDNP_LEGEND_FS * 0.55 / TRENDNP_W

    def _item_width_nonphys(label: str) -> float:
        return LEG_LINE_LEN + LEG_TEXT_GAP + len(label) * char_w

    # Same series order as the lines themselves; first two ride row 1.
    # Row 2 holds the physical-tasks series alongside the OLS projection
    # style key. The projection isn't a series of its own — it's a style
    # annotation over each line — so we render it once in neutral text
    # color and use the 4th tuple slot to flag the dotted+× swatch.
    PROJ_LABEL = "2-yr Linear OLS Projection"
    legend_rows: list[list[tuple[str, str, str, bool]]] = [
        [
            (series_spec[0][3], series_spec[0][4], series_spec[0][5], False),
            (series_spec[1][3], series_spec[1][4], series_spec[1][5], False),
        ],
        [
            (series_spec[2][3], series_spec[2][4], series_spec[2][5], False),
            (PROJ_LABEL, PAPER_PALETTE["text"], "dot", True),
        ],
    ]

    for row_idx, row_items in enumerate(legend_rows):
        row_total_w = (
            sum(_item_width_nonphys(lbl) for lbl, _, _, _ in row_items)
            + LEG_ITEM_SPACING * (len(row_items) - 1)
        )
        cursor_x = 0.5 - row_total_w / 2
        leg_y = LEG_ROW1_Y if row_idx == 0 else LEG_ROW2_Y
        for label, color, dash_style, is_proj in row_items:
            line_start = cursor_x
            line_end = cursor_x + LEG_LINE_LEN
            text_x = line_end + LEG_TEXT_GAP
            if is_proj:
                # Two short dotted segments straddling a centered ×,
                # mirroring the chart's dotted-line + × marker pattern
                # without overlap.
                gap_half = 0.006
                mid_x = (line_start + line_end) / 2
                fig.add_shape(
                    type="line", xref="paper", yref="paper",
                    x0=line_start, x1=mid_x - gap_half,
                    y0=leg_y, y1=leg_y,
                    line=dict(color=color, width=2, dash="dot"),
                )
                fig.add_shape(
                    type="line", xref="paper", yref="paper",
                    x0=mid_x + gap_half, x1=line_end,
                    y0=leg_y, y1=leg_y,
                    line=dict(color=color, width=2, dash="dot"),
                )
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=mid_x, y=leg_y,
                    text="×", showarrow=False,
                    xanchor="center", yanchor="middle",
                    font=dict(size=TRENDNP_LEGEND_FS, family=FONT_FAMILY,
                              color=color),
                )
            else:
                fig.add_shape(
                    type="line",
                    xref="paper", yref="paper",
                    x0=line_start, x1=line_end,
                    y0=leg_y, y1=leg_y,
                    line=dict(color=color, width=3, dash=dash_style),
                )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=text_x, y=leg_y,
                text=label, showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=TRENDNP_LEGEND_FS, family=FONT_FAMILY,
                          color=PAPER_PALETTE["text"]),
            )
            cursor_x += _item_width_nonphys(label) + LEG_ITEM_SPACING

    # style_paper_figure resets axis tick/title fonts — re-apply ours.
    fig.update_xaxes(
        tickfont=dict(size=TRENDNP_TICK_FS, family=FONT_FAMILY),
        title_font=dict(size=TRENDNP_AXIS_FS, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        tickfont=dict(size=TRENDNP_TICK_FS, family=FONT_FAMILY),
        title_font=dict(size=TRENDNP_AXIS_FS, family=FONT_FAMILY),
    )

    save_figure(fig, results / "figures" / "temporal_trend_nonphys.png")
    shutil.copy(results / "figures" / "temporal_trend_nonphys.png",
                figures / "temporal_trend_nonphys.png")
    print("  -> temporal_trend_nonphys.png")


# ──────────────────────────────────────────────────────────────────────────
# major_categories_trend — relocated from Part 2.
# Three side-by-side panels (% tasks / workers / wages) for the top-10
# major occupational categories ranked by absolute change in
# pct_tasks_affected from first to final all_confirmed snapshot. Same 10
# majors and same ordering across all panels so the workers/wages context
# reads against the % tasks ranking. Each bar is a three-segment stack:
# solid = first-snapshot value, mid-opacity = observed jump
# (current − first), hatched = 2-year linear OLS projected jump.
# ──────────────────────────────────────────────────────────────────────────

def _major_trend_series() -> pd.DataFrame:
    """Stack the all_confirmed time series at major level into long form.

    Uses ANALYSIS_CONFIG_SERIES['all_confirmed'] (which excludes the 2024
    dates per the trend-series invariant)."""
    from backend.compute import get_group_data

    series = ANALYSIS_CONFIG_SERIES["all_confirmed"]
    rows: list[dict] = []
    for ds in series:
        date_str = ds.rsplit(" ", 1)[-1]
        cfg = {
            "selected_datasets": [ds],
            "combine_method": "Average",
            "method": "freq",
            "use_auto_aug": True,
            "physical_mode": "all",
            "geo": "nat",
            "agg_level": "major",
            "sort_by": "% Tasks Affected",
            "top_n": 9999,
            "search_query": "",
            "context_size": 3,
        }
        data = get_group_data(cfg)
        assert data is not None, f"No data for {ds}"
        df: pd.DataFrame = data["df"].rename(
            columns={data["group_col"]: "category"}
        )
        for _, r in df.iterrows():
            rows.append({
                "date": pd.Timestamp(date_str),
                "category": r["category"],
                "pct_tasks_affected": float(r["pct_tasks_affected"]),
                "workers_affected":   float(r["workers_affected"]),
                "wages_affected":     float(r["wages_affected"]),
            })
        print(f"  loaded {ds}: {len(df)} majors")
    return pd.DataFrame(rows)


def _annot_fmt(metric: str, panel_max: float):
    """Compact 3-value formatter — '{start}→{current}→{proj}<unit>'.

    Picks one unit per panel from ``panel_max`` so all three numbers share
    the same scale (K / M / B / T). Values ≥ 10 print as integers, smaller
    values as one decimal — keeps the annotation short enough to clear
    the per-panel right-side headroom set by the x-axis range factor."""
    if metric == "pct_tasks_affected":
        def fmt_tasks(a, b, c):
            return f"{a:.0f}→{b:.0f}→{c:.0f}%"
        return fmt_tasks

    # Note: no leading "$" — plotly interprets paired "$...$" as MathJax
    # delimiters and silently truncates the annotation at the second one.
    # The wages-panel context comes from the axis title, not the values.
    if metric == "wages_affected":
        if panel_max >= 1e12:
            scale, unit = 1e12, "T"
        elif panel_max >= 1e9:
            scale, unit = 1e9, "B"
        elif panel_max >= 1e6:
            scale, unit = 1e6, "M"
        else:
            scale, unit = 1e3, "K"
    else:  # workers_affected
        if panel_max >= 1e9:
            scale, unit = 1e9, "B"
        elif panel_max >= 1e6:
            scale, unit = 1e6, "M"
        else:
            scale, unit = 1e3, "K"

    def num(v: float) -> str:
        s = v / scale
        return f"{s:.0f}" if abs(s) >= 10 else f"{s:.1f}"

    def fmt(a, b, c):
        return f"{num(a)}→{num(b)}→{num(c)}{unit}"
    return fmt


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a `#rrggbb` string to a Plotly `rgba(r,g,b,a)` string.
    Used to draw legend swatches with explicit per-swatch opacity."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _hatch_png_data_uri(hex_color: str, opacity: float = 0.30) -> str:
    """Generate a data: URI PNG that reproduces plotly's bar
    `pattern(shape="/", solidity=0.25, fgcolor="white")` look — used
    as a layout image for the projected-change legend swatch so the
    pattern matches the bars exactly. Stock plotly Shape doesn't
    support fillpattern; this image overlay is the workaround.
    """
    import io, base64
    from PIL import Image, ImageDraw

    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = int(round(255 * opacity))

    # Generated at higher res than the rendered swatch (22 px) for
    # crisp downscaling. Stripe geometry produces ~5–6 thin diagonals
    # in the rendered swatch — close to plotly's default `/` density.
    SIZE = 110
    img = Image.new("RGBA", (SIZE, SIZE), (r, g, b, a))
    draw = ImageDraw.Draw(img)
    stripe_w = 5
    spacing  = 20    # ≈ solidity 0.25 → 5/20 of perpendicular area is stripe
    # "/" stripes — in PIL coords (y axis points down), draw from
    # (offset, SIZE) to (offset + SIZE, 0) for a bottom-left to
    # top-right tilt.
    for off in range(-SIZE, 2 * SIZE, spacing):
        draw.line(
            [(off, SIZE), (off + SIZE, 0)],
            fill=(255, 255, 255, 255),
            width=stripe_w,
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _strip_occupations_suffix(s: str) -> str:
    """SOC major-group titles all end in ' Occupations'. Drop it — context
    is obvious from the chart and it just consumes label width."""
    suffix = " Occupations"
    return s[: -len(suffix)] if s.endswith(suffix) else s


def _wrap_label(s: str, max_chars: int = 24) -> str:
    """Wrap a long label at word boundaries, joining with <br>. Long
    SOC major titles (e.g. 'Arts, Design, Entertainment, Sports, and Media')
    don't fit on one line at print pt when two panels both carry their
    own y-axis labels — wrapping keeps them legible without dropping below
    the 8 pt floor."""
    if len(s) <= max_chars:
        return s
    words = s.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def _major_trend_stats(
    trend: pd.DataFrame, metric: str,
) -> dict[str, dict]:
    """Compute first / current / projected / jump per major for one metric.

    `_linear_project` lives in Part 2 and is exercised by its tests —
    importing it inline keeps a single canonical implementation."""
    from lib.builders.part2 import _linear_project

    out: dict[str, dict] = {}
    for cat, sub in trend.groupby("category"):
        sub = sub.sort_values("date")
        dates = list(sub["date"])
        yvals = [float(v) for v in sub[metric]]
        if len(dates) < 2 or (yvals[-1] == 0 and yvals[0] == 0):
            continue
        _slope, projected, r2 = _linear_project(dates, yvals, PROJECTION_DAYS)
        out[cat] = {
            "first":     yvals[0],
            "current":   yvals[-1],
            "projected": max(0.0, float(projected)),
            "jump":      yvals[-1] - yvals[0],
            "r2":        r2,
        }
    return out


def _render_major_trend_single(
    results: Path,
    figures: Path,
    metric: str,
    metric_key: str,
    axis_title: str,
    chart_title: str,
    out_name: str,
    stats: dict[str, dict],
) -> None:
    """Render one single-panel horizontal-bar trend chart for one metric.

    ``stats`` provides first / current / projected per category. Top-10
    by absolute jump are plotted, top-mover at top of chart. Inline
    observed-Δ values are placed on the middle bar segment, and the
    full start→current→projected triplet is shown to the right of
    each bar.
    """
    ranked = sorted(stats.items(),
                    key=lambda kv: abs(kv[1]["jump"]),
                    reverse=True)[:10]
    # Plotly h-bars stack bottom→top, so reverse for top-mover-at-top.
    plot_cats = [cat for cat, _ in ranked][::-1]

    firsts = [stats[c]["first"]     for c in plot_cats]
    currs  = [stats[c]["current"]   for c in plot_cats]
    projs  = [stats[c]["projected"] for c in plot_cats]
    seg_start = firsts
    seg_jump  = [c - f for f, c in zip(firsts, currs)]
    # Projection segment clipped at 0 — never extends backward past the
    # current value.
    seg_proj  = [max(0.0, p - c) for c, p in zip(currs, projs)]

    # Y-axis labels: strip the redundant " Occupations" suffix from SOC
    # major titles. With a single panel at PAPER_W, no wrapping needed —
    # long titles like "Arts, Design, Entertainment, Sports, and Media"
    # fit on a single line via plotly automargin.
    y_labels = [_strip_occupations_suffix(c) for c in plot_cats]

    base_color = METRIC_COLORS[metric_key]
    bar_ends = [max(c, p) for c, p in zip(currs, projs)]
    max_x = float(max(bar_ends) or 1.0)

    # ── Fonts from the standardized pt ladder at PAPER_W.
    px = paper_fonts(PAPER_W)

    fig = go.Figure()

    # Three-segment stacked bar: solid start | mid-opacity observed Δ |
    # hatched 2-yr projection. Legend is built manually below as
    # shapes+annotations — gives bigger swatches and reliable layout
    # (plotly's auto-legend wraps to vertical with long y-axis labels).
    fig.add_trace(go.Bar(
        y=y_labels, x=seg_start, orientation="h",
        marker=dict(color=base_color, opacity=1.0, line=dict(width=0)),
        showlegend=False,
        hovertemplate="Start: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=y_labels, x=seg_jump, orientation="h",
        marker=dict(color=base_color, opacity=0.55, line=dict(width=0)),
        showlegend=False,
        hovertemplate="Observed Δ: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=y_labels, x=seg_proj, orientation="h",
        marker=dict(
            color=base_color,
            opacity=0.30,
            line=dict(width=0),
            pattern=dict(shape="/", solidity=0.25, fgcolor="white"),
        ),
        showlegend=False,
        hovertemplate="Projected Δ: %{x:.2f}<extra></extra>",
    ))

    # Right-side annotation per bar — start → current → projected.
    # Full black so values read clearly against the white background.
    annot_fmt = _annot_fmt(metric, max_x)
    for y_lbl, f_v, c_v, p_v in zip(y_labels, firsts, currs, projs):
        fig.add_annotation(
            x=max(c_v, p_v), y=y_lbl,
            xref="x", yref="y",
            text="  " + annot_fmt(f_v, c_v, p_v),
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=px["in_chart_floor"],
                      color="#000000",
                      family=FONT_FAMILY),
        )

    # Inline observed-change value, centered on the middle segment of
    # each bar. Black text so it reads on the mid-opacity fill.
    def _delta_fmt(d: float) -> str:
        if metric == "pct_tasks_affected":
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.0f}pp"
        # workers — scale to match the right-side annotation unit
        abs_max = max_x
        if abs_max >= 1e9:
            scale, unit = 1e9, "B"
        elif abs_max >= 1e6:
            scale, unit = 1e6, "M"
        elif abs_max >= 1e3:
            scale, unit = 1e3, "K"
        else:
            scale, unit = 1.0, ""
        sign = "+" if d >= 0 else ""
        return f"{sign}{d/scale:.1f}{unit}"

    # Inline label sits centered on the Δ segment. When the segment is
    # very narrow (Δ < 4% of x-range) the label visually collides with
    # the bar-end annotation that follows; nudge those a bit to the
    # right of the bar end instead so they don't crowd the start/current
    # arrow string.
    NUDGE_THRESHOLD = 0.04
    for y_lbl, f_v, c_v, p_v in zip(y_labels, firsts, currs, projs):
        delta = c_v - f_v
        bar_end = max(c_v, p_v)
        if abs(delta) / max_x < NUDGE_THRESHOLD:
            # Narrow Δ — center across the full bar (start → projected)
            # so the label stays inside the bar instead of bleeding back
            # past x=0 into the y-axis label area.
            label_x = bar_end / 2.0
        else:
            label_x = f_v + delta / 2.0
        fig.add_annotation(
            x=label_x, y=y_lbl,
            xref="x", yref="y",
            text=_delta_fmt(delta),
            showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(size=px["in_chart_floor"],
                      color="#000000",
                      family=FONT_FAMILY),
        )

    # Headroom factor — tight enough that the x-axis stops just past
    # the right-side annotation rather than wasting half the panel.
    headroom = 1.30

    x_axis_kwargs = dict(
        range=[0, max_x * headroom],
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title=dict(text=axis_title,
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
    )
    if metric == "pct_tasks_affected":
        x_axis_kwargs["ticksuffix"] = "%"
    fig.update_xaxes(**x_axis_kwargs)

    # Disable automargin and set margin_l explicitly so the plot
    # position is deterministic. This lets the manual legend convert
    # container-center to paper coords precisely (shapes only accept
    # xref="paper"; centering on the full canvas requires knowing
    # where paper x=0 sits in the container).
    longest_chars = max(len(c) for c in y_labels)
    # Width budget: Inter char width @ tick pt ≈ 0.50 × pt-px (kaleido
    # renders Inter slightly narrower than 0.55), plus the rotated
    # y-axis title (font height) and a small gap. Tight margin removes
    # dead whitespace between the y-axis title and the canvas left edge.
    margin_l = max(
        60,
        int(longest_chars * 0.50 * px["tick"])  # tick labels
        + px["axis_title"]                       # rotated y title width
        + 25,                                    # gap + small padding
    )
    margin_r = 40
    margin_t = 100
    margin_b = 220   # extra so legend sits well below x-axis title
    chart_h  = 720

    fig.update_yaxes(
        title=dict(text="Major Occupational Category",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        automargin=False,
    )

    style_paper_figure(
        fig,
        chart_title,
        subtitle="",
        height=chart_h,
        width=PAPER_W,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
    )

    # Plotly's bar legend reproduces each bar's fill (including the
    # Projected Change hatch pattern) directly in the swatch, which
    # is what the bars actually look like. Container coords center
    # the legend horizontally under the whole canvas, not the plot
    # area; that's only possible because automargin is off and the
    # plot position is deterministic.
    fig.update_layout(barmode="stack", showlegend=False)

    # ── Manual legend ──────────────────────────────────────────────────
    # Big custom swatches centered under the whole canvas. The projected
    # swatch is striped via dense diagonal line shapes that visually
    # match the bar's plotly fillpattern.
    plot_w_px = PAPER_W - margin_l - margin_r
    plot_h_px = chart_h - margin_t - margin_b
    paper_x_at_canvas_center = (0.5 * PAPER_W - margin_l) / plot_w_px

    legend_specs = [
        (1.00, "Start",                     False),
        (0.55, "Observed Change",           False),
        (0.30, "Projected Change (2-year)", True),
    ]

    char_w_paper = (0.62 * px["legend"]) / plot_w_px
    # 22-px swatches to match the econ aggregate chart's Scatter
    # marker.size=22 swatches.
    SWATCH_W_PAPER = 22 / plot_w_px
    SWATCH_H_PAPER = 22 / plot_h_px
    SW_LABEL_GAP   = 10 / plot_w_px
    ITEM_GAP       = 60 / plot_w_px

    item_widths = [
        SWATCH_W_PAPER + SW_LABEL_GAP + len(lbl) * char_w_paper
        for _, lbl, _ in legend_specs
    ]
    total_w = sum(item_widths) + ITEM_GAP * (len(legend_specs) - 1)
    cursor = paper_x_at_canvas_center - total_w / 2.0

    LEGEND_Y = -0.36   # paper y — extra whitespace below the x-axis title

    # Pre-generate the hatched PNG once (same color for all charts'
    # projected swatches in this function — base_color varies by metric).
    hatch_uri = _hatch_png_data_uri(base_color, opacity=0.30)

    for (opacity, label, hatched), item_w in zip(legend_specs, item_widths):
        sw_x0 = cursor
        sw_x1 = sw_x0 + SWATCH_W_PAPER
        sw_y0 = LEGEND_Y - SWATCH_H_PAPER / 2
        sw_y1 = LEGEND_Y + SWATCH_H_PAPER / 2

        if hatched:
            # Layout image carries the actual hatch pattern. Plotly Shape
            # doesn't support fillpattern, so a PNG overlay is how we
            # get a swatch fill that visually matches the bar segments.
            fig.add_layout_image(dict(
                source=hatch_uri,
                xref="paper", yref="paper",
                x=sw_x0, y=sw_y1,
                sizex=SWATCH_W_PAPER, sizey=SWATCH_H_PAPER,
                xanchor="left", yanchor="top",
                sizing="stretch",
                layer="above",
            ))
            # Border on top so it reads like the other swatches.
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=sw_x0, x1=sw_x1, y0=sw_y0, y1=sw_y1,
                fillcolor="rgba(0,0,0,0)",
                line=dict(color=base_color, width=1),
                layer="above",
            )
        else:
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=sw_x0, x1=sw_x1, y0=sw_y0, y1=sw_y1,
                fillcolor=_hex_to_rgba(base_color, opacity),
                line=dict(color=base_color, width=1),
                layer="above",
            )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=sw_x1 + SW_LABEL_GAP, y=LEGEND_Y,
            text=label, showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=px["legend"],
                      color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        )

        cursor = sw_x0 + item_w + ITEM_GAP

    save_figure(fig, results / "figures" / out_name, scale=2)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_major_categories_trend(results: Path, figures: Path) -> None:
    """Two separate single-panel charts (tasks, workers). Each ranks its
    own top-10 majors by absolute change start → current on its own
    metric. Bars are three-segment stacks: solid start | mid-opacity
    observed Δ | hatched 2-yr projection.

    The workers chart additionally shows each row's value as a percent
    of national employment — start / current / projected — on a second
    annotation line.

    All font sizes resolved from the standardized pt ladder via
    paper_fonts(PAPER_W) so the charts print at canonical paper pt sizes.
    """
    trend = _major_trend_series()
    save_csv(trend, results / "major_trend_data.csv")

    # Per-metric raw stats
    tasks_stats   = _major_trend_stats(trend, "pct_tasks_affected")
    workers_stats = _major_trend_stats(trend, "workers_affected")

    summary_rows: list[dict] = []
    for metric, src in [("pct_tasks_affected", tasks_stats),
                        ("workers_affected",   workers_stats)]:
        for cat, s in src.items():
            summary_rows.append({
                "metric": metric, "category": cat,
                "first": s["first"], "current": s["current"],
                "jump_observed": s["current"] - s["first"],
                "projected_2yr": s["projected"],
                "delta_projected_2yr": s["projected"] - s["current"],
                "r2": s["r2"],
            })
    save_csv(pd.DataFrame(summary_rows),
             results / "major_trend_projections.csv")

    _render_major_trend_single(
        results, figures,
        metric="pct_tasks_affected",
        metric_key="tasks",
        axis_title="Tasks Exposed",
        chart_title=(
            "Major Occupational Category Tasks Exposed — "
            "Trend and 2-Year Linear Projection"
        ),
        out_name="major_categories_trend_tasks.png",
        stats=tasks_stats,
    )

    _render_major_trend_single(
        results, figures,
        metric="workers_affected",
        metric_key="workers",
        axis_title="Workers Exposed",
        chart_title=(
            "Major Occupational Category Workers Exposed — "
            "Trend and 2-Year Linear Projection"
        ),
        out_name="major_categories_trend_workers.png",
        stats=workers_stats,
    )


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def build_eloundou_divergence_major(results: Path, figures: Path) -> None:
    """Single-panel z-score divergence by Major Occupational Category.

    Per-occupation z-scores of our `all_confirmed` pct_tasks_affected and
    Eloundou et al. (2024) GPT-4 β are differenced, then averaged within
    each Major Occupational Category. Positive (blue) = we read more
    exposure than Eloundou; negative (orange) = Eloundou reads more.

    Mirrors the All Confirmed panel of `extcompare_eloundou_diff`'s
    `major_diverging_zscore` chart, formatted for the paper appendix.
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
    pct = get_pct_tasks_affected(PRIMARY_DATASET)

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
        results / "eloundou_divergence_major.csv",
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

    pad = (s.abs().max() or 1) * 0.22
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

    style_paper_figure(
        fig,
        title="Where We and Eloundou Disagree by Major Occupational Category",
        width=PAPER_W,
        height=fig_height,
        # Bottom margin holds x-tick row + axis title + ~50 px gap +
        # the one-line legend (visible breathing room from the axis title).
        margin=dict(l=margin_l, r=margin_r, t=90, b=210),
    )

    fig.update_xaxes(
        title=dict(
            text="z-score difference: All Confirmed − Eloundou GPT-4 β",
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

    out_name = "eloundou_divergence_major.png"
    save_figure(fig, results / "figures" / out_name)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_gwa_wkrs_wages(results: Path, figures: Path) -> None:
    """Appendix counterpart to part_2's `build_gwa_pct`. Three All Confirmed
    panels for all 41 O*NET GWAs: % Tasks Exposed | Workers Exposed |
    Wages Exposed. Same y-ordering and styling as the main-text gwa_pct
    chart so the reader can bridge between them."""
    base = _gwa_base_data()

    categories_r = [_wrap_gwa_label(c) for c in reversed(base["category"].tolist())]
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

    workers_max_v = float(base["workers_affected"].max()) if not base.empty else 0.0
    wages_max_v   = float(base["wages_affected"].max())   if not base.empty else 0.0
    # Higher threshold than the major chart (50% of max vs 30%) — the 41
    # GWA rows compress vertically so even small visual collisions
    # between inside-white text and adjacent bar regions stand out. Only
    # the clearly-tall bars get the inside-white treatment; everything
    # else reads as outside-dark text past the bar end.
    PCT_INSIDE_THRESHOLD = 50.0
    WKR_INSIDE_THRESHOLD = 0.50 * workers_max_v
    WAG_INSIDE_THRESHOLD = 0.50 * wages_max_v
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

    _style_gwa_split(
        fig,
        "AI Exposure by General Work Activity",
        n_cats=n_cats,
        panel_titles=set(),
        bottom_margin=130,  # 1-line axis titles only
    )

    # % axis uses the same uniform [0, 100] / [0, 50, 100] presentation
    # as the part_2 gwa_pct chart so the scale anchor reads consistently
    # figure-to-figure.
    r_wkr, t_wkr = _axis_max_and_ticks(workers_max_v)
    # Wages: 2 ticks only (0 and the largest "nice" round number below
    # the max). The 3-tick variant from _axis_max_and_ticks ($0/$200B/$400B)
    # crowds visually against the panel width here; a single inner tick
    # at the nearest $100B floor of the max gives a clear scale anchor
    # without packing labels together.
    import math
    wag_inner = math.floor(wages_max_v / 100e9) * 100e9
    r_wag = wages_max_v * 1.05
    t_wag = [0.0, float(wag_inner)]
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
            text="O*NET General Work Activity",
            font=dict(size=_MAJ_LABEL_FS),
            standoff=4,
        ),
        # Force every category label — plotly auto-thins categorical ticks
        # when the plot is short; tickmode="array" pins one label per bar.
        tickmode="array", tickvals=categories_r, ticktext=categories_r,
        row=1, col=1,
    )

    save_figure(fig, results / "figures" / "gwa_wkrs_wages.png", scale=2)
    _copy_fig(results, figures, "gwa_wkrs_wages.png")
    print("  -> gwa_wkrs_wages.png")


# ──────────────────────────────────────────────────────────────────────────
# state_clusters_each_ranked — companion to Part 3's state_clusters_map.
# Same cluster colors, but each panel sorts the 51 states by its own metric
# (left = % workforce exposed, right = % in High AI Exp & <0 Emp Proj occs).
# Lets the reader see how the two axes disagree on which states top out.
# ──────────────────────────────────────────────────────────────────────────

def build_state_clusters_each_ranked(results: Path, figures: Path) -> None:
    """Two-panel ranked bar chart with each panel sorted independently.

    Cluster colors and naming come from
    `deepdive_state_clusters.compute_clusters()` so this chart stays
    consistent with the main-body map in Part 3.
    """
    try:
        from lib.exploratory.state_clusters import (
            compute_clusters, OUTLIER_CLUSTER_ID, ALL_FEATURES,
            _load_state_features, CLUSTER_FEATURES, OUTLIER_GEOS,
            _pick_k_from_linkage, K_MIN, K_MAX,
        )
        from lib.exploratory.state_signal import (
            _load_focused_set,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/deepdive_state_clusters not available ({exc})")
        return

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster

    # Pin k=3 to match Part 3's state_clusters_map (and the version of
    # this chart that ran prior to the n_focused fix). Without pinning,
    # `_pick_k_from_linkage` can drift to k=4 when upstream feature values
    # shift (e.g. when the SKA-gated focused set changes from 38 → 44),
    # which would split one cluster into two and add a legend row.
    K_PIN = 3
    import lib.exploratory.state_clusters as _dsc_mod
    _orig_k_min, _orig_k_max = _dsc_mod.K_MIN, _dsc_mod.K_MAX
    _dsc_mod.K_MIN = _dsc_mod.K_MAX = K_PIN
    try:
        pkg = compute_clusters()
    finally:
        _dsc_mod.K_MIN, _dsc_mod.K_MAX = _orig_k_min, _orig_k_max
    state_df       = pkg["state_df"]
    cluster_names  = pkg["cluster_names"]
    cluster_color  = pkg["cluster_color"]
    order          = pkg["order"]

    # Recompute K-means alignment so we can mark disagreement states with
    # a diagonal stripe overlay in the K-means cluster's color.
    raw = _load_state_features()
    sub_in = raw[~raw["geo"].isin(OUTLIER_GEOS)].copy().reset_index(drop=True)
    Xz = StandardScaler().fit_transform(sub_in[CLUSTER_FEATURES].to_numpy(dtype=float))
    Z = linkage(Xz, method="ward")
    k_, _ = _pick_k_from_linkage(Z, K_PIN, K_PIN)
    ward_lab = fcluster(Z, t=k_, criterion="maxclust")
    km_lab = KMeans(n_clusters=k_, n_init=20, random_state=42).fit_predict(Xz) + 1
    aligned = np.zeros_like(km_lab)
    for w in np.unique(ward_lab):
        idx = ward_lab == w
        majority = pd.Series(km_lab[idx]).mode().iloc[0]
        aligned[km_lab == majority] = w
    for i, v in enumerate(aligned):
        if v == 0:
            aligned[i] = km_lab[i] + 100
    sub_in["ward_lab"] = ward_lab
    sub_in["km_lab"] = aligned
    km_alt_color: dict[str, str] = {
        r["geo"].upper(): cluster_color.get(int(r["km_lab"]), "#777777")
        for _, r in sub_in.iterrows()
        if r["ward_lab"] != r["km_lab"]
    }

    # SKA-gated focused set size — derived from the same builder Part 3's
    # risk_score_5f uses, so this label can't drift out of sync if the
    # underlying flags shift.
    n_focused = len(_load_focused_set())

    save_csv(
        state_df[["geo", "cluster", "cluster_name",
                  "pct_emp_wtd", "focused_share_pct"]],
        results / "state_clusters_each_ranked.csv",
        float_format="%.3f",
    )

    base = state_df.dropna(subset=list(ALL_FEATURES)).copy()
    n_states = len(base)

    left_sorted  = base.sort_values("pct_emp_wtd",       ascending=False).reset_index(drop=True)
    right_sorted = base.sort_values("focused_share_pct", ascending=False).reset_index(drop=True)

    def _rev(df: pd.DataFrame, value_col: str) -> tuple[list, list, list]:
        geos = list(reversed(df["geo"].str.upper().tolist()))
        vals = list(reversed(df[value_col].tolist()))
        cols = list(reversed([cluster_color[c] for c in df["cluster"].tolist()]))
        return geos, vals, cols

    geos_left,  exp_vals,     colors_left  = _rev(left_sorted,  "pct_emp_wtd")
    geos_right, focused_vals, colors_right = _rev(right_sorted, "focused_share_pct")

    # Plotly 6.6 doesn't honor per-bar `marker.pattern.fgcolor` arrays —
    # all stripes render in one color regardless of the list. To get
    # K-means alternative cluster colors as the stripe color, we split
    # the overlay into one trace per unique K-means color and use
    # barmode="overlay" so the stripes sit on top of the base bars.
    def _overlay_groups(geos: list[str], vals: list[float]) -> dict[str, dict]:
        groups: dict[str, dict] = {}
        for g, v in zip(geos, vals):
            if g not in km_alt_color:
                continue
            color = km_alt_color[g]
            groups.setdefault(color, {"y": [], "x": []})
            groups[color]["y"].append(g)
            groups[color]["x"].append(v)
        return groups

    # Per-bar border arrays: disagreement states get a thick border in
    # the K-means cluster color; all others get a zero-width (invisible)
    # border. The border is solid color (not a pattern), so it reads as
    # the exact K-means cluster hex rather than a visual blend.
    BORDER_W = 3
    border_color_left  = [km_alt_color.get(g, "rgba(0,0,0,0)") for g in geos_left]
    border_width_left  = [BORDER_W if g in km_alt_color else 0 for g in geos_left]
    border_color_right = [km_alt_color.get(g, "rgba(0,0,0,0)") for g in geos_right]
    border_width_right = [BORDER_W if g in km_alt_color else 0 for g in geos_right]

    # Both subtitles wrap onto two lines so the visual baselines align.
    # Without the left also breaking, the right (2-line) renders centered
    # on the same y as the left (1-line) and visually sits higher.
    panel_left  = "Sorted by<br>% of State Workforce Exposed"
    panel_right = (
        f"Sorted by % of State Emp in<br>"
        f"High AI Exp & <0 Emp Proj Occs (n={n_focused})"
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[panel_left, panel_right],
        horizontal_spacing=0.12,
        shared_yaxes=False,
    )

    # Base bars: every state, Ward color fill, K-means color border on
    # disagreement states (zero-width border elsewhere).
    fig.add_trace(go.Bar(
        y=geos_left, x=exp_vals, orientation="h",
        marker=dict(color=colors_left,
                    line=dict(color=border_color_left, width=border_width_left)),
        text=[f"{v:.1f}%" for v in exp_vals],
        textposition="outside",
        textfont=dict(size=ANNOT_FS, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Exposed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=geos_right, x=focused_vals, orientation="h",
        marker=dict(color=colors_right,
                    line=dict(color=border_color_right, width=border_width_right)),
        text=[f"{v:.1f}%" for v in focused_vals],
        textposition="outside",
        textfont=dict(size=ANNOT_FS, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>% of state emp in High AI Exp & <0 Emp Proj occs: %{x:.2f}%<extra></extra>",
    ), row=1, col=2)

    for cid in order:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=cluster_color[cid]),
            name=cluster_names[cid],
            showlegend=True,
        ), row=1, col=1)
    # Legend entry for the Ward / K-means disagreement marker — a
    # neutral-gray bar with a dark border, mirroring the per-bar border
    # treatment so the reader knows what the colored frames mean.
    fig.add_trace(go.Bar(
        y=[None], x=[None],
        marker=dict(
            color="#cccccc",
            line=dict(color="#333333", width=BORDER_W),
        ),
        name="Ward / K-means disagreement (border = K-means)",
        showlegend=True,
    ), row=1, col=1)

    height = max(PAPER_H + 250, n_states * 30 + 280)

    style_paper_figure(
        fig,
        "Workforce Exposure by State, Colored by Cluster",
        subtitle=(
            "Companion to the state-cluster map (Part 3). Each panel sorts the "
            "51 states by its own metric so the two rankings can be compared "
            "directly. Cluster colors are identical to the map; a state's color "
            "is the same in both panels. Note how the knowledge-economy states "
            "(mid blue) top the left panel but sit near the bottom of the right; "
            "the high-vulnerability cluster (dark blue) dominates the top of the "
            "right panel and spreads across the middle of the left."
        ),
        height=height,
        width=PAPER_W,
        # t at 170 keeps a comfortable gap between the (top-pinned) figure
        # title and the 2-line panel subtitles without yawning empty space
        # in between. b at 290 gives the x-axis titles room above the legend.
        margin=dict(l=40, r=80, t=170, b=290),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        ticksuffix="%",
    )
    fig.update_xaxes(
        title=dict(text="% State Workforce Exposed",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text="% State Emp in High AI Exp & <0 Emp Proj Occs",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=2,
    )

    # dtick=1 forces every state label to render on a categorical axis.
    fig.update_yaxes(
        showgrid=False, showline=False,
        tickmode="linear", dtick=1,
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        title=dict(text="State", font=dict(size=LABEL_FS - 2)),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="State", font=dict(size=LABEL_FS - 2)),
        row=1, col=2,
    )

    panel_set = {panel_left, panel_right}
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_set:
            ann.font = dict(size=LABEL_FS - 2, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    fig.update_layout(
        bargap=0.28,
        # `barmode="overlay"` puts the K-means stripe overlay traces on
        # top of the base bars at the same y position; without it they'd
        # render side-by-side and the stripes would land at different bar
        # positions than the colored fill.
        barmode="overlay",
        # Pin the figure title to the top of the canvas. style_paper_figure
        # leaves title.y unset, so Plotly centers it inside the t margin —
        # with t=210 that leaves a big gap above the title. Anchoring it
        # near y=1.0 collapses that gap while preserving the subtitle space.
        title=dict(y=0.985, yanchor="top"),
        legend=dict(
            orientation="h",
            # Pushed from -0.06 → -0.11 to add room between the x-axis
            # titles and the legend chips.
            yanchor="top", y=-0.11,
            xanchor="center", x=0.5,
            font=dict(size=TICK_FS - 1, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    save_figure(fig, results / "figures" / "state_clusters_each_ranked.png", scale=3)
    shutil.copy(
        results / "figures" / "state_clusters_each_ranked.png",
        figures / "state_clusters_each_ranked.png",
    )
    print("  -> state_clusters_each_ranked.png")


# ──────────────────────────────────────────────────────────────────────────
# state_clusters_combined_ranked — companion to state_clusters_each_ranked.
# Sums each state's rank from the two panels of the prior chart and sorts
# ascending. Single bar per state colored by cluster; end-of-bar label
# carries the sum plus the two component ranks in parentheses.
# ──────────────────────────────────────────────────────────────────────────

def build_state_clusters_combined_ranked(results: Path, figures: Path) -> None:
    """Single-panel bar chart of the summed rank from each panel of
    state_clusters_each_ranked.

    For every state, the rank on % workforce exposed (rank 1 = highest)
    is summed with the rank on % emp in High AI Exp & <0 Emp Proj occs
    (rank 1 = highest). Lower combined rank means a state ranks high
    on both. Bar color = state's Ward cluster. End-of-bar label =
    combined rank sum, with the two component ranks in parentheses
    (workforce rank, focused-set rank).
    """
    try:
        from lib.exploratory.state_clusters import (
            compute_clusters, ALL_FEATURES,
            _load_state_features, CLUSTER_FEATURES, OUTLIER_GEOS,
            _pick_k_from_linkage,
        )
        from lib.exploratory.state_signal import (
            _load_focused_set,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/deepdive_state_clusters not available ({exc})")
        return

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster

    # Pin k=3 to match Part 3's state_clusters_map and the companion
    # each_ranked chart above. See that function for the rationale.
    K_PIN = 3
    import lib.exploratory.state_clusters as _dsc_mod
    _orig_k_min, _orig_k_max = _dsc_mod.K_MIN, _dsc_mod.K_MAX
    _dsc_mod.K_MIN = _dsc_mod.K_MAX = K_PIN
    try:
        pkg = compute_clusters()
    finally:
        _dsc_mod.K_MIN, _dsc_mod.K_MAX = _orig_k_min, _orig_k_max
    state_df       = pkg["state_df"]
    cluster_names  = pkg["cluster_names"]
    cluster_color  = pkg["cluster_color"]
    order          = pkg["order"]

    # K-means / Ward disagreement alignment, lifted from each_ranked so
    # the two charts mark the same set of striped states.
    raw = _load_state_features()
    sub_in = raw[~raw["geo"].isin(OUTLIER_GEOS)].copy().reset_index(drop=True)
    Xz = StandardScaler().fit_transform(
        sub_in[CLUSTER_FEATURES].to_numpy(dtype=float)
    )
    Z = linkage(Xz, method="ward")
    k_, _ = _pick_k_from_linkage(Z, K_PIN, K_PIN)
    ward_lab = fcluster(Z, t=k_, criterion="maxclust")
    km_lab = KMeans(n_clusters=k_, n_init=20, random_state=42).fit_predict(Xz) + 1
    aligned = np.zeros_like(km_lab)
    for w in np.unique(ward_lab):
        idx = ward_lab == w
        majority = pd.Series(km_lab[idx]).mode().iloc[0]
        aligned[km_lab == majority] = w
    for i, v in enumerate(aligned):
        if v == 0:
            aligned[i] = km_lab[i] + 100
    sub_in["ward_lab"] = ward_lab
    sub_in["km_lab"] = aligned
    km_alt_color: dict[str, str] = {
        r["geo"].upper(): cluster_color.get(int(r["km_lab"]), "#777777")
        for _, r in sub_in.iterrows()
        if r["ward_lab"] != r["km_lab"]
    }

    n_focused = len(_load_focused_set())

    base = state_df.dropna(subset=list(ALL_FEATURES)).copy()
    n_states = len(base)

    # Descending rank: 1 = highest exposure. method="min" matches the
    # ranks reported alongside the each_ranked CSV.
    base["rank_workforce"] = base["pct_emp_wtd"].rank(
        ascending=False, method="min"
    ).astype(int)
    base["rank_focused"] = base["focused_share_pct"].rank(
        ascending=False, method="min"
    ).astype(int)
    base["rank_sum"] = base["rank_workforce"] + base["rank_focused"]

    sorted_df = base.sort_values(
        ["rank_sum", "geo"], ascending=[True, True]
    ).reset_index(drop=True)

    # Plotly horizontal bars render bottom-to-top, so reverse for
    # top-to-bottom display (lowest rank_sum = most exposed = top).
    geos    = list(reversed(sorted_df["geo"].str.upper().tolist()))
    rank_wf = list(reversed(sorted_df["rank_workforce"].tolist()))
    rank_fc = list(reversed(sorted_df["rank_focused"].tolist()))
    rank_sm = list(reversed(sorted_df["rank_sum"].tolist()))
    pct_wf  = list(reversed(sorted_df["pct_emp_wtd"].tolist()))
    pct_fc  = list(reversed(sorted_df["focused_share_pct"].tolist()))
    colors = list(reversed(
        [cluster_color[c] for c in sorted_df["cluster"].tolist()]
    ))

    save_csv(
        sorted_df.assign(geo=sorted_df["geo"].str.upper())[
            ["geo", "cluster", "cluster_name",
             "pct_emp_wtd", "rank_workforce",
             "focused_share_pct", "rank_focused",
             "rank_sum"]
        ],
        results / "state_clusters_combined_ranked.csv",
        float_format="%.3f",
    )

    # Per-bar border: K-means disagreement states get a thick colored
    # outline in the K-means alt cluster's color. Agreement states get
    # no border. Unlike marker.pattern.fgcolor (which Plotly 6.6
    # silently ignores when given an array), marker.line.color IS
    # honored per-bar.
    BORDER_W_DISAGREE = 3
    line_colors = [
        km_alt_color.get(g, "rgba(0,0,0,0)") for g in geos
    ]
    line_widths = [
        BORDER_W_DISAGREE if g in km_alt_color else 0 for g in geos
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=geos, x=rank_sm, orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=line_colors, width=line_widths),
        ),
        showlegend=False, cliponaxis=False,
        customdata=list(zip(rank_wf, rank_fc, pct_wf, pct_fc)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Combined rank sum: %{x}<br>"
            "Rank on %% Workforce Exposed: %{customdata[0]} "
            "(%{customdata[2]:.1f}%%)<br>"
            "Rank on %% in High AI Exp & <0 Emp Proj Occs: "
            "%{customdata[1]} (%{customdata[3]:.2f}%%)<extra></extra>"
        ),
    ))

    # Cluster legend swatches + disagreement-border legend swatch.
    for cid in order:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=cluster_color[cid]),
            name=cluster_names[cid],
            showlegend=True,
        ))
    fig.add_trace(go.Bar(
        y=[None], x=[None],
        marker=dict(
            color="#e6e6e6",
            line=dict(color="#333333", width=BORDER_W_DISAGREE),
        ),
        name="Ward / K-means disagreement (border = K-means)",
        showlegend=True,
    ))

    height = max(PAPER_H + 250, n_states * 30 + 280)

    style_paper_figure(
        fig,
        ("Combined Exposure Rank — Sum of Both State-Ranking Panels "
         f"(focused set n={n_focused})"),
        subtitle=(
            "Each state's rank on the two panels of the previous figure summed. "
            "Lower combined rank means a state ranks high on BOTH "
            "% workforce exposed and % employment in High AI Exposure & <0 "
            "Emp Proj occupations. States sorted ascending — most exposed on "
            "both metrics at top. Bar color = Ward cluster (matches previous "
            "figure); colored border = K-means disagreement (border color is "
            "the K-means alternative cluster). End-of-bar label: combined "
            "rank sum (bold), followed by (workforce rank, focused-set rank) "
            "in parentheses."
        ),
        height=height,
        width=PAPER_W,
        # Tighter top margin (was 170) but still wide enough to stack
        # title above the label-format key without them overlapping.
        margin=dict(l=40, r=160, t=130, b=200),
    )

    # End-of-bar label needs room for the longest case e.g. "97 (49, 48)".
    x_upper = max(rank_sm) + 24
    fig.update_xaxes(
        title=dict(
            text=("Combined Rank Sum (lower = more exposed on both metrics; "
                  f"min possible = 2, max possible = {2 * n_states})"),
            font=dict(size=LABEL_FS - 4),
        ),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        range=[0, x_upper],
    )
    fig.update_yaxes(
        title=dict(text="State", font=dict(size=LABEL_FS - 2)),
        showgrid=False, showline=False,
        tickmode="linear", dtick=1,
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
    )

    # Sum (bold) + component ranks in parentheses at the end of each bar.
    for g, s, wf, fc in zip(geos, rank_sm, rank_wf, rank_fc):
        fig.add_annotation(
            x=s + 0.9, y=g,
            text=f"<b>{s}</b> ({wf}, {fc})",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=ANNOT_FS, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        )

    # Label-format key sitting between the title and the plot area in
    # the top margin, so the reader knows which number in the parens is
    # which without consulting the caption.
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.0, y=1.022,
        xanchor="left", yanchor="bottom",
        text=("<b>Label format:</b> "
              "<b>combined rank sum</b> "
              "(<i>workforce-exposure rank</i>, "
              "<i>focused-set rank</i>) — e.g. <b>12</b> (11, 1)"),
        showarrow=False,
        font=dict(size=ANNOT_FS, color=PAPER_PALETTE["text"],
                  family=FONT_FAMILY),
    )

    fig.update_layout(
        bargap=0.28,
        # Title anchored at the top of the canvas so the label-format
        # key annotation has room beneath it before the plot area.
        title=dict(y=0.985, yanchor="top"),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.07,
            xanchor="center", x=0.5,
            font=dict(size=TICK_FS - 1, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    save_figure(fig, results / "figures" / "state_clusters_combined_ranked.png", scale=3)
    shutil.copy(
        results / "figures" / "state_clusters_combined_ranked.png",
        figures / "state_clusters_combined_ranked.png",
    )
    print("  -> state_clusters_combined_ranked.png")


# ──────────────────────────────────────────────────────────────────────────
# underadoption_gap — per-major ratio of % tasks exposed to share of AI
# usage (ratio_full_pct from the eco-baseline-normalized intensity model).
# High = exposure outpacing where AI is actually being used. Same visual
# language as Part 3's intensity_anchor_fulleco for an easy paired read.
# ──────────────────────────────────────────────────────────────────────────

def build_underadoption_gap(results: Path, figures: Path) -> None:
    """Underadoption relative to potential, normalized on the median major.

    Per major:
      raw_gap   = pct_tasks_affected / ratio_full_pct
      gap_ratio = raw_gap / median(raw_gap)

    Numerator (pct_tasks_affected): % of an occupation's task completions
    AI can affect — the 'potential' informed by task exposure (Part 2
    major_categories metric). Denominator (ratio_full_pct): share of
    total AI usage that maps to that major, normalized over the full eco
    employment×freq baseline (same backbone as Part 3
    intensity_anchor_fulleco).

    Normalizing on the median raw_gap across the 22 majors makes lift read
    as 'X times more underadopted than the median major'. The median dashed
    line sits at x=1 by construction.
    """
    try:
        from lib.exploratory.intensity import (
            BIAS_VARIANTS, compute_bias_ratios,
        )
        from lib.exploratory.intensity_v3 import (
            compute_v3_intensity, compute_major_full_eco_denominator,
        )
        from lib.builders.part3 import _run_config
    except ImportError as exc:
        print(f"  -> SKIPPED: dependency not available ({exc})")
        return

    # ── Compute ratio_full_pct (share of all AI usage, full-eco baseline)
    # Numerator uses the AEI-only eco_2025-rebased pool — same as the rest
    # of the paper's intensity figures. Equal 3-source debias still applies.
    base = compute_v3_intensity(
        _INTENSITY_V3_KEY, compute_bias_ratios(BIAS_VARIANTS["equal"])
    ).copy()
    full_den = compute_major_full_eco_denominator()
    base["den_full"] = base["category"].map(full_den).fillna(0.0)
    base["ratio_full"] = np.where(
        base["den_full"] > 0, base["num"] / base["den_full"], 0.0
    )
    total_full = base["ratio_full"].sum()
    base["ratio_full_pct"] = (
        base["ratio_full"] / total_full * 100.0 if total_full > 0 else 0.0
    )

    # ── pct_tasks_affected from the same AEI-only file so both sides of
    # the gap ratio are coherent (numerator and denominator share a data
    # source). Diverges from the Part 2 major_categories chart's colorbar
    # by design — the intensity-series pair stays end-to-end on no-MS data.
    major_df = _run_config(_INTENSITY_DATASET, "major")
    pct_aff = major_df.set_index("category")["pct_tasks_affected"]
    base["pct_tasks_affected"] = base["category"].map(pct_aff).fillna(0.0)

    # ── Raw gap, then normalized to the median major so lift reads as a
    # clean multiple ("× more underadopted than the median major"). pandas
    # .median() skips the NaN rows (majors with no AI usage mass).
    base["raw_gap"] = np.where(
        base["ratio_full_pct"] > 0,
        base["pct_tasks_affected"] / base["ratio_full_pct"],
        np.nan,
    )
    median_gap = float(base["raw_gap"].median())
    assert median_gap > 0, "Median raw_gap must be > 0"
    base["gap_ratio"] = base["raw_gap"] / median_gap

    out = base[
        ["category", "pct_tasks_affected", "ratio_full_pct",
         "raw_gap", "gap_ratio"]
    ].sort_values("gap_ratio", ascending=False)
    out["median_value"] = median_gap
    save_csv(out, results / "underadoption_gap.csv", float_format="%.4f")

    plot_df = base.sort_values("gap_ratio", ascending=True).reset_index(drop=True)
    plot_df["display_category"] = (
        plot_df["category"].str.replace(r"\s*Occupations\s*$", "", regex=True)
    )
    cvals = plot_df["pct_tasks_affected"].to_numpy(dtype=float)
    cmin, cmax = float(cvals.min()), float(cvals.max())

    W = PAPER_W
    px = paper_fonts(W)

    TASKS_LIGHT = "#cfe0ec"
    TASKS_DARK = "#2c4f6b"

    # Per-bar inside/outside text — wider bars get the value label inside
    # in white (matches intensity_anchor_fulleco / part_2 major_categories);
    # narrower bars stay outside in dark.
    x_top = float(plot_df["gap_ratio"].max()) * 1.06
    INSIDE_THRESHOLD = x_top * 0.12
    pos = [
        "inside" if v >= INSIDE_THRESHOLD else "outside"
        for v in plot_df["gap_ratio"]
    ]
    inside_font  = dict(size=px["tick"], color="white",                family=FONT_FAMILY)
    outside_font = dict(size=px["tick"], color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["display_category"], x=plot_df["gap_ratio"], orientation="h",
        marker=dict(
            color=cvals,
            colorscale=[[0, TASKS_LIGHT], [1, TASKS_DARK]],
            cmin=cmin, cmax=cmax,
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{v:.2f}x" for v in plot_df["gap_ratio"]],
        textposition=pos,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        constraintext="none",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>underadoption: %{x:.2f}x<br>"
            "% tasks exposed: %{marker.color:.1f}%<extra></extra>"
        ),
        showlegend=False,
    ))

    # Median reference line — the median major sits at x = 1 by construction.
    fig.add_vline(
        x=1.0, line_dash="dash",
        line_color=PAPER_PALETTE["negative"], line_width=1.5,
    )
    fig.add_annotation(
        x=1.0, y=1.005,
        xref="x", yref="paper",
        text="median",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["negative"], family=FONT_FAMILY),
    )

    # ── Bottom legend — same HTML-swatch gradient as Part 3 intensity chart.
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    rgb_l = _hex_to_rgb(TASKS_LIGHT)
    rgb_d = _hex_to_rgb(TASKS_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"Tasks Exposed&nbsp;&nbsp;{cmin:.0f}%&nbsp;"
        f"{swatch_html}&nbsp;{cmax:.0f}%"
    )
    # Compact layout matching the main intensity_anchor_fulleco chart:
    # 22 majors × 38 px/row + 90/170 margins ≈ 1096 px (~5.1").
    n = len(plot_df)
    MARGIN_T, MARGIN_B = 90, 170
    chart_h = n * 38 + MARGIN_T + MARGIN_B
    plot_h_px = chart_h - MARGIN_T - MARGIN_B
    legend_y = -(MARGIN_B - 50) / plot_h_px
    fig.add_annotation(
        x=0.14, y=legend_y,
        xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    style_paper_figure(
        fig,
        "Underadoption Relative to Potential as Informed by Task Exposure",
        height=chart_h, width=W,
        margin=dict(l=20, r=80, t=MARGIN_T, b=MARGIN_B),
    )
    fig.update_layout(bargap=0.15)
    fig.update_xaxes(
        title=dict(
            text="Underadoption Relative to Median (×)",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
        ),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        range=[0, x_top],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )
    # tickmode="array" pins every category label at this tight row pitch.
    y_labels = list(plot_df["display_category"])
    fig.update_yaxes(
        title=dict(
            text="Major Occupational Category",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
        ),
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        tickmode="array", tickvals=y_labels, ticktext=y_labels,
    )

    save_figure(fig, results / "figures" / "underadoption_gap.png", scale=2)
    _copy_fig(results, figures, "underadoption_gap.png")
    print("  -> underadoption_gap.png")


# ─────────────────────────────────────────────────────────────────────────
# Within-major intensity drivers — decomposes the three high-lift bars of
# Part 3's intensity_anchor_fulleco chart (Life/Phys/Sci, Arts, Comp/Math)
# into the top-10 occupations and top-10 tasks driving each major's lift.
#
# Per occ (or per task) ratio = Σ debiased adj_pct / Σ (freq × emp), then
# normalized by the within-major median ratio so the dashed median line
# sits at x=1 and lifts read directly as "× the major's median row."
# ─────────────────────────────────────────────────────────────────────────

TARGET_MAJORS_DRIVERS: list[tuple[str, str, str]] = [
    ("Life, Physical, and Social Science Occupations",
     "life_phys_soc_sci",
     "Life, Physical & Social Science"),
    ("Arts, Design, Entertainment, Sports, and Media Occupations",
     "arts_design_ent",
     "Arts, Design & Entertainment"),
    ("Computer and Mathematical Occupations",
     "comp_math",
     "Computer and Mathematical"),
]


def _intensity_drivers_dedup() -> pd.DataFrame:
    """Deduped (task, occ) table for all_confirmed with debiased adj_pct,
    raw pct_normalized, eco_weight (freq×emp), auto_aug_mean, and the
    original (punctuated, capitalized) `task` statement keyed on
    `task_normalized` for human-readable y-tick labels.

    Replicates the dedup + bias logic that
    deepdive_intensity_drivers._build_task_occ_table uses, so this builder
    stays self-contained even if that exploratory folder is absent.
    Imports from audit_pct_norm_eco (already a paper dependency for
    intensity_anchor_fulleco and underadoption_gap).
    """
    from lib.exploratory.intensity import (
        BIAS_VARIANTS, EMP_COL, compute_bias_ratios,
    )
    from lib.exploratory.intensity_v3 import (
        DATA_DIR, V3_CONFIGS, load_v3_config,
    )

    bias_ratios = compute_bias_ratios(BIAS_VARIANTS["equal"])
    assert bias_ratios is not None, "equal bias must be present"
    # Intensity figures use the AEI-only, eco_2025-rebased pool — not
    # V3_CONFIGS["all_confirmed"] (which still points at AEI + Microsoft).
    cfg = V3_CONFIGS[_INTENSITY_V3_KEY]
    occ_col = cfg["occ_col"]  # "title_current"

    df = load_v3_config(_INTENSITY_V3_KEY)
    # Separate read for the original `task` text (load_v3_config has a
    # fixed usecols that excludes it). Dedup to one statement per
    # task_normalized — duplicates across the file are punctuation /
    # capitalization variants of the same task.
    task_text = pd.read_csv(
        DATA_DIR / cfg["file"], usecols=["task_normalized", "task"]
    ).dropna(subset=["task_normalized", "task"])
    task_text = (
        task_text.drop_duplicates(subset=["task_normalized"])
        .set_index("task_normalized")["task"]
    )
    ai = df[df["pct_normalized"].notna() & df["major_occ_category"].notna()].copy()
    ai["eco_weight"] = ai["freq_mean"].fillna(0.0) * ai[EMP_COL].fillna(0.0)

    gwa_pairs = (
        ai.dropna(subset=["gwa_title"])
        .drop_duplicates(["task_normalized", occ_col, "gwa_title"])[
            ["task_normalized", occ_col, "gwa_title"]
        ].copy()
    )
    gwa_pairs["bias"] = gwa_pairs["gwa_title"].map(bias_ratios).fillna(1.0)
    avg_bias = (
        gwa_pairs.groupby(["task_normalized", occ_col])["bias"].mean()
        .reset_index(name="avg_bias")
    )

    keep_cols = [
        "task_normalized", occ_col, "major_occ_category",
        "pct_normalized", "auto_aug_mean", "eco_weight",
    ]
    dedup = ai.drop_duplicates(["task_normalized", occ_col])[keep_cols].copy()
    dedup = dedup.merge(avg_bias, on=["task_normalized", occ_col], how="left")
    dedup["avg_bias"] = dedup["avg_bias"].fillna(1.0).replace(0.0, 1.0)
    dedup["adj_pct"] = dedup["pct_normalized"] / dedup["avg_bias"]
    # Attach the original task statement. Falls back to the normalized
    # form for any task_normalized that doesn't have a paired `task` row.
    dedup["task_display"] = (
        dedup["task_normalized"].map(task_text).fillna(dedup["task_normalized"])
    )
    return dedup


_ORPHAN_STARTS = {"and", "or", "but", "nor", "yet", "&", "the", "a", "an", "of",
                  "to", "in", "on", "for", "with", "by", "at", "from"}


def _balance_two_lines(words: list[str], width: int) -> str | None:
    """Find the word-boundary split that minimizes the longer of two
    lines, with both lines ≤ width. Splits that would start line 2 with
    an orphan word (conjunction / preposition / article) get a small
    char-budget penalty so balanced splits without orphans are preferred
    even when slightly less even. On a true tie, the later split wins
    (more text on line 1). Returns "<br>"-joined string, or None if no
    valid 2-line split exists."""
    n = len(words)
    if n == 0:
        return ""
    if n == 1:
        return words[0] if len(words[0]) <= width else None
    best_split = None
    best_score: tuple[int, int] | None = None
    for split in range(1, n):
        line1 = " ".join(words[:split])
        line2 = " ".join(words[split:])
        if len(line1) > width or len(line2) > width:
            continue
        this_max = max(len(line1), len(line2))
        starts_orphan = (
            words[split].lower().rstrip(",.;:!?") in _ORPHAN_STARTS
        )
        # Penalty adds 5 chars to the comparison max — small enough that
        # a strongly-balanced orphan split still wins over a heavily
        # lopsided non-orphan split, large enough to tip ties.
        adjusted_max = this_max + (5 if starts_orphan else 0)
        score = (adjusted_max, -split)  # later split wins on tie
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    if best_split is None:
        return None
    return f"{' '.join(words[:best_split])}<br>{' '.join(words[best_split:])}"


def _wrap_driver_label(s: str, width: int, max_lines: int = 2) -> str:
    """Balanced word-wrap for y-tick labels. When text wraps to two
    lines, the split is chosen to minimize the longer line so the two
    lines are roughly the same length. Past max_lines, the trailing
    word gets an ellipsis. max_lines=2 keeps every row at uniform
    height in plotly bar charts.
    """
    import textwrap
    words = str(s).split()
    if not words:
        return ""
    full = " ".join(words)
    if len(full) <= width:
        return full

    if max_lines == 2:
        # Try the full text; if it doesn't fit two balanced lines, drop
        # trailing words one at a time and try again. The first success
        # is the longest text we can show with balanced 2-line wrap.
        truncated = False
        trial = list(words)
        while trial:
            result = _balance_two_lines(trial, width)
            if result is not None:
                if truncated:
                    # Apply ellipsis to the last visible word on line 2.
                    line1, _, line2 = result.partition("<br>")
                    line2 = line2.rstrip(",.;:") + "…"
                    return f"{line1}<br>{line2}"
                return result
            trial = trial[:-1]
            truncated = True
        return ""

    # Fallback for max_lines != 2 (not used in current charts).
    lines = textwrap.wrap(full, width=width, break_long_words=False,
                          break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        if len(last) > width - 1:
            last = last[: width - 1].rstrip()
        lines[-1] = last.rstrip(",.;:") + "…"
    return "<br>".join(lines)


def _render_intensity_driver_chart(
    plot_df: pd.DataFrame,
    results: Path,
    figures: Path,
    *,
    level: str,            # "occ" or "task"
    slug: str,
    major_short: str,
    label_col: str,
    color_col: str,
    color_label: str,
    color_fmt: str,
    label_wrap: int,
    margin_left: int | None = None,
) -> None:
    """Render one within-major intensity-driver chart, mirroring the
    main-body intensity_anchor_fulleco style: horizontal bars, dashed
    median reference at x=1, TASKS_LIGHT→TASKS_DARK color, HTML-swatch
    bottom legend, raw-pct + lift label outside each bar.

    `plot_df` must already be sorted ascending by lift so Plotly draws
    the largest bar at the top.
    """
    out_name = f"intensity_drivers_{level}_{slug}.png"

    display_labels = plot_df[label_col].astype(str).map(
        lambda s: _wrap_driver_label(s, label_wrap)
    )
    # Auto-fit margin_left to the actual longest rendered line so the
    # y-axis title sits flush against the PNG left edge regardless of
    # what occupations/tasks happen to be in a given major. Uses a
    # per-character width table calibrated against 9pt sans-serif
    # rendered at PAPER_W=1400 — narrow chars ('i','l','t') are ~5–7px,
    # wide chars ('M','W','m','w') are ~17–19px, defaults ~12.
    if margin_left is None:
        _CHAR_W = {
            'i': 6, 'l': 6, 'j': 7, 't': 9, 'f': 9, 'r': 9,
            '.': 6, ',': 6, ';': 6, ':': 6, "'": 5, ' ': 6,
            '(': 8, ')': 8, '-': 9, '/': 8, '!': 6, '|': 6,
            '0': 14, '1': 9, '2': 14, '3': 14, '4': 14, '5': 14,
            '6': 14, '7': 13, '8': 14, '9': 14,
            'I': 8, 'J': 10,
            'M': 20, 'W': 21, 'm': 19, 'w': 18,
        }
        DEFAULT_W = 15  # most upper/lowercase letters at 9pt sans-serif
        def _est(line: str) -> float:
            return sum(_CHAR_W.get(c, DEFAULT_W) for c in line)
        max_line_px = max(
            (max(_est(line) for line in lbl.split("<br>"))
             for lbl in display_labels),
            default=120.0,
        )
        # +70 covers the vertical y-axis title (~30px wide at 10pt), the
        # standoff between title and labels (~18px), and the axis-to-
        # label gap (~12px), plus a ~10px safety buffer so the title
        # doesn't overlap the longest label (e.g. the life_phys_soc_sci
        # chart's "Anthropologists and Archeologists" pushed right up
        # against "Occupations" at the prior +50 setting).
        margin_left = int(max_line_px + 70)
    cvals = plot_df[color_col].to_numpy(dtype=float)
    cmin, cmax = float(np.nanmin(cvals)), float(np.nanmax(cvals))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
        cmin, cmax = (cmin if np.isfinite(cmin) else 0.0,
                      (cmin if np.isfinite(cmin) else 0.0) + 1.0)

    W = PAPER_W
    px = paper_fonts(W)

    TASKS_LIGHT = "#cfe0ec"
    TASKS_DARK = "#2c4f6b"

    text_labels = [
        f"{lift:.2f}×   ({raw:.3f}% raw pct)"
        for lift, raw in zip(plot_df["lift"], plot_df["raw_pct"])
    ]

    level_word = "Occupations" if level == "occ" else "Tasks"
    # One line — shortened from the main-body "Actual Equalized AI Usage
    # as a Multiple of Median Usage" so the major name fits on the same
    # line at 11pt across PAPER_W=1400 (cap ≈ 85 chars). "Occupations"
    # is 6 chars longer than "Tasks" so the prefix has to be terse.
    title = (
        f"AI Usage as Multiple of Median — Top {level_word} in {major_short}"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=display_labels, x=plot_df["lift"], orientation="h",
        marker=dict(
            color=cvals,
            colorscale=[[0, TASKS_LIGHT], [1, TASKS_DARK]],
            cmin=cmin, cmax=cmax,
            showscale=False,
            line=dict(width=0),
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=px["tick"], color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        cliponaxis=False,
        showlegend=False,
    ))

    fig.add_vline(
        x=1.0, line_dash="dash",
        line_color=PAPER_PALETTE["negative"], line_width=1.5,
    )
    # Median label sits just above the plot, right of the dashed line,
    # on both occ and task charts (1-line title leaves enough top space).
    fig.add_annotation(
        x=1.0, y=1.0, xref="x", yref="paper",
        text="median", showarrow=False,
        xanchor="left", yanchor="bottom",
        xshift=2, yshift=6,
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["negative"], family=FONT_FAMILY),
    )

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    rgb_l = _hex_to_rgb(TASKS_LIGHT)
    rgb_d = _hex_to_rgb(TASKS_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"{color_label}&nbsp;&nbsp;{color_fmt.format(cmin)}&nbsp;"
        f"{swatch_html}&nbsp;{color_fmt.format(cmax)}"
    )
    # Compact layout: per-row pitch tightened so 10-row charts land
    # ~3.7–4.3" tall. Task charts get more room because labels are
    # 2-line wrapped (`_wrap_driver_label(width=52, max_lines=2)`); occ
    # charts also have 2-line wraps at width=36 (e.g. "Fine Artists,
    # Including Painters, Sculptors, and Illustrators"), so occ pitch
    # bumped from 50→58 to give 2-line occ labels room without crowding
    # adjacent rows. margin_b unified to 170 across all intensity charts
    # (intensity_anchor_fulleco + underadoption_gap + drivers) so the
    # bottom gradient legend lands the same 120 px below the plot bottom
    # and 50 px above the canvas bottom on every one.
    n_rows = len(plot_df)
    row_pitch = 60 if level == "task" else 58
    margin_top = 90
    margin_bottom = 170
    margin_right = 90 if level == "task" else 110
    height = n_rows * row_pitch + margin_top + margin_bottom
    style_paper_figure(
        fig, title,
        height=height, width=W,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
    )
    fig.update_layout(bargap=0.15)

    # Legend centered on the PNG itself, not on the plot area. xref="paper"
    # is plot-area-relative — to land on PNG midpoint W/2 in pixel terms,
    # solve: PNG_center = margin_left + x_paper × plot_width. With
    # yaxis automargin=False below, margin_left is exactly what we pass
    # here, so the formula is exact.
    plot_w_px = float(W - margin_left - margin_right)
    legend_x = (float(W) / 2.0 - float(margin_left)) / plot_w_px
    plot_h_px = float(height - margin_top - margin_bottom)
    legend_y = -(margin_bottom - 50) / plot_h_px  # matches underadoption_gap spacing
    fig.add_annotation(
        x=legend_x, y=legend_y, xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    # x_top sized so the longest bar's outside text ("X.XX×   (Y.YYY raw
    # pct)") extends into the trimmed margin_right and lands near, but
    # inside, the PNG right edge — minimises right-side whitespace. Task
    # charts have a wider margin_left (for long wrapped task labels), so
    # the plot is narrower → needs a larger x_top multiplier to keep the
    # max-lift bar from pushing its outside text past x=W.
    # Multipliers bumped (task 1.78→1.92, occ 1.38→1.48) so the wider
    # "X.XX×   (Y.YYY% raw pct)" outside labels — one char wider since the
    # raw pct gained a "%" — clear the PNG right edge without clipping.
    x_mult = 1.92 if level == "task" else 1.48
    x_top = float(plot_df["lift"].max()) * x_mult
    # Task chart lift values reach 7,000×+ so plotly's auto-tick spacing
    # crams 5–6 ticks into a narrow plot area. Constrain to ~4 ticks and
    # use the SI suffix format (1k, 2k, …) for compact readability.
    use_si = level == "task" and float(plot_df["lift"].max()) >= 1000.0
    fig.update_xaxes(
        title=dict(text="Usage Relative to Median (×)",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY),
                   standoff=18),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        range=[0, x_top],
        tickangle=0,
        nticks=4 if level == "task" else 6,
        tickformat="~s" if use_si else ",.0f",
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )
    # automargin=False pins margin_left to exactly what we pass to
    # style_paper_figure() so the PNG-centered legend math above is
    # exact. Margins below at the call sites are sized to fit the
    # longest wrapped label without plotly needing to expand.
    # tickmode="array" pins every category label — at the tight 50–60 px
    # row pitch, plotly auto-thins categorical ticks otherwise.
    fig.update_yaxes(
        title=dict(text=level_word,
                   font=dict(size=px["axis_title"], family=FONT_FAMILY),
                   standoff=12),
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        automargin=False,
        tickmode="array", tickvals=list(display_labels), ticktext=list(display_labels),
    )

    save_figure(fig, results / "figures" / out_name, scale=2)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_intensity_drivers(results: Path, figures: Path,
                            only: tuple[str, ...] = ()) -> None:
    """Top-10 occupations and top-10 tasks within each of the three high-lift
    majors. Six PNGs total (or filter with `only=("occ_comp_math",...)`)."""
    try:
        dedup = _intensity_drivers_dedup()
    except ImportError as exc:
        print(f"  -> SKIPPED: dependency not available ({exc})")
        return

    # pct_tasks_affected for occ chart color comes from the same AEI-only
    # eco_2025 file the dedup table uses — keeps the chart internally
    # consistent (intentionally diverges from PRIMARY_DATASET).
    occ_pct_aff = get_pct_tasks_affected(_INTENSITY_DATASET)

    for major_full, slug, major_short in TARGET_MAJORS_DRIVERS:
        sub = dedup[dedup["major_occ_category"] == major_full].copy()
        if sub.empty:
            print(f"  -> skip {slug}: no rated rows for this major")
            continue

        # ── Occ-level ─────────────────────────────────────────────────────
        occ_key = f"occ_{slug}"
        if not only or occ_key in only:
            occ_grp = (
                sub.groupby("title_current")
                .agg(num=("adj_pct", "sum"),
                     raw_pct=("pct_normalized", "sum"),
                     den=("eco_weight", "sum"))
                .reset_index()
            )
            occ_grp = occ_grp[occ_grp["den"] > 0].copy()
            occ_grp["ratio"] = occ_grp["num"] / occ_grp["den"]
            median_ratio = float(occ_grp["ratio"].median())
            assert median_ratio > 0, f"non-positive median occ ratio in {slug}"
            occ_grp["lift"] = occ_grp["ratio"] / median_ratio
            occ_grp["pct_tasks_affected"] = (
                occ_grp["title_current"].map(occ_pct_aff).astype(float)
            )
            # Drop occs with no pct_tasks_affected match (rare; keeps color clean).
            occ_grp = occ_grp.dropna(subset=["pct_tasks_affected"])
            top_occ = occ_grp.nlargest(10, "lift").copy()
            save_csv(
                top_occ.sort_values("lift", ascending=False),
                results / f"intensity_drivers_occ_{slug}.csv",
                float_format="%.4f",
            )
            _render_intensity_driver_chart(
                top_occ.sort_values("lift", ascending=True),
                results, figures,
                level="occ", slug=slug, major_short=major_short,
                label_col="title_current", color_col="pct_tasks_affected",
                color_label="Tasks Exposed", color_fmt="{:.0f}%",
                label_wrap=36,
            )

        # ── Task-level ────────────────────────────────────────────────────
        task_key = f"task_{slug}"
        if not only or task_key in only:
            task_grp = (
                sub.groupby("task_normalized")
                .agg(num=("adj_pct", "sum"),
                     raw_pct=("pct_normalized", "sum"),
                     den=("eco_weight", "sum"),
                     auto_aug=("auto_aug_mean", "mean"),
                     task_display=("task_display", "first"))
                .reset_index()
            )
            task_grp = task_grp[task_grp["den"] > 0].copy()
            task_grp["ratio"] = task_grp["num"] / task_grp["den"]
            median_ratio_t = float(task_grp["ratio"].median())
            assert median_ratio_t > 0, f"non-positive median task ratio in {slug}"
            task_grp["lift"] = task_grp["ratio"] / median_ratio_t
            top_task = task_grp.nlargest(10, "lift").copy()
            save_csv(
                top_task.sort_values("lift", ascending=False),
                results / f"intensity_drivers_task_{slug}.csv",
                float_format="%.4f",
            )
            # Manual margin override for comp_math task — its top-10
            # labels lean heavier on default-width chars (no wide
            # 'M'/'W's, fewer narrow 'i'/'l's) than Arts or Life, so the
            # auto-fit per-char estimate under-counts the actual rendered
            # width and one label clips. +30 px gives clean clearance.
            margin_override = 730 if slug == "comp_math" else None
            _render_intensity_driver_chart(
                top_task.sort_values("lift", ascending=True),
                results, figures,
                level="task", slug=slug, major_short=major_short,
                label_col="task_display", color_col="auto_aug",
                color_label="Auto-Aug (1–5)", color_fmt="{:.2f}",
                label_wrap=52, margin_left=margin_override,
            )


# ─────────────────────────────────────────────────────────────────────────
# adoption_friction_scatter — appendix figure testing the Section 3
# adoption-layer prediction that risk and deployment friction
# discriminate exposure once capability is roughly held constant.
#
# Source: data/final_eco_2025_with_task_properties.csv (12 LLM-rated
# structural properties, 1–5, on 23,850 (task, occ) rows). The same
# property file used by analysis/exploratory/audit_task_properties.
#
# Capability is held roughly constant by restricting to occupations
# whose task load is mostly non-physical (`pct_physical < 33%` — the
# same cut Part 2 uses for its Non-Physical phys-mix tier). Inside that
# slice, each occupation contributes one dot: x = unweighted mean
# rating of the friction property across ALL of that occupation's
# tasks, y = % tasks exposed under all_confirmed. Two panels — left
# for `r`, right for `df`. The non-physical restriction lives at the
# occupation level; no second filter at the task level since these
# occupations are already >67% non-physical by construction.
#
# Why no weighting: weighting by freq×emp concentrates each occ's
# property mean onto a few high-frequency tasks (often routine office
# work with mid-range friction), washing the cross-occ variance out.
# Unweighted preserves the cross-occ signal (ρ ≈ −0.50 for r and −0.42
# for df at occ level; the major-level rollup within non-phys shows
# almost nothing — discrimination lives at occupation level here).
# ─────────────────────────────────────────────────────────────────────────

ADOPTION_FRICTION_PROPS: list[tuple[str, str]] = [
    ("r", "Objective Risk (r)"),
    ("df", "Deployment Friction (df)"),
]

# Cut occupations into Non-Physical bucket: <33% physical tasks. Same
# threshold Part 2 uses for its phys-mix tier scheme.
NONPHYS_PCT_CUTOFF = 33.0


def _load_props_deduped() -> pd.DataFrame:
    """Property file deduped to one row per (task_normalized, title_current).

    eco_2025 expands tasks across the GWA/IWA/DWA hierarchy non-
    proportionally between physical and non-physical, so dedup is
    required before any per-occ summary — same pitfall noted in the
    appendix README under data conventions.
    """
    df = pd.read_csv(DATA_DIR / "final_eco_2025_with_task_properties.csv")
    keep_cols = [
        "task_normalized", "title_current", "physical", "freq_mean",
        "emp_tot_nat_2024",
        "m", "d", "s", "r", "h", "e", "t", "tf", "df", "de", "nt", "ac",
    ]
    return df[keep_cols].drop_duplicates(["task_normalized", "title_current"]).copy()


def build_adoption_friction_scatter(results: Path, figures: Path) -> None:
    """Per-occupation scatter of two friction properties (`r`, `df`) vs
    % tasks exposed, restricted to occupations whose task load is mostly
    non-physical (Part 2's Non-Physical bucket: <33% physical tasks).
    Each dot is one occupation. Spearman ρ + OLS fit per panel.
    """
    from scipy import stats

    props = _load_props_deduped()

    # Per-occ pct_physical over UNIQUE (occ, task) pairs.
    occ_struct = (
        props.groupby("title_current")
        .agg(n_tasks=("physical", "count"),
             n_physical=("physical", "sum"))
        .reset_index()
    )
    occ_struct["pct_physical"] = (
        occ_struct["n_physical"] / occ_struct["n_tasks"] * 100
    )
    nonphys_occs = set(
        occ_struct.loc[occ_struct["pct_physical"] < NONPHYS_PCT_CUTOFF,
                       "title_current"]
    )

    # Per-occ unweighted mean of each friction property over ALL of the
    # occupation's tasks. The non-physical restriction lives at the
    # occupation level above (pct_physical < 33%); double-filtering at
    # the task level adds nothing since these occupations are already
    # >67% non-physical by construction. Weighting by freq×emp was
    # tried and concentrates the mean on a few high-frequency tasks,
    # erasing the cross-occ variance; keep unweighted.
    occ_means = (
        props.groupby("title_current")
        .agg(r=("r", "mean"),
             df=("df", "mean"),
             n_tasks=("r", "count"))
        .reset_index()
    )

    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    occ_means["pct_tasks_affected"] = occ_means["title_current"].map(pct)

    plot_df = occ_means[
        occ_means["title_current"].isin(nonphys_occs)
        & occ_means["pct_tasks_affected"].notna()
    ].copy()
    assert len(plot_df) > 0, "No mostly-non-physical occupations matched"
    n_occs = len(plot_df)

    save_csv(
        plot_df.sort_values("pct_tasks_affected", ascending=False),
        results / "adoption_friction_scatter.csv",
        float_format="%.4f",
    )

    # Spearman ρ + linear fit per property
    stat_rows: list[dict] = []
    fit_lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, _ in ADOPTION_FRICTION_PROPS:
        x = plot_df[code].astype(float).to_numpy()
        y = plot_df["pct_tasks_affected"].astype(float).to_numpy()
        rho, p_val = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 50)
        ys = intercept + slope * xs
        fit_lines[code] = (xs, ys)
        stat_rows.append({"property": code, "spearman_rho": rho,
                          "p_value": p_val, "slope": slope,
                          "intercept": intercept, "n": int(len(x))})
    save_csv(pd.DataFrame(stat_rows),
             results / "adoption_friction_scatter_stats.csv",
             float_format="%.4f")

    W = PAPER_W
    px = paper_fonts(W)

    # Color encodes exposure value so darker = more exposed (mirrors
    # the part_3 underadoption ramp).
    SCATTER_LIGHT = "#cfe0ec"
    SCATTER_DARK = "#2c4f6b"
    cvals = plot_df["pct_tasks_affected"].to_numpy(dtype=float)
    cmin, cmax = float(cvals.min()), float(cvals.max())

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.11,
        subplot_titles=[lbl for _, lbl in ADOPTION_FRICTION_PROPS],
    )

    for col_idx, (code, _label) in enumerate(ADOPTION_FRICTION_PROPS, start=1):
        x = plot_df[code].astype(float)
        y = plot_df["pct_tasks_affected"].astype(float)

        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(
                    size=6,
                    color=y,
                    colorscale=[[0, SCATTER_LIGHT], [1, SCATTER_DARK]],
                    cmin=cmin, cmax=cmax,
                    line=dict(width=0.4, color="rgba(0,0,0,0.25)"),
                    opacity=0.85,
                ),
                customdata=plot_df["title_current"],
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    f"mean {code} (all tasks): %{{x:.2f}}<br>"
                    "% tasks exposed: %{y:.1f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        xs, ys = fit_lines[code]
        # Show legend entry only on the first panel's fit trace so the
        # legend has a single "OLS fit" row.
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                name="OLS fit",
                line=dict(color=PAPER_PALETTE["negative"], width=2, dash="dash"),
                hoverinfo="skip", showlegend=(col_idx == 1),
            ),
            row=1, col=col_idx,
        )

        rho = next(r["spearman_rho"] for r in stat_rows if r["property"] == code)
        p_val = next(r["p_value"] for r in stat_rows if r["property"] == code)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        # Subplot 1 uses bare "x"/"y"; subsequent panels use "x2"/"y2", etc.
        xref_str = "x domain" if col_idx == 1 else f"x{col_idx} domain"
        yref_str = "y domain" if col_idx == 1 else f"y{col_idx} domain"
        fig.add_annotation(
            x=0.02, y=0.97,
            xref=xref_str, yref=yref_str,
            text=f"Spearman ρ = {rho:+.2f}{sig}<br>n = {len(x)} occs",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            align="left",
            bgcolor="rgba(255,255,255,0.78)", borderpad=4,
        )

    for col_idx, (code, _) in enumerate(ADOPTION_FRICTION_PROPS, start=1):
        fig.update_xaxes(
            title=dict(
                text=f"Mean {code} rating across all tasks in occupation",
                font=dict(size=px["axis_title"], family=FONT_FAMILY),
            ),
            # Start at 2.0 — no data below ~2.3 and 1.5 would overlap
            # the y-axis 0% label at the bottom-left corner.
            range=[2.0, 4.5], dtick=0.5,
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
            row=1, col=col_idx,
        )
        y_title = "Tasks Exposed" if col_idx == 1 else None
        fig.update_yaxes(
            title=dict(text=y_title,
                       font=dict(size=px["axis_title"], family=FONT_FAMILY))
                       if y_title else None,
            ticksuffix="%",
            range=[0, 100], dtick=20,
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
            row=1, col=col_idx,
        )

    style_paper_figure(
        fig,
        "Adoption Frictions vs Exposure — Non-Physical Occupations",
        width=W,
        height=600,
        margin=dict(l=90, r=50, t=130, b=110),
    )
    # Pin title below the top edge — Plotly's default title.y for short
    # canvases sometimes places the text flush with the top.
    fig.update_layout(
        title=dict(y=0.96, yanchor="top"),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center", y=-0.22, yanchor="top",
            font=dict(size=px["legend"], family=FONT_FAMILY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    for ann in fig.layout.annotations:
        if ann.text in {lbl for _, lbl in ADOPTION_FRICTION_PROPS}:
            ann.font = dict(size=px["panel_title"], family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    save_figure(fig, results / "figures" / "adoption_friction_scatter.png", scale=2)
    _copy_fig(results, figures, "adoption_friction_scatter.png")
    print("  -> adoption_friction_scatter.png")


# ─────────────────────────────────────────────────────────────────────────
# capability_vs_adoption_all_occs — companion to adoption_friction_scatter.
# Shows the framework's two-layer structure across all 923 occupations:
# capability properties (Schaal ag/da, our s/d) discriminate strongly
# (ρ ≈ +0.55 to +0.68 against exposure), while adoption properties
# (our r, df) barely discriminate at the all-occ level (ρ ≈ −0.17 to
# −0.19). The capability props are riding the phys/non-phys split; the
# adoption signal only emerges once you restrict to non-phys (see
# adoption_friction_scatter).
# ─────────────────────────────────────────────────────────────────────────

CAP_ADO_PANELS: list[tuple[str, str, str, str]] = [
    # (col_key, panel_title, x-axis label, kind)
    ("schaal_ag", "Schaal: Algorithmic Similarity", "Schaal AG",                       "capability"),
    ("schaal_da", "Schaal: Data Abundance",         "Schaal DA",                       "capability"),
    ("s",         "Our: Algorithmic Similarity",    "Mean s across all tasks in occ",  "capability"),
    ("d",         "Our: Data Abundance",            "Mean d across all tasks in occ",  "capability"),
    ("r",         "Our: Objective Risk",            "Mean r across all tasks in occ",  "adoption"),
    ("df",        "Our: Deployment Friction",       "Mean df across all tasks in occ", "adoption"),
]


def build_capability_vs_adoption_all_occs(results: Path, figures: Path) -> None:
    """4-row panel chart contrasting structural / capability / adoption
    discrimination across all 923 occupations.

    Row 1 (one wide panel, gray ramp): pct_physical — the raw structural
       variable. Shows the phys/non-phys split that the capability
       properties subsequently ride.
    Rows 2–3 (blue ramp, 4 panels): capability properties — Schaal ag,
       Schaal da, our s, our d.
    Row 4 (gold ramp, 2 panels): adoption properties — our r, our df.

    Per-occupation property means are unweighted across all of the
    occupation's tasks; y-axis is all_confirmed pct_tasks_affected.
    """
    from scipy import stats
    from lib.builders.part1 import _load_schaal_occ

    props = _load_props_deduped()
    occ = (
        props.groupby("title_current")
        .agg(s=("s", "mean"), d=("d", "mean"),
             r=("r", "mean"), df=("df", "mean"),
             n_tasks=("physical", "count"),
             n_physical=("physical", "sum"))
        .reset_index()
    )
    # pct_physical = share of an occupation's UNIQUE tasks that are
    # physical. Same dedup pitfall as elsewhere (eco_2025 expands tasks
    # over GWA/IWA/DWA non-proportionally between phys and non-phys).
    occ["pct_physical"] = occ["n_physical"] / occ["n_tasks"] * 100

    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    occ["pct_tasks_affected"] = occ["title_current"].map(pct)
    occ = occ.merge(_load_schaal_occ(), on="title_current", how="left")
    occ = occ.dropna(subset=["pct_tasks_affected"]).copy()
    n_total = len(occ)
    assert n_total > 0, "No occupations matched all_confirmed pct"

    # Per-panel Spearman ρ + OLS fit lines. Include pct_physical as the
    # top-row structural panel.
    all_cols = ["pct_physical"] + [p[0] for p in CAP_ADO_PANELS]
    stat_rows: list[dict] = []
    fit_lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code in all_cols:
        sub = occ.dropna(subset=[code])
        x = sub[code].astype(float).to_numpy()
        y = sub["pct_tasks_affected"].astype(float).to_numpy()
        rho, p_val = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 50)
        ys = intercept + slope * xs
        fit_lines[code] = (xs, ys)
        stat_rows.append({"property": code, "spearman_rho": rho,
                          "p_value": p_val, "slope": slope,
                          "intercept": intercept, "n": int(len(x))})
    save_csv(pd.DataFrame(stat_rows),
             results / "capability_vs_adoption_all_occs_stats.csv",
             float_format="%.4f")
    save_csv(occ.sort_values("pct_tasks_affected", ascending=False),
             results / "capability_vs_adoption_all_occs.csv",
             float_format="%.4f")

    W = PAPER_W
    px = paper_fonts(W)

    # Three color ramps so the three layers read as distinct.
    STR_LIGHT, STR_DARK = "#e0e0d8", "#5a5a55"   # gray ramp (structural)
    CAP_LIGHT, CAP_DARK = "#cfe0ec", "#2c4f6b"   # blue ramp (capability)
    ADO_LIGHT, ADO_DARK = "#f4e0c0", "#8a5a1a"   # gold ramp (adoption)

    # 4 rows × 2 cols, with row 1 a single wide panel that spans both
    # columns. Subsequent rows are normal 2-column splits.
    fig = make_subplots(
        rows=4, cols=2,
        horizontal_spacing=0.13,
        vertical_spacing=0.11,
        specs=[
            [{"colspan": 2}, None],       # row 1: pct_physical wide
            [{}, {}],                     # row 2: schaal ag / schaal da
            [{}, {}],                     # row 3: our s / our d
            [{}, {}],                     # row 4: our r / our df
        ],
        subplot_titles=(
            ["Share of Occupation's Tasks That Are Physical"]
            + [p[1] for p in CAP_ADO_PANELS]
        ),
    )

    # ── Row 1: pct_physical (wide) ──────────────────────────────────
    sub = occ.dropna(subset=["pct_physical"])
    x = sub["pct_physical"].astype(float)
    y = sub["pct_tasks_affected"].astype(float)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=5, color=y,
                    colorscale=[[0, STR_LIGHT], [1, STR_DARK]],
                    line=dict(width=0.3, color="rgba(0,0,0,0.25)"),
                    opacity=0.8),
        customdata=sub["title_current"],
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "% physical tasks: %{x:.1f}%<br>"
            "tasks exposed: %{y:.1f}%<extra></extra>"
        ),
        showlegend=False,
    ), row=1, col=1)
    xs, ys = fit_lines["pct_physical"]
    # Show the OLS-fit legend entry once, on this very first trace.
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name="OLS fit",
        line=dict(color=PAPER_PALETTE["negative"], width=2, dash="dash"),
        hoverinfo="skip", showlegend=True,
    ), row=1, col=1)
    rho_phys = next(r["spearman_rho"] for r in stat_rows if r["property"] == "pct_physical")
    fig.add_annotation(
        x=0.015, y=0.96, xref="x domain", yref="y domain",
        text=f"Spearman ρ = {rho_phys:+.2f}<br>n = {len(x)} occs",
        showarrow=False, xanchor="left", yanchor="top",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
        align="left", bgcolor="rgba(255,255,255,0.82)", borderpad=3,
    )

    # ── Rows 2–4: capability + adoption panels ───────────────────────
    # Subplot indices start at 2 for the second trace (pct_physical
    # consumed index 1). CAP_ADO_PANELS go into x2/y2 through x7/y7.
    for i, (code, _title, _xlabel, kind) in enumerate(CAP_ADO_PANELS):
        # Map flat index → (row, col): panels 0–1 → row 2, 2–3 → row 3, 4–5 → row 4
        row = i // 2 + 2
        col = i % 2 + 1
        sub = occ.dropna(subset=[code])
        x = sub[code].astype(float)
        y = sub["pct_tasks_affected"].astype(float)
        light, dark = (CAP_LIGHT, CAP_DARK) if kind == "capability" else (ADO_LIGHT, ADO_DARK)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color=y,
                        colorscale=[[0, light], [1, dark]],
                        line=dict(width=0.3, color="rgba(0,0,0,0.25)"),
                        opacity=0.8),
            customdata=sub["title_current"],
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                f"{code}: %{{x:.2f}}<br>tasks exposed: %{{y:.1f}}%<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=col)

        xs, ys = fit_lines[code]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=PAPER_PALETTE["negative"], width=2, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=col)

        rho = next(r["spearman_rho"] for r in stat_rows if r["property"] == code)
        # Plotly axis index: pct_physical took x1/y1, panel i takes x(i+2)/y(i+2).
        axis_idx = i + 2
        xref = f"x{axis_idx} domain"
        yref = f"y{axis_idx} domain"
        fig.add_annotation(
            x=0.03, y=0.96, xref=xref, yref=yref,
            text=f"Spearman ρ = {rho:+.2f}<br>n = {len(x)} occs",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            align="left", bgcolor="rgba(255,255,255,0.82)", borderpad=3,
        )

    # ── Axes: pct_physical (row 1) on 0–100% with a % suffix ─────────
    fig.update_xaxes(
        title=dict(
            text="% Physical Tasks in Occupation",
            font=dict(size=px["axis_title"], family=FONT_FAMILY),
        ),
        range=[0, 100], dtick=20, ticksuffix="%",
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="Tasks Exposed",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        ticksuffix="%", range=[0, 100], dtick=20,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        row=1, col=1,
    )

    # Rows 2–4: capability + adoption panels.
    for i, (code, _title, xlabel, _kind) in enumerate(CAP_ADO_PANELS):
        row = i // 2 + 2
        col = i % 2 + 1
        if code in ("s", "d", "r", "df"):
            x_range, x_dtick = [2.0, 4.5], 0.5
        else:
            sub = occ.dropna(subset=[code])
            xmin, xmax = float(sub[code].min()), float(sub[code].max())
            margin = (xmax - xmin) * 0.05
            x_range, x_dtick = [xmin - margin, xmax + margin], None
        xkw = dict(
            title=dict(text=xlabel,
                       font=dict(size=px["axis_title"], family=FONT_FAMILY)),
            range=x_range,
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        )
        if x_dtick:
            xkw["dtick"] = x_dtick
        fig.update_xaxes(row=row, col=col, **xkw)

        y_title = "Tasks Exposed" if col == 1 else None
        fig.update_yaxes(
            title=(dict(text=y_title,
                        font=dict(size=px["axis_title"], family=FONT_FAMILY))
                   if y_title else None),
            ticksuffix="%", range=[0, 100], dtick=20,
            tickfont=dict(size=px["tick"], family=FONT_FAMILY),
            showgrid=True, gridcolor=PAPER_PALETTE["grid"],
            row=row, col=col,
        )

    style_paper_figure(
        fig,
        "Structural, Capability, and Adoption Properties — All Occupations",
        width=W,
        height=1500,
        margin=dict(l=100, r=50, t=120, b=110),
    )
    fig.update_layout(
        title=dict(y=0.975, yanchor="top"),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center", y=-0.10, yanchor="top",
            font=dict(size=px["legend"], family=FONT_FAMILY),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    # Subplot titles get sized + colored from the paper ladder. Include
    # the top wide-row title alongside the CAP_ADO_PANELS titles.
    panel_titles = {p[1] for p in CAP_ADO_PANELS} | {
        "Share of Occupation's Tasks That Are Physical",
    }
    for ann in fig.layout.annotations:
        if ann.text in panel_titles:
            ann.font = dict(size=px["panel_title"], family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    save_figure(
        fig, results / "figures" / "capability_vs_adoption_all_occs.png",
        scale=2,
    )
    _copy_fig(results, figures, "capability_vs_adoption_all_occs.png")
    print("  -> capability_vs_adoption_all_occs.png")


def main() -> None:
    results = ensure_results_dir(HERE)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)

    print("=" * 60)
    print("Appendix figures")
    print("=" * 60)

    print("\n[1/12] convergence_full (one full-matrix per SOC level)")
    for lvl_key, lvl_title in [("major", "Major level"),
                                ("minor", "Minor level"),
                                ("broad", "Broad level"),
                                ("occupation", "Occupation level")]:
        short = "occ" if lvl_key == "occupation" else lvl_key
        build_convergence_full(
            results, figures,
            levels=[(lvl_key, lvl_title)],
            out_name=f"convergence_full_{short}.png",
            csv_name=f"spearman_combined_full_{short}.csv",
        )

    print("\n[2/12] overview_no_autoaug (paper part_1 overview, no auto_aug)")
    build_overview_no_autoaug(results, figures)

    print("\n[3/12] temporal_trend_nonphys (Part 1 trend, non-physical tasks only)")
    build_temporal_trend_nonphys(results, figures)

    print("\n[4/12] major_categories_trend (Part 2 trend chart, relocated)")
    build_major_categories_trend(results, figures)

    print("\n[5/12] eloundou_divergence_major (z-score divergence by major occ cat)")
    build_eloundou_divergence_major(results, figures)

    print("\n[6/12] ska_full (full element-level SKA)")
    build_ska_full(results, figures)

    print("\n[7/12] gwa_wkrs_wages (workers/wages counterpart to part_2 gwa_pct)")
    build_gwa_wkrs_wages(results, figures)

    print("\n[8/13] state_clusters_each_ranked (companion to Part 3 cluster map)")
    build_state_clusters_each_ranked(results, figures)

    print("\n[9/13] state_clusters_combined_ranked (sum of each_ranked panel ranks)")
    build_state_clusters_combined_ranked(results, figures)

    print("\n[10/13] underadoption_gap (% tasks exposed ÷ share of AI usage)")
    build_underadoption_gap(results, figures)

    print("\n[11/13] intensity_drivers (top occs + tasks within 3 high-lift majors)")
    build_intensity_drivers(results, figures)

    print("\n[12/13] adoption_friction_scatter (Section 3 adoption props × non-phys occs)")
    build_adoption_friction_scatter(results, figures)

    print("\n[13/13] capability_vs_adoption_all_occs (capability vs adoption across all occs)")
    build_capability_vs_adoption_all_occs(results, figures)

    print("\nDone — figures in results/figures/ and figures/")


if __name__ == "__main__":
    main()
