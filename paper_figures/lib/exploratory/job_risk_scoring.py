"""
run.py — Job Exposure: Job Risk Scoring

Which jobs are most at risk of replacement (not just transformation)?

Eight binary risk flags per occupation with weighted scoring. Max = 10.

Exposure depth (weight 2 each — the "AI is actually reaching this job" signal):
  1. pct_tasks_affected > 50%                       (absolute, matches the gate intent)
  2. overall_ska_pct > median                       (AI covers more of this occ than typical)

Exposure velocity (weight 1 each — the "and it's still growing" signal):
  3. pct_delta > 0 AND > median(pct_delta)
  4. ska_delta > 0 AND > median(ska_delta)

Exposure depth (weight 1 — complementary signal to flag 1):
  8. auto_avg_with_vals > median                    (tasks are highly auto/augmentable)

Structural vulnerability (weight 1 each — the worker's safety net is thin):
  5. job_zone ∈ {1, 2, 3}
  6. outlook ∈ {2, 3}                               (1 = good outlook/low wages — NOT at risk)
  7. n_software > median

Risk tiers (on the 0–10 score):
  - Low:        0–2
  - Mod-Low:    3–4
  - Mod-High:   5–7
  - High:       8–10  (requires pct_tasks_affected ≥ 33% — exposure gate)

Exposure gate: a score of 8+ with pct < 33% is downgraded to Mod-High.
Rationale: without meaningful task exposure, no amount of structural + trend
signals can justify a "high risk" label on a job AI hasn't reached yet.

SKA flag note: flag 2 uses `overall_pct` (ratio-of-sums percentage framing),
not the legacy `overall_gap`. This is consistent with the SKA reporting overhaul
— see METHODOLOGY.md §2.12. A value of 100% means AI's demonstrated capability
matches the occupation's skill requirement exactly; above 100% = AI leads.

Primary config: all_confirmed. Cross-config comparison shows tier volatility.

Run from project root:
    venv/Scripts/python -m analysis.questions.job_exposure.job_risk_scoring.run
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.config import (
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ANALYSIS_CONFIG_SERIES,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.compute_ska import SKAData, compute_ska, load_ska_data
from lib.utils import (
    COLORS,
    CATEGORY_PALETTE,
    FONT_FAMILY,
    format_workers,
    save_csv,
    save_figure,
    style_figure,
)

HERE = Path(__file__).resolve().parent
# This file lives at <repo_root>/paper_figures/lib/exploratory/job_risk_scoring.py;
# reference data (incl. tech_skills_simple.csv) lives at <repo_root>/data/reference/.
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"
TECH_SKILLS_FILE = DATA_DIR / "tech_skills_simple.csv"

PRIMARY_KEY = "all_confirmed"
CEILING_KEY = "all_ceiling"

RISK_AT_RISK_ZONE = {1, 2, 3}
RISK_AT_RISK_OUTLOOK = {2, 3}
PIVOT_N = 10

EXPOSURE_GATE = 33.0          # pct below this cannot be high risk
PCT_ABS_THRESHOLD = 50.0      # flag 1: absolute pct cutoff (was median — now fixed)

# Weights: flags 1 and 2 (exposure depth) get 2x, everything else 1x. Max = 10.
FLAG_WEIGHTS = {
    "flag1_pct":        2,   # pct_tasks_affected > 50%
    "flag2_ska":        2,   # overall_ska_pct > median
    "flag3_pct_trend":  1,   # pct_delta > 0 AND > median pct_delta
    "flag4_ska_trend":  1,   # ska_delta > 0 AND > median ska_delta
    "flag5_job_zone":   1,   # job_zone ∈ {1, 2, 3}
    "flag6_outlook":    1,   # outlook ∈ {2, 3}
    "flag7_n_software": 1,   # n_software > median
    "flag8_auto_aug":   1,   # auto_avg_with_vals > median
}

MAX_SCORE = sum(FLAG_WEIGHTS.values())  # 10

# Four-tier structure on the 0–10 score
TIER_LABELS = {
    "high":     "High Risk (8–10)",
    "mod_high": "Mod-High Risk (5–7)",
    "mod_low":  "Mod-Low Risk (3–4)",
    "low":      "Low Risk (0–2)",
}
TIER_ORDER = ["high", "mod_high", "mod_low", "low"]
TIER_COLORS = {
    "high":     COLORS["negative"],
    "mod_high": COLORS["accent"],
    "mod_low":  COLORS["secondary"],
    "low":      COLORS["muted"],
}


def _assign_risk_tier(score: int, pct: float) -> str:
    """Assign risk tier with exposure gate.

    High requires both (a) score ≥ 8 and (b) pct_tasks_affected ≥ 33%.
    If the score is 8+ but the gate fails, downgrade to mod_high.
    """
    if score >= 8:
        return "high" if pct >= EXPOSURE_GATE else "mod_high"
    if score >= 5:
        return "mod_high"
    if score >= 3:
        return "mod_low"
    return "low"


# ── Employment + structural lookup ────────────────────────────────────────────

def _get_structural_data() -> pd.DataFrame:
    """Return DataFrame with title_current, emp_nat, wage_nat, major, job_zone,
    outlook, and auto_avg_with_vals (for flag 8)."""
    from backend.compute import get_explorer_occupations

    rows = []
    for occ in get_explorer_occupations():
        rows.append({
            "title_current": occ["title_current"],
            "emp_nat": occ.get("emp") or 0,
            "wage_nat": occ.get("wage") or 0,
            "major": occ.get("major", ""),
            "job_zone": occ.get("job_zone"),
            "outlook": occ.get("dws_star_rating"),
            "auto_avg_with_vals": occ.get("auto_avg_with_vals"),
        })
    return pd.DataFrame(rows)


# ── Trend helpers ─────────────────────────────────────────────────────────────

def _compute_pct_trend(config_key: str) -> pd.Series:
    """Return pct_delta (last - first) per occ for a config's time series."""
    series = ANALYSIS_CONFIG_SERIES[config_key]
    if len(series) < 2:
        return pd.Series(dtype=float)
    pct_first = get_pct_tasks_affected(series[0])
    pct_last = get_pct_tasks_affected(series[-1])
    combined = pd.DataFrame({"first": pct_first, "last": pct_last})
    combined["delta"] = combined["last"].fillna(0) - combined["first"].fillna(0)
    return combined["delta"]


def _compute_ska_trend(config_key: str, ska_data: SKAData) -> pd.Series:
    """Return ska_delta (overall_pct last - first) per occ for a config's series.

    Uses overall_pct (ratio-of-sums percentage framing) to stay consistent with
    flag 2. Delta is in percentage points on the AI-as-%-of-occ scale.
    """
    series = ANALYSIS_CONFIG_SERIES[config_key]
    if len(series) < 2:
        return pd.Series(dtype=float)
    pct_first = get_pct_tasks_affected(series[0])
    pct_last = get_pct_tasks_affected(series[-1])
    result_first = compute_ska(pct_first, ska_data)
    result_last = compute_ska(pct_last, ska_data)
    pct_first_series = result_first.occ_gaps.set_index("title_current")["overall_pct"]
    pct_last_series = result_last.occ_gaps.set_index("title_current")["overall_pct"]
    delta = (pct_last_series - pct_first_series).rename("ska_delta")
    return delta


# ── Flag computation ──────────────────────────────────────────────────────────

def _compute_flags(
    df: pd.DataFrame,
    pct: pd.Series,
    ska_pct: pd.Series,
    pct_delta: pd.Series,
    ska_delta: pd.Series,
) -> pd.DataFrame:
    """Compute all 8 binary flags with weighted scoring + exposure gate.

    Flag 1 uses an ABSOLUTE threshold (pct > 50%); the rest use medians.
    Flag 2 uses overall SKA percentage (ratio-of-sums), not the raw gap.
    """
    out = df.copy()
    out["pct"] = out["title_current"].map(pct).fillna(0.0)
    out["ska_pct"] = out["title_current"].map(ska_pct).fillna(np.nan)
    out["pct_delta"] = out["title_current"].map(pct_delta).fillna(np.nan)
    out["ska_delta"] = out["title_current"].map(ska_delta).fillna(np.nan)
    # auto_avg_with_vals may be None → coerce to NaN then fill 0 for median comparison
    out["auto_avg"] = pd.to_numeric(out.get("auto_avg_with_vals"), errors="coerce")

    # Medians (flag 1 bypasses median in favor of absolute threshold)
    ska_pct_median = out["ska_pct"].median()
    pct_delta_median = out["pct_delta"].median()
    ska_delta_median = out["ska_delta"].median()
    n_software_median = out["n_software"].median()
    auto_median = out["auto_avg"].median()

    # Exposure depth
    out["flag1_pct"] = (out["pct"] > PCT_ABS_THRESHOLD).astype(int)
    out["flag2_ska"] = (out["ska_pct"] > ska_pct_median).astype(int)

    # Exposure velocity
    out["flag3_pct_trend"] = (
        (out["pct_delta"] > 0) & (out["pct_delta"] > pct_delta_median)
    ).astype(int)
    out["flag4_ska_trend"] = (
        (out["ska_delta"] > 0) & (out["ska_delta"] > ska_delta_median)
    ).astype(int)

    # Structural vulnerability
    out["flag5_job_zone"] = out["job_zone"].apply(
        lambda z: 1 if pd.notna(z) and int(z) in RISK_AT_RISK_ZONE else 0
    )
    out["flag6_outlook"] = out["outlook"].apply(
        lambda o: 1 if pd.notna(o) and int(o) in RISK_AT_RISK_OUTLOOK else 0
    )
    out["flag7_n_software"] = (out["n_software"] > n_software_median).astype(int)

    # Auto-aug depth (flag 8)
    out["flag8_auto_aug"] = (out["auto_avg"] > auto_median).astype(int)

    # Weighted score
    out["risk_score"] = sum(
        out[col] * weight for col, weight in FLAG_WEIGHTS.items()
    )

    # Apply exposure gate + 4-tier assignment
    out["risk_tier"] = [
        _assign_risk_tier(score, pct_val)
        for score, pct_val in zip(out["risk_score"], out["pct"])
    ]

    # Track which occs were gated (score ≥ 8 but pct < 33%)
    out["exposure_gated"] = (out["risk_score"] >= 8) & (out["pct"] < EXPOSURE_GATE)

    # Store thresholds for reporting
    out.attrs["pct_threshold"] = PCT_ABS_THRESHOLD
    out.attrs["ska_pct_median"] = ska_pct_median
    out.attrs["pct_delta_median"] = pct_delta_median
    out.attrs["ska_delta_median"] = ska_delta_median
    out.attrs["n_software_median"] = n_software_median
    out.attrs["auto_median"] = auto_median

    return out


# ── Figures ───────────────────────────────────────────────────────────────────

def _risk_distribution_bar(df: pd.DataFrame, config_label: str) -> go.Figure:
    """Bar chart: number of occupations in each risk tier (4-tier)."""
    counts = df["risk_tier"].value_counts()
    emp_by_tier = df.groupby("risk_tier")["emp_nat"].sum()
    fig = go.Figure()
    for tier in TIER_ORDER:
        n = counts.get(tier, 0)
        e = emp_by_tier.get(tier, 0)
        fig.add_trace(go.Bar(
            x=[TIER_LABELS[tier]], y=[n],
            marker=dict(color=TIER_COLORS[tier], line=dict(width=0)),
            text=[f"{n} occs<br>{e/1e6:.1f}M workers"],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["neutral"], family=FONT_FAMILY),
            name=TIER_LABELS[tier],
            showlegend=False,
        ))
    gated = df["exposure_gated"].sum()
    subtitle = f"{config_label} | Weighted scoring (max {MAX_SCORE}) + 33% exposure gate"
    if gated > 0:
        subtitle += f" | {gated} occs downgraded to Mod-High by gate"
    style_figure(
        fig,
        "Risk Tier Distribution",
        subtitle=subtitle,
        x_title=None, y_title="Number of Occupations",
        height=550, width=800, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=60, r=40, t=80, b=100),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"]),
        xaxis=dict(showgrid=False),
        bargap=0.35,
    )
    return fig


def _risk_vs_pct_scatter(df: pd.DataFrame, config_label: str) -> go.Figure:
    """Scatter: risk_score (x) vs pct_tasks_affected (y)."""
    fig = go.Figure()
    for tier in TIER_ORDER:
        sub = df[df["risk_tier"] == tier]
        fig.add_trace(go.Scatter(
            x=sub["risk_score"] + (np.random.default_rng(42).uniform(-0.25, 0.25, len(sub))),
            y=sub["pct"],
            mode="markers",
            name=TIER_LABELS[tier],
            marker=dict(color=TIER_COLORS[tier], size=6, opacity=0.6,
                        line=dict(width=0.5, color=COLORS["bg"])),
            text=sub["title_current"],
            hovertemplate="<b>%{text}</b><br>Risk Score: %{x:.0f}<br>% Tasks: %{y:.1f}%<extra></extra>",
        ))
    # Gate line
    fig.add_hline(y=EXPOSURE_GATE, line_dash="dot", line_color=COLORS["accent"], line_width=1,
                  annotation_text=f"Exposure gate ({EXPOSURE_GATE:.0f}%)",
                  annotation_position="right",
                  annotation_font=dict(size=9, color=COLORS["accent"], family=FONT_FAMILY))
    style_figure(
        fig,
        "Risk Score vs Task Exposure",
        subtitle=f"{config_label} | Jitter on x for visibility | Horizontal line = exposure gate",
        x_title=f"Weighted Risk Score (0–{MAX_SCORE})", y_title="% Tasks Affected",
        height=650, width=950, show_legend=True,
    )
    fig.update_layout(
        margin=dict(l=60, r=40, t=80, b=120),
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=10, color=COLORS["neutral"], family=FONT_FAMILY)),
    )
    return fig


def _flag_breakdown_by_score(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: for each risk score, show which flags are typically active."""
    flag_cols = list(FLAG_WEIGHTS.keys())
    flag_labels = {
        "flag1_pct":        "% Tasks > 50%",
        "flag2_ska":        "SKA pct > median",
        "flag3_pct_trend":  "Pct trend ↑",
        "flag4_ska_trend":  "SKA trend ↑",
        "flag5_job_zone":   "Job zone 1-3",
        "flag6_outlook":    "Outlook 2-3",
        "flag7_n_software": "Software > median",
        "flag8_auto_aug":   "Auto-aug > median",
    }
    scores = sorted(df["risk_score"].unique())
    fig = go.Figure()
    for i, flag_col in enumerate(flag_cols):
        pcts = []
        for score in scores:
            score_df = df[df["risk_score"] == score]
            if len(score_df) == 0:
                pcts.append(0)
            else:
                pcts.append(score_df[flag_col].mean() * 100)
        fig.add_trace(go.Bar(
            x=[str(s) for s in scores], y=pcts,
            name=flag_labels[flag_col],
            marker=dict(color=CATEGORY_PALETTE[i], line=dict(width=0)),
        ))
    style_figure(
        fig,
        "Which Flags Drive Each Risk Score Level?",
        subtitle="% of occupations at each score that have each flag active",
        x_title="Weighted Risk Score", y_title="% with Flag Active",
        height=600, width=1000, show_legend=True,
    )
    fig.update_layout(
        barmode="group", bargap=0.15, bargroupgap=0.05,
        margin=dict(l=60, r=40, t=80, b=120),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY)),
    )
    return fig


def _tier_top_occs_bar(df: pd.DataFrame, tier: str, top_n: int = 20) -> go.Figure:
    """Horizontal bar: top occupations in a single tier, sized by workers affected,
    labeled with pct, risk_score, and active flag count."""
    sub = df[df["risk_tier"] == tier].copy()
    if sub.empty:
        return go.Figure()
    sub["workers_affected"] = sub["pct"] / 100.0 * sub["emp_nat"]

    # Sort by workers_affected descending, take top N
    top = sub.nlargest(top_n, "workers_affected").sort_values("workers_affected", ascending=True)

    flag_cols = [c for c in top.columns if c.startswith("flag")]
    top["n_flags"] = top[flag_cols].sum(axis=1).astype(int)

    labels = [
        f"{r['pct']:.0f}% tasks | score {r['risk_score']:.0f} | "
        f"{r['n_flags']}/{len(flag_cols)} flags | "
        f"{r['workers_affected']/1e6:.1f}M workers"
        for _, r in top.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top["title_current"],
        x=top["workers_affected"],
        orientation="h",
        marker=dict(color=TIER_COLORS[tier], line=dict(width=0)),
        text=labels,
        textposition="outside",
        textfont=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False,
    ))
    n_total = len(sub)
    total_workers = sub["workers_affected"].sum()
    chart_h = max(500, len(top) * 28 + 200)
    style_figure(
        fig,
        f"{TIER_LABELS[tier]} — Top {len(top)} by Workers Affected",
        subtitle=(
            f"{n_total} total occupations in tier | "
            f"{total_workers/1e6:.1f}M total workers affected"
        ),
        x_title=None, height=chart_h, width=1050, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=20, r=260, t=80, b=80),
        xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=9, family=FONT_FAMILY)),
        bargap=0.25,
    )
    return fig


def _tier_flag_profile_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: for each tier, show the average activation rate of each flag.
    Rows = tiers, columns = flags. Cell = % of occs in that tier with the flag active."""
    flag_cols = list(FLAG_WEIGHTS.keys())
    flag_labels = {
        "flag1_pct":        "% Tasks\n> 50%",
        "flag2_ska":        "SKA pct\n> median",
        "flag3_pct_trend":  "Pct\ntrend ↑",
        "flag4_ska_trend":  "SKA\ntrend ↑",
        "flag5_job_zone":   "Job zone\n1-3",
        "flag6_outlook":    "Outlook\n2-3",
        "flag7_n_software": "Software\n> median",
        "flag8_auto_aug":   "Auto-aug\n> median",
    }

    rows_data = []
    for tier in TIER_ORDER:
        sub = df[df["risk_tier"] == tier]
        if sub.empty:
            continue
        row = {"tier": TIER_LABELS[tier]}
        for fc in flag_cols:
            row[fc] = sub[fc].mean() * 100
        row["n_occs"] = len(sub)
        row["avg_pct"] = sub["pct"].mean()
        rows_data.append(row)
    summary = pd.DataFrame(rows_data)

    z = summary[flag_cols].values
    y_labels = [f"{r['tier']} ({r['n_occs']} occs, avg {r['avg_pct']:.0f}%)"
                for _, r in summary.iterrows()]
    x_labels = [flag_labels[fc] for fc in flag_cols]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#f5f5f0"], [0.5, COLORS["accent"]], [1.0, COLORS["negative"]]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, family=FONT_FAMILY),
        hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
        showscale=True,
        colorbar=dict(title="% with<br>flag active", ticksuffix="%", tickfont=dict(size=10)),
    ))
    style_figure(
        fig,
        "Flag Activation Profile by Risk Tier",
        subtitle="What percentage of occupations in each tier have each flag active?",
        x_title=None, y_title=None,
        height=400, width=900, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=20, r=80, t=80, b=100),
        xaxis=dict(tickfont=dict(size=9, family=FONT_FAMILY), tickangle=0),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10, family=FONT_FAMILY)),
    )
    return fig


def _cross_config_volatility(risk_all: pd.DataFrame, primary_df: pd.DataFrame) -> go.Figure:
    """
    Show occupations that change tiers across configs.
    Focus on the most volatile ones + base ranking.
    """
    tier_to_num = {"high": 4, "mod_high": 3, "mod_low": 2, "low": 1}
    config_order = list(ANALYSIS_CONFIGS.keys())

    # Build pivot of tier assignments
    pivot = risk_all.pivot(index="title_current", columns="config", values="risk_tier")
    pivot = pivot.reindex(columns=config_order)
    pivot_num = pivot.map(lambda t: tier_to_num.get(t, 0))

    # Compute volatility: range of tier numbers across configs
    pivot_num["tier_range"] = pivot_num[config_order].max(axis=1) - pivot_num[config_order].min(axis=1)

    # Get occupations that actually change tier (range > 0)
    volatile = pivot_num[pivot_num["tier_range"] > 0].nlargest(30, "tier_range")

    # Also get top 10 by primary risk score for context
    top_primary = primary_df.nlargest(10, "risk_score")["title_current"].tolist()

    # Combine: volatile + top primary, deduplicated
    show_occs = list(dict.fromkeys(volatile.index.tolist() + top_primary))[:40]

    pivot_show = pivot.loc[[o for o in show_occs if o in pivot.index]]
    pivot_num_show = pivot_num.loc[pivot_show.index, config_order]

    z = pivot_num_show.values
    text = pivot_show.values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[ANALYSIS_CONFIG_LABELS[k] for k in config_order],
        y=pivot_show.index.tolist(),
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.00, COLORS["muted"]],
            [0.33, COLORS["secondary"]],
            [0.66, COLORS["accent"]],
            [1.00, COLORS["negative"]],
        ],
        showscale=False,
        hovertemplate="<b>%{y}</b><br>Config: %{x}<br>Tier: %{text}<extra></extra>",
    ))
    chart_h = max(700, len(pivot_show) * 20 + 250)
    style_figure(
        fig,
        "Which Occupations Change Risk Tier Across Configs?",
        subtitle="Top volatile occs + top-risk by primary config | Tiers: high / mod_high / mod_low / low",
        x_title=None, y_title=None, height=chart_h, width=950, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=20, r=40, t=80, b=100),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9, family=FONT_FAMILY)),
        xaxis=dict(tickfont=dict(size=10, family=FONT_FAMILY)),
    )
    return fig


def _cross_config_tier_shifts(risk_all: pd.DataFrame) -> pd.DataFrame:
    """Find occupations that jump between tiers across configs."""
    tier_to_num = {"high": 3, "moderate": 2, "low": 1}
    config_order = list(ANALYSIS_CONFIGS.keys())

    pivot = risk_all.pivot(index="title_current", columns="config", values="risk_tier")
    pivot = pivot.reindex(columns=config_order)
    pivot_num = pivot.map(lambda t: tier_to_num.get(t, 0))

    shifts = []
    for occ in pivot.index:
        tiers = pivot.loc[occ]
        tier_nums = pivot_num.loc[occ]
        if tier_nums.max() - tier_nums.min() > 0:
            shifts.append({
                "title_current": occ,
                "min_tier": tiers[tier_nums.idxmin()],
                "max_tier": tiers[tier_nums.idxmax()],
                "tier_range": int(tier_nums.max() - tier_nums.min()),
                **{f"tier_{k}": tiers[k] for k in config_order},
            })
    return pd.DataFrame(shifts).sort_values("tier_range", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    fig_dir = results / "figures"
    print("Job Risk Scoring -- generating outputs...\n")

    # ── Structural data ───────────────────────────────────────────────────────
    print("Loading structural data...")
    struct = _get_structural_data()
    print(f"  {len(struct)} occupations")

    # ── n_software ────────────────────────────────────────────────────────────
    if not TECH_SKILLS_FILE.exists():
        raise FileNotFoundError(
            f"tech_skills_simple.csv not found at {TECH_SKILLS_FILE}. "
            "Run: venv/Scripts/python -m analysis.data.compute_tech_skills"
        )
    tech = pd.read_csv(TECH_SKILLS_FILE)
    struct = struct.merge(
        tech[["title", "n_software"]].rename(columns={"title": "title_current"}),
        on="title_current", how="left",
    )
    struct["n_software"] = struct["n_software"].fillna(0).astype(int)
    print(f"  n_software joined: {struct['n_software'].gt(0).sum()} occs matched\n")

    # ── Load SKA data once ────────────────────────────────────────────────────
    print("Loading SKA base data...")
    ska_data = load_ska_data()

    # ── Primary config: all_confirmed ────────────────────────────────────────
    primary_dataset = ANALYSIS_CONFIGS[PRIMARY_KEY]
    print(f"\n== Primary config: {PRIMARY_KEY} ({primary_dataset}) ==")

    print("  Computing pct_tasks_affected...")
    pct_primary = get_pct_tasks_affected(primary_dataset)

    print("  Computing SKA (overall_pct)...")
    ska_primary = compute_ska(pct_primary, ska_data)
    ska_pct_primary = ska_primary.occ_gaps.set_index("title_current")["overall_pct"]

    print("  Computing pct trend (first -> last date)...")
    pct_delta_primary = _compute_pct_trend(PRIMARY_KEY)

    print("  Computing SKA trend (first -> last date)...")
    ska_delta_primary = _compute_ska_trend(PRIMARY_KEY, ska_data)

    print("  Computing risk flags (weighted scoring, 8 flags, max 10)...")
    primary_df = _compute_flags(struct, pct_primary, ska_pct_primary,
                                pct_delta_primary, ska_delta_primary)

    # Print tier summary
    tier_emp = primary_df.groupby("risk_tier")["emp_nat"].sum()
    for tier in TIER_ORDER:
        n = (primary_df["risk_tier"] == tier).sum()
        e = tier_emp.get(tier, 0)
        print(f"  {TIER_LABELS[tier]}: {n} occs, {e/1e6:.1f}M workers")

    gated = primary_df["exposure_gated"].sum()
    if gated > 0:
        print(f"  Exposure gate: {gated} occs downgraded from high to moderate (pct < {EXPOSURE_GATE}%)")

    # Flag frequency
    flag_cols = list(FLAG_WEIGHTS.keys())
    flag_freq = primary_df[flag_cols].sum().reset_index()
    flag_freq.columns = ["flag", "n_triggered"]
    flag_freq["pct_of_occs"] = flag_freq["n_triggered"] / len(primary_df) * 100
    flag_freq["weight"] = flag_freq["flag"].map(FLAG_WEIGHTS)
    save_csv(flag_freq, results / "flags_breakdown.csv")

    # Per-score flag breakdown
    print("\n  Flag composition by risk score:")
    for score in sorted(primary_df["risk_score"].unique()):
        score_df = primary_df[primary_df["risk_score"] == score]
        n = len(score_df)
        tier = score_df["risk_tier"].mode().iloc[0] if n > 0 else "?"
        avg_pct = score_df["pct"].mean()
        flag_pcts = {col: f"{score_df[col].mean()*100:.0f}%" for col in flag_cols}
        print(f"    Score {score} ({tier}): {n} occs, avg pct={avg_pct:.1f}%, "
              f"flags: {flag_pcts}")

    # Save primary results
    out_cols = ["title_current", "emp_nat", "wage_nat", "major", "job_zone", "outlook",
                "n_software", "auto_avg", "pct", "ska_pct", "pct_delta", "ska_delta"] + flag_cols + \
               ["risk_score", "risk_tier", "exposure_gated"]
    primary_out = primary_df[out_cols].rename(columns={
        "emp_nat": "employment", "wage_nat": "median_wage", "pct": "pct_tasks_affected"
    })
    save_csv(primary_out.sort_values("risk_score", ascending=False),
             results / "risk_scores_primary.csv")
    print("\nSaved risk_scores_primary.csv")

    # Tier summary
    tier_summary = primary_df.groupby("risk_tier").agg(
        n_occs=("title_current", "count"),
        total_emp=("emp_nat", "sum"),
        avg_pct=("pct", "mean"),
        avg_risk_score=("risk_score", "mean"),
    ).reset_index()
    # Compute total wages affected per tier
    primary_df["workers_affected"] = primary_df["pct"] / 100.0 * primary_df["emp_nat"]
    primary_df["wages_affected"] = primary_df["workers_affected"] * primary_df["wage_nat"]
    wages_by_tier = primary_df.groupby("risk_tier").agg(
        total_workers_affected=("workers_affected", "sum"),
        total_wages_affected=("wages_affected", "sum"),
    ).reset_index()
    tier_summary = tier_summary.merge(wages_by_tier, on="risk_tier", how="left")
    save_csv(tier_summary, results / "risk_tier_summary.csv")

    # ── Cross-config comparison ───────────────────────────────────────────────
    print("\n== Cross-config risk scores ==")
    all_config_rows = []
    for config_key, dataset_name in ANALYSIS_CONFIGS.items():
        print(f"  {config_key}: {dataset_name}")
        pct_cfg = get_pct_tasks_affected(dataset_name)
        ska_cfg = compute_ska(pct_cfg, ska_data)
        ska_pct_cfg = ska_cfg.occ_gaps.set_index("title_current")["overall_pct"]
        pct_delta_cfg = _compute_pct_trend(config_key)
        ska_delta_cfg = _compute_ska_trend(config_key, ska_data)
        flags_cfg = _compute_flags(struct, pct_cfg, ska_pct_cfg, pct_delta_cfg, ska_delta_cfg)
        flags_cfg["config"] = config_key
        all_config_rows.append(
            flags_cfg[["title_current", "config", "pct", "ska_pct",
                        "risk_score", "risk_tier", "exposure_gated"]].copy()
        )

    risk_all = pd.concat(all_config_rows, ignore_index=True)
    save_csv(risk_all, results / "risk_scores_all_configs.csv")
    print("Saved risk_scores_all_configs.csv")

    # Tier shifts across configs
    tier_shifts = _cross_config_tier_shifts(risk_all)
    save_csv(tier_shifts, results / "cross_config_tier_shifts.csv")
    n_volatile = len(tier_shifts)
    n_big_jumps = len(tier_shifts[tier_shifts["tier_range"] >= 2])
    print(f"Saved cross_config_tier_shifts.csv ({n_volatile} volatile, {n_big_jumps} big jumps)")

    # Show interesting examples
    if not tier_shifts.empty:
        print("\n  Notable tier shifts:")
        for _, row in tier_shifts.head(10).iterrows():
            configs = " | ".join(f"{k}={row.get(f'tier_{k}', '?')}" for k in ANALYSIS_CONFIGS)
            print(f"    {row['title_current']}: {configs}")

    # ── Save pivot-distance inputs (top/bottom 10 per zone) ───────────────────
    # Tiebreakers:
    #   high_risk bucket — after risk_score desc, use pct desc (higher exposure = more at risk)
    #   low_risk bucket  — after risk_score asc, use pct asc (lower exposure = safer)
    zone_pivot_rows = []
    for zone in [1, 2, 3, 4, 5]:
        zone_df = primary_df[primary_df["job_zone"].apply(
            lambda z: pd.notna(z) and int(z) == zone
        )].copy()
        if zone_df.empty:
            continue
        top_n = min(PIVOT_N, len(zone_df))
        high_risk = (
            zone_df.sort_values(["risk_score", "pct"], ascending=[False, False])
            .head(top_n)[["title_current", "risk_score", "pct"]]
            .assign(group="high_risk", job_zone=zone)
        )
        low_risk = (
            zone_df.sort_values(["risk_score", "pct"], ascending=[True, True])
            .head(top_n)[["title_current", "risk_score", "pct"]]
            .assign(group="low_risk", job_zone=zone)
        )
        zone_pivot_rows.extend([high_risk, low_risk])
    if zone_pivot_rows:
        pivot_inputs = pd.concat(zone_pivot_rows, ignore_index=True)
        save_csv(pivot_inputs, results / "pivot_distance_inputs.csv")
        print("Saved pivot_distance_inputs.csv (for pivot_distance sub-question)")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig = _risk_distribution_bar(primary_df, ANALYSIS_CONFIG_LABELS[PRIMARY_KEY])
    save_figure(fig, fig_dir / "risk_tier_distribution.png")
    print("  risk_tier_distribution.png")

    fig = _risk_vs_pct_scatter(primary_df, ANALYSIS_CONFIG_LABELS[PRIMARY_KEY])
    save_figure(fig, fig_dir / "risk_vs_pct_scatter.png")
    print("  risk_vs_pct_scatter.png")

    fig = _flag_breakdown_by_score(primary_df)
    save_figure(fig, fig_dir / "flag_breakdown_by_score.png")
    print("  flag_breakdown_by_score.png")

    fig = _cross_config_volatility(risk_all, primary_df)
    save_figure(fig, fig_dir / "cross_config_volatility.png")
    print("  cross_config_volatility.png")

    # Flag activation profile heatmap (tier × flag)
    fig = _tier_flag_profile_heatmap(primary_df)
    save_figure(fig, fig_dir / "tier_flag_profile.png")
    print("  tier_flag_profile.png")

    # Top occupations per tier
    for tier in TIER_ORDER:
        fig = _tier_top_occs_bar(primary_df, tier, top_n=20)
        if fig.data:
            fname = f"tier_top_occs_{tier}.png"
            save_figure(fig, fig_dir / fname)
            print(f"  {fname}")

    # ── Copy key figures ──────────────────────────────────────────────────────
    committed = HERE / "figures"
    committed.mkdir(exist_ok=True)
    key_figs = [
        "risk_tier_distribution.png", "risk_vs_pct_scatter.png",
        "flag_breakdown_by_score.png", "cross_config_volatility.png",
        "tier_flag_profile.png",
    ] + [f"tier_top_occs_{t}.png" for t in TIER_ORDER]
    for fname in key_figs:
        src = fig_dir / fname
        if src.exists():
            shutil.copy2(src, committed / fname)

    # ── PDF ────────────────────────────────────────────────────────────────────
    from lib.utils import generate_pdf
    md_path = HERE / "job_risk_scoring_report.md"
    if md_path.exists():
        generate_pdf(md_path, results / "job_risk_scoring_report.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()
