"""
risk_score_audit — Is the 8-flag risk-score composite doing more than echoing
pct_tasks_affected and known structural features?

Four sections:

  1. Flag-validity diagnostics
       For each of the 8 flags, activation rate split by pct_physical and
       job_zone. Eta-squared (between-bin variance / total variance) per flag
       quantifies how much of the flag is just structural.

  2. SKA mechanicalness
       SKA overall_pct as a function of pct_physical and job_zone. R^2 of a
       structural-only OLS gives a number on "X% of SKA-pct variance is
       structural."

  3. Level-vs-trend independence
       Eight 2x2 contingency tables for four pairings (pct level x pct trend,
       SKA level x SKA trend, pct level x SKA trend, SKA level x pct trend),
       each at median and 75p thresholds. Off-diagonal mass = whether trend
       adds info beyond level.

  4. Flagging variants
       Six candidate "high-exposure" definitions:
         A. pct > 50%
         B. pct > p75
         C. pct > 50% AND outlook in {2, 3}
         D. pct > 50% AND pct trend in top half
         E. Trimmed weighted score (pct, pct-trend, outlook, auto-aug only)
         F. Full 8-flag (current dashboard)
       Per variant: set size, Jaccard with full-8, occs added/dropped vs A.

Run from project root:
    venv/Scripts/python -m analysis.exploratory.risk_score_audit.run
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.config import (
    ANALYSIS_CONFIG_SERIES,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.compute_ska import SKAData, load_ska_data
from lib.utils import COLORS, FONT_FAMILY, save_csv, save_figure, style_figure

from lib.exploratory.job_risk_scoring import (
    EXPOSURE_GATE,
    FLAG_WEIGHTS,
    PCT_ABS_THRESHOLD,
    PRIMARY_KEY,
    TECH_SKILLS_FILE,
    _assign_risk_tier,
    _compute_flags,
    _compute_pct_trend,
    _get_structural_data,
)

HERE = Path(__file__).resolve().parent
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]
PRIMARY_LABEL = ANALYSIS_CONFIG_LABELS[PRIMARY_KEY]

# SKA AI-capability aggregation override.
# compute_ska.py uses 95th percentile across occupations per element.
# This audit uses the mean of the top-N ai_product values instead — a higher,
# stricter "what AI demonstrably can do" floor that matches the dashboard's
# per-row top-10 reference. Set TOP_N_MEAN = None to fall back to p95.
TOP_N_MEAN = 10
SKA_LABEL = f"top-{TOP_N_MEAN} mean" if TOP_N_MEAN else "p95"

# Flag-6 override: replace DWS-outlook-based signal with the new
# emp_change_pct_2024_2034 column from eco_2025. Flag fires when projected
# employment change is negative — a direct employment-trajectory signal that
# avoids the 2-axis outlook+wages tradeoff that confused the original DWS scale.
FLAG6_LABEL = "F6: emp proj < 0"
FLAG6_RULE = "emp_change_pct_2024_2034 < 0"

CONFIG_SUB = (
    f"{PRIMARY_LABEL} | National | freq, auto-aug ON | "
    f"SKA: {SKA_LABEL} | {FLAG6_LABEL}"
)


# ──────────────────────────────────────────────────────────────────────────
# Local SKA computation — overrides compute_ska's p95 with top-N mean.
# Returns a Series of overall_pct (ratio-of-sums of ai_score / occ_score
# across all SKA elements with importance >= 3) keyed by title_current.
# ──────────────────────────────────────────────────────────────────────────

def _compute_ska_overall_pct(
    pct: pd.Series,
    ska_data: SKAData,
    top_n: int = 10,
) -> pd.Series:
    type_map = {
        "skills":    ska_data.skills,
        "abilities": ska_data.abilities,
        "knowledge": ska_data.knowledge,
    }
    rows = []
    for onet_df in type_map.values():
        df = onet_df.copy()
        df["pct"] = df["title"].map(pct)
        df = df.dropna(subset=["pct", "importance", "level"])
        df = df[df["importance"] >= 3].copy()
        df["occ_score"] = df["importance"] * df["level"]
        df["ai_product"] = (df["pct"] / 100.0) * df["occ_score"]
        ai_cap_series = (
            df.groupby("element_name")["ai_product"]
            .apply(lambda s: s.nlargest(top_n).mean())
        )
        df["ai_score"] = df["element_name"].map(ai_cap_series)
        rows.append(df[["title", "occ_score", "ai_score"]])
    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby("title").agg(
        sum_ai=("ai_score", "sum"),
        sum_occ=("occ_score", "sum"),
    )
    overall_pct = (
        grouped["sum_ai"] / grouped["sum_occ"].replace(0, np.nan) * 100.0
    )
    overall_pct.index.name = "title_current"
    return overall_pct


def _compute_ska_trend_topn(config_key: str, ska_data: SKAData, top_n: int = 10) -> pd.Series:
    series = ANALYSIS_CONFIG_SERIES[config_key]
    if len(series) < 2:
        return pd.Series(dtype=float)
    first = get_pct_tasks_affected(series[0])
    last = get_pct_tasks_affected(series[-1])
    delta = (
        _compute_ska_overall_pct(last, ska_data, top_n=top_n)
        - _compute_ska_overall_pct(first, ska_data, top_n=top_n)
    )
    return delta.rename("ska_delta")

FLAG_COLS: list[str] = list(FLAG_WEIGHTS.keys())
FLAG_LABELS: dict[str, str] = {
    "flag1_pct":        "F1: pct > 50%",
    "flag2_ska":        "F2: SKA pct > med",
    "flag3_pct_trend":  "F3: pct trend up",
    "flag4_ska_trend":  "F4: SKA trend up",
    "flag5_job_zone":   "F5: zone in 1-3",
    "flag6_outlook":    FLAG6_LABEL,   # was "F6: outlook in 2-3" — see header for swap
    "flag7_n_software": "F7: software > med",
    "flag8_auto_aug":   "F8: auto-aug > med",
}


def _get_emp_projections() -> pd.Series:
    """Return per-occupation projected employment change pct (2025-2034) from
    eco_2025. Source for the new flag-6 signal.
    """
    from backend.compute import load_eco_raw
    eco = load_eco_raw()
    assert eco is not None, "eco_2025 not loaded"
    col = "emp_change_pct__PROJ_2025_2034__"
    assert col in eco.columns, f"{col} missing from eco_2025"
    proj = (
        eco.groupby("title_current")[col]
        .first()
        .astype(float)
    )
    return proj


# ──────────────────────────────────────────────────────────────────────────
# Shared loader
# ──────────────────────────────────────────────────────────────────────────

def _load_flag_df() -> pd.DataFrame:
    """Build the same flags dataframe job_risk_scoring builds, plus
    pct_physical (joined from get_explorer_occupations).
    """
    print("  Loading structural data...")
    struct = _get_structural_data()

    # n_software
    assert TECH_SKILLS_FILE.exists(), f"Missing {TECH_SKILLS_FILE}"
    tech = pd.read_csv(TECH_SKILLS_FILE)
    struct = struct.merge(
        tech[["title", "n_software"]].rename(columns={"title": "title_current"}),
        on="title_current", how="left",
    )
    struct["n_software"] = struct["n_software"].fillna(0).astype(int)

    # pct_physical and other explorer fields not in _get_structural_data
    from backend.compute import get_explorer_occupations
    occ_extra = pd.DataFrame([
        {
            "title_current": o["title_current"],
            "pct_physical": o.get("pct_physical"),
            "n_tasks": o.get("n_tasks"),
        }
        for o in get_explorer_occupations()
    ])
    struct = struct.merge(occ_extra, on="title_current", how="left")

    print("  Loading SKA base data...")
    ska_data = load_ska_data()

    print("  Computing pct_tasks_affected (primary)...")
    pct = get_pct_tasks_affected(PRIMARY_DATASET)

    print(f"  Computing SKA overall_pct (top-{TOP_N_MEAN} mean)...")
    ska_pct = _compute_ska_overall_pct(pct, ska_data, top_n=TOP_N_MEAN)

    print("  Computing pct trend (first -> last)...")
    pct_delta = _compute_pct_trend(PRIMARY_KEY)

    print(f"  Computing SKA trend (top-{TOP_N_MEAN} mean, first -> last)...")
    ska_delta = _compute_ska_trend_topn(PRIMARY_KEY, ska_data, top_n=TOP_N_MEAN)

    print("  Computing flags...")
    flags_df = _compute_flags(struct, pct, ska_pct, pct_delta, ska_delta)

    # Override flag6 — replace DWS-outlook signal with the new
    # emp_change_pct_2024_2034 < 0 signal. Recompute score and tier so
    # downstream sections see the new composite.
    print(f"  Overriding flag6 with {FLAG6_RULE}...")
    emp_proj = _get_emp_projections()
    flags_df["emp_proj_pct"] = flags_df["title_current"].map(emp_proj)
    flags_df["flag6_outlook"] = (
        flags_df["emp_proj_pct"].fillna(0) < 0
    ).astype(int)
    flags_df["risk_score"] = sum(
        flags_df[col] * weight for col, weight in FLAG_WEIGHTS.items()
    )
    flags_df["risk_tier"] = [
        _assign_risk_tier(score, pct_val)
        for score, pct_val in zip(flags_df["risk_score"], flags_df["pct"])
    ]
    flags_df["exposure_gated"] = (
        (flags_df["risk_score"] >= 8) & (flags_df["pct"] < EXPOSURE_GATE)
    )
    return flags_df


# ──────────────────────────────────────────────────────────────────────────
# Section 1 — Flag-validity diagnostics
# ──────────────────────────────────────────────────────────────────────────

PHYS_BINS = [
    ("0% (no phys)",   0.0,    0.0001),
    ("0-25%",          0.0001, 0.25),
    ("25-50%",         0.25,   0.50),
    ("50-75%",         0.50,   0.75),
    (">75%",           0.75,   1.01),
]


def _phys_bin(p: Any) -> str:
    if p is None or pd.isna(p):
        return "missing"
    p = float(p)
    for label, lo, hi in PHYS_BINS:
        if lo <= p < hi:
            return label
    return "missing"


def _zone_bin(z: Any) -> str:
    if z is None or pd.isna(z):
        return "missing"
    return f"Z{int(z)}"


def _eta_sq(values: np.ndarray, groups: np.ndarray) -> float:
    """Eta-squared = between-group variance / total variance.
    Matches one-way ANOVA's effect size; works for binary outcomes too.
    Returns 0.0 if total variance is zero.
    """
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    total_var = df["v"].var(ddof=0)
    if total_var <= 0:
        return 0.0
    grand_mean = df["v"].mean()
    grp_means = df.groupby("g")["v"].agg(["mean", "size"])
    between = ((grp_means["mean"] - grand_mean) ** 2 * grp_means["size"]).sum() / len(df)
    return float(between / total_var)


def section_1_flag_validity(flags_df: pd.DataFrame, results: Path, fig_dir: Path) -> None:
    print("\n[1/4] Flag-validity diagnostics")

    df = flags_df.copy()
    df["phys_bin"] = df["pct_physical"].apply(_phys_bin)
    df["zone_bin"] = df["job_zone"].apply(_zone_bin)

    # Compute activation-rate matrices (flag x bin)
    phys_order = [b[0] for b in PHYS_BINS]
    zone_order = ["Z1", "Z2", "Z3", "Z4", "Z5"]

    def _matrix(group_col: str, order: list[str]) -> tuple[np.ndarray, list[str]]:
        sub = df[df[group_col].isin(order)]
        present = [b for b in order if b in sub[group_col].unique()]
        mat = np.zeros((len(FLAG_COLS), len(present)))
        for i, fc in enumerate(FLAG_COLS):
            for j, b in enumerate(present):
                m = sub[sub[group_col] == b]
                mat[i, j] = m[fc].mean() * 100 if len(m) else np.nan
        return mat, present

    phys_mat, phys_present = _matrix("phys_bin", phys_order)
    zone_mat, zone_present = _matrix("zone_bin", zone_order)

    # Eta-squared per flag x stratum
    rows = []
    for fc in FLAG_COLS:
        rows.append({
            "flag":         fc,
            "label":        FLAG_LABELS[fc],
            "eta2_phys":    _eta_sq(df[fc].astype(float).values,
                                    df["phys_bin"].astype(str).values),
            "eta2_zone":    _eta_sq(df[fc].astype(float).values,
                                    df["zone_bin"].astype(str).values),
            "eta2_major":   _eta_sq(df[fc].astype(float).values,
                                    df["major"].astype(str).values),
            "activation_rate": float(df[fc].mean()),
        })
    eta_df = pd.DataFrame(rows).sort_values(
        ["eta2_zone", "eta2_phys"], ascending=[False, False]
    )
    save_csv(eta_df, results / "flag_validity_eta2.csv", float_format="%.4f")

    print("  Eta^2 by structural feature (between-bin variance / total variance):")
    print("  Higher = flag more determined by structural feature alone.")
    print(eta_df[["label", "eta2_phys", "eta2_zone", "eta2_major"]].to_string(index=False))

    # Figure: paired heatmaps (flag x phys bins | flag x zone bins)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=("Activation rate by % physical tasks",
                        "Activation rate by job zone"),
        horizontal_spacing=0.18,
    )
    y_labels = [FLAG_LABELS[fc] for fc in FLAG_COLS]
    fig.add_trace(go.Heatmap(
        z=phys_mat, x=phys_present, y=y_labels,
        colorscale=[[0, "#f5f5f0"], [0.5, COLORS["accent"]], [1, COLORS["negative"]]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in r] for r in phys_mat],
        texttemplate="%{text}",
        textfont=dict(size=10, family=FONT_FAMILY),
        showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=zone_mat, x=zone_present, y=y_labels,
        colorscale=[[0, "#f5f5f0"], [0.5, COLORS["accent"]], [1, COLORS["negative"]]],
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in r] for r in zone_mat],
        texttemplate="%{text}",
        textfont=dict(size=10, family=FONT_FAMILY),
        colorbar=dict(title="% with<br>flag active", ticksuffix="%",
                      tickfont=dict(size=10), len=0.7),
        hovertemplate="%{y}<br>%{x}: %{z:.0f}%<extra></extra>",
    ), row=1, col=2)
    style_figure(
        fig,
        "Section 1: Flag Activation by Structural Feature",
        subtitle=f"{CONFIG_SUB} | Cells: % of occupations in stratum where flag is active",
        x_title=None, y_title=None,
        height=600, width=1200, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=180, r=120, t=100, b=60),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "01_flag_activation_by_structure.png")
    print("  -> 01_flag_activation_by_structure.png")

    # Eta-squared bar chart (3 series: phys, zone, major)
    fig2 = go.Figure()
    eta_plot = eta_df.iloc[::-1]  # bottom-up for plotly
    fig2.add_trace(go.Bar(
        y=eta_plot["label"], x=eta_plot["eta2_phys"], orientation="h",
        name="vs. % physical bins", marker=dict(color=COLORS["primary"]),
    ))
    fig2.add_trace(go.Bar(
        y=eta_plot["label"], x=eta_plot["eta2_zone"], orientation="h",
        name="vs. job zone", marker=dict(color=COLORS["secondary"]),
    ))
    fig2.add_trace(go.Bar(
        y=eta_plot["label"], x=eta_plot["eta2_major"], orientation="h",
        name="vs. major (22 sectors)", marker=dict(color=COLORS["accent"]),
    ))
    style_figure(
        fig2,
        "Section 1: How Much of Each Flag Is Just Structural?",
        subtitle=(f"{CONFIG_SUB} | Eta^2 = between-bin variance / total variance | "
                  "Higher = flag is more determined by structure alone"),
        x_title="Eta^2", y_title=None,
        height=500, width=1100, show_legend=True,
    )
    fig2.update_layout(barmode="group", bargap=0.18, bargroupgap=0.08,
                       margin=dict(l=180, r=40, t=100, b=80))
    fig2.update_xaxes(range=[0, max(0.5, eta_df[["eta2_phys", "eta2_zone", "eta2_major"]].values.max() * 1.1)])
    save_figure(fig2, fig_dir / "01b_flag_eta2.png")
    print("  -> 01b_flag_eta2.png")


# ──────────────────────────────────────────────────────────────────────────
# Section 2 — SKA mechanicalness
# ──────────────────────────────────────────────────────────────────────────

def section_2_ska_mechanicalness(flags_df: pd.DataFrame, results: Path, fig_dir: Path) -> None:
    print("\n[2/4] SKA mechanicalness")

    df = flags_df[["title_current", "ska_pct", "pct_physical", "job_zone", "pct"]].copy()
    df = df.dropna(subset=["ska_pct", "pct_physical", "job_zone"])

    # OLS: ska_pct ~ pct_physical + job_zone (linear, no interactions)
    X = np.column_stack([
        np.ones(len(df)),
        df["pct_physical"].astype(float).values,
        df["job_zone"].astype(float).values,
    ])
    y = df["ska_pct"].astype(float).values
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Single-feature R^2 for each
    def _r2_single(x: np.ndarray, y: np.ndarray) -> float:
        Xs = np.column_stack([np.ones(len(x)), x])
        c, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        yh = Xs @ c
        ssr = float(((y - yh) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ssr / sst if sst > 0 else 0.0

    r2_phys = _r2_single(df["pct_physical"].astype(float).values, y)
    r2_zone = _r2_single(df["job_zone"].astype(float).values, y)

    summary = pd.DataFrame([
        {"model": "ska_pct ~ pct_physical",                  "r2": r2_phys},
        {"model": "ska_pct ~ job_zone",                      "r2": r2_zone},
        {"model": "ska_pct ~ pct_physical + job_zone (OLS)", "r2": r2},
        {"model": "OLS intercept",                           "r2": float(coefs[0])},
        {"model": "OLS coef pct_physical",                   "r2": float(coefs[1])},
        {"model": "OLS coef job_zone",                       "r2": float(coefs[2])},
    ])
    save_csv(summary, results / "ska_mechanicalness.csv", float_format="%.4f")

    print(f"  R^2(ska_pct ~ pct_physical):           {r2_phys:.3f}")
    print(f"  R^2(ska_pct ~ job_zone):               {r2_zone:.3f}")
    print(f"  R^2(ska_pct ~ pct_physical + zone):    {r2:.3f}")
    print(f"  OLS slopes: pct_physical {coefs[1]:+.2f}, job_zone {coefs[2]:+.2f}")

    # Figure: scatter (left) + box plot by zone (right)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=("SKA pct vs % physical tasks (color = job zone)",
                        "SKA pct by job zone"),
        horizontal_spacing=0.12,
    )

    zones = sorted(df["job_zone"].dropna().unique())
    palette = ["#3a5f83", "#4a7c6f", "#c05621", "#7b5ea7", "#8b4513"]
    zone_color = {z: palette[int(z) - 1] for z in zones if 1 <= int(z) <= 5}

    # Scatter
    for z in zones:
        sub = df[df["job_zone"] == z]
        fig.add_trace(go.Scatter(
            x=sub["pct_physical"] * 100, y=sub["ska_pct"],
            mode="markers",
            marker=dict(
                color=zone_color.get(z, COLORS["neutral"]),
                size=5, opacity=0.55,
                line=dict(width=0.3, color="white"),
            ),
            name=f"Zone {int(z)}",
            text=sub["title_current"],
            hovertemplate="%{text}<br>%phys: %{x:.0f}%<br>SKA pct: %{y:.0f}%<extra></extra>",
            showlegend=True,
            legendgroup="zones",
        ), row=1, col=1)

    # OLS line over the scatter (predicted ska_pct at mean job_zone)
    mean_zone = df["job_zone"].mean()
    px = np.linspace(0, 1, 50)
    py = coefs[0] + coefs[1] * px + coefs[2] * mean_zone
    fig.add_trace(go.Scatter(
        x=px * 100, y=py, mode="lines",
        line=dict(color="black", width=1.2, dash="dot"),
        name=f"OLS (R²={r2:.2f}, mean zone)",
        hoverinfo="skip", showlegend=True,
    ), row=1, col=1)

    # Box plot (right)
    for z in zones:
        sub = df[df["job_zone"] == z]
        fig.add_trace(go.Box(
            y=sub["ska_pct"], name=f"Zone {int(z)}",
            marker=dict(color=zone_color.get(z, COLORS["neutral"])),
            boxmean=True, showlegend=False,
        ), row=1, col=2)

    style_figure(
        fig,
        "Section 2: How Much of SKA pct Is Structural?",
        subtitle=(f"{CONFIG_SUB} | OLS R^2(ska_pct ~ pct_physical + zone) = {r2:.3f} | "
                  f"single-feature: phys {r2_phys:.2f}, zone {r2_zone:.2f}"),
        x_title=None, y_title=None,
        height=620, width=1300, show_legend=True,
    )
    fig.update_xaxes(title_text="% physical tasks", row=1, col=1, ticksuffix="%")
    fig.update_yaxes(title_text="SKA overall_pct (%)", row=1, col=1, ticksuffix="%")
    fig.update_xaxes(title_text="Job zone", row=1, col=2)
    fig.update_yaxes(title_text=None, row=1, col=2, ticksuffix="%")
    fig.update_layout(margin=dict(l=70, r=40, t=110, b=80))
    save_figure(fig, fig_dir / "02_ska_mechanicalness.png")
    print("  -> 02_ska_mechanicalness.png")


# ──────────────────────────────────────────────────────────────────────────
# Section 3 — Level vs trend independence
# ──────────────────────────────────────────────────────────────────────────

PAIRINGS = [
    ("pct_level vs pct_trend",   "pct",     "pct_delta"),
    ("ska_level vs ska_trend",   "ska_pct", "ska_delta"),
    ("pct_level vs ska_trend",   "pct",     "ska_delta"),
    ("ska_level vs pct_trend",   "ska_pct", "pct_delta"),
]

THRESHOLDS = [("median", 0.50), ("p75", 0.75)]


def section_3_level_vs_trend(flags_df: pd.DataFrame, results: Path, fig_dir: Path) -> None:
    print("\n[3/4] Level vs trend independence")

    rows = []
    panels: dict[tuple[str, str], dict] = {}

    for thr_name, q in THRESHOLDS:
        for pair_name, lvl_col, trd_col in PAIRINGS:
            sub = flags_df[[lvl_col, trd_col]].dropna()
            lvl_thr = sub[lvl_col].quantile(q)
            trd_thr = sub[trd_col].quantile(q)
            lvl_hi = sub[lvl_col] > lvl_thr
            trd_hi = sub[trd_col] > trd_thr
            n11 = int(((lvl_hi) & (trd_hi)).sum())
            n10 = int(((lvl_hi) & (~trd_hi)).sum())
            n01 = int(((~lvl_hi) & (trd_hi)).sum())
            n00 = int(((~lvl_hi) & (~trd_hi)).sum())
            n_total = n11 + n10 + n01 + n00
            off_diag = (n10 + n01) / n_total if n_total else 0.0

            # Phi (Pearson on binary indicators)
            try:
                phi = float(
                    np.corrcoef(lvl_hi.astype(int).values,
                                trd_hi.astype(int).values)[0, 1]
                )
            except Exception:
                phi = float("nan")

            # Of "level high", how many are NOT trend high? — what user asked.
            n_lvl_hi = int(lvl_hi.sum())
            pct_lvl_hi_not_trd_hi = (n10 / n_lvl_hi * 100) if n_lvl_hi else 0.0

            rows.append({
                "pairing": pair_name,
                "threshold": thr_name,
                "lvl_thr": float(lvl_thr),
                "trd_thr": float(trd_thr),
                "n11_lvl_hi_trd_hi": n11,
                "n10_lvl_hi_trd_lo": n10,
                "n01_lvl_lo_trd_hi": n01,
                "n00_lvl_lo_trd_lo": n00,
                "off_diag_pct": off_diag * 100,
                "phi": phi,
                "n_lvl_hi": n_lvl_hi,
                "pct_lvl_hi_not_trd_hi": pct_lvl_hi_not_trd_hi,
            })
            panels[(thr_name, pair_name)] = {
                "matrix": np.array([[n11, n10], [n01, n00]]),
                "phi": phi,
                "off_diag": off_diag * 100,
                "lvl_thr": lvl_thr,
                "trd_thr": trd_thr,
            }

    summary = pd.DataFrame(rows)
    save_csv(summary, results / "level_vs_trend.csv", float_format="%.3f")
    print("  Off-diagonal mass (% of occs where level and trend disagree):")
    show = summary[["pairing", "threshold", "off_diag_pct", "phi",
                    "n_lvl_hi", "pct_lvl_hi_not_trd_hi"]]
    print(show.to_string(index=False))

    # Figure: 4 columns (pairings) x 2 rows (thresholds) of 2x2 heatmaps
    pair_names = [p[0] for p in PAIRINGS]
    thr_names = [t[0] for t in THRESHOLDS]
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{p} ({t})" for t in thr_names for p in pair_names],
        vertical_spacing=0.18, horizontal_spacing=0.06,
    )
    for r_idx, (thr_name, _) in enumerate(THRESHOLDS):
        for c_idx, pair_name in enumerate(pair_names):
            d = panels[(thr_name, pair_name)]
            mat = d["matrix"]
            row_tot = mat.sum()
            pct_mat = mat / row_tot * 100 if row_tot else mat
            # Cell labels: count + share
            text = [
                [f"{mat[i, j]}<br>({pct_mat[i, j]:.0f}%)" for j in range(2)]
                for i in range(2)
            ]
            fig.add_trace(go.Heatmap(
                z=pct_mat, x=["trend hi", "trend lo"], y=["level hi", "level lo"],
                colorscale=[[0, "#f5f5f0"], [0.5, COLORS["primary"]], [1, COLORS["negative"]]],
                zmin=0, zmax=60,
                text=text, texttemplate="%{text}",
                textfont=dict(size=10, family=FONT_FAMILY),
                showscale=False,
                hovertemplate="%{y} & %{x}: %{z:.0f}%<extra></extra>",
            ), row=r_idx + 1, col=c_idx + 1)
            # Annotate phi + off-diag at the bottom of each panel.
            # Plotly's first subplot uses xref="x" (not "x1"); subsequent are x2, x3, ...
            n_panel = r_idx * 4 + c_idx + 1
            xref = "x domain" if n_panel == 1 else f"x{n_panel} domain"
            yref = "y domain" if n_panel == 1 else f"y{n_panel} domain"
            fig.add_annotation(
                x=0.5, y=-0.35, xref=xref, yref=yref,
                text=(f"phi={d['phi']:.2f} | off-diag={d['off_diag']:.0f}%"),
                showarrow=False,
                font=dict(size=10, color=COLORS["neutral"], family=FONT_FAMILY),
            )

    style_figure(
        fig,
        "Section 3: Does Trend Add Info Beyond Level?",
        subtitle=(f"{CONFIG_SUB} | 2x2 contingencies (level x trend) at median (top row) "
                  "and p75 (bottom) | cell = count (% of total) | phi = Pearson on indicators"),
        x_title=None, y_title=None,
        height=720, width=1500, show_legend=False,
    )
    fig.update_layout(margin=dict(l=80, r=40, t=110, b=120))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10), autorange="reversed")
    save_figure(fig, fig_dir / "03_level_vs_trend.png")
    print("  -> 03_level_vs_trend.png")


# ──────────────────────────────────────────────────────────────────────────
# Section 4 — Flagging variants
# ──────────────────────────────────────────────────────────────────────────

def _build_variant_sets(flags_df: pd.DataFrame) -> dict[str, set[str]]:
    """Nine candidate "high-exposure" definitions. Returns dict of name -> set of titles."""
    df = flags_df.copy()

    # Variant E (trimmed weighted score): pct + pct-trend + outlook + auto-aug,
    # weights {flag1=2, flag3=1, flag6=1, flag8=1}, max=5, "high" = score >= 4
    # AND pct >= EXPOSURE_GATE.
    trimmed_score = (
        df["flag1_pct"]        * 2 +
        df["flag3_pct_trend"]  * 1 +
        df["flag6_outlook"]    * 1 +
        df["flag8_auto_aug"]   * 1
    )
    trimmed_high = (trimmed_score >= 4) & (df["pct"] >= EXPOSURE_GATE)

    pct50 = 50.0
    pct_med = df["pct"].median()
    pct75 = df["pct"].quantile(0.75)
    pct_delta_med = df["pct_delta"].median()

    # Variants G/H/I: 4-condition intersect (pct + SKA + pct trend + outlook).
    # Same SKA / pct-trend / outlook conditions; only the pct threshold varies.
    def _quad(pct_thr: float) -> set[str]:
        mask = (
            (df["pct"] > pct_thr)
            & (df["flag2_ska"] == 1)         # SKA pct > median
            & (df["pct_delta"] > pct_delta_med)  # pct trend > median
            & (df["flag6_outlook"] == 1)     # outlook in {2, 3}
        )
        return set(df.loc[mask, "title_current"])

    sets: dict[str, set[str]] = {
        "A: pct > 50%":
            set(df.loc[df["pct"] > pct50, "title_current"]),
        "B: pct > p75":
            set(df.loc[df["pct"] > pct75, "title_current"]),
        "C: pct > 50% & emp proj<0":
            set(df.loc[(df["pct"] > pct50) & (df["flag6_outlook"] == 1),
                       "title_current"]),
        "D: pct > 50% & pct trend top half":
            set(df.loc[(df["pct"] > pct50) & (df["pct_delta"] > pct_delta_med),
                       "title_current"]),
        "E: trimmed score (4 flags, score>=4)":
            set(df.loc[trimmed_high, "title_current"]),
        "F: full 8-flag (score>=8 + gate)":
            set(df.loc[df["risk_tier"] == "high", "title_current"]),
        "G: pct>50% & SKA>med & trend>med & emp proj<0":
            _quad(pct50),
        "H: pct>med & SKA>med & trend>med & emp proj<0":
            _quad(pct_med),
        "I: pct>p75 & SKA>med & trend>med & emp proj<0":
            _quad(pct75),
    }
    return sets


def section_4_flagging_variants(flags_df: pd.DataFrame, results: Path, fig_dir: Path) -> None:
    print("\n[4/4] Flagging variants")

    sets = _build_variant_sets(flags_df)
    df = flags_df.set_index("title_current")
    df["workers_affected"] = df["pct"] / 100.0 * df["emp_nat"]
    df["wages_affected"] = df["workers_affected"] * df["wage_nat"]

    # Per-variant summary
    rows = []
    for name, occ_set in sets.items():
        sub = df.loc[df.index.isin(occ_set)]
        rows.append({
            "variant": name,
            "n_high": len(occ_set),
            "share_of_923_pct": len(occ_set) / len(df) * 100,
            "workers_affected": float(sub["workers_affected"].sum()),
            "wages_affected": float(sub["wages_affected"].sum()),
            "avg_pct": float(sub["pct"].mean()) if len(sub) else 0.0,
            "share_zone_1_3_pct": float(
                sub["job_zone"].apply(
                    lambda z: pd.notna(z) and int(z) in {1, 2, 3}
                ).mean() * 100
            ) if len(sub) else 0.0,
            "share_emp_proj_neg_pct": float(
                (sub["emp_proj_pct"].fillna(0) < 0).mean() * 100
            ) if len(sub) else 0.0,
        })
    summary = pd.DataFrame(rows)
    save_csv(summary, results / "variant_summary.csv", float_format="%.2f")
    print("  Per-variant summary:")
    print(summary.to_string(index=False))

    # Jaccard 6x6
    names = list(sets.keys())
    jacc = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            a, b = sets[n1], sets[n2]
            if not a and not b:
                jacc[i, j] = 1.0
                continue
            inter = len(a & b)
            uni = len(a | b)
            jacc[i, j] = inter / uni if uni else 0.0
    jacc_df = pd.DataFrame(jacc, index=names, columns=names).round(3)
    save_csv(jacc_df.reset_index().rename(columns={"index": "variant"}),
             results / "variant_jaccard.csv", float_format="%.3f")

    # Added/dropped vs A and vs F
    base_A = sets["A: pct > 50%"]
    base_F = sets["F: full 8-flag (score>=8 + gate)"]
    diff_rows = []
    for name, occ_set in sets.items():
        if name == "A: pct > 50%":
            continue
        added_vs_A = sorted(occ_set - base_A)
        dropped_vs_A = sorted(base_A - occ_set)
        added_vs_F = sorted(occ_set - base_F)
        dropped_vs_F = sorted(base_F - occ_set)
        diff_rows.append({
            "variant": name,
            "n_added_vs_A": len(added_vs_A),
            "n_dropped_vs_A": len(dropped_vs_A),
            "n_added_vs_F": len(added_vs_F),
            "n_dropped_vs_F": len(dropped_vs_F),
            "examples_added_vs_A": " | ".join(added_vs_A[:5]),
            "examples_dropped_vs_A": " | ".join(dropped_vs_A[:5]),
        })
    save_csv(pd.DataFrame(diff_rows), results / "variant_set_diff.csv")

    # Figure 4A: bar chart of n_high per variant + workers (right axis labels)
    summary_plot = summary.copy().iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary_plot["variant"], x=summary_plot["n_high"],
        orientation="h",
        marker=dict(color=COLORS["primary"], line=dict(width=0)),
        text=[
            f"{int(r['n_high'])} occs ({r['share_of_923_pct']:.0f}%)<br>"
            f"{r['workers_affected']/1e6:.1f}M wkrs | "
            f"avg pct {r['avg_pct']:.0f}% | "
            f"zone1-3 {r['share_zone_1_3_pct']:.0f}% | "
            f"emp proj<0 {r['share_emp_proj_neg_pct']:.0f}%"
            for _, r in summary_plot.iterrows()
        ],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False,
    ))
    style_figure(
        fig, "Section 4: How Many Occupations Each Variant Calls 'High Exposure'",
        subtitle=f"{CONFIG_SUB} | Annotations show set size, workforce, and structural composition",
        x_title="Number of occupations in 'high' set", y_title=None,
        height=720, width=1350, show_legend=False,
    )
    fig.update_layout(margin=dict(l=380, r=480, t=100, b=80), bargap=0.30)
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "04a_variant_set_sizes.png")
    print("  -> 04a_variant_set_sizes.png")

    # Figure 4B: Jaccard heatmap
    fig2 = go.Figure(go.Heatmap(
        z=jacc, x=names, y=names,
        colorscale=[[0, "#f5f5f0"], [0.5, COLORS["secondary"]], [1, COLORS["primary"]]],
        zmin=0, zmax=1,
        text=[[f"{v:.2f}" for v in r] for r in jacc],
        texttemplate="%{text}",
        textfont=dict(size=11, family=FONT_FAMILY),
        colorbar=dict(title="Jaccard", tickfont=dict(size=10), len=0.7),
        hovertemplate="%{y}<br>vs %{x}: %{z:.2f}<extra></extra>",
    ))
    style_figure(
        fig2, "Section 4: Variant Overlap (Jaccard)",
        subtitle="1.00 = identical sets | 0.00 = disjoint",
        x_title=None, y_title=None,
        height=820, width=1150, show_legend=False,
    )
    fig2.update_layout(margin=dict(l=400, r=40, t=100, b=240))
    fig2.update_xaxes(tickangle=-30, tickfont=dict(size=10, family=FONT_FAMILY))
    fig2.update_yaxes(autorange="reversed", tickfont=dict(size=10, family=FONT_FAMILY))
    save_figure(fig2, fig_dir / "04b_variant_jaccard.png")
    print("  -> 04b_variant_jaccard.png")


# ──────────────────────────────────────────────────────────────────────────
# Section 5 — Focused set for Part 3 chart variations
#
# 56 occupations meeting (pct > 50% AND pct trend > median AND emp proj < 0).
# Subset of those (43) ALSO meet SKA pct > median ("gated"); the other 13
# pass when the SKA filter is dropped ("added"). Four chart variations
# produced for the paper to choose from.
# ──────────────────────────────────────────────────────────────────────────

GATED_COLOR = COLORS["primary"]      # slate blue — strict 4-condition tier
ADDED_COLOR = COLORS["secondary"]    # teal green — added when SKA filter dropped


def _build_focused_set(flags_df: pd.DataFrame) -> pd.DataFrame:
    pct_delta_med = flags_df["pct_delta"].median()
    base_mask = (
        (flags_df["pct"] > 50.0)
        & (flags_df["pct_delta"] > pct_delta_med)
        & (flags_df["flag6_outlook"] == 1)
    )
    sub = flags_df.loc[base_mask].copy()
    sub["ska_gated"] = (sub["flag2_ska"] == 1).astype(int)
    sub["tier"] = sub["ska_gated"].map({1: "SKA-gated (G, n=43)",
                                         0: "Added without SKA (n=13)"})
    sub["workers_affected"] = sub["pct"] / 100.0 * sub["emp_nat"]
    sub["wages_affected"] = sub["workers_affected"] * sub["wage_nat"]
    sub["major_short"] = sub["major"].str.replace(" Occupations", "", regex=False)
    return sub


def section_5_focused_set(flags_df: pd.DataFrame, results: Path, fig_dir: Path) -> None:
    print("\n[5/5] Focused set for Part 3 (G + 13 SKA-released)")
    sub = _build_focused_set(flags_df)
    print(f"  Total: {len(sub)} occs ({(sub['ska_gated']==1).sum()} SKA-gated, "
          f"{(sub['ska_gated']==0).sum()} added without SKA)")

    save_csv(
        sub.sort_values(["ska_gated", "pct"], ascending=[False, False])[
            ["title_current", "major_short", "job_zone", "emp_proj_pct",
             "pct", "ska_pct", "pct_delta",
             "emp_nat", "wage_nat", "workers_affected", "wages_affected",
             "ska_gated", "tier"]
        ],
        results / "focused_set.csv",
        float_format="%.3f",
    )

    cap = (
        f"{CONFIG_SUB} | Focused set: pct > 50% AND pct trend > median "
        f"AND emp proj < 0 (n={len(sub)}) | "
        f"Color = whether SKA pct > median also fires"
    )

    # ── Variation 5a: horizontal bar sorted by pct desc ──────────────────
    s5a = sub.sort_values("pct", ascending=True).reset_index(drop=True)  # plotly bottom-up
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=s5a["title_current"], x=s5a["pct"], orientation="h",
        marker=dict(
            color=[GATED_COLOR if g == 1 else ADDED_COLOR for g in s5a["ska_gated"]],
            line=dict(width=0),
        ),
        text=[
            f"{r['pct']:.0f}%  |  emp proj {r['emp_proj_pct']:+.1f}%  |  "
            f"trend +{r['pct_delta']:.1f}pp  |  zone {int(r['job_zone'])}"
            for _, r in s5a.iterrows()
        ],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
    ))
    # Manual legend (two phantom traces with showlegend)
    for label, color in [
        ("SKA-gated (G, n=43)", GATED_COLOR),
        ("Added without SKA (n=13)", ADDED_COLOR),
    ]:
        fig.add_trace(go.Bar(
            y=[None], x=[0], orientation="h", name=label,
            marker=dict(color=color), showlegend=True,
        ))
    style_figure(
        fig,
        "Section 5a: Focused 56 — sorted by % tasks affected",
        subtitle=cap,
        x_title="% tasks affected", y_title=None,
        height=1200, width=1500, show_legend=True,
    )
    fig.update_layout(
        margin=dict(l=380, r=480, t=110, b=140),
        bargap=0.20,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "05a_focused_by_pct.png")
    print("  -> 05a_focused_by_pct.png")

    # ── Variation 5b: horizontal bar sorted by emp_proj asc ──────────────
    # Most-declining at top — flips the framing to "BLS says these jobs are
    # going away first" rather than "these are the most AI-exposed."
    s5b = sub.sort_values("emp_proj_pct", ascending=False).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=s5b["title_current"], x=s5b["emp_proj_pct"], orientation="h",
        marker=dict(
            color=[GATED_COLOR if g == 1 else ADDED_COLOR for g in s5b["ska_gated"]],
            line=dict(width=0),
        ),
        text=[
            f"{r['emp_proj_pct']:+.1f}%  |  pct {r['pct']:.0f}%  |  "
            f"trend +{r['pct_delta']:.1f}pp  |  zone {int(r['job_zone'])}"
            for _, r in s5b.iterrows()
        ],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
    ))
    for label, color in [
        ("SKA-gated (G, n=43)", GATED_COLOR),
        ("Added without SKA (n=13)", ADDED_COLOR),
    ]:
        fig.add_trace(go.Bar(
            y=[None], x=[0], orientation="h", name=label,
            marker=dict(color=color), showlegend=True,
        ))
    style_figure(
        fig,
        "Section 5b: Focused 56 — sorted by BLS employment projection (most declining first)",
        subtitle=cap,
        x_title="BLS projected % change in employment, 2024-2034",
        y_title=None,
        height=1200, width=1500, show_legend=True,
    )
    fig.update_layout(
        margin=dict(l=380, r=480, t=110, b=140),
        bargap=0.20,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "05b_focused_by_emp_proj.png")
    print("  -> 05b_focused_by_emp_proj.png")

    # ── Variation 5c: scatter emp_proj × pct, sized by workers ────────────
    fig = go.Figure()
    for tier_label, tier_val, color in [
        ("SKA-gated (G, n=43)", 1, GATED_COLOR),
        ("Added without SKA (n=13)", 0, ADDED_COLOR),
    ]:
        m = sub["ska_gated"] == tier_val
        fig.add_trace(go.Scatter(
            x=sub.loc[m, "emp_proj_pct"], y=sub.loc[m, "pct"],
            mode="markers",
            marker=dict(
                size=np.sqrt(sub.loc[m, "workers_affected"].clip(lower=1)) / 50.0 + 6,
                color=color, opacity=0.75,
                line=dict(width=0.6, color="white"),
            ),
            name=tier_label,
            text=sub.loc[m, "title_current"]
                 + " | " + sub.loc[m, "major_short"]
                 + " | zone " + sub.loc[m, "job_zone"].astype(int).astype(str),
            customdata=np.column_stack([
                sub.loc[m, "ska_pct"], sub.loc[m, "pct_delta"],
                sub.loc[m, "workers_affected"],
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "emp proj: %{x:+.1f}%<br>"
                "pct: %{y:.0f}%<br>"
                "SKA pct: %{customdata[0]:.0f}%<br>"
                "trend: +%{customdata[1]:.1f}pp<br>"
                "workers affected: %{customdata[2]:,.0f}<extra></extra>"
            ),
        ))
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["neutral"], line_width=1)
    style_figure(
        fig,
        "Section 5c: Focused 56 — emp projection vs. % tasks affected",
        subtitle=cap + " | Marker size = workers affected | Vertical line at emp proj = 0",
        x_title="BLS projected % change in employment, 2024-2034",
        y_title="% tasks affected",
        height=720, width=1300, show_legend=True,
    )
    fig.update_xaxes(ticksuffix="%", zeroline=False)
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(margin=dict(l=70, r=40, t=110, b=120))
    save_figure(fig, fig_dir / "05c_focused_scatter.png")
    print("  -> 05c_focused_scatter.png")

    # ── Variation 5d: stacked bar by major ────────────────────────────────
    by_major = (
        sub.groupby(["major_short", "ska_gated"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={1: "gated", 0: "added"})
    )
    by_major["total"] = by_major.get("gated", 0) + by_major.get("added", 0)
    by_major = by_major.sort_values("total", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=by_major.index, x=by_major.get("gated", 0), orientation="h",
        name="SKA-gated (G, n=43)",
        marker=dict(color=GATED_COLOR, line=dict(width=0)),
        text=[str(int(v)) if v > 0 else "" for v in by_major.get("gated", 0)],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white", family=FONT_FAMILY),
    ))
    fig.add_trace(go.Bar(
        y=by_major.index, x=by_major.get("added", 0), orientation="h",
        name="Added without SKA (n=13)",
        marker=dict(color=ADDED_COLOR, line=dict(width=0)),
        text=[str(int(v)) if v > 0 else "" for v in by_major.get("added", 0)],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white", family=FONT_FAMILY),
    ))
    style_figure(
        fig,
        "Section 5d: Focused 56 — major occupational composition",
        subtitle=cap + " | Stacked: SKA-gated (left) + added without SKA (right)",
        x_title="Number of occupations", y_title=None,
        height=580, width=1100, show_legend=True,
    )
    fig.update_layout(
        barmode="stack", bargap=0.25,
        margin=dict(l=300, r=40, t=110, b=140),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "05d_focused_by_major.png")
    print("  -> 05d_focused_by_major.png")

    # ── Variation 5e: paper candidate ─────────────────────────────────────
    # 5b's sort but flipped right (length = |emp proj|), with pct encoded
    # via bar color, SKA tier via leading marker shape (filled = gated,
    # open = added), and major composition shown in a right-side sidebar.
    # A leading colored chip at x=0 ensures pct color is visible even when
    # the bar itself is tiny (e.g., −0.1% emp proj).
    s5e = sub.copy()
    s5e["abs_emp"] = s5e["emp_proj_pct"].abs()
    # Plotly horizontal bars render bottom-up; sort ascending by abs so the
    # largest decline is at the top of the visual.
    s5e = s5e.sort_values("abs_emp", ascending=True).reset_index(drop=True)

    pct_min = float(s5e["pct"].min())
    pct_max = float(s5e["pct"].max())
    PCT_SCALE = [[0, "#c4d9d2"], [1, "#0a2e25"]]

    fig = go.Figure()

    # Bars: length = |emp proj|, color = pct
    fig.add_trace(go.Bar(
        y=s5e["title_current"], x=s5e["abs_emp"], orientation="h",
        marker=dict(
            color=s5e["pct"].values,
            colorscale=PCT_SCALE,
            cmin=pct_min, cmax=pct_max,
            line=dict(width=0),
        ),
        text=[
            f"{r['emp_proj_pct']:+.1f}% emp proj  |  {r['major_short']}  |  "
            f"pct {r['pct']:.0f}%  |  trend +{r['pct_delta']:.1f}pp  |  zone {int(r['job_zone'])}"
            for _, r in s5e.iterrows()
        ],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
        hovertemplate="<b>%{y}</b><br>emp proj: -%{x:.1f}%<extra></extra>",
    ))

    # Leading chips at x=0 — fixed-size markers so SKA tier reads even on
    # tiny bars. Filled diamond = SKA-gated, open square = added.
    # Filled in slate / open in teal so the chips encode tier even before
    # color-shape is parsed.
    for tier_val, symbol, line_color, fill_color, name in [
        (1, "diamond",     COLORS["text"], COLORS["primary"],   "SKA-gated tier (filled, n=43)"),
        (0, "square-open", COLORS["secondary"], "white",         "Added without SKA (open, n=13)"),
    ]:
        m = s5e["ska_gated"] == tier_val
        fig.add_trace(go.Scatter(
            x=[0] * int(m.sum()), y=s5e.loc[m, "title_current"],
            mode="markers",
            marker=dict(
                symbol=symbol,
                color=fill_color,
                size=14,
                line=dict(width=2.0, color=line_color),
            ),
            name=name,
            hoverinfo="skip",
            showlegend=True,
        ))

    # No separate colorbar widget — the right-side text annotation on each
    # bar shows "pct X%" so the exact value is always readable, and the
    # color shading is a visual scan cue (lighter = lower pct, darker =
    # higher). Subtitle calls this out.

    # Major composition sidebar
    by_major = sub["major_short"].value_counts()
    by_major_gated = sub[sub["ska_gated"] == 1]["major_short"].value_counts()
    by_major_added = sub[sub["ska_gated"] == 0]["major_short"].value_counts()
    rows = []
    for m in by_major.index:
        g = int(by_major_gated.get(m, 0))
        a = int(by_major_added.get(m, 0))
        if a > 0:
            rows.append(f"{m}<br>  {g + a} ({g} gated, +{a} added)")
        else:
            rows.append(f"{m}<br>  {g} ({g} gated)")
    sidebar = (
        "<b>Major occ composition (n=56)</b><br>"
        + "<br>".join(rows)
    )
    fig.add_annotation(
        text=sidebar,
        xref="paper", yref="paper",
        x=1.005, y=0.62,
        showarrow=False, align="left",
        xanchor="left", yanchor="top",
        font=dict(size=10, color=COLORS["text"], family=FONT_FAMILY),
        bordercolor=COLORS["grid"], borderwidth=1, borderpad=10,
        bgcolor="white",
    )

    style_figure(
        fig,
        "Section 5e: Focused 56 — sorted by BLS employment projection (most declining first)",
        subtitle=(
            f"{CONFIG_SUB} | Bar length = |% projected employment decline 2024-2034| | "
            f"Bar shade: lighter = lower pct tasks affected (min {pct_min:.0f}%), "
            f"darker = higher (max {pct_max:.0f}%) | Marker shape = SKA tier"
        ),
        x_title="BLS projected employment decline (% over 2024-2034, absolute value)",
        y_title=None,
        height=1400, width=1800, show_legend=True,
    )
    fig.update_layout(
        margin=dict(l=400, r=600, t=110, b=160),
        bargap=0.20,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.08,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"], ticksuffix="%"),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, family=FONT_FAMILY)),
    )
    save_figure(fig, fig_dir / "05e_focused_paper_candidate.png")
    print("  -> 05e_focused_paper_candidate.png")

    # ── Variation 5f: paper main-text candidate ───────────────────────────
    # 5e but restricted to the 43 SKA-gated occs only — single tier, so
    # no chip markers, no major sidebar. Bars going right (length =
    # |emp proj decline|), color = pct tasks affected, per-row annotation.
    s5f = sub[sub["ska_gated"] == 1].copy()
    s5f["abs_emp"] = s5f["emp_proj_pct"].abs()
    s5f = s5f.sort_values("abs_emp", ascending=True).reset_index(drop=True)

    pct_min_f = float(s5f["pct"].min())
    pct_max_f = float(s5f["pct"].max())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=s5f["title_current"], x=s5f["abs_emp"], orientation="h",
        marker=dict(
            color=s5f["pct"].values,
            colorscale=PCT_SCALE,
            cmin=pct_min_f, cmax=pct_max_f,
            line=dict(width=0),
        ),
        text=[
            f"{r['emp_proj_pct']:+.1f}% emp proj  |  {r['major_short']}  |  "
            f"pct {r['pct']:.0f}%  |  trend +{r['pct_delta']:.1f}pp  |  zone {int(r['job_zone'])}"
            for _, r in s5f.iterrows()
        ],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False, showlegend=False,
        hovertemplate="<b>%{y}</b><br>emp proj: -%{x:.1f}%<extra></extra>",
    ))
    style_figure(
        fig,
        "Section 5f: SKA-gated focused 43 — sorted by BLS employment projection",
        subtitle=(
            f"{CONFIG_SUB} | "
            f"pct > 50% AND pct trend > median AND emp proj < 0 AND SKA pct > median (n=43) | "
            f"Bar length = |% projected employment decline 2024-2034| | "
            f"Bar shade: lighter = lower pct ({pct_min_f:.0f}%), "
            f"darker = higher ({pct_max_f:.0f}%)"
        ),
        x_title="BLS projected employment decline (% over 2024-2034, absolute value)",
        y_title=None,
        height=1100, width=1500, show_legend=False,
    )
    fig.update_layout(
        margin=dict(l=400, r=520, t=110, b=120),
        bargap=0.20,
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], ticksuffix="%")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "05f_main_text_candidate.png")
    print("  -> 05f_main_text_candidate.png")

    # ── Variation 5g: job zone composition (mirror of 5d) ─────────────────
    # Same stacked-bar pattern as 5d but on job zones instead of majors.
    # Pairs with 5f for the appendix sectoral/zone summary.
    by_zone = (
        sub.groupby([sub["job_zone"].astype(int), "ska_gated"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={1: "gated", 0: "added"})
    )
    by_zone["total"] = by_zone.get("gated", 0) + by_zone.get("added", 0)
    by_zone = by_zone.sort_index(ascending=False)  # Z5 at top, Z1 at bottom
    zone_labels = [f"Job zone {z}" for z in by_zone.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=zone_labels, x=by_zone.get("gated", 0), orientation="h",
        name="SKA-gated (G, n=43)",
        marker=dict(color=GATED_COLOR, line=dict(width=0)),
        text=[str(int(v)) if v > 0 else "" for v in by_zone.get("gated", 0)],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white", family=FONT_FAMILY),
    ))
    fig.add_trace(go.Bar(
        y=zone_labels, x=by_zone.get("added", 0), orientation="h",
        name="Added without SKA (n=13)",
        marker=dict(color=ADDED_COLOR, line=dict(width=0)),
        text=[str(int(v)) if v > 0 else "" for v in by_zone.get("added", 0)],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(size=11, color="white", family=FONT_FAMILY),
    ))
    style_figure(
        fig,
        "Section 5g: Focused 56 — job zone composition",
        subtitle=cap + " | Stacked: SKA-gated (left) + added without SKA (right)",
        x_title="Number of occupations", y_title=None,
        height=480, width=1000, show_legend=True,
    )
    fig.update_layout(
        barmode="stack", bargap=0.30,
        margin=dict(l=160, r=40, t=110, b=140),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, family=FONT_FAMILY))
    save_figure(fig, fig_dir / "05g_focused_by_zone.png")
    print("  -> 05g_focused_by_zone.png")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    fig_dir = results / "figures"

    print("=" * 64)
    print("risk_score_audit — Are the 8 flags doing real work?")
    print("=" * 64)

    flags_df = _load_flag_df()
    save_csv(
        flags_df[
            ["title_current", "major", "job_zone", "outlook", "emp_proj_pct",
             "pct_physical", "n_software", "auto_avg", "pct", "ska_pct",
             "pct_delta", "ska_delta",
             *FLAG_COLS, "risk_score", "risk_tier", "exposure_gated"]
        ],
        results / "flags_dataframe.csv",
        float_format="%.3f",
    )
    print(f"\nLoaded {len(flags_df)} occupations.")

    section_1_flag_validity(flags_df, results, fig_dir)
    section_2_ska_mechanicalness(flags_df, results, fig_dir)
    section_3_level_vs_trend(flags_df, results, fig_dir)
    section_4_flagging_variants(flags_df, results, fig_dir)
    section_5_focused_set(flags_df, results, fig_dir)

    print("\n" + "=" * 64)
    print("Done. Figures in results/figures/, CSVs in results/.")
    print("=" * 64)


if __name__ == "__main__":
    main()
