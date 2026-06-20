"""
deepdive_state_signal — Where's the real state-level signal?

The archived state_clusters bucket established the headline finding:
pct_tasks_affected is essentially uniform across states (~36% everywhere).
The clusterings it ran on top of that produced low-ARI groupings — the lenses
disagree because each one is measuring something different, but no single
candidate chart emerged from that work as a paper-ready visual.

This deep dive isolates five candidate state-level signals and computes them
side-by-side under a single, consistent methodology so we can compare spread
and pick the strongest one(s) to surface as a paper figure.

Five signals (all employment-weighted, all using the all_confirmed config):

  A. Wages-affected per worker         — dollar stakes per state employee
  B. Focused-set workforce share       — % of state emp in the 43 SKA-gated
                                          high-exposure occs (from
                                          audit_risk_score's paper-aligned set)
  C. SKA-weighted exposure             — emp-weighted avg of overall_pct
                                          (AI share of SKA need), an
                                          importance/level-weighted alternative
                                          to pct_tasks_affected
  D. Wage premium of exposed work      — avg wage among exposed / avg wage
                                          overall — is exposure tilted high or
                                          low on the state's wage distribution?
  E. High-zone share of exposed work   — % of state's workers_affected sitting
                                          in O*NET zones 4–5 (bachelor's+)

For comparison, the script also reports pct_tasks_affected employment-weighted
at the state level — the flat baseline.

Restricted to 50 states + DC. Territories (PR/VI/GU) excluded per project lead.

Run from project root:
    venv/Scripts/python -m lib.exploratory.state_signal
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.config import (
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.compute_ska import load_ska_data
from lib.utils import (
    COLORS,
    FONT_FAMILY,
    save_csv,
    save_figure,
    style_figure,
)
from lib.paper_config import (
    METRIC_COLORS,
    PAPER_PALETTE,
    PAPER_W,
    PAPER_H,
    LABEL_FS,
    TICK_FS,
    ANNOT_FS,
    fmt_wages,
    style_paper_figure,
)
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent

PRIMARY_KEY = "all_confirmed"
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]
PRIMARY_LABEL = ANALYSIS_CONFIG_LABELS[PRIMARY_KEY]

# 50 states + DC (territories excluded by request).
STATE_GEOS: list[str] = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma",
    "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny",
    "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx",
    "ut", "vt", "va", "wa", "wv", "wi", "wy",
]

# Plot styling.
BAR_COLOR_DEFAULT = COLORS["primary"]
BAR_COLOR_HIGH = COLORS["accent"]
BAR_COLOR_LOW = COLORS["secondary"]
N_HIGHLIGHT = 5  # top/bottom N annotated per chart


# ─────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────

def _load_occ_table() -> pd.DataFrame:
    """Build the per-occupation feature table that all 5 signals consume.

    Columns:
      title_current, pct, job_zone, physical_share
      For each state geo: emp_{geo}, wage_{geo}
    """
    from backend.compute import load_eco_raw

    eco = load_eco_raw()
    assert eco is not None and not eco.empty, "eco_2025 not available"
    assert "title_current" in eco.columns, "eco_2025 missing title_current"
    assert "job_zone" in eco.columns, "eco_2025 missing job_zone"
    assert "physical" in eco.columns, "eco_2025 missing physical"

    # pct_tasks_affected per occupation (national, freq, auto-aug on).
    pct = get_pct_tasks_affected(PRIMARY_DATASET).rename("pct")
    pct.index.name = "title_current"

    # One row per occupation. Take first non-null for emp/wage cols (they're
    # repeated per task row) and the modal job_zone.
    agg: dict[str, str] = {"job_zone": "first"}
    for geo in STATE_GEOS:
        emp_col = f"emp_tot_{geo}_2025"
        wage_col = f"a_med_{geo}_2025"
        if emp_col in eco.columns:
            agg[emp_col] = "first"
        if wage_col in eco.columns:
            agg[wage_col] = "first"

    occ = eco.groupby("title_current").agg(agg).reset_index()

    # Physical share per occupation (fraction of an occ's task rows flagged
    # physical=True). Used as a cross-check feature.
    eco["_phys_int"] = eco["physical"].fillna(False).astype(bool).astype(int)
    phys = eco.groupby("title_current")["_phys_int"].mean().rename("physical_share")
    occ = occ.merge(phys.reset_index(), on="title_current", how="left")

    occ = occ.merge(pct.reset_index(), on="title_current", how="left")

    # Drop unrated occupations (pct null). Drop unrated rows so weighted
    # averages don't NaN out — these get zero contribution to numerators
    # but should also drop out of denominators (handled per-metric below).
    occ["pct"] = occ["pct"].astype(float)
    occ = occ.dropna(subset=["pct"]).copy()

    # Standardize emp/wage column dtypes.
    for geo in STATE_GEOS:
        emp_col, wage_col = f"emp_tot_{geo}_2025", f"a_med_{geo}_2025"
        if emp_col in occ.columns:
            occ[emp_col] = pd.to_numeric(occ[emp_col], errors="coerce").fillna(0.0)
        if wage_col in occ.columns:
            occ[wage_col] = pd.to_numeric(occ[wage_col], errors="coerce")
        # rename for clarity in downstream computations
        occ = occ.rename(columns={emp_col: f"emp_{geo}", wage_col: f"wage_{geo}"})

    occ["job_zone"] = pd.to_numeric(occ["job_zone"], errors="coerce")

    return occ


def _load_ska_overall_pct() -> pd.Series:
    """SKA overall_pct per occupation under all_confirmed pct."""
    from lib.compute_ska import compute_ska
    ska_data = load_ska_data()
    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    result = compute_ska(pct, ska_data)
    # occ_gaps has title_current; overall_pct lives there.
    g = result.occ_gaps
    assert "title_current" in g.columns, "compute_ska occ_gaps missing title_current"
    assert "overall_pct" in g.columns, "compute_ska occ_gaps missing overall_pct"
    return g.set_index("title_current")["overall_pct"].astype(float)


def _load_focused_set() -> set[str]:
    """The 43-occ SKA-gated focused set from audit_risk_score.

    Same set the paper's risk_score_5f chart uses.
    """
    from lib.exploratory.risk_score import (
        _load_flag_df, _build_focused_set,
    )
    flags_df = _load_flag_df()
    sub = _build_focused_set(flags_df)
    gated = sub[sub["ska_gated"] == 1]["title_current"].tolist()
    return set(gated)


# ─────────────────────────────────────────────────────────────────────
# Per-state metric computations
# ─────────────────────────────────────────────────────────────────────

def compute_state_metrics(
    occ: pd.DataFrame,
    ska_overall_pct: pd.Series,
    focused_set: set[str],
) -> pd.DataFrame:
    """For each state, compute the 5 candidate signals (+ baselines)."""
    rows: list[dict] = []

    ska_lookup = ska_overall_pct.to_dict()
    is_focused = occ["title_current"].isin(focused_set)
    is_zone_45 = occ["job_zone"].isin([4, 5])

    for geo in STATE_GEOS:
        emp_col = f"emp_{geo}"
        wage_col = f"wage_{geo}"
        if emp_col not in occ.columns or wage_col not in occ.columns:
            continue

        emp = occ[emp_col].astype(float).fillna(0.0)
        wage = occ[wage_col].astype(float)
        pct = occ["pct"].astype(float)

        # Workers / wages affected per occupation in this state.
        wa = (pct / 100.0) * emp
        wages_aff = wa * wage.fillna(0.0)

        total_emp = emp.sum()
        total_wage_pool = (emp * wage.fillna(0.0)).sum()
        total_wa = wa.sum()
        total_wages_aff = wages_aff.sum()

        if total_emp <= 0:
            continue

        # ── Baseline: emp-weighted pct_tasks_affected ───────────────────
        pct_emp_wtd = total_wa / total_emp * 100.0

        # ── % of state wages exposed (parallel to pct_emp_wtd) ──────────
        pct_wages_exposed = (
            total_wages_aff / total_wage_pool * 100.0
            if total_wage_pool > 0 else np.nan
        )

        # ── A. Wages-affected per worker (full state denominator) ───────
        wages_per_worker = total_wages_aff / total_emp

        # ── B. Focused-set workforce share ──────────────────────────────
        focused_emp = emp[is_focused].sum()
        focused_share = focused_emp / total_emp * 100.0

        # ── C. SKA-weighted exposure (emp-weighted overall_pct) ─────────
        ska = occ["title_current"].map(ska_lookup).astype(float)
        ska_mask = ska.notna() & (emp > 0)
        if ska_mask.sum() > 0:
            ska_weighted = (ska[ska_mask] * emp[ska_mask]).sum() / emp[ska_mask].sum()
        else:
            ska_weighted = np.nan

        # ── D. Wage premium of exposed work ─────────────────────────────
        # Numerator: avg wage of an exposed worker = Σ wage * wa / Σ wa
        # Denominator: avg wage in state = Σ wage * emp / Σ emp
        wage_mask = wage.notna()
        if total_wa > 0 and wage_mask.sum() > 0:
            avg_wage_exposed = (
                (wage[wage_mask].fillna(0.0) * wa[wage_mask]).sum()
                / wa[wage_mask].sum()
            )
            avg_wage_all = (
                (wage[wage_mask].fillna(0.0) * emp[wage_mask]).sum()
                / emp[wage_mask].sum()
            )
            wage_premium = (
                avg_wage_exposed / avg_wage_all if avg_wage_all > 0 else np.nan
            )
        else:
            wage_premium = np.nan
            avg_wage_exposed = np.nan
            avg_wage_all = np.nan

        # ── E. High-zone share of exposed work ──────────────────────────
        if total_wa > 0:
            zone45_wa = wa[is_zone_45].sum()
            zone45_share = zone45_wa / total_wa * 100.0
        else:
            zone45_share = np.nan

        rows.append({
            "geo": geo,
            "total_emp": total_emp,
            "total_workers_affected": total_wa,
            "total_wages_affected": total_wages_aff,
            # Baseline (flat)
            "pct_emp_wtd": pct_emp_wtd,
            "pct_wages_exposed": pct_wages_exposed,
            # A
            "wages_per_worker": wages_per_worker,
            # B
            "focused_share_pct": focused_share,
            # C
            "ska_emp_wtd": ska_weighted,
            # D
            "avg_wage_exposed": avg_wage_exposed,
            "avg_wage_all": avg_wage_all,
            "wage_premium": wage_premium,
            # E
            "zone45_share_pct": zone45_share,
        })

    df = pd.DataFrame(rows).sort_values("geo").reset_index(drop=True)
    return df


def compute_spread_summary(state_df: pd.DataFrame) -> pd.DataFrame:
    """For each candidate signal, compute spread vs. the flat baseline.

    Reports min, max, max/min ratio, p10-p90 range, and coefficient of
    variation. The higher these, the stronger the state-level signal.
    """
    metrics = {
        "pct_emp_wtd": "Baseline: % of state emp exposed",
        "pct_wages_exposed": "% of state wages exposed",
        "wages_per_worker": "A. Wages affected per worker ($)",
        "focused_share_pct": "B. Focused-set share of state emp (%)",
        "ska_emp_wtd": "C. SKA-weighted exposure (emp-wtd overall_pct)",
        "wage_premium": "D. Wage premium of exposed (×)",
        "zone45_share_pct": "E. Zone 4-5 share of exposed (%)",
    }
    rows = []
    for col, label in metrics.items():
        s = state_df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "metric": label,
            "min":   s.min(),
            "p10":   s.quantile(0.10),
            "median": s.median(),
            "p90":   s.quantile(0.90),
            "max":   s.max(),
            "min_state": state_df.loc[s.idxmin(), "geo"].upper(),
            "max_state": state_df.loc[s.idxmax(), "geo"].upper(),
            "max_over_min": float(s.max() / s.min()) if s.min() > 0 else np.nan,
            "p90_over_p10": float(s.quantile(0.90) / s.quantile(0.10))
                            if s.quantile(0.10) > 0 else np.nan,
            "cv_pct": float(s.std() / s.mean() * 100) if s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def _bar_with_highlights(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    subtitle: str,
    x_title: str,
    *,
    text_fmt: str = "{:.1f}",
    ref_line: Optional[float] = None,
    ref_label: str = "U.S. average",
    high_is_good: bool = True,
    n_highlight: int = N_HIGHLIGHT,
) -> go.Figure:
    """Horizontal bar chart over states, top/bottom N highlighted.

    high_is_good only controls coloring of top/bottom labels — the chart
    itself just shows numeric spread.
    """
    plot = df[["geo", value_col]].dropna().copy()
    plot = plot.sort_values(value_col, ascending=True).reset_index(drop=True)
    plot["geo_label"] = plot["geo"].str.upper()

    # Color top/bottom N.
    n = len(plot)
    top_idx = set(plot.tail(n_highlight).index.tolist())
    bot_idx = set(plot.head(n_highlight).index.tolist())
    bar_colors = []
    for i in range(n):
        if i in top_idx:
            bar_colors.append(BAR_COLOR_HIGH if high_is_good else BAR_COLOR_LOW)
        elif i in bot_idx:
            bar_colors.append(BAR_COLOR_LOW if high_is_good else BAR_COLOR_HIGH)
        else:
            bar_colors.append(BAR_COLOR_DEFAULT)

    text = [text_fmt.format(v) for v in plot[value_col]]

    fig = go.Figure(go.Bar(
        x=plot[value_col],
        y=plot["geo_label"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=text,
        textposition="outside",
        textfont=dict(size=10, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
    ))

    style_figure(
        fig, title,
        subtitle=subtitle,
        x_title=x_title,
        show_legend=False,
        height=1200, width=900,
    )
    fig.update_layout(
        margin=dict(l=60, r=110, t=100, b=60),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        bargap=0.18,
    )

    if ref_line is not None:
        fig.add_vline(
            x=ref_line,
            line=dict(color=COLORS["neutral"], width=1.5, dash="dash"),
            annotation=dict(
                text=f"{ref_label}: {text_fmt.format(ref_line)}",
                font=dict(size=10, color=COLORS["neutral"]),
                yref="paper", y=1.0, yanchor="bottom",
                showarrow=False,
            ),
        )

    return fig


def _build_spread_panel(spread_df: pd.DataFrame) -> go.Figure:
    """Dot-and-line chart: each metric on its own row, normalized to its own
    median so we can visually compare relative spread across signals."""
    rows = []
    for _, r in spread_df.iterrows():
        med = r["median"]
        if med == 0 or pd.isna(med):
            continue
        rows.append({
            "metric": r["metric"],
            "rel_min": r["min"] / med,
            "rel_max": r["max"] / med,
            "rel_p10": r["p10"] / med,
            "rel_p90": r["p90"] / med,
            "max_over_min": r["max_over_min"],
            "cv_pct": r["cv_pct"],
        })
    s = pd.DataFrame(rows)
    s = s.iloc[::-1].reset_index(drop=True)  # plot top→bottom

    fig = go.Figure()
    for _, r in s.iterrows():
        # min-max range (light)
        fig.add_trace(go.Scatter(
            x=[r["rel_min"], r["rel_max"]],
            y=[r["metric"], r["metric"]],
            mode="lines",
            line=dict(color=COLORS["grid"], width=8),
            showlegend=False,
            hoverinfo="skip",
        ))
        # p10-p90 range (dark)
        fig.add_trace(go.Scatter(
            x=[r["rel_p10"], r["rel_p90"]],
            y=[r["metric"], r["metric"]],
            mode="lines",
            line=dict(color=COLORS["primary"], width=8),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Median marker.
        fig.add_trace(go.Scatter(
            x=[1.0], y=[r["metric"]],
            mode="markers+text",
            marker=dict(color=COLORS["accent"], size=11),
            text=[f"max/min {r['max_over_min']:.2f}× · CV {r['cv_pct']:.1f}%"],
            textposition="middle right",
            textfont=dict(size=10, color=COLORS["neutral"], family=FONT_FAMILY),
            showlegend=False,
            hoverinfo="skip",
        ))

    style_figure(
        fig,
        "How much do the 5 candidate state signals actually vary?",
        subtitle="Each row normalized to its own median (vertical line at 1.0). "
                 "Light bar = min–max · Dark bar = p10–p90 · Annotations: max/min ratio and CV.",
        x_title="Ratio to median (1.0 = median)",
        show_legend=False,
        height=600, width=1200,
    )
    fig.update_layout(
        margin=dict(l=420, r=300, t=110, b=70),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
    )
    fig.add_vline(x=1.0, line=dict(color=COLORS["neutral"], width=1, dash="dot"))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Paper-style side-by-side panel: Baseline · A · B  (preview of how this
# would look if it landed in the paper next to major_categories.png).
# ─────────────────────────────────────────────────────────────────────

def _build_side_by_side(state_df: pd.DataFrame, focused_n: int) -> go.Figure:
    """Two-panel paper-style chart: % of state employment exposed |
    focused-set share of state employment. The B panel deliberately
    flips the ranking — knowledge-economy states (DC, CA, MA, WA) top
    the left panel but bottom the right panel."""
    plot = state_df.dropna(
        subset=["pct_emp_wtd", "focused_share_pct"]
    ).copy()
    plot = plot.sort_values("pct_emp_wtd", ascending=False).reset_index(drop=True)

    # Plotly: bottom-up traces, so reverse.
    geos_r    = list(reversed(plot["geo"].str.upper().tolist()))
    emp_r     = list(reversed(plot["pct_emp_wtd"].tolist()))
    focused_r = list(reversed(plot["focused_share_pct"].tolist()))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "% of State Employment Exposed",
            f"% of State Employment in Focused-Set ({focused_n} Occupations)",
        ],
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )

    # Left panel — workers gold.
    fig.add_trace(go.Bar(
        y=geos_r, x=emp_r, orientation="h",
        marker=dict(color=METRIC_COLORS["workers"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in emp_r],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="% emp exposed: %{x:.2f}%<extra></extra>",
    ), row=1, col=1)

    # Right panel — wages green (the "focused-set" framing reads as a
    # structural composition metric; using a distinct color keeps the
    # two panels from collapsing visually).
    fig.add_trace(go.Bar(
        y=geos_r, x=focused_r, orientation="h",
        marker=dict(color=METRIC_COLORS["wages"], line=dict(width=0)),
        text=[f"{v:.1f}%" for v in focused_r],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="% emp in focused set: %{x:.2f}%<extra></extra>",
    ), row=1, col=2)

    n = len(plot)
    height = max(PAPER_H + 200, n * 30 + 220)

    style_paper_figure(
        fig,
        "AI Exposure by State — Broad Exposure vs. Focused-Set Concentration",
        subtitle=(
            "Left: Σ workers_affected ÷ Σ total state emp. "
            f"Right: Σ emp in the {focused_n}-occupation SKA-gated focused set ÷ Σ total state emp. "
            "States sorted by left panel descending — note how the right panel reverses for the "
            f"top knowledge-economy states. 50 states + DC · {PRIMARY_LABEL}."
        ),
        height=height,
        width=PAPER_W,
        margin=dict(l=40, r=80, t=150, b=110),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        ticksuffix="%",
    )
    fig.update_xaxes(title=dict(text="% of state employment exposed",       font=dict(size=LABEL_FS - 4)), row=1, col=1)
    fig.update_xaxes(title=dict(text="% of state employment in focused set", font=dict(size=LABEL_FS - 4)), row=1, col=2)

    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_yaxes(
        title=dict(text="State", font=dict(size=LABEL_FS - 2)),
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        row=1, col=1,
    )

    # Build the set of dynamic panel-title strings we expect to find so
    # the font upsize only touches our titles (not subtitle annotations).
    panel_titles = {
        "% of State Employment Exposed",
        f"% of State Employment in Focused-Set ({focused_n} Occupations)",
    }
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_titles:
            ann.font = dict(size=LABEL_FS - 2, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    fig.update_layout(bargap=0.28)
    return fig


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    figs_dir = HERE / "figures"
    figs_dir.mkdir(exist_ok=True)

    print(f"deepdive_state_signal: primary config = {PRIMARY_LABEL} "
          f"({PRIMARY_DATASET})")
    print("  Loading occupation feature table...")
    occ = _load_occ_table()
    print(f"    {len(occ)} occupations with pct_tasks_affected.")

    print("  Computing SKA overall_pct per occupation...")
    ska_overall = _load_ska_overall_pct()
    print(f"    {len(ska_overall)} occs with SKA score.")

    print("  Loading focused set (audit_risk_score SKA-gated 43)...")
    focused = _load_focused_set()
    print(f"    {len(focused)} occupations in focused set.")

    print("  Computing per-state metrics for 50 states + DC...")
    state_df = compute_state_metrics(occ, ska_overall, focused)
    print(f"    {len(state_df)} state rows.")

    save_csv(state_df, results / "state_metrics.csv", float_format="%.4f")

    print("  Computing spread summary...")
    spread = compute_spread_summary(state_df)
    save_csv(spread, results / "spread_summary.csv", float_format="%.4f")
    print(spread.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────
    print("\n  Building figures...")
    figures: list[tuple[str, go.Figure]] = []

    # Baseline: emp-weighted pct (flat, for comparison).
    nat_pct_emp = float(
        (state_df["total_workers_affected"].sum()
         / state_df["total_emp"].sum()) * 100.0
    )
    fig_baseline = _bar_with_highlights(
        state_df, "pct_emp_wtd",
        title="Baseline — Employment-weighted % Tasks Affected by State",
        subtitle=f"{PRIMARY_LABEL} · 50 states + DC · flat ~36% confirms the archived finding",
        x_title="% tasks affected (emp-weighted)",
        text_fmt="{:.1f}%",
        ref_line=nat_pct_emp,
        n_highlight=N_HIGHLIGHT,
    )
    figures.append(("00_baseline_pct_emp_wtd.png", fig_baseline))

    # A. Wages per worker.
    nat_wages_per_worker = float(
        state_df["total_wages_affected"].sum()
        / state_df["total_emp"].sum()
    )
    fig_a = _bar_with_highlights(
        state_df, "wages_per_worker",
        title="A — AI-Exposed Wages per State Worker",
        subtitle=f"{PRIMARY_LABEL} · Σ wages_affected ÷ Σ total state employment · {len(state_df)} states + DC",
        x_title="Wages affected per worker ($)",
        text_fmt="${:,.0f}",
        ref_line=nat_wages_per_worker,
    )
    figures.append(("01_wages_per_worker.png", fig_a))

    # B. Focused-set share.
    nat_focused = float(
        sum(
            occ[occ["title_current"].isin(focused)][f"emp_{g}"].sum()
            for g in STATE_GEOS if f"emp_{g}" in occ.columns
        )
        / sum(
            occ[f"emp_{g}"].sum()
            for g in STATE_GEOS if f"emp_{g}" in occ.columns
        )
        * 100.0
    )
    fig_b = _bar_with_highlights(
        state_df, "focused_share_pct",
        title="B — Focused-Set Share of State Employment",
        subtitle=f"% of state emp in the {len(focused)} SKA-gated focused-set occs (pct>50 · trend>med · emp proj<0 · SKA>med)",
        x_title="% of state employment",
        text_fmt="{:.2f}%",
        ref_line=nat_focused,
        ref_label="50-state aggregate",
        high_is_good=False,  # high = more workers in vulnerable set
    )
    figures.append(("02_focused_share.png", fig_b))

    # C. SKA-weighted exposure.
    ska_lookup = ska_overall.to_dict()
    occ_with_ska = occ.copy()
    occ_with_ska["ska"] = occ_with_ska["title_current"].map(ska_lookup)
    valid = occ_with_ska.dropna(subset=["ska"])
    nat_emp_ska = sum(
        (valid["ska"] * valid[f"emp_{g}"]).sum()
        for g in STATE_GEOS if f"emp_{g}" in valid.columns
    ) / sum(
        valid[f"emp_{g}"].sum()
        for g in STATE_GEOS if f"emp_{g}" in valid.columns
    )
    fig_c = _bar_with_highlights(
        state_df, "ska_emp_wtd",
        title="C — Employment-Weighted SKA AI-Share-of-Need by State",
        subtitle=f"Avg overall_pct (AI capability ÷ occupation's SKA need × 100), weighted by state employment",
        x_title="SKA AI-share-of-need (%)",
        text_fmt="{:.1f}%",
        ref_line=float(nat_emp_ska),
        ref_label="50-state aggregate",
    )
    figures.append(("03_ska_emp_wtd.png", fig_c))

    # D. Wage premium of exposed.
    fig_d = _bar_with_highlights(
        state_df, "wage_premium",
        title="D — Wage Premium of AI-Exposed Work by State",
        subtitle="avg wage among workers_affected ÷ avg wage among all workers · >1.0 = exposure tilts high-wage",
        x_title="Wage premium (×)",
        text_fmt="{:.3f}",
        ref_line=1.0,
        ref_label="parity (1.000)",
    )
    figures.append(("04_wage_premium.png", fig_d))

    # E. High-zone share of exposed.
    nat_zone45 = float(
        sum(
            ((occ["pct"] / 100.0) * occ[f"emp_{g}"])[occ["job_zone"].isin([4, 5])].sum()
            for g in STATE_GEOS if f"emp_{g}" in occ.columns
        ) / sum(
            ((occ["pct"] / 100.0) * occ[f"emp_{g}"]).sum()
            for g in STATE_GEOS if f"emp_{g}" in occ.columns
        ) * 100.0
    )
    fig_e = _bar_with_highlights(
        state_df, "zone45_share_pct",
        title="E — Share of Exposed Workforce in Job Zones 4–5",
        subtitle="% of workers_affected in zones 4–5 (bachelor's+) — knowledge-tilt of each state's exposure",
        x_title="% of workers_affected in zones 4-5",
        text_fmt="{:.1f}%",
        ref_line=nat_zone45,
        ref_label="50-state aggregate",
    )
    figures.append(("05_zone45_share.png", fig_e))

    # Spread panel.
    fig_spread = _build_spread_panel(spread)
    figures.append(("06_spread_panel.png", fig_spread))

    # Paper-style side-by-side: % emp exposed vs. focused-set share.
    fig_sxs = _build_side_by_side(state_df, focused_n=len(focused))
    figures.append(("07_side_by_side_paper_preview.png", fig_sxs))

    # Save all.
    for fname, fig in figures:
        out_path = results / "figures" / fname
        save_figure(fig, out_path)
        shutil.copy(out_path, figs_dir / fname)
        print(f"    -> {fname}")

    # Quick stdout summary so the user can eyeball winners in the log.
    print("\nState-signal headlines:")
    print(f"  Baseline pct (emp-wtd) range: "
          f"{state_df['pct_emp_wtd'].min():.1f}% – {state_df['pct_emp_wtd'].max():.1f}% "
          f"(max/min {state_df['pct_emp_wtd'].max() / state_df['pct_emp_wtd'].min():.2f}×)")
    print(f"  % wages exposed range:        "
          f"{state_df['pct_wages_exposed'].min():.1f}% – {state_df['pct_wages_exposed'].max():.1f}% "
          f"(max/min {state_df['pct_wages_exposed'].max() / state_df['pct_wages_exposed'].min():.2f}×)")
    print(f"  A wages/worker:               "
          f"${state_df['wages_per_worker'].min():,.0f} – ${state_df['wages_per_worker'].max():,.0f} "
          f"(max/min {state_df['wages_per_worker'].max() / state_df['wages_per_worker'].min():.2f}×)")
    print(f"  B focused-set share:          "
          f"{state_df['focused_share_pct'].min():.2f}% – {state_df['focused_share_pct'].max():.2f}% "
          f"(max/min {state_df['focused_share_pct'].max() / state_df['focused_share_pct'].min():.2f}×)")
    print(f"  C SKA emp-wtd:                "
          f"{state_df['ska_emp_wtd'].min():.1f}% – {state_df['ska_emp_wtd'].max():.1f}% "
          f"(max/min {state_df['ska_emp_wtd'].max() / state_df['ska_emp_wtd'].min():.2f}×)")
    print(f"  D wage premium:               "
          f"{state_df['wage_premium'].min():.3f} – {state_df['wage_premium'].max():.3f} "
          f"(max/min {state_df['wage_premium'].max() / state_df['wage_premium'].min():.3f}×)")
    print(f"  E zone 4-5 share:             "
          f"{state_df['zone45_share_pct'].min():.1f}% – {state_df['zone45_share_pct'].max():.1f}% "
          f"(max/min {state_df['zone45_share_pct'].max() / state_df['zone45_share_pct'].min():.2f}×)")

    print("\ndeepdive_state_signal: done.")


if __name__ == "__main__":
    main()
