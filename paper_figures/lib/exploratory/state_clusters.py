"""
deepdive_state_clusters — Group 50 states + DC into AI-exposure clusters.

The paper's state_exposure_at_risk.png ranks every state on two axes (broad
employment exposed, focused-set share). Useful but visually flat — 51 bars
in two columns. Alice asked whether we could collapse states into a small
number of named groups and show the result as both a recolored bar chart
and a US map.

This deep dive does exactly that. Initial scoping with three features
(pct_emp_wtd, wages_per_worker, focused_share_pct) revealed that
wages_per_worker correlates r ≈ 0.92 with % workers exposed — same axis,
just expressed in dollars — so wages_per_worker was dropped entirely to
keep the analysis on its two genuinely independent axes:

  CLUSTERING features:
    1. pct_emp_wtd        — share of state employment in AI-exposed work
                             ("% workforce exposed")
    2. focused_share_pct  — share of state emp in the 38-occ set with
                             High AI Exposure & Negative Emp Projection
                             (the same set the paper's risk_score_5f
                             chart visualizes; SKA-gated)

DC is treated as a labeled outlier — its broad exposure (45.9%) is so
extreme that including it in the z-scoring distorts the rest of the
structure. So we cluster on 50 states and visualize DC separately.

All features come from `deepdive_state_signal.compute_state_metrics()`
under the `all_confirmed` config (AEI Both + Micro 2026-02-12).

Method:
  - Z-score the 2 clustering features so distances are scale-free.
  - Ward hierarchical clustering → dendrogram → pick k from the largest
    relative jump in merge distance within k ∈ [2, 7].
  - K-means at the same k as a robustness cross-check; report Adjusted
    Rand Index between the two labelings.
  - Cluster names describe each cluster's position on each axis using
    the long-form paper language ("Mid Workforce Exposed / Highest in
    High AI Exp & <0 Emp Proj").

Run from project root:
    venv/Scripts/python -m lib.exploratory.state_clusters
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from lib.config import (
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ensure_results_dir,
)
from lib.utils import (
    CATEGORY_PALETTE,
    COLORS,
    FONT_FAMILY,
    save_csv,
    save_figure,
    style_figure,
)
from lib.paper_config import (
    ANNOT_FS,
    LABEL_FS,
    PAPER_H,
    PAPER_PALETTE,
    PAPER_W,
    TICK_FS,
    style_paper_figure,
)

HERE = Path(__file__).resolve().parent

PRIMARY_KEY = "all_confirmed"
PRIMARY_LABEL = ANALYSIS_CONFIG_LABELS[PRIMARY_KEY]
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]

CLUSTER_FEATURES: list[str] = [
    "pct_emp_wtd",
    "focused_share_pct",
]
ALL_FEATURES: list[str] = list(CLUSTER_FEATURES)

# Long-form display names — match the paper's risk_score_5f chart title
# language so the same concept reads consistently across paper figures.
FEATURE_LABELS: dict[str, str] = {
    "pct_emp_wtd":       "% State Workforce Exposed",
    "focused_share_pct": "% in High AI Exposure & Negative Emp Proj Occs",
}
# Compact aliases for axis labels / cluster names where the long form
# would overrun.
FEATURE_LABELS_SHORT: dict[str, str] = {
    "pct_emp_wtd":       "% Workforce Exposed",
    "focused_share_pct": "% in High AI Exp & <0 Emp Proj Occs",
}

# Candidate k range. Outside this we don't even consider.
# Set to [2, 7] because we cluster on 50 non-outlier states (DC excluded —
# see OUTLIER_GEOS) and the natural k is 3 there.
K_MIN, K_MAX = 2, 7

# State codes excluded from the clustering algorithm and shown as labeled
# outliers in the figures. DC is excluded because its broad-exposure score
# is so extreme (45.9% vs the next-highest 38.9%) that it inflates the
# z-scoring std and forces the algorithm to find finer-grained distinctions
# among the remaining states than the data actually supports. See the
# robustness section of state_clusters_report.md for the full reasoning.
OUTLIER_GEOS: list[str] = ["dc"]
OUTLIER_CLUSTER_ID: int = -1  # sentinel cluster id reserved for outliers
OUTLIER_COLOR: str = "#c05621"  # accent orange, distinct from the blue gradient

# Blue gradient applied to non-outlier clusters in order of "exposure
# severity," defined as the rank sum (broad_rank + atrisk_rank) where
# rank 1 = highest on each axis. Lowest rank sum = darkest blue.
# The palette has enough entries to cover up to 5 non-outlier clusters
# (we expect 3 with k=3 on 50 states, but leave room).
BLUE_GRADIENT: list[str] = [
    "#1f3a5f",  # darkest — worst exposure score (sum 3)
    "#5a8aab",  # mid
    "#b8cee0",  # light
    "#dbe7f1",  # very light
    "#f0f4f8",  # almost white
]


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def _load_state_features() -> pd.DataFrame:
    """Build the 51 × 3 state feature table.

    Tries the deepdive_state_signal cached CSV first (fast); falls back to
    recomputing if the CSV isn't there. Either way it returns a frame with
    columns: geo, pct_emp_wtd, focused_share_pct.
    """
    sig_csv = (
        HERE.parent / "deepdive_state_signal"
        / "results" / "state_metrics.csv"
    )
    if sig_csv.exists():
        print(f"  Loaded cached state metrics from {sig_csv}.")
        df = pd.read_csv(sig_csv)
    else:
        print("  Cached state_metrics.csv not found — recomputing from upstream.")
        from lib.exploratory.state_signal import (
            _load_occ_table,
            _load_ska_overall_pct,
            _load_focused_set,
            compute_state_metrics,
        )
        occ = _load_occ_table()
        ska = _load_ska_overall_pct()
        focused = _load_focused_set()
        df = compute_state_metrics(occ, ska, focused)

    # Restrict to the cluster + descriptive features + geo.
    keep = ["geo"] + ALL_FEATURES
    missing = [c for c in keep if c not in df.columns]
    assert not missing, f"state_metrics missing columns: {missing}"
    df = df[keep].dropna().reset_index(drop=True).copy()
    df["geo"] = df["geo"].str.lower()
    df["geo_label"] = df["geo"].str.upper()
    return df


# ─────────────────────────────────────────────────────────────────────
# Correlation matrix
# ─────────────────────────────────────────────────────────────────────

def _build_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    labels = [FEATURE_LABELS[c] for c in corr.columns]
    z = corr.values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#3a5f83"],
            [0.5, "#f7f7f4"],
            [1.0, "#c05621"],
        ],
        zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, family=FONT_FAMILY, color=COLORS["text"]),
        showscale=True,
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=14,
            len=0.65,
        ),
        hovertemplate="%{x}<br>%{y}<br>r = %{z:.3f}<extra></extra>",
    ))

    style_figure(
        fig,
        "Feature Correlations (Pearson r, 51 states + DC)",
        subtitle="Wages-per-worker correlates r=0.92 with % workforce exposed — same axis in dollars. "
                 "Clustering uses only the two independent axes: % workforce exposed and "
                 "% in High AI Exp & <0 Emp Proj occupations. Wage axis dropped entirely.",
        height=560, width=720,
        show_legend=False,
    )
    fig.update_layout(margin=dict(l=200, r=120, t=110, b=140))
    fig.update_xaxes(tickangle=-25, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


# ─────────────────────────────────────────────────────────────────────
# Hierarchical (Ward) — dendrogram + automatic k selection
# ─────────────────────────────────────────────────────────────────────

def _pick_k_from_linkage(Z: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> tuple[int, dict[int, float]]:
    """Find k ∈ [k_min, k_max] with the largest jump in merge distance.

    The linkage matrix's third column (Z[:, 2]) is the sequence of merge
    distances in ascending order. The merge that takes us from k clusters
    to k - 1 happens at Z[-(k-1), 2]. A "natural" break at k means the
    merge that *would* collapse cluster k+1 into k has a much smaller
    distance than the merge that collapses k into k-1.

    Score: ratio = dist_at(k → k-1) / dist_at(k+1 → k). Higher = bigger
    jump = more reason to stop at k.
    """
    distances = Z[:, 2]
    # Map: dist_at[k] = merge distance going from k clusters down to k-1.
    dist_at = {k: float(distances[-(k - 1)]) for k in range(2, len(distances) + 1)}
    ratios: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        below = dist_at.get(k, np.nan)            # k → k-1
        above = dist_at.get(k + 1, np.nan)        # k+1 → k
        if above and above > 0 and not np.isnan(below):
            ratios[k] = below / above
        else:
            ratios[k] = np.nan
    # Pick the k with the largest ratio (ignore NaNs).
    best_k = max(ratios, key=lambda k: (ratios[k] if not np.isnan(ratios[k]) else -np.inf))
    return best_k, ratios


def _build_dendrogram(
    Z: np.ndarray,
    labels: list[str],
    chosen_k: int,
) -> go.Figure:
    """Ward dendrogram with a horizontal cut line at the chosen k."""
    # scipy's dendrogram in 'no_plot' mode gives us the icoord/dcoord lists.
    ddata = dendrogram(Z, no_plot=True, labels=labels)
    icoord = np.array(ddata["icoord"])
    dcoord = np.array(ddata["dcoord"])
    leaf_labels = ddata["ivl"]

    fig = go.Figure()
    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=COLORS["primary"], width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Cut line: at the merge distance for k → k-1.
    distances = Z[:, 2]
    cut_height = float(distances[-(chosen_k - 1)])
    # Place between the merge for k → k-1 and (k+1) → k so it visibly slices
    # exactly into chosen_k branches.
    if chosen_k + 1 - 1 <= len(distances):
        cut_height_above = float(distances[-chosen_k])
        cut_height = (cut_height + cut_height_above) / 2
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, x1=1, y0=cut_height, y1=cut_height,
        line=dict(color=COLORS["accent"], width=2, dash="dash"),
    )
    fig.add_annotation(
        text=f"Cut for k = {chosen_k}",
        font=dict(size=12, color=COLORS["accent"]),
        xref="paper", x=1.0, xanchor="right",
        yref="y", y=cut_height, yanchor="bottom",
        showarrow=False,
    )

    # Leaf positions sit at 5, 15, 25, ... in icoord units.
    n = len(leaf_labels)
    tickvals = [5 + i * 10 for i in range(n)]

    style_figure(
        fig,
        f"Ward Dendrogram of U.S. States (k = {chosen_k} chosen by largest merge-distance jump)",
        subtitle="Bottom-up merges of states on 2 standardized features (% workforce exposed, "
                 "% in High AI Exp & <0 Emp Proj occs). DC excluded as an outlier; cut at k = chosen.",
        x_title="State",
        y_title="Ward merge distance",
        height=600, width=1400,
        show_legend=False,
    )
    fig.update_layout(margin=dict(l=70, r=40, t=110, b=100))
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=leaf_labels,
        tickfont=dict(size=10, family=FONT_FAMILY),
        tickangle=-90,
        showgrid=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=COLORS["grid"],
        tickfont=dict(size=11, family=FONT_FAMILY),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Cluster naming
# ─────────────────────────────────────────────────────────────────────

def _rank_word(rank: int, n: int) -> str:
    """Convert a 1-indexed rank within n clusters to 'Highest'/'Mid'/'Lowest'.

    For n=3: rank 1 -> 'Highest', 2 -> 'Mid', 3 -> 'Lowest'.
    For n=2: rank 1 -> 'Higher', 2 -> 'Lower'.
    Larger n falls back to 'Rank {i}'.
    """
    if n <= 1:
        return "Only"
    if n == 2:
        return ["Higher", "Lower"][rank - 1]
    if n == 3:
        return ["Highest", "Mid", "Lowest"][rank - 1]
    # 4+ clusters — use ordinal-ish descriptors that still imply ordering.
    if rank == 1:
        return "Highest"
    if rank == n:
        return "Lowest"
    return f"Mid (#{rank})"


def _build_cluster_profile(state_df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Compute the per-cluster centroid table + rank columns + rank-sum score.

    Index is cluster_id. Columns: n_states, pct_emp_wtd, focused_share_pct,
    workforce_rank, vuln_rank, rank_sum.
    rank=1 means highest on that axis. rank_sum = workforce_rank + vuln_rank;
    lowest rank_sum = "worst exposed" cluster.
    """
    centroids = state_df.groupby(cluster_col).agg(
        n_states=("geo", "count"),
        pct_emp_wtd=("pct_emp_wtd", "mean"),
        focused_share_pct=("focused_share_pct", "mean"),
    )
    centroids["workforce_rank"] = centroids["pct_emp_wtd"].rank(
        ascending=False, method="dense"
    ).astype(int)
    centroids["vuln_rank"] = centroids["focused_share_pct"].rank(
        ascending=False, method="dense"
    ).astype(int)
    centroids["rank_sum"] = centroids["workforce_rank"] + centroids["vuln_rank"]
    return centroids


def _name_clusters_by_rank(profile: pd.DataFrame) -> dict[int, str]:
    """Build cluster labels of the form
    '{rank} Workforce Exposed / {rank} Emp Share in High AI Exp & <0 Emp Proj Occs'.

    The "Emp Share in ..." phrasing makes explicit that focused_share_pct
    is the share of state employment sitting in the 38-occ at-risk set —
    not a ranking within that set. Reads workforce_rank and vuln_rank
    from the profile table.
    """
    n = len(profile)
    names: dict[int, str] = {}
    for cid, row in profile.iterrows():
        workforce_word = _rank_word(int(row["workforce_rank"]), n)
        vuln_word = _rank_word(int(row["vuln_rank"]), n)
        names[int(cid)] = (
            f"{workforce_word} Workforce Exposed / "
            f"{vuln_word} Emp Share in High AI Exp & <0 Emp Proj Occs"
        )
    return names


def _legend_order(state_df: pd.DataFrame, profile: pd.DataFrame) -> list[int]:
    """Canonical ordering of cluster IDs for legends and color ramps.

    Outliers first (so the chart's callout reads at the top of the legend),
    then real clusters by rank_sum ascending (worst exposed = darkest blue
    listed first). Clusters present in state_df but missing from profile
    (e.g. extra outliers) are appended at the end.
    """
    present = list(state_df["cluster"].unique())
    outliers = [c for c in present if int(c) == OUTLIER_CLUSTER_ID]
    real = [c for c in present if int(c) != OUTLIER_CLUSTER_ID]
    real_sorted = sorted(
        real,
        key=lambda c: (
            profile.loc[c, "rank_sum"] if c in profile.index else 999,
            profile.loc[c, "workforce_rank"] if c in profile.index else 999,
        ),
    )
    return outliers + real_sorted


def _assign_cluster_colors_by_rank_sum(profile: pd.DataFrame) -> dict[int, str]:
    """Apply the blue gradient in order of rank_sum ascending.

    Lowest rank_sum (worst combined workforce-and-vulnerable position) gets
    the darkest blue; highest rank_sum gets the lightest. Ties (same rank_sum)
    break on workforce_rank ascending so behavior is deterministic.
    """
    ordered = profile.sort_values(
        ["rank_sum", "workforce_rank"], ascending=[True, True]
    )
    return {
        int(cid): BLUE_GRADIENT[min(i, len(BLUE_GRADIENT) - 1)]
        for i, cid in enumerate(ordered.index)
    }


# ─────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────

def _build_scatter(
    state_df: pd.DataFrame,
    cluster_names: dict[int, str],
    cluster_color: dict[int, str],
    order: list[int],
) -> go.Figure:
    """2D scatter: % workforce exposed (x) × % in High AI Exp & <0 Emp Proj (y), colored by cluster."""
    fig = go.Figure()

    for cid in order:
        sub = state_df[state_df["cluster"] == cid]
        fig.add_trace(go.Scatter(
            x=sub["pct_emp_wtd"],
            y=sub["focused_share_pct"],
            mode="markers+text",
            text=sub["geo_label"],
            textposition="top center",
            textfont=dict(size=10, color=COLORS["text"], family=FONT_FAMILY),
            marker=dict(
                size=14,
                color=cluster_color[cid],
                line=dict(color="white", width=1.5),
            ),
            name=cluster_names[cid],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "% workforce exposed: %{x:.1f}%<br>"
                "% in High AI Exp & <0 Emp Proj: %{y:.1f}%<extra></extra>"
            ),
        ))

    # Centroid markers — only for real clusters (skip outliers, which
    # weren't part of the clustering computation).
    real_clusters = state_df[state_df["cluster"] != OUTLIER_CLUSTER_ID]
    if not real_clusters.empty:
        centroids = real_clusters.groupby("cluster")[CLUSTER_FEATURES].mean()
        fig.add_trace(go.Scatter(
            x=centroids["pct_emp_wtd"],
            y=centroids["focused_share_pct"],
            mode="markers",
            marker=dict(
                size=22, symbol="x-thin",
                color=COLORS["text"],
                line=dict(width=3, color=COLORS["text"]),
            ),
            name="Centroid",
            hoverinfo="skip",
        ))

    style_figure(
        fig,
        "State Clusters in the (% Workforce Exposed × % in High AI Exp & <0 Emp Proj) plane",
        subtitle="These are the two features actually used for clustering — the scatter IS the cluster "
                 "space (after z-scoring). Cluster labels follow the paper's risk_score_5f language.",
        x_title="% of state workforce exposed",
        y_title="% of state employment in High AI Exp & <0 Emp Proj occupations",
        width=1300, height=760,
    )
    fig.update_layout(margin=dict(l=80, r=40, t=110, b=120))
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor=COLORS["grid"])
    return fig


def _build_choropleth(
    state_df: pd.DataFrame,
    cluster_names: dict[int, str],
    cluster_color: dict[int, str],
    order: list[int],
) -> go.Figure:
    """U.S. state choropleth colored by cluster.

    Plotly's USA-states locationmode accepts 2-letter postal codes including
    DC. To keep the discrete colorscale clean with non-contiguous cluster
    IDs (e.g. -1 for the outlier alongside 1, 2, 3), we remap to a 0..n-1
    display index that runs in the canonical `order`. Colorbar tickvals
    sit at the band centers.
    """
    # Reverse the canonical order for the colorbar so the darkest blue
    # (worst exposure) sits at the TOP of the bar rather than the bottom.
    # Plotly renders colorbars low-z at the bottom, so we reverse the
    # display-index mapping.
    display_order = list(reversed(order))
    display_idx = {int(cid): i for i, cid in enumerate(display_order)}
    n = len(display_order)

    # Build a discrete colorscale — one flat band per cluster.
    colorscale: list[list] = []
    for i, cid in enumerate(display_order):
        lo = i / n
        hi = (i + 1) / n
        colorscale.append([lo, cluster_color[cid]])
        colorscale.append([hi, cluster_color[cid]])

    z_vals = state_df["cluster"].map(display_idx).astype(float)

    fig = go.Figure(data=go.Choropleth(
        locations=state_df["geo_label"],
        z=z_vals,
        locationmode="USA-states",
        colorscale=colorscale,
        zmin=-0.5,
        zmax=n - 0.5,
        marker_line_color="white",
        marker_line_width=0.8,
        colorbar=dict(
            title=dict(text="Cluster", font=dict(size=11)),
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=[cluster_names[cid] for cid in display_order],
            tickfont=dict(size=10, family=FONT_FAMILY),
            thickness=15,
            len=0.75,
        ),
        text=[
            f"{r['geo_label']}<br>"
            f"% workforce exposed: {r['pct_emp_wtd']:.1f}%<br>"
            f"% in High AI Exp & <0 Emp Proj: {r['focused_share_pct']:.1f}%"
            for _, r in state_df.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True, landcolor=COLORS["bg_page"],
        showlakes=False, showsubunits=True,
        subunitcolor="white",
    )

    style_figure(
        fig,
        "U.S. States by AI-Exposure Cluster",
        subtitle="Ward hierarchical clustering on % workforce exposed and "
                 "% in High AI Exposure & Negative Emp Proj occupations. DC shown as outlier.",
        width=1400, height=820,
        show_legend=False,
    )
    fig.update_layout(margin=dict(l=20, r=180, t=110, b=80))
    return fig


def _build_recolored_bars(
    state_df: pd.DataFrame,
    cluster_names: dict[int, str],
    cluster_color: dict[int, str],
    order: list[int],
    n_focused: int,
) -> go.Figure:
    """Recolored copy of the paper's state_exposure_at_risk figure.

    Mirrors `build_state_exposure_at_risk` in paper/results/part_3/run.py:
    same layout, same paper styling, same panel titles, same axis labels —
    the only difference is that each bar's fill is its cluster color instead
    of the gold/sage-green pair the paper version uses.
    """
    plot = state_df.dropna(subset=ALL_FEATURES).copy()
    plot = plot.sort_values("pct_emp_wtd", ascending=False).reset_index(drop=True)
    n_states = len(plot)

    geos_r    = list(reversed(plot["geo_label"].tolist()))
    exp_r     = list(reversed(plot["pct_emp_wtd"].tolist()))
    focused_r = list(reversed(plot["focused_share_pct"].tolist()))
    colors_r  = list(reversed([cluster_color[c] for c in plot["cluster"].tolist()]))

    panel_left  = "% of State Workforce Exposed"
    panel_right = f"% in High AI Exp & <0 Emp Proj Occs (n={n_focused})"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[panel_left, panel_right],
        horizontal_spacing=0.08,
        shared_yaxes=True,
    )

    fig.add_trace(go.Bar(
        y=geos_r, x=exp_r, orientation="h",
        marker=dict(color=colors_r, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in exp_r],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Exposed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=geos_r, x=focused_r, orientation="h",
        marker=dict(color=colors_r, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in focused_r],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Most At Risk: %{x:.2f}%<extra></extra>",
    ), row=1, col=2)

    # Legend dummies — one invisible trace per cluster so the cluster
    # colors get an interpretable legend. Use the canonical `order` so
    # the outlier sits first and the blue gradient runs darkest -> lightest.
    for cid in order:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=cluster_color[cid]),
            name=cluster_names[cid],
            showlegend=True,
        ), row=1, col=1)

    height = max(PAPER_H + 250, n_states * 30 + 280)

    style_paper_figure(
        fig,
        "AI Exposure by State, Colored by Cluster",
        subtitle=(
            "Same layout and ranking as the paper's state_exposure_at_risk figure, "
            "but each bar's fill is its Ward cluster (50 states clustered, DC shown "
            f"as outlier). Legend at bottom. n = {n_states}."
        ),
        height=height,
        width=PAPER_W,
        # Big BOTTOM margin so the legend has dedicated space below the
        # axis titles. With chart heights of ~1780 px, plot-area-relative
        # paper coords need a generous absolute margin to actually clear
        # the bars.
        margin=dict(l=40, r=80, t=160, b=240),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        ticksuffix="%",
    )
    fig.update_xaxes(
        title=dict(text="Share of state workforce exposed (%)",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text="Share in High AI Exp & <0 Emp Proj occupations (%)",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=2,
    )

    # Force every state to render — categorical y-axes can auto-skip when
    # the chart is dense; dtick=1 keeps all 51 labels visible.
    fig.update_yaxes(
        showgrid=False, showline=False,
        tickmode="linear", dtick=1,
    )
    fig.update_yaxes(
        title=dict(text="State", font=dict(size=LABEL_FS - 2)),
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        row=1, col=1,
    )

    panel_set = {panel_left, panel_right}
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_set:
            ann.font = dict(size=LABEL_FS - 2, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    fig.update_layout(
        bargap=0.28,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.06,  # top of legend 6% below plot bottom
            xanchor="center", x=0.5,
            font=dict(size=TICK_FS - 1, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    return fig


def _build_recolored_bars_each_ranked(
    state_df: pd.DataFrame,
    cluster_names: dict[int, str],
    cluster_color: dict[int, str],
    order: list[int],
    n_focused: int,
) -> go.Figure:
    """Sibling of `_build_recolored_bars` where each panel sorts itself.

    The original recolored chart shares its y-axis ordering with the paper
    chart — both panels are ranked by % workforce exposed descending, so the
    right (High AI Exp & <0 Emp Proj) panel reads as out-of-order. This
    variant breaks the shared y-axis: left panel sorted by % workforce
    exposed descending, right panel sorted by % in High AI Exp & <0 Emp Proj
    descending. Same states in both, different rank positions; cluster colors
    stay consistent.
    """
    base = state_df.dropna(subset=ALL_FEATURES).copy()
    n_states = len(base)

    left_sorted = base.sort_values("pct_emp_wtd", ascending=False).reset_index(drop=True)
    right_sorted = base.sort_values("focused_share_pct", ascending=False).reset_index(drop=True)

    # Plotly's horizontal bars draw bottom-up — reverse so rank 1 lands at the top.
    def _rev(df: pd.DataFrame, value_col: str) -> tuple[list, list, list]:
        geos = list(reversed(df["geo_label"].tolist()))
        vals = list(reversed(df[value_col].tolist()))
        cols = list(reversed([cluster_color[c] for c in df["cluster"].tolist()]))
        return geos, vals, cols

    geos_left,  exp_vals,     colors_left  = _rev(left_sorted, "pct_emp_wtd")
    geos_right, focused_vals, colors_right = _rev(right_sorted, "focused_share_pct")

    panel_left  = "Sorted by % Workforce Exposed"
    panel_right = (
        f"Sorted by % in High AI Exp & <0 Emp Proj Occs (n={n_focused})"
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[panel_left, panel_right],
        horizontal_spacing=0.12,
        shared_yaxes=False,  # the key difference — each panel owns its order
    )

    fig.add_trace(go.Bar(
        y=geos_left, x=exp_vals, orientation="h",
        marker=dict(color=colors_left, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in exp_vals],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Exposed: %{x:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=geos_right, x=focused_vals, orientation="h",
        marker=dict(color=colors_right, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in focused_vals],
        textposition="outside",
        textfont=dict(size=ANNOT_FS - 1, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        showlegend=False, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Most At Risk: %{x:.2f}%<extra></extra>",
    ), row=1, col=2)

    # Legend dummies in the canonical order.
    for cid in order:
        fig.add_trace(go.Bar(
            y=[None], x=[None],
            marker=dict(color=cluster_color[cid]),
            name=cluster_names[cid],
            showlegend=True,
        ), row=1, col=1)

    height = max(PAPER_H + 250, n_states * 30 + 280)

    style_paper_figure(
        fig,
        "AI Exposure by State, Colored by Cluster — Each Panel Ranked Independently",
        subtitle=(
            "Same data and cluster colors as the previous figure, but each "
            "panel sorts the 51 states by its own metric. Compare left vs. "
            "right ranks to see how % workforce exposed and concentration "
            "in High AI Exp & <0 Emp Proj occupations disagree on which "
            "states sit at the top."
        ),
        height=height,
        width=PAPER_W,
        margin=dict(l=40, r=80, t=160, b=240),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
        ticksuffix="%",
    )
    fig.update_xaxes(
        title=dict(text="Share of state workforce exposed (%)",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text="Share in High AI Exp & <0 Emp Proj occupations (%)",
                   font=dict(size=LABEL_FS - 4)),
        row=1, col=2,
    )

    # Each panel gets its own state labels on the y-axis (not shared).
    # dtick=1 forces every state to render — without it, plotly auto-skips
    # labels when the chart is dense.
    fig.update_yaxes(
        showgrid=False, showline=False,
        tickmode="linear", dtick=1,
        tickfont=dict(size=TICK_FS - 2, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        title=dict(text="State (by % workforce exposed)",
                   font=dict(size=LABEL_FS - 2)),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="State (by % in High AI Exp & <0 Emp Proj)",
                   font=dict(size=LABEL_FS - 2)),
        row=1, col=2,
    )

    panel_set = {panel_left, panel_right}
    for ann in fig.layout.annotations:
        if hasattr(ann, "text") and ann.text in panel_set:
            ann.font = dict(size=LABEL_FS - 2, family=FONT_FAMILY,
                            color=PAPER_PALETTE["text"])

    fig.update_layout(
        bargap=0.28,
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.06,
            xanchor="center", x=0.5,
            font=dict(size=TICK_FS - 1, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Importable pipeline — used by paper part_3 and appendix without
# duplicating the clustering logic.
# ─────────────────────────────────────────────────────────────────────

def compute_clusters() -> dict:
    """Run the full clustering pipeline and return everything the
    paper-side chart builders need.

    Returns a dict with:
      - state_df:        full 51-row DataFrame with cluster, cluster_name,
                          and feature columns
      - profile:         per-cluster centroid + rank table
      - cluster_names:   dict[cluster_id, str]
      - cluster_color:   dict[cluster_id, hex]
      - order:           canonical legend order (outlier first, then
                          rank_sum ascending)
      - chosen_k:        number of non-outlier clusters
      - ari:             Ward vs K-means Adjusted Rand Index
    """
    state_df = _load_state_features()

    outlier_mask = state_df["geo"].isin(OUTLIER_GEOS)
    outliers_df = state_df[outlier_mask].copy().reset_index(drop=True)
    cluster_input_df = state_df[~outlier_mask].copy().reset_index(drop=True)

    X = cluster_input_df[CLUSTER_FEATURES].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)

    Z = linkage(Xz, method="ward")
    chosen_k, _ = _pick_k_from_linkage(Z, K_MIN, K_MAX)
    ward_labels = fcluster(Z, t=chosen_k, criterion="maxclust")

    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
    km_labels = km.fit_predict(Xz) + 1
    ari = adjusted_rand_score(ward_labels, km_labels)

    cluster_input_df["cluster"] = ward_labels
    profile = _build_cluster_profile(cluster_input_df, "cluster")
    cluster_names = _name_clusters_by_rank(profile)
    cluster_color = _assign_cluster_colors_by_rank_sum(profile)
    cluster_input_df["cluster_name"] = cluster_input_df["cluster"].map(cluster_names)

    if not outliers_df.empty:
        outlier_rows = []
        for _, r in outliers_df.iterrows():
            label = (
                f"{r['geo_label']} (outlier — {r['pct_emp_wtd']:.1f}% workforce exposed, "
                f"{r['focused_share_pct']:.1f}% emp share in High AI Exp & <0 Emp Proj occs)"
            )
            outlier_rows.append({
                **r.to_dict(),
                "cluster": OUTLIER_CLUSTER_ID,
                "cluster_name": label,
            })
            cluster_names[OUTLIER_CLUSTER_ID] = label
        cluster_color[OUTLIER_CLUSTER_ID] = OUTLIER_COLOR
        full_state_df = pd.concat(
            [cluster_input_df, pd.DataFrame(outlier_rows)],
            ignore_index=True,
        )
    else:
        full_state_df = cluster_input_df.copy()

    order = _legend_order(full_state_df, profile)

    return {
        "state_df": full_state_df,
        "profile": profile,
        "cluster_names": cluster_names,
        "cluster_color": cluster_color,
        "order": order,
        "chosen_k": chosen_k,
        "ari": ari,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    figs_dir = HERE / "figures"
    figs_dir.mkdir(exist_ok=True)

    print(f"deepdive_state_clusters: primary config = {PRIMARY_LABEL} ({PRIMARY_DATASET})")

    state_df = _load_state_features()
    print(f"  Loaded {len(state_df)} state rows.")
    print(f"  Clustering features: {CLUSTER_FEATURES}")

    # Correlation matrix — over all features so the reader can see why the
    # wage signals were demoted to descriptive-only.
    corr = state_df[ALL_FEATURES].corr(method="pearson")
    print("\nFeature correlations (Pearson r, all features):")
    print(corr.round(3).to_string())
    high_pairs = []
    for i, a in enumerate(ALL_FEATURES):
        for b in ALL_FEATURES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.9:
                high_pairs.append((a, b, r))
    if high_pairs:
        print("\n  NOTE: Highly correlated pairs (|r| >= 0.9):")
        for a, b, r in high_pairs:
            cluster_a = "cluster" if a in CLUSTER_FEATURES else "descriptive"
            cluster_b = "cluster" if b in CLUSTER_FEATURES else "descriptive"
            print(f"    {a} ({cluster_a}) <-> {b} ({cluster_b}): r = {r:.3f}")
    else:
        print("\n  No pair exceeds |r| >= 0.9 in the feature set.")

    save_csv(corr.reset_index().rename(columns={"index": "feature"}),
             results / "correlations.csv", float_format="%.4f")

    # ── Split outliers from the clustering input ──────────────────────
    outlier_mask = state_df["geo"].isin(OUTLIER_GEOS)
    outliers_df = state_df[outlier_mask].copy().reset_index(drop=True)
    cluster_input_df = state_df[~outlier_mask].copy().reset_index(drop=True)
    print(f"\nOutliers excluded from clustering: "
          f"{outliers_df['geo_label'].tolist() if not outliers_df.empty else 'none'}")
    print(f"  Clustering on {len(cluster_input_df)} states.")

    # ── Standardize features (on the non-outlier subset only) ─────────
    X = cluster_input_df[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # ── Ward hierarchical ─────────────────────────────────────────────
    Z = linkage(Xz, method="ward")
    chosen_k, k_ratios = _pick_k_from_linkage(Z, K_MIN, K_MAX)
    print(f"\nWard k-selection (ratio of merge-distance at k->k-1 vs. k+1->k):")
    for k, r in sorted(k_ratios.items()):
        flag = "  <-- chosen" if k == chosen_k else ""
        print(f"  k = {k}: ratio = {r:.3f}{flag}")
    print(f"  Chosen k = {chosen_k}")

    ward_labels = fcluster(Z, t=chosen_k, criterion="maxclust")
    cluster_input_df["ward_cluster"] = ward_labels

    # ── K-means at the same k (robustness) ────────────────────────────
    km = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
    km_labels = km.fit_predict(Xz) + 1  # 1-indexed to match fcluster

    # Re-map km cluster IDs to align with Ward IDs by majority overlap so the
    # ARI / membership-diff comparison is interpretable (label ID is arbitrary,
    # the *partitioning* is what we compare).
    aligned = np.zeros_like(km_labels)
    for w in np.unique(ward_labels):
        idx = ward_labels == w
        if idx.sum() == 0:
            continue
        majority = pd.Series(km_labels[idx]).mode().iloc[0]
        aligned[km_labels == majority] = w
    for i, v in enumerate(aligned):
        if v == 0:
            aligned[i] = km_labels[i] + 100
    cluster_input_df["kmeans_cluster"] = aligned

    ari = adjusted_rand_score(ward_labels, km_labels)
    n_same = int((cluster_input_df["ward_cluster"]
                  == cluster_input_df["kmeans_cluster"]).sum())
    n_total = len(cluster_input_df)
    print(f"\nWard vs. K-means agreement (on clustered states only):")
    print(f"  Adjusted Rand Index = {ari:.3f}")
    print(f"  Same cluster:       {n_same} / {n_total} states "
          f"({n_same / n_total * 100:.1f}%)")

    # ── Build cluster profile (centroids + ranks + rank-sum) ──────────
    cluster_input_df["cluster"] = cluster_input_df["ward_cluster"]
    profile = _build_cluster_profile(cluster_input_df, "cluster")
    cluster_names = _name_clusters_by_rank(profile)
    cluster_color = _assign_cluster_colors_by_rank_sum(profile)
    cluster_input_df["cluster_name"] = cluster_input_df["cluster"].map(cluster_names)

    # ── Re-attach outliers to the visualization frame ─────────────────
    # Outliers get sentinel cluster_id, outlier color, and a special name
    # that includes their own per-state values so they stand out cleanly.
    if not outliers_df.empty:
        outlier_rows = []
        for _, r in outliers_df.iterrows():
            label = (
                f"{r['geo_label']} (outlier — {r['pct_emp_wtd']:.1f}% workforce exposed, "
                f"{r['focused_share_pct']:.1f}% emp share in High AI Exp & <0 Emp Proj occs)"
            )
            outlier_rows.append({
                **r.to_dict(),
                "cluster": OUTLIER_CLUSTER_ID,
                "cluster_name": label,
                "ward_cluster": OUTLIER_CLUSTER_ID,
                "kmeans_cluster": OUTLIER_CLUSTER_ID,
            })
            cluster_names[OUTLIER_CLUSTER_ID] = label
        cluster_color[OUTLIER_CLUSTER_ID] = OUTLIER_COLOR
        outliers_attached = pd.DataFrame(outlier_rows)
        state_df = pd.concat([cluster_input_df, outliers_attached],
                             ignore_index=True)
    else:
        state_df = cluster_input_df.copy()

    print(f"\nCluster rank profile (lowest rank_sum = worst exposed):")
    print(profile.sort_values("rank_sum")
          [["n_states", "pct_emp_wtd", "focused_share_pct",
            "workforce_rank", "vuln_rank", "rank_sum"]]
          .round(2).to_string())

    # Persist outputs.
    out_cols = ["geo", "geo_label", "cluster", "cluster_name",
                "ward_cluster", "kmeans_cluster", *ALL_FEATURES]
    save_csv(state_df[out_cols], results / "state_clusters.csv", float_format="%.4f")

    # Cluster summary table. Merge in the rank columns from the profile so
    # the CSV captures the scoring logic. Outliers don't have ranks (they
    # weren't clustered), so they get NaN there.
    summ = (
        state_df.groupby(["cluster", "cluster_name"])
        .agg(
            n_states=("geo", "count"),
            pct_emp_wtd_mean=("pct_emp_wtd", "mean"),
            focused_share_pct_mean=("focused_share_pct", "mean"),
            states=("geo_label", lambda s: ", ".join(sorted(s))),
        )
        .reset_index()
    )
    rank_cols = profile[["workforce_rank", "vuln_rank", "rank_sum"]].reset_index()
    summ = summ.merge(rank_cols, on="cluster", how="left")
    # Sort: outliers first (so the chart's DC bar reads as the "callout"
    # row), then real clusters by rank_sum ascending (worst exposed first).
    summ["_order"] = summ["cluster"].apply(
        lambda c: -1 if int(c) == OUTLIER_CLUSTER_ID else int(c)
    )
    summ = summ.sort_values(["_order", "rank_sum"]).drop(columns="_order").reset_index(drop=True)
    save_csv(summ, results / "cluster_summary.csv", float_format="%.2f")
    print("\nCluster summary (sorted: outliers first, then by rank_sum ascending):")
    print(summ.drop(columns=["states"]).to_string(index=False))
    print()
    for _, r in summ.iterrows():
        print(f"  Cluster {int(r['cluster'])} ({r['cluster_name']}): {r['states']}")

    # ── Figures ───────────────────────────────────────────────────────
    print("\nBuilding figures...")
    figures: list[tuple[str, go.Figure]] = []

    fig_corr = _build_corr_heatmap(corr)
    figures.append(("00_feature_correlations.png", fig_corr))

    # Dendrogram labels come from the clustering input (DC excluded).
    fig_dend = _build_dendrogram(
        Z, cluster_input_df["geo_label"].tolist(), chosen_k
    )
    figures.append(("01_dendrogram.png", fig_dend))

    order = _legend_order(state_df, profile)

    fig_scatter = _build_scatter(state_df, cluster_names, cluster_color, order)
    figures.append(("02_cluster_scatter.png", fig_scatter))

    fig_map = _build_choropleth(state_df, cluster_names, cluster_color, order)
    figures.append(("03_us_choropleth.png", fig_map))

    # Focused-set size matches the paper's state_exposure_at_risk caption (n=38,
    # the SKA-gated Most-At-Risk set from audit_risk_score).
    fig_bars = _build_recolored_bars(state_df, cluster_names, cluster_color,
                                     order, n_focused=38)
    figures.append(("04_recolored_bars.png", fig_bars))

    fig_bars_each = _build_recolored_bars_each_ranked(
        state_df, cluster_names, cluster_color, order, n_focused=38
    )
    figures.append(("05_recolored_bars_each_ranked.png", fig_bars_each))

    for fname, fig in figures:
        out_path = results / "figures" / fname
        save_figure(fig, out_path)
        shutil.copy(out_path, figs_dir / fname)
        print(f"  -> {fname}")

    print("\ndeepdive_state_clusters: done.")


if __name__ == "__main__":
    main()
