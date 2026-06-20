"""
run_v3.py — pct_norm vs eco intensity, V3 deliverable.

Major-occupational-category-only intensity ranking, sliced by source dataset and
optional bias correction. Same metric mechanics as v2:

    ratio[major]    = Σ pct_normalized (optionally bias-corrected)
                      / Σ (freq × emp)
                      over unique (task, occ) pairs in the major
    ratio_pct[major] = ratio renormalized so all majors sum to 100

Coverage (11 charts):

    01  all_confirmed       — no bias
    02  microsoft_only      — no bias
    03  aei_all             — no bias
    04  aei_conv            — no bias
    05  aei_api             — no bias
    06  all_confirmed       — equal 3-source bias correction
    07  aei_all             — equal 3-source bias correction
    08  aei_conv            — equal 3-source bias correction
    09  aei_api             — equal 3-source bias correction
    10  copilot_allocation  — synth pct from Copilot's GWA share split evenly
                              across all eco_2025 tasks under each GWA
    11  chatgpt_allocation  — same construction with ChatGPT's GWA share

For 06–09 the bias variant is `equal` (Claude/Copilot/ChatGPT each weighted 1×),
matching v1 / v2's existing methodology — `bias_ratio[gwa] = claude_share /
mean(claude, copilot, chatgpt)`. The Claude-share prior comes from the existing
hardcoded AEI conversation distribution and is reused for AEI Conv / AEI API /
AEI All / All Confirmed alike (same convention as the rest of this analysis).

For 10–11 the synth pct is built per-task: each canonical GWA's published share
is divided evenly among unique tasks under that GWA in eco_2025; tasks under
multiple GWAs sum across their GWAs. The intensity ratio is then computed over
the eco_2025 (task, occ) universe.

Run from project root:
    venv/Scripts/python -m analysis.exploratory.pct_norm_vs_eco.run_v3
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.config import ROOT, ensure_results_dir
from lib.utils import COLORS, FONT_FAMILY, save_csv, save_figure
from lib.exploratory.intensity import (
    AEI_GWA_RENAME,
    BIAS_LABELS,
    BIAS_VARIANTS,
    CANONICAL_GWAS,
    CHATGPT_SHARE,
    CLAUDE_SHARE,
    COPILOT_SHARE,
    EMP_COL,
    compute_bias_ratios,
    load_eco_df,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


# ── V3 configs (separate dict so we don't mutate run.py's CONFIGS) ──────────

V3_CONFIGS: dict[str, dict[str, str]] = {
    "all_confirmed": {
        "file": "final_all_confirmed_usage_2026-02-12.csv",
        "label": "All Confirmed (AEI Both + Micro 2026-02-12)",
        "occ_col": "title_current",
    },
    "microsoft_only": {
        "file": "final_microsoft.csv",
        "label": "Microsoft Copilot",
        "occ_col": "title_current",
    },
    "aei_all": {
        "file": "final_aei_all_usage_2026-02-12.csv",
        "label": "AEI All Usage (Conv + API 2026-02-12)",
        "occ_col": "title",
    },
    "aei_all_eco2025": {
        # AEI Conv + AEI API pooled with task_prop normalization onto the
        # eco_2025 universe (already 2019 SOC, no crosswalk). Used by the
        # paper's intensity figures so the numerator drops Microsoft while
        # still keeping the equal 3-source bias correction (the bias prior
        # is GWA-level and applies regardless of which dataset is measured).
        "file": "final_aei_all_usage_2025_2026-02-12.csv",
        "label": "AEI All Usage — eco_2025 baseline (Conv + API 2026-02-12, no Microsoft)",
        "occ_col": "title_current",
    },
    "aei_conv": {
        "file": "final_aei_human_usage_2026-02-12.csv",
        "label": "AEI Conv. (Human 2026-02-12)",
        "occ_col": "title",
    },
    "aei_api": {
        "file": "final_aei_agentic_usage_2026-02-12.csv",
        "label": "AEI API (Agentic 2026-02-12)",
        "occ_col": "title",
    },
}

# AEI-family configs that use eco_2015 GWA names → rename to canonical eco_2025
_AEI_FAMILY = {"aei_all", "aei_conv", "aei_api"}

_CONFIG_CACHE: dict[str, pd.DataFrame] = {}


def load_v3_config(config_key: str) -> pd.DataFrame:
    """Load a V3 config CSV, applying AEI GWA rename for AEI-family configs."""
    if config_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_key]
    cfg = V3_CONFIGS[config_key]
    path = DATA_DIR / cfg["file"]
    assert path.exists(), f"Missing config file: {path}"
    occ_col = cfg["occ_col"]
    usecols = [
        "task_normalized",
        occ_col,
        "major_occ_category",
        "gwa_title",
        "pct_normalized",
        "auto_aug_mean",
        "freq_mean",
        EMP_COL,
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    assert not df.empty, f"Empty: {path}"
    for c in ("pct_normalized", "auto_aug_mean", "freq_mean", EMP_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if config_key in _AEI_FAMILY:
        df["gwa_title"] = df["gwa_title"].replace(AEI_GWA_RENAME)
    _CONFIG_CACHE[config_key] = df
    return df


# ── Intensity computation (major-only, dataset-driven) ──────────────────────

def compute_v3_intensity(
    config_key: str,
    bias_ratios: Optional[dict[str, float]],
    weight_by_auto_aug: bool = False,
) -> pd.DataFrame:
    """Major-level intensity ratio with optional bias correction.

    Numerator and denominator both restricted to (task, occ) pairs that are
    rated by the config (i.e. pct_normalized non-null) and have a major.
    Bias correction averages bias_ratio across the GWAs each (task, occ) maps
    to (consistent with v1's occ-hierarchy treatment).

    weight_by_auto_aug: if True, multiply each (task, occ)'s adj_pct by
    (auto_aug_mean / 5) before summing into the major numerator. Tasks
    flagged as more automatable contribute proportionally more; tasks with
    low or missing auto_aug shrink toward zero (NaN treated as 0).
    """
    df = load_v3_config(config_key)
    occ_col = V3_CONFIGS[config_key]["occ_col"]

    ai = df[df["pct_normalized"].notna() & df["major_occ_category"].notna()].copy()
    ai["eco_weight"] = ai["freq_mean"].fillna(0.0) * ai[EMP_COL].fillna(0.0)

    avg_bias_df: Optional[pd.DataFrame] = None
    if bias_ratios is not None:
        gwa_pairs = (
            ai.dropna(subset=["gwa_title"])
            .drop_duplicates(["task_normalized", occ_col, "gwa_title"])[
                ["task_normalized", occ_col, "gwa_title"]
            ]
            .copy()
        )
        gwa_pairs["bias"] = gwa_pairs["gwa_title"].map(bias_ratios).fillna(1.0)
        avg_bias_df = (
            gwa_pairs.groupby(["task_normalized", occ_col])["bias"].mean()
            .reset_index(name="avg_bias")
        )

    keep = [
        "task_normalized",
        occ_col,
        "major_occ_category",
        "pct_normalized",
        "auto_aug_mean",
        "eco_weight",
    ]
    dedup = ai.drop_duplicates(["task_normalized", occ_col])[keep].copy()
    if avg_bias_df is not None:
        dedup = dedup.merge(avg_bias_df, on=["task_normalized", occ_col], how="left")
        dedup["avg_bias"] = dedup["avg_bias"].fillna(1.0).replace(0.0, 1.0)
        dedup["adj_pct"] = dedup["pct_normalized"] / dedup["avg_bias"]
    else:
        dedup["adj_pct"] = dedup["pct_normalized"]

    if weight_by_auto_aug:
        weight = dedup["auto_aug_mean"].fillna(0.0) / 5.0
        dedup["adj_pct"] = dedup["adj_pct"] * weight

    grp = (
        dedup.groupby("major_occ_category")
        .agg(
            num=("adj_pct", "sum"),
            den=("eco_weight", "sum"),
            raw_pct=("pct_normalized", "sum"),
        )
        .reset_index()
        .rename(columns={"major_occ_category": "category"})
    )
    grp["ratio"] = np.where(grp["den"] > 0, grp["num"] / grp["den"], 0.0)
    total = grp["ratio"].sum()
    grp["ratio_pct"] = grp["ratio"] / total * 100.0 if total > 0 else 0.0
    # raw_pct: un-debiased Σ pct_normalized per major over the same deduped
    # (task, occ) pairs as num. Same definition as the appendix
    # intensity_drivers "raw pct" bar text; additive column, callers select
    # by name so it's safe.
    return grp[["category", "num", "den", "raw_pct", "ratio", "ratio_pct"]]


# ── pct_tasks_affected per major (standard dashboard formula) ───────────────

def compute_major_pct_tasks_affected() -> pd.Series:
    """Compute pct_tasks_affected per major from the all_confirmed dataset.

    Standard dashboard formula at major level — ratio-of-totals over rated
    (task, occ) pairs:
        pct[major] = Σ (auto_aug/5 × freq × emp) / Σ (freq × emp)  × 100

    Returns Series keyed by major_occ_category, values in 0–100.
    """
    df = load_v3_config("all_confirmed")
    occ_col = V3_CONFIGS["all_confirmed"]["occ_col"]
    rated = df[
        df["pct_normalized"].notna() & df["major_occ_category"].notna()
    ].copy()
    pairs = rated.drop_duplicates(["task_normalized", occ_col]).copy()
    pairs["eco_weight"] = pairs["freq_mean"].fillna(0.0) * pairs[EMP_COL].fillna(0.0)
    pairs["ai_weight"] = (
        pairs["eco_weight"] * pairs["auto_aug_mean"].fillna(0.0) / 5.0
    )
    grp = pairs.groupby("major_occ_category").agg(
        ai_w=("ai_weight", "sum"),
        eco_w=("eco_weight", "sum"),
    )
    pct = np.where(grp["eco_w"] > 0, grp["ai_w"] / grp["eco_w"] * 100.0, 0.0)
    return pd.Series(pct, index=grp.index, name="pct_tasks_affected")


def compute_major_full_eco_denominator() -> pd.Series:
    """Σ (freq × emp) per major over the FULL eco_2025 universe (every
    deduped task-occ pair, regardless of dataset coverage)."""
    eco = load_eco_df()
    eco = eco[eco["major_occ_category"].notna()].copy()
    pairs = eco.drop_duplicates(["task_normalized", "title_current"]).copy()
    pairs["eco_weight"] = pairs["freq_mean"].fillna(0.0) * pairs[EMP_COL].fillna(0.0)
    return pairs.groupby("major_occ_category")["eco_weight"].sum().rename("den_full")


def _interpolate_hex(t: float, light_hex: str = "#dde6ee", dark_hex: Optional[str] = None) -> str:
    """Linear RGB interpolation. t ∈ [0, 1]; t=0 → light_hex, t=1 → dark_hex."""
    if dark_hex is None:
        dark_hex = COLORS["primary"]
    t = max(0.0, min(1.0, float(t)))

    def _to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    light = _to_rgb(light_hex)
    dark = _to_rgb(dark_hex)
    rgb = tuple(int(light[i] + t * (dark[i] - light[i])) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ── Source-allocation charts (Copilot / ChatGPT priors) ─────────────────────

def compute_source_allocation_intensity(
    source_share: dict[str, float],
) -> pd.DataFrame:
    """Build a synthetic per-task pct from a published GWA share distribution,
    spread evenly across unique tasks in eco_2025 under each GWA, then compute
    the major-level intensity ratio.

    Allocation rule:
      per_task_share[gwa, task] = source_share[gwa] / n_unique_eco_tasks_in[gwa]
      task_alloc[task] = Σ per_task_share over GWAs the task maps to
      (task, occ) pairs in eco_2025 are deduped; each pair inherits its task's
      task_alloc. Numerator and denominator (Σ freq×emp) sum over all eco_2025
      (task, occ) pairs in each major, regardless of whether the task has a
      GWA (tasks without a canonical GWA contribute 0 to the numerator).
    """
    eco = load_eco_df()
    eco = eco[eco["major_occ_category"].notna()].copy()

    # Restrict the (task, gwa) link to canonical GWAs the source covers.
    task_gwa = (
        eco[eco["gwa_title"].isin(source_share.keys())]
        .drop_duplicates(["task_normalized", "gwa_title"])[
            ["task_normalized", "gwa_title"]
        ]
        .copy()
    )
    # Unique-task-per-GWA count for the allocation denominator.
    gwa_task_count = task_gwa.groupby("gwa_title")["task_normalized"].nunique().to_dict()
    task_gwa["per_task_share"] = task_gwa.apply(
        lambda r: (source_share[r["gwa_title"]] / gwa_task_count[r["gwa_title"]])
        if gwa_task_count.get(r["gwa_title"], 0) > 0
        else 0.0,
        axis=1,
    )
    task_alloc = (
        task_gwa.groupby("task_normalized")["per_task_share"].sum().rename("alloc_pct")
    )

    pairs = (
        eco.drop_duplicates(["task_normalized", "title_current"])[
            ["task_normalized", "title_current", "major_occ_category", "freq_mean", EMP_COL]
        ]
        .copy()
    )
    pairs["eco_weight"] = pairs["freq_mean"].fillna(0.0) * pairs[EMP_COL].fillna(0.0)
    pairs = pairs.merge(task_alloc, on="task_normalized", how="left")
    pairs["alloc_pct"] = pairs["alloc_pct"].fillna(0.0)

    grp = (
        pairs.groupby("major_occ_category")
        .agg(num=("alloc_pct", "sum"), den=("eco_weight", "sum"))
        .reset_index()
        .rename(columns={"major_occ_category": "category"})
    )
    grp["ratio"] = np.where(grp["den"] > 0, grp["num"] / grp["den"], 0.0)
    total = grp["ratio"].sum()
    grp["ratio_pct"] = grp["ratio"] / total * 100.0 if total > 0 else 0.0
    return grp[["category", "num", "den", "ratio", "ratio_pct"]]


# ── Chart helper ─────────────────────────────────────────────────────────────

def ranking_chart(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    value_col: str = "ratio_pct",
    value_fmt: str = ".2f",
    x_axis_title: str = "AI intensity share (% of total; Σpct / Σ(freq×emp) renormalized)",
    ref_lines: Optional[list[tuple[float, str]]] = None,
    color_by: Optional[str] = None,
) -> go.Figure:
    """Single-color horizontal bar of value_col, all majors visible.

    ref_lines: optional list of (x_value, label) pairs to draw as dashed
    vertical reference lines (e.g., a median or anchor marker).
    color_by: optional column name; if present, bars are shaded light→dark
    based on that column's value across the rows (darker = higher).
    """
    plot_df = df.sort_values(value_col, ascending=True).copy()

    max_val = plot_df[value_col].max() if len(plot_df) else 1.0
    pad = max_val * 0.18 if max_val > 0 else 1.0

    if color_by and color_by in plot_df.columns:
        cvals = plot_df[color_by].to_numpy(dtype=float)
        cmin, cmax = cvals.min(), cvals.max()
        if cmax > cmin:
            ts = (cvals - cmin) / (cmax - cmin)
        else:
            ts = np.full_like(cvals, 0.5)
        bar_colors = [_interpolate_hex(t) for t in ts]
    else:
        bar_colors = COLORS["primary"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=plot_df["category"],
            x=plot_df[value_col],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:{value_fmt}}" for v in plot_df[value_col]],
            textposition="outside",
            textfont=dict(size=12, family=FONT_FAMILY, color=COLORS["text"]),
            customdata=(
                plot_df[color_by].to_numpy() if color_by and color_by in plot_df.columns else None
            ),
            hovertemplate=(
                "%{y}<br>" + x_axis_title.split(" — ")[-1].split(" (")[0] +
                ": %{x:" + value_fmt + "}<br>" +
                (f"{color_by}: %{{customdata:.1f}}<extra></extra>"
                 if color_by and color_by in plot_df.columns else "<extra></extra>")
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub style='color:#5a5a5a'>{subtitle}</sub>",
            x=0.02,
            xanchor="left",
            y=0.97,
            font=dict(family=FONT_FAMILY, size=18, color=COLORS["text"]),
        ),
        xaxis=dict(
            title=x_axis_title,
            range=[0, max_val + pad],
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, automargin=True),
        font=dict(family=FONT_FAMILY, size=13, color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        height=max(620, 28 * len(plot_df) + 220),
        width=1500,
        margin=dict(l=420, r=120, t=120, b=70),
        showlegend=False,
    )
    if ref_lines:
        for x_val, label in ref_lines:
            fig.add_vline(
                x=x_val,
                line_dash="dash",
                line_color="#c0392b",
                line_width=2,
                annotation_text=label,
                annotation_position="top",
                annotation_font=dict(size=12, color="#c0392b", family=FONT_FAMILY),
            )
    return fig


# ── Spec ─────────────────────────────────────────────────────────────────────

# (chart_id, kind, config_key_or_share_name, bias_key)
V3_SPEC: list[tuple[str, str, str, Optional[str]]] = [
    ("01_all_confirmed",        "config", "all_confirmed",   None),
    ("02_microsoft",            "config", "microsoft_only",  None),
    ("03_aei_all",              "config", "aei_all",         None),
    ("04_aei_conv",             "config", "aei_conv",        None),
    ("05_aei_api",              "config", "aei_api",         None),
    ("06_all_confirmed_bias",   "config", "all_confirmed",   "equal"),
    ("07_aei_all_bias",         "config", "aei_all",         "equal"),
    ("08_aei_conv_bias",        "config", "aei_conv",        "equal"),
    ("09_aei_api_bias",         "config", "aei_api",         "equal"),
    ("10_copilot_allocation",   "alloc",  "copilot",         None),
    ("11_chatgpt_allocation",   "alloc",  "chatgpt",         None),
]

ALLOC_SHARES = {
    "copilot": ("Microsoft Copilot — published GWA share", COPILOT_SHARE),
    "chatgpt": ("ChatGPT (Weidinger et al.) — published GWA share", CHATGPT_SHARE),
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results_dir = ensure_results_dir(HERE)
    fig_dir = results_dir / "figures" / "v3"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = results_dir / "v3"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets …")
    for key in V3_CONFIGS:
        load_v3_config(key)
    load_eco_df()

    print(f"Producing {len(V3_SPEC)} charts …")

    # Combined CSV: one row per (chart_id, major) for cross-comparison.
    all_rows: list[pd.DataFrame] = []

    for i, (chart_id, kind, ref, bias) in enumerate(V3_SPEC, 1):
        if kind == "config":
            bias_ratios = compute_bias_ratios(BIAS_VARIANTS[bias]) if bias else None
            result = compute_v3_intensity(ref, bias_ratios)
            cfg_label = V3_CONFIGS[ref]["label"]
            bias_disp = BIAS_LABELS[bias] if bias else "no bias correction"
            subtitle = (
                f"{cfg_label} · {bias_disp} · rated-task denominator. "
                f"Per-major ratio renormalized to 100%."
            )
        else:
            label, share_dict = ALLOC_SHARES[ref]
            result = compute_source_allocation_intensity(share_dict)
            subtitle = (
                f"{label} · evenly split across eco_2025 tasks per GWA · "
                f"full eco_2025 denominator. Per-major ratio renormalized to 100%."
            )

        fig = ranking_chart(
            result,
            "AI Intensity Ranking — Major Occupational Categories",
            subtitle,
        )
        save_figure(fig, fig_dir / f"{chart_id}.png")

        save_csv(
            result.sort_values("ratio_pct", ascending=False),
            csv_dir / f"{chart_id}.csv",
            float_format="%.4f",
        )

        tagged = result.copy()
        tagged.insert(0, "chart_id", chart_id)
        all_rows.append(tagged)

        print(f"  {i:2d}/{len(V3_SPEC)}  {chart_id}")

    combined = pd.concat(all_rows, ignore_index=True)
    save_csv(combined, csv_dir / "all_variants_combined.csv", float_format="%.4f")

    # Wide pivot for side-by-side comparison: rows = major, cols = chart_id.
    wide = combined.pivot(index="category", columns="chart_id", values="ratio_pct")
    wide = wide.reset_index().rename(columns={"category": "major"})
    save_csv(wide, csv_dir / "all_variants_wide.csv", float_format="%.4f")

    # Charts 12–14 — All Confirmed (bias-corrected) intensity, anchored on the
    # higher of the two median-ranked majors so that anchor reads as 1.00×.
    # 12 — basic ratio (Σ pct / Σ ew)
    # 13 — sqrt smoothing (Σ pct / sqrt(Σ ew))
    # 14 — additive smoothing (Σ pct / (Σ ew + median(ew)))
    # Each chart also draws a dashed vertical line at the lift-distribution's
    # statistical median across the 22 majors (avg of 11th & 12th ranks).
    print("Producing chart 12–15 (anchored variants on chart 06 data) …")
    base = compute_v3_intensity(
        "all_confirmed", compute_bias_ratios(BIAS_VARIANTS["equal"])
    ).copy()

    # Anchor major: 12th of 22 sorted ascending (the higher of the two middle
    # majors). Resolved from the basic ratio's ordering and reused for 13/14/15
    # so the four charts are directly comparable.
    base_sorted = base.sort_values("ratio_pct", ascending=True).reset_index(drop=True)
    anchor_major = base_sorted.iloc[11]["category"]
    print(f"  Anchor major: {anchor_major}")

    # pct_tasks_affected per major (used as a continuous color signal on 12 & 15).
    pct_aff = compute_major_pct_tasks_affected()
    base["pct_tasks_affected"] = base["category"].map(pct_aff).fillna(0.0)

    def _anchor_and_plot(
        df: pd.DataFrame,
        ratio_col: str,
        chart_id: str,
        chart_title: str,
        method_label: str,
        color_by_pct: bool = False,
    ) -> pd.DataFrame:
        """Divide ratio_col by anchor major's value → lift; plot with median line.
        If color_by_pct, shade bars by pct_tasks_affected (darker = higher)."""
        anchor_val = df.loc[df["category"] == anchor_major, ratio_col].iloc[0]
        assert anchor_val > 0, f"Anchor value for {anchor_major} must be > 0"
        df = df.copy()
        df["lift"] = df[ratio_col] / anchor_val
        median_lift = df["lift"].median()

        color_note = (
            " · bar shading: darker = higher pct_tasks_affected"
            if color_by_pct else ""
        )
        fig = ranking_chart(
            df,
            chart_title,
            f"All Confirmed (AEI Both + Micro 2026-02-12) · equal 3-source consensus "
            f"· {method_label}. Anchor: {anchor_major} = 1.00×. Dashed line = "
            f"median lift across the 22 majors ({median_lift:.2f}×).{color_note}",
            value_col="lift",
            value_fmt=".2f",
            x_axis_title=f"AI usage relative to anchor major (×) — {method_label}",
            ref_lines=[(median_lift, f"median = {median_lift:.2f}×")],
            color_by="pct_tasks_affected" if color_by_pct else None,
        )
        save_figure(fig, fig_dir / f"{chart_id}.png")

        out_cols = ["category", ratio_col, "lift"]
        if color_by_pct:
            out_cols.append("pct_tasks_affected")
        out = df[out_cols].copy()
        out["anchor_value"] = anchor_val
        out["median_lift"] = median_lift
        save_csv(
            out.sort_values("lift", ascending=False),
            csv_dir / f"{chart_id}.csv",
            float_format="%.4f",
        )
        return df

    # Chart 12 — basic, colored by pct_tasks_affected
    c12 = _anchor_and_plot(
        base,
        "ratio_pct",
        "12_all_confirmed_bias_anchor_basic",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, basic)",
        "Σ pct / Σ (freq × emp), renormalized · rated-task denominator",
        color_by_pct=True,
    )
    print(f"  12 max lift = {c12['lift'].max():.2f}×")

    # Chart 13 — sqrt-den smoothing
    base["ratio_sqrt"] = np.where(
        base["den"] > 0, base["num"] / np.sqrt(base["den"]), 0.0
    )
    total_sqrt = base["ratio_sqrt"].sum()
    base["ratio_sqrt_pct"] = (
        base["ratio_sqrt"] / total_sqrt * 100.0 if total_sqrt > 0 else 0.0
    )
    c13 = _anchor_and_plot(
        base,
        "ratio_sqrt_pct",
        "13_all_confirmed_bias_anchor_sqrt",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, sqrt smoothing)",
        "Σ pct / √(Σ (freq × emp)), renormalized · rated-task denominator",
    )
    print(f"  13 max lift = {c13['lift'].max():.2f}×")

    # Chart 14 — additive smoothing (den + median(den))
    alpha = base.loc[base["den"] > 0, "den"].median()
    base["ratio_add"] = base["num"] / (base["den"] + alpha)
    total_add = base["ratio_add"].sum()
    base["ratio_add_pct"] = (
        base["ratio_add"] / total_add * 100.0 if total_add > 0 else 0.0
    )
    c14 = _anchor_and_plot(
        base,
        "ratio_add_pct",
        "14_all_confirmed_bias_anchor_additive",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, additive smoothing)",
        f"Σ pct / (Σ (freq × emp) + α), α = median den ({alpha:,.0f}), renormalized · rated-task denominator",
    )
    print(f"  14 max lift = {c14['lift'].max():.2f}×")

    # Chart 15 — same numerator as 12 but full-eco denominator
    full_den = compute_major_full_eco_denominator()
    base["den_full"] = base["category"].map(full_den).fillna(0.0)
    base["ratio_full"] = np.where(
        base["den_full"] > 0, base["num"] / base["den_full"], 0.0
    )
    total_full = base["ratio_full"].sum()
    base["ratio_full_pct"] = (
        base["ratio_full"] / total_full * 100.0 if total_full > 0 else 0.0
    )
    c15 = _anchor_and_plot(
        base,
        "ratio_full_pct",
        "15_all_confirmed_bias_anchor_fulleco",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, full eco_2025 denominator)",
        "Σ pct (rated) / Σ (freq × emp) over FULL eco_2025, renormalized",
        color_by_pct=True,
    )
    print(f"  15 max lift = {c15['lift'].max():.2f}×")

    # Charts 16 & 17 — auto-aug-weighted versions of charts 12 and 15.
    # Numerator now sums adj_pct × (auto_aug_mean / 5) per (task, occ): tasks
    # flagged as more automatable contribute more; low-auto-aug tasks shrink.
    # 16 keeps chart 12's rated-task denominator. 17 keeps chart 15's full-eco.
    print("Producing chart 16 & 17 (auto-aug-weighted variants of 12 & 15) ...")
    base_aa = compute_v3_intensity(
        "all_confirmed",
        compute_bias_ratios(BIAS_VARIANTS["equal"]),
        weight_by_auto_aug=True,
    ).copy()
    # Map pct_tasks_affected onto base_aa for color shading (same as base).
    base_aa["pct_tasks_affected"] = base_aa["category"].map(pct_aff).fillna(0.0)

    c16 = _anchor_and_plot(
        base_aa,
        "ratio_pct",
        "16_all_confirmed_bias_autoaug_anchor_basic",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, auto-aug weighted, basic)",
        "Σ pct × (auto_aug/5) / Σ (freq × emp), renormalized · rated-task denominator",
        color_by_pct=True,
    )
    print(f"  16 max lift = {c16['lift'].max():.2f}×")

    # 17 keeps the auto-aug-weighted numerator but swaps in chart 15's full-eco
    # denominator (the same full_den series we computed above for chart 15).
    base_aa["den_full"] = base_aa["category"].map(full_den).fillna(0.0)
    base_aa["ratio_full"] = np.where(
        base_aa["den_full"] > 0, base_aa["num"] / base_aa["den_full"], 0.0
    )
    total_aa_full = base_aa["ratio_full"].sum()
    base_aa["ratio_full_pct"] = (
        base_aa["ratio_full"] / total_aa_full * 100.0 if total_aa_full > 0 else 0.0
    )
    c17 = _anchor_and_plot(
        base_aa,
        "ratio_full_pct",
        "17_all_confirmed_bias_autoaug_anchor_fulleco",
        "AI Intensity vs. Median-Rank Anchor — All Confirmed (bias-corrected, auto-aug weighted, full eco_2025 denominator)",
        "Σ pct × (auto_aug/5) / Σ (freq × emp) over FULL eco_2025, renormalized",
        color_by_pct=True,
    )
    print(f"  17 max lift = {c17['lift'].max():.2f}×")

    print(f"\nSaved figures to: {fig_dir}")
    print(f"Saved CSVs to:    {csv_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
