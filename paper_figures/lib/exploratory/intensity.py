"""
run.py — pct_norm vs eco comparison (overhauled).

Two configs × up to 7 levels × 2 eco scopes × up to 5 bias variants.

Output scope (per user spec):
  - major_occ + gwa: BOTH configs × all 5 bias variants × both eco variants
      → 80 charts (40 per level)
  - minor/broad/occ/iwa/dwa: all_confirmed only × equal-consensus bias only × both eco
      → 20 charts (4 per level × 5 levels)
  - Total ≈ 100 PNGs

For each (config, level, bias, eco) combination:
  - AI bar = Σ pct_normalized (optionally bias-corrected per task), renormed to 100%
  - Eco bar = freq-method emp allocation (config-scoped or full eco_2025), renormed to 100%
  - Overlay chart + signed-delta chart

Bias correction:
  consensus_share[gwa] = weighted mean of {Claude/AEI, Copilot, ChatGPT} share for that
    gwa, excluding sources where the gwa is absent; weights = (1, 1, chatgpt_weight).
  bias_ratio[gwa] = claude_share[gwa] / consensus_share[gwa]
  Each task's pct is divided by bias_ratio[gwa]. At occ-hierarchy levels a task can
  map to multiple GWAs, so we average the bias_ratio across those GWAs per (task, occ).

Run (from project root):
    venv/Scripts/python -m analysis.exploratory.pct_norm_vs_eco.run
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.config import ROOT, ensure_results_dir
from lib.utils import COLORS, FONT_FAMILY, save_csv, save_figure

HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ── Configs ──────────────────────────────────────────────────────────────────

CONFIGS = {
    "all_confirmed": {
        "file": "final_all_confirmed_usage_2026-02-12.csv",
        "label": "All Confirmed (AEI Both + Micro 2026-02-12)",
        "occ_col": "title_current",
    },
    "aei_all_usage": {
        "file": "final_aei_all_usage_2026-02-12.csv",
        "label": "AEI All Usage (Conv + API 2026-02-12)",
        "occ_col": "title",
    },
}

ECO_FILE = "final_eco_2025.csv"
EMP_COL = "emp_tot_nat_2025"

# ── GWA name map: AEI dataset (eco_2015 style) → canonical eco_2025 ──────────

AEI_GWA_RENAME = {
    "Interacting With Computers": "Working with Computers",
    "Provide Consultation and Advice to Others": "Providing Consultation and Advice to Others",
    "Communicating with Persons Outside Organization": "Communicating with People Outside the Organization",
    "Monitor Processes, Materials, or Surroundings": "Monitoring Processes, Materials, or Surroundings",
    "Inspecting Equipment, Structures, or Material": "Inspecting Equipment, Structures, or Materials",
    "Judging the Qualities of Things, Services, or People": "Judging the Qualities of Objects, Services, or People",
}

# ── Source GWA distributions (hardcoded from user input) ─────────────────────
# All keyed by canonical eco_2025 GWA names. Renormalized to sum to 100 below.

CLAUDE_SHARE_RAW = {
    "Thinking Creatively": 33.7,
    "Working with Computers": 11.3,
    "Documenting/Recording Information": 8.7,
    "Analyzing Data or Information": 8.4,
    "Providing Consultation and Advice to Others": 4.0,
    "Training and Teaching Others": 3.7,
    "Making Decisions and Solving Problems": 3.7,
    "Getting Information": 3.6,
    "Inspecting Equipment, Structures, or Materials": 2.7,
    "Developing Objectives and Strategies": 2.4,
    "Judging the Qualities of Objects, Services, or People": 2.2,
    "Interpreting the Meaning of Information for Others": 2.0,
    "Guiding, Directing, and Motivating Subordinates": 2.0,
    "Communicating with Supervisors, Peers, or Subordinates": 1.8,
    "Performing for or Working Directly with the Public": 1.4,
    "Processing Information": 1.3,
    "Communicating with People Outside the Organization": 0.9,
    "Repairing and Maintaining Mechanical Equipment": 0.9,
    "Updating and Using Relevant Knowledge": 0.8,
    "Monitoring Processes, Materials, or Surroundings": 0.7,
    "Performing Administrative Activities": 0.6,
    "Assisting and Caring for Others": 0.5,
    "Selling or Influencing Others": 0.5,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 0.4,
    "Handling and Moving Objects": 0.4,
    "Identifying Objects, Actions, and Events": 0.3,
    "Monitoring and Controlling Resources": 0.2,
    "Organizing, Planning, and Prioritizing Work": 0.2,
    "Resolving Conflicts and Negotiating with Others": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Staffing Organizational Units": 0.2,
    "Controlling Machines and Processes": 0.1,
    "Scheduling Work and Activities": 0.1,
    "Establishing and Maintaining Interpersonal Relationships": 0.1,
    "Coaching and Developing Others": 0.1,
    "Performing General Physical Activities": 0.0,
    "Operating Vehicles, Mechanized Devices, or Equipment": 0.0,
}

COPILOT_SHARE_RAW = {
    "Getting Information": 24.3,
    "Communicating with People Outside the Organization": 15.4,
    "Performing for or Working Directly with the Public": 12.7,
    "Assisting and Caring for Others": 8.4,
    "Interpreting the Meaning of Information for Others": 5.2,
    "Documenting/Recording Information": 5.1,
    "Thinking Creatively": 4.4,
    "Providing Consultation and Advice to Others": 3.5,
    "Updating and Using Relevant Knowledge": 3.3,
    "Making Decisions and Solving Problems": 3.2,
    "Working with Computers": 2.7,
    "Communicating with Supervisors, Peers, or Subordinates": 2.2,
    "Analyzing Data or Information": 1.4,
    "Coaching and Developing Others": 1.3,
    "Training and Teaching Others": 1.3,
    "Judging the Qualities of Objects, Services, or People": 1.0,
    "Processing Information": 0.7,
    "Handling and Moving Objects": 0.6,
    "Selling or Influencing Others": 0.6,
    "Performing Administrative Activities": 0.5,
    "Monitoring Processes, Materials, or Surroundings": 0.5,
    "Monitoring and Controlling Resources": 0.3,
    "Performing General Physical Activities": 0.3,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 0.2,
    "Organizing, Planning, and Prioritizing Work": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Inspecting Equipment, Structures, or Materials": 0.1,
    "Developing Objectives and Strategies": 0.1,
    "Controlling Machines and Processes": 0.1,
    "Repairing and Maintaining Mechanical Equipment": 0.1,
    "Identifying Objects, Actions, and Events": 0.0,
    "Establishing and Maintaining Interpersonal Relationships": 0.0,
}
# Copilot is missing: Guiding/Directing/Motivating Subordinates, Operating Vehicles,
# Resolving Conflicts, Scheduling Work, Staffing Organizational Units.

CHATGPT_SHARE_RAW = {
    # Ambiguous (1.1%) and Suppressed (0.1%) dropped before renormalization.
    "Documenting/Recording Information": 18.4,
    "Making Decisions and Solving Problems": 14.9,
    "Thinking Creatively": 13.0,
    "Working with Computers": 10.8,
    "Interpreting the Meaning of Information for Others": 10.1,
    "Getting Information": 9.3,
    "Providing Consultation and Advice to Others": 4.4,
    "Analyzing Data or Information": 3.0,
    "Communicating with Supervisors, Peers, or Subordinates": 2.8,
    "Judging the Qualities of Objects, Services, or People": 2.0,
    "Communicating with People Outside the Organization": 1.4,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 1.0,
    "Performing Administrative Activities": 1.0,
    "Training and Teaching Others": 0.8,
    "Selling or Influencing Others": 0.8,
    "Assisting and Caring for Others": 0.6,
    "Organizing, Planning, and Prioritizing Work": 0.6,
    "Scheduling Work and Activities": 0.5,
    "Developing Objectives and Strategies": 0.4,
    "Processing Information": 0.4,
    "Staffing Organizational Units": 0.3,
    "Updating and Using Relevant Knowledge": 0.3,
    "Resolving Conflicts and Negotiating with Others": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Handling and Moving Objects": 0.2,
    "Coaching and Developing Others": 0.2,
    "Monitoring and Controlling Resources": 0.2,
    "Monitoring Processes, Materials, or Surroundings": 0.2,
    "Identifying Objects, Actions, and Events": 0.2,
    "Establishing and Maintaining Interpersonal Relationships": 0.2,
    "Inspecting Equipment, Structures, or Materials": 0.2,
    "Guiding, Directing, and Motivating Subordinates": 0.2,
    "Performing for or Working Directly with the Public": 0.1,
    "Repairing and Maintaining Mechanical Equipment": 0.1,
    "Performing General Physical Activities": 0.1,
}
# ChatGPT is missing: Operating Vehicles, Controlling Machines and Processes.


def _renorm_100(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    assert total > 0, "Empty distribution"
    return {k: v * 100.0 / total for k, v in d.items()}


CLAUDE_SHARE = _renorm_100(CLAUDE_SHARE_RAW)
COPILOT_SHARE = _renorm_100(COPILOT_SHARE_RAW)
CHATGPT_SHARE = _renorm_100(CHATGPT_SHARE_RAW)

# Canonical GWA list = union of Claude + Copilot + ChatGPT keys; Claude has all 37.
CANONICAL_GWAS = sorted(
    set(CLAUDE_SHARE) | set(COPILOT_SHARE) | set(CHATGPT_SHARE)
)

# ── Bias variants ────────────────────────────────────────────────────────────

BIAS_VARIANTS: dict[str, Optional[tuple[float, float, float]]] = {
    "no_bias": None,
    "equal": (1.0, 1.0, 1.0),
    "chatgpt_2x": (1.0, 1.0, 2.0),
    "chatgpt_5x": (1.0, 1.0, 5.0),
    "chatgpt_10x": (1.0, 1.0, 10.0),
}

BIAS_LABELS = {
    "no_bias": "no bias correction",
    "equal": "equal 3-source consensus",
    "chatgpt_2x": "ChatGPT weight 2×",
    "chatgpt_5x": "ChatGPT weight 5×",
    "chatgpt_10x": "ChatGPT weight 10×",
}


def compute_bias_ratios(
    weights: Optional[tuple[float, float, float]],
) -> Optional[dict[str, float]]:
    """bias_ratio[gwa] = claude_share / consensus_share.

    Missing-source rule: a GWA not listed in a source is excluded from that
    source's contribution to the consensus for that GWA (weight drops to 0).

    Returns None when weights is None (no correction).
    """
    if weights is None:
        return None
    w_cl, w_co, w_gp = weights
    ratios: dict[str, float] = {}
    for gwa in CANONICAL_GWAS:
        parts: list[tuple[float, float]] = []
        if gwa in CLAUDE_SHARE:
            parts.append((CLAUDE_SHARE[gwa], w_cl))
        if gwa in COPILOT_SHARE:
            parts.append((COPILOT_SHARE[gwa], w_co))
        if gwa in CHATGPT_SHARE:
            parts.append((CHATGPT_SHARE[gwa], w_gp))
        w_sum = sum(w for _, w in parts)
        if w_sum <= 0 or gwa not in CLAUDE_SHARE:
            ratios[gwa] = 1.0
            continue
        consensus = sum(s * w for s, w in parts) / w_sum
        ratios[gwa] = CLAUDE_SHARE[gwa] / consensus if consensus > 0 else 1.0
    return ratios


# ── Data loading ─────────────────────────────────────────────────────────────

_CONFIG_CACHE: dict[str, pd.DataFrame] = {}
_ECO_CACHE: Optional[pd.DataFrame] = None


def load_config_df(config_key: str) -> pd.DataFrame:
    if config_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_key]
    cfg = CONFIGS[config_key]
    path = DATA_DIR / cfg["file"]
    assert path.exists(), f"Missing config file: {path}"
    occ_col = cfg["occ_col"]
    usecols = [
        "task_normalized", occ_col,
        "major_occ_category", "minor_occ_category", "broad_occ",
        "gwa_title", "iwa_title", "dwa_title",
        "pct_normalized", "auto_aug_mean", "freq_mean", EMP_COL,
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    assert not df.empty, f"Empty: {path}"
    for c in ("pct_normalized", "auto_aug_mean", "freq_mean", EMP_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if config_key == "aei_all_usage":
        df["gwa_title"] = df["gwa_title"].replace(AEI_GWA_RENAME)
    _CONFIG_CACHE[config_key] = df
    return df


def load_eco_df() -> pd.DataFrame:
    global _ECO_CACHE
    if _ECO_CACHE is not None:
        return _ECO_CACHE
    path = DATA_DIR / ECO_FILE
    assert path.exists(), f"Missing eco file: {path}"
    usecols = [
        "task_normalized", "title_current",
        "major_occ_category", "minor_occ_category", "broad_occ",
        "gwa_title", "iwa_title", "dwa_title",
        "freq_mean", EMP_COL,
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for c in ("freq_mean", EMP_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _ECO_CACHE = df
    return df


# ── AI bar computation (per config × level × bias) ───────────────────────────

WA_LEVELS = {"gwa": "gwa_title", "iwa": "iwa_title", "dwa": "dwa_title"}
OCC_HIER_LEVELS = {
    "major_occ": "major_occ_category",
    "minor_occ": "minor_occ_category",
    "broad_occ": "broad_occ",
    "occupation": None,  # resolved per config via occ_col
}


def _level_col(level_key: str, config_key: str) -> str:
    if level_key in WA_LEVELS:
        return WA_LEVELS[level_key]
    if level_key == "occupation":
        return CONFIGS[config_key]["occ_col"]
    return OCC_HIER_LEVELS[level_key]


def compute_ai_bar(
    config_key: str,
    level_key: str,
    bias_ratios: Optional[dict[str, float]],
) -> pd.DataFrame:
    """Return DataFrame: category, ai_sum, ai_pct."""
    df = load_config_df(config_key)
    col = _level_col(level_key, config_key)
    occ_col = CONFIGS[config_key]["occ_col"]
    ai = df[df["pct_normalized"].notna() & df[col].notna()].copy()

    if level_key in WA_LEVELS:
        # Unique (wa, task) pairs; bias uses row's gwa_title.
        keep_cols = list({col, "task_normalized", "pct_normalized", "gwa_title"})
        dedup = ai.drop_duplicates([col, "task_normalized"])[keep_cols].copy()
        if bias_ratios is not None:
            dedup["bias"] = dedup["gwa_title"].map(bias_ratios).fillna(1.0)
            dedup["adj"] = dedup["pct_normalized"] / dedup["bias"].replace(0.0, 1.0)
        else:
            dedup["adj"] = dedup["pct_normalized"]
        grp = dedup.groupby(col)["adj"].sum().reset_index(name="ai_sum")
    else:
        # Occ-hierarchy: unique (task, occ) pairs. Bias = mean bias_ratio across
        # all GWAs the (task, occ) maps to.
        if bias_ratios is not None:
            gwa_pairs = (
                ai.dropna(subset=["gwa_title"])
                .drop_duplicates(["task_normalized", occ_col, "gwa_title"])[
                    ["task_normalized", occ_col, "gwa_title"]
                ]
                .copy()
            )
            gwa_pairs["bias"] = gwa_pairs["gwa_title"].map(bias_ratios).fillna(1.0)
            avg = (
                gwa_pairs.groupby(["task_normalized", occ_col])["bias"]
                .mean()
                .reset_index(name="avg_bias")
            )
        keep_cols = list({"task_normalized", occ_col, col, "pct_normalized"})
        dedup = ai.drop_duplicates(["task_normalized", occ_col])[keep_cols].copy()
        if bias_ratios is not None:
            dedup = dedup.merge(avg, on=["task_normalized", occ_col], how="left")
            dedup["avg_bias"] = dedup["avg_bias"].fillna(1.0).replace(0.0, 1.0)
            dedup["adj"] = dedup["pct_normalized"] / dedup["avg_bias"]
        else:
            dedup["adj"] = dedup["pct_normalized"]
        grp = dedup.groupby(col)["adj"].sum().reset_index(name="ai_sum")

    total = grp["ai_sum"].sum()
    grp["ai_pct"] = grp["ai_sum"] / total * 100.0 if total > 0 else 0.0
    return grp.rename(columns={col: "category"})[["category", "ai_sum", "ai_pct"]]


# ── Eco bar computation (per source × level) ─────────────────────────────────

def _compute_eco_bar_from(df: pd.DataFrame, level_key: str, occ_col: str) -> pd.DataFrame:
    """Compute eco distribution per category from a source dataframe."""
    col = WA_LEVELS.get(level_key, None) or (
        occ_col if level_key == "occupation" else OCC_HIER_LEVELS[level_key]
    )
    d = df.copy()
    d["freq_mean"] = d["freq_mean"].fillna(0.0)
    d[EMP_COL] = d[EMP_COL].fillna(0.0)

    if level_key in WA_LEVELS:
        # Freq-method emp allocation per (task, occ); sum over (wa, task, occ) triples.
        pairs = d.drop_duplicates(["task_normalized", occ_col])[
            ["task_normalized", occ_col, "freq_mean", EMP_COL]
        ].copy()
        pairs["freq_sum_in_occ"] = pairs.groupby(occ_col)["freq_mean"].transform("sum")
        pairs["freq_frac"] = np.where(
            pairs["freq_sum_in_occ"] > 0,
            pairs["freq_mean"] / pairs["freq_sum_in_occ"],
            0.0,
        )
        pairs["emp_per_task"] = pairs["freq_frac"] * pairs[EMP_COL]
        triples = (
            d.dropna(subset=[col])
            .drop_duplicates([col, "task_normalized", occ_col])[
                [col, "task_normalized", occ_col]
            ]
            .merge(
                pairs[["task_normalized", occ_col, "emp_per_task"]],
                on=["task_normalized", occ_col],
                how="left",
            )
        )
        grp = triples.groupby(col)["emp_per_task"].sum().reset_index(name="eco_sum")
    else:
        pairs = d.drop_duplicates(["task_normalized", occ_col]).copy()
        pairs["eco_weight"] = pairs["freq_mean"] * pairs[EMP_COL]
        grp = pairs.groupby(col)["eco_weight"].sum().reset_index(name="eco_sum")

    total = grp["eco_sum"].sum()
    grp["eco_pct"] = grp["eco_sum"] / total * 100.0 if total > 0 else 0.0
    return grp.rename(columns={col: "category"})[["category", "eco_sum", "eco_pct"]]


def compute_eco_bar_configscoped(config_key: str, level_key: str) -> pd.DataFrame:
    df = load_config_df(config_key)
    occ_col = CONFIGS[config_key]["occ_col"]
    # Config-scoped eco = freq×emp computed over the config's own rows
    # (all pct-rated rows by construction).
    return _compute_eco_bar_from(df, level_key, occ_col)


def compute_eco_bar_full(level_key: str) -> pd.DataFrame:
    # Full eco = freq×emp computed over the full eco_2025 universe.
    df = load_eco_df()
    return _compute_eco_bar_from(df, level_key, "title_current")


# ── Chart helpers ────────────────────────────────────────────────────────────

def _overlay_bar(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    top_n: Optional[int] = None,
) -> go.Figure:
    """Horizontal grouped bar: AI pct vs Eco pct per category."""
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.nlargest(top_n, "ai_pct")
    plot_df = plot_df.sort_values("ai_pct", ascending=True)
    max_val = max(plot_df["ai_pct"].max(), plot_df["eco_pct"].max())
    x_pad = max_val * 0.18 if max_val else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["category"],
        x=plot_df["ai_pct"],
        name="AI usage — Σ pct_norm (% of total)",
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}" for v in plot_df["ai_pct"]],
        textposition="outside",
        textfont=dict(size=11, family=FONT_FAMILY, color=COLORS["text"]),
    ))
    fig.add_trace(go.Bar(
        y=plot_df["category"],
        x=plot_df["eco_pct"],
        name="Economic baseline (% of total)",
        orientation="h",
        marker_color=COLORS["secondary"],
        text=[f"{v:.1f}" for v in plot_df["eco_pct"]],
        textposition="outside",
        textfont=dict(size=11, family=FONT_FAMILY, color=COLORS["text"]),
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub style='color:#5a5a5a'>{subtitle}</sub>",
            x=0.02, xanchor="left", y=0.97,
            font=dict(family=FONT_FAMILY, size=18, color=COLORS["text"]),
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        xaxis=dict(
            title="% of respective total",
            range=[0, max_val + x_pad],
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, automargin=True),
        font=dict(family=FONT_FAMILY, size=12, color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        height=max(560, 28 * len(plot_df) + 200),
        width=1400,
        margin=dict(l=320, r=80, t=120, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=12)),
    )
    return fig


def _delta_bar(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    split_top_n: Optional[int] = None,
) -> go.Figure:
    """Horizontal bar of (AI pct − Eco pct). If split_top_n given, show that
    many most-positive and that many most-negative deltas."""
    plot_df = df.copy()
    if split_top_n is not None:
        top_pos = plot_df.nlargest(split_top_n, "delta_pct")
        top_neg = plot_df.nsmallest(split_top_n, "delta_pct")
        plot_df = pd.concat([top_pos, top_neg]).drop_duplicates("category")
    plot_df = plot_df.sort_values("delta_pct", ascending=True)
    colors = [
        COLORS["positive"] if v > 0 else COLORS["negative"]
        for v in plot_df["delta_pct"]
    ]
    max_abs = plot_df["delta_pct"].abs().max()
    pad = max_abs * 0.22 if max_abs else 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["category"],
        x=plot_df["delta_pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in plot_df["delta_pct"]],
        textposition="outside",
        textfont=dict(size=11, family=FONT_FAMILY, color=COLORS["text"]),
    ))
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub style='color:#5a5a5a'>{subtitle}</sub>",
            x=0.02, xanchor="left", y=0.97,
            font=dict(family=FONT_FAMILY, size=18, color=COLORS["text"]),
        ),
        xaxis=dict(
            title="AI pct − Eco pct (percentage points)",
            range=[-max_abs - pad, max_abs + pad],
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=True,
            zerolinecolor=COLORS["neutral"],
            zerolinewidth=1,
        ),
        yaxis=dict(showgrid=False, automargin=True),
        font=dict(family=FONT_FAMILY, size=12, color=COLORS["text"]),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        height=max(560, 28 * len(plot_df) + 200),
        width=1400,
        margin=dict(l=320, r=80, t=120, b=70),
        showlegend=False,
    )
    return fig


# ── Output matrix ────────────────────────────────────────────────────────────

LEVEL_DISPLAY = {
    "major_occ": "Major Occupational Categories",
    "minor_occ": "Minor Occupational Categories",
    "broad_occ": "Broad Occupations",
    "occupation": "Occupations",
    "gwa": "General Work Activities",
    "iwa": "Intermediate Work Activities",
    "dwa": "Detailed Work Activities",
}

# top_n only applied for overlay. None = show all.
LEVEL_TOPN = {
    "major_occ": None,
    "minor_occ": None,
    "broad_occ": 40,
    "occupation": 40,
    "gwa": None,
    "iwa": 40,
    "dwa": 40,
}
# delta split (top-N over + top-N under); None = show all.
LEVEL_DELTA_SPLIT = {
    "major_occ": None,
    "minor_occ": None,
    "broad_occ": 20,
    "occupation": 20,
    "gwa": None,
    "iwa": 20,
    "dwa": 20,
}


def build_output_matrix() -> list[tuple[str, str, str, str]]:
    """Return list of (level_key, config_key, bias_key, eco_scope) tuples to produce."""
    out: list[tuple[str, str, str, str]] = []
    full_bias = list(BIAS_VARIANTS.keys())  # all 5
    eco_scopes = ["configscoped", "full"]
    # major_occ + gwa: both configs × all bias × both eco
    for level in ("major_occ", "gwa"):
        for cfg in CONFIGS.keys():
            for b in full_bias:
                for e in eco_scopes:
                    out.append((level, cfg, b, e))
    # Other levels: all_confirmed × equal only × both eco
    for level in ("minor_occ", "broad_occ", "occupation", "iwa", "dwa"):
        for e in eco_scopes:
            out.append((level, "all_confirmed", "equal", e))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results_dir = ensure_results_dir(HERE)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    print("Loading datasets …")
    for cfg in CONFIGS:
        load_config_df(cfg)
    load_eco_df()

    # Precompute bias_ratios for each variant
    bias_ratio_cache = {
        b: compute_bias_ratios(w) for b, w in BIAS_VARIANTS.items()
    }

    # Save bias ratios CSV for reference
    bias_rows = []
    for gwa in CANONICAL_GWAS:
        row = {"gwa": gwa,
               "claude_share": CLAUDE_SHARE.get(gwa, np.nan),
               "copilot_share": COPILOT_SHARE.get(gwa, np.nan),
               "chatgpt_share": CHATGPT_SHARE.get(gwa, np.nan)}
        for b, ratios in bias_ratio_cache.items():
            row[f"bias_ratio_{b}"] = ratios.get(gwa, 1.0) if ratios else 1.0
        bias_rows.append(row)
    bias_df = pd.DataFrame(bias_rows).sort_values("claude_share", ascending=False)
    save_csv(bias_df, results_dir / "bias_ratios.csv", float_format="%.4f")

    # Cache AI bars: keyed by (config, level, bias) → DataFrame(category, ai_sum, ai_pct)
    ai_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    # Cache Eco bars: (config, level, scope) or (None, level, "full")
    eco_cache: dict[tuple[Optional[str], str, str], pd.DataFrame] = {}

    matrix = build_output_matrix()
    total = len(matrix)
    print(f"Producing {total * 2} charts (overlay + delta per combination) …")

    # Also aggregate CSVs per (config, level): all variants in one file.
    csv_agg: dict[tuple[str, str], pd.DataFrame] = {}

    for i, (level, cfg, bias, scope) in enumerate(matrix, 1):
        # AI bar
        ai_key = (cfg, level, bias)
        if ai_key not in ai_cache:
            ai_cache[ai_key] = compute_ai_bar(cfg, level, bias_ratio_cache[bias])
        ai_df = ai_cache[ai_key]

        # Eco bar
        eco_key = (cfg, level, scope) if scope == "configscoped" else (None, level, "full")
        if eco_key not in eco_cache:
            if scope == "configscoped":
                eco_cache[eco_key] = compute_eco_bar_configscoped(cfg, level)
            else:
                eco_cache[eco_key] = compute_eco_bar_full(level)
        eco_df = eco_cache[eco_key]

        merged = ai_df.merge(eco_df, on="category", how="outer").fillna(0.0)
        merged["delta_pct"] = merged["ai_pct"] - merged["eco_pct"]

        # Update CSV aggregate
        agg_key = (cfg, level)
        col_ai = f"ai_pct_{bias}"
        col_eco = f"eco_pct_{scope}"
        col_delta = f"delta_pct_{bias}_{scope}"
        if agg_key not in csv_agg:
            csv_agg[agg_key] = merged[["category"]].copy()
        agg = csv_agg[agg_key]
        if col_ai not in agg.columns:
            agg = agg.merge(merged[["category", "ai_pct"]].rename(columns={"ai_pct": col_ai}),
                            on="category", how="outer")
        if col_eco not in agg.columns:
            agg = agg.merge(merged[["category", "eco_pct"]].rename(columns={"eco_pct": col_eco}),
                            on="category", how="outer")
        agg = agg.merge(merged[["category", "delta_pct"]].rename(columns={"delta_pct": col_delta}),
                        on="category", how="outer")
        csv_agg[agg_key] = agg

        # Figures
        cfg_label = CONFIGS[cfg]["label"]
        level_disp = LEVEL_DISPLAY[level]
        scope_disp = "config-scoped eco" if scope == "configscoped" else "full eco_2025"
        subtitle = f"{cfg_label} · {BIAS_LABELS[bias]} · {scope_disp}"

        base = f"{level}_{cfg}_{bias}_{scope}"
        fig_ov = _overlay_bar(
            merged, f"AI Usage vs. Economic Baseline — {level_disp}",
            subtitle, top_n=LEVEL_TOPN[level],
        )
        save_figure(fig_ov, figures_dir / f"{base}_overlay.png")

        fig_dl = _delta_bar(
            merged, f"AI Over / Under-Indexing — {level_disp}",
            f"{subtitle}. AI pct − Eco pct. Green = AI over-indexed, red = under-indexed.",
            split_top_n=LEVEL_DELTA_SPLIT[level],
        )
        save_figure(fig_dl, figures_dir / f"{base}_delta.png")

        if i % 10 == 0 or i == total:
            print(f"  {i}/{total}  {base}")

    # Write aggregated CSVs
    for (cfg, level), df in csv_agg.items():
        save_csv(
            df.sort_values(df.columns[1] if len(df.columns) > 1 else "category",
                           ascending=False),
            results_dir / f"{level}_{cfg}_bars.csv",
            float_format="%.4f",
        )

    print(f"\nSaved figures to: {figures_dir}")
    print(f"Saved CSVs to:    {results_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
