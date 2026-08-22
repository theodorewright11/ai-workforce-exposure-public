"""
risk_score.py — the per-occupation flag table behind the focused set.

`_load_flag_df()` assembles one row per occupation carrying everything the
focused-set chart gates and labels on: exposure level and trend, SKA coverage
level and trend, job zone, major category, BLS 2024-34 employment projection,
and the eight binary risk flags with their weighted score.

The focused set itself applies just two gates (work time exposed >= 67% and a
negative BLS projection); the remaining columns are what the chart labels and
sorts by.

SKA coverage here overrides the p95 definition in `lib.compute_ska` with a
top-N mean (N = TOP_N_MEAN) — see `_compute_ska_overall_pct`.
"""
from __future__ import annotations


import numpy as np
import pandas as pd


from lib.config import ANALYSIS_CONFIG_SERIES, ANALYSIS_CONFIGS, get_pct_tasks_affected
from lib.compute_ska import SKAData, load_ska_data


from lib.exploratory.job_risk_scoring import (
    EXPOSURE_GATE,
    FLAG_WEIGHTS,
    PRIMARY_KEY,
    TECH_SKILLS_FILE,
    _assign_risk_tier,
    _compute_flags,
    _compute_pct_trend,
    _get_structural_data,
)


PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]


# SKA AI-capability aggregation override.
# compute_ska.py uses 95th percentile across occupations per element.
# This audit uses the mean of the top-N ai_product values instead — a higher,
# stricter "what AI demonstrably can do" floor that matches the dashboard's
# per-row top-10 reference. Set TOP_N_MEAN = None to fall back to p95.
TOP_N_MEAN = 10


FLAG6_RULE = "emp_change_pct_2024_2034 < 0"


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
