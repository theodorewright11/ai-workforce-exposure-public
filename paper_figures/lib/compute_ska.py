"""
compute_ska.py — Real-time SKA gap computation for analysis.

Given a pct_tasks_affected Series (occupation title → 0-100), computes the
Skills / Abilities / Knowledge gap between AI capability and individual
occupation requirements.

Formula (locked-in spec)
------------------------
- Filter: only (occ, element) rows where importance >= 3
- Occ score per element:   importance × level  (the occ's own values)
- AI product per row:      (pct_tasks_affected / 100) × importance × level
- AI capability per element: 95th percentile of AI product across occupations
  (see analysis/data/ai_capability_method_comparison.ipynb for the three-line
   defense of 95th pct: demonstrated-capability floor, outlier stability,
   bootstrap robustness)
- Gap per (occ, element):  AI capability score − occ score  (raw units)
- AI-as-% per (occ, element): AI capability / occ_score × 100  (percentage frame)
- Eco baseline per element: mean of (importance × level) across all occs,
  no pct weighting — reference only, not used in gap calculation

Per-occupation summaries (two framings, both emitted):
- Raw gap summary (legacy): mean gap across elements per type, then overall
  mean gap across types.
- Percentage summary (ratio-of-sums): sum(ai_capability) / sum(occ_score)
  across the occ's importance>=3 elements, per type and overall.
  Above 100% = AI leads the occ's requirement; below 100% = human advantage.
  Ratio-of-sums (not mean of ratios) — consistent with the dashboard's
  pct_tasks_affected pattern and avoids small-occ_score outliers dominating.

The AI capability score is recomputed fresh from the pct_tasks_affected input,
so it varies by dataset config. Nothing is pre-cached here.

Typical usage
-------------
    from lib.config import get_pct_tasks_affected, ANALYSIS_CONFIGS
    from lib.compute_ska import load_ska_data, compute_ska

    ska_data = load_ska_data()                          # load once, reuse

    pct = get_pct_tasks_affected(ANALYSIS_CONFIGS["all_ceiling"])
    result = compute_ska(pct, ska_data)

    # result.occ_gaps   — DataFrame: title_current, skills_gap, abilities_gap,
    #                                 knowledge_gap, overall_gap
    # result.ai_capability — DataFrame: element_name, type, ai_score
    # result.eco_baseline  — DataFrame: element_name, type, eco_score
    # result.occ_element_scores — dict[type → DataFrame with per-element detail]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# This file lives at <repo_root>/paper_figures/lib/compute_ska.py; the O*NET
# SKA reference CSVs live under <repo_root>/data/reference/.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "reference"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────

IMPORTANCE_THRESHOLD: float = 3.0   # include only elements rated >= this
AI_PERCENTILE: int = 95             # percentile used for AI capability

ELEMENT_FILES: dict[str, Path] = {
    "skills":    DATA_DIR / "skills_v30.1.csv",
    "abilities": DATA_DIR / "abilities_v30.1.csv",
    "knowledge": DATA_DIR / "knowledge_v30.1.csv",
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SKAData:
    """Raw O*NET SKA data, loaded once and reused across compute calls."""
    skills: pd.DataFrame
    abilities: pd.DataFrame
    knowledge: pd.DataFrame


@dataclass
class SKAResult:
    """Output of compute_ska()."""
    # 95th-percentile AI capability per element (one row per element × type)
    ai_capability: pd.DataFrame
    # Mean occ score per element — reference baseline (mean of imp × level across occs)
    eco_baseline: pd.DataFrame
    # 95th percentile of occ_score per element — alternate "top practitioners" baseline
    eco_baseline_p95: pd.DataFrame
    # Per-occupation gap summary (raw gap per type + overall, AND pct-of-occ per type + overall)
    #   Columns: title_current, skills_gap, abilities_gap, knowledge_gap, overall_gap,
    #            skills_pct, abilities_pct, knowledge_pct, overall_pct
    occ_gaps: pd.DataFrame
    # Per-occupation × per-element detail, keyed by type
    # Each DF has columns: title_current, element_name, occ_score, ai_product, ai_score, gap
    occ_element_scores: dict[str, pd.DataFrame]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_onet_file(path: Path) -> pd.DataFrame:
    """
    Load an O*NET Skills / Abilities / Knowledge file and pivot to produce
    one row per (soc_code, title, element_name) with columns:
        importance  (Scale ID = IM, 1-5)
        level       (Scale ID = LV, 0-7)
    """
    assert path.exists(), f"O*NET file not found: {path}"
    df = pd.read_csv(path, dtype=str)

    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code",
        "Title":          "title",
        "Element Name":   "element_name",
        "Scale ID":       "scale_id",
        "Data Value":     "data_value",
    })

    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["scale_id"].isin(["IM", "LV"])].copy()

    pivoted = (
        df.pivot_table(
            index=["soc_code", "title", "element_name"],
            columns="scale_id",
            values="data_value",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"})

    # Drop rows missing either scale
    pivoted = pivoted.dropna(subset=["importance", "level"])
    return pivoted


def load_ska_data() -> SKAData:
    """Load all three O*NET SKA files. Call once; pass SKAData to compute_ska()."""
    return SKAData(
        skills=_load_onet_file(ELEMENT_FILES["skills"]),
        abilities=_load_onet_file(ELEMENT_FILES["abilities"]),
        knowledge=_load_onet_file(ELEMENT_FILES["knowledge"]),
    )


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_type(
    onet_df: pd.DataFrame,
    pct_tasks_affected: pd.Series,
    type_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute AI capability, eco baseline (mean + p95), and per-occ element data.

    Parameters
    ----------
    onet_df : pivoted O*NET DataFrame for this type
    pct_tasks_affected : Series keyed by title_current (0-100)
    type_name : "skills" | "abilities" | "knowledge"

    Returns
    -------
    ai_capability    : DataFrame[element_name, ai_score]
    eco_baseline     : DataFrame[element_name, eco_score]        — mean occ_score
    eco_baseline_p95 : DataFrame[element_name, eco_score_p95]    — 95th pct occ_score
    occ_element      : DataFrame[title_current, element_name, occ_score,
                                  ai_product, ai_score, gap]
    """
    df = onet_df.copy()

    # Map pct onto each occ row (O*NET "title" matches title_current in eco_2025)
    df["pct"] = df["title"].map(pct_tasks_affected)
    df = df.dropna(subset=["pct", "importance", "level"])

    # Filter to elements with importance >= threshold
    df = df[df["importance"] >= IMPORTANCE_THRESHOLD].copy()

    assert len(df) > 0, (
        f"No {type_name} rows remain after importance >= {IMPORTANCE_THRESHOLD} filter. "
        "Check that pct_tasks_affected titles match O*NET titles."
    )

    # Occ score = importance × level
    df["occ_score"] = df["importance"] * df["level"]

    # AI product = (pct / 100) × importance × level
    df["ai_product"] = (df["pct"] / 100.0) * df["occ_score"]

    # AI capability per element: 95th percentile across occupations
    ai_cap = (
        df.groupby("element_name")["ai_product"]
        .quantile(AI_PERCENTILE / 100.0)
        .reset_index()
        .rename(columns={"ai_product": "ai_score"})
    )

    # Eco baseline per element: mean of occ_score (no pct weighting) — primary
    eco_base = (
        df.groupby("element_name")["occ_score"]
        .mean()
        .reset_index()
        .rename(columns={"occ_score": "eco_score"})
    )

    # Eco baseline (p95): 95th percentile of occ_score per element — alternate
    # "top practitioners" reference for economy-level charts
    eco_base_p95 = (
        df.groupby("element_name")["occ_score"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"occ_score": "eco_score_p95"})
    )

    # Per-occ element detail
    occ_elem = df[["title", "element_name", "occ_score", "ai_product"]].copy()
    occ_elem = occ_elem.rename(columns={"title": "title_current"})
    occ_elem = occ_elem.merge(ai_cap, on="element_name", how="left")
    occ_elem["gap"] = occ_elem["ai_score"] - occ_elem["occ_score"]

    return ai_cap, eco_base, eco_base_p95, occ_elem


def compute_ska(
    pct_tasks_affected: pd.Series,
    ska_data: SKAData,
) -> SKAResult:
    """
    Compute SKA gap analysis for a given pct_tasks_affected mapping.

    Parameters
    ----------
    pct_tasks_affected : Series keyed by title_current, values 0-100.
    ska_data : Loaded O*NET data from load_ska_data().

    Returns
    -------
    SKAResult
    """
    assert len(pct_tasks_affected) > 0, "pct_tasks_affected must not be empty"

    type_map: dict[str, pd.DataFrame] = {
        "skills":    ska_data.skills,
        "abilities": ska_data.abilities,
        "knowledge": ska_data.knowledge,
    }

    ai_caps: dict[str, pd.DataFrame] = {}
    eco_bases: dict[str, pd.DataFrame] = {}
    eco_bases_p95: dict[str, pd.DataFrame] = {}
    occ_elems: dict[str, pd.DataFrame] = {}

    for type_name, onet_df in type_map.items():
        ai_cap, eco_base, eco_base_p95, occ_elem = _compute_type(
            onet_df, pct_tasks_affected, type_name
        )
        ai_cap["type"] = type_name
        eco_base["type"] = type_name
        eco_base_p95["type"] = type_name
        ai_caps[type_name] = ai_cap
        eco_bases[type_name] = eco_base
        eco_bases_p95[type_name] = eco_base_p95
        occ_elems[type_name] = occ_elem

    # Combined reference tables
    ai_capability = pd.concat(ai_caps.values(), ignore_index=True)
    eco_baseline = pd.concat(eco_bases.values(), ignore_index=True)
    eco_baseline_p95 = pd.concat(eco_bases_p95.values(), ignore_index=True)

    # Per-occ summary:
    #   raw gap   = mean of (ai_score − occ_score) across elements per type
    #   pct       = ratio of sums, sum(ai_score) / sum(occ_score) × 100 per type
    #               (consistent with dashboard ratio-of-totals pattern)
    per_type_frames: dict[str, pd.DataFrame] = {}
    for type_name, occ_elem in occ_elems.items():
        grouped = occ_elem.groupby("title_current")
        gap_mean = grouped["gap"].mean().rename(f"{type_name}_gap")
        sum_ai = grouped["ai_score"].sum()
        sum_occ = grouped["occ_score"].sum()
        pct = (sum_ai / sum_occ.replace(0, np.nan) * 100.0).rename(f"{type_name}_pct")
        per_type_frames[type_name] = pd.concat([gap_mean, pct], axis=1)

    occ_gaps = pd.concat(per_type_frames.values(), axis=1).reset_index()
    # overall_gap = mean of the three per-type gaps (legacy — kept for backward compat)
    occ_gaps["overall_gap"] = occ_gaps[
        ["skills_gap", "abilities_gap", "knowledge_gap"]
    ].mean(axis=1)
    # overall_pct = ratio of sums across ALL types combined (primary, percentage framing)
    #   numerator  = sum(ai_score) across skills+abilities+knowledge qualifying elements
    #   denom      = sum(occ_score) across the same elements
    all_elems = pd.concat(
        [df[["title_current", "occ_score", "ai_score"]] for df in occ_elems.values()],
        ignore_index=True,
    )
    overall = all_elems.groupby("title_current").agg(
        _sum_ai=("ai_score", "sum"),
        _sum_occ=("occ_score", "sum"),
    )
    overall["overall_pct"] = (
        overall["_sum_ai"] / overall["_sum_occ"].replace(0, np.nan) * 100.0
    )
    occ_gaps = occ_gaps.merge(
        overall[["overall_pct"]], left_on="title_current", right_index=True, how="left"
    )
    occ_gaps = occ_gaps.sort_values("overall_gap", ascending=False).reset_index(drop=True)

    n_matched = len(occ_gaps)
    n_input = len(pct_tasks_affected)
    if n_matched < n_input * 0.9:
        import warnings
        warnings.warn(
            f"compute_ska: only {n_matched}/{n_input} occupations matched O*NET titles. "
            "Check that pct_tasks_affected index uses title_current strings."
        )

    return SKAResult(
        ai_capability=ai_capability,
        eco_baseline=eco_baseline,
        eco_baseline_p95=eco_baseline_p95,
        occ_gaps=occ_gaps,
        occ_element_scores=occ_elems,
    )


# ── Quick smoke-test ──────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Run a quick check: load data, run compute with dummy pct, print shapes."""
    print("Loading SKA data...")
    ska_data = load_ska_data()
    print(f"  Skills:    {len(ska_data.skills)} rows")
    print(f"  Abilities: {len(ska_data.abilities)} rows")
    print(f"  Knowledge: {len(ska_data.knowledge)} rows")

    # Build a dummy pct series from unique titles in skills file
    titles = ska_data.skills["title"].unique()
    rng = np.random.default_rng(42)
    pct = pd.Series(rng.uniform(0, 80, size=len(titles)), index=titles)

    print(f"\nRunning compute_ska with {len(pct)} occupations (dummy pct)...")
    result = compute_ska(pct, ska_data)

    print(f"  ai_capability rows:    {len(result.ai_capability)}")
    print(f"  eco_baseline rows:     {len(result.eco_baseline)}")
    print(f"  eco_baseline_p95 rows: {len(result.eco_baseline_p95)}")
    print(f"  occ_gaps rows:         {len(result.occ_gaps)}")
    cols = ["title_current", "overall_gap", "overall_pct"]
    print("\nTop 5 occ (AI leads most, highest overall_pct):")
    print(result.occ_gaps.sort_values("overall_pct", ascending=False).head(5)[cols].to_string(index=False))
    print("\nBottom 5 occ (Human advantage largest, lowest overall_pct):")
    print(result.occ_gaps.sort_values("overall_pct", ascending=True).head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    _smoke_test()
