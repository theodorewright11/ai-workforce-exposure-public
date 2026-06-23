"""usage_intensity.py — debiased AI-usage intensity for the Actual Usage tab.

Generalizes the paper's major-only `compute_v3_intensity`
(paper_figures/lib/exploratory/intensity_v3.py) to every hierarchy level
(major / minor / broad / occupation / task) so the dashboard can let users
drill the hierarchy. Methodology is identical to the paper:

    ratio[group]     = Σ debias_pct / Σ (freq × emp)   over (task, occ) pairs in group
    intensity[group] = ratio[group] / ratio[anchor]    (× the reference category)

where debias_pct = pct_normalized ÷ avg_bias, and avg_bias is the mean
equal-3-source (Claude / ChatGPT / Copilot) GWA bias ratio across the GWAs each
(task, occ) maps to. Fixed dataset: AEI Conv + API pooled onto eco_2025, no
Microsoft (`final_aei_all_usage_2025_2026-02-12.csv`). No config selector, no
trend — this is the one view that does not use the five configs.

The bias ratios + GWA rename are reused verbatim from lib.exploratory.intensity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.exploratory.intensity import (
    AEI_GWA_RENAME,
    BIAS_VARIANTS,
    compute_bias_ratios,
    load_eco_df,
)
from config import DATA_DIR

# Fixed intensity dataset — AEI Conv + API on eco_2025, no Microsoft.
_INTENSITY_FILE = DATA_DIR / "final_aei_all_usage_2025_2026-02-12.csv"
_OCC_COL = "title_current"
_EMP_COL = "emp_tot_nat_2025"

# Reference anchors (paper uses Office and Administrative Support — a near-median
# major — for the economy-wide chart; within-major drill-downs use the level median).
_MAJOR_ANCHOR = "Office and Administrative Support Occupations"

_LEVEL_COL = {
    "major": "major_occ_category",
    "minor": "minor_occ_category",
    "broad": "broad_occ",
    "occupation": _OCC_COL,
    "task": "task_normalized",
}

_df_cache: Optional[pd.DataFrame] = None
_bias_cache: Optional[dict[str, float]] = None


def _load() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    assert _INTENSITY_FILE.exists(), f"Missing intensity dataset: {_INTENSITY_FILE}"
    usecols = [
        "task_normalized", _OCC_COL,
        "major_occ_category", "minor_occ_category", "broad_occ",
        "gwa_title", "pct_normalized", "freq_mean", _EMP_COL,
    ]
    df = pd.read_csv(_INTENSITY_FILE, usecols=usecols, low_memory=False)
    assert not df.empty, f"Empty intensity dataset: {_INTENSITY_FILE}"
    for c in ("pct_normalized", "freq_mean", _EMP_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # eco_2025 file already uses canonical GWA names, but apply the rename
    # defensively in case any eco_2015-style labels slipped through.
    df["gwa_title"] = df["gwa_title"].replace(AEI_GWA_RENAME)
    _df_cache = df
    return df


def _bias_ratios() -> dict[str, float]:
    global _bias_cache
    if _bias_cache is None:
        _bias_cache = compute_bias_ratios(BIAS_VARIANTS["equal"])
    return _bias_cache


def _deduped_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, occ): debiased pct + freq×emp weight + hierarchy."""
    ai = df[df["pct_normalized"].notna()].copy()
    ai["eco_weight"] = ai["freq_mean"].fillna(0.0) * ai[_EMP_COL].fillna(0.0)

    # avg bias ratio across the GWAs each (task, occ) maps to
    bias = _bias_ratios()
    gwa_pairs = (
        ai.dropna(subset=["gwa_title"])
        .drop_duplicates(["task_normalized", _OCC_COL, "gwa_title"])[
            ["task_normalized", _OCC_COL, "gwa_title"]
        ]
        .copy()
    )
    gwa_pairs["bias"] = gwa_pairs["gwa_title"].map(bias).fillna(1.0)
    avg_bias = (
        gwa_pairs.groupby(["task_normalized", _OCC_COL])["bias"].mean()
        .reset_index(name="avg_bias")
    )

    keep = [
        "task_normalized", _OCC_COL,
        "major_occ_category", "minor_occ_category", "broad_occ",
        "pct_normalized", "eco_weight",
    ]
    dedup = ai.drop_duplicates(["task_normalized", _OCC_COL])[keep].copy()
    dedup = dedup.merge(avg_bias, on=["task_normalized", _OCC_COL], how="left")
    dedup["avg_bias"] = dedup["avg_bias"].fillna(1.0).replace(0.0, 1.0)
    dedup["debias_pct"] = dedup["pct_normalized"] / dedup["avg_bias"]
    return dedup


_EMP_FULL = "emp_tot_nat_2025"


def _full_eco_den(level_col: str, parent_col: Optional[str], parent_val: Optional[str]) -> "pd.Series":
    """Σ (freq × emp) per group over the FULL eco_2025 universe (every deduped
    task-occ pair, regardless of whether the usage dataset rated it) — the paper's
    `compute_major_full_eco_denominator`, generalized to any level + parent filter."""
    eco = load_eco_df().copy()  # title_current + hierarchy + gwa/iwa/dwa + freq + emp
    if parent_col and parent_val is not None and parent_col in eco.columns:
        eco = eco[eco[parent_col] == parent_val]
    pairs = eco.dropna(subset=[level_col]).drop_duplicates(
        ["task_normalized", "title_current"]
    ).copy()
    pairs["ew"] = pairs["freq_mean"].fillna(0.0) * pairs[_EMP_FULL].fillna(0.0)
    return pairs.groupby(level_col)["ew"].sum()


def compute_intensity(
    level: str,
    parent_col: Optional[str] = None,
    parent_val: Optional[str] = None,
) -> list[dict]:
    """Intensity rows at `level`, optionally restricted to a parent category.

    Matches the paper's `intensity_anchor_fulleco` exactly:
        ratio[group]     = Σ debias_pct (rated)  /  Σ (freq × emp) over FULL eco_2025
        intensity[group] = ratio[group] / ratio[anchor]   (× the reference)

    The renormalize-to-100 step the paper applies before dividing by the anchor
    cancels in the ratio, so we skip it.

    - `level`        in major|minor|broad|occupation|task
    - `parent_col`   eco column to filter on (major_occ_category|…|title_current)
    - `parent_val`   value to filter on (for hierarchy drill-down)

    Anchor: economy-wide major level uses Office & Admin Support; every other case
    (drill-down or finer level) uses the median ratio across the rows shown.
    """
    assert level in _LEVEL_COL, f"Unknown level: {level}"
    col = _LEVEL_COL[level]

    # numerator — debiased usage mass per group, from the fixed usage dataset
    dedup = _deduped_pairs(_load())
    if parent_col and parent_val is not None and parent_col in dedup.columns:
        dedup = dedup[dedup[parent_col] == parent_val].copy()
    num = (
        dedup.dropna(subset=[col])
        .groupby(col)
        .agg(num=("debias_pct", "sum"), raw_pct=("pct_normalized", "sum"))
    )

    # denominator — full eco_2025 freq×emp per group
    den_full = _full_eco_den(col, parent_col, parent_val)

    grp = num.join(den_full.rename("den_full"), how="left").reset_index()
    grp = grp.rename(columns={col: "category"})
    grp["den_full"] = grp["den_full"].fillna(0.0)
    grp["ratio"] = np.where(grp["den_full"] > 0, grp["num"] / grp["den_full"], 0.0)

    # Anchor (x = 1 reference). The economy-wide major view anchors on Office &
    # Admin Support (matches the paper's main intensity chart exactly). Every
    # drill-down / finer level re-anchors on the MEDIAN ratio of the rows shown —
    # ratios aren't comparable across aggregation grains (a single niche
    # occupation has a far more concentrated ratio than a major aggregate), so a
    # global anchor would make finer levels unreadable. This mirrors the paper's
    # within-major driver charts ("× the median occupation/task in this major").
    anchor: Optional[float] = None
    if level == "major" and parent_val is None:
        match = grp.loc[grp["category"] == _MAJOR_ANCHOR, "ratio"]
        if not match.empty and float(match.iloc[0]) > 0:
            anchor = float(match.iloc[0])
    if anchor is None:
        nz = grp.loc[grp["ratio"] > 0, "ratio"]
        anchor = float(nz.median()) if not nz.empty else 1.0
    anchor = anchor if anchor > 0 else 1.0

    grp["intensity"] = grp["ratio"] / anchor
    grp = grp.sort_values("intensity", ascending=False)
    return [
        {
            "category": str(r["category"]),
            "intensity": float(r["intensity"]),
            "raw_pct": float(r["raw_pct"]),
            "ratio": float(r["ratio"]),
        }
        for _, r in grp.iterrows()
    ]
