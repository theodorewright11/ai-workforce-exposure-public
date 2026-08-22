"""
job_risk_scoring.py — the eight binary risk flags and their weighted score.

One row per occupation, max score 10. Used by `lib.exploratory.risk_score` to
build the focused-set flag table.

Exposure depth (weight 2 each — "AI is actually reaching this job"):
  1. pct_tasks_affected > 50%
  2. overall_ska_pct > median

Exposure velocity (weight 1 each — "and it is still growing"):
  3. pct_delta > 0 AND > median(pct_delta)
  4. ska_delta > 0 AND > median(ska_delta)

Exposure depth (weight 1 — complementary to flag 1):
  8. auto_avg_with_vals > median

Structural vulnerability (weight 1 each — the safety net is thin):
  5. job_zone in {1, 2, 3}
  6. outlook in {2, 3}          (1 = good outlook / low wages — NOT at risk)
  7. n_software > median

Risk tiers on the 0-10 score: Low 0-2, Mod-Low 3-4, Mod-High 5-7, High 8-10.
A score of 8+ with pct < 33% is downgraded to Mod-High (EXPOSURE_GATE):
without meaningful task exposure, structural and trend signals alone cannot
justify a "high risk" label on a job AI has not reached.
"""
from __future__ import annotations


from pathlib import Path


import numpy as np
import pandas as pd


from lib.config import ANALYSIS_CONFIG_SERIES, get_pct_tasks_affected


# This file lives at <repo_root>/paper_figures/lib/exploratory/job_risk_scoring.py;
# reference data (incl. tech_skills_simple.csv) lives at <repo_root>/data/reference/.
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "reference"
TECH_SKILLS_FILE = DATA_DIR / "tech_skills_simple.csv"


PRIMARY_KEY = "all_confirmed"


RISK_AT_RISK_ZONE = {1, 2, 3}
RISK_AT_RISK_OUTLOOK = {2, 3}


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
