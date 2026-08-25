"""The focused set — "the 31".

Occupations that are both heavily exposed and already projected to shed
employment. Membership is exactly what the name says:

    work time exposed >= EXPOSURE_MIN  AND  BLS 2025-34 projection < 0

and nothing else. The paper's earlier set (`audit_risk_score`) layered an SKA
gate and an exposure-trend gate on top, which made it impossible to reproduce
from the chart; dropping them moved the set from 40 occupations to 31 and
bought a rule a reader can check.

Both gates are columns of `final_eco_2025.csv`, so this module deliberately
does NOT go through `lib.exploratory.risk_score._load_flag_df` the way
`builders/focused.py` historically did. That loader pulls in the whole SKA
stack (O*NET skills/knowledge/abilities, the tech-skills file, the explorer
occupation list) to supply three columns we already have. The dashboard
imports this module, and SKA is retired from the dashboard.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from lib import figure_data
from lib.figure_data import DEFAULT_GEO

__all__ = ["EXPOSURE_MIN", "PROJECTION_COL", "focused_set"]

# Work-time exposure gate, in percent. Set with the hours weighting on the
# 05-31 files, so the gate the title states is the gate the bars show.
EXPOSURE_MIN = 67.0

PROJECTION_COL = "emp_change_pct__PROJ_2025_2034__"

_focused_cache: dict[tuple[str, str, float], pd.DataFrame] = {}


def _occ_projection() -> pd.DataFrame:
    """One row per occupation: BLS projection, job zone, major label."""
    eco = figure_data.load_eco_raw()
    assert eco is not None and not eco.empty, "final_eco_2025.csv missing"
    assert PROJECTION_COL in eco.columns, f"{PROJECTION_COL} missing from eco"
    out = (
        eco[["title_current", PROJECTION_COL, "job_zone", "major_occ_category"]]
        .drop_duplicates("title_current")
        .rename(columns={PROJECTION_COL: "emp_proj_pct",
                         "major_occ_category": "major"})
        .reset_index(drop=True)
    )
    out["emp_proj_pct"] = pd.to_numeric(out["emp_proj_pct"], errors="coerce")
    out["job_zone"] = pd.to_numeric(out["job_zone"], errors="coerce")
    return out


def focused_set(
    dataset: Optional[str] = None,
    geo: str = DEFAULT_GEO,
    exposure_min: float = EXPOSURE_MIN,
) -> pd.DataFrame:
    """The focused set, ranked most-exposed first.

    Columns: title_current, major, major_short, job_zone, emp_proj_pct, pct,
    pct_unexposed, usage_x, workers_affected, wages_affected.

    Magnitudes are built per occupation (p x emp, p x emp x wage) and never
    by multiplying a group percentage back out.
    """
    dataset = dataset or figure_data.PRIMARY_DATASET
    key = (dataset, geo, float(exposure_min))
    if key in _focused_cache:
        return _focused_cache[key]

    exposure = figure_data.occ_exposure(dataset, geo=geo).set_index("title_current")
    df = _occ_projection()
    df["pct"] = df["title_current"].map(exposure["p"] * 100.0)
    df = df[(df["pct"] >= exposure_min) & (df["emp_proj_pct"] < 0)].copy()
    assert not df.empty, (
        f"No occupations clear {exposure_min}% exposure + a negative projection"
    )

    df["pct_unexposed"] = 100.0 - df["pct"]
    df["major_short"] = df["major"].str.replace(" Occupations", "", regex=False)
    df["workers_affected"] = df["title_current"].map(
        exposure["p"] * exposure["emp"]).fillna(0.0)
    df["wages_affected"] = df["title_current"].map(
        exposure["p"] * exposure["emp"] * exposure["wage"]).fillna(0.0)
    # Anchored against the whole economy, not inside the set -- see
    # figure_data.occ_usage_lift. 1.00x means the same thing here as anywhere.
    df["usage_x"] = df["title_current"].map(
        figure_data.occ_usage_lift(geo)).fillna(0.0)

    out = df.sort_values("pct", ascending=False).reset_index(drop=True)[
        ["title_current", "major", "major_short", "job_zone", "emp_proj_pct",
         "pct", "pct_unexposed", "usage_x", "workers_affected", "wages_affected"]
    ]
    _focused_cache[key] = out
    return out
