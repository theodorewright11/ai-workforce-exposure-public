"""Economy at a Glance — the six blocks (PRD §3.2).

Zero controls, so the whole page is one cacheable payload. Each block mirrors
a paper figure and is computed by the same shared-layer call the figure uses,
so the two cannot drift:

    1  headline + trend        trend_phys
    2  verb families           verb_family_all_confirmed
    3  major categories        major_categories_stacked + major_adoption
    4  general work activities gwa_stacked + gwa_adoption
    5  the 31                  focused_set_usage
    6  states                  (new — see figure_data.state_exposure)

Every block carries a contrast: exposed vs. not, exposure vs. usage, or now
vs. then.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from lib import families as fam_lib
from lib import figure_data
from lib.figure_data import DEFAULT_GEO
from lib.focused import EXPOSURE_MIN, focused_set

DATASET = figure_data.PRIMARY_DATASET

# Physical mode → the series name the trend block reports it under.
_TREND_SERIES = [("all", "All work"), ("exclude", "Non-physical"), ("only", "Physical")]

_economy_cache: dict[str, dict] = {}


def _pct(rows: pd.DataFrame) -> float:
    """Employment-weighted work-time exposure of a pair-grain frame."""
    total = float(rows["hours"].sum())
    assert total > 0, "empty exposure denominator"
    return float(rows["hours_exposed"].sum()) / total * 100.0


# ── 1 · Headline + trend ──────────────────────────────────────────────────

def _trend_block(geo: str) -> dict:
    series: list[dict] = []
    for mode, label in _TREND_SERIES:
        points = []
        for ds in figure_data.AC_SERIES:
            rows = figure_data.pair_exposure_rows(ds, mode, geo=geo)
            points.append({"date": ds.split()[-1], "pct": round(_pct(rows), 1)})
        series.append({"key": mode, "label": label, "points": points})

    allw = next(s for s in series if s["key"] == "all")["points"]
    first, last = allw[0], allw[-1]
    return {
        "series": series,
        "headline_pct": last["pct"],
        "headline_unexposed": round(100.0 - last["pct"], 1),
        "first_pct": first["pct"],
        "first_date": first["date"],
        "latest_date": last["date"],
        "change_pp": round(last["pct"] - first["pct"], 1),
    }


# ── 2 · Verb families ─────────────────────────────────────────────────────

def _family_block(geo: str) -> list[dict]:
    fam = fam_lib.family_rows(DATASET, geo=geo)
    # Hours are /n-split across each pair's DWAs, so the family totals sum
    # back to the economy's daily hours and this share is a true share.
    total_hours = float(fam["hours"].sum())
    assert total_hours > 0, "empty family hours denominator"
    return [
        {
            "family": r.family,
            "label": r.label,
            "short": r.short,
            "pct_exposed": round(float(r.pct_exposed), 1),
            "pct_unexposed": round(float(r.pct_unexposed), 1),
            "usage_x": round(float(r.usage_x), 2),
            "share_of_day": round(float(r.hours) / total_hours * 100.0, 1),
            "n_tasks": int(r.n_tasks),
        }
        for r in fam.itertuples()
    ]


# ── 3 / 4 · Majors and GWAs ───────────────────────────────────────────────

def _group_block(kind: str, geo: str, top_n: Optional[int] = None) -> list[dict]:
    """Exposed/not plus workers, wages and usage for majors or GWAs."""
    if kind == "major":
        exp = figure_data.major_exposure(DATASET, geo=geo)
        usage = figure_data.group_usage_x("major_occ_category", geo=geo)
    else:
        exp = figure_data.act_exposure(DATASET, "gwa_title", geo=geo)
        usage = figure_data.group_usage_x("gwa_title", geo=geo)

    exp = exp.sort_values("pct", ascending=False)
    if top_n:
        exp = exp.head(top_n)
    return [
        {
            "category": str(r.category),
            "pct_exposed": round(float(r.pct), 1),
            "pct_unexposed": round(100.0 - float(r.pct), 1),
            "workers_exposed": float(r.workers_fte),
            "wages_exposed": float(r.wages_exposed),
            "usage_x": round(float(usage.get(str(r.category), 0.0)), 2),
        }
        for r in exp.itertuples()
    ]


# ── 5 · The 31 ────────────────────────────────────────────────────────────

def _focused_block(geo: str) -> dict:
    df = focused_set(DATASET, geo=geo)
    return {
        "exposure_min": EXPOSURE_MIN,
        "count": int(len(df)),
        "total_workers": float(df["workers_affected"].sum()),
        "rows": [
            {
                "title": str(r.title_current),
                "major": str(r.major_short),
                "job_zone": int(r.job_zone) if pd.notna(r.job_zone) else None,
                "pct_exposed": round(float(r.pct), 1),
                "pct_unexposed": round(float(r.pct_unexposed), 1),
                "emp_proj_pct": round(float(r.emp_proj_pct), 1),
                "usage_x": round(float(r.usage_x), 2),
                "workers_exposed": float(r.workers_affected),
            }
            for r in df.itertuples()
        ],
    }


# ── 6 · States ────────────────────────────────────────────────────────────

def _state_block(top_n: int = 10) -> dict:
    st = figure_data.state_exposure(DATASET)
    def _rows(sub: pd.DataFrame) -> list[dict]:
        return [
            {
                "geo": str(r.geo),
                "state": str(r.state),
                "rank": int(r.rank),
                "pct_exposed": round(float(r.pct), 1),
                "pct_unexposed": round(float(r.pct_unexposed), 1),
                "employment": float(r.employment),
                "workers_exposed": float(r.workers_affected),
            }
            for r in sub.itertuples()
        ]
    return {
        "total": int(len(st)),
        "top": _rows(st.head(top_n)),
        "bottom": _rows(st.tail(5).iloc[::-1]),
    }


# ── Public API ────────────────────────────────────────────────────────────

def get_economy(geo: str = DEFAULT_GEO) -> dict:
    """All six blocks. Cached per geo — the page takes no other input."""
    if geo in _economy_cache:
        return _economy_cache[geo]
    payload = {
        "trend": _trend_block(geo),
        "families": _family_block(geo),
        "majors": _group_block("major", geo),
        "gwas": _group_block("gwa", geo, top_n=15),
        "focused": _focused_block(geo),
        "states": _state_block(),
        "dataset": DATASET,
    }
    _economy_cache[geo] = payload
    return payload
