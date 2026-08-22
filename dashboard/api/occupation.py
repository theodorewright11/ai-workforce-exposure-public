"""The My-Occupation card.

Replaces v1's `occupation_report.py` (1,597 lines: SKA, similarity matrix,
four-gate risk score, software commodities, sector chain). v2's card is four
headline numbers and the verb families, and every number comes from the same
shared layer the paper figures use — see `dashboard/ARCHITECTURE.md` §2.

Payload shape, matching PRD §3.1:

    headline   four stats and the hierarchy
    families   one row per verb family present in this job
    tasks      the job's tasks, grouped by family (the expand)

Exposure and usage always travel together (PRD §2.3).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from lib import families as fam_lib
from lib import figure_data
from lib.figure_data import DEFAULT_GEO

figure_data.register_datasets()

DATASET = figure_data.PRIMARY_DATASET
FIRST_DATASET = figure_data.AC_SERIES[0]      # 2025-08-11
FIRST_DATE = "2025-08-11"
LATEST_DATE = "2026-05-31"

_titles_cache: Optional[list[str]] = None
_hierarchy_cache: Optional[list[dict]] = None
_occ_usage_cache: dict[str, pd.DataFrame] = {}


# ── Occupation universe ───────────────────────────────────────────────────

def _eco_occ_index() -> pd.DataFrame:
    """One row per occupation: hierarchy labels, deduped from eco_2025."""
    eco = figure_data.load_eco_raw()
    assert eco is not None and not eco.empty, "final_eco_2025.csv missing"
    cols = ["title_current", "broad_occ", "minor_occ_category",
            "major_occ_category", "job_zone"]
    return eco[cols].drop_duplicates("title_current").reset_index(drop=True)


def get_occupation_titles() -> list[str]:
    global _titles_cache
    if _titles_cache is None:
        _titles_cache = sorted(_eco_occ_index()["title_current"].astype(str))
    return _titles_cache


def get_occupation_hierarchy() -> list[dict]:
    """Rows for the browse-by-category picker."""
    global _hierarchy_cache
    if _hierarchy_cache is None:
        idx = _eco_occ_index()
        _hierarchy_cache = [
            {"title": str(r.title_current), "broad": str(r.broad_occ),
             "minor": str(r.minor_occ_category), "major": str(r.major_occ_category)}
            for r in idx.itertuples()
        ]
    return _hierarchy_cache


# ── Occupation-level usage (the paper's quantity) ─────────────────────────

def _occ_usage(geo: str) -> pd.DataFrame:
    """Per-occupation usage lift, anchored on the median rated occupation.

    Same construction as `builders/drivers.occ_usage_frame`, but across the
    whole economy rather than inside one major — the card compares this job
    to every job, not to its neighbours.

    Only occupations with at least one rated pair enter the anchor
    population. Entering the rest as zeros would drag the median down and
    inflate every reading.
    """
    if geo in _occ_usage_cache:
        return _occ_usage_cache[geo]

    pairs = figure_data.intensity_pairs(geo=geo)
    emp = figure_data.pair_level_emp("title_current", geo=geo)

    frame = pd.DataFrame({"num": pairs.groupby("title_current")["adj_pct"].sum()})
    frame["den"] = emp.reindex(frame.index)
    frame = frame[frame["den"] > 0].copy()
    assert not frame.empty, "no occupation survived the positive-denominator filter"
    frame["ratio"] = frame["num"] / frame["den"]
    frame["usage_x"] = figure_data.anchor_lower_median(frame["ratio"])
    frame["usage_rank"] = frame["usage_x"].rank(ascending=False, method="min").astype(int)
    frame = frame.reset_index()
    _occ_usage_cache[geo] = frame
    return frame


# ── Headline ──────────────────────────────────────────────────────────────

def _headline(title: str, geo: str) -> dict:
    occ = figure_data.occ_exposure(DATASET, geo=geo).set_index("title_current")
    assert title in occ.index, f"Occupation not in exposure frame: {title!r}"
    row = occ.loc[title]

    pct = float(row["p"]) * 100.0
    ranks = occ["p"].rank(ascending=False, method="min")

    # Direction: the same occupation on the first snapshot of the paper window.
    first = figure_data.occ_exposure(FIRST_DATASET, geo=geo).set_index("title_current")
    pct_first = float(first.loc[title, "p"]) * 100.0 if title in first.index else None

    usage = _occ_usage(geo).set_index("title_current")
    has_usage = title in usage.index

    idx = _eco_occ_index().set_index("title_current")
    meta = idx.loc[title] if title in idx.index else None

    emp = float(row["emp"])
    wage = float(row["wage"]) if pd.notna(row["wage"]) else None

    return {
        "title": title,
        "major": str(meta["major_occ_category"]) if meta is not None else None,
        "minor": str(meta["minor_occ_category"]) if meta is not None else None,
        "broad": str(meta["broad_occ"]) if meta is not None else None,
        "job_zone": int(meta["job_zone"]) if meta is not None and pd.notna(meta["job_zone"]) else None,
        # 1 — how much of the workday
        "pct_exposed": round(pct, 1),
        "pct_unexposed": round(100.0 - pct, 1),
        "pct_rank": int(ranks.loc[title]),
        "total_occupations": int(len(occ)),
        # 2 — usage, paired with it
        "usage_x": round(float(usage.loc[title, "usage_x"]), 2) if has_usage else 0.0,
        "usage_rank": int(usage.loc[title, "usage_rank"]) if has_usage else None,
        "usage_of": int(len(usage)),
        # 3 — direction
        "pct_first": round(pct_first, 1) if pct_first is not None else None,
        "change_pp": round(pct - pct_first, 1) if pct_first is not None else None,
        "first_date": FIRST_DATE,
        "latest_date": LATEST_DATE,
        # 4 — scale
        "employment": emp,
        "median_wage": wage,
        "workers_exposed": round(pct / 100.0 * emp),
    }


# ── Public API ────────────────────────────────────────────────────────────

def get_occupation_card(title: str, geo: str = DEFAULT_GEO) -> Optional[dict]:
    """The whole card. Returns None if the occupation is unknown."""
    if title not in set(get_occupation_titles()):
        return None

    head = _headline(title, geo)
    fam = fam_lib.occupation_family_rows(title, DATASET, geo=geo)
    tasks = fam_lib.occupation_family_tasks(title, DATASET, geo=geo)

    families_out = [
        {
            "family": r.family,
            "label": r.label,
            "short": r.short,
            "pct_exposed": round(float(r.pct_exposed), 1),
            "pct_unexposed": round(float(r.pct_unexposed), 1),
            "usage_x": round(float(r.usage_x), 2),
            "usage_share": round(float(r.usage_share), 1),
            "share_of_day": round(float(r.share_of_day), 1),
            "n_tasks": int(r.n_tasks),
        }
        for r in fam.itertuples()
    ]

    tasks_out: dict[str, list[dict]] = {}
    for r in tasks.itertuples():
        tasks_out.setdefault(r.family, []).append({
            "task": str(r.task_normalized),
            "activities": list(r.dwas),
            "pct_exposed": round(float(r.pct_exposed), 1),
            "pct_unexposed": round(float(r.pct_unexposed), 1),
            "usage_x": round(float(r.usage_x), 2),
            "usage_share": round(float(r.usage_share), 1),
            "auto_aug": round(float(r.auto_aug), 2) if pd.notna(r.auto_aug) else None,
        })

    return {"headline": head, "families": families_out, "tasks": tasks_out}
