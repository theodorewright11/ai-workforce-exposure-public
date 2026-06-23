"""main.py — FastAPI backend for the PUBLIC AI Workforce Exposure dashboard.

A thin, read-only layer over the existing compute engine (`backend/compute.py`)
and figure-lib helpers (`paper_figures/lib/`). Exposes the five paper configs,
the three data-page views (occupation exposure, work-activity exposure, actual
usage), trends, and the (copied) occupation report.

Endpoints:
  GET  /api/health
  GET  /api/config                  — 5 configs + level/geo options
  POST /api/exposure                — single-snapshot bars (occ or wa)
  POST /api/exposure/children       — hierarchy drill-down (children of a category)
  POST /api/trend                   — time series for a config (occ or wa)
  POST /api/usage                   — debiased actual-usage intensity (drill-down)
  GET  /api/occupation-report/titles
  GET  /api/occupation-report       — full occupation report (copied page)
"""
from __future__ import annotations

import dashboard.api  # noqa: F401  — runs sys.path bootstrap (see __init__.py)

import math
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import AGG_LEVEL_COL, GEO_OPTIONS  # backend/config.py (flat import)
from compute import (
    get_group_data,
    compute_work_activities,
    compute_trends,
    compute_wa_trends,
    load_eco_raw,
)
from lib.config import ANALYSIS_CONFIG_LABELS, ANALYSIS_CONFIG_SERIES
from lib.paper_config import paper_dataset_for

from dashboard.api.occupation_report import (
    get_occupation_titles,
    get_occupation_hierarchy,
    get_occupation_report,
)
from dashboard.api.usage_intensity import compute_intensity

# ── Config metadata ─────────────────────────────────────────────────────────────

# Display order — all_confirmed (primary) first.
CONFIG_ORDER = [
    "all_confirmed",
    "human_conversation",
    "agentic_confirmed",
    "all_ceiling",
    "agentic_ceiling",
]

# config_key → DATASET_SERIES sub_type key (for compute_trends / compute_wa_trends)
CONFIG_TREND_SUBTYPE = {
    "all_confirmed":      "AEI Both + Micro",
    "human_conversation": "AEI Conv + Micro",
    "agentic_confirmed":  "AEI API",
    "all_ceiling":        "All",
    "agentic_ceiling":    "MCP + API",
}

OCC_LEVELS = {"Major Category": "major", "Minor Category": "minor",
              "Broad Occupation": "broad", "Occupation": "occupation"}
WA_LEVELS = {"General (GWA)": "gwa", "Intermediate (IWA)": "iwa", "Detailed (DWA)": "dwa"}
USAGE_LEVELS = {"Major Category": "major", "Minor Category": "minor",
                "Broad Occupation": "broad", "Occupation": "occupation", "Task": "task"}

# ── App ─────────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Workforce Exposure — Public Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


def _safe(v) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return float(v)


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _settings(config: str, level: str, geo: str, top_n: int = 9999) -> dict:
    """Locked-default settings dict for one config (single group)."""
    return {
        "selected_datasets": [paper_dataset_for(config)],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": "all",
        "geo": geo,
        "agg_level": level,
        "sort_by": "% Tasks Affected",
        "top_n": top_n,
        "search_query": "",
        "context_size": 5,
    }


# ── Hierarchy parent maps (for drill-down) ──────────────────────────────────────

_parent_maps: Optional[dict] = None


def _hierarchy_parent_maps() -> dict:
    """Cached child→parent lookups from eco_2025 for occ + WA hierarchies."""
    global _parent_maps
    if _parent_maps is not None:
        return _parent_maps
    eco = load_eco_raw()
    maps: dict[str, dict[str, str]] = {}

    def _pmap(child_col: str, parent_col: str) -> dict[str, str]:
        sub = eco[[child_col, parent_col]].dropna().drop_duplicates(child_col)
        return dict(zip(sub[child_col].astype(str), sub[parent_col].astype(str)))

    maps["minor"] = _pmap("minor_occ_category", "major_occ_category")
    maps["broad"] = _pmap("broad_occ", "minor_occ_category")
    maps["occupation"] = _pmap("title_current", "broad_occ")
    maps["iwa"] = _pmap("iwa_title", "gwa_title")
    maps["dwa"] = _pmap("dwa_title", "iwa_title")
    _parent_maps = maps
    return maps


# Next level down in each hierarchy
_OCC_CHILD = {"major": "minor", "minor": "broad", "broad": "occupation"}
_WA_CHILD = {"gwa": "iwa", "iwa": "dwa"}
_USAGE_CHILD = {"major": "minor", "minor": "broad", "broad": "occupation", "occupation": "task"}
_USAGE_PARENT_COL = {"major": "major_occ_category", "minor": "minor_occ_category",
                     "broad": "broad_occ", "occupation": "title_current"}


# ── Models ──────────────────────────────────────────────────────────────────────

class ExposureRow(BaseModel):
    category: str
    pct_tasks_affected: float
    workers_affected: float
    wages_affected: float
    rank_pct: int = 0
    rank_workers: int = 0
    rank_wages: int = 0


class ExposureResponse(BaseModel):
    rows: list[ExposureRow]
    total_categories: int = 0
    total_workers: float = 0.0
    total_wages: float = 0.0


class UsageRow(BaseModel):
    category: str
    intensity: float
    raw_pct: float
    ratio: float


# ── /api/health, /api/config ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "configs": [
            {"key": k, "label": ANALYSIS_CONFIG_LABELS[k]} for k in CONFIG_ORDER
        ],
        "occ_levels": OCC_LEVELS,
        "wa_levels": WA_LEVELS,
        "usage_levels": USAGE_LEVELS,
        "geo_options": GEO_OPTIONS,
        "default_config": "all_confirmed",
    }


# ── Exposure (occ + wa bars) ────────────────────────────────────────────────────

def _occ_rows(config: str, level: str, geo: str) -> ExposureResponse:
    result = get_group_data(_settings(config, level, geo, top_n=9999))
    if result is None or result.get("df") is None or result["df"].empty:
        return ExposureResponse(rows=[])
    df = result["df"]
    col = AGG_LEVEL_COL[level]
    rows = [
        ExposureRow(
            category=str(r[col]),
            pct_tasks_affected=_safe(r.get("pct_tasks_affected", 0)),
            workers_affected=_safe(r.get("workers_affected", 0)),
            wages_affected=_safe(r.get("wages_affected", 0)),
            rank_pct=_safe_int(r.get("rank_pct", 0)),
            rank_workers=_safe_int(r.get("rank_workers", 0)),
            rank_wages=_safe_int(r.get("rank_wages", 0)),
        )
        for _, r in df.iterrows()
    ]
    return ExposureResponse(
        rows=rows,
        total_categories=result.get("total_categories", len(rows)),
        total_workers=_safe(result.get("total_emp", 0.0)),
        total_wages=_safe(result.get("total_wages", 0.0)),
    )


def _wa_rows(config: str, level: str, geo: str) -> ExposureResponse:
    result = compute_work_activities(_settings(config, level, geo, top_n=9999))
    group = result.get("mcp_group") or result.get("aei_group")
    if group is None:
        return ExposureResponse(rows=[])
    raw = group.get(level, [])
    if not raw:
        return ExposureResponse(rows=[])
    # ranks + totals computed over the full set (no top-N truncation here)
    def _ranked(metric: str) -> dict[str, int]:
        order = sorted(raw, key=lambda r: r.get(metric, 0) or 0, reverse=True)
        return {str(r["category"]): i + 1 for i, r in enumerate(order)}
    rk_pct = _ranked("pct_tasks_affected")
    rk_wk = _ranked("workers_affected")
    rk_wg = _ranked("wages_affected")
    rows = [
        ExposureRow(
            category=str(r["category"]),
            pct_tasks_affected=_safe(r.get("pct_tasks_affected", 0)),
            workers_affected=_safe(r.get("workers_affected", 0)),
            wages_affected=_safe(r.get("wages_affected", 0)),
            rank_pct=rk_pct[str(r["category"])],
            rank_workers=rk_wk[str(r["category"])],
            rank_wages=rk_wg[str(r["category"])],
        )
        for r in raw
    ]
    return ExposureResponse(
        rows=rows,
        total_categories=len(rows),
        total_workers=_safe(sum(r.get("workers_affected", 0) or 0 for r in raw)),
        total_wages=_safe(sum(r.get("wages_affected", 0) or 0 for r in raw)),
    )


class ExposureRequest(BaseModel):
    config: str = "all_confirmed"
    level: str = "major"
    geo: str = "nat"
    kind: str = "occ"   # "occ" | "wa"


@app.post("/api/exposure", response_model=ExposureResponse)
def exposure(req: ExposureRequest):
    if req.kind == "wa":
        return _wa_rows(req.config, req.level, req.geo)
    return _occ_rows(req.config, req.level, req.geo)


class ChildrenRequest(ExposureRequest):
    parent: str = ""   # category whose children to return


@app.post("/api/exposure/children", response_model=ExposureResponse)
def exposure_children(req: ChildrenRequest):
    maps = _hierarchy_parent_maps()
    if req.kind == "wa":
        child = _WA_CHILD.get(req.level)
        if child is None:
            return ExposureResponse(rows=[])
        full = _wa_rows(req.config, child, req.geo)
        pmap = maps[child]
    else:
        child = _OCC_CHILD.get(req.level)
        if child is None:
            return ExposureResponse(rows=[])
        full = _occ_rows(req.config, child, req.geo)
        pmap = maps[child]
    kept = [r for r in full.rows if pmap.get(r.category) == req.parent]
    return ExposureResponse(
        rows=kept, total_categories=full.total_categories,
        total_workers=full.total_workers, total_wages=full.total_wages,
    )


# ── Trends ──────────────────────────────────────────────────────────────────────

class TrendDataPoint(BaseModel):
    dataset: str
    date: str
    rows: list[ExposureRow]


class TrendResponse(BaseModel):
    data_points: list[TrendDataPoint]
    top_categories: list[str]


class TrendRequest(BaseModel):
    config: str = "all_confirmed"
    level: str = "major"
    geo: str = "nat"
    kind: str = "occ"


@app.post("/api/trend", response_model=TrendResponse)
def trend(req: TrendRequest):
    sub = CONFIG_TREND_SUBTYPE[req.config]
    allowed = set(ANALYSIS_CONFIG_SERIES.get(req.config, []))  # paper series dates
    base = {
        "series": [sub], "method": "freq", "use_auto_aug": True,
        "physical_mode": "all", "geo": req.geo,
        "top_n": 9999, "sort_by": "% Tasks Affected",
    }
    if req.kind == "wa":
        base["activity_level"] = req.level
        result = compute_wa_trends(base)
    else:
        base["agg_level"] = req.level
        result = compute_trends(base)

    dps: list[TrendDataPoint] = []
    top: list[str] = []
    for s in result.get("series", []):
        top = s.get("top_categories", [])
        for dp in s.get("data_points", []):
            if dp["dataset"] not in allowed:
                continue  # restrict to the paper's config series (drops 2024 anchors)
            rows = [
                ExposureRow(
                    category=str(r["category"]),
                    pct_tasks_affected=_safe(r.get("pct_tasks_affected", 0)),
                    workers_affected=_safe(r.get("workers_affected", 0)),
                    wages_affected=_safe(r.get("wages_affected", 0)),
                )
                for r in dp.get("rows", [])
            ]
            dps.append(TrendDataPoint(dataset=dp["dataset"], date=dp["date"], rows=rows))
    dps.sort(key=lambda d: d.date)
    return TrendResponse(data_points=dps, top_categories=top)


# ── Actual usage (intensity) ────────────────────────────────────────────────────

class UsageRequest(BaseModel):
    level: str = "major"
    parent_level: Optional[str] = None   # for drill-down: the parent's level
    parent: Optional[str] = None         # the parent category value


@app.post("/api/usage")
def usage(req: UsageRequest):
    parent_col = None
    if req.parent_level and req.parent:
        parent_col = _USAGE_PARENT_COL.get(req.parent_level)
    rows = compute_intensity(req.level, parent_col=parent_col, parent_val=req.parent)
    return {"rows": rows, "child_level": _USAGE_CHILD.get(req.level)}


# ── Occupation report (copied page) ─────────────────────────────────────────────

@app.get("/api/occupation-report/titles")
def occupation_report_titles():
    return {"titles": get_occupation_titles(), "hierarchy": get_occupation_hierarchy()}


@app.get("/api/occupation-report")
def occupation_report(
    title: str = Query(..., description="Occupation title (title_current)"),
    geo: str = Query("nat", description="Geography code"),
):
    if geo not in GEO_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown geo: {geo}")
    payload = get_occupation_report(title, geo)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Occupation not found: {title}")
    return payload
