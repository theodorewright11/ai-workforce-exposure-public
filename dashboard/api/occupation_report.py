"""
occupation_report.py — Per-occupation actionable report.

Composes one big payload for a single occupation, drawn from the dashboard's
existing compute pipeline plus the SKA gap / intensity / tech / risk logic
that the analysis folder uses for its tips-and-tricks worker_resilience runs.

Everything is computed against the **all-confirmed** dataset
(`AEI Both + Micro 2026-02-12`) to match the framing used in
worker_resilience. Per-task auto_aug values come from the explorer task
lookup — same source breakdown shown in the explorers' task accordion.

Public entrypoint:
    get_occupation_report(title: str, geo: str = "nat") -> dict | None

Module-level caches (all lazy, computed on first call) avoid recomputing
the cross-occupation references (top-10 SKA per element, risk medians,
intensity ranks, similarity matrix) on every request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, DATASETS, GEO_OPTIONS
from compute import (
    _build_explorer_task_lookup,
    _build_top_mcps_lookup,
    _safe_num,
    compute_work_activities,
    get_explorer_groups,
    get_explorer_occupations,
    get_group_data,
    load_eco_raw,
)


# ── Constants ─────────────────────────────────────────────────────────────────

PRIMARY_DATASET: str = "AEI Both + Micro 2026-02-12"

# Time series we walk for the trend sparkline. Mirrors all_confirmed in
# analysis/config.py: ANALYSIS_CONFIG_SERIES["all_confirmed"].
TREND_SERIES: list[str] = [
    "AEI Both + Micro 2025-03-06",
    "AEI Both + Micro 2025-08-11",
    "AEI Both + Micro 2025-11-13",
    "AEI Both + Micro 2026-02-12",
]

# Color thresholds (auto_aug 0–5 scale)
# Color-bucket cutoffs on the 0–5 auto-aug scale. high = rounds to 5 (≥4.5),
# mid = rounds to 3–4 (2.5–4.5), low = rounds to 1–2 (<2.5).
AUTO_HIGH: float = 4.5
AUTO_MID: float = 2.5
# SKA color thresholds (AI as % of occ need; > 100 means AI exceeds need)
SKA_PCT_HIGH: float = 100.0
SKA_PCT_MID: float = 66.0

IMPORTANCE_THRESHOLD: float = 3.0
TOP_N_FOR_AI_REFERENCE: int = 10
N_SIMILAR_OCCS: int = 5
N_TASK_MCPS: int = 5

# Risk-scoring (mirrors analysis/questions/job_exposure/job_risk_scoring/run.py)
RISK_AT_RISK_ZONE: set[int] = {1, 2, 3}
RISK_AT_RISK_OUTLOOK: set[int] = {2, 3}
EXPOSURE_GATE: float = 33.0
PCT_ABS_THRESHOLD: float = 50.0
FLAG_WEIGHTS: dict[str, int] = {
    "flag1_pct":        2,
    "flag2_ska":        2,
    "flag3_pct_trend":  1,
    "flag4_ska_trend":  1,
    "flag5_job_zone":   1,
    "flag6_outlook":    1,
    "flag7_n_software": 1,
    "flag8_auto_aug":   1,
}

# Static reference CSVs. In this public repo the O*NET SKA + tech reference
# files live under `data/reference/` (alongside the paper-figure reference data);
# `mcp_titles_desc.csv` was copied into `data/` root for the occupation report.
REFERENCE_DIR   = DATA_DIR / "reference"
SKILLS_FILE     = REFERENCE_DIR / "skills_v30.1.csv"
ABILITIES_FILE  = REFERENCE_DIR / "abilities_v30.1.csv"
KNOWLEDGE_FILE  = REFERENCE_DIR / "knowledge_v30.1.csv"
TECH_SKILLS_FILE        = REFERENCE_DIR / "technology_skills_v30.1.csv"
TECH_SKILLS_SIMPLE_FILE = REFERENCE_DIR / "tech_skills_simple.csv"
MCP_TITLES_DESC_FILE    = DATA_DIR / "mcp_titles_desc.csv"


# ── Bias-ratio constants for intensity correction ────────────────────────────
# Source: analysis/exploratory/pct_norm_vs_eco/run.py (CLAUDE/COPILOT/CHATGPT
# GWA shares, equal 3-source consensus). Inlined here so backend has no
# cross-folder import dependency. If those numbers change in the analysis,
# update both places.

_CLAUDE_GWA_SHARE_RAW: dict[str, float] = {
    "Thinking Creatively": 33.7, "Working with Computers": 11.3,
    "Documenting/Recording Information": 8.7, "Analyzing Data or Information": 8.4,
    "Providing Consultation and Advice to Others": 4.0, "Training and Teaching Others": 3.7,
    "Making Decisions and Solving Problems": 3.7, "Getting Information": 3.6,
    "Inspecting Equipment, Structures, or Materials": 2.7, "Developing Objectives and Strategies": 2.4,
    "Judging the Qualities of Objects, Services, or People": 2.2,
    "Interpreting the Meaning of Information for Others": 2.0,
    "Guiding, Directing, and Motivating Subordinates": 2.0,
    "Communicating with Supervisors, Peers, or Subordinates": 1.8,
    "Performing for or Working Directly with the Public": 1.4, "Processing Information": 1.3,
    "Communicating with People Outside the Organization": 0.9,
    "Repairing and Maintaining Mechanical Equipment": 0.9,
    "Updating and Using Relevant Knowledge": 0.8,
    "Monitoring Processes, Materials, or Surroundings": 0.7,
    "Performing Administrative Activities": 0.6, "Assisting and Caring for Others": 0.5,
    "Selling or Influencing Others": 0.5,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 0.4,
    "Handling and Moving Objects": 0.4, "Identifying Objects, Actions, and Events": 0.3,
    "Monitoring and Controlling Resources": 0.2,
    "Organizing, Planning, and Prioritizing Work": 0.2,
    "Resolving Conflicts and Negotiating with Others": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Staffing Organizational Units": 0.2, "Controlling Machines and Processes": 0.1,
    "Scheduling Work and Activities": 0.1,
    "Establishing and Maintaining Interpersonal Relationships": 0.1,
    "Coaching and Developing Others": 0.1, "Performing General Physical Activities": 0.0,
    "Operating Vehicles, Mechanized Devices, or Equipment": 0.0,
}

_COPILOT_GWA_SHARE_RAW: dict[str, float] = {
    "Getting Information": 24.3, "Communicating with People Outside the Organization": 15.4,
    "Performing for or Working Directly with the Public": 12.7, "Assisting and Caring for Others": 8.4,
    "Interpreting the Meaning of Information for Others": 5.2,
    "Documenting/Recording Information": 5.1, "Thinking Creatively": 4.4,
    "Providing Consultation and Advice to Others": 3.5, "Updating and Using Relevant Knowledge": 3.3,
    "Making Decisions and Solving Problems": 3.2, "Working with Computers": 2.7,
    "Communicating with Supervisors, Peers, or Subordinates": 2.2,
    "Analyzing Data or Information": 1.4, "Coaching and Developing Others": 1.3,
    "Training and Teaching Others": 1.3,
}

_CHATGPT_GWA_SHARE_RAW: dict[str, float] = {
    "Getting Information": 19.3, "Documenting/Recording Information": 13.8,
    "Thinking Creatively": 13.4, "Communicating with People Outside the Organization": 10.5,
    "Working with Computers": 8.6, "Interpreting the Meaning of Information for Others": 7.3,
    "Analyzing Data or Information": 5.9, "Updating and Using Relevant Knowledge": 4.4,
    "Making Decisions and Solving Problems": 4.2, "Providing Consultation and Advice to Others": 4.1,
    "Performing for or Working Directly with the Public": 3.5, "Assisting and Caring for Others": 1.8,
    "Communicating with Supervisors, Peers, or Subordinates": 1.7, "Coaching and Developing Others": 1.5,
}


def _renorm_to_100(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: 100.0 * v / total for k, v in d.items()}


_CLAUDE_GWA  = _renorm_to_100(_CLAUDE_GWA_SHARE_RAW)
_COPILOT_GWA = _renorm_to_100(_COPILOT_GWA_SHARE_RAW)
_CHATGPT_GWA = _renorm_to_100(_CHATGPT_GWA_SHARE_RAW)
_CANONICAL_GWAS: list[str] = sorted(set(_CLAUDE_GWA) | set(_COPILOT_GWA) | set(_CHATGPT_GWA))


def _equal_consensus_bias_ratios() -> dict[str, float]:
    """bias_ratio[gwa] = claude_share / equal-weight consensus_share."""
    ratios: dict[str, float] = {}
    for gwa in _CANONICAL_GWAS:
        parts: list[float] = []
        if gwa in _CLAUDE_GWA:  parts.append(_CLAUDE_GWA[gwa])
        if gwa in _COPILOT_GWA: parts.append(_COPILOT_GWA[gwa])
        if gwa in _CHATGPT_GWA: parts.append(_CHATGPT_GWA[gwa])
        if not parts or gwa not in _CLAUDE_GWA:
            ratios[gwa] = 1.0
            continue
        consensus = sum(parts) / len(parts)
        ratios[gwa] = _CLAUDE_GWA[gwa] / consensus if consensus > 0 else 1.0
    return ratios


# ── In-process caches ─────────────────────────────────────────────────────────

_pct_by_dataset_cache: dict[str, pd.Series] = {}
_ska_data_cache: Optional["_SKAData"] = None
_ska_result_cache: dict[str, "_SKAResult"] = {}    # keyed by dataset name
_intensity_rank_cache: Optional[dict[str, dict]] = None     # title → {occ_rank, major_rank, ...}
_risk_table_cache: Optional[pd.DataFrame] = None
_tech_commodity_rank_cache: Optional[pd.DataFrame] = None
_ska_top10_per_element_cache: Optional[pd.DataFrame] = None
_mcp_titles_desc_cache: Optional[dict[str, str]] = None
_explorer_occ_index_cache: Optional[dict[str, dict]] = None
_ska_profile_matrix_cache: Optional[tuple[list[str], np.ndarray]] = None  # (titles, profile)
_eco_wa_stats_cache: dict[str, dict[str, dict[str, dict]]] = {}  # geo → level → wa_name → eco_stats
_sector_level_cache: dict[tuple[str, str], pd.DataFrame] = {}    # (level, geo) → ranked group df


# ── SKA loader & compute (inlined from analysis/data/compute_ska.py) ──────────

@dataclass
class _SKAData:
    skills: pd.DataFrame
    abilities: pd.DataFrame
    knowledge: pd.DataFrame


@dataclass
class _SKAResult:
    occ_gaps: pd.DataFrame                        # per-occ skills_pct/abilities_pct/knowledge_pct/overall_pct
    occ_element_scores: dict[str, pd.DataFrame]   # per type → per (occ, element) detail
    ai_capability: pd.DataFrame                   # element_name, type, ai_score (95th pct)


def _load_onet_ska_file(path: Path) -> pd.DataFrame:
    """Pivot one O*NET SKA CSV to (soc_code, title, element_name, importance, level)."""
    assert path.exists(), f"O*NET SKA file not found: {path}"
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code", "Title": "title",
        "Element Name": "element_name", "Scale ID": "scale_id", "Data Value": "data_value",
    })
    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["scale_id"].isin(["IM", "LV"])]
    pivoted = df.pivot_table(
        index=["soc_code", "title", "element_name"],
        columns="scale_id", values="data_value", aggfunc="mean",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"})
    return pivoted.dropna(subset=["importance", "level"])


def _load_ska_data() -> _SKAData:
    global _ska_data_cache
    if _ska_data_cache is None:
        _ska_data_cache = _SKAData(
            skills=_load_onet_ska_file(SKILLS_FILE),
            abilities=_load_onet_ska_file(ABILITIES_FILE),
            knowledge=_load_onet_ska_file(KNOWLEDGE_FILE),
        )
    return _ska_data_cache


def _compute_ska_for_pct(pct: pd.Series) -> _SKAResult:
    """Run SKA pipeline for a pct_tasks_affected Series. Sames as
    analysis/data/compute_ska.compute_ska but trimmed to fields we use."""
    assert len(pct) > 0, "pct must not be empty"
    ska = _load_ska_data()

    type_map = {"skills": ska.skills, "abilities": ska.abilities, "knowledge": ska.knowledge}
    occ_elems: dict[str, pd.DataFrame] = {}
    ai_caps: list[pd.DataFrame] = []

    for type_name, onet_df in type_map.items():
        df = onet_df.copy()
        df["pct"] = df["title"].map(pct)
        df = df.dropna(subset=["pct"])
        df = df[df["importance"] >= IMPORTANCE_THRESHOLD].copy()
        if df.empty:
            occ_elems[type_name] = pd.DataFrame(columns=[
                "title_current", "element_name", "occ_score", "ai_product", "ai_score", "gap",
            ])
            continue
        df["occ_score"] = df["importance"] * df["level"]
        df["ai_product"] = (df["pct"] / 100.0) * df["occ_score"]
        ai_cap = (df.groupby("element_name")["ai_product"].quantile(0.95)
                  .reset_index().rename(columns={"ai_product": "ai_score"}))
        ai_cap["type"] = type_name
        ai_caps.append(ai_cap)
        occ_elem = df[["title", "element_name", "importance", "level", "occ_score", "ai_product"]].copy()
        occ_elem = occ_elem.rename(columns={"title": "title_current"})
        occ_elem = occ_elem.merge(ai_cap[["element_name", "ai_score"]], on="element_name", how="left")
        occ_elem["gap"] = occ_elem["ai_score"] - occ_elem["occ_score"]
        occ_elems[type_name] = occ_elem

    # Per-occ summary: ratio-of-sums per type + overall
    per_type_frames: list[pd.DataFrame] = []
    for type_name, occ_elem in occ_elems.items():
        if occ_elem.empty:
            continue
        grouped = occ_elem.groupby("title_current")
        sum_ai  = grouped["ai_score"].sum()
        sum_occ = grouped["occ_score"].sum()
        pct_col = (sum_ai / sum_occ.replace(0, np.nan) * 100.0).rename(f"{type_name}_pct")
        per_type_frames.append(pct_col)

    if per_type_frames:
        occ_gaps = pd.concat(per_type_frames, axis=1).reset_index()
    else:
        occ_gaps = pd.DataFrame(columns=["title_current"])

    all_elems = pd.concat(
        [df[["title_current", "occ_score", "ai_score"]] for df in occ_elems.values() if not df.empty],
        ignore_index=True,
    )
    if not all_elems.empty:
        overall = all_elems.groupby("title_current").agg(
            _sum_ai=("ai_score", "sum"), _sum_occ=("occ_score", "sum"),
        )
        overall["overall_pct"] = (
            overall["_sum_ai"] / overall["_sum_occ"].replace(0, np.nan) * 100.0
        )
        occ_gaps = occ_gaps.merge(
            overall[["overall_pct"]], left_on="title_current", right_index=True, how="left",
        )

    ai_cap_combined = pd.concat(ai_caps, ignore_index=True) if ai_caps else pd.DataFrame()
    return _SKAResult(occ_gaps=occ_gaps, occ_element_scores=occ_elems, ai_capability=ai_cap_combined)


def _ska_for(dataset_name: str) -> _SKAResult:
    if dataset_name in _ska_result_cache:
        return _ska_result_cache[dataset_name]
    pct = _pct_for(dataset_name)
    result = _compute_ska_for_pct(pct)
    _ska_result_cache[dataset_name] = result
    return result


def _ska_top10_per_element() -> pd.DataFrame:
    """For each (type, element_name), the mean of the top-10 ai_product values
    across occupations. This is the AI reference value used by the SKA gap
    table — same definition as analysis/exploratory/ska_levels."""
    global _ska_top10_per_element_cache
    if _ska_top10_per_element_cache is not None:
        return _ska_top10_per_element_cache
    result = _ska_for(PRIMARY_DATASET)
    out_rows: list[dict] = []
    for type_name, occ_elem in result.occ_element_scores.items():
        if occ_elem.empty:
            continue
        for elem, sub in occ_elem.groupby("element_name"):
            vals = sub["ai_product"].dropna()
            if vals.empty:
                continue
            top_n = min(TOP_N_FOR_AI_REFERENCE, len(vals))
            top10 = vals.nlargest(top_n).mean()
            out_rows.append({
                "type": type_name, "element_name": elem,
                "ai_top10": float(top10),
            })
    df = pd.DataFrame(out_rows)
    _ska_top10_per_element_cache = df
    return df


# ── Pct loaders (one geo, occupation level, primary dataset) ─────────────────

def _pct_for(dataset_name: str, geo: str = "nat") -> pd.Series:
    """Run the full compute pipeline for one dataset → Series keyed by title_current.
    Equivalent to analysis.config.get_pct_tasks_affected but inlined."""
    cache_key = f"{dataset_name}|{geo}"
    if cache_key in _pct_by_dataset_cache:
        return _pct_by_dataset_cache[cache_key]
    settings = {
        "selected_datasets": [dataset_name], "combine_method": "Average",
        "method": "freq", "use_auto_aug": True,
        "physical_mode": "all", "geo": geo, "agg_level": "occupation",
        "sort_by": "% Tasks Affected", "top_n": 9999,
        "search_query": "", "context_size": 3,
    }
    data = get_group_data(settings)
    if data is None:
        _pct_by_dataset_cache[cache_key] = pd.Series(dtype=float)
        return _pct_by_dataset_cache[cache_key]
    df = data["df"]
    group_col = data["group_col"]
    series = df.set_index(group_col)["pct_tasks_affected"]
    _pct_by_dataset_cache[cache_key] = series
    return series


def _emp_wage_for(dataset_name: str, geo: str = "nat") -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (workers_affected, wages_affected, pct) Series keyed by title_current."""
    settings = {
        "selected_datasets": [dataset_name], "combine_method": "Average",
        "method": "freq", "use_auto_aug": True,
        "physical_mode": "all", "geo": geo, "agg_level": "occupation",
        "sort_by": "% Tasks Affected", "top_n": 9999,
        "search_query": "", "context_size": 3,
    }
    data = get_group_data(settings)
    if data is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    df = data["df"]
    group_col = data["group_col"]
    workers = df.set_index(group_col)["workers_affected"]
    wages   = df.set_index(group_col)["wages_affected"]
    pct     = df.set_index(group_col)["pct_tasks_affected"]
    return workers, wages, pct


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color_bucket_auto(score: Optional[float]) -> str:
    """Map auto_aug score → color bucket. Three neutral framings."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "none"
    if score >= AUTO_HIGH:
        return "high"     # "more automated usage seen"
    if score >= AUTO_MID:
        return "mid"      # "more augmentative"
    return "low"          # "not much usage seen"


def _color_bucket_ska(pct_of_need: Optional[float]) -> str:
    """Map SKA AI-as-%-of-occ-need → color bucket."""
    if pct_of_need is None or (isinstance(pct_of_need, float) and math.isnan(pct_of_need)):
        return "none"
    if pct_of_need >= SKA_PCT_HIGH:
        return "high"
    if pct_of_need >= SKA_PCT_MID:
        return "mid"
    return "low"


def _max_or_none(vals: list[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(nums) if nums else None


def _round_or_none(v: Optional[float], n: int = 2) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), n)


def _occ_index() -> dict[str, dict]:
    """title_current → explorer-occ dict (national emp/wage, hierarchy, dws, jz, n_tasks)."""
    global _explorer_occ_index_cache
    if _explorer_occ_index_cache is None:
        _explorer_occ_index_cache = {
            occ["title_current"]: occ for occ in get_explorer_occupations(geo="nat")
        }
    return _explorer_occ_index_cache


def _mcp_titles_desc_lookup() -> dict[str, str]:
    """title (lowercased / titlecased) → text_for_llm description."""
    global _mcp_titles_desc_cache
    if _mcp_titles_desc_cache is not None:
        return _mcp_titles_desc_cache
    out: dict[str, str] = {}
    if MCP_TITLES_DESC_FILE.exists():
        df = pd.read_csv(MCP_TITLES_DESC_FILE)
        if "title" in df.columns and "text_for_llm" in df.columns:
            for _, row in df.iterrows():
                t = str(row["title"]).strip()
                d = row.get("text_for_llm")
                if not t or pd.isna(d):
                    continue
                out[t] = str(d)
                # also index by lowercase for case-insensitive lookup
                out[t.lower()] = str(d)
    _mcp_titles_desc_cache = out
    return out


# ── Tech commodity rank (mirrors skills_landscape Chart 1 metric) ─────────────

def _tech_commodity_rank() -> pd.DataFrame:
    """For every commodity, average pct_tasks_affected across (occ, software) rows
    that use it. Returns DataFrame with columns: commodity, avg_pct, rank.
    rank is 1-indexed, 1 = highest avg_pct."""
    global _tech_commodity_rank_cache
    if _tech_commodity_rank_cache is not None:
        return _tech_commodity_rank_cache
    if not TECH_SKILLS_FILE.exists():
        _tech_commodity_rank_cache = pd.DataFrame(columns=["commodity", "avg_pct", "rank"])
        return _tech_commodity_rank_cache
    pct = _pct_for(PRIMARY_DATASET)
    tech = pd.read_csv(TECH_SKILLS_FILE)
    tech.columns = [c.strip() for c in tech.columns]
    if "Title" not in tech.columns or "Commodity Title" not in tech.columns:
        _tech_commodity_rank_cache = pd.DataFrame(columns=["commodity", "avg_pct", "rank"])
        return _tech_commodity_rank_cache
    tech = tech[["Title", "Commodity Title", "Example"]].copy()
    tech["pct"] = tech["Title"].map(pct).fillna(0.0)
    by_comm = tech.groupby("Commodity Title")["pct"].mean().reset_index()
    by_comm = by_comm.rename(columns={"Commodity Title": "commodity", "pct": "avg_pct"})
    by_comm = by_comm.sort_values("avg_pct", ascending=False).reset_index(drop=True)
    by_comm["rank"] = by_comm.index + 1
    _tech_commodity_rank_cache = by_comm
    return by_comm


def _tech_for_occ(title: str) -> list[dict]:
    """Return softwares this occ uses, with commodity rank info."""
    if not TECH_SKILLS_FILE.exists():
        return []
    tech = pd.read_csv(TECH_SKILLS_FILE)
    tech.columns = [c.strip() for c in tech.columns]
    if "Title" not in tech.columns:
        return []
    occ_tech = tech[tech["Title"] == title].copy()
    if occ_tech.empty:
        return []
    rank_df = _tech_commodity_rank()
    rank_map = dict(zip(rank_df["commodity"], rank_df["rank"]))
    pct_map = dict(zip(rank_df["commodity"], rank_df["avg_pct"]))
    n_total = len(rank_df)
    out: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for _, row in occ_tech.iterrows():
        comm = str(row.get("Commodity Title", "") or "").strip()
        software = str(row.get("Example", "") or "").strip()
        if not software or not comm:
            continue
        if (software, comm) in seen_pairs:
            continue
        seen_pairs.add((software, comm))
        rank = rank_map.get(comm)
        avg_pct = pct_map.get(comm)
        out.append({
            "software": software,
            "commodity": comm,
            "commodity_rank": int(rank) if rank is not None else None,
            "commodity_total": n_total,
            "commodity_avg_pct": _round_or_none(avg_pct, 2),
        })
    # sort by commodity rank ascending (more exposed → top)
    out.sort(key=lambda r: r["commodity_rank"] if r["commodity_rank"] is not None else 1e9)
    return out


# ── Risk score (mirrors job_risk_scoring) ─────────────────────────────────────

def _risk_table() -> pd.DataFrame:
    """Per-occ DataFrame with risk_score (0–10), tier, and per-flag breakdown.
    Computed once across all 923 occs because medians are needed. Cached."""
    global _risk_table_cache
    if _risk_table_cache is not None:
        return _risk_table_cache

    pct       = _pct_for(PRIMARY_DATASET)
    pct_first = _pct_for(TREND_SERIES[0])
    pct_last  = _pct_for(TREND_SERIES[-1])

    ska_now   = _ska_for(PRIMARY_DATASET)
    ska_first = _compute_ska_for_pct(pct_first)
    ska_last  = _compute_ska_for_pct(pct_last)

    # per-occ ska overall_pct
    ska_pct_now   = ska_now.occ_gaps.set_index("title_current").get(
        "overall_pct", pd.Series(dtype=float))
    ska_pct_first = ska_first.occ_gaps.set_index("title_current").get(
        "overall_pct", pd.Series(dtype=float))
    ska_pct_last  = ska_last.occ_gaps.set_index("title_current").get(
        "overall_pct", pd.Series(dtype=float))

    # BLS 2025–34 projected employment change per occ (for the focused-set gate)
    eco_proj = load_eco_raw()
    emp_proj_map: dict[str, float] = {}
    if eco_proj is not None and "emp_change_pct__PROJ_2025_2034__" in eco_proj.columns:
        _sub = eco_proj.drop_duplicates("title_current")
        emp_proj_map = dict(zip(
            _sub["title_current"],
            pd.to_numeric(_sub["emp_change_pct__PROJ_2025_2034__"], errors="coerce"),
        ))

    occ_idx = _occ_index()
    rows: list[dict] = []
    for title, occ in occ_idx.items():
        rows.append({
            "title_current": title,
            "pct":       float(pct.get(title, 0.0) or 0.0),
            "emp_proj":  _safe_num(emp_proj_map.get(title)),
            "pct_delta": float(pct_last.get(title, 0.0) or 0.0)
                       - float(pct_first.get(title, 0.0) or 0.0),
            "ska_pct":   _safe_num(ska_pct_now.get(title)),
            "ska_delta": (
                _safe_num(ska_pct_last.get(title)) - _safe_num(ska_pct_first.get(title))
                if ska_pct_last.get(title) is not None and ska_pct_first.get(title) is not None
                else np.nan
            ),
            "job_zone":  occ.get("job_zone"),
            "outlook":   occ.get("dws_star_rating"),
            "auto_avg":  occ.get("auto_avg_with_vals"),
        })
    df = pd.DataFrame(rows)

    # n_software lookup
    n_soft = pd.read_csv(TECH_SKILLS_SIMPLE_FILE) if TECH_SKILLS_SIMPLE_FILE.exists() else pd.DataFrame()
    if not n_soft.empty and "title" in n_soft.columns and "n_software" in n_soft.columns:
        df = df.merge(
            n_soft[["title", "n_software"]].rename(columns={"title": "title_current"}),
            on="title_current", how="left",
        )
    else:
        df["n_software"] = 0
    df["n_software"] = df["n_software"].fillna(0).astype(int)

    # Coerce numerics
    for c in ("ska_pct", "ska_delta", "auto_avg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Medians
    medians = {
        "ska_pct":   df["ska_pct"].median(),
        "pct_delta": df["pct_delta"].median(),
        "ska_delta": df["ska_delta"].median(),
        "n_software": df["n_software"].median(),
        "auto_avg":  df["auto_avg"].median(),
    }

    # Flags
    df["flag1_pct"]        = (df["pct"] > PCT_ABS_THRESHOLD).astype(int)
    df["flag2_ska"]        = (df["ska_pct"] > medians["ska_pct"]).astype(int)
    df["flag3_pct_trend"]  = ((df["pct_delta"] > 0) & (df["pct_delta"] > medians["pct_delta"])).astype(int)
    df["flag4_ska_trend"]  = ((df["ska_delta"] > 0) & (df["ska_delta"] > medians["ska_delta"])).astype(int)
    df["flag5_job_zone"]   = df["job_zone"].apply(
        lambda z: 1 if z is not None and not (isinstance(z, float) and math.isnan(z))
                  and int(round(float(z))) in RISK_AT_RISK_ZONE else 0)
    df["flag6_outlook"]    = df["outlook"].apply(
        lambda o: 1 if o is not None and not (isinstance(o, float) and math.isnan(o))
                  and int(round(float(o))) in RISK_AT_RISK_OUTLOOK else 0)
    df["flag7_n_software"] = (df["n_software"] > medians["n_software"]).astype(int)
    df["flag8_auto_aug"]   = (df["auto_avg"] > medians["auto_avg"]).astype(int)

    df["risk_score"] = sum(df[c] * w for c, w in FLAG_WEIGHTS.items())

    # ── Focused-set exposure gates (the 4 paper conditions) ──────────────────
    # (1) % tasks exposed > 50%, (2) SKA reach above median, (3) exposure growth
    # above median, (4) BLS 2025–34 employment projection negative.
    df["gate_pct"]    = (df["pct"] > PCT_ABS_THRESHOLD).astype(int)
    df["gate_ska"]    = df["flag2_ska"]
    df["gate_growth"] = df["flag3_pct_trend"]
    df["gate_emp"]    = (df["emp_proj"] < 0).astype(int)
    df["gates_count"] = df[["gate_pct", "gate_ska", "gate_growth", "gate_emp"]].sum(axis=1)

    def _tier(score: int, pct_val: float) -> str:
        if score >= 8:
            return "high" if pct_val >= EXPOSURE_GATE else "mod_high"
        if score >= 5:
            return "mod_high"
        if score >= 3:
            return "mod_low"
        return "low"

    df["risk_tier"] = [_tier(int(s), float(p)) for s, p in zip(df["risk_score"], df["pct"])]

    df.attrs["medians"] = medians
    _risk_table_cache = df
    return df


# ── Intensity rank (bias-corrected, equal-consensus, configscoped, no smoothing) ──

def _intensity_rank_table() -> dict[str, dict]:
    """For every occupation, compute its intensity ratio
    (Σ pct_norm bias-corrected / Σ freq×emp) and rank globally; also compute
    each major's intensity ratio and rank.

    Returns dict keyed by title_current → {
        occ_intensity_pct, occ_intensity_rank, occ_intensity_total,
        major_intensity_pct, major_intensity_rank, major_intensity_total,
    }
    """
    global _intensity_rank_cache
    if _intensity_rank_cache is not None:
        return _intensity_rank_cache

    meta = DATASETS.get(PRIMARY_DATASET, {})
    fpath = meta.get("file", "")
    if not Path(fpath).exists():
        _intensity_rank_cache = {}
        return _intensity_rank_cache

    bias = _equal_consensus_bias_ratios()
    df = pd.read_csv(fpath, low_memory=False)
    needed = {"task_normalized", "title_current", "major_occ_category", "gwa_title",
              "pct_normalized", "freq_mean", "emp_tot_nat_2025"}
    missing = needed - set(df.columns)
    assert not missing, f"Primary dataset missing columns: {missing}"
    for c in ("pct_normalized", "freq_mean", "emp_tot_nat_2025"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ai = df[df["pct_normalized"].notna()].copy()
    ai["eco_weight"] = ai["freq_mean"].fillna(0.0) * ai["emp_tot_nat_2025"].fillna(0.0)

    # ── Occupation level: dedupe (task, occ), bias-correct via avg over occ's GWAs
    dedup_occ = ai.drop_duplicates(["task_normalized", "title_current"])[
        ["task_normalized", "title_current", "pct_normalized", "eco_weight"]
    ].copy()
    gwa_pairs_occ = (ai.dropna(subset=["gwa_title"])
                     .drop_duplicates(["task_normalized", "title_current", "gwa_title"])
                     [["task_normalized", "title_current", "gwa_title"]].copy())
    gwa_pairs_occ["bias"] = gwa_pairs_occ["gwa_title"].map(bias).fillna(1.0)
    avg_bias_occ = (gwa_pairs_occ.groupby(["task_normalized", "title_current"])["bias"].mean()
                    .reset_index(name="avg_bias"))
    dedup_occ = dedup_occ.merge(avg_bias_occ, on=["task_normalized", "title_current"], how="left")
    dedup_occ["avg_bias"] = dedup_occ["avg_bias"].fillna(1.0).replace(0.0, 1.0)
    dedup_occ["adj"] = dedup_occ["pct_normalized"] / dedup_occ["avg_bias"]
    occ_num = dedup_occ.groupby("title_current")["adj"].sum().rename("num")
    occ_den = dedup_occ.groupby("title_current")["eco_weight"].sum().rename("den")
    occ_df = pd.concat([occ_num, occ_den], axis=1).fillna(0.0)
    occ_df["ratio"] = np.where(occ_df["den"] > 0, occ_df["num"] / occ_df["den"], 0.0)
    total_occ = occ_df["ratio"].sum() or 1.0
    occ_df["ratio_pct"] = occ_df["ratio"] / total_occ * 100.0
    occ_df = occ_df.sort_values("ratio_pct", ascending=False).reset_index()
    occ_df["rank"] = occ_df.index + 1
    occ_total = len(occ_df)

    # ── Major level: same logic but grouped by major_occ_category
    dedup_maj = ai.drop_duplicates(["task_normalized", "title_current"])[
        ["task_normalized", "title_current", "major_occ_category", "pct_normalized", "eco_weight"]
    ].copy()
    dedup_maj = dedup_maj.merge(avg_bias_occ, on=["task_normalized", "title_current"], how="left")
    dedup_maj["avg_bias"] = dedup_maj["avg_bias"].fillna(1.0).replace(0.0, 1.0)
    dedup_maj["adj"] = dedup_maj["pct_normalized"] / dedup_maj["avg_bias"]
    maj_num = dedup_maj.groupby("major_occ_category")["adj"].sum().rename("num")
    maj_den = dedup_maj.groupby("major_occ_category")["eco_weight"].sum().rename("den")
    maj_df = pd.concat([maj_num, maj_den], axis=1).fillna(0.0)
    maj_df["ratio"] = np.where(maj_df["den"] > 0, maj_df["num"] / maj_df["den"], 0.0)
    total_maj = maj_df["ratio"].sum() or 1.0
    maj_df["ratio_pct"] = maj_df["ratio"] / total_maj * 100.0
    maj_df = maj_df.sort_values("ratio_pct", ascending=False).reset_index()
    maj_df["rank"] = maj_df.index + 1
    maj_total = len(maj_df)

    occ_to_major = dict(zip(ai["title_current"], ai["major_occ_category"]))
    maj_lookup = dict(zip(maj_df["major_occ_category"], zip(maj_df["ratio_pct"], maj_df["rank"])))

    out: dict[str, dict] = {}
    for _, row in occ_df.iterrows():
        title = row["title_current"]
        major = occ_to_major.get(title)
        maj_pct, maj_rank = maj_lookup.get(major, (None, None))
        out[title] = {
            "occ_intensity_pct":   _round_or_none(row["ratio_pct"], 3),
            "occ_intensity_rank":  int(row["rank"]),
            "occ_intensity_total": occ_total,
            "major_intensity_pct":   _round_or_none(maj_pct, 3),
            "major_intensity_rank":  int(maj_rank) if maj_rank is not None else None,
            "major_intensity_total": maj_total,
        }
    _intensity_rank_cache = out
    return out


# ── Similarity matrix (L1 over SKA imp×lv profile) ────────────────────────────

def _ska_profile_matrix() -> tuple[list[str], np.ndarray, list[str]]:
    """Build a (n_occs × n_elements) matrix of imp×lv values (importance>=3 only).
    Cells where the occ has no record for that element get 0. Returns
    (titles, matrix, element_names_in_col_order)."""
    global _ska_profile_matrix_cache
    if _ska_profile_matrix_cache is not None:
        titles, matrix = _ska_profile_matrix_cache
        # element names not cached separately (not needed for similarity calc),
        # but we recompute on demand below
        return titles, matrix, []
    ska = _load_ska_data()
    pieces: list[pd.DataFrame] = []
    for type_name, df in [("skills", ska.skills), ("abilities", ska.abilities), ("knowledge", ska.knowledge)]:
        sub = df[df["importance"] >= IMPORTANCE_THRESHOLD].copy()
        sub["score"] = sub["importance"] * sub["level"]
        sub["element_id"] = type_name + "::" + sub["element_name"]
        pieces.append(sub[["title", "element_id", "score"]])
    long = pd.concat(pieces, ignore_index=True)
    pivot = long.pivot_table(index="title", columns="element_id", values="score", aggfunc="first")
    pivot = pivot.fillna(0.0)
    titles = pivot.index.astype(str).tolist()
    matrix = pivot.to_numpy(dtype=float)
    _ska_profile_matrix_cache = (titles, matrix)
    return titles, matrix, pivot.columns.tolist()


def _similar_occs(title: str, n: int = N_SIMILAR_OCCS) -> list[dict]:
    """L1 distance over SKA profile vector. Returns list of nearest occ dicts.
    Each row also carries the occupation's exposure (risk) score / tier / flags
    so the UI can render an exposure profile column."""
    titles, matrix, _ = _ska_profile_matrix()
    if title not in titles:
        return []
    idx = titles.index(title)
    target = matrix[idx]
    # L1 distance to all rows
    dists = np.abs(matrix - target).sum(axis=1)
    # Sort ascending, drop self
    order = np.argsort(dists)
    occ_idx = _occ_index()
    pct_map = _pct_for(PRIMARY_DATASET)
    risk_df = _risk_table().set_index("title_current")

    out: list[dict] = []
    for j in order:
        if j == idx:
            continue
        other = titles[j]
        meta = occ_idx.get(other)
        if meta is None:
            continue
        risk_payload: Optional[dict] = None
        if other in risk_df.index:
            r = risk_df.loc[other]
            risk_payload = {
                "score": int(r["risk_score"]),
                "tier":  str(r["risk_tier"]),
                "flags": {k: int(r[k]) for k in FLAG_WEIGHTS.keys()},
            }
        out.append({
            "title": other,
            "distance": _round_or_none(float(dists[j]), 1),
            "pct_tasks_affected": _round_or_none(float(pct_map.get(other, 0.0) or 0.0), 1),
            "wage": _safe_num(meta.get("wage")),
            "job_zone": _safe_num(meta.get("job_zone")),
            "dws_star_rating": _safe_num(meta.get("dws_star_rating")),
            "major": meta.get("major"),
            "risk": risk_payload,
        })
        if len(out) >= n:
            break
    return out


# ── Section builders ──────────────────────────────────────────────────────────

def _raw_emp_wage(title: str, geo: str) -> tuple[Optional[float], Optional[float]]:
    """Return (occupation total emp, median wage) for the given geo from eco_2025.
    No AI weighting — just BLS OEWS numbers for this occ."""
    eco = load_eco_raw()
    if eco is None:
        return None, None
    emp_col  = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"
    if emp_col not in eco.columns:
        emp_col = "emp_tot_nat_2025"
    if wage_col not in eco.columns:
        wage_col = "a_med_nat_2025"
    sub = eco[eco["title_current"] == title]
    if sub.empty:
        return None, None
    return _safe_num(sub.iloc[0].get(emp_col)), _safe_num(sub.iloc[0].get(wage_col))


def _build_headline(title: str, geo: str) -> dict:
    workers, wages, pct = _emp_wage_for(PRIMARY_DATASET, geo)
    occ_idx = _occ_index()
    meta = occ_idx.get(title, {})

    raw_emp, raw_wage = _raw_emp_wage(title, geo)

    # Risk score
    risk_df = _risk_table()
    risk_row = risk_df[risk_df["title_current"] == title]
    if not risk_row.empty:
        r = risk_row.iloc[0]
        risk_payload = {
            "score": int(r["risk_score"]),
            "tier":  str(r["risk_tier"]),
            "flags": {k: int(r[k]) for k in FLAG_WEIGHTS.keys()},
        }
    else:
        risk_payload = {"score": 0, "tier": "low", "flags": {k: 0 for k in FLAG_WEIGHTS}}

    # Focused-set exposure gates (the 4-signal tier shown on the page)
    if not risk_row.empty:
        rr = risk_row.iloc[0]
        gates_payload = {
            "pct":         int(rr["gate_pct"]),
            "ska":         int(rr["gate_ska"]),
            "growth":      int(rr["gate_growth"]),
            "emp_decline": int(rr["gate_emp"]),
            "count":       int(rr["gates_count"]),
            "emp_proj":    _round_or_none(_safe_num(rr.get("emp_proj")), 1),
        }
    else:
        gates_payload = {"pct": 0, "ska": 0, "growth": 0, "emp_decline": 0, "count": 0, "emp_proj": None}

    # Intensity
    intensity = _intensity_rank_table().get(title, {})

    return {
        "title":              title,
        "major":              meta.get("major"),
        "minor":              meta.get("minor"),
        "broad":              meta.get("broad"),
        "job_zone":           meta.get("job_zone"),
        "dws_star_rating":    meta.get("dws_star_rating"),
        "n_tasks":            meta.get("n_tasks"),
        "pct_physical":       meta.get("pct_physical"),
        "emp":                _round_or_none(raw_emp, 0),
        "wage":               _round_or_none(raw_wage, 0),
        "pct_tasks_affected": _round_or_none(pct.get(title), 1),
        "workers_affected":   _round_or_none(workers.get(title), 0),
        "wages_affected":     _round_or_none(wages.get(title), 0),
        "risk":               risk_payload,
        "gates":              gates_payload,
        "intensity":          intensity,
    }


# ── All-confirmed per-(task, occ) lookup + interpretive auto label ────────────

_primary_pairs_cache: Optional[dict] = None


def _primary_pairs_lookup() -> dict:
    """{(title_current, task_normalized): {auto_aug, pct_norm, freq}} from the
    PRIMARY (All Confirmed) dataset — the single source for a task's headline
    auto value, color bucket, and usage multiplier on the occupation page."""
    global _primary_pairs_cache
    if _primary_pairs_cache is not None:
        return _primary_pairs_cache
    out: dict = {}
    fpath = DATASETS.get(PRIMARY_DATASET, {}).get("file", "")
    if Path(fpath).exists():
        df = pd.read_csv(fpath, low_memory=False)
        for c in ("auto_aug_mean", "pct_normalized", "freq_mean"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if {"title_current", "task_normalized"} <= set(df.columns):
            sub = df.drop_duplicates(subset=["title_current", "task_normalized"])
            for _, r in sub.iterrows():
                out[(r["title_current"], r["task_normalized"])] = {
                    "auto_aug": _safe_num(r.get("auto_aug_mean")),
                    "pct_norm": _safe_num(r.get("pct_normalized")),
                    "freq":     _safe_num(r.get("freq_mean")),
                }
    _primary_pairs_cache = out
    return out


def _auto_label(score: Optional[float]) -> str:
    """Interpretive label for a 0–5 auto-aug value, so the number reads in place."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "no usage seen"
    if score >= 4.5:
        return "most automated usage seen"
    if score >= 3.5:
        return "automated usage seen"
    if score >= 2.5:
        return "mixed usage seen"
    if score >= 1.5:
        return "augmentative usage seen"
    return "low automation usage seen"


def _build_tasks(title: str) -> list[dict]:
    """Per-task rows for the occ. Each row carries: the All-Confirmed auto value
    (single source) + interpretive label + color bucket, centrality (freq×imp×rel,
    ranked client-side), a usage-vs-median multiplier (pct/freq ÷ occ median over
    rated tasks), the task's GWA/IWA/DWA with each activity's auto + tasks-exposed
    rank, and the top-5 MCP servers."""
    eco = load_eco_raw()
    if eco is None:
        return []
    occ_tasks = eco[eco["title_current"] == title].copy()
    if occ_tasks.empty:
        return []
    occ_tasks = occ_tasks.sort_values("task")
    pairs = _primary_pairs_lookup()           # (title, tn) → all-confirmed auto/pct/freq
    top_mcps_lookup = _build_top_mcps_lookup()
    desc_lookup = _mcp_titles_desc_lookup()
    wa_stats = _eco_wa_stats("nat")           # auto + tasks-exposed rank per WA (geo-independent)

    # usage-vs-median denominator: median of (pct / freq) across this occ's rated tasks
    usage_vals: list[float] = []
    for _, row in occ_tasks.drop_duplicates(subset=["task_normalized"]).iterrows():
        pp = pairs.get((title, row["task_normalized"]))
        if pp and pp.get("pct_norm") and pp.get("freq") and pp["freq"] > 0:
            usage_vals.append(pp["pct_norm"] / pp["freq"])
    usage_median = float(np.median(usage_vals)) if usage_vals else 0.0

    def _wa_detail(level: str, name) -> Optional[dict]:
        if not name:
            return None
        rec = wa_stats.get(level, {}).get(str(name))
        if not rec:
            return {"name": str(name), "auto": None, "rank_pct": None, "total": None}
        return {
            "name": str(name),
            "auto": rec.get("auto_aug_mean"),
            "rank_pct": rec.get("rank_pct"),
            "total": rec.get("total"),
        }

    rows: list[dict] = []
    seen_tn: set = set()
    for _, row in occ_tasks.iterrows():
        tn = row["task_normalized"]
        if tn in seen_tn:
            continue
        seen_tn.add(tn)

        pp = pairs.get((title, tn), {})
        auto = pp.get("auto_aug")               # All-Confirmed auto value (the single source)
        pct_norm = pp.get("pct_norm")
        freq = pp.get("freq") or _safe_num(row.get("freq_mean"))

        imp = _safe_num(row.get("importance")) or 0.0
        rel = _safe_num(row.get("relevance")) or 0.0
        fq  = _safe_num(row.get("freq_mean")) or 0.0
        centrality = fq * imp * rel             # freq×imp×rel; ranked client-side

        usage_mult = None
        if pct_norm and freq and freq > 0 and usage_median > 0:
            usage_mult = (pct_norm / freq) / usage_median

        mcps_raw = top_mcps_lookup.get(tn, [])
        mcps: list[dict] = []
        for m in mcps_raw[:N_TASK_MCPS]:
            t = (m.get("title") or "").strip()
            mcps.append({
                "title": t,
                "rating": m.get("rating"),
                "url": m.get("url"),
                "description": desc_lookup.get(t) or desc_lookup.get(t.lower()),
            })

        rows.append({
            "task":             row["task"],
            "task_normalized":  tn,
            "importance":       imp,
            "freq_mean":        fq,
            "relevance":        rel,
            "centrality":       _round_or_none(centrality, 1),
            "physical":         bool(row["physical"]) if pd.notna(row.get("physical")) else False,
            "auto":             _round_or_none(auto, 1),     # All-Confirmed auto-aug
            "auto_label":       _auto_label(auto),
            "color_bucket":     _color_bucket_auto(auto),
            "usage_mult":       _round_or_none(usage_mult, 1),
            "gwa":              _wa_detail("gwa", row.get("gwa_title")),
            "iwa":              _wa_detail("iwa", row.get("iwa_title")),
            "dwa":              _wa_detail("dwa", row.get("dwa_title")),
            "top_mcps":         mcps,
        })
    # Sort by All-Confirmed auto desc (most-automated first); rank within the occ
    rows.sort(key=lambda r: (r["auto"] is None, -(r["auto"] or 0.0)))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _eco_wa_stats(geo: str) -> dict[str, dict[str, dict]]:
    """Economy-wide stats per WA at gwa/iwa/dwa, computed against PRIMARY_DATASET.

    Returns: {level: {wa_name: {pct_tasks_affected, workers_affected, wages_affected,
                                 auto_aug_mean, rank_pct, rank_workers, rank_wages,
                                 rank_auto, total}}}.

    Cached per geo. Workers/wages depend on geo; pct/auto_aug do not, but we store
    them under the same key for simplicity.
    """
    if geo in _eco_wa_stats_cache:
        return _eco_wa_stats_cache[geo]

    settings = {
        "selected_datasets": [PRIMARY_DATASET],
        "combine_method":    "Average",
        "method":            "freq",
        "use_auto_aug":      True,
        "physical_mode":     "all",
        "geo":               geo,
        "agg_level":         "occupation",
        "sort_by":           "Workers Affected",
        "top_n":             9999,
        "search_query":      "",
        "context_size":      3,
    }
    wa_result = compute_work_activities(settings)
    # PRIMARY is is_aei=False → comes back as mcp_group
    group = (wa_result or {}).get("mcp_group") or (wa_result or {}).get("aei_group") or {}

    # Per-WA auto_aug_mean from the dataset CSV (one pass, level-agnostic)
    auto_by_level: dict[str, dict[str, float]] = {"gwa": {}, "iwa": {}, "dwa": {}}
    fpath = DATASETS.get(PRIMARY_DATASET, {}).get("file", "")
    if Path(fpath).exists():
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if "auto_aug_mean" in df.columns:
                df["auto_aug_mean"] = pd.to_numeric(df["auto_aug_mean"], errors="coerce")
                for level_key, col in (("gwa", "gwa_title"), ("iwa", "iwa_title"), ("dwa", "dwa_title")):
                    if col not in df.columns:
                        continue
                    sub = df[df[col].notna() & df["auto_aug_mean"].notna()].copy()
                    # Dedup by (task_normalized, wa) so a task counted once per WA
                    if "task_normalized" in sub.columns:
                        sub = sub.drop_duplicates(subset=["task_normalized", col])
                    auto_by_level[level_key] = sub.groupby(col)["auto_aug_mean"].mean().to_dict()
        except Exception:
            pass  # leave auto empty; eco_stats just won't have auto

    out: dict[str, dict[str, dict]] = {"gwa": {}, "iwa": {}, "dwa": {}}
    for level_key in ("gwa", "iwa", "dwa"):
        rows = group.get(level_key) or []
        if not rows:
            continue
        rec_df = pd.DataFrame(rows)
        if rec_df.empty or "category" not in rec_df.columns:
            continue
        # Attach auto and compute per-metric ranks (1 = highest)
        rec_df["auto_aug_mean"] = rec_df["category"].map(auto_by_level.get(level_key, {})).astype(float)
        for metric in ("pct_tasks_affected", "workers_affected", "wages_affected", "auto_aug_mean"):
            if metric not in rec_df.columns:
                rec_df[metric] = np.nan
            rec_df[f"rank_{metric.split('_')[0]}"] = (
                rec_df[metric].rank(ascending=False, method="min", na_option="bottom").astype("Int64")
            )
        total = int(len(rec_df))
        for _, r in rec_df.iterrows():
            name = str(r["category"])
            out[level_key][name] = {
                "pct_tasks_affected": _round_or_none(r.get("pct_tasks_affected"), 1),
                "workers_affected":   _round_or_none(r.get("workers_affected"), 0),
                "wages_affected":     _round_or_none(r.get("wages_affected"), 0),
                "auto_aug_mean":      _round_or_none(r.get("auto_aug_mean"), 2),
                "rank_pct":           int(r["rank_pct"])     if pd.notna(r["rank_pct"])     else None,
                "rank_workers":       int(r["rank_workers"]) if pd.notna(r["rank_workers"]) else None,
                "rank_wages":         int(r["rank_wages"])   if pd.notna(r["rank_wages"])   else None,
                "rank_auto":          int(r["rank_auto"])    if pd.notna(r["rank_auto"])    else None,
                "total":              total,
            }

    _eco_wa_stats_cache[geo] = out
    return out


def _build_was(tasks: list[dict], geo: str) -> dict:
    """Roll up tasks → GWA/IWA/DWA. For each WA, average the per-task
    auto_aug values across the same color buckets, and attach economy-wide
    eco_stats (pct/workers/wages/auto + ranks within all WAs at that level)."""
    eco_stats = _eco_wa_stats(geo)

    def _rollup(level_key: str, eco_key: str) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for t in tasks:
            name = t.get(level_key)
            if not name:
                continue
            groups.setdefault(name, []).append(t)
        out: list[dict] = []
        eco_for_level = eco_stats.get(eco_key, {})
        for name, ts in groups.items():
            n = len(ts)
            def _avg(field: str) -> Optional[float]:
                xs = [t[field] for t in ts if t.get(field) is not None]
                return round(sum(xs) / len(xs), 2) if xs else None
            color_avg = _avg("color_driver")
            out.append({
                "name":           name,
                "n_tasks":        n,
                "aei_conv_max":   _avg("aei_conv_max"),
                "aei_api_max":    _avg("aei_api_max"),
                "microsoft":      _avg("microsoft"),
                "mcp":            _avg("mcp"),
                "color_driver":   color_avg,
                "color_bucket":   _color_bucket_auto(color_avg),
                "avg_importance": _avg("importance"),
                "eco_stats":      eco_for_level.get(name),
            })
        out.sort(key=lambda r: (r["color_driver"] is None, -(r["color_driver"] or 0.0)))
        for i, r in enumerate(out, start=1):
            r["rank"] = i
        return out

    return {
        "gwa": _rollup("gwa_title", "gwa"),
        "iwa": _rollup("iwa_title", "iwa"),
        "dwa": _rollup("dwa_title", "dwa"),
    }


def _build_group_ranks(title: str, geo: str) -> dict:
    occ_idx = _occ_index()
    meta = occ_idx.get(title, {})
    major = meta.get("major"); minor = meta.get("minor"); broad = meta.get("broad")
    workers, wages, pct = _emp_wage_for(PRIMARY_DATASET, geo)

    # Build dataframes per major/minor/broad based on pct/workers/wages
    df = pd.DataFrame([
        {"title": t, "major": occ_idx[t].get("major"),
         "minor": occ_idx[t].get("minor"), "broad": occ_idx[t].get("broad"),
         "pct":     float(pct.get(t, 0.0) or 0.0),
         "workers": float(workers.get(t, 0.0) or 0.0),
         "wages":   float(wages.get(t, 0.0) or 0.0)}
        for t in occ_idx
    ])

    def _ranks(group_col: str, group_val: Optional[str]) -> Optional[dict]:
        if not group_val:
            return None
        sub = df[df[group_col] == group_val].copy()
        if sub.empty or title not in set(sub["title"]):
            return None
        n = len(sub)
        out: dict[str, int] = {}
        for metric in ("pct", "workers", "wages"):
            sorted_sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)
            sorted_sub["r"] = sorted_sub.index + 1
            row = sorted_sub[sorted_sub["title"] == title]
            if not row.empty:
                out[metric] = int(row.iloc[0]["r"])
        out["total"] = n
        return out

    # economy-wide ranks (out of all 923)
    n_total = len(df)
    eco_ranks: dict[str, int] = {}
    for metric in ("pct", "workers", "wages"):
        sorted_df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        sorted_df["r"] = sorted_df.index + 1
        row = sorted_df[sorted_df["title"] == title]
        if not row.empty:
            eco_ranks[metric] = int(row.iloc[0]["r"])
    eco_ranks["total"] = n_total

    return {
        "economy": eco_ranks,
        "major":   _ranks("major", major),
        "minor":   _ranks("minor", minor),
        "broad":   _ranks("broad", broad),
    }


def _build_trend(title: str, geo: str) -> list[dict]:
    """Per-snapshot pct_tasks_affected over the all_confirmed series."""
    out: list[dict] = []
    for ds in TREND_SERIES:
        if not Path(DATASETS.get(ds, {}).get("file", "")).exists():
            continue
        pct = _pct_for(ds, geo)
        meta = DATASETS.get(ds, {})
        # Date is in the dataset name
        date = ds.split(" ")[-1]
        out.append({
            "dataset": ds,
            "date":    date,
            "pct_tasks_affected": _round_or_none(pct.get(title), 1),
        })
    return out


def _build_ska(title: str) -> dict:
    """SKA gap rows for one occ. Each row: type, element, importance, level,
    occ_score, ai_top10 reference, gap (ai_top10 − occ_score),
    pct_of_need (ai_top10 / occ_score × 100), color bucket. Sorted by gap desc
    (biggest AI lead first) within each type."""
    result = _ska_for(PRIMARY_DATASET)
    top10_df = _ska_top10_per_element()
    top10_lookup = {(r["type"], r["element_name"]): r["ai_top10"] for _, r in top10_df.iterrows()}

    out: dict[str, list[dict]] = {"skills": [], "abilities": [], "knowledge": []}
    for type_name, occ_elem in result.occ_element_scores.items():
        if occ_elem.empty:
            continue
        sub = occ_elem[occ_elem["title_current"] == title].copy()
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            elem = row["element_name"]
            occ_score = float(row["occ_score"])
            ai_top10 = top10_lookup.get((type_name, elem))
            pct_of_need = (ai_top10 / occ_score * 100.0) if (ai_top10 is not None and occ_score > 0) else None
            gap = (ai_top10 - occ_score) if ai_top10 is not None else None
            out[type_name].append({
                "element":      elem,
                "importance":   _round_or_none(row.get("importance"), 1),
                "level":        _round_or_none(row.get("level"), 1),
                "occ_score":    _round_or_none(occ_score, 1),
                "ai_top10":     _round_or_none(ai_top10, 2),
                "gap":          _round_or_none(gap, 2),
                "pct_of_need":  _round_or_none(pct_of_need, 1),
                "color_bucket": _color_bucket_ska(pct_of_need),
            })
        # Sort by gap desc (biggest AI advantage first), then by element name
        out[type_name].sort(key=lambda r: (
            -1e9 if r["gap"] is None else -r["gap"],
            r["element"] or "",
        ))

    # Per-occ summary (overall_pct etc.)
    summary_row = result.occ_gaps[result.occ_gaps["title_current"] == title]
    if not summary_row.empty:
        s = summary_row.iloc[0]
        summary = {
            "skills_pct":    _round_or_none(s.get("skills_pct"), 1),
            "abilities_pct": _round_or_none(s.get("abilities_pct"), 1),
            "knowledge_pct": _round_or_none(s.get("knowledge_pct"), 1),
            "overall_pct":   _round_or_none(s.get("overall_pct"), 1),
        }
    else:
        summary = {}

    return {"summary": summary, "rows": out}


_SECTOR_LEVEL_GROUP_COL: dict[str, str] = {
    "major": "major_occ_category",
    "minor": "minor_occ_category",
    "broad": "broad_occ",
}


def _ranked_group_df(level: str, geo: str) -> Optional[pd.DataFrame]:
    """Compute the full ranked group dataframe at one SOC level for PRIMARY_DATASET.
    Cached per (level, geo). Returns DataFrame with rank columns added."""
    cache_key = (level, geo)
    if cache_key in _sector_level_cache:
        return _sector_level_cache[cache_key]
    settings = {
        "selected_datasets": [PRIMARY_DATASET], "combine_method": "Average",
        "method": "freq", "use_auto_aug": True,
        "physical_mode": "all", "geo": geo, "agg_level": level,
        "sort_by": "% Tasks Affected", "top_n": 9999,
        "search_query": "", "context_size": 3,
    }
    data = get_group_data(settings)
    if data is None:
        return None
    df: pd.DataFrame = data["df"].copy().reset_index(drop=True)
    df["rank_pct"]     = df["pct_tasks_affected"].rank(ascending=False, method="min").astype(int)
    df["rank_workers"] = df["workers_affected"].rank(ascending=False, method="min").astype(int)
    df["rank_wages"]   = df["wages_affected"].rank(ascending=False, method="min").astype(int)
    _sector_level_cache[cache_key] = df
    return df


def _sector_stats_at_level(title: str, level: str, geo: str) -> Optional[dict]:
    """Stats for the occ's category at one SOC level (major/minor/broad)."""
    occ_idx = _occ_index()
    meta = occ_idx.get(title, {})
    level_meta_key = {"major": "major", "minor": "minor", "broad": "broad"}[level]
    cat = meta.get(level_meta_key)
    if not cat:
        return None
    df = _ranked_group_df(level, geo)
    if df is None:
        return None
    group_col = _SECTOR_LEVEL_GROUP_COL[level]
    if group_col not in df.columns:
        return None
    row = df[df[group_col] == cat]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "name":               cat,
        "level":              level,
        "pct_tasks_affected": _round_or_none(r["pct_tasks_affected"], 1),
        "workers_affected":   _round_or_none(r["workers_affected"], 0),
        "wages_affected":     _round_or_none(r["wages_affected"], 0),
        "rank_pct":           int(r["rank_pct"]),
        "rank_workers":       int(r["rank_workers"]),
        "rank_wages":         int(r["rank_wages"]),
        "total":              int(len(df)),
    }


def _build_sector_stats(title: str, geo: str) -> dict:
    """Major sector stats with the legacy field names (kept for backwards
    compatibility with the existing `sector` payload key)."""
    s = _sector_stats_at_level(title, "major", geo)
    if not s:
        return {}
    return {
        "major":              s["name"],
        "pct_tasks_affected": s["pct_tasks_affected"],
        "workers_affected":   s["workers_affected"],
        "wages_affected":     s["wages_affected"],
        "rank_pct":           s["rank_pct"],
        "rank_workers":       s["rank_workers"],
        "rank_wages":         s["rank_wages"],
        "n_majors":           s["total"],
    }


def _build_sector_chain(title: str, geo: str) -> dict:
    """Per-level (major/minor/broad) economy-wide stats + ranks for the occ."""
    return {
        "major": _sector_stats_at_level(title, "major", geo),
        "minor": _sector_stats_at_level(title, "minor", geo),
        "broad": _sector_stats_at_level(title, "broad", geo),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_occupation_titles() -> list[str]:
    """All occupation titles available — used by the frontend's occupation picker."""
    return sorted(_occ_index().keys())


def get_occupation_hierarchy() -> list[dict]:
    """Same set of occupations, plus their major/minor/broad. Sorted by title.

    Used by the frontend's Browse-by-category picker so it can build cascading
    Major → Minor → Broad → Occupation dropdowns without an extra round-trip.
    """
    occ_idx = _occ_index()
    out: list[dict] = []
    for title in sorted(occ_idx.keys()):
        meta = occ_idx[title]
        out.append({
            "title": title,
            "major": meta.get("major"),
            "minor": meta.get("minor"),
            "broad": meta.get("broad"),
        })
    return out


def _build_major_ranking(title: str, window: int = 5) -> dict:
    """The occupation's place within its MAJOR category, on two metrics:
    % tasks exposed and AI adoption (intensity). Returns a ±`window` slice of
    neighbours around the occ for each, so the page can show two small ranked
    charts with the occ centred."""
    occ_idx = _occ_index()
    meta = occ_idx.get(title, {})
    major = meta.get("major")
    if not major:
        return {}

    same_major = [t for t, m in occ_idx.items() if m.get("major") == major]

    pct_map = _pct_for(PRIMARY_DATASET)
    intens = _intensity_rank_table()

    def _window(value_of) -> dict:
        ranked = sorted(same_major, key=lambda t: value_of(t), reverse=True)
        try:
            i = ranked.index(title)
        except ValueError:
            return {"rank": None, "total": len(ranked), "window": []}
        lo = max(0, i - window)
        hi = min(len(ranked), i + window + 1)
        win = [{
            "title": t,
            "value": _round_or_none(float(value_of(t)), 2),
            "rank":  j + 1,
            "is_occ": t == title,
        } for j, t in list(enumerate(ranked))[lo:hi]]
        return {"rank": i + 1, "total": len(ranked), "window": win}

    return {
        "major": major,
        "pct":      _window(lambda t: float(pct_map.get(t, 0.0) or 0.0)),
        "adoption": _window(lambda t: float((intens.get(t, {}) or {}).get("occ_intensity_pct", 0.0) or 0.0)),
    }


def get_occupation_report(title: str, geo: str = "nat") -> Optional[dict]:
    """Build the full per-occupation report payload."""
    occ_idx = _occ_index()
    if title not in occ_idx:
        return None
    if geo not in GEO_OPTIONS:
        geo = "nat"

    headline = _build_headline(title, geo)
    tasks    = _build_tasks(title)
    was      = _build_was(tasks, geo)
    group_ranks = _build_group_ranks(title, geo)
    trend    = _build_trend(title, geo)
    ska      = _build_ska(title)
    major_ranking = _build_major_ranking(title)
    sector   = _build_sector_stats(title, geo)
    sector_chain = _build_sector_chain(title, geo)
    similar  = _similar_occs(title, n=N_SIMILAR_OCCS)
    tech     = _tech_for_occ(title)

    return {
        "title":     title,
        "geo":       geo,
        "primary_dataset": PRIMARY_DATASET,
        "headline":  headline,
        "tasks":     tasks,
        "work_activities": was,
        "group_ranks":     group_ranks,
        "trend":     trend,
        "ska":       ska,
        "major_ranking": major_ranking,
        "sector":    sector,
        "sector_chain": sector_chain,
        "similar":   similar,
        "tech":      tech,
    }
