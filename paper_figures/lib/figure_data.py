"""Shared data layer for the paper figures.

Every figure in `run_main_figures.py` and `run_supplemental_figures.py` is
built from this module: the 2026-05-31 dataset pins, the work-time exposure
helpers, the debiased usage-intensity helpers, the verb-family taxonomy and
the compact label maps.

Exposure here is weighted by **work time**, not task frequency. Each
(task, occupation) row carries `time_per_day` — estimated hours per day,
normalised upstream so an occupation's tasks sum to exactly OCC_DAY_HOURS —
so an exposure percentage is the share of the workday AI reaches, and at
group level it is employment-weighted. Usage intensity is deliberately
different: it divides by employment alone, because dividing by time would
penalise slow tasks.

The module pins its own dataset constants rather than reading
ANALYSIS_CONFIGS, and registers the handful of eco_2025-rebased files it
needs into `backend.config.DATASETS` at import time, so the committed
dataset registry stays untouched. Bias ratios for the usage numerator come
from `lib.exploratory.intensity` (the equal 3-source consensus GWA prior).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# lib.config owns the sys.path bootstrap (repo root, backend/, paper_figures/)
# so `backend.*` and `lib.*` resolve however a figure script is launched.
from lib.config import DATA_DIR, ROOT  # noqa: F401,E402

from backend import config as backend_config           # noqa: E402
from lib.exploratory.intensity import (  # noqa: E402
    BIAS_VARIANTS,
    EMP_COL,
    compute_bias_ratios,
)

# ── Dataset pins (May 2026 vintage) ───────────────────────────────────────

PRIMARY_DATASET = "AEI Both + Micro 2026-05-31"          # all_confirmed
AGENTIC_DATASET = "AEI API eco2025 2026-05-31"           # agentic_confirmed (eco_2025)
AGENTIC_CEILING_DATASET = "MCP + API 2026-05-31"         # agentic_ceiling
WA_INTENSITY_DATASET = "AEI Both eco2025 2026-05-31"     # intensity numerator as a dataset
CONV_DATASET = "AEI Conv + Micro 2026-05-31"             # human_conversation
ALL_CEILING_DATASET = "All 2026-05-31"                   # all_ceiling

# Every weight in this folder is the per-(task, occ) estimated hours per day.
# It is normalised upstream so an occupation's tasks sum to exactly
# OCC_DAY_HOURS, which is what makes an exposure percentage a share of the
# workday and makes `time_per_day / OCC_DAY_HOURS` a set of task shares that
# sum to 1 within an occupation.
TIME_COL = "time_per_day"
OCC_DAY_HOURS = 7.0

# EMP_COL comes from audit_pct_norm_eco (national OEWS employment); the wage
# column is its median-annual-wage counterpart for the same geography.
WAGE_COL = "a_med_nat_2025"

# ── Geography ─────────────────────────────────────────────────────────────
# The figures are national and pass nothing; the dashboard passes a state
# code. Every loader below takes `geo` and adds canonical `emp` / `wage`
# columns resolved for it, so downstream math never names a geography.
#
# Group exposure percentages are employment-weighted (see major_exposure),
# so a state's number is not the national one relabelled — it re-weights
# that state's occupational mix. Read it as "the exposure of this state's
# job mix"; a single occupation's exposure is identical everywhere.
#
# When geo == "nat" the raw national columns are left on the frame as well,
# so existing figure code that reaches for EMP_COL directly keeps working.

DEFAULT_GEO = "nat"


def emp_col(geo: str = DEFAULT_GEO) -> str:
    """Employment column for a geography ("nat", "ut", "ia", …)."""
    return f"emp_tot_{geo}_2025"


def wage_col(geo: str = DEFAULT_GEO) -> str:
    """Median-annual-wage column for a geography."""
    return f"a_med_{geo}_2025"


def _add_geo_cols(df: pd.DataFrame, geo: str, *, wage: bool = True) -> pd.DataFrame:
    """Attach canonical `emp` (and `wage`) columns for `geo`, in place.

    Asserts the geography actually exists in the file rather than letting a
    typo silently produce an all-NaN employment column — which would sail
    through every downstream sum as a zero.
    """
    ec = emp_col(geo)
    assert ec in df.columns, f"Unknown geography {geo!r}: {ec} not in frame"
    df["emp"] = pd.to_numeric(df[ec], errors="coerce").fillna(0.0)
    if wage:
        wc = wage_col(geo)
        assert wc in df.columns, f"Unknown geography {geo!r}: {wc} not in frame"
        df["wage"] = pd.to_numeric(df[wc], errors="coerce")
    return df

# The five canonical ANALYSIS_CONFIGS keys pinned to the May 2026 vintage.
# Every eco_2025-based (is_aei=False) so pair/act-row helpers below apply to
# all of them uniformly.
CONFIG_DATASETS: dict[str, str] = {
    "all_confirmed":      PRIMARY_DATASET,
    "human_conversation": CONV_DATASET,
    "agentic_confirmed":  AGENTIC_DATASET,
    "agentic_ceiling":    AGENTIC_CEILING_DATASET,
    "all_ceiling":        ALL_CEILING_DATASET,
}

CONFIG_LABELS: dict[str, str] = {
    "all_confirmed":      "All Confirmed Usage",
    "human_conversation": "Conversational Usage",
    "agentic_confirmed":  "Agentic Confirmed Usage",
    "agentic_ceiling":    "Agentic Ceiling",
    "all_ceiling":        "All Ceiling",
}

# Most recent standalone MCP pipeline snapshot (tool specs, not usage).
MCP_DATASET = "MCP Cumul. v4"

# All Confirmed paper-window series through May 2026.
AC_SERIES: list[str] = [
    "AEI Both + Micro 2025-08-11",
    "AEI Both + Micro 2025-11-13",
    "AEI Both + Micro 2026-02-12",
    "AEI Both + Micro 2026-04-30",
    "AEI Both + Micro 2026-05-31",
]

# Cumulative AEI API files rebased onto eco_2025 for every window snapshot.
AGENTIC_ECO2025_SERIES: list[str] = [
    "AEI API eco2025 2025-08-11",
    "AEI API eco2025 2025-11-13",
    "AEI API eco2025 2026-02-12",
    "AEI API eco2025 2026-04-30",
    "AEI API eco2025 2026-05-31",
]

_EXTRA_DATASETS: dict[str, dict] = {
    f"AEI API eco2025 {d}": {
        "file": str(DATA_DIR / f"final_aei_agentic_usage_eco2025_{d}.csv"),
        "is_aei": False,
        "is_mcp": False,
    }
    for d in ["2025-08-11", "2025-11-13", "2026-02-12", "2026-04-30", "2026-05-31"]
}
_EXTRA_DATASETS.update({
    "AEI Conv eco2025 2026-05-31": {
        "file": str(DATA_DIR / "final_aei_human_usage_eco2025_2026-05-31.csv"),
        "is_aei": False, "is_mcp": False,
    },
    "AEI Both + Micro 2026-04-30": {
        "file": str(DATA_DIR / "final_all_confirmed_usage_2026-04-30.csv"),
        "is_aei": False, "is_mcp": False,
    },
    "AEI Both + Micro 2026-05-31": {
        "file": str(DATA_DIR / "final_all_confirmed_usage_2026-05-31.csv"),
        "is_aei": False, "is_mcp": False,
    },
    # The `_2025_` files (Aug 2026 build) predate the time_per_day column;
    # the `_eco2025_` twins are the same rows with it, so every weight in
    # this folder can come from one place.
    "AEI Both eco2025 2026-05-31": {
        "file": str(DATA_DIR / "final_aei_all_usage_eco2025_2026-05-31.csv"),
        "is_aei": False, "is_mcp": False,
    },
    "MCP + API 2026-05-31": {
        "file": str(DATA_DIR / "final_all_agentic_usage_2026-05-31.csv"),
        "is_aei": False, "is_mcp": True,
    },
    "AEI Conv + Micro 2026-05-31": {
        "file": str(DATA_DIR / "final_confirmed_human_usage_2026-05-31.csv"),
        "is_aei": False, "is_mcp": False,
    },
    "All 2026-05-31": {
        "file": str(DATA_DIR / "final_all_usage_2026-05-31.csv"),
        "is_aei": False, "is_mcp": False,
    },
})


def register_datasets() -> None:
    """Add the eco_2025-rebased datasets to the live DATASETS registry.
    Idempotent; only
    fills names the registry doesn't already have.

    backend/config.py gets imported under TWO module names — 'backend.config'
    (package import) and top-level 'config' (backend/ sits on sys.path;
    backend/compute.py does `from config import DATASETS`). Each instance has
    its own DATASETS dict, so update every loaded instance of the file.
    """
    for name, meta in _EXTRA_DATASETS.items():
        assert Path(meta["file"]).exists(), f"Missing data file for {name}: {meta['file']}"

    cfg_file = str(Path(backend_config.__file__).resolve())
    import importlib

    try:
        importlib.import_module("config")
    except ImportError:
        pass
    for mod in list(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if mod_file and str(Path(mod_file).resolve()) == cfg_file:
            datasets = getattr(mod, "DATASETS", None)
            if isinstance(datasets, dict):
                for name, meta in _EXTRA_DATASETS.items():
                    datasets.setdefault(name, meta)


register_datasets()

# Import backend.compute only after registration so a fresh top-level
# 'config' module instance (created by compute's own import) is covered.
from backend.compute import load_eco_raw  # noqa: E402

# ── Verb-family taxonomy (audit_verb_taxonomy/dwa_buckets.csv) ────────────

DWA_BUCKETS_FILE = DATA_DIR / "dwa_buckets.csv"

# Display labels + palette matching the deepdive_task_texture charts.
FAMILY_LABELS: dict[str, str] = {
    "produce_create": "Create / Design / Develop",
    "analyze_compute": "Analyze / Compute",
    "evaluate_inspect": "Evaluate / Inspect / Monitor",
    "info_acquisition": "Acquire Info / Research / Collect",
    "direct_manage": "Manage / Direct / Coordinate",
    "communicate_teach": "Communicate / Teach / Advise",
    "record_document": "Document / Record",
    "operate_physical": "Operate / Repair / Maintain",
}

FAMILY_SHORT: dict[str, str] = {
    "produce_create": "Create",
    "analyze_compute": "Analyze",
    "evaluate_inspect": "Evaluate",
    "info_acquisition": "Acquire",
    "direct_manage": "Manage",
    "communicate_teach": "Comm",
    "record_document": "Doc",
    "operate_physical": "Operate",
}

FAMILY_COLORS: dict[str, str] = {
    "produce_create": "#3d5a80",     # slate blue
    "analyze_compute": "#4f7c72",    # teal green
    "evaluate_inspect": "#c0592b",   # burnt orange
    "info_acquisition": "#8e6bb3",   # purple
    "direct_manage": "#2e7ea3",      # cyan blue
    "communicate_teach": "#6b8f3c",  # olive green
    "record_document": "#c09a1f",    # mustard
    "operate_physical": "#9a4f1e",   # brown
}

FAMILY_ORDER: list[str] = list(FAMILY_LABELS)

_dwa_bucket_map: Optional[dict[str, str]] = None


def dwa_bucket_map() -> dict[str, str]:
    """dwa_title (stripped) → bucket_key. Exact-title join, same as the
    deepdive_task_texture loader."""
    global _dwa_bucket_map
    if _dwa_bucket_map is None:
        df = pd.read_csv(DWA_BUCKETS_FILE, usecols=["dwa_title", "bucket_key"])
        assert not df.empty, f"Empty: {DWA_BUCKETS_FILE}"
        bad = set(df["bucket_key"]) - set(FAMILY_LABELS)
        assert not bad, f"Unknown bucket keys in {DWA_BUCKETS_FILE}: {bad}"
        _dwa_bucket_map = dict(zip(df["dwa_title"].str.strip(), df["bucket_key"]))
    return _dwa_bucket_map


def assign_family(dwa_series: pd.Series) -> pd.Series:
    """Map a dwa_title Series to bucket keys; asserts (near-)full coverage."""
    mapped = dwa_series.str.strip().map(dwa_bucket_map())
    coverage = mapped.notna().mean()
    assert coverage >= 0.99, f"DWA bucket coverage only {coverage:.1%}"
    return mapped


# ── Compact single-line labels (split two-column charts) ─────────────────
# The split charts put a y-label column on BOTH halves, so label width is
# charged to the canvas twice. At a fixed print pt the label's share of the
# canvas is width-invariant — widening the canvas doesn't buy bar room, only
# shorter labels do.
#
# Two tiers, because the charts do not all have the same room.
#
# SHORT (majors ≤ 17, GWAs ≤ 19) is for the stacked charts, whose caps came
# down when they absorbed the workers/wages figure: three number columns per
# panel have to come out of somewhere, and a shorter label costs less than a
# bar segment that can no longer carry its own percentage.
#
# MEDIUM (majors ≤ 24, GWAs ≤ 26) is for the adoption charts, which carry one
# number column instead of three and were left with ~290 px of empty gutter
# between the halves. Spending that on fuller category names is worth more
# than leaving it blank. The two tiers do mean a major reads "Computer &
# Math" on one figure and "Computer & Mathematical" on another.

MAJOR_SHORT_LABELS: dict[str, str] = {
    "Architecture and Engineering Occupations": "Architect. & Eng",
    "Arts, Design, Entertainment, Sports, and Media Occupations": "Arts & Design",
    "Building and Grounds Cleaning and Maintenance Occupations": "Building & Grnds",
    "Business and Financial Operations Occupations": "Business & Fin.",
    "Community and Social Service Occupations": "Community & Soc.",
    "Computer and Mathematical Occupations": "Computer & Math",
    "Construction and Extraction Occupations": "Construction",
    "Educational Instruction and Library Occupations": "Educ. & Library",
    "Farming, Fishing, and Forestry Occupations": "Farm & Forestry",
    "Food Preparation and Serving Related Occupations": "Food Prep",
    "Healthcare Practitioners and Technical Occupations": "Healthcare Pract.",
    "Healthcare Support Occupations": "Healthcare Supp.",
    "Installation, Maintenance, and Repair Occupations": "Install & Repair",
    "Legal Occupations": "Legal",
    "Life, Physical, and Social Science Occupations": "Life/Phys/Soc Sci",
    "Management Occupations": "Management",
    "Office and Administrative Support Occupations": "Office & Admin",
    "Personal Care and Service Occupations": "Personal Care",
    "Production Occupations": "Production",
    "Protective Service Occupations": "Protective Svc",
    "Sales and Related Occupations": "Sales and Related",
    "Transportation and Material Moving Occupations": "Transportation",
}

MAJOR_MEDIUM_LABELS: dict[str, str] = {
    "Architecture and Engineering Occupations": "Architecture & Eng",
    "Arts, Design, Entertainment, Sports, and Media Occupations": "Arts, Design & Media",
    "Building and Grounds Cleaning and Maintenance Occupations": "Building & Grounds",
    "Business and Financial Operations Occupations": "Business & Financial",
    "Community and Social Service Occupations": "Community & Social Svc",
    "Computer and Mathematical Occupations": "Computer & Mathematical",
    "Construction and Extraction Occupations": "Construction & Extract.",
    "Educational Instruction and Library Occupations": "Education & Library",
    "Farming, Fishing, and Forestry Occupations": "Farming & Forestry",
    "Food Preparation and Serving Related Occupations": "Food Prep & Serving",
    "Healthcare Practitioners and Technical Occupations": "Healthcare Practitioners",
    "Healthcare Support Occupations": "Healthcare Support",
    "Installation, Maintenance, and Repair Occupations": "Install, Maint. & Repair",
    "Legal Occupations": "Legal",
    "Life, Physical, and Social Science Occupations": "Life/Phys/Social Science",
    "Management Occupations": "Management",
    "Office and Administrative Support Occupations": "Office & Admin Support",
    "Personal Care and Service Occupations": "Personal Care & Service",
    "Production Occupations": "Production",
    "Protective Service Occupations": "Protective Service",
    "Sales and Related Occupations": "Sales and Related",
    "Transportation and Material Moving Occupations": "Transportation & Moving",
}

# Applied on top of _GWA_SHORT_LABELS below — only the entries that are
# still over 30 chars after that map runs.
_GWA_SHORT_PASS2: dict[str, str] = {
    "Establishing Interpersonal Relationships": "Building Relationships",
    "Estimating Quantifiable Characteristics": "Estimating Quantities",
    "Updating and Using Relevant Knowledge": "Updating Knowledge",
    "Making Decisions and Solving Problems": "Deciding, Solving Problems",
    "Communicating with People Outside Org": "Communicating Externally",
    "Performing Administrative Activities": "Administrative Activities",
    "Monitoring and Controlling Resources": "Controlling Resources",
    "Evaluating Compliance with Standards": "Evaluating Compliance",
    "Developing Objectives and Strategies": "Developing Strategy",
    "Communicating with Supervisors/Peers": "Communicating Internally",
    "Resolving Conflicts and Negotiating": "Resolving Conflicts",
    "Judging Qualities of Objects/People": "Judging Qualities",
    "Interpreting Information for Others": "Interpreting Information",
    "Identifying Objects/Actions/Events": "Identifying Objects/Events",
    "Guiding and Directing Subordinates": "Directing Subordinates",
    "Controlling Machines and Processes": "Controlling Machines",
    "Providing Consultation and Advice": "Consulting and Advising",
    "Performing for or with the Public": "Working with the Public",
    "Documenting/Recording Information": "Documenting Information",
    "Operating Vehicles and Equipment": "Operating Equipment",
    "Inspecting Equipment/Structures": "Inspecting Equipment",
    "Assisting and Caring for Others": "Assisting and Caring",
    # Second pass: the split two-column charts charge the label column to the
    # canvas twice, so the gutter between the halves has to clear the right
    # panel's longest label plus the left panel's number column. Dropping the
    # cap from 30 to 26 buys each half roughly a fifth more bar.
    "Coaching and Developing Others": "Coaching and Developing",
    "Monitoring Processes/Materials": "Monitoring Processes",
    "Organizing, Planning, and Prioritizing Work": "Planning and Prioritizing",
    "Planning and Prioritizing Work": "Planning and Prioritizing",
    "Performing Physical Activities": "Physical Activities",
    "Repairing Mechanical Equipment": "Repairing Equipment",
    "Scheduling Work and Activities": "Scheduling Work",
    "Analyzing Data or Information": "Analyzing Data",
    "Selling or Influencing Others": "Selling or Influencing",
    "Staffing Organizational Units": "Staffing Org Units",
    "Training and Teaching Others": "Training and Teaching",
    "Handling and Moving Objects": "Handling/Moving Objects",
}

# Third pass, keyed on what the two maps above leave behind: only the labels
# still over the 19-char cap once the number columns took their share.
_GWA_SHORT_PASS3: dict[str, str] = {
    "Assisting and Caring": "Assisting & Caring",
    "Controlling Machines": "Machines/Processes",
    "Controlling Resources": "Managing Resources",
    "Estimating Quantities": "Estimating Quantity",
    "Evaluating Compliance": "Checking Compliance",
    "Inspecting Equipment": "Inspecting Equip.",
    "Monitoring Processes": "Monitoring Process",
    "Training and Teaching": "Training & Teaching",
    "Working with Computers": "Using Computers",
    "Administrative Activities": "Admin. Activities",
    "Building Relationships": "Building Relations",
    "Coaching and Developing": "Coaching Others",
    "Communicating Externally": "Communicating Ext.",
    "Communicating Internally": "Communicating Int.",
    "Consulting and Advising": "Consulting/Advising",
    "Deciding, Solving Problems": "Deciding & Solving",
    "Directing Subordinates": "Directing Reports",
    "Documenting Information": "Documenting Info",
    "Handling/Moving Objects": "Handling Objects",
    "Identifying Objects/Events": "Identifying Objects",
    "Interpreting Information": "Interpreting Info",
    "Planning and Prioritizing": "Prioritizing Work",
    "Processing Information": "Processing Info",
    "Selling or Influencing": "Selling/Influencing",
    "Working with the Public": "Working w/ Public",
}

MAJOR_LABEL_MAX = 17
GWA_LABEL_MAX = 19
MAJOR_MEDIUM_MAX = 24
GWA_MEDIUM_MAX = 26


def short_major_label(major: str) -> str:
    """Single-line major-category label, ≤ MAJOR_LABEL_MAX chars."""
    label = MAJOR_SHORT_LABELS.get(major, major.replace(" Occupations", ""))
    assert len(label) <= MAJOR_LABEL_MAX, f"Major label too long: {label!r}"
    return label


def short_gwa_label(gwa: str) -> str:
    """Single-line GWA label, ≤ GWA_LABEL_MAX chars."""
    label = medium_gwa_label(gwa)
    label = _GWA_SHORT_PASS3.get(label, label)
    assert len(label) <= GWA_LABEL_MAX, f"GWA label too long: {label!r}"
    return label


def medium_major_label(major: str) -> str:
    """Fuller major-category label, ≤ MAJOR_MEDIUM_MAX chars, for the charts
    that have the room (see the tier note above)."""
    label = MAJOR_MEDIUM_LABELS.get(major, major.replace(" Occupations", ""))
    assert len(label) <= MAJOR_MEDIUM_MAX, f"Major label too long: {label!r}"
    return label


def medium_gwa_label(gwa: str) -> str:
    """Fuller GWA label, ≤ GWA_MEDIUM_MAX chars — the short label without the
    third abbreviation pass."""
    label = _GWA_SHORT_LABELS.get(gwa, gwa)
    label = _GWA_SHORT_PASS2.get(label, label)
    assert len(label) <= GWA_MEDIUM_MAX, f"GWA label too long: {label!r}"
    return label


# ── Physical flag ─────────────────────────────────────────────────────────

def coerce_phys_bool(s: pd.Series) -> pd.Series:
    """The physical column is stored heterogeneously (1/0/True/'True'/…)."""
    return s.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})


# The physical-makeup cuts. A unit is Non-physical under a third, Physical
# over two thirds, Mixed between.
PHYS_LOWER, PHYS_UPPER = 33.0, 67.0


def phys_tier(pct_physical: pd.Series) -> pd.Series:
    """Bucket a 0–100 physical share into the three tiers.

    The share fed in is a share of WORK TIME, not of task count — see
    `occ_phys_hours_share` and the note there for why.
    """
    return pd.cut(
        pct_physical,
        bins=[-0.1, PHYS_LOWER, PHYS_UPPER, 100.1],
        labels=["Non-physical", "Mixed", "Physical"],
    )


def occ_phys_hours_share() -> pd.Series:
    """Per occupation: the share of its WORK TIME spent on physical tasks,
    0–100, indexed by title_current.

    Time-weighted, not a count of tasks. Two occupations with the same number
    of physical tasks are not equally physical if one of them spends four
    hours a day on them and the other twenty minutes, and every other number
    in this folder is already hours.

    Weighted on `time_per_day` rather than on `hours` (= time x emp) even
    though employment cancels inside an occupation: it cancels only where
    employment is positive, and Fishing and Hunting Workers carries emp = 0
    in OEWS, which would make every one of its hours zero and land a 72%
    physical occupation in the non-physical bucket. `time_per_day` sums to 7
    for every occupation, so this is Σ time_per_day over physical tasks ÷ 7.
    """
    pairs = eco_pairs()
    t = pairs[TIME_COL].fillna(0.0)
    total = t.groupby(pairs["title_current"]).sum()
    phys = t.where(pairs["physical"], 0.0).groupby(pairs["title_current"]).sum()
    share = np.where(total > 0, phys.reindex(total.index).fillna(0.0) / total * 100.0, 0.0)
    return pd.Series(share, index=total.index, name="pct_physical_hours")


# ── ECO universe (deduped views of final_eco_2025.csv) ────────────────────

def eco_pairs(extra_cols: tuple[str, ...] = (),
              geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Full eco_2025 universe deduped to one row per (title_current,
    task_normalized), with time/emp/physical/major + extras.

    Two weights ride along, and they answer different questions:

    - `hours`      = time_per_day x emp — the labour hours per day the
                     economy spends on that (task, occupation). This is the
                     denominator of every EXPOSURE number.
    - `eco_weight` = emp — the denominator of every USAGE number. Usage asks
                     "per worker doing this work, how much AI activity is
                     there", so it deliberately does not re-weight by how
                     long the task takes.
    """
    eco = load_eco_raw()
    assert eco is not None and not eco.empty, "final_eco_2025.csv missing"
    keep = [
        "title_current", "task_normalized", "freq_mean", TIME_COL,
        "auto_aug_mean", "physical", "major_occ_category",
        EMP_COL, WAGE_COL, emp_col(geo), wage_col(geo), *extra_cols,
    ]
    keep = [c for c in dict.fromkeys(keep) if c in eco.columns]
    pairs = (
        eco[keep]
        .groupby(["title_current", "task_normalized"], sort=False)
        .first()
        .reset_index()
    )
    assert TIME_COL in pairs.columns, f"{TIME_COL} missing from final_eco_2025.csv"
    pairs["physical"] = coerce_phys_bool(pairs["physical"])
    _add_geo_cols(pairs, geo)
    pairs["hours"] = pairs[TIME_COL].fillna(0.0) * pairs["emp"]
    pairs["eco_weight"] = pairs["emp"]
    return pairs


def eco_act_rows(act_col: str, geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Eco universe deduped to (title_current, task_normalized, act_col) —
    the WA pipeline's denominator grain. Unsplit; see eco_act_split_rows for
    the /n-split version."""
    eco = load_eco_raw()
    assert eco is not None and not eco.empty
    keep = [
        "title_current", "task_normalized", act_col, "freq_mean", TIME_COL,
        "physical", "major_occ_category",
        EMP_COL, WAGE_COL, emp_col(geo), wage_col(geo),
    ]
    keep = [c for c in dict.fromkeys(keep) if c in eco.columns]
    rows = (
        eco[keep]
        .dropna(subset=[act_col])
        .groupby(["title_current", "task_normalized", act_col], sort=False)
        .first()
        .reset_index()
    )
    rows["physical"] = coerce_phys_bool(rows["physical"])
    _add_geo_cols(rows, geo)
    return rows


# ── Exposure pairs from a dataset file (eco_2025-based, is_aei=False) ─────

def dataset_pairs(dataset_name: str) -> pd.DataFrame:
    """Rated (title_current, task_normalized) pairs of an eco_2025-based
    dataset file, with the FILE's freq/auto_aug (what the dashboard numerator
    uses) and the physical flag."""
    meta = backend_config.DATASETS[dataset_name]
    assert not meta["is_aei"], f"{dataset_name} is eco_2015-based; expected eco_2025"
    df = pd.read_csv(
        meta["file"],
        usecols=["title_current", "task_normalized", "freq_mean",
                 "auto_aug_mean", "physical", "pct_normalized"],
    )
    assert not df.empty, f"Empty dataset file: {meta['file']}"
    pairs = (
        df.groupby(["title_current", "task_normalized"], sort=False)
        .first()
        .reset_index()
    )
    pairs["physical"] = coerce_phys_bool(pairs["physical"])
    return pairs


def dataset_act_rows(dataset_name: str, act_col: str) -> pd.DataFrame:
    """Rated rows of a dataset deduped at (title_current, task_normalized,
    act_col) — the WA pipeline's numerator grain."""
    meta = backend_config.DATASETS[dataset_name]
    assert not meta["is_aei"], f"{dataset_name} is eco_2015-based; expected eco_2025"
    df = pd.read_csv(
        meta["file"],
        usecols=["title_current", "task_normalized", act_col, "freq_mean",
                 "auto_aug_mean", "physical"],
    )
    rows = (
        df.dropna(subset=[act_col])
        .groupby(["title_current", "task_normalized", act_col], sort=False)
        .first()
        .reset_index()
    )
    rows["physical"] = coerce_phys_bool(rows["physical"])
    return rows


# ── Usage intensity (paper methodology: AEI-only numerator, equal debias) ─

INTENSITY_FILE = DATA_DIR / "final_aei_all_usage_eco2025_2026-05-31.csv"

_intensity_pairs_cache: dict[str, pd.DataFrame] = {}


def intensity_pairs(geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Rated (task, occ) pairs of the paper's intensity dataset (AEI Conv +
    API pooled on eco_2025, no Microsoft; 2026-05-31), debiased with the
    equal 3-source consensus GWA ratios:

        adj_pct   = pct_normalized / mean(bias_ratio over the pair's GWAs)
        intensity = adj_pct / emp                    (NaN if denominator 0)

    The denominator is employment alone, NOT freq x emp or time x emp. The
    question usage answers is "per worker doing this work, how much AI
    activity shows up" — re-weighting by how long the task takes would
    penalise slow tasks, which is a claim about how AI use scales that the
    data does not support.

    One row per (title_current, task_normalized) with: adj_pct, raw pct,
    eco_weight (= emp), intensity, major, job_zone, physical.
    """
    if geo in _intensity_pairs_cache:
        return _intensity_pairs_cache[geo]

    usecols = [
        "task_normalized", "title_current", "major_occ_category", "gwa_title",
        "pct_normalized", "auto_aug_mean", "freq_mean", "physical",
        "job_zone", EMP_COL, emp_col(geo),
    ]
    usecols = list(dict.fromkeys(usecols))
    df = pd.read_csv(INTENSITY_FILE, usecols=usecols, low_memory=False)
    assert not df.empty, f"Empty: {INTENSITY_FILE}"
    for c in dict.fromkeys(("pct_normalized", "auto_aug_mean", "freq_mean",
                            "job_zone", EMP_COL, emp_col(geo))):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ai = df[df["pct_normalized"].notna()].copy()

    bias = compute_bias_ratios(BIAS_VARIANTS["equal"])
    gwa_pairs = (
        ai.dropna(subset=["gwa_title"])
        .drop_duplicates(["task_normalized", "title_current", "gwa_title"])[
            ["task_normalized", "title_current", "gwa_title"]
        ]
        .copy()
    )
    gwa_pairs["bias"] = gwa_pairs["gwa_title"].map(bias).fillna(1.0)
    avg_bias = (
        gwa_pairs.groupby(["task_normalized", "title_current"])["bias"].mean()
        .reset_index(name="avg_bias")
    )

    pairs = ai.drop_duplicates(["task_normalized", "title_current"])[
        ["task_normalized", "title_current", "major_occ_category",
         "pct_normalized", "auto_aug_mean", "freq_mean", "physical",
         "job_zone", *dict.fromkeys((EMP_COL, emp_col(geo)))]
    ].copy()
    pairs = pairs.merge(avg_bias, on=["task_normalized", "title_current"], how="left")
    pairs["avg_bias"] = pairs["avg_bias"].fillna(1.0).replace(0.0, 1.0)
    pairs["adj_pct"] = pairs["pct_normalized"] / pairs["avg_bias"]
    _add_geo_cols(pairs, geo, wage=False)
    pairs["eco_weight"] = pairs["emp"]
    pairs["intensity"] = np.where(
        pairs["eco_weight"] > 0, pairs["adj_pct"] / pairs["eco_weight"], np.nan
    )
    pairs["physical"] = coerce_phys_bool(pairs["physical"])
    _intensity_pairs_cache[geo] = pairs
    return pairs


# ── Frame caches ──────────────────────────────────────────────────────────
# The three builders below each rebuild a ~60k-row join from CSV. The figures
# call them a handful of times; the dashboard calls them once per page view,
# where 0.3s each is the difference between a click feeling instant and not.
#
# Callers routinely tag the result in place (`rows["family"] = ...`), so every
# cached frame is handed out as a COPY. Copying 60k rows costs ~10ms against a
# ~300ms rebuild, and returning the live object would let one caller's tag
# leak into the next one's frame.

_eco_act_split_cache: dict[tuple[str, str], pd.DataFrame] = {}
_act_exposure_rows_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
_intensity_act_rows_cache: dict[tuple[str, str], pd.DataFrame] = {}


def _build_intensity_act_rows(act_col: str = "gwa_title",
                              geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Rated (task, occ, act) rows of the intensity dataset with each pair's
    adj_pct, raw pct, and eco_weight split /n across its distinct values of
    `act_col`, so per-activity sums stay a true decomposition (Σ = the pair
    totals). Carries the pair's physical flag and major for downstream cuts.

    `act_col` is "gwa_title" for the GWA adoption chart and "dwa_title" for
    the verb-family charts — a task sits in several DWAs, so without the /n
    split its usage would be counted once per DWA.
    """
    usecols = ["task_normalized", "title_current", act_col, "pct_normalized"]
    df = pd.read_csv(INTENSITY_FILE, usecols=usecols, low_memory=False)
    rows = (
        df[df["pct_normalized"].notna()]
        .dropna(subset=[act_col])
        .drop_duplicates(["task_normalized", "title_current", act_col])[
            ["task_normalized", "title_current", act_col]
        ]
        .copy()
    )
    pairs = intensity_pairs(geo=geo)
    rows = rows.merge(
        pairs[["task_normalized", "title_current", "adj_pct", "pct_normalized",
               "eco_weight", "physical", "major_occ_category"]],
        on=["task_normalized", "title_current"],
        how="inner",
    )
    n = rows.groupby(["task_normalized", "title_current"])[act_col].transform("nunique")
    rows["adj_pct_split"] = rows["adj_pct"] / n
    rows["raw_pct_split"] = rows["pct_normalized"] / n
    rows["eco_weight_split"] = rows["eco_weight"] / n
    return rows


def intensity_gwa_rows(geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """GWA-grain intensity rows (back-compat alias for the adoption chart)."""
    return intensity_act_rows("gwa_title", geo=geo)


def _build_eco_act_split_rows(act_col: str = "gwa_title",
                              geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """FULL eco_2025 universe at (task, occ, act) grain, /n-split across the
    pair's distinct activities.

    Carries both weights, split the same way:

    - `eco_weight_split` = emp / n         — usage denominator
    - `hours_split`      = (time x emp)/n  — exposure denominator

    The /n split is what keeps a task that sits in three GWAs from being
    counted three times: its hours are a third of the workday's claim in
    each, so the per-activity sums add back to the pair total (and thence to
    the whole economy) instead of inventing work.
    """
    eco = load_eco_raw()
    assert eco is not None and not eco.empty, "final_eco_2025.csv missing"
    keep = [
        "task_normalized", "title_current", act_col, "freq_mean", TIME_COL,
        "physical", "major_occ_category",
        EMP_COL, WAGE_COL, emp_col(geo), wage_col(geo),
    ]
    keep = [c for c in dict.fromkeys(keep) if c in eco.columns]
    rows = (
        eco.dropna(subset=[act_col])
        .drop_duplicates(["task_normalized", "title_current", act_col])[keep]
        .copy()
    )
    rows["physical"] = coerce_phys_bool(rows["physical"])
    _add_geo_cols(rows, geo)
    rows["eco_weight"] = rows["emp"]
    rows["hours"] = rows[TIME_COL].fillna(0.0) * rows["emp"]
    # ~88 of 18.8k pairs carry two freq_mean/time values (one normalized task
    # text mapping to two raw tasks). Collapse to one weight per pair, the way
    # eco_pairs() and the intensity numerator already do — otherwise the /n
    # split stops summing back to the pair total and the denominator is
    # built on a different grain than the numerator it divides.
    pair = ["task_normalized", "title_current"]
    for col in ("eco_weight", "hours"):
        rows[col] = rows.groupby(pair)[col].transform("first")
    n = rows.groupby(pair)[act_col].transform("nunique")
    rows["eco_weight_split"] = rows["eco_weight"] / n
    rows["hours_split"] = rows["hours"] / n
    return rows


def eco_gwa_weight_split(geo: str = DEFAULT_GEO) -> pd.Series:
    """Σ emp/n_gwas per GWA over the FULL eco_2025 universe — the
    full-economy denominator for the GWA adoption chart, /n-split so the
    GWA denominators sum to the economy total."""
    rows = eco_act_split_rows("gwa_title", geo=geo)
    return rows.groupby("gwa_title")["eco_weight_split"].sum().rename("den_full")


# ── Usage rate (ratio of sums) + median anchoring ─────────────────────────

def usage_rate(
    num_rows: pd.DataFrame,
    den_rows: pd.DataFrame,
    group_cols: list[str],
    num_col: str = "adj_pct_split",
) -> pd.Series:
    """Ratio-of-sums usage rate per group: Σ usage ÷ Σ emp.

    This is the same construction as the job-zone / GWA / occupation usage
    numbers. Groups present in the eco denominator but absent from the AI
    numerator come back as 0, not NaN — no observed usage is a real zero.
    """
    assert group_cols, "group_cols cannot be empty"
    num = num_rows.groupby(group_cols)[num_col].sum()
    den = den_rows.groupby(group_cols)["eco_weight_split"].sum()
    aligned_num = num.reindex(den.index).fillna(0.0)
    rate = aligned_num / den.replace(0.0, np.nan)
    return rate.rename("usage_rate")


def pair_level_emp(group_col: str, extra_cols: tuple[str, ...] = (),
                   geo: str = DEFAULT_GEO) -> pd.Series:
    """Σ emp per group over (task, occupation) pairs.

    The usage denominator for groupings that are occupation attributes
    (major category, job zone). An occupation's employment is counted once
    per task it has, which is the same rule the activity-level denominator
    uses — there, a pair appears once per activity and is split /n; here it
    appears exactly once, so there is nothing to split. Stating it as one
    rule is why this counts pairs rather than deduplicating to occupations.
    """
    pairs = eco_pairs(extra_cols, geo=geo)
    return pairs.groupby(group_col)["emp"].sum().rename("emp_den")


def lower_median(vals: pd.Series) -> float:
    """The lower of the two middle values (the exact median when odd).

    `median()` interpolates between the two middle values on an even-length
    series, which leaves no entry sitting at 1.00 once you divide through by
    it. Taking the lower of the two middles instead means one group always
    anchors the scale — the reader can point at it and say "that one is the
    median".
    """
    clean = vals.dropna()
    assert len(clean), "No values to anchor on"
    ordered = clean.sort_values()
    anchor = float(ordered.iloc[(len(ordered) - 1) // 2])
    if anchor <= 0:
        positive = ordered[ordered > 0]
        anchor = float(positive.iloc[(len(positive) - 1) // 2]) if len(positive) else 1.0
    return anchor


def anchor_lower_median(vals: pd.Series) -> pd.Series:
    """Rescale so the lower-middle entry reads exactly 1.00×."""
    return vals / lower_median(vals)


# ── Exposure: work time, workers and wages ────────────────────────────────
# Every exposure figure in this folder resolves to one quantity —
#
#     exposed hours = time_per_day x emp x (auto_aug / 5)
#
# summed over (task, occupation) rows. Divide by the group's total hours and
# you have the share; divide by OCC_DAY_HOURS and you have the full-time-
# equivalent headcount; multiply by the wage first and you have exposed
# wages. Nothing is allocated to anybody, so there is no split to defend:
# hours are hours.

_occ_exposure_cache: dict[tuple[str, str, str], pd.DataFrame] = {}


def occ_exposure(dataset_name: str, physical_mode: str = "all",
                 geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """One row per occupation: `p` (exposed share of work time, 0-1), emp,
    wage, major.

    `p` comes from the dashboard pipeline at occupation level with
    method="time_day". Occupation level is the only grain where the pipeline
    needs no interpretation — there is no employment weighting question
    inside a single occupation — so the group rollups below are done here
    rather than asking the pipeline for them.
    """
    assert physical_mode in {"all", "exclude", "only"}, physical_mode
    # geo belongs in the key: `p` is geo-invariant inside one occupation, but
    # the emp/wage merged on below is not.
    key = (dataset_name, physical_mode, geo)
    if key in _occ_exposure_cache:
        return _occ_exposure_cache[key]

    from backend.compute import get_group_data

    data = get_group_data({
        "selected_datasets": [dataset_name], "combine_method": "Average",
        "method": "time_day", "use_auto_aug": True,
        "physical_mode": physical_mode, "geo": geo,
        "agg_level": "occupation", "sort_by": "% Tasks Affected",
        "top_n": 9999, "search_query": "", "context_size": 3,
    })
    assert data is not None, f"No data for {dataset_name} ({physical_mode})"
    df = data["df"].rename(columns={data["group_col"]: "title_current"})
    out = df[["title_current", "pct_tasks_affected"]].copy()
    out["p"] = out["pct_tasks_affected"] / 100.0

    struct = (
        eco_pairs(geo=geo)
        .groupby("title_current")
        .agg(major_occ_category=("major_occ_category", "first"),
             emp=("emp", "first"), wage=("wage", "first"))
        .reset_index()
    )
    out = out.merge(struct, on="title_current", how="inner")
    assert len(out) > 800, f"Only {len(out)} occupations matched"
    _occ_exposure_cache[key] = out
    return out


def major_exposure(dataset_name: str, physical_mode: str = "all",
                   geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Per major: work time exposed (%), FTE workers exposed, wages exposed.

    The percentage is employment-weighted — "of all the hours worked in this
    major, this share is exposed" — which is exactly workers_exposed divided
    by total employment. Magnitudes are built per occupation and summed, never
    derived by multiplying a category percentage back out: wages would be
    wrong, because exposure and pay are correlated across occupations.
    """
    occ = occ_exposure(dataset_name, physical_mode, geo=geo)
    g = (
        occ.assign(
            hours=lambda d: d["emp"] * OCC_DAY_HOURS,
            hours_exposed=lambda d: d["p"] * d["emp"] * OCC_DAY_HOURS,
            workers_fte=lambda d: d["p"] * d["emp"],
            wages_exposed=lambda d: d["p"] * d["emp"] * d["wage"],
        )
        .groupby("major_occ_category")
        .agg(emp=("emp", "sum"), hours=("hours", "sum"),
             hours_exposed=("hours_exposed", "sum"),
             workers_fte=("workers_fte", "sum"),
             wages_exposed=("wages_exposed", "sum"),
             n_occs=("title_current", "size"))
        .reset_index()
        .rename(columns={"major_occ_category": "category"})
    )
    g["pct"] = g["hours_exposed"] / g["hours"].replace(0.0, np.nan) * 100.0
    return g


def pair_exposure_rows(
    dataset_name: str,
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """(task, occupation) rows with hours and exposed hours — the pair-grain
    counterpart of act_exposure_rows, used wherever a chart cuts the economy
    by something other than a work activity (majors, physicality, job zone).
    """
    assert physical_mode in {"all", "exclude", "only"}, physical_mode
    rows = eco_pairs(geo=geo)
    if physical_mode == "exclude":
        rows = rows[~rows["physical"]].copy()
    elif physical_mode == "only":
        rows = rows[rows["physical"]].copy()

    ai = dataset_pairs(dataset_name)[
        ["title_current", "task_normalized", "auto_aug_mean"]
    ].rename(columns={"auto_aug_mean": "ai_aug"})
    rows = rows.merge(ai, on=["title_current", "task_normalized"], how="left")
    rows["rated"] = rows["ai_aug"].notna()
    rows["exposed_frac"] = rows["ai_aug"].fillna(0.0) / 5.0
    rows["hours_exposed"] = rows["hours"] * rows["exposed_frac"]
    return rows


def _build_act_exposure_rows(
    dataset_name: str,
    act_col: str = "gwa_title",
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """(task, occ, activity) rows with /n-split hours and exposed hours.

    The auto-aug score is joined from the AI dataset; unrated pairs score 0
    (a real zero — nothing observed). Hours come from eco on BOTH sides,
    which is safe because time_per_day is a task property and is identical
    in eco and every dataset file for shared pairs — so exposed hours can
    never exceed total hours.
    """
    assert physical_mode in {"all", "exclude", "only"}, physical_mode
    rows = eco_act_split_rows(act_col, geo=geo)
    if physical_mode == "exclude":
        rows = rows[~rows["physical"]].copy()
    elif physical_mode == "only":
        rows = rows[rows["physical"]].copy()

    ai = dataset_pairs(dataset_name)[
        ["title_current", "task_normalized", "auto_aug_mean"]
    ]
    rows = rows.merge(ai, on=["title_current", "task_normalized"], how="left")
    rows["exposed_frac"] = rows["auto_aug_mean"].fillna(0.0) / 5.0
    rows["hours_exposed_split"] = rows["hours_split"] * rows["exposed_frac"]
    rows["wages_exposed_split"] = (
        rows["hours_exposed_split"] / OCC_DAY_HOURS * rows["wage"].fillna(0.0)
    )
    return rows


def act_exposure(
    dataset_name: str,
    act_col: str = "gwa_title",
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """Per activity: work time exposed (%), exposed hours/day, FTE workers.

    Because the hours are /n-split, the `hours` column sums across activities
    to the whole economy's daily hours and `hours_exposed` sums to the
    national exposed total — the same number major_exposure() reports.
    """
    rows = act_exposure_rows(dataset_name, act_col, physical_mode, geo=geo)
    g = (
        rows.groupby(act_col)
        .agg(hours=("hours_split", "sum"),
             hours_exposed=("hours_exposed_split", "sum"),
             wages_exposed=("wages_exposed_split", "sum"),
             n_rows=("hours_split", "size"))
        .reset_index()
        .rename(columns={act_col: "category"})
    )
    g["pct"] = g["hours_exposed"] / g["hours"].replace(0.0, np.nan) * 100.0
    g["workers_fte"] = g["hours_exposed"] / OCC_DAY_HOURS
    return g



# ── Cached public wrappers ────────────────────────────────────────────────
# Each returns a COPY: callers tag these frames in place (families.py adds a
# `family` column), and handing out the live cached object would let one
# caller's tag appear in the next caller's frame.


def intensity_act_rows(act_col: str = "gwa_title",
                       geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Cached `_build_intensity_act_rows`. See it for the semantics."""
    key = (act_col, geo)
    if key not in _intensity_act_rows_cache:
        _intensity_act_rows_cache[key] = _build_intensity_act_rows(act_col, geo)
    return _intensity_act_rows_cache[key].copy()


def eco_act_split_rows(act_col: str = "gwa_title",
                       geo: str = DEFAULT_GEO) -> pd.DataFrame:
    """Cached `_build_eco_act_split_rows`. See it for the semantics."""
    key = (act_col, geo)
    if key not in _eco_act_split_cache:
        _eco_act_split_cache[key] = _build_eco_act_split_rows(act_col, geo)
    return _eco_act_split_cache[key].copy()


def act_exposure_rows(
    dataset_name: str,
    act_col: str = "gwa_title",
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """Cached `_build_act_exposure_rows`. See it for the semantics."""
    key = (dataset_name, act_col, physical_mode, geo)
    if key not in _act_exposure_rows_cache:
        _act_exposure_rows_cache[key] = _build_act_exposure_rows(
            dataset_name, act_col, physical_mode, geo
        )
    return _act_exposure_rows_cache[key].copy()


# ── Standalone MCP snapshot (tool specs, not usage) ───────────────────────

def mcp_pairs() -> pd.DataFrame:
    """Rated (title_current, task_normalized) pairs of the standalone MCP
    snapshot, with auto-aug, pct, and the mean rating of the matched
    servers. `top_mcps` is a '||'-separated "Name (rating)" list capped at
    the top 5, so the count saturates — the rating mean is the usable
    depth signal, and presence in this file is the coverage signal."""
    meta = backend_config.DATASETS[MCP_DATASET]
    df = pd.read_csv(
        meta["file"],
        usecols=["title_current", "task_normalized", "auto_aug_mean",
                 "pct_normalized", "physical", "top_mcps"],
        low_memory=False,
    )
    assert not df.empty, f"Empty MCP file: {meta['file']}"
    pairs = (
        df.groupby(["title_current", "task_normalized"], sort=False)
        .first()
        .reset_index()
    )
    pairs["physical"] = coerce_phys_bool(pairs["physical"])
    pairs["mcp_rating_mean"] = pairs["top_mcps"].map(_mean_mcp_rating)
    return pairs


_RATING_RE = r"\(([0-9.]+)\)\s*$"


def _mean_mcp_rating(cell: object) -> float:
    """Mean of the parenthesised ratings in a '||'-separated top_mcps cell."""
    if not isinstance(cell, str) or not cell.strip():
        return float("nan")
    import re

    vals = [
        float(m.group(1))
        for part in cell.split("||")
        if (m := re.search(_RATING_RE, part.strip()))
    ]
    return float(np.mean(vals)) if vals else float("nan")


# ── Misc ──────────────────────────────────────────────────────────────────

def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation without a scipy dependency."""
    xr = pd.Series(x).rank()
    yr = pd.Series(y).rank()
    return float(np.corrcoef(xr, yr)[0, 1])


# ── Per-occupation structural data (count-based physical share, job zone,
#    employment, wage). The HOURS-based physical share used to classify
#    occupations for the charts is occ_phys_hours_share / phys_tier above;
#    this loader supplies job_zone / emp / wage and the count-based share.

def _load_occ_structural() -> pd.DataFrame:
    """Load eco_2025 and compute per-occupation structural data:
    pct_physical, occ_group, job_zone.

    pct_physical is computed over UNIQUE (occ, task) pairs — eco_2025 expands
    each task across its GWA/IWA/DWA classifications, and that expansion is
    not proportional between physical and non-physical tasks. Counting raw
    rows weights tasks by their WA-expansion factor and produces the wrong
    per-occ physical share. This matches the dashboard backend pipeline.
    """
    eco = pd.read_csv(DATA_DIR / "final_eco_2025.csv")
    assert "title_current" in eco.columns
    assert "physical" in eco.columns
    assert "job_zone" in eco.columns
    assert "task_normalized" in eco.columns

    # Dedup on (occ, task) before counting. job_zone, emp, wage are occ-level
    # constants so the dedup leaves them untouched.
    eco_unique = eco.drop_duplicates(["title_current", "task_normalized"])

    occ = (
        eco_unique.groupby("title_current")
        .agg(
            n_tasks=("physical", "count"),
            n_physical=("physical", "sum"),
            job_zone=("job_zone", "first"),
            emp=("emp_tot_nat_2025", "first"),
            wage=("a_med_nat_2025", "first"),
        )
        .reset_index()
    )
    occ["pct_physical"] = occ["n_physical"] / occ["n_tasks"] * 100

    occ["occ_group"] = "Mixed"
    occ.loc[occ["pct_physical"] < PHYS_LOWER, "occ_group"] = "Non-physical"
    occ.loc[occ["pct_physical"] > PHYS_UPPER, "occ_group"] = "Physical"

    return occ


# ── GWA label abbreviation, pass 1 ──────────────────────────────────────
# _GWA_SHORT_PASS2 / _GWA_SHORT_PASS3 above run on top of this.

# Single-line GWA labels for paper charts. Long O*NET names get a
# hand-picked shorter form so 41 rows fit on one page without wrapping
# (a per-row pitch tight enough for one line of 8 pt text would cause
# 2-line wraps to collide). Shortenings preserve the semantic core of
# the GWA — trimming connectives, abbreviating "Information" → "Info"
# only where needed, and using "/" for list-or-list expansions.
_GWA_SHORT_LABELS: dict[str, str] = {
    "Interpreting the Meaning of Information for Others":
        "Interpreting Information for Others",
    "Communicating with People Outside the Organization":
        "Communicating with People Outside Org",
    "Establishing and Maintaining Interpersonal Relationships":
        "Establishing Interpersonal Relationships",
    "Providing Consultation and Advice to Others":
        "Providing Consultation and Advice",
    "Organizing, Planning, and Prioritizing Work":
        "Planning and Prioritizing Work",
    "Performing for or Working Directly with the Public":
        "Performing for or with the Public",
    "Judging the Qualities of Objects, Services, or People":
        "Judging Qualities of Objects/People",
    "Evaluating Information to Determine Compliance with Standards":
        "Evaluating Compliance with Standards",
    "Resolving Conflicts and Negotiating with Others":
        "Resolving Conflicts and Negotiating",
    "Communicating with Supervisors, Peers, or Subordinates":
        "Communicating with Supervisors/Peers",
    "Estimating the Quantifiable Characteristics of Products, Events, or Information":
        "Estimating Quantifiable Characteristics",
    "Identifying Objects, Actions, and Events":
        "Identifying Objects/Actions/Events",
    "Guiding, Directing, and Motivating Subordinates":
        "Guiding and Directing Subordinates",
    "Monitoring Processes, Materials, or Surroundings":
        "Monitoring Processes/Materials",
    "Inspecting Equipment, Structures, or Materials":
        "Inspecting Equipment/Structures",
    "Repairing and Maintaining Mechanical Equipment":
        "Repairing Mechanical Equipment",
    "Repairing and Maintaining Electronic Equipment":
        "Repairing Electronic Equipment",
    "Performing General Physical Activities":
        "Performing Physical Activities",
    "Operating Vehicles, Mechanized Devices, or Equipment":
        "Operating Vehicles and Equipment",
}
