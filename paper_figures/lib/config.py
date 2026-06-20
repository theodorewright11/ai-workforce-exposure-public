"""
config.py — Shared configuration for analysis scripts.

Provides path setup, dataset presets, and default configs so question
scripts don't have to repeat boilerplate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at <repo_root>/paper_figures/lib/config.py
ROOT = Path(__file__).resolve().parents[2]          # repo root
PKG_DIR = Path(__file__).resolve().parents[1]        # paper_figures/  (so `lib.*` imports resolve)
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"               # O*NET SKA + external-index reference CSVs
ANALYSIS_DIR = ROOT                                  # back-compat alias (reference files now live in REFERENCE_DIR)

# Put repo root on path so `from backend.compute import ...` works; backend dir
# itself so compute.py's `from config import ...` resolves; paper_figures dir so
# `from lib.* import ...` resolves regardless of how a script is launched.
for _p in (str(ROOT), str(BACKEND_DIR), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Dataset presets ───────────────────────────────────────────────────────────

# Standard occupation-level analysis: cumulative AEI + latest MCP + Microsoft
ALL_DATASETS: list[str] = [
    "AEI Cumul. (Both) v4",
    "MCP Cumul. v4",
    "Microsoft",
]

# AEI family only (for WA analysis — uses O*NET 2015 baseline)
AEI_DATASETS: list[str] = ["AEI Cumul. (Both) v4"]

# MCP + Microsoft (for WA analysis — uses O*NET 2025 baseline)
MCP_MS_DATASETS: list[str] = ["MCP Cumul. v4", "Microsoft"]

# WA analysis presets (same as above, explicit aliases for clarity)
WA_AEI_DATASETS: list[str] = AEI_DATASETS
WA_MCP_MS_DATASETS: list[str] = MCP_MS_DATASETS


# ── Default configs ───────────────────────────────────────────────────────────

DEFAULT_OCC_CONFIG: dict[str, Any] = {
    "selected_datasets": ALL_DATASETS,
    "combine_method": "Average",
    "method": "freq",
    "use_auto_aug": True,
    "physical_mode": "all",
    "geo": "nat",
    "agg_level": "major",
    "sort_by": "Workers Affected",
    "top_n": 30,
    "search_query": "",
    "context_size": 3,
}

DEFAULT_WA_AEI_CONFIG: dict[str, Any] = {
    "selected_datasets": WA_AEI_DATASETS,
    "combine_method": "Average",
    "method": "freq",
    "use_auto_aug": True,
    "physical_mode": "all",
    "geo": "nat",
    "agg_level": "dwa",
    "sort_by": "Workers Affected",
    "top_n": 30,
    "search_query": "",
    "context_size": 3,
}

DEFAULT_WA_MCP_MS_CONFIG: dict[str, Any] = {
    "selected_datasets": WA_MCP_MS_DATASETS,
    "combine_method": "Average",
    "method": "freq",
    "use_auto_aug": True,
    "physical_mode": "all",
    "geo": "nat",
    "agg_level": "dwa",
    "sort_by": "Workers Affected",
    "top_n": 30,
    "search_query": "",
    "context_size": 3,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_config(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Create a config dict from a base with overrides.

    Example:
        cfg = make_config(DEFAULT_OCC_CONFIG, geo="ut", agg_level="occupation", top_n=20)
    """
    cfg = base.copy()
    cfg.update(overrides)
    return cfg


def run_occ_query(config: dict[str, Any]) -> tuple[Any, str] | None:
    """Run get_group_data and return (DataFrame with 'category' column, group_col).

    Handles the column rename from the internal agg column name to 'category'
    so question scripts don't need to know the internal column names.

    Returns None if no data is available.
    """
    from backend.compute import get_group_data
    import pandas as pd

    data = get_group_data(config)
    if data is None:
        return None
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    df = df.rename(columns={group_col: "category"})
    return df, group_col


def ensure_results_dir(question_dir: Path) -> Path:
    """Create and return the results/ directory for a question folder."""
    results = question_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(exist_ok=True)
    return results


# ── Five primary analysis configs ─────────────────────────────────────────────
# Single pre-combined datasets — no combine_method needed.
# All use method="freq" (time-weighted), use_auto_aug=True, geo="nat".
# These are the canonical configs for job_exposure and subsequent analyses.
ANALYSIS_CONFIGS: dict[str, str] = {
    "all_ceiling":        "All 2026-02-18",               # AEI Both + MCP + Microsoft — ceiling
    "human_conversation": "AEI Conv + Micro 2026-02-12",  # confirmed human conversation usage
    "agentic_confirmed":  "AEI API 2026-02-12",           # confirmed agentic tool-use (AEI API only, natural eco_2015 / 2010 SOC file matching the trend series below). Paper static charts that need an eco_2025-baselined comparison use the override in analysis/paper/paper_config.py (PAPER_CONFIG_DATASET_OVERRIDES).
    "all_confirmed":      "AEI Both + Micro 2026-02-12",  # all confirmed usage (conv + API + Microsoft)
    "agentic_ceiling":    "MCP + API 2026-02-18",         # agentic ceiling (most recent)
}

# Config labels for charts/reports
ANALYSIS_CONFIG_LABELS: dict[str, str] = {
    "all_ceiling":        "All Sources (Ceiling)",
    "human_conversation": "Conversational Confirmed",
    "agentic_confirmed":  "Agentic Confirmed",
    "all_confirmed":      "All Confirmed",
    "agentic_ceiling":    "Agentic Ceiling",
}

# Full time series for each config (for trend analysis)
ANALYSIS_CONFIG_SERIES: dict[str, list[str]] = {
    "all_ceiling": [
        "All 2025-03-06", "All 2025-04-24",
        "All 2025-05-24", "All 2025-07-23", "All 2025-08-11", "All 2025-11-13",
        "All 2026-02-12", "All 2026-02-18",
    ],
    "human_conversation": [
        "AEI Conv + Micro 2025-03-06", "AEI Conv + Micro 2025-08-11",
        "AEI Conv + Micro 2025-11-13", "AEI Conv + Micro 2026-02-12",
    ],
    "agentic_confirmed": [
        "AEI API 2025-08-11", "AEI API 2025-11-13", "AEI API 2026-02-12",
    ],
    "all_confirmed": [
        "AEI Both + Micro 2025-03-06", "AEI Both + Micro 2025-08-11",
        "AEI Both + Micro 2025-11-13", "AEI Both + Micro 2026-02-12",
    ],
    "agentic_ceiling": [
        "MCP + API 2025-04-24", "MCP + API 2025-05-24", "MCP + API 2025-07-23",
        "MCP + API 2025-08-11", "MCP + API 2025-11-13", "MCP + API 2026-02-12",
        "MCP + API 2026-02-18",
    ],
}

# Occupations of interest (matched against title_current in eco_2025)
OCCS_OF_INTEREST: list[str] = [
    # High-profile / high-employment
    "Registered Nurses",
    "Software Developers",
    "General and Operations Managers",
    "Cashiers",
    "Customer Service Representatives",
    "Retail Salespersons",
    "Heavy and Tractor-Trailer Truck Drivers",
    "Elementary School Teachers, Except Special Education",
    "Waiters and Waitresses",
    "Janitors and Cleaners, Except Maids and Housekeeping Cleaners",
    "Accountants and Auditors",
    "Secretaries and Administrative Assistants, Except Legal, Medical, and Executive",
    # AI-controversial / interesting
    "Lawyers",
    "Physicians, All Other",
    "Financial Analysts",
    "Graphic Designers",
    "Technical Writers",
    "Web Developers",
    "Paralegals and Legal Assistants",
    "Data Scientists",
    "Human Resources Specialists",
    "Market Research Analysts and Marketing Specialists",
    "Editors",
    "Interpreters and Translators",
    # Utah-relevant
    "Computer Systems Analysts",
    "Medical and Health Services Managers",
    "Construction Laborers",
    "Sales Representatives, Wholesale and Manufacturing, Except Technical and Scientific Products",
    "Network and Computer Systems Administrators",
]


def get_pct_tasks_affected(
    dataset_name: str,
    method: str = "freq",
    use_auto_aug: bool = True,
) -> "pd.Series":
    """
    Run the backend compute pipeline for a single dataset and return
    pct_tasks_affected as a Series keyed by title_current (occupation name).

    Parameters
    ----------
    dataset_name : exact key from backend config, e.g. "All 2026-02-18"
    method       : "freq" (time-weighted) or "imp" (value-weighted)
    use_auto_aug : whether to apply the auto-aug multiplier

    Returns
    -------
    pd.Series  index=title_current, values=pct_tasks_affected (0-100)
    """
    import pandas as pd
    from backend.compute import get_group_data

    config: dict[str, Any] = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": method,
        "use_auto_aug": use_auto_aug,
        "physical_mode": "all",
        "geo": "nat",
        "agg_level": "occupation",
        "sort_by": "% Tasks Affected",
        "top_n": 9999,
        "search_query": "",
        "context_size": 3,
    }
    data = get_group_data(config)
    assert data is not None, f"get_pct_tasks_affected: no data returned for '{dataset_name}'"
    df: pd.DataFrame = data["df"]
    group_col: str = data["group_col"]
    return df.set_index(group_col)["pct_tasks_affected"]
