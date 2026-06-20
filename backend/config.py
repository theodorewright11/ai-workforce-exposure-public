"""
config.py — Backend configuration (paths, dataset registry, metrics).
"""
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

ECO_BASELINE_FILE = str(DATA_DIR / "final_eco_2025.csv")
ECO_2015_FILE     = str(DATA_DIR / "final_eco_2015.csv")

CROSSWALK_PATHS = [
    str(DATA_DIR / "2010_to_2019_soc_crosswalk.csv"),
    str(ROOT.parent / "aea_dashboard_dev" / "data" / "2010_to_2019_soc_crosswalk.csv"),
    str(ROOT.parent / "automation_exposure_analysis" / "data" / "2010_to_2019_soc_crosswalk.csv"),
]

# All datasets
DATASETS = {
    # ── Snapshots ────────────────────────────────────────────────────────────
    # AEI Conversation snapshots (2010 SOC, needs crosswalk)
    "AEI Conv. v1":   {"file": str(DATA_DIR / "final_aei_v1.csv"),     "is_aei": True,  "is_mcp": False},
    "AEI Conv. v2":   {"file": str(DATA_DIR / "final_aei_v2.csv"),     "is_aei": True,  "is_mcp": False},
    "AEI Conv. v3":   {"file": str(DATA_DIR / "final_aei_v3.csv"),     "is_aei": True,  "is_mcp": False},
    "AEI Conv. v4":   {"file": str(DATA_DIR / "final_aei_v4.csv"),     "is_aei": True,  "is_mcp": False},
    "AEI Conv. v5":   {"file": str(DATA_DIR / "final_aei_v5.csv"),     "is_aei": True,  "is_mcp": False},
    # AEI API snapshots (2010 SOC, needs crosswalk)
    "AEI API v3":     {"file": str(DATA_DIR / "final_aei_api_v3.csv"), "is_aei": True,  "is_mcp": False},
    "AEI API v4":     {"file": str(DATA_DIR / "final_aei_api_v4.csv"), "is_aei": True,  "is_mcp": False},
    "AEI API v5":     {"file": str(DATA_DIR / "final_aei_api_v5.csv"), "is_aei": True,  "is_mcp": False},
    # Microsoft (2019 SOC)
    "Microsoft":      {"file": str(DATA_DIR / "final_microsoft.csv"),  "is_aei": False, "is_mcp": False},

    # ── Usage (cumulative) ───────────────────────────────────────────────────
    # AEI Both + Micro — All confirmed usage (2019 SOC)
    "AEI Both + Micro 2024-09-30": {"file": str(DATA_DIR / "final_all_confirmed_usage_2024-09-30.csv"), "is_aei": False, "is_mcp": False},
    "AEI Both + Micro 2024-12-23": {"file": str(DATA_DIR / "final_all_confirmed_usage_2024-12-23.csv"), "is_aei": False, "is_mcp": False},
    "AEI Both + Micro 2025-03-06": {"file": str(DATA_DIR / "final_all_confirmed_usage_2025-03-06.csv"), "is_aei": False, "is_mcp": False},
    "AEI Both + Micro 2025-08-11": {"file": str(DATA_DIR / "final_all_confirmed_usage_2025-08-11.csv"), "is_aei": False, "is_mcp": False},
    "AEI Both + Micro 2025-11-13": {"file": str(DATA_DIR / "final_all_confirmed_usage_2025-11-13.csv"), "is_aei": False, "is_mcp": False},
    "AEI Both + Micro 2026-02-12": {"file": str(DATA_DIR / "final_all_confirmed_usage_2026-02-12.csv"), "is_aei": False, "is_mcp": False},
    # AEI Conv + Micro — All confirmed human usage (2019 SOC)
    "AEI Conv + Micro 2024-09-30": {"file": str(DATA_DIR / "final_confirmed_human_usage_2024-09-30.csv"), "is_aei": False, "is_mcp": False},
    "AEI Conv + Micro 2024-12-23": {"file": str(DATA_DIR / "final_confirmed_human_usage_2024-12-23.csv"), "is_aei": False, "is_mcp": False},
    "AEI Conv + Micro 2025-03-06": {"file": str(DATA_DIR / "final_confirmed_human_usage_2025-03-06.csv"), "is_aei": False, "is_mcp": False},
    "AEI Conv + Micro 2025-08-11": {"file": str(DATA_DIR / "final_confirmed_human_usage_2025-08-11.csv"), "is_aei": False, "is_mcp": False},
    "AEI Conv + Micro 2025-11-13": {"file": str(DATA_DIR / "final_confirmed_human_usage_2025-11-13.csv"), "is_aei": False, "is_mcp": False},
    "AEI Conv + Micro 2026-02-12": {"file": str(DATA_DIR / "final_confirmed_human_usage_2026-02-12.csv"), "is_aei": False, "is_mcp": False},
    # AEI Both — AEI all confirmed usage (2010 SOC, needs crosswalk)
    "AEI Both 2024-12-23": {"file": str(DATA_DIR / "final_aei_all_usage_2024-12-23.csv"), "is_aei": True, "is_mcp": False},
    "AEI Both 2025-03-06": {"file": str(DATA_DIR / "final_aei_all_usage_2025-03-06.csv"), "is_aei": True, "is_mcp": False},
    "AEI Both 2025-08-11": {"file": str(DATA_DIR / "final_aei_all_usage_2025-08-11.csv"), "is_aei": True, "is_mcp": False},
    "AEI Both 2025-11-13": {"file": str(DATA_DIR / "final_aei_all_usage_2025-11-13.csv"), "is_aei": True, "is_mcp": False},
    "AEI Both 2026-02-12": {"file": str(DATA_DIR / "final_aei_all_usage_2026-02-12.csv"), "is_aei": True, "is_mcp": False},
    # AEI Conv — AEI all confirmed human use (2010 SOC, needs crosswalk)
    "AEI Conv 2024-12-23": {"file": str(DATA_DIR / "final_aei_human_usage_2024-12-23.csv"), "is_aei": True, "is_mcp": False},
    "AEI Conv 2025-03-06": {"file": str(DATA_DIR / "final_aei_human_usage_2025-03-06.csv"), "is_aei": True, "is_mcp": False},
    "AEI Conv 2025-08-11": {"file": str(DATA_DIR / "final_aei_human_usage_2025-08-11.csv"), "is_aei": True, "is_mcp": False},
    "AEI Conv 2025-11-13": {"file": str(DATA_DIR / "final_aei_human_usage_2025-11-13.csv"), "is_aei": True, "is_mcp": False},
    "AEI Conv 2026-02-12": {"file": str(DATA_DIR / "final_aei_human_usage_2026-02-12.csv"), "is_aei": True, "is_mcp": False},
    # AEI API — AEI all confirmed agentic usage (2010 SOC, needs crosswalk)
    "AEI API 2025-08-11": {"file": str(DATA_DIR / "final_aei_agentic_usage_2025-08-11.csv"), "is_aei": True, "is_mcp": False},
    "AEI API 2025-11-13": {"file": str(DATA_DIR / "final_aei_agentic_usage_2025-11-13.csv"), "is_aei": True, "is_mcp": False},
    "AEI API 2026-02-12": {"file": str(DATA_DIR / "final_aei_agentic_usage_2026-02-12.csv"), "is_aei": True, "is_mcp": False},
    # AEI API rebased to eco_2025 (2019 SOC, no crosswalk) — used by static charts
    # so agentic_confirmed pct/workers/wages live on the same baseline as the
    # other ANALYSIS_CONFIGS. Trend series stays on the eco_2015 family above.
    "AEI API 2025 2026-02-12": {"file": str(DATA_DIR / "final_aei_agentic_usage_2025_2026-02-12.csv"), "is_aei": False, "is_mcp": False},
    # AEI Both rebased to eco_2025 (2019 SOC, no crosswalk). AEI Conv + AEI API
    # pooled with task_prop normalization onto the eco_2025 task universe. Used
    # by the paper's intensity figures (main intensity_anchor_fulleco + 6
    # intensity_drivers + underadoption_gap) so they reflect AEI-only usage
    # with no Microsoft inclusion. Not in DATASET_CATEGORIES (backstage only).
    "AEI Both 2025 2026-02-12": {"file": str(DATA_DIR / "final_aei_all_usage_2025_2026-02-12.csv"), "is_aei": False, "is_mcp": False},

    # ── Agentic (cumulative) ─────────────────────────────────────────────────
    # MCP + API — All possible agentic usage (2019 SOC)
    "MCP + API 2025-04-24": {"file": str(DATA_DIR / "final_all_agentic_usage_2025-04-24.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2025-05-24": {"file": str(DATA_DIR / "final_all_agentic_usage_2025-05-24.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2025-07-23": {"file": str(DATA_DIR / "final_all_agentic_usage_2025-07-23.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2025-08-11": {"file": str(DATA_DIR / "final_all_agentic_usage_2025-08-11.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2025-11-13": {"file": str(DATA_DIR / "final_all_agentic_usage_2025-11-13.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2026-02-12": {"file": str(DATA_DIR / "final_all_agentic_usage_2026-02-12.csv"), "is_aei": False, "is_mcp": False},
    "MCP + API 2026-02-18": {"file": str(DATA_DIR / "final_all_agentic_usage_2026-02-18.csv"), "is_aei": False, "is_mcp": False},
    # MCP — standalone (2019 SOC)
    "MCP Cumul. v1":  {"file": str(DATA_DIR / "final_mcp_v1.csv"),     "is_aei": False, "is_mcp": True},
    "MCP Cumul. v2":  {"file": str(DATA_DIR / "final_mcp_v2.csv"),     "is_aei": False, "is_mcp": True},
    "MCP Cumul. v3":  {"file": str(DATA_DIR / "final_mcp_v3.csv"),     "is_aei": False, "is_mcp": True},
    "MCP Cumul. v4":  {"file": str(DATA_DIR / "final_mcp_v4.csv"),     "is_aei": False, "is_mcp": True},

    # ── All (cumulative) ─────────────────────────────────────────────────────
    # AEI Both + MCP + Microsoft — all usage potential (2019 SOC)
    "All 2024-09-30": {"file": str(DATA_DIR / "final_all_usage_2024-09-30.csv"), "is_aei": False, "is_mcp": False},
    "All 2024-12-23": {"file": str(DATA_DIR / "final_all_usage_2024-12-23.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-03-06": {"file": str(DATA_DIR / "final_all_usage_2025-03-06.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-04-24": {"file": str(DATA_DIR / "final_all_usage_2025-04-24.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-05-24": {"file": str(DATA_DIR / "final_all_usage_2025-05-24.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-07-23": {"file": str(DATA_DIR / "final_all_usage_2025-07-23.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-08-11": {"file": str(DATA_DIR / "final_all_usage_2025-08-11.csv"), "is_aei": False, "is_mcp": False},
    "All 2025-11-13": {"file": str(DATA_DIR / "final_all_usage_2025-11-13.csv"), "is_aei": False, "is_mcp": False},
    "All 2026-02-12": {"file": str(DATA_DIR / "final_all_usage_2026-02-12.csv"), "is_aei": False, "is_mcp": False},
    "All 2026-02-18": {"file": str(DATA_DIR / "final_all_usage_2026-02-18.csv"), "is_aei": False, "is_mcp": False},
}

# Eco 2015 is used internally as the baseline for AEI work-activity analysis (not user-selectable)
ECO_2015_META = {"file": ECO_2015_FILE, "is_aei": True, "is_mcp": False}

# ── Dataset categories (for UI organization) ──────────────────────────────────
# Each category contains sub_types, each sub_type maps to an ordered list of
# {name, date} entries. The frontend uses this to build category→sub_type→date
# selection UIs.
DATASET_CATEGORIES = [
    {
        "key": "snapshot",
        "label": "Snapshots",
        "sub_types": [
            {
                "key": "AEI Conv.",
                "label": "AEI Conversation",
                "datasets": [
                    {"name": "AEI Conv. v1", "date": "2024-12-23"},
                    {"name": "AEI Conv. v2", "date": "2025-03-06"},
                    {"name": "AEI Conv. v3", "date": "2025-08-11"},
                    {"name": "AEI Conv. v4", "date": "2025-11-13"},
                    {"name": "AEI Conv. v5", "date": "2026-02-12"},
                ],
            },
            {
                "key": "AEI API (Snapshot)",
                "label": "AEI API",
                "datasets": [
                    {"name": "AEI API v3", "date": "2025-08-11"},
                    {"name": "AEI API v4", "date": "2025-11-13"},
                    {"name": "AEI API v5", "date": "2026-02-12"},
                ],
            },
            {
                "key": "Microsoft",
                "label": "Microsoft",
                "datasets": [
                    {"name": "Microsoft", "date": "2024-09-30"},
                ],
            },
        ],
    },
    {
        "key": "usage",
        "label": "Usage",
        "sub_types": [
            {
                "key": "AEI Both + Micro",
                "label": "All Confirmed Usage",
                "datasets": [
                    {"name": "AEI Both + Micro 2024-09-30", "date": "2024-09-30"},
                    {"name": "AEI Both + Micro 2024-12-23", "date": "2024-12-23"},
                    {"name": "AEI Both + Micro 2025-03-06", "date": "2025-03-06"},
                    {"name": "AEI Both + Micro 2025-08-11", "date": "2025-08-11"},
                    {"name": "AEI Both + Micro 2025-11-13", "date": "2025-11-13"},
                    {"name": "AEI Both + Micro 2026-02-12", "date": "2026-02-12"},
                ],
            },
            {
                "key": "AEI Conv + Micro",
                "label": "All Confirmed Human Usage",
                "datasets": [
                    {"name": "AEI Conv + Micro 2024-09-30", "date": "2024-09-30"},
                    {"name": "AEI Conv + Micro 2024-12-23", "date": "2024-12-23"},
                    {"name": "AEI Conv + Micro 2025-03-06", "date": "2025-03-06"},
                    {"name": "AEI Conv + Micro 2025-08-11", "date": "2025-08-11"},
                    {"name": "AEI Conv + Micro 2025-11-13", "date": "2025-11-13"},
                    {"name": "AEI Conv + Micro 2026-02-12", "date": "2026-02-12"},
                ],
            },
            {
                "key": "AEI Both",
                "label": "AEI All Confirmed Usage",
                "datasets": [
                    {"name": "AEI Both 2024-12-23", "date": "2024-12-23"},
                    {"name": "AEI Both 2025-03-06", "date": "2025-03-06"},
                    {"name": "AEI Both 2025-08-11", "date": "2025-08-11"},
                    {"name": "AEI Both 2025-11-13", "date": "2025-11-13"},
                    {"name": "AEI Both 2026-02-12", "date": "2026-02-12"},
                ],
            },
            {
                "key": "AEI Conv",
                "label": "AEI All Confirmed Human Use",
                "datasets": [
                    {"name": "AEI Conv 2024-12-23", "date": "2024-12-23"},
                    {"name": "AEI Conv 2025-03-06", "date": "2025-03-06"},
                    {"name": "AEI Conv 2025-08-11", "date": "2025-08-11"},
                    {"name": "AEI Conv 2025-11-13", "date": "2025-11-13"},
                    {"name": "AEI Conv 2026-02-12", "date": "2026-02-12"},
                ],
            },
            {
                "key": "AEI API",
                "label": "AEI All Confirmed Agentic Usage",
                "datasets": [
                    {"name": "AEI API 2025-08-11", "date": "2025-08-11"},
                    {"name": "AEI API 2025-11-13", "date": "2025-11-13"},
                    {"name": "AEI API 2026-02-12", "date": "2026-02-12"},
                ],
            },
        ],
    },
    {
        "key": "agentic",
        "label": "Agentic",
        "sub_types": [
            {
                "key": "MCP + API",
                "label": "All Possible Agentic Usage",
                "datasets": [
                    {"name": "MCP + API 2025-04-24", "date": "2025-04-24"},
                    {"name": "MCP + API 2025-05-24", "date": "2025-05-24"},
                    {"name": "MCP + API 2025-07-23", "date": "2025-07-23"},
                    {"name": "MCP + API 2025-08-11", "date": "2025-08-11"},
                    {"name": "MCP + API 2025-11-13", "date": "2025-11-13"},
                    {"name": "MCP + API 2026-02-12", "date": "2026-02-12"},
                    {"name": "MCP + API 2026-02-18", "date": "2026-02-18"},
                ],
            },
            {
                "key": "MCP",
                "label": "MCP",
                "datasets": [
                    {"name": "MCP Cumul. v1", "date": "2025-04-24"},
                    {"name": "MCP Cumul. v2", "date": "2025-05-24"},
                    {"name": "MCP Cumul. v3", "date": "2025-07-23"},
                    {"name": "MCP Cumul. v4", "date": "2026-02-18"},
                ],
            },
        ],
    },
    {
        "key": "all",
        "label": "All",
        "sub_types": [
            {
                "key": "All",
                "label": "AEI Both + MCP + Microsoft",
                "datasets": [
                    {"name": "All 2024-09-30", "date": "2024-09-30"},
                    {"name": "All 2024-12-23", "date": "2024-12-23"},
                    {"name": "All 2025-03-06", "date": "2025-03-06"},
                    {"name": "All 2025-04-24", "date": "2025-04-24"},
                    {"name": "All 2025-05-24", "date": "2025-05-24"},
                    {"name": "All 2025-07-23", "date": "2025-07-23"},
                    {"name": "All 2025-08-11", "date": "2025-08-11"},
                    {"name": "All 2025-11-13", "date": "2025-11-13"},
                    {"name": "All 2026-02-12", "date": "2026-02-12"},
                    {"name": "All 2026-02-18", "date": "2026-02-18"},
                ],
            },
        ],
    },
]

# ── Dataset series for time-trend analysis ─────────────────────────────────
# Each key is a sub_type key; value is the ordered list of dataset names.
DATASET_SERIES = {
    # Snapshots
    "AEI Conv.":          ["AEI Conv. v1", "AEI Conv. v2", "AEI Conv. v3", "AEI Conv. v4", "AEI Conv. v5"],
    "AEI API (Snapshot)": ["AEI API v3", "AEI API v4", "AEI API v5"],
    "Microsoft":          ["Microsoft"],
    # Usage cumulative
    "AEI Both + Micro":   ["AEI Both + Micro 2024-09-30", "AEI Both + Micro 2024-12-23", "AEI Both + Micro 2025-03-06", "AEI Both + Micro 2025-08-11", "AEI Both + Micro 2025-11-13", "AEI Both + Micro 2026-02-12"],
    "AEI Conv + Micro":   ["AEI Conv + Micro 2024-09-30", "AEI Conv + Micro 2024-12-23", "AEI Conv + Micro 2025-03-06", "AEI Conv + Micro 2025-08-11", "AEI Conv + Micro 2025-11-13", "AEI Conv + Micro 2026-02-12"],
    "AEI Both":           ["AEI Both 2024-12-23", "AEI Both 2025-03-06", "AEI Both 2025-08-11", "AEI Both 2025-11-13", "AEI Both 2026-02-12"],
    "AEI Conv":           ["AEI Conv 2024-12-23", "AEI Conv 2025-03-06", "AEI Conv 2025-08-11", "AEI Conv 2025-11-13", "AEI Conv 2026-02-12"],
    "AEI API":            ["AEI API 2025-08-11", "AEI API 2025-11-13", "AEI API 2026-02-12"],
    # Agentic cumulative
    "MCP + API":          ["MCP + API 2025-04-24", "MCP + API 2025-05-24", "MCP + API 2025-07-23", "MCP + API 2025-08-11", "MCP + API 2025-11-13", "MCP + API 2026-02-12", "MCP + API 2026-02-18"],
    "MCP":                ["MCP Cumul. v1", "MCP Cumul. v2", "MCP Cumul. v3", "MCP Cumul. v4"],
    # All
    "All":                ["All 2024-09-30", "All 2024-12-23", "All 2025-03-06", "All 2025-04-24", "All 2025-05-24", "All 2025-07-23", "All 2025-08-11", "All 2025-11-13", "All 2026-02-12", "All 2026-02-18"],
}

AGG_LEVEL_COL = {
    "occupation": "title_current",
    "broad":      "broad_occ",
    "minor":      "minor_occ_category",
    "major":      "major_occ_category",
}

AGG_LEVEL_OPTIONS = {
    "Major Category":   "major",
    "Minor Category":   "minor",
    "Broad Occupation": "broad",
    "Occupation":       "occupation",
}

SORT_OPTIONS = ["Workers Affected", "Wages Affected", "% Tasks Affected"]
SORT_COL_MAP = {
    "Workers Affected": "workers_affected",
    "Wages Affected":   "wages_affected",
    "% Tasks Affected": "pct_tasks_affected",
}

# All supported geography codes (national + 50 states + DC + territories)
GEO_OPTIONS = {
    "nat": "National",
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "dc": "District of Columbia", "fl": "Florida", "ga": "Georgia", "hi": "Hawaii",
    "id": "Idaho", "il": "Illinois", "in": "Indiana", "ia": "Iowa",
    "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "me": "Maine",
    "md": "Maryland", "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota",
    "ms": "Mississippi", "mo": "Missouri", "mt": "Montana", "ne": "Nebraska",
    "nv": "Nevada", "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico",
    "ny": "New York", "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio",
    "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island",
    "sc": "South Carolina", "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas",
    "ut": "Utah", "vt": "Vermont", "va": "Virginia", "wa": "Washington",
    "wv": "West Virginia", "wi": "Wisconsin", "wy": "Wyoming",
    "gu": "Guam", "pr": "Puerto Rico", "vi": "U.S. Virgin Islands",
}
