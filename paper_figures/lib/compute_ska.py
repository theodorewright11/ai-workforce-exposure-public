"""
compute_ska.py — O*NET Skills / Abilities / Knowledge element loader.

Reads the three O*NET element files from `data/reference/` and pivots each to
one row per (soc_code, title, element_name) carrying `importance` (Scale ID
IM, 1-5) and `level` (Scale ID LV, 0-7).

`load_ska_data()` returns all three as a single `SKAData` bundle; the focused-
set chart uses it to score how much of an occupation's skill requirements AI
already covers.
"""
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path


import pandas as pd


# This file lives at <repo_root>/paper_figures/lib/compute_ska.py; the O*NET
# SKA reference CSVs live under <repo_root>/data/reference/.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "reference"


ELEMENT_FILES: dict[str, Path] = {
    "skills":    DATA_DIR / "skills_v30.1.csv",
    "abilities": DATA_DIR / "abilities_v30.1.csv",
    "knowledge": DATA_DIR / "knowledge_v30.1.csv",
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SKAData:
    """Raw O*NET SKA data, loaded once and reused across compute calls."""
    skills: pd.DataFrame
    abilities: pd.DataFrame
    knowledge: pd.DataFrame


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_onet_file(path: Path) -> pd.DataFrame:
    """
    Load an O*NET Skills / Abilities / Knowledge file and pivot to produce
    one row per (soc_code, title, element_name) with columns:
        importance  (Scale ID = IM, 1-5)
        level       (Scale ID = LV, 0-7)
    """
    assert path.exists(), f"O*NET file not found: {path}"
    df = pd.read_csv(path, dtype=str)

    df = df.rename(columns={
        "O*NET-SOC Code": "soc_code",
        "Title":          "title",
        "Element Name":   "element_name",
        "Scale ID":       "scale_id",
        "Data Value":     "data_value",
    })

    df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
    df = df[df["scale_id"].isin(["IM", "LV"])].copy()

    pivoted = (
        df.pivot_table(
            index=["soc_code", "title", "element_name"],
            columns="scale_id",
            values="data_value",
            aggfunc="mean",
        )
        .reset_index()
    )
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"IM": "importance", "LV": "level"})

    # Drop rows missing either scale
    pivoted = pivoted.dropna(subset=["importance", "level"])
    return pivoted


def load_ska_data() -> SKAData:
    """Load all three O*NET SKA files. Call once; pass SKAData to compute_ska()."""
    return SKAData(
        skills=_load_onet_file(ELEMENT_FILES["skills"]),
        abilities=_load_onet_file(ELEMENT_FILES["abilities"]),
        knowledge=_load_onet_file(ELEMENT_FILES["knowledge"]),
    )
