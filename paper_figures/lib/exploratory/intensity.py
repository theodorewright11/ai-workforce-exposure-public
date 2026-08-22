"""
intensity.py — the equal 3-source GWA bias prior, and the eco_2025 loader.

Observed AI-usage shares are platform-biased: Claude's mix of general work
activities is not the economy's mix, and neither is Copilot's or ChatGPT's.
This module holds each platform's published GWA share, forms a consensus
across the three, and exposes the per-GWA correction factor

    bias_ratio[gwa] = claude_share[gwa] / consensus_share[gwa]

that every usage-intensity figure divides its numerator by. `BIAS_VARIANTS`
names the weighting schemes; the paper uses `equal` (1, 1, 1).

Consumed by `lib.figure_data` (the figures) and by the dashboard's
`usage_intensity.py`, which reuses the ratios and the GWA rename verbatim so
the dashboard's Actual-Usage tab matches the paper's method exactly.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from lib.config import DATA_DIR

EMP_COL = "emp_tot_nat_2025"


# ── Source GWA distributions (hardcoded from user input) ─────────────────────
# All keyed by canonical eco_2025 GWA names. Renormalized to sum to 100 below.

CLAUDE_SHARE_RAW = {
    "Thinking Creatively": 33.7,
    "Working with Computers": 11.3,
    "Documenting/Recording Information": 8.7,
    "Analyzing Data or Information": 8.4,
    "Providing Consultation and Advice to Others": 4.0,
    "Training and Teaching Others": 3.7,
    "Making Decisions and Solving Problems": 3.7,
    "Getting Information": 3.6,
    "Inspecting Equipment, Structures, or Materials": 2.7,
    "Developing Objectives and Strategies": 2.4,
    "Judging the Qualities of Objects, Services, or People": 2.2,
    "Interpreting the Meaning of Information for Others": 2.0,
    "Guiding, Directing, and Motivating Subordinates": 2.0,
    "Communicating with Supervisors, Peers, or Subordinates": 1.8,
    "Performing for or Working Directly with the Public": 1.4,
    "Processing Information": 1.3,
    "Communicating with People Outside the Organization": 0.9,
    "Repairing and Maintaining Mechanical Equipment": 0.9,
    "Updating and Using Relevant Knowledge": 0.8,
    "Monitoring Processes, Materials, or Surroundings": 0.7,
    "Performing Administrative Activities": 0.6,
    "Assisting and Caring for Others": 0.5,
    "Selling or Influencing Others": 0.5,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 0.4,
    "Handling and Moving Objects": 0.4,
    "Identifying Objects, Actions, and Events": 0.3,
    "Monitoring and Controlling Resources": 0.2,
    "Organizing, Planning, and Prioritizing Work": 0.2,
    "Resolving Conflicts and Negotiating with Others": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Staffing Organizational Units": 0.2,
    "Controlling Machines and Processes": 0.1,
    "Scheduling Work and Activities": 0.1,
    "Establishing and Maintaining Interpersonal Relationships": 0.1,
    "Coaching and Developing Others": 0.1,
    "Performing General Physical Activities": 0.0,
    "Operating Vehicles, Mechanized Devices, or Equipment": 0.0,
}


COPILOT_SHARE_RAW = {
    "Getting Information": 24.3,
    "Communicating with People Outside the Organization": 15.4,
    "Performing for or Working Directly with the Public": 12.7,
    "Assisting and Caring for Others": 8.4,
    "Interpreting the Meaning of Information for Others": 5.2,
    "Documenting/Recording Information": 5.1,
    "Thinking Creatively": 4.4,
    "Providing Consultation and Advice to Others": 3.5,
    "Updating and Using Relevant Knowledge": 3.3,
    "Making Decisions and Solving Problems": 3.2,
    "Working with Computers": 2.7,
    "Communicating with Supervisors, Peers, or Subordinates": 2.2,
    "Analyzing Data or Information": 1.4,
    "Coaching and Developing Others": 1.3,
    "Training and Teaching Others": 1.3,
    "Judging the Qualities of Objects, Services, or People": 1.0,
    "Processing Information": 0.7,
    "Handling and Moving Objects": 0.6,
    "Selling or Influencing Others": 0.6,
    "Performing Administrative Activities": 0.5,
    "Monitoring Processes, Materials, or Surroundings": 0.5,
    "Monitoring and Controlling Resources": 0.3,
    "Performing General Physical Activities": 0.3,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 0.2,
    "Organizing, Planning, and Prioritizing Work": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Inspecting Equipment, Structures, or Materials": 0.1,
    "Developing Objectives and Strategies": 0.1,
    "Controlling Machines and Processes": 0.1,
    "Repairing and Maintaining Mechanical Equipment": 0.1,
    "Identifying Objects, Actions, and Events": 0.0,
    "Establishing and Maintaining Interpersonal Relationships": 0.0,
}
# Copilot is missing: Guiding/Directing/Motivating Subordinates, Operating Vehicles,
# Resolving Conflicts, Scheduling Work, Staffing Organizational Units.

CHATGPT_SHARE_RAW = {
    # Ambiguous (1.1%) and Suppressed (0.1%) dropped before renormalization.
    "Documenting/Recording Information": 18.4,
    "Making Decisions and Solving Problems": 14.9,
    "Thinking Creatively": 13.0,
    "Working with Computers": 10.8,
    "Interpreting the Meaning of Information for Others": 10.1,
    "Getting Information": 9.3,
    "Providing Consultation and Advice to Others": 4.4,
    "Analyzing Data or Information": 3.0,
    "Communicating with Supervisors, Peers, or Subordinates": 2.8,
    "Judging the Qualities of Objects, Services, or People": 2.0,
    "Communicating with People Outside the Organization": 1.4,
    "Estimating the Quantifiable Characteristics of Products, Events, or Information": 1.0,
    "Performing Administrative Activities": 1.0,
    "Training and Teaching Others": 0.8,
    "Selling or Influencing Others": 0.8,
    "Assisting and Caring for Others": 0.6,
    "Organizing, Planning, and Prioritizing Work": 0.6,
    "Scheduling Work and Activities": 0.5,
    "Developing Objectives and Strategies": 0.4,
    "Processing Information": 0.4,
    "Staffing Organizational Units": 0.3,
    "Updating and Using Relevant Knowledge": 0.3,
    "Resolving Conflicts and Negotiating with Others": 0.2,
    "Evaluating Information to Determine Compliance with Standards": 0.2,
    "Handling and Moving Objects": 0.2,
    "Coaching and Developing Others": 0.2,
    "Monitoring and Controlling Resources": 0.2,
    "Monitoring Processes, Materials, or Surroundings": 0.2,
    "Identifying Objects, Actions, and Events": 0.2,
    "Establishing and Maintaining Interpersonal Relationships": 0.2,
    "Inspecting Equipment, Structures, or Materials": 0.2,
    "Guiding, Directing, and Motivating Subordinates": 0.2,
    "Performing for or Working Directly with the Public": 0.1,
    "Repairing and Maintaining Mechanical Equipment": 0.1,
    "Performing General Physical Activities": 0.1,
}
# ChatGPT is missing: Operating Vehicles, Controlling Machines and Processes.


def _renorm_100(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    assert total > 0, "Empty distribution"
    return {k: v * 100.0 / total for k, v in d.items()}


CLAUDE_SHARE = _renorm_100(CLAUDE_SHARE_RAW)
COPILOT_SHARE = _renorm_100(COPILOT_SHARE_RAW)
CHATGPT_SHARE = _renorm_100(CHATGPT_SHARE_RAW)


# Canonical GWA list = union of Claude + Copilot + ChatGPT keys; Claude has all 37.
CANONICAL_GWAS = sorted(
    set(CLAUDE_SHARE) | set(COPILOT_SHARE) | set(CHATGPT_SHARE)
)


# ── Bias variants ────────────────────────────────────────────────────────────

BIAS_VARIANTS: dict[str, Optional[tuple[float, float, float]]] = {
    "no_bias": None,
    "equal": (1.0, 1.0, 1.0),
    "chatgpt_2x": (1.0, 1.0, 2.0),
    "chatgpt_5x": (1.0, 1.0, 5.0),
    "chatgpt_10x": (1.0, 1.0, 10.0),
}


def compute_bias_ratios(
    weights: Optional[tuple[float, float, float]],
) -> Optional[dict[str, float]]:
    """bias_ratio[gwa] = claude_share / consensus_share.

    Missing-source rule: a GWA not listed in a source is excluded from that
    source's contribution to the consensus for that GWA (weight drops to 0).

    Returns None when weights is None (no correction).
    """
    if weights is None:
        return None
    w_cl, w_co, w_gp = weights
    ratios: dict[str, float] = {}
    for gwa in CANONICAL_GWAS:
        parts: list[tuple[float, float]] = []
        if gwa in CLAUDE_SHARE:
            parts.append((CLAUDE_SHARE[gwa], w_cl))
        if gwa in COPILOT_SHARE:
            parts.append((COPILOT_SHARE[gwa], w_co))
        if gwa in CHATGPT_SHARE:
            parts.append((CHATGPT_SHARE[gwa], w_gp))
        w_sum = sum(w for _, w in parts)
        if w_sum <= 0 or gwa not in CLAUDE_SHARE:
            ratios[gwa] = 1.0
            continue
        consensus = sum(s * w for s, w in parts) / w_sum
        ratios[gwa] = CLAUDE_SHARE[gwa] / consensus if consensus > 0 else 1.0
    return ratios


# ── eco_2025 universe ────────────────────────────────────────────────────────

ECO_FILE = "final_eco_2025.csv"

_ECO_CACHE: Optional[pd.DataFrame] = None


def load_eco_df() -> pd.DataFrame:
    global _ECO_CACHE
    if _ECO_CACHE is not None:
        return _ECO_CACHE
    path = DATA_DIR / ECO_FILE
    assert path.exists(), f"Missing eco file: {path}"
    usecols = [
        "task_normalized", "title_current",
        "major_occ_category", "minor_occ_category", "broad_occ",
        "gwa_title", "iwa_title", "dwa_title",
        "freq_mean", EMP_COL,
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    for c in ("freq_mean", EMP_COL):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _ECO_CACHE = df
    return df


# ── GWA name map: AEI dataset (eco_2015 style) → canonical eco_2025 ──────────

AEI_GWA_RENAME = {
    "Interacting With Computers": "Working with Computers",
    "Provide Consultation and Advice to Others": "Providing Consultation and Advice to Others",
    "Communicating with Persons Outside Organization": "Communicating with People Outside the Organization",
    "Monitor Processes, Materials, or Surroundings": "Monitoring Processes, Materials, or Surroundings",
    "Inspecting Equipment, Structures, or Material": "Inspecting Equipment, Structures, or Materials",
    "Judging the Qualities of Things, Services, or People": "Judging the Qualities of Objects, Services, or People",
}
