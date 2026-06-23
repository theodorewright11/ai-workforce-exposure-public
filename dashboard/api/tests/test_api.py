"""Smoke + regression tests for the public dashboard API.

Run from the repo root:
    .venv/Scripts/python -m pytest dashboard/api/tests -q

These pin the headline numbers to the paper so a refactor that silently changes a
baseline (e.g. the agentic_confirmed eco_2025 rebasing, or the usage full-eco
denominator) fails loudly. Tolerances are loose (±0.5) — we're guarding the wiring,
not re-deriving the figures.
"""
import dashboard.api  # noqa: F401  — sys.path bootstrap

from dashboard.api.main import (
    config, exposure, ExposureRequest, exposure_children, ChildrenRequest,
    trend, TrendRequest, usage, UsageRequest,
    occupation_report, occupation_report_titles,
)


def _top(rows, key="pct_tasks_affected"):
    return sorted(rows, key=lambda r: getattr(r, key), reverse=True)[0]


def test_config_has_five_configs():
    c = config()
    keys = [x["key"] for x in c["configs"]]
    assert keys == ["all_confirmed", "human_conversation", "agentic_confirmed",
                    "all_ceiling", "agentic_ceiling"]
    assert c["default_config"] == "all_confirmed"
    assert "task" in c["usage_levels"].values()


def test_occ_exposure_matches_paper():
    r = exposure(ExposureRequest(config="all_confirmed", level="major", geo="nat", kind="occ"))
    assert len(r.rows) == 22
    top = _top(r.rows)
    assert top.category.startswith("Computer and Mathematical")
    assert abs(top.pct_tasks_affected - 70.9) < 0.6   # paper Fig 11/12


def test_wa_exposure_matches_paper():
    r = exposure(ExposureRequest(config="all_confirmed", level="gwa", geo="nat", kind="wa"))
    assert len(r.rows) == 37
    top = _top(r.rows)
    assert top.category.startswith("Working with Computers")
    assert abs(top.pct_tasks_affected - 76.0) < 0.6   # paper Fig 13


def test_agentic_confirmed_uses_eco2025_baseline():
    # paper_dataset_for() rebases agentic_confirmed onto eco_2025 → Comp&Math ~44.3
    r = exposure(ExposureRequest(config="agentic_confirmed", level="major", geo="nat", kind="occ"))
    top = _top(r.rows)
    assert top.category.startswith("Computer and Mathematical")
    assert abs(top.pct_tasks_affected - 44.3) < 0.8   # paper Fig 17


def test_drilldown_children():
    ch = exposure_children(ChildrenRequest(
        config="all_confirmed", level="major", geo="nat", kind="occ",
        parent="Computer and Mathematical Occupations"))
    cats = {r.category for r in ch.rows}
    assert "Computer Occupations" in cats


def test_trend_uses_paper_series_dates():
    t = trend(TrendRequest(config="all_confirmed", level="major", geo="nat", kind="occ"))
    dates = [d.date for d in t.data_points]
    # paper all_confirmed series drops the 2024 anchor dates
    assert dates == ["2025-03-06", "2025-08-11", "2025-11-13", "2026-02-12"]


def test_usage_intensity_matches_paper():
    u = usage(UsageRequest(level="major"))
    assert u["child_level"] == "minor"
    rows = u["rows"]
    top = rows[0]
    assert top["category"].startswith("Life, Physical")
    assert abs(top["intensity"] - 28.8) < 0.8         # paper Fig 23
    office = [r for r in rows if r["category"].startswith("Office and Admin")][0]
    assert abs(office["intensity"] - 1.0) < 0.05      # anchor


def test_occupation_report():
    titles = occupation_report_titles()
    assert len(titles["titles"]) == 923
    rep = occupation_report(title="Computer Programmers", geo="nat")
    assert rep["title"] == "Computer Programmers"
    assert abs(rep["headline"]["pct_tasks_affected"] - 78.0) < 1.0
    # per-source fields + MCP servers populated
    assert any(t.get("top_mcps") for t in rep["tasks"])
