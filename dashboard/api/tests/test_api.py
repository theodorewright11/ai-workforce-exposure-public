"""Smoke + regression tests for the public dashboard API.

Run from the repo root:
    .venv/Scripts/python -m pytest dashboard/api/tests -q

These pin the dashboard's headline numbers on the 2026-05-31 snapshot so a
refactor that silently changes a baseline (e.g. the agentic_confirmed eco_2025
rebasing, or the usage full-eco denominator) fails loudly. Tolerances are loose
(±0.5) — we're guarding the wiring, not re-deriving the figures.

The expected values are NOT read off a paper figure. The dashboard is
freq-weighted; the paper figures moved to work-time weighting (`time_day`), so
the two no longer produce the same quantity. Each constant below was
independently re-derived from the raw CSVs in data/ using the PRD §4 formula --
sum(AI task_comp) / sum(ECO task_comp) x 100 over unique (occupation, task)
pairs, task_comp = freq_mean x auto_aug_mean / 5 -- without going through
backend/ or dashboard/ code. Re-verify the same way if the datasets are
refreshed again; do not copy numbers from the figures.
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
    assert abs(top.pct_tasks_affected - 68.9) < 0.6   # re-derived from raw CSVs


def test_wa_exposure_matches_paper():
    r = exposure(ExposureRequest(config="all_confirmed", level="gwa", geo="nat", kind="wa"))
    assert len(r.rows) == 37
    top = _top(r.rows)
    assert top.category.startswith("Working with Computers")
    assert abs(top.pct_tasks_affected - 74.0) < 0.6   # re-derived from raw CSVs


def test_agentic_confirmed_uses_eco2025_baseline():
    # paper_dataset_for() rebases agentic_confirmed onto eco_2025 -> Comp&Math ~70.0
    r = exposure(ExposureRequest(config="agentic_confirmed", level="major", geo="nat", kind="occ"))
    top = _top(r.rows)
    assert top.category.startswith("Computer and Mathematical")
    assert abs(top.pct_tasks_affected - 70.0) < 0.8   # re-derived from raw CSVs


def test_drilldown_children():
    ch = exposure_children(ChildrenRequest(
        config="all_confirmed", level="major", geo="nat", kind="occ",
        parent="Computer and Mathematical Occupations"))
    cats = {r.category for r in ch.rows}
    assert "Computer Occupations" in cats


def test_trend_uses_paper_series_dates():
    t = trend(TrendRequest(config="all_confirmed", level="major", geo="nat", kind="occ"))
    dates = [d.date for d in t.data_points]
    # paper series window: v3 snapshot (2025-08-11) → latest
    assert dates == ["2025-08-11", "2025-11-13", "2026-02-12",
                     "2026-04-30", "2026-05-31"]


def test_usage_intensity_matches_paper():
    u = usage(UsageRequest(level="major"))
    assert u["child_level"] == "minor"
    rows = u["rows"]
    top = rows[0]
    # On the 2026-05-31 intensity dataset the top major shifted from
    # Life/Phys/Soc Science to Computer & Mathematical.
    assert top["category"].startswith("Computer and Mathematical")
    assert abs(top["intensity"] - 21.3) < 0.8         # 2026-05-31 intensity set
    office = [r for r in rows if r["category"].startswith("Office and Admin")][0]
    assert abs(office["intensity"] - 1.0) < 0.05      # anchor


def test_occupation_report():
    titles = occupation_report_titles()
    assert len(titles["titles"]) == 923
    rep = occupation_report(title="Computer Programmers", geo="nat")
    assert rep["title"] == "Computer Programmers"
    assert abs(rep["headline"]["pct_tasks_affected"] - 72.1) < 1.0  # re-derived
    # per-source fields + MCP servers populated
    assert any(t.get("top_mcps") for t in rep["tasks"])
