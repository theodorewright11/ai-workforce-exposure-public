"""Regression tests for the public dashboard API (v2).

Run from the repo root:
    .venv/Scripts/python -m pytest dashboard/api/tests -q

The governing invariant of v2 is **the dashboard shows the same number as the
paper figure**, so most of these assert exactly that: they read the committed
per-chart CSV under `paper_figures/results/` and check the API agrees.

That is a stronger guard than pinning literals. A pinned constant only catches
a change to the dashboard; comparing against the figure output catches the
dashboard and the figures drifting apart, which is the failure v1 actually had
(v1 was freq-weighted while the figures moved to work time, so the two stopped
measuring the same quantity and nothing failed).

The figure CSVs are stored at 2-4 decimal places, hence the 0.05 tolerances.
"""
import math
from pathlib import Path

import pandas as pd
import pytest

import dashboard.api  # noqa: F401  — sys.path bootstrap

from fastapi.testclient import TestClient

from dashboard.api.main import app
from dashboard.api.economy import get_economy
from dashboard.api.occupation import get_occupation_card
from lib import families as fam_lib
from lib import figure_data
from lib.focused import focused_set

RESULTS = Path(__file__).resolve().parents[3] / "paper_figures" / "results"

client = TestClient(app)


def _figure_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run paper_figures/run_main_figures.py")
    return pd.read_csv(path)


# ── The dashboard/figure invariant ──────────────────────────────────────────

def test_headline_matches_trend_figure():
    """37.9% of the U.S. workday, straight off trend_phys.csv."""
    ref = _figure_csv("trend_phys.csv")
    expected = float(
        ref[(ref.series == "confirmed_all") & (ref.date == "2026-05-31")]
        .pct_tasks_affected.iloc[0]
    )
    got = get_economy("nat")["trend"]["headline_pct"]
    assert abs(got - expected) < 0.05, f"headline {got} vs figure {expected}"


def test_trend_series_matches_figure():
    """Every point of all three series, not just the endpoint."""
    ref = _figure_csv("trend_phys.csv")
    key = {"all": "confirmed_all", "exclude": "confirmed_nonphys", "only": "confirmed_phys"}
    for series in get_economy("nat")["trend"]["series"]:
        name = key[series["key"]]
        for point in series["points"]:
            row = ref[(ref.series == name) & (ref.date == point["date"])]
            assert len(row) == 1, f"no figure row for {name} {point['date']}"
            expected = float(row.pct_tasks_affected.iloc[0])
            assert abs(point["pct"] - expected) < 0.05, (
                f"{name} {point['date']}: {point['pct']} vs figure {expected}"
            )


def test_verb_families_match_figure():
    ref = _figure_csv("verb_family_all_confirmed.csv").set_index("family")
    for row in get_economy("nat")["families"]:
        exp_pct = float(ref.loc[row["family"], "pct_work_time_exposed"])
        exp_use = float(ref.loc[row["family"], "usage_x"])
        assert abs(row["pct_exposed"] - exp_pct) < 0.05, row["family"]
        assert abs(row["usage_x"] - exp_use) < 0.05, row["family"]


def test_majors_match_figure():
    ref = _figure_csv("major_categories_stacked.csv").set_index("major_occ_category")
    for row in get_economy("nat")["majors"]:
        expected = float(ref.loc[row["category"], "pct_exposed"])
        assert abs(row["pct_exposed"] - expected) < 0.05, row["category"]


def test_focused_set_matches_figure():
    """The 31 — membership and values, rebuilt without the SKA loader."""
    ref = _figure_csv("focused_set_usage.csv")
    got = focused_set()
    assert set(got.title_current) == set(ref.title_current)
    assert len(got) == len(ref) == 31
    merged = got.merge(ref, on="title_current", suffixes=("", "_ref"))
    assert (merged["pct"] - merged["pct_ref"]).abs().max() < 0.05


# ── Structural invariants ───────────────────────────────────────────────────

def test_exposure_and_usage_are_always_paired():
    """PRD §2.3 — a rated row carries both, by construction.

    If this ever fails the data has changed shape, and the whole card design
    (which never shows an exposure number alone) needs revisiting.
    """
    df = figure_data.dataset_pairs(figure_data.PRIMARY_DATASET)
    exposed = df["auto_aug_mean"].fillna(0) > 0
    usage = df["pct_normalized"].fillna(0) > 0
    assert int((exposed & ~usage).sum()) == 0, "exposure without usage"


def test_verb_family_taxonomy_covers_everything():
    eco = figure_data.load_eco_raw()
    mapped = figure_data.assign_family(eco["dwa_title"].dropna())
    assert mapped.notna().mean() > 0.999


def test_state_exposure_is_employment_weighted():
    """Geography has to actually move the number, or the states block is a lie."""
    st = figure_data.state_exposure()
    assert len(st) >= 50
    assert st["pct"].max() - st["pct"].min() > 5.0, "states barely differ — check emp col"
    # A single occupation is geo-invariant; only the mix moves.
    nat = figure_data.occ_exposure(figure_data.PRIMARY_DATASET, geo="nat").set_index("title_current")
    ut = figure_data.occ_exposure(figure_data.PRIMARY_DATASET, geo="ut").set_index("title_current")
    common = nat.index.intersection(ut.index)[:50]
    assert (nat.loc[common, "p"] - ut.loc[common, "p"]).abs().max() < 1e-9


def test_family_usage_anchor_is_per_grain():
    """Economy-wide families anchor on the lower-middle family = 1.00x."""
    fam = fam_lib.family_rows(figure_data.PRIMARY_DATASET)
    assert math.isclose(float(fam["usage_x"].sort_values().iloc[(len(fam) - 1) // 2]), 1.0, abs_tol=1e-6)


# ── Endpoints ───────────────────────────────────────────────────────────────

def test_health_and_config():
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert len(cfg["configs"]) == 5
    assert cfg["default_config"] == "all_confirmed"


def test_economy_endpoint_has_six_blocks():
    body = client.get("/api/economy").json()
    for block in ("trend", "families", "majors", "gwas", "focused", "states"):
        assert block in body, block
    assert len(body["families"]) == 8
    assert body["focused"]["count"] == 31
    assert len(body["states"]["top"]) == 10


def test_families_endpoint_and_drilldown():
    body = client.get("/api/families").json()
    assert len(body["rows"]) == 8
    for row in body["rows"]:
        # the complement is always present, and the pair always adds to 100
        assert abs(row["pct_exposed"] + row["pct_unexposed"] - 100.0) < 0.05
    drill = client.get("/api/families", params={"parent": "evaluate_inspect"}).json()
    assert len(drill["rows"]) > 50
    assert drill["parent"] == "evaluate_inspect"


def test_occupation_card_shape():
    card = get_occupation_card("Computer Programmers")
    assert card is not None
    head = card["headline"]
    assert abs(head["pct_exposed"] + head["pct_unexposed"] - 100.0) < 0.05
    assert head["total_occupations"] == 923
    assert card["families"], "no verb families on the card"
    # Every family row shown carries both signals.
    for row in card["families"]:
        assert "pct_exposed" in row and "usage_x" in row
    # and its tasks resolve
    assert any(card["tasks"].get(f["family"]) for f in card["families"])


def test_occupation_card_unknown_title():
    assert get_occupation_card("Not A Real Job") is None


def test_wa_task_list_survives_the_ska_deletion():
    body = client.post("/api/wa-tasks", json={"level": "dwa", "name": "Evaluate student work."}).json()
    assert len(body["tasks"]) > 0
    assert "auto" in body["tasks"][0]


def test_explore_exposure_still_works():
    r = client.post("/api/exposure", json={
        "config": "all_confirmed", "level": "major", "geo": "nat", "kind": "occ",
    }).json()
    assert len(r["rows"]) == 22
