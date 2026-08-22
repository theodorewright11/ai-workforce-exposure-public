"""Tests for the layer the dashboard and the figures share.

Two things are being defended here, and they are the whole premise of the v2
dashboard:

1. **Adding `geo` moved no national number.** Every figure calls these
   helpers without a geography, so the default path has to be byte-for-byte
   what it was before geography existed.
2. **The dashboard and the figure render the same quantity.** Not "agree to
   two decimals today" — literally one implementation, asserted against the
   committed figure CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from lib import families, figure_data  # noqa: E402

RESULTS = HERE.parent / "results"


@pytest.fixture(scope="module")
def dataset() -> str:
    figure_data.register_datasets()
    return figure_data.PRIMARY_DATASET


# ── Geography ─────────────────────────────────────────────────────────────

def test_geo_resolvers_name_real_columns():
    assert figure_data.emp_col() == "emp_tot_nat_2025"
    assert figure_data.emp_col("ut") == "emp_tot_ut_2025"
    assert figure_data.wage_col("ia") == "a_med_ia_2025"


def test_unknown_geography_raises_rather_than_zeroing():
    # A typo'd geography used to be indistinguishable from an all-NaN
    # employment column, which sums to zero and quietly produces a
    # plausible-looking wrong answer.
    with pytest.raises(AssertionError, match="Unknown geography"):
        figure_data.eco_pairs(geo="zz")


def test_default_geo_is_national(dataset):
    assert figure_data.major_exposure(dataset).equals(
        figure_data.major_exposure(dataset, geo="nat")
    )


def test_national_totals_match_the_committed_trend_figure(dataset):
    g = figure_data.major_exposure(dataset)
    economy = g["hours_exposed"].sum() / g["hours"].sum() * 100.0
    # trend_phys.csv, confirmed_all @ 2026-05-31
    assert economy == pytest.approx(37.90, abs=0.05)


def test_state_reweights_the_occupational_mix(dataset):
    """A state number is not the national one relabelled — group exposure is
    employment-weighted, so a state's job mix moves it."""
    def economy(geo: str) -> float:
        g = figure_data.major_exposure(dataset, geo=geo)
        return g["hours_exposed"].sum() / g["hours"].sum() * 100.0

    ny, wv = economy("ny"), economy("wv")
    assert ny > economy("nat") > wv, (ny, wv)
    assert ny - wv > 2.0, "states should spread by more than rounding"


def test_state_employment_is_smaller_than_national(dataset):
    nat = figure_data.occ_exposure(dataset)["emp"].sum()
    ut = figure_data.occ_exposure(dataset, geo="ut")["emp"].sum()
    assert 0 < ut < nat / 10


def test_occupation_exposure_is_geo_invariant(dataset):
    """Inside one occupation there is no employment weighting question, so
    the exposed share of its work time is the same everywhere. Only the
    headcount and payroll behind it move."""
    nat = figure_data.occ_exposure(dataset).set_index("title_current")["p"]
    ut = figure_data.occ_exposure(dataset, geo="ut").set_index("title_current")["p"]
    shared = nat.index.intersection(ut.index)
    assert len(shared) > 800
    assert (nat[shared] - ut[shared]).abs().max() < 1e-9


def test_geo_is_in_the_cache_key(dataset):
    """Caching on (dataset, physical_mode) alone would serve one state's
    employment for another's."""
    figure_data.occ_exposure(dataset, geo="ut")
    figure_data.occ_exposure(dataset, geo="wv")
    ut = figure_data.occ_exposure(dataset, geo="ut")["emp"].sum()
    wv = figure_data.occ_exposure(dataset, geo="wv")["emp"].sum()
    assert ut != wv


# ── Verb families ─────────────────────────────────────────────────────────

def test_family_rows_reproduce_the_paper_figure(dataset):
    """The dashboard's family numbers ARE the figure's family numbers."""
    csv = RESULTS / "verb_family_all_confirmed.csv"
    if not csv.exists():
        pytest.skip("run_main_figures.py has not been run in this checkout")
    import pandas as pd

    want = pd.read_csv(csv).set_index("family")
    got = families.family_rows(dataset).set_index("family")
    assert set(got.index) == set(want.index)
    for fam in want.index:
        assert got.loc[fam, "pct_exposed"] == pytest.approx(
            want.loc[fam, "pct_work_time_exposed"], abs=0.01), fam
        assert got.loc[fam, "usage_x"] == pytest.approx(
            want.loc[fam, "usage_x"], abs=0.01), fam


def test_every_family_row_carries_both_signals(dataset):
    """PRD §2.3 — no exposure number is ever rendered without its usage
    number, at any grain."""
    for frame in (
        families.family_rows(dataset),
        families.occupation_family_rows("Computer Programmers", dataset),
        families.family_dwa_rows("analyze_compute", dataset),
    ):
        assert not frame["pct_exposed"].isna().any()
        assert not frame["usage_x"].isna().any()


def test_exposure_and_complement_sum_to_100(dataset):
    rows = families.family_rows(dataset)
    assert (rows["pct_exposed"] + rows["pct_unexposed"]).sub(100.0).abs().max() < 1e-9


def test_taxonomy_covers_every_dwa(dataset):
    rows = families.dwa_task_rows(dataset)
    assert rows["family"].notna().all()
    assert set(rows["family"]) <= set(figure_data.FAMILY_LABELS)


def test_family_hours_decompose_the_economy(dataset):
    """The /n split is what makes this true: a task in three DWAs must not
    be counted three times."""
    fam = families.family_rows(dataset)["hours"].sum()
    act = figure_data.act_exposure(dataset, "dwa_title")["hours"].sum()
    assert fam == pytest.approx(act, rel=1e-9)


def test_occupation_usage_reanchors_within_the_occupation(dataset):
    """Ratios are not comparable across aggregation grains. Anchoring an
    occupation's families on the economy-wide anchor put Computer
    Programmers in the hundreds; the card anchors on the job's own
    lower-middle family instead."""
    rows = families.occupation_family_rows("Computer Programmers", dataset)
    assert rows["usage_x"].max() < 50.0
    assert rows["usage_x"].eq(1.0).any(), "one family must sit exactly at the anchor"


def test_usage_share_is_bounded_and_sums_to_100(dataset):
    for title in ("Computer Programmers", "Registered Nurses",
                  "Paralegals and Legal Assistants"):
        rows = families.occupation_family_rows(title, dataset)
        assert rows["usage_share"].between(0.0, 100.0).all(), title
        assert rows["usage_share"].sum() == pytest.approx(100.0, abs=0.01), title


def test_occupation_family_tasks_name_their_activities(dataset):
    tasks = families.occupation_family_tasks("Computer Programmers", dataset)
    assert not tasks.empty
    assert tasks["dwas"].map(len).gt(0).all()
    assert tasks["usage_share"].sum() == pytest.approx(100.0, abs=0.01)


def test_families_present_per_occupation_is_plausible(dataset):
    """881 of 923 occupations carry rated work; the median job shows about
    five of the eight families. A change that drops this to one or pushes it
    to eight everywhere means the join broke."""
    n = len(families.occupation_family_rows("Registered Nurses", dataset))
    assert 3 <= n <= 8
