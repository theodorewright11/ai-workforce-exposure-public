"""Tests for the paper-figure helpers.

Run from the repo root:
    python -m pytest paper_figures/tests/

Covers the pieces where a silent regression would change published numbers
rather than just the layout: the hours weighting and the reconciliation
identities it buys, the usage ratio-of-sums construction, the median anchor,
the /n activity splits, the non-physical filter, and the label caps the
split charts depend on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]     # paper_figures/  (for `lib.*`)
ROOT = Path(__file__).resolve().parents[2]        # repo root       (for `backend.*`)
for _p in (str(ROOT), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib import figure_data  # noqa: E402


# ── anchor_lower_median ───────────────────────────────────────────────────

def test_anchor_puts_exactly_one_entry_at_one_when_even():
    """An even-length series has no natural median entry — the lower middle
    is chosen so a category can be pointed at as 'the median one'."""
    vals = pd.Series([0.5, 1.0, 2.0, 4.0], index=list("abcd"))
    out = figure_data.anchor_lower_median(vals)
    assert (out == 1.0).sum() == 1
    assert out["b"] == pytest.approx(1.0)      # 2nd of 4 = lower middle
    assert out["d"] == pytest.approx(4.0)


def test_anchor_uses_true_median_when_odd():
    vals = pd.Series([1.0, 3.0, 9.0])
    out = figure_data.anchor_lower_median(vals)
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(3.0)


def test_anchor_falls_back_past_zero_median():
    """Zero-heavy sets would divide by zero on a plain median."""
    vals = pd.Series([0.0, 0.0, 0.0, 2.0, 6.0])
    out = figure_data.anchor_lower_median(vals)
    assert np.isfinite(out).all()
    assert out.iloc[3] == pytest.approx(1.0)   # lower middle of the positives


def test_anchor_rejects_empty():
    with pytest.raises(AssertionError):
        figure_data.anchor_lower_median(pd.Series(dtype=float))


def test_lower_median_exposes_the_anchor_value():
    """The heatmaps need the anchor itself, not the rescaled series, so both
    the economy row and the per-major cells divide by the same number."""
    vals = pd.Series([0.5, 1.0, 2.0, 4.0])
    assert figure_data.lower_median(vals) == pytest.approx(1.0)
    assert np.allclose(vals / figure_data.lower_median(vals),
                       figure_data.anchor_lower_median(vals))


# ── usage_rate ────────────────────────────────────────────────────────────

def _rate_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    num = pd.DataFrame({
        "grp": ["a", "a", "b"],
        "adj_pct_split": [1.0, 3.0, 5.0],
    })
    den = pd.DataFrame({
        "grp": ["a", "a", "b", "c"],
        "eco_weight_split": [2.0, 2.0, 10.0, 4.0],
    })
    return num, den


def test_usage_rate_is_ratio_of_sums_not_mean_of_ratios():
    """Σ4 ÷ Σ4 = 1.0 for group a. Averaging the two row ratios would give
    (0.5 + 1.5)/2 = 1.0 here only by luck — the assertion below on group b
    is what separates the two."""
    num, den = _rate_frames()
    rate = figure_data.usage_rate(num, den, ["grp"])
    assert rate["a"] == pytest.approx(4.0 / 4.0)
    assert rate["b"] == pytest.approx(5.0 / 10.0)


def test_usage_rate_zero_fills_groups_with_no_observed_usage():
    """A group present in the eco denominator but absent from the AI
    numerator has genuinely zero observed usage, not missing data."""
    num, den = _rate_frames()
    rate = figure_data.usage_rate(num, den, ["grp"])
    assert rate["c"] == pytest.approx(0.0)


def test_usage_rate_requires_group_cols():
    num, den = _rate_frames()
    with pytest.raises(AssertionError):
        figure_data.usage_rate(num, den, [])


# ── MCP rating parsing ────────────────────────────────────────────────────

def test_mean_mcp_rating_parses_pipe_separated_cell():
    cell = "Memory MCP (5.0) || Anbani MCP (4.0) || Inception (3.0)"
    assert figure_data._mean_mcp_rating(cell) == pytest.approx(4.0)


def test_mean_mcp_rating_handles_missing_and_unrated():
    assert np.isnan(figure_data._mean_mcp_rating(None))
    assert np.isnan(figure_data._mean_mcp_rating(""))
    assert np.isnan(figure_data._mean_mcp_rating("Some Server With No Rating"))


# ── Label caps (the split charts budget canvas width on these) ────────────

def test_every_major_has_a_short_label_within_the_cap():
    rows = figure_data.eco_act_split_rows("gwa_title")
    majors = rows["major_occ_category"].dropna().unique()
    assert len(majors) == 22
    for major in majors:
        assert len(figure_data.short_major_label(major)) <= figure_data.MAJOR_LABEL_MAX


def test_every_gwa_has_a_short_label_within_the_cap():
    rows = figure_data.eco_act_split_rows("gwa_title")
    gwas = rows["gwa_title"].dropna().unique()
    assert len(gwas) >= 35
    for gwa in gwas:
        assert len(figure_data.short_gwa_label(gwa)) <= figure_data.GWA_LABEL_MAX


def test_medium_labels_stay_within_their_own_cap_and_are_never_shorter():
    """The adoption charts carry one number column instead of three, so they
    use a fuller label tier. It has to clear its own cap and it has to
    actually be fuller, or the tier is buying nothing."""
    rows = figure_data.eco_act_split_rows("gwa_title")
    majors = rows["major_occ_category"].dropna().unique()
    gwas = rows["gwa_title"].dropna().unique()
    for major in majors:
        med, short = (figure_data.medium_major_label(major),
                      figure_data.short_major_label(major))
        assert len(med) <= figure_data.MAJOR_MEDIUM_MAX
        assert len(med) >= len(short)
    for gwa in gwas:
        med, short = figure_data.medium_gwa_label(gwa), figure_data.short_gwa_label(gwa)
        assert len(med) <= figure_data.GWA_MEDIUM_MAX
        assert len(med) >= len(short)
    assert any(len(figure_data.medium_major_label(m)) > len(figure_data.short_major_label(m))
               for m in majors)
    assert len({figure_data.medium_major_label(m) for m in majors}) == len(majors)
    assert len({figure_data.medium_gwa_label(g) for g in gwas}) == len(gwas)


def test_short_labels_are_unique_so_two_rows_never_read_the_same():
    rows = figure_data.eco_act_split_rows("gwa_title")
    majors = [figure_data.short_major_label(m)
              for m in rows["major_occ_category"].dropna().unique()]
    gwas = [figure_data.short_gwa_label(g)
            for g in rows["gwa_title"].dropna().unique()]
    assert len(set(majors)) == len(majors)
    assert len(set(gwas)) == len(gwas)


# ── Stacked-chart number columns ──────────────────────────────────────────
# The two opening charts absorbed the standalone workers/wages figure as two
# extra number columns. What can silently break: the magnitudes stop matching
# the canonical helpers, or a column drifts into its neighbour.

@pytest.mark.parametrize("col_x, x_top", [
    ((128.0, 179.0, 222.0), 225.0),
    ((131.0, 187.0, 234.0), 236.0),
])
def test_number_columns_sit_past_the_bars_and_inside_the_axis(col_x, x_top):
    from lib.builders import occupation

    live = [(occupation._MAJOR_COL_X, occupation._MAJOR_X_TOP),
            (occupation._GWA_COL_X, occupation._GWA_X_TOP)]
    assert (col_x, x_top) in live, "chart geometry moved — update the test"
    assert col_x[0] > 100.0, "first column overlaps the 100% bar end"
    assert list(col_x) == sorted(col_x), "columns must run left to right"
    assert col_x[-1] < x_top, "last column is clipped by the axis"
    # Each gap has to hold the wider of the two columns' headers ("Workers"
    # is 7 characters at ~0.5 em, so ~36 x-units on these panels).
    assert all(b - a >= 36.0 for a, b in zip(col_x, col_x[1:]))


def test_stacked_magnitudes_match_the_canonical_helpers():
    """The columns are joined onto the stacked frame from major_exposure /
    act_exposure — never re-derived by multiplying a category percentage
    back out, which would get wages wrong."""
    from lib.builders import occupation

    ds = figure_data.PRIMARY_DATASET
    df = occupation._stacked_frame(
        figure_data.pair_exposure_rows(ds), "major_occ_category",
        "hours", "hours_exposed",
    )
    ref = figure_data.major_exposure(ds).set_index("category")
    for _, row in df.iterrows():
        r = ref.loc[row["major_occ_category"]]
        assert row["pct_exposed"] == pytest.approx(float(r["pct"]), abs=0.01)
        assert float(r["workers_fte"]) > 0
        assert float(r["wages_exposed"]) > 0


def test_magnitude_labels_round_to_whole_units():
    from lib.builders import occupation

    assert occupation._fmt_workers_round(11_400_000) == "11M"
    assert occupation._fmt_workers_round(935_000) == "935K"
    assert occupation._fmt_workers_round(36_400) == "36K"
    assert occupation._fmt_wages_round(562_000_000_000) == "$562B"
    assert occupation._fmt_wages_round(1_400_000_000) == "$1B"
    assert occupation._fmt_wages_round(700_000_000) == "$700M"
    for fmt in (occupation._fmt_workers_round, occupation._fmt_wages_round):
        assert all("." not in fmt(v) for v in (1.5e6, 9.94e8, 1.26e12))


# ── Physical makeup is weighted by work time, not by task count ───────────

def test_occ_phys_share_is_time_weighted_not_a_task_count():
    """Every physical share in this folder is a share of HOURS. Employment
    cancels inside an occupation, so it must equal Σ time_per_day over the
    occupation's physical tasks ÷ 7."""
    share = figure_data.occ_phys_hours_share()
    assert len(share) >= 900
    assert share.between(0.0, 100.0).all()

    pairs = figure_data.eco_pairs()
    by_time = (
        pairs.assign(t=np.where(pairs["physical"], pairs[figure_data.TIME_COL], 0.0))
        .groupby("title_current")
        .apply(lambda d: d["t"].sum() / d[figure_data.TIME_COL].sum() * 100.0,
               include_groups=False)
    )
    assert np.allclose(share.sort_index(), by_time.sort_index(), atol=1e-6)


def test_time_weighting_moves_occupations_across_the_tiers():
    """The whole point of the change: a task count and a share of the workday
    do not classify the same occupations, because physical tasks take longer."""
    from lib.figure_data import _load_occ_structural

    occ = _load_occ_structural()
    by_count = occ.set_index("title_current")["occ_group"]
    by_time = figure_data.phys_tier(figure_data.occ_phys_hours_share()).astype(str)
    aligned = by_time.reindex(by_count.index)
    assert (aligned != by_count).sum() > 50
    # Time weighting pulls toward physical — physical tasks are the long ones.
    assert (aligned == "Physical").sum() > (by_count == "Physical").sum()


def test_phys_tier_cuts_at_a_third_and_two_thirds():
    tiers = figure_data.phys_tier(pd.Series([0.0, 32.9, 33.0, 50.0, 67.0, 67.1, 100.0]))
    assert list(tiers.astype(str)) == [
        "Non-physical", "Non-physical", "Non-physical", "Mixed",
        "Mixed", "Physical", "Physical",
    ]


def test_dwa_phys_share_weights_by_hours_not_rows():
    """One long physical task must outweigh several short non-physical ones,
    which is exactly what counting rows gets wrong."""
    from lib.builders import verbs

    rows = pd.DataFrame({
        "dwa_title": ["D"] * 4,
        "family": ["f"] * 4,
        "physical": [True, False, False, False],
        "hours_split": [90.0, 5.0, 3.0, 2.0],
        "hours_exposed_split": [0.0, 0.0, 0.0, 0.0],
    })
    units = verbs._dwa_units(rows)
    assert units.loc[0, "phys_share"] == pytest.approx(0.90)      # by hours
    assert figure_data.phys_tier(units["phys_share"] * 100.0)[0] == "Physical"


# ── Activity /n splits ────────────────────────────────────────────────────

def test_dwa_split_preserves_pair_totals():
    """A task sitting in several DWAs must not have its usage counted once
    per DWA — the /n split is what keeps the per-DWA sums a decomposition."""
    rows = figure_data.intensity_act_rows("dwa_title")
    per_pair = rows.groupby(["task_normalized", "title_current"]).agg(
        split_sum=("adj_pct_split", "sum"), whole=("adj_pct", "first")
    )
    assert np.allclose(per_pair["split_sum"], per_pair["whole"], rtol=1e-9)


def test_eco_dwa_split_preserves_pair_weights():
    rows = figure_data.eco_act_split_rows("dwa_title")
    per_pair = rows.groupby(["task_normalized", "title_current"]).agg(
        split_sum=("eco_weight_split", "sum"), whole=("eco_weight", "first")
    )
    assert np.allclose(per_pair["split_sum"], per_pair["whole"], rtol=1e-9)


def test_gwa_denominator_sums_to_the_full_economy():
    """The /n split means the per-GWA denominators add back up to the whole
    eco weight, so GWA usage shares are comparable to each other."""
    den = figure_data.eco_gwa_weight_split()
    rows = figure_data.eco_act_split_rows("gwa_title")
    pair_total = rows.drop_duplicates(
        ["task_normalized", "title_current"]
    )["eco_weight"].sum()
    assert den.sum() == pytest.approx(pair_total, rel=1e-9)


# ── Non-physical filtering ────────────────────────────────────────────────

def test_nonphys_filter_shrinks_both_sides_of_the_ratio():
    from lib.builders import verbs

    num_all, den_all = verbs._usage_rows(nonphys=False)
    num_np, den_np = verbs._usage_rows(nonphys=True)
    assert len(num_np) < len(num_all)
    assert len(den_np) < len(den_all)
    assert not num_np["physical"].any()
    assert not den_np["physical"].any()


def test_family_usage_anchors_one_family_at_one():
    from lib.builders import verbs

    usage = verbs._family_usage(nonphys=False)
    assert len(usage) == len(figure_data.FAMILY_ORDER)
    assert (usage.round(6) == 1.0).sum() == 1


# ── The hours weighting ───────────────────────────────────────────────────

def test_time_per_day_sums_to_the_workday_for_every_occupation():
    """The whole design rests on this: time_per_day is normalised so an
    occupation's tasks sum to a constant workday. If that ever stops holding,
    a percentage stops being a share of the day and the group rollups stop
    reconciling."""
    pairs = figure_data.eco_pairs()
    per_occ = pairs.groupby("title_current")[figure_data.TIME_COL].sum()
    assert len(per_occ) == 923
    assert np.allclose(per_occ, figure_data.OCC_DAY_HOURS, atol=1e-6)


def test_eco_act_split_preserves_pair_hours():
    rows = figure_data.eco_act_split_rows("gwa_title")
    per_pair = rows.groupby(["task_normalized", "title_current"]).agg(
        split_sum=("hours_split", "sum"), whole=("hours", "first")
    )
    assert np.allclose(per_pair["split_sum"], per_pair["whole"], rtol=1e-9)


def test_major_and_gwa_exposure_reconcile():
    """The same exposed hours reached two different ways — summed over
    occupations, and summed over /n-split work-activity rows. If these ever
    drift apart, one of the two chart families is lying."""
    ds = figure_data.PRIMARY_DATASET
    maj = figure_data.major_exposure(ds)
    gwa = figure_data.act_exposure(ds, "gwa_title")
    for col in ("hours", "hours_exposed", "wages_exposed"):
        assert maj[col].sum() == pytest.approx(gwa[col].sum(), rel=1e-6), col


def test_total_hours_equal_workday_times_employment():
    ds = figure_data.PRIMARY_DATASET
    maj = figure_data.major_exposure(ds)
    expected = maj["emp"].sum() * figure_data.OCC_DAY_HOURS
    assert maj["hours"].sum() == pytest.approx(expected, rel=1e-9)


def test_group_pct_equals_workers_over_employment():
    """Employment-weighted % of work time exposed IS workers exposed divided
    by employment — the identity that lets the two charts be read together."""
    maj = figure_data.major_exposure(figure_data.PRIMARY_DATASET)
    assert np.allclose(maj["pct"], maj["workers_fte"] / maj["emp"] * 100.0)


def test_exposed_hours_never_exceed_total_hours():
    for level, frame in (
        ("major", figure_data.major_exposure(figure_data.PRIMARY_DATASET)),
        ("gwa", figure_data.act_exposure(figure_data.PRIMARY_DATASET, "gwa_title")),
    ):
        assert (frame["hours_exposed"] <= frame["hours"] + 1e-6).all(), level
        assert (frame["pct"] <= 100.0 + 1e-9).all(), level


def test_pair_level_emp_counts_employment_once_per_task():
    """One denominator rule everywhere: employment summed over (task, occ)
    pairs. Activity groupings split the same pair /n across its activities;
    occupation groupings see each pair once, so there is nothing to split."""
    by_major = figure_data.pair_level_emp("major_occ_category")
    total = figure_data.eco_pairs()[figure_data.EMP_COL].sum()
    assert by_major.sum() == pytest.approx(total, rel=1e-9)
    # And it matches what the activity-level denominator sums to, since the
    # /n split puts a pair's employment back together across its activities.
    assert by_major.sum() == pytest.approx(
        figure_data.eco_gwa_weight_split().sum(), rel=1e-9
    )


# ── Backend time_day method ───────────────────────────────────────────────

def test_backend_time_day_uses_the_time_column():
    from backend.compute import compute_task_comp

    df = pd.DataFrame({
        "freq_mean": [1.0, 9.0], "relevance": [50.0, 50.0],
        "importance": [3.0, 3.0], "time_per_day": [5.0, 2.0],
        "auto_aug_mean": [5.0, 5.0],
    })
    tc = compute_task_comp(df, "time_day", use_auto_aug=False)
    assert list(tc) == [5.0, 2.0]


def test_backend_rejects_an_unknown_method():
    """The branch used to fall through to freq x rel x imp for anything it
    did not recognise, so a typo silently changed the metric."""
    from backend.compute import compute_task_comp

    df = pd.DataFrame({"freq_mean": [1.0], "relevance": [1.0],
                       "importance": [1.0], "time_per_day": [1.0]})
    with pytest.raises(AssertionError):
        compute_task_comp(df, "time", use_auto_aug=False)


# ── Verb-family totals (the number the overview chart ranks on) ───────────

def test_family_total_matches_the_hours_ratio_of_totals():
    """The family-total column is the same ratio of totals the major and GWA
    charts report, cut by verb family — exposed hours over hours for the
    family's DWA rows."""
    from lib.builders import verbs

    rows = verbs._dwa_task_rows(figure_data.PRIMARY_DATASET)
    aug = verbs._family_autoaug(rows)
    direct = (
        rows.groupby("family")["hours_exposed_split"].sum()
        / rows.groupby("family")["hours_split"].sum() * 100.0
    )
    assert np.allclose(aug["pct_exposed"], direct.reindex(aug.index), rtol=1e-9)


def test_automation_level_is_an_unweighted_task_mean():
    """The two automation bars are plain means of the 0-5 score, NOT the
    hours-weighted version the family total uses — they answer "how capable
    is AI on these tasks", not "how much of the work is reached"."""
    from lib.builders import verbs

    rows = verbs._dwa_task_rows(figure_data.PRIMARY_DATASET)
    aug = verbs._family_autoaug(rows)
    plain_all = rows.groupby("family")["auto_aug_mean"].apply(
        lambda s: float(s.fillna(0.0).mean())
    )
    plain_rated = rows[rows["exposed"]].groupby("family")["auto_aug_mean"].mean()
    assert np.allclose(aug["aug_all"], plain_all.reindex(aug.index), rtol=1e-9)
    assert np.allclose(aug["aug_exposed"], plain_rated.reindex(aug.index), rtol=1e-9)
    # Unrated rows enter the all-tasks mean as 0, so it can never exceed the
    # rated-only mean.
    assert (aug["aug_all"] <= aug["aug_exposed"] + 1e-9).all()
    # And it is no longer the family total rescaled.
    assert not np.allclose(aug["aug_all"] / 5.0 * 100.0, aug["pct_exposed"])


def test_exemplar_contributions_sum_to_zero_in_every_cell():
    """A cell's percentage is the hours-weighted mean of its DWAs, so each
    DWA's `share × (pct − cell)` must cancel across the cell. That identity is
    what lets the exemplar be chosen without a minimum-size floor — break it
    and the selection rule is picking on something that isn't a decomposition.
    """
    from lib.builders import verbs

    _rows, cells, units = verbs._family_major_cells(figure_data.PRIMARY_DATASET)
    for _i, cell in cells.iterrows():
        c = units[(units["major_occ_category"] == cell["major_occ_category"])
                  & (units["family"] == cell["family"])]
        share = c["hours"] / c["hours"].sum()
        assert abs(float((share * (c["pct"] - cell["pct"])).sum())) < 1e-6, (
            f"{cell['major_occ_category']} / {cell['family']}"
        )


def test_exemplar_is_the_largest_contributor_not_the_most_extreme():
    """The pick must beat every other DWA on contribution — not on its own
    percentage. Ranking on the rate alone surfaced slivers holding a tenth of
    a percent of the cell, which is the failure this rule exists to avoid.
    """
    from lib.builders import verbs

    _rows, cells, units = verbs._family_major_cells(figure_data.PRIMARY_DATASET)
    cell = cells.sort_values("pct", ascending=False).iloc[0]
    for most in (True, False):
        pick = verbs._exemplar(units, cell["major_occ_category"],
                               cell["family"], cell["pct"], most)
        c = units[(units["major_occ_category"] == cell["major_occ_category"])
                  & (units["family"] == cell["family"])].copy()
        c["contrib"] = (c["hours"] / c["hours"].sum()) * (c["pct"] - cell["pct"])
        best = c["contrib"].max() if most else c["contrib"].min()
        got = (pick["share"] * (pick["pct"] - cell["pct"]))
        assert abs(got - best) < 1e-9
        # A winning contribution implies real size: a DWA cannot carry the
        # cell while holding a rounding error of its hours.
        assert pick["share"] > 0.01


def test_exemplar_majors_come_from_the_top_pool():
    """Unrestricted, the comparison set is dominated by physical majors and
    the figure just re-tells the phys/non-phys split. The pool is the gate."""
    from lib.builders import verbs

    _rows, cells, _units = verbs._family_major_cells(figure_data.PRIMARY_DATASET)
    pool = set(
        figure_data.major_exposure(figure_data.PRIMARY_DATASET)
        .sort_values("pct", ascending=False)["category"]
        .tolist()[:verbs.TOP_MAJORS_POOL]
    )
    assert set(cells["major_occ_category"]) <= pool
    assert len(pool) == verbs.TOP_MAJORS_POOL


def _standouts():
    from lib.builders import verbs

    rows, cells, units = verbs._family_major_cells(figure_data.PRIMARY_DATASET)
    eco = verbs._family_autoaug(rows)["pct_exposed"]
    cells = cells.copy()
    cells["share"] = (cells["hours"]
                      / cells.groupby("major_occ_category")["hours"]
                      .transform("sum") * 100.0)
    cells["gap"] = cells["pct"] - cells["family"].map(eco)
    return verbs._major_standouts(cells, eco, units), cells, eco


def test_every_pooled_major_gets_exactly_one_row():
    df, cells, _eco = _standouts()
    assert len(df) == cells["major_occ_category"].nunique()
    assert df["major"].is_unique
    # Sorted by the signed gap, so the negatives sit below every positive.
    assert df["gap"].is_monotonic_decreasing


def test_the_named_family_is_a_big_part_of_that_majors_day():
    """The eligibility rule is what stops a 0.2%-of-the-day family winning a
    row on a large percentage-point gap — the failure this figure had before.
    """
    df, cells, _eco = _standouts()
    for _i, r in df.iterrows():
        grp = cells[cells["major_occ_category"] == r["major"]]
        assert r["family_share"] >= grp["share"].median(), r["major"]
    # Every row's family is a real slice of the job, not a corner of it.
    assert df["family_share"].min() > 5.0
    # And the eligible set is most of the working day, which is what lets the
    # rule be stated as "the larger half of their work".
    assert df["eligible_share"].min() > 60.0


def test_the_named_gap_is_the_largest_among_eligible_families():
    df, cells, _eco = _standouts()
    for _i, r in df.iterrows():
        grp = cells[cells["major_occ_category"] == r["major"]]
        elig = grp[grp["share"] >= grp["share"].median()]
        assert abs(r["gap"]) == pytest.approx(elig["gap"].abs().max())


# ── MCP high-rated slice ──────────────────────────────────────────────────

def test_occ_group_rollup_matches_the_major_helper():
    """The generic occupation-group rollup has to reproduce
    figure_data.major_exposure exactly at major level, or the minor and broad
    blocks are computed on a different footing from the rest of the set."""
    from lib.builders import agentic

    mine = agentic._occ_group_exposure(
        figure_data.PRIMARY_DATASET, "major_occ_category")
    ref = figure_data.major_exposure(figure_data.PRIMARY_DATASET).set_index("category")["pct"]
    assert len(mine) == 22
    assert np.allclose(mine.sort_index(), ref.reindex(mine.sort_index().index),
                       rtol=1e-9)


def test_the_agentic_level_frame_has_both_sides_counts_and_shares():
    """Each panel must resolve a confirmed percentage, a tooling percentage,
    its rating counts and its share of the catalogue's attention."""
    from lib.builders import agentic

    col, is_act = agentic.LEVEL_COL, agentic.LEVEL_IS_ACTIVITY
    df = agentic._level_frame(col, is_act)
    assert len(df) >= agentic.TOP_N, col
    for c in ("pct_conf", "pct_mcp", "n_high", "n_rated", "mcp_pct_share"):
        assert df[c].notna().all(), f"{col}: {c}"
    assert (df["n_high"] <= df["n_rated"]).all(), col
    assert (df["pct_mcp"] <= 100.0 + 1e-9).all(), col
    # Ranked by the tooling-minus-confirmed gap, descending — the headroom
    # the figure is about, not coverage alone.
    assert df["gap"].is_monotonic_decreasing, col
    assert np.allclose(df["gap"], df["pct_mcp"] - df["pct_conf"])
    # The two halves have to cover the top N exactly once between them.
    assert [i for lo, hi in agentic._SPLITS for i in range(lo, hi)] == list(
        range(agentic.TOP_N))


def test_mcp_attention_share_sums_to_one_hundred_down_each_panel():
    """The catalogue's pct is already a share over unique (task, occ) pairs,
    so a correct rollup sums back to 100 at every level. At activity grain
    that only holds if the pair's share is /n-split across its activities —
    without the split the same attention is counted two or three times."""
    from lib.builders import agentic

    for col, is_act in ((agentic.LEVEL_COL, agentic.LEVEL_IS_ACTIVITY),
                        ("minor_occ_category", False)):
        share = agentic._mcp_pct_share(col, is_act)
        assert abs(float(share.sum()) - 100.0) < 1e-6, col
        assert (share >= 0).all(), col


def test_the_tooling_column_clears_every_tooling_bar():
    """The MCP number moved off its bar into a fixed right-anchored column so
    it stops colliding with the confirmed label. That only works while the
    column sits past the longest bar and short of the ratings column."""
    from lib.builders import agentic

    top = agentic._level_frame(
        agentic.LEVEL_COL, agentic.LEVEL_IS_ACTIVITY).head(agentic.TOP_N)
    # A 5-glyph label is ~41 x-units on this panel; the column is right-
    # anchored, so its left edge is the anchor minus that.
    assert agentic._MCP_COL_X - 41.0 >= float(top["pct_mcp"].max()) - 1.0
    assert agentic._SHARE_X > agentic._MCP_COL_X


def test_wrapped_labels_stay_within_two_lines():
    from lib.builders import agentic

    top = agentic._level_frame(
        agentic.LEVEL_COL, agentic.LEVEL_IS_ACTIVITY).head(agentic.TOP_N)
    for name in top["category"]:
        lines = agentic._wrap(name).split("<br>")
        assert len(lines) <= 2, name
        assert all(len(l) <= agentic._LABEL_LINE + 6 for l in lines), name


def test_rating_counts_exclude_the_not_automatable_level():
    """n_rated is the count of ratings that say anything — levels 2 through
    5. A rating of 1 means the server judged the task not automatable and is
    deliberately not counted."""
    from lib.builders import agentic

    assert "n_rating_1" not in agentic.RATING_COLS
    counts = agentic._mcp_rating_counts("major_occ_category")
    # _mcp_rating_counts renames the top two levels for the chart's headers.
    parts = ["n_rating_2", "n_rating_3", "n_4", "n_5"]
    assert np.allclose(counts["n_rated"], counts[parts].sum(axis=1))
    assert (counts["n_4"] <= counts["n_rated"]).all()
    assert (counts["n_5"] <= counts["n_rated"]).all()


# ── Trend series ──────────────────────────────────────────────────────────

def test_trend_is_three_all_confirmed_lines_on_one_dataset():
    """The agentic line was dropped: every series now comes off the same
    All Confirmed snapshots and differs only by physical filter, which is
    what lets the dataset be named in the title and the legend be one row."""
    from lib.builders import trend

    spec = trend._series_spec()
    assert len(spec) == 3
    assert {s[3] for s in spec} == {"all", "exclude", "only"}
    for _key, _label, datasets, _mode, _color, _dash in spec:
        assert datasets is figure_data.AC_SERIES


# ── Trend label placement ─────────────────────────────────────────────────

def _no_crossings(placed, ys, label_h):
    """No label box contains another series' line, and no two boxes overlap."""
    half = label_h * 0.58
    for i, p in enumerate(placed):
        for y in ys:
            if p - half < y < p + half:
                return False
        for q in placed[i + 1:]:
            if abs(p - q) < label_h * 1.15:
                return False
    return True


def test_above_below_keeps_every_label_off_every_line():
    """The real last-observed values at the chart's real geometry: four
    series in a 36-unit band with a 5.3-unit label."""
    from lib.builders import trend

    ys = [55.2, 37.9, 30.1, 19.0]
    label_h = 5.3
    placed = trend._assign_above_below(ys, label_h, offset=label_h * 1.15)
    assert _no_crossings(placed, ys, label_h)


def test_above_below_handles_a_pair_too_close_to_sit_between():
    """16.1 and 13.9 are 2.2 units apart — no label fits between them, so the
    solver has to send one label out to the far side of its own line."""
    from lib.builders import trend

    ys = [45.4, 30.4, 16.1, 13.9]
    label_h = 5.3
    placed = trend._assign_above_below(ys, label_h, offset=label_h * 1.15)
    assert _no_crossings(placed, ys, label_h)


def test_min_gap_positions_separates_without_reordering():
    from lib.builders import trend

    out = trend._min_gap_positions([45.4, 30.4, 16.1, 13.9], 5.0)
    ordered = sorted(out)
    assert all(b - a >= 5.0 - 1e-9 for a, b in zip(ordered, ordered[1:]))
    assert [sorted(out).index(v) for v in out] == [3, 2, 1, 0]


# ── Focused set membership ────────────────────────────────────────────────

def test_focused_set_gate_is_exposure_and_projection_only():
    """Every member clears the stated exposure floor and has a negative BLS
    projection — the two gates the title names, and no others."""
    from lib.exploratory.risk_score import _load_flag_df
    from lib.builders import focused

    flags = _load_flag_df()
    exposure = figure_data.occ_exposure(figure_data.PRIMARY_DATASET).set_index("title_current")
    pct = flags["title_current"].map(exposure["p"] * 100.0)
    members = flags[(pct >= focused.EXPOSURE_MIN) & (flags["emp_proj_pct"] < 0)]
    assert 20 <= len(members) <= 45, len(members)
    assert (members["emp_proj_pct"] < 0).all()


def test_focused_worker_labels_go_outside_bars_they_would_overrun():
    """A workers label sits inside its bar only if the bar is wider than the
    label. The old share-of-max rule put 516K inside a 66 px bar under a
    79 px label, which spilled left into the occupation names."""
    from lib.builders import focused

    x_top = 2_037_891.6 * 2.05        # the chart's own axis top
    floor_px = 32                     # the 8 pt floor at this canvas
    inside = {
        v: focused.worker_label_inside(v, t, x_top, floor_px)
        for v, t in [(2_037_891, "2.0M"), (1_697_218, "1.7M"),
                     (1_274_388, "1.3M"), (1_033_075, "1.0M"),
                     (997_237, "997K"), (516_095, "516K"), (299_376, "299K")]
    }
    assert inside[516_095] is False
    assert inside[299_376] is False
    assert all(inside[v] for v in (997_237, 1_033_075, 1_274_388,
                                   1_697_218, 2_037_891))


# ── Config pins ───────────────────────────────────────────────────────────

def test_all_five_configs_are_registered_and_eco_2025_based():
    from backend import config as backend_config

    assert len(figure_data.CONFIG_DATASETS) == 5
    for key, dataset in figure_data.CONFIG_DATASETS.items():
        meta = backend_config.DATASETS[dataset]
        assert Path(meta["file"]).exists(), f"{key}: missing {meta['file']}"
        assert not meta["is_aei"], f"{key} is eco_2015-based; expected eco_2025"
        assert key in figure_data.CONFIG_LABELS


# ── Supplement: usage drivers ─────────────────────────────────────────────

def test_dwa_sits_under_exactly_one_gwa():
    """The four DWA charts slice the economy's activities by GWA. That is
    only a clean slice because the nesting is 1:1 — if a DWA sat under two
    GWAs its usage would appear on two charts."""
    from lib.builders import drivers

    nest = drivers.dwa_to_gwa()
    assert nest.index.is_unique
    assert len(nest) > 2000
    assert nest.nunique() == 37


def test_gwa_ranking_reproduces_the_published_adoption_chart():
    """The top-4 GWAs are picked off `gwa_usage_ranking`, but the number a
    reader sees for them is the one on the main-body adoption chart. If the
    two constructions drift, the supplement charts silently stop being the
    top 4 of the figure they claim to decompose."""
    from lib.builders import (
        adoption, drivers,
    )

    mine = drivers.gwa_usage_ranking().set_index("gwa_title")["lift"]

    rows = figure_data.intensity_gwa_rows()
    den = figure_data.eco_gwa_weight_split()
    theirs = adoption._usage_frame(
        rows.groupby("gwa_title")["adj_pct_split"].sum().reindex(den.index).fillna(0.0),
        den,
        pd.Series(0.0, index=den.index),
    ).set_index("category")["lift"]

    aligned = pd.concat([mine, theirs], axis=1, join="inner")
    assert len(aligned) == len(den)
    assert np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])


def test_dwa_anchor_is_economy_wide_not_per_gwa():
    """Deliberate departure from the folder's anchor rule, so it is pinned:
    exactly one DWA in the ECONOMY reads 1.00x, and the four per-GWA slices
    do not each carry their own 1.00x row."""
    from lib.builders import drivers

    frame = drivers.dwa_usage_frame()
    at_one = np.isclose(frame["lift"], 1.0)
    assert at_one.sum() == 1

    # The single 1.00x row lives in whichever GWA holds the economy's median
    # activity, so at most one of the four charts can show one — which is
    # the whole point of anchoring economy-wide rather than per GWA.
    top4 = set(drivers.gwa_usage_ranking().head(4)["gwa_title"])
    assert frame.loc[at_one, "gwa_title"].isin(top4).sum() <= 1


def test_usage_frames_are_ratios_of_sums_with_positive_denominators():
    from lib.builders import drivers

    for frame in (drivers.occ_usage_frame(drivers.TARGET_MAJORS[0][0]),
                  drivers.dwa_usage_frame()):
        assert (frame["den"] > 0).all()
        assert np.allclose(frame["ratio"], frame["num"] / frame["den"])
        assert (frame["raw_pct"] >= 0).all()


def test_driver_charts_carry_intensity_and_raw_pct_only():
    """The supplement's usage charts print the lift and the summed RAW pct.
    A debiased-share column was deliberately left off — the debias is a
    correction applied to the numerator, not an observation to report."""
    from lib.builders import drivers

    frame = drivers.dwa_usage_frame()
    assert {"lift", "raw_pct"} <= set(frame.columns)
    assert "debias_share" not in frame.columns


def test_gwa_title_labels_fit_a_chart_title():
    from lib.builders import drivers

    for gwa in drivers.gwa_usage_ranking()["gwa_title"]:
        label = drivers.gwa_title_label(gwa)
        assert 0 < len(label) <= drivers._GWA_TITLE_MAX
        # The full rendered title has to clear the ~85-char cap at 11 pt.
        assert len(f"AI Usage as Multiple of Median — Top Activities in {label}") <= 85


# ── Supplement: the adoption charts' number column ────────────────────────

def test_adoption_number_column_is_raw_not_debiased():
    """These are different numbers, not a relabelling — Computer & Math is
    ~38% of raw usage and ~27% of debiased usage. The renderer asserts the
    raw column is attached; this pins that they actually differ."""
    from lib.builders import adoption

    pairs = figure_data.intensity_pairs()
    num = pairs.groupby("major_occ_category")["adj_pct"].sum()
    den = figure_data.pair_level_emp("major_occ_category")
    exposure = figure_data.major_exposure(figure_data.PRIMARY_DATASET).set_index(
        "category")["pct"]
    df = adoption._usage_frame(num, den, exposure)

    raw = pairs.groupby("major_occ_category")["pct_normalized"].sum()
    df["raw_share"] = df["category"].map(raw / raw.sum() * 100.0).fillna(0.0)

    assert df["raw_share"].sum() == pytest.approx(100.0, abs=0.01)
    assert df["debias_share"].sum() == pytest.approx(100.0, abs=0.01)
    assert not np.allclose(df["raw_share"], df["debias_share"])


def test_adoption_renderer_requires_the_raw_column():
    from lib.builders import adoption

    bare = pd.DataFrame({"lift": [1.0, 2.0], "pct_exposed": [10.0, 20.0]})
    with pytest.raises(AssertionError):
        adoption._render_adoption(
            bare, ["a", "b"], "t", 1400, 44, 200, Path("unused.png"),
        )


# ── Supplement: benchmark series ──────────────────────────────────────────

def test_internal_series_is_hours_weighted_at_both_levels():
    from lib.builders import benchmarks

    occ = benchmarks.internal_series("all_confirmed", "occupation")
    major = benchmarks.internal_series("all_confirmed", "major")
    assert len(occ) > 800 and len(major) == 22
    for s in (occ, major):
        assert s.between(0, 100).all()


def test_major_convergence_series_is_employment_weighted():
    """The reason these figures do not just call the paper's `_run_config`
    with method="time_day": time_per_day sums to a constant 7.0 per
    occupation, so a plain ratio-of-totals at major level collapses to the
    UNWEIGHTED mean of the member occupations. `major_exposure` puts
    employment back, and the two readings genuinely differ."""
    from lib.builders import benchmarks

    major = benchmarks.internal_series("all_confirmed", "major")
    occ = figure_data.occ_exposure(figure_data.PRIMARY_DATASET)
    unweighted = occ.groupby("major_occ_category")["p"].mean() * 100.0

    aligned = pd.concat([major, unweighted], axis=1, join="inner").dropna()
    assert len(aligned) == 22
    assert not np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], atol=0.5)

    weighted = figure_data.major_exposure(figure_data.PRIMARY_DATASET).set_index(
        "category")["pct"]
    assert np.allclose(major.reindex(weighted.index), weighted)


def test_internal_series_falls_back_for_an_unpinned_key():
    """Returning None is what tells `build_convergence_full` to use its own
    freq-weighted series, so a measure added to the paper's CORR_ORDER
    degrades to the paper reading instead of vanishing from the matrix."""
    from lib.builders import benchmarks

    assert benchmarks.internal_series("not_a_real_key", "major") is None


def test_every_internal_measure_is_pinned_and_registered():
    from backend import config as backend_config
    from lib.builders.benchmarks import CORR_ORDER, CONFIG_ORDER
    from lib.builders import benchmarks

    assert set(benchmarks.INTERNAL_DATASETS) == set(CORR_ORDER) | set(CONFIG_ORDER)
    for key, dataset in benchmarks.INTERNAL_DATASETS.items():
        meta = backend_config.DATASETS[dataset]
        assert Path(meta["file"]).exists(), f"{key}: missing {meta['file']}"
        assert not meta["is_aei"], f"{key} is eco_2015-based"


def test_driver_labels_fit_the_canvas():
    """The supplement's usage charts pad the left margin (`WIDE_LABEL_PAD`)
    so the rotated axis title clears the longest row label, and pay for it
    with matching x headroom (`WIDE_LABEL_X_MULT`) so the widest bar's
    outside label still lands inside the PNG. Both were tuned against the
    current top-10s, and the thing that would break them is a longer label
    — the outside text is `"{lift:.2f}x   ({raw:.3f}% raw pct)"`, so its
    width is set by the integer digits of the lift.
    """
    from lib.builders import drivers

    assert drivers.WIDE_LABEL_X_MULT > 1.92, (
        "padding the left margin without paying for it on the x axis clips "
        "the widest bar's label"
    )

    LABEL_CHARS_MAX = 30
    frames = [drivers.task_usage_frame(m) for m, _, _ in drivers.TARGET_MAJORS]
    dwa = drivers.dwa_usage_frame()
    top4 = drivers.gwa_usage_ranking().head(4)["gwa_title"]
    frames += [dwa[dwa["gwa_title"] == g] for g in top4]

    for frame in frames:
        top = frame.nlargest(drivers.TOP_N_ROWS, "lift")
        widest = max(
            len(f"{lift:.2f}x   ({raw:.3f}% raw pct)")
            for lift, raw in zip(top["lift"], top["raw_pct"])
        )
        assert widest <= LABEL_CHARS_MAX, (
            f"outside label grew to {widest} chars; re-check "
            "WIDE_LABEL_X_MULT against the render"
        )
