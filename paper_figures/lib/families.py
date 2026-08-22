"""Verb-family math — shared by the paper figures and the dashboard.

The eight families (`figure_data.FAMILY_LABELS`) re-bucket O*NET detailed
work activities by the action the work *is* — Analyze, Create, Evaluate,
Manage — cutting across the GWA → IWA → DWA hierarchy rather than sitting
inside it. That is why a family is never a level in the activity selector.

Everything here resolves to two quantities, and they are the same two the
rest of the product reports:

    exposure = Σ hours×(auto_aug/5) over the family's rows
             ÷ Σ hours over the family's rows            (work-time weighted)

    usage    = Σ debiased pct ÷ Σ emp                    (employment-weighted)
               anchored so the lower-middle FAMILY reads 1.00×

**The /n split is load-bearing.** A task mapping to n distinct DWAs has its
hours, its usage and its economic weight divided by n, so summing by family
decomposes the workday instead of counting a multi-DWA task several times.
Every aggregation below goes through `figure_data`'s split rows for exactly
this reason; do not re-derive one from unsplit rows.

**Anchors are per-grain, and that is not a preference.** A usage rate is
Σ usage ÷ Σ emp, so its scale depends on how many occupations are inside the
group. Dividing a single occupation's family rate by the economy-wide family
anchor produced readings of 270× for Computer Programmers — arithmetically
true, useless on a card, and comparing two quantities that do not live on
one scale. So:

- economy-wide / per-major family rows anchor on the **economy-wide
  lower-middle family** (the paper's chart anchor);
- an occupation's family rows anchor on **that occupation's own lower-middle
  family**, i.e. "relative to a typical part of this job".

Cross-job comparability is carried by the single occupation-level ×median
number at the top of the card, which is a paper quantity. Each number
answers one question; neither is asked to answer both.

**Two usage columns on the occupation grain, and the card picks one.**
`usage_x` is the paper's ratio. At n = 1-3 rated tasks per cell it is
volatile: when a job has several zero-usage families the anchor lands on a
near-zero family and the rest read in the tens (Paralegals: 45x). `usage_share`
is the same numerator expressed as a share of the occupation's own observed
AI use — bounded 0-100, sums to 100 across the job's families, no anchor to
be volatile about. It is a subtotal of the published numerator rather than a
new metric, and it answers the card's actual question ("which parts of my job
get the AI use") more plainly than a ratio does.

Exposure and usage are always returned together (PRD §2.3): a rated row
carries both by construction, so any shape that could show one without the
other is a bug waiting to happen.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from lib import figure_data
from lib.figure_data import DEFAULT_GEO, FAMILY_LABELS, FAMILY_ORDER, FAMILY_SHORT

__all__ = [
    "dwa_task_rows", "family_usage_rate", "family_usage_anchor",
    "family_rows", "occupation_family_rows", "occupation_family_tasks",
    "family_dwa_rows", "FAMILY_LABELS", "FAMILY_ORDER", "FAMILY_SHORT",
]


def dwa_task_rows(
    dataset: str,
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """(occupation, task, DWA) rows tagged with family, carrying /n-split
    hours and the dataset's exposed hours (0 where the pair went unrated).
    """
    rows = figure_data.act_exposure_rows(
        dataset, "dwa_title", physical_mode=physical_mode, geo=geo
    )
    rows["family"] = figure_data.assign_family(rows["dwa_title"])
    rows["exposed"] = rows["auto_aug_mean"].notna()
    return rows


def _usage_frames(
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(numerator, denominator) rows at DWA grain, family-tagged."""
    num = figure_data.intensity_act_rows("dwa_title", geo=geo)
    den = figure_data.eco_act_split_rows("dwa_title", geo=geo)
    if physical_mode == "exclude":
        num, den = num[~num["physical"]].copy(), den[~den["physical"]].copy()
    elif physical_mode == "only":
        num, den = num[num["physical"]].copy(), den[den["physical"]].copy()
    num["family"] = figure_data.assign_family(num["dwa_title"])
    den["family"] = figure_data.assign_family(den["dwa_title"])
    return num, den


def family_usage_rate(
    group_cols: Optional[list[str]] = None,
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.Series:
    """Raw (un-anchored) usage rate per group. `group_cols` defaults to
    ["family"]; pass e.g. ["title_current", "family"] for the card."""
    num, den = _usage_frames(physical_mode, geo)
    return figure_data.usage_rate(num, den, group_cols or ["family"])


def family_usage_anchor(
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> float:
    """The economy-wide lower-middle family rate. Divide any family usage
    rate by this — at any grain — so 1.00× always means the same thing."""
    return figure_data.lower_median(
        family_usage_rate(["family"], physical_mode, geo)
    )


def _attach_usage(
    out: pd.DataFrame,
    rate: pd.Series,
    keys: list[str],
    anchor: float,
) -> pd.DataFrame:
    """Join an anchored usage column onto an exposure frame.

    Groups absent from the numerator come back as 0 from `usage_rate`, which
    is a real zero (nothing observed), not missing data — so the column is
    never NaN and no row can render an exposure number with a blank beside
    it.
    """
    idx = out.set_index(keys).index if len(keys) > 1 else out[keys[0]]
    out = out.copy()
    out["usage_x"] = pd.Series(rate.reindex(idx).to_numpy(), index=out.index).fillna(0.0) / anchor
    return out


def _share_of(mass: pd.Series) -> pd.Series:
    """Express a usage mass as a percentage of its own total (0 if empty)."""
    total = float(mass.sum())
    return (mass / total * 100.0) if total > 0 else mass * 0.0


def family_rows(
    dataset: str,
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
    extra_group: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Economy-wide (or per `extra_group`) family rows: exposure, its
    complement, usage ×median, hours and task counts.

    Pass `extra_group=["major_occ_category"]` for the family × major grid.
    """
    keys = (extra_group or []) + ["family"]
    rows = dwa_task_rows(dataset, physical_mode, geo)
    out = (
        rows.groupby(keys)
        .agg(hours=("hours_split", "sum"),
             hours_exposed=("hours_exposed_split", "sum"),
             n_tasks=("task_normalized", "nunique"),
             n_dwas=("dwa_title", "nunique"))
        .reset_index()
    )
    out["pct_exposed"] = np.where(
        out["hours"] > 0, out["hours_exposed"] / out["hours"] * 100.0, 0.0
    ).clip(0.0, 100.0)
    out["pct_unexposed"] = 100.0 - out["pct_exposed"]

    anchor = family_usage_anchor(physical_mode, geo)
    rate = family_usage_rate(keys, physical_mode, geo)
    out = _attach_usage(out, rate, keys, anchor)

    out["label"] = out["family"].map(FAMILY_LABELS)
    out["short"] = out["family"].map(FAMILY_SHORT)
    return out.sort_values("pct_exposed", ascending=False).reset_index(drop=True)


def occupation_family_rows(
    title: str,
    dataset: str,
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """The occupation card's family rows — one per family present in this
    job (median 5 of 8), with exposure, its complement, and usage anchored
    on this occupation's own lower-middle family (see module docstring).

    `n_tasks` is on every row deliberately: the median (occupation, family)
    cell holds 2 rated tasks, so a single task can swing a family and the
    reader has to be able to see that.
    """
    rows = dwa_task_rows(dataset, "all", geo)
    rows = rows[rows["title_current"] == title]
    assert not rows.empty, f"No task rows for occupation {title!r}"

    out = (
        rows.groupby("family")
        .agg(hours=("hours_split", "sum"),
             hours_exposed=("hours_exposed_split", "sum"),
             n_tasks=("task_normalized", "nunique"),
             n_rated=("exposed", "sum"))
        .reset_index()
    )
    out["pct_exposed"] = np.where(
        out["hours"] > 0, out["hours_exposed"] / out["hours"] * 100.0, 0.0
    ).clip(0.0, 100.0)
    out["pct_unexposed"] = 100.0 - out["pct_exposed"]
    # Share of this job's whole workday the family accounts for — what makes
    # "Evaluate is barely exposed" weigh differently in two jobs.
    total = out["hours"].sum()
    out["share_of_day"] = np.where(total > 0, out["hours"] / total * 100.0, 0.0)

    num, _ = _usage_frames("all", geo)
    usage_mass = (
        num[num["title_current"] == title].groupby("family")["adj_pct_split"].sum()
    )

    rate = family_usage_rate(["title_current", "family"], "all", geo)
    rate = rate.loc[rate.index.get_level_values("title_current") == title]
    rate.index = rate.index.droplevel("title_current")
    # Re-anchored on this occupation's own lower-middle family: 1.00x means
    # "a typical part of this job", not "a typical family in the economy".
    out = _attach_usage(out, rate, ["family"], figure_data.lower_median(rate))
    out["usage_share"] = _share_of(out["family"].map(usage_mass).fillna(0.0))

    out["label"] = out["family"].map(FAMILY_LABELS)
    out["short"] = out["family"].map(FAMILY_SHORT)
    return out.sort_values("pct_exposed", ascending=False).reset_index(drop=True)


def occupation_family_tasks(
    title: str,
    dataset: str,
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """Every task of one occupation, with its family, its DWAs, its exposure
    and its usage — the rows behind an expanded family on the card."""
    rows = dwa_task_rows(dataset, "all", geo)
    rows = rows[rows["title_current"] == title]
    assert not rows.empty, f"No task rows for occupation {title!r}"

    out = (
        rows.groupby(["family", "task_normalized"])
        .agg(hours=("hours_split", "sum"),
             hours_exposed=("hours_exposed_split", "sum"),
             auto_aug=("auto_aug_mean", "max"),
             dwas=("dwa_title", lambda s: sorted(set(s))))
        .reset_index()
    )
    out["pct_exposed"] = np.where(
        out["hours"] > 0, out["hours_exposed"] / out["hours"] * 100.0, 0.0
    ).clip(0.0, 100.0)
    out["pct_unexposed"] = 100.0 - out["pct_exposed"]

    pairs = figure_data.intensity_pairs(geo=geo)
    pairs = pairs[pairs["title_current"] == title]
    rate = pairs.set_index("task_normalized")["intensity"]
    # Anchored on this occupation's own median TASK, for the same reason the
    # family rows re-anchor: a task rate and an economy-wide family rate are
    # not on one scale.
    anchor = figure_data.lower_median(rate) if len(rate.dropna()) else 1.0
    out["usage_x"] = out["task_normalized"].map(rate).fillna(0.0) / anchor
    mass = pairs.set_index("task_normalized")["adj_pct"]
    out["usage_share"] = _share_of(out["task_normalized"].map(mass).fillna(0.0))
    out["label"] = out["family"].map(FAMILY_LABELS)
    return out.sort_values(["family", "pct_exposed"], ascending=[True, False]).reset_index(drop=True)


def family_dwa_rows(
    family: str,
    dataset: str,
    physical_mode: str = "all",
    geo: str = DEFAULT_GEO,
) -> pd.DataFrame:
    """The DWAs inside one family — the Explore drill-down target."""
    assert family in FAMILY_LABELS, f"Unknown family {family!r}"
    rows = dwa_task_rows(dataset, physical_mode, geo)
    rows = rows[rows["family"] == family]
    assert not rows.empty, f"No rows for family {family!r}"

    out = (
        rows.groupby("dwa_title")
        .agg(hours=("hours_split", "sum"),
             hours_exposed=("hours_exposed_split", "sum"),
             n_tasks=("task_normalized", "nunique"))
        .reset_index()
    )
    out["pct_exposed"] = np.where(
        out["hours"] > 0, out["hours_exposed"] / out["hours"] * 100.0, 0.0
    ).clip(0.0, 100.0)
    out["pct_unexposed"] = 100.0 - out["pct_exposed"]

    anchor = family_usage_anchor(physical_mode, geo)
    rate = family_usage_rate(["dwa_title"], physical_mode, geo)
    out = _attach_usage(out, rate, ["dwa_title"], anchor)
    return out.sort_values("pct_exposed", ascending=False).reset_index(drop=True)
