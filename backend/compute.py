"""
compute.py — Core compute engine for the AEA Dashboard.
All @st.cache_data decorators replaced with simple in-process dict caches.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    ECO_BASELINE_FILE, ECO_2015_FILE, CROSSWALK_PATHS,
    AGG_LEVEL_COL, DATASETS, DATASET_SERIES, SORT_COL_MAP,
)

AEI_EXPLORER_DATASETS = ["AEI Conv. v1", "AEI Conv. v2", "AEI Conv. v3", "AEI Conv. v4", "AEI Conv. v5", "AEI API v3", "AEI API v4", "AEI API v5"]
EXPLORER_SOURCE_NAMES: list[str] = AEI_EXPLORER_DATASETS + ["MCP", "Microsoft"]


def get_explorer_source_names() -> list[str]:
    """Return the list of source names used in the explorer task lookup."""
    return list(EXPLORER_SOURCE_NAMES)


# ── Simple in-process caches ───────────────────────────────────────────────────
_crosswalk_cache: Optional[pd.DataFrame] = None
_eco_raw_cache:   Optional[pd.DataFrame] = None
_eco2015_raw_cache: Optional[pd.DataFrame] = None
_eco_baseline_cache: dict = {}
_dataset_cache:      dict = {}
_explorer_occ_base_cache: dict = {}          # keyed by frozenset|None
_explorer_task_cache: dict = {}
_wa_cache: dict = {}
_trends_cache: dict = {}
_explorer_task_lookup_cache: Optional[dict] = None
_explorer_groups_base_cache: dict = {}       # keyed by frozenset|None
_wa_explorer_geo_cache: dict = {}
_all_tasks_geo_cache: dict = {}
_all_eco_tasks_geo_cache: dict = {}
_top_mcps_cache: Optional[dict] = None
_task_changes_cache: dict = {}
_eco2015_baseline_set_cache: Optional[set] = None


def _find_crosswalk() -> Optional[Path]:
    for p in CROSSWALK_PATHS:
        if Path(p).exists():
            return Path(p)
    return None


def crosswalk_available() -> bool:
    return _find_crosswalk() is not None


def load_crosswalk() -> Optional[pd.DataFrame]:
    global _crosswalk_cache
    if _crosswalk_cache is None:
        path = _find_crosswalk()
        if path is None:
            return None
        _crosswalk_cache = pd.read_csv(path)
    return _crosswalk_cache


def load_eco_raw() -> Optional[pd.DataFrame]:
    global _eco_raw_cache
    if _eco_raw_cache is None:
        if not Path(ECO_BASELINE_FILE).exists():
            return None
        _eco_raw_cache = pd.read_csv(ECO_BASELINE_FILE)
    return _eco_raw_cache


def load_eco2015_raw() -> Optional[pd.DataFrame]:
    global _eco2015_raw_cache
    if _eco2015_raw_cache is None:
        if not Path(ECO_2015_FILE).exists():
            return None
        _eco2015_raw_cache = pd.read_csv(ECO_2015_FILE)
    return _eco2015_raw_cache


def dataset_exists(name: str) -> bool:
    meta = DATASETS.get(name)
    if meta is None:
        return False
    return Path(meta["file"]).exists()


def eco2015_available() -> bool:
    return Path(ECO_2015_FILE).exists()


# ── Task-level transformations ─────────────────────────────────────────────────

def apply_physical_filter(df: pd.DataFrame, physical_mode: str) -> pd.DataFrame:
    if "physical" not in df.columns or physical_mode == "all":
        return df
    if physical_mode == "exclude":
        return df[df["physical"] != True].copy()
    if physical_mode == "only":
        return df[df["physical"] == True].copy()
    return df


def compute_task_comp(
    df: pd.DataFrame,
    method: str,
    use_auto_aug: bool,
) -> pd.Series:
    if method == "freq":
        tc = df["freq_mean"].copy().fillna(0.0)
    else:
        tc = df["freq_mean"].fillna(0.0) * df["relevance"].fillna(0.0) * df["importance"].fillna(0.0)

    if use_auto_aug:
        if "auto_aug_mean" in df.columns:
            aug = df["auto_aug_mean"].fillna(0.0)
        else:
            aug = pd.Series(1.0, index=df.index)
        tc = tc * (aug / 5.0)

    return tc


def dedup_and_compute(
    df: pd.DataFrame,
    title_col: str,
    emp_col: str,
    wage_col: str,
    method: str,
    use_auto_aug: bool,
) -> pd.DataFrame:
    keep = [
        emp_col, wage_col,
        "broad_occ", "minor_occ_category", "major_occ_category",
        "freq_mean", "importance", "relevance", "auto_aug_mean",
    ]

    agg_dict = {c: "first" for c in keep if c in df.columns}
    deduped = (
        df.groupby([title_col, "task_normalized"], sort=False)
        .agg(agg_dict)
        .reset_index()
    )
    deduped["task_comp"] = compute_task_comp(deduped, method, use_auto_aug)
    return deduped


# ── ECO Baseline ───────────────────────────────────────────────────────────────

def load_eco_baseline(method: str, physical_mode: str, geo: str) -> Optional[pd.DataFrame]:
    key = (method, physical_mode, geo)
    if key in _eco_baseline_cache:
        return _eco_baseline_cache[key]

    eco = load_eco_raw()
    if eco is None:
        return None
    eco = apply_physical_filter(eco, physical_mode)
    if eco.empty:
        return None

    emp_col  = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"
    result = dedup_and_compute(eco, "title_current", emp_col, wage_col, method, False)
    _eco_baseline_cache[key] = result
    return result


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_results(
    ai_df:     pd.DataFrame,
    eco_df:    pd.DataFrame,
    title_col: str,
    agg_level: str,
    emp_col:   str,
    wage_col:  str,
) -> pd.DataFrame:
    group_col = AGG_LEVEL_COL[agg_level]

    ai_by_occ = (
        ai_df.groupby(title_col)["task_comp"]
        .sum()
        .reset_index()
        .rename(columns={"task_comp": "ai_task_comp"})
    )

    eco_agg: dict = {"eco_task_comp": ("task_comp", "sum")}
    for c in [emp_col, wage_col, "broad_occ", "minor_occ_category", "major_occ_category"]:
        if c in eco_df.columns:
            eco_agg[c] = (c, "first")

    eco_by_occ = eco_df.groupby("title_current").agg(**eco_agg).reset_index()

    occ = eco_by_occ.merge(
        ai_by_occ, left_on="title_current", right_on=title_col, how="left"
    )
    if title_col != "title_current" and title_col in occ.columns:
        occ.drop(columns=[title_col], inplace=True)

    occ["ai_task_comp"] = occ["ai_task_comp"].fillna(0.0)
    occ["pct_tasks_affected"] = (
        (occ["ai_task_comp"] / occ["eco_task_comp"].replace(0, np.nan)) * 100
    ).fillna(0.0).clip(upper=100.0)

    if emp_col in occ.columns:
        occ["workers_affected"] = occ["pct_tasks_affected"] / 100.0 * occ[emp_col]
        if wage_col in occ.columns:
            occ["wages_affected"] = (
                occ["pct_tasks_affected"] / 100.0 * occ[emp_col] * occ[wage_col]
            )
        else:
            occ["wages_affected"] = np.nan
    else:
        occ["workers_affected"] = np.nan
        occ["wages_affected"]   = np.nan

    if agg_level == "occupation":
        return occ[["title_current", "pct_tasks_affected",
                    "workers_affected", "wages_affected"]].copy()

    ai_by_group = (
        ai_df.groupby(group_col)["task_comp"]
        .sum().reset_index()
        .rename(columns={"task_comp": "ai_task_comp_group"})
    )
    eco_by_group = (
        eco_df.groupby(group_col)["task_comp"]
        .sum().reset_index()
        .rename(columns={"task_comp": "eco_task_comp_group"})
    )

    grp = eco_by_group.merge(ai_by_group, on=group_col, how="left")
    grp["ai_task_comp_group"] = grp["ai_task_comp_group"].fillna(0.0)
    grp["pct_tasks_affected"] = (
        (grp["ai_task_comp_group"] / grp["eco_task_comp_group"].replace(0, np.nan)) * 100
    ).fillna(0.0).clip(upper=100.0)

    if group_col in occ.columns:
        occ_by_group = (
            occ.groupby(group_col)
            .agg(workers_affected=("workers_affected", "sum"),
                 wages_affected=("wages_affected", "sum"))
            .reset_index()
        )
        grp = grp.merge(occ_by_group, on=group_col, how="left")
    else:
        grp["workers_affected"] = np.nan
        grp["wages_affected"]   = np.nan

    return grp[[group_col, "pct_tasks_affected", "workers_affected", "wages_affected"]].copy()


# ── Single-dataset compute ─────────────────────────────────────────────────────

def compute_single_dataset(
    file_path:     str,
    is_aei:        bool,
    method:        str,
    use_auto_aug:  bool,
    physical_mode: str,
    geo:           str,
    agg_level:     str,
) -> Optional[pd.DataFrame]:
    key = (file_path, is_aei, method, use_auto_aug, physical_mode, geo, agg_level)
    if key in _dataset_cache:
        return _dataset_cache[key]

    if not Path(file_path).exists():
        return None

    eco_deduped = load_eco_baseline(method, physical_mode, geo)
    if eco_deduped is None or eco_deduped.empty:
        return None

    df = pd.read_csv(file_path)
    df = apply_physical_filter(df, physical_mode)
    if df.empty:
        return None

    emp_col  = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"

    if is_aei:
        crosswalk = load_crosswalk()
        if crosswalk is None:
            return None

        ai_deduped = dedup_and_compute(
            df, "title", emp_col, wage_col, method, use_auto_aug
        )

        soc_lookup = df[["title", "soc_code_2010"]].drop_duplicates("title")
        ai_deduped = ai_deduped.merge(soc_lookup, on="title", how="left")

        ai_deduped = ai_deduped.merge(
            crosswalk[["O*NET-SOC 2010 Code", "O*NET-SOC 2019 Title"]],
            left_on="soc_code_2010", right_on="O*NET-SOC 2010 Code",
            how="left",
        )

        split_counts = (
            crosswalk.groupby("O*NET-SOC 2010 Code")["O*NET-SOC 2019 Title"]
            .nunique()
            .reset_index(name="split_count")
        )
        ai_deduped = ai_deduped.merge(
            split_counts,
            left_on="soc_code_2010", right_on="O*NET-SOC 2010 Code",
            how="left", suffixes=("", "_sc"),
        )
        ai_deduped.drop(
            columns=[c for c in ai_deduped.columns if c.endswith("_sc")],
            inplace=True,
        )
        ai_deduped["split_count"] = ai_deduped["split_count"].fillna(1.0)
        ai_deduped["task_comp"] /= ai_deduped["split_count"]
        if emp_col in ai_deduped.columns:
            ai_deduped[emp_col] /= ai_deduped["split_count"]

        agg_cols: dict = {"task_comp": "sum"}
        if emp_col in ai_deduped.columns:
            agg_cols[emp_col] = "sum"
        for c in ["broad_occ", "minor_occ_category", "major_occ_category", wage_col]:
            if c in ai_deduped.columns:
                agg_cols[c] = "first"

        ai_final = (
            ai_deduped
            .groupby(["O*NET-SOC 2019 Title", "task_normalized"], sort=False)
            .agg(agg_cols)
            .reset_index()
            .rename(columns={"O*NET-SOC 2019 Title": "title_current"})
        )

        eco_raw = load_eco_raw()
        if eco_raw is not None and "task_prop" in eco_raw.columns:
            tp = eco_raw[["title_current", "task_prop"]].drop_duplicates("title_current")
            ai_final = ai_final.merge(tp, on="title_current", how="left")
            ai_final["task_prop"] = ai_final["task_prop"].fillna(1.0).clip(lower=1.0)
            ai_final["task_comp"] /= ai_final["task_prop"]
            ai_final.drop(columns=["task_prop"], inplace=True)

        if eco_raw is not None:
            eco_groups = eco_raw[
                ["title_current", "broad_occ", "minor_occ_category", "major_occ_category"]
            ].drop_duplicates("title_current")
            for gc in ["broad_occ", "minor_occ_category", "major_occ_category"]:
                if gc in ai_final.columns:
                    mask = ai_final[gc].isna()
                    if mask.any():
                        fill = ai_final.loc[mask, ["title_current"]].merge(
                            eco_groups[["title_current", gc]],
                            on="title_current", how="left",
                        )
                        ai_final.loc[mask, gc] = fill[gc].values

        title_col_for_agg = "title_current"

    else:
        ai_final = dedup_and_compute(
            df, "title_current", emp_col, wage_col, method, use_auto_aug
        )
        title_col_for_agg = "title_current"

    result = aggregate_results(
        ai_final, eco_deduped, title_col_for_agg, agg_level, emp_col, wage_col
    )
    _dataset_cache[key] = result
    return result


# ── Multi-dataset combination ──────────────────────────────────────────────────

def combine_results(
    results:        list[Optional[pd.DataFrame]],
    group_col:      str,
    combine_method: str,
) -> pd.DataFrame:
    valid = [r for r in results if r is not None and not r.empty]
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1:
        return valid[0].copy()

    metric_cols = ["pct_tasks_affected", "workers_affected", "wages_affected"]

    renamed = []
    for i, r in enumerate(valid):
        sub = r[[group_col] + metric_cols].copy()
        sub = sub.rename(columns={mc: f"{mc}_{i}" for mc in metric_cols})
        renamed.append(sub)

    combined = renamed[0]
    for sub in renamed[1:]:
        combined = combined.merge(sub, on=group_col, how="outer")

    for mc in metric_cols:
        cols = [f"{mc}_{i}" for i in range(len(valid))]
        if combine_method == "Max":
            combined[mc] = combined[cols].max(axis=1)
        else:
            combined[mc] = combined[cols].mean(axis=1)

    return combined[[group_col] + metric_cols].copy()


# ── Group data (occupation-level overview) ─────────────────────────────────────

def get_group_data(settings: dict) -> Optional[dict]:
    """
    Returns a dict with:
      - df:               DataFrame of rows (descending order, top_n or search window)
      - group_col:        column name for the category
      - total_categories: total number of categories before top_n/search filter
      - total_emp:        sum of workers_affected across ALL categories
      - total_wages:      sum of wages_affected across ALL categories
      - matched_category: str or None (set when search_query was used)
    """
    selected = settings.get("selected_datasets", [])
    if not selected:
        return None

    method        = settings["method"]
    use_auto_aug  = settings["use_auto_aug"]
    physical_mode = settings["physical_mode"]
    geo           = settings["geo"]
    agg_level     = settings["agg_level"]
    sort_by       = settings["sort_by"]
    top_n         = int(settings["top_n"])
    combine       = settings.get("combine_method", "Average")
    search_query  = (settings.get("search_query") or "").strip()
    context_size  = int(settings.get("context_size") or 5)

    results = []
    for name in selected:
        meta = DATASETS.get(name)
        if meta is None:
            continue
        r = compute_single_dataset(
            file_path     = meta["file"],
            is_aei        = meta["is_aei"],
            method        = method,
            use_auto_aug  = use_auto_aug,
            physical_mode = physical_mode,
            geo           = geo,
            agg_level     = agg_level,
        )
        results.append(r)

    group_col = AGG_LEVEL_COL[agg_level]
    df = combine_results(results, group_col, combine)
    if df is None or df.empty:
        return None

    sort_col = SORT_COL_MAP.get(sort_by, "workers_affected")
    if sort_col not in df.columns:
        for c in ["workers_affected", "wages_affected", "pct_tasks_affected"]:
            if c in df.columns:
                sort_col = c
                break
        else:
            sort_col = group_col

    # Sort descending — highest value first (appears at TOP of horizontal bar chart)
    df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    # Compute ranks across ALL categories (rank 1 = highest value)
    for metric_col, rank_col in [
        ("workers_affected",   "rank_workers"),
        ("wages_affected",     "rank_wages"),
        ("pct_tasks_affected", "rank_pct"),
    ]:
        if metric_col in df.columns:
            df[rank_col] = df[metric_col].rank(ascending=False, method="min").astype(int)
        else:
            df[rank_col] = 0

    total_categories = len(df)
    total_emp   = float(df["workers_affected"].sum()) if "workers_affected" in df.columns else 0.0
    total_wages = float(df["wages_affected"].sum())   if "wages_affected"   in df.columns else 0.0

    # Apply search filter or top_n
    matched_category: Optional[str] = None
    if search_query:
        q = search_query.lower()
        mask = df[group_col].str.lower().str.contains(q, na=False, regex=False)
        if mask.any():
            pos = int(mask.idxmax())
            matched_category = str(df.loc[pos, group_col])
            start = max(0, pos - context_size)
            end   = min(len(df), pos + context_size + 1)
            df = df.iloc[start:end].copy()
        else:
            df = df.iloc[0:0].copy()  # empty — no match
    else:
        df = df.head(top_n).copy()

    return {
        "df":               df,
        "group_col":        group_col,
        "total_categories": total_categories,
        "total_emp":        total_emp,
        "total_wages":      total_wages,
        "matched_category": matched_category,
    }


# ── Work Activities (DWA / IWA / GWA) ─────────────────────────────────────────

def _combine_activity_dfs(
    frames: list[pd.DataFrame],
    cat_col: str,
    combine_method: str,
) -> pd.DataFrame:
    """Combine multiple activity-level DataFrames via average or max."""
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].copy()

    metric_cols = ["pct_tasks_affected", "workers_affected", "wages_affected"]
    renamed = []
    for i, f in enumerate(frames):
        sub = f[[cat_col] + [c for c in metric_cols if c in f.columns]].copy()
        sub = sub.rename(columns={mc: f"{mc}_{i}" for mc in metric_cols if mc in f.columns})
        renamed.append(sub)

    combined = renamed[0]
    for sub in renamed[1:]:
        combined = combined.merge(sub, on=cat_col, how="outer")

    for mc in metric_cols:
        cols = [f"{mc}_{i}" for i in range(len(frames)) if f"{mc}_{i}" in combined.columns]
        if not cols:
            continue
        if combine_method == "Max":
            combined[mc] = combined[cols].max(axis=1)
        else:
            combined[mc] = combined[cols].mean(axis=1)

    keep = [cat_col] + [mc for mc in metric_cols if mc in combined.columns]
    return combined[keep].copy()


def _compute_wa_for_group(
    dataset_names: list[str],
    settings: dict,
    use_eco2015: bool,
) -> Optional[dict]:
    """
    Compute DWA/IWA/GWA metrics for a group of datasets sharing the same SOC taxonomy.
    Returns dict with keys "gwa", "iwa", "dwa" → list of activity row dicts.

    Dedup strategy:
    - n_tasks_per_occ uses (title, task_normalized) dedup — for emp_per_task allocation
    - Each activity level uses (title, task_normalized, act_col) dedup — preserves all
      DWA/IWA/GWA associations a task may have (a task can map to multiple DWAs)
    """
    method        = settings["method"]
    use_auto_aug  = settings["use_auto_aug"]
    physical_mode = settings["physical_mode"]
    geo           = settings["geo"]
    combine       = settings.get("combine_method", "Average")
    top_n         = int(settings.get("top_n", 20))
    sort_by       = settings.get("sort_by", "workers_affected")

    emp_col  = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"

    # -- Load ECO baseline
    eco_raw = load_eco2015_raw() if use_eco2015 else load_eco_raw()
    if eco_raw is None:
        return None

    eco = apply_physical_filter(eco_raw, physical_mode)
    if eco.empty:
        return None

    title_col = "title" if use_eco2015 else "title_current"

    activity_cols = {
        "gwa": "gwa_title",
        "iwa": "iwa_title",
        "dwa": "dwa_title",
    }

    # -- Emp weighting per occ: freq-based or freq×rel×imp-based instead of equal split
    eco_task_dedup = (
        eco.groupby([title_col, "task_normalized"], sort=False)
        .first()
        .reset_index()
    )

    # Compute per-task weight for emp allocation
    if method == "freq":
        eco_task_dedup["_emp_weight"] = eco_task_dedup["freq_mean"].fillna(0.0)
    else:
        eco_task_dedup["_emp_weight"] = (
            eco_task_dedup["freq_mean"].fillna(0.0)
            * eco_task_dedup["relevance"].fillna(0.0)
            * eco_task_dedup["importance"].fillna(0.0)
        )
    # Normalised weight per occ: weight / sum(weight in occ)
    eco_task_dedup["_emp_weight_sum"] = eco_task_dedup.groupby(title_col)["_emp_weight"].transform("sum")
    eco_task_dedup["_emp_frac"] = (
        eco_task_dedup["_emp_weight"] / eco_task_dedup["_emp_weight_sum"].replace(0, np.nan)
    ).fillna(0.0)

    emp_frac_lookup = eco_task_dedup[[title_col, "task_normalized", "_emp_frac"]].copy()

    # Employment / wage per occupation (from raw, first occurrence)
    occ_emp_wage = (
        eco.groupby(title_col)[[emp_col, wage_col]]
        .first()
        .reset_index()
    )

    # -- Pre-compute eco_for_act per activity level
    # Each uses (title, task_normalized, act_col) dedup to preserve all DWA/IWA/GWA associations
    eco_for_acts: dict[str, pd.DataFrame] = {}
    for act_key, act_col in activity_cols.items():
        if act_col not in eco.columns:
            continue
        eco_for_act = (
            eco.groupby([title_col, "task_normalized", act_col], sort=False)
            .first()
            .reset_index()
        )
        eco_for_act["eco_tc"] = compute_task_comp(eco_for_act, method, False)
        eco_for_act = eco_for_act.merge(emp_frac_lookup, on=[title_col, "task_normalized"], how="left")
        eco_for_act = eco_for_act.merge(occ_emp_wage, on=title_col, how="left", suffixes=("", "_ow"))
        for col in [emp_col, wage_col]:
            ow = f"{col}_ow"
            if ow in eco_for_act.columns:
                eco_for_act[col] = eco_for_act[ow].fillna(eco_for_act[col])
                eco_for_act.drop(columns=[ow], inplace=True)
        eco_for_act["emp_per_task"] = (eco_for_act["_emp_frac"] * eco_for_act[emp_col]).fillna(0.0)
        eco_for_acts[act_key] = eco_for_act

    # -- Process each AI dataset
    per_dataset: dict[str, list] = {}

    for name in dataset_names:
        meta = DATASETS.get(name)
        if meta is None or not Path(meta["file"]).exists():
            continue

        ai_raw = pd.read_csv(meta["file"])
        ai_raw = apply_physical_filter(ai_raw, physical_mode)
        if ai_raw.empty:
            continue

        ai_title_col  = "title" if meta["is_aei"] else "title_current"

        for act_key, act_col in activity_cols.items():
            if act_key not in eco_for_acts:
                continue
            if act_col not in ai_raw.columns:
                continue

            eco_for_act = eco_for_acts[act_key]

            # Dedup AI on (ai_title, task_normalized, act_col) to match eco dedup
            ai_for_act = (
                ai_raw.groupby([ai_title_col, "task_normalized", act_col], sort=False)
                .first()
                .reset_index()
            )
            ai_for_act["ai_tc"] = compute_task_comp(ai_for_act, method, use_auto_aug)

            merged = eco_for_act.merge(
                ai_for_act[[ai_title_col, "task_normalized", act_col, "ai_tc"]].rename(
                    columns={ai_title_col: title_col}
                ),
                on=[title_col, "task_normalized", act_col],
                how="left",
            )
            merged["ai_tc"] = merged["ai_tc"].fillna(0.0)

            eco_tc_safe = merged["eco_tc"].replace(0, np.nan)
            merged["workers_contribution"] = (
                (merged["ai_tc"] / eco_tc_safe) * merged["emp_per_task"]
            ).fillna(0.0).clip(lower=0)
            merged["wages_contribution"] = (
                merged["workers_contribution"] * merged[wage_col].fillna(0.0)
            )

            by_act = (
                merged.groupby(act_col)
                .agg(
                    ai_tc_sum=("ai_tc", "sum"),
                    eco_tc_sum=("eco_tc", "sum"),
                    workers_affected=("workers_contribution", "sum"),
                    wages_affected=("wages_contribution", "sum"),
                )
                .reset_index()
                .rename(columns={act_col: "category"})
            )
            by_act["pct_tasks_affected"] = (
                by_act["ai_tc_sum"] / by_act["eco_tc_sum"].replace(0, np.nan) * 100
            ).fillna(0.0).clip(upper=100.0)
            by_act = by_act[["category", "pct_tasks_affected", "workers_affected", "wages_affected"]]

            per_dataset.setdefault(act_key, []).append(by_act)

    if not per_dataset:
        return None

    # -- Combine across datasets and sort descending (highest at top of chart)
    combined: dict = {}
    sort_col_map = {
        "Workers Affected":   "workers_affected",
        "Wages Affected":     "wages_affected",
        "% Tasks Affected":   "pct_tasks_affected",
        "workers_affected":   "workers_affected",
        "wages_affected":     "wages_affected",
        "pct_tasks_affected": "pct_tasks_affected",
    }
    sort_col = sort_col_map.get(sort_by, "workers_affected")

    for act_key, frames in per_dataset.items():
        df = _combine_activity_dfs(frames, "category", combine)
        if df.empty:
            continue
        sc = sort_col if sort_col in df.columns else "pct_tasks_affected"
        # Sort descending — highest at top of chart
        df = (
            df
            .sort_values(sc, ascending=False, na_position="last")
            .head(top_n)
            .reset_index(drop=True)
        )
        combined[act_key] = df.to_dict(orient="records")

    return combined if combined else None


def compute_work_activities(settings: dict) -> dict:
    """
    Splits selected datasets into AEI group (eco_2015 baseline) and
    MCP/Microsoft group (eco_2025 baseline), then computes DWA/IWA/GWA metrics.
    Returns {"aei_group": {...}, "mcp_group": {...}}.
    """
    selected = settings.get("selected_datasets", [])

    aei_datasets    = [d for d in selected if DATASETS.get(d, {}).get("is_aei")]
    mcp_ms_datasets = [d for d in selected if d in DATASETS and not DATASETS[d]["is_aei"]]

    result: dict = {}

    if aei_datasets:
        aei_result = _compute_wa_for_group(aei_datasets, settings, use_eco2015=True)
        if aei_result:
            result["aei_group"] = {"datasets": aei_datasets, **aei_result}

    if mcp_ms_datasets:
        mcp_result = _compute_wa_for_group(mcp_ms_datasets, settings, use_eco2015=False)
        if mcp_result:
            result["mcp_group"] = {"datasets": mcp_ms_datasets, **mcp_result}

    return result


# ── Time Trends ────────────────────────────────────────────────────────────────

def _get_dataset_date(file_path: str) -> str:
    try:
        row = pd.read_csv(file_path, nrows=1)
        if "date" in row.columns:
            return str(row["date"].iloc[0])
    except Exception:
        pass
    return ""


def _safe_num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def compute_trends(settings: dict) -> dict:
    """
    For each requested series (AEI, MCP, etc.), runs compute_single_dataset for
    every version and returns time-series data grouped by date.
    """
    series_names  = settings.get("series", ["AEI", "MCP"])
    method        = settings["method"]
    use_auto_aug  = settings["use_auto_aug"]
    physical_mode = settings["physical_mode"]
    geo           = settings["geo"]
    agg_level     = settings["agg_level"]
    top_n         = int(settings.get("top_n", 10))
    sort_by       = settings.get("sort_by", "Workers Affected")

    group_col = AGG_LEVEL_COL[agg_level]
    sort_col  = SORT_COL_MAP.get(sort_by, "workers_affected")

    result_series = []

    for series_name in series_names:
        ds_list = DATASET_SERIES.get(series_name, [])
        series_data = []
        latest_categories: list[str] = []

        for ds_name in ds_list:
            meta = DATASETS.get(ds_name)
            if meta is None or not Path(meta["file"]).exists():
                continue

            date = _get_dataset_date(meta["file"])

            df = compute_single_dataset(
                file_path     = meta["file"],
                is_aei        = meta["is_aei"],
                method        = method,
                use_auto_aug  = use_auto_aug,
                physical_mode = physical_mode,
                geo           = geo,
                agg_level     = agg_level,
            )

            if df is None or df.empty:
                continue

            sc = sort_col if sort_col in df.columns else group_col
            top_df = (
                df
                .sort_values(sc, ascending=False, na_position="last")
                .head(top_n)
            )
            latest_categories = top_df[group_col].tolist()

            rows = []
            for _, row in df.iterrows():
                rows.append({
                    "category":           str(row[group_col]),
                    "pct_tasks_affected": _safe_num(row.get("pct_tasks_affected", 0)),
                    "workers_affected":   _safe_num(row.get("workers_affected", 0)),
                    "wages_affected":     _safe_num(row.get("wages_affected", 0)),
                })

            series_data.append({
                "dataset": ds_name,
                "date":    date,
                "rows":    rows,
            })

        if series_data:
            result_series.append({
                "name":              series_name,
                "data_points":       series_data,
                "top_categories":    latest_categories,
                "group_col":         group_col,
            })

    return {"series": result_series}


# ── Work Activity Trends ───────────────────────────────────────────────────────

def compute_wa_trends(settings: dict) -> dict:
    """
    Computes work-activity time trends per sub_type series.
    Series names are sub_type keys from DATASET_SERIES (e.g. "AEI Conv.", "MCP").
    AEI sub_types (keys starting with "AEI") use eco_2015 baseline; others use eco_2025.
    Returns {"series": [{"name": ..., "data_points": [...], "top_categories": [...]}]}
    """
    series_names  = settings.get("series", [])
    method        = settings["method"]
    use_auto_aug  = settings["use_auto_aug"]
    physical_mode = settings["physical_mode"]
    geo           = settings["geo"]
    top_n         = int(settings.get("top_n", 10))
    sort_by       = settings.get("sort_by", "Workers Affected")
    activity_level = settings.get("activity_level", "gwa")  # gwa | iwa | dwa

    result_series = []

    for series_name in series_names:
        ds_list = DATASET_SERIES.get(series_name, [])
        if not ds_list:
            continue
        # AEI sub_types use eco_2015 baseline; others use eco_2025
        use_eco2015 = series_name.startswith("AEI")

        series_data = []
        latest_categories: list[str] = []

        for ds_name in ds_list:
            meta = DATASETS.get(ds_name)
            if meta is None or not Path(meta["file"]).exists():
                continue

            date = _get_dataset_date(meta["file"])

            wa_settings = {
                "method":        method,
                "use_auto_aug":  use_auto_aug,
                "physical_mode": physical_mode,
                "geo":           geo,
                "combine_method": "Average",
                "top_n":         top_n,
                "sort_by":       sort_by,
            }

            wa_result = _compute_wa_for_group([ds_name], wa_settings, use_eco2015)
            if wa_result is None or activity_level not in wa_result:
                continue

            rows_raw = wa_result[activity_level]
            # rows_raw is already sorted descending, top_n already applied
            latest_categories = [r["category"] for r in rows_raw]

            # For time series we want ALL categories, not just top_n
            # Re-run without top_n limit
            wa_settings_all = dict(wa_settings)
            wa_settings_all["top_n"] = 9999
            wa_result_all = _compute_wa_for_group([ds_name], wa_settings_all, use_eco2015)
            rows_all = wa_result_all.get(activity_level, []) if wa_result_all else rows_raw

            series_data.append({
                "dataset": ds_name,
                "date":    date,
                "rows": [
                    {
                        "category":           str(r["category"]),
                        "pct_tasks_affected": _safe_num(r.get("pct_tasks_affected", 0)),
                        "workers_affected":   _safe_num(r.get("workers_affected", 0)),
                        "wages_affected":     _safe_num(r.get("wages_affected", 0)),
                    }
                    for r in rows_all
                ],
            })

        if series_data:
            result_series.append({
                "name":           series_name,
                "data_points":    series_data,
                "top_categories": latest_categories,
                "group_col":      activity_level,
            })

    return {"series": result_series}


# ── Explorer ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> float:
    f = _safe_num(v)
    return 0.0 if f is None else f


def _build_explorer_occ_base(selected_sources: Optional[frozenset] = None) -> list:
    """
    Build and cache the geo-independent base data for explorer occupations.
    Returns a list of dicts with hierarchy, dws, task counts, and AI metrics
    (but NO emp/wage — those are overlaid per geo).
    Cache is keyed by selected_sources (frozenset or None).
    """
    global _explorer_occ_base_cache
    cache_key = selected_sources
    if cache_key in _explorer_occ_base_cache:
        return _explorer_occ_base_cache[cache_key]

    eco = load_eco_raw()
    if eco is None:
        return []

    lookup = _build_explorer_task_lookup()

    # Unique (title, task_norm) pairs with physical flag
    def _phys_bool(v):
        if v is None: return False
        if isinstance(v, float) and np.isnan(v): return False
        return bool(v)

    task_pairs = eco[["title_current", "task_normalized", "physical"]].drop_duplicates(
        subset=["title_current", "task_normalized"]
    ).copy()
    task_pairs["physical_bool"] = task_pairs["physical"].apply(_phys_bool)

    # Build occ → task list and physical count
    occ_task_map: dict = {}
    occ_physical_map: dict = {}
    for _, row in task_pairs.iterrows():
        title = row["title_current"]
        tn = row["task_normalized"]
        if title not in occ_task_map:
            occ_task_map[title] = []
            occ_physical_map[title] = 0
        occ_task_map[title].append(tn)
        if row["physical_bool"]:
            occ_physical_map[title] += 1

    # Basic occ stats (geo-independent)
    agg_dict = {
        "major":    ("major_occ_category", "first"),
        "minor":    ("minor_occ_category", "first"),
        "broad":    ("broad_occ",           "first"),
    }
    if "dws_star_rating" in eco.columns:
        agg_dict["dws_star_rating"] = ("dws_star_rating", "first")
    if "job_zone" in eco.columns:
        agg_dict["job_zone"] = ("job_zone", "first")
    occ_stats = eco.groupby("title_current").agg(**agg_dict).reset_index()

    result = []
    for _, row in occ_stats.iterrows():
        title = row["title_current"]
        task_norms = occ_task_map.get(title, [])
        n_tasks = len(task_norms)
        n_phys  = occ_physical_map.get(title, 0)

        metrics = _compute_task_metrics(task_norms, lookup, selected_sources)

        dws_val = _safe_num(row.get("dws_star_rating")) if "dws_star_rating" in row.index else None
        jz_val = _safe_num(row.get("job_zone")) if "job_zone" in row.index else None

        occ_dict: dict = {
            "title_current": title,
            "major":   row.get("major"),
            "minor":   row.get("minor"),
            "broad":   row.get("broad"),
            "dws_star_rating": round(dws_val, 1) if dws_val is not None else None,
            "job_zone": round(jz_val, 1) if jz_val is not None else None,
            "n_tasks":  n_tasks,
            "n_physical_tasks": n_phys,
            "pct_physical": round(n_phys / n_tasks, 4) if n_tasks else None,
            **metrics,
        }
        result.append(occ_dict)

    _explorer_occ_base_cache[cache_key] = result
    return result


def get_explorer_occupations(geo: str = "nat", selected_sources: Optional[frozenset] = None) -> list:
    """
    Returns list of all occupations from eco_2025 with hierarchy, employment,
    wage stats, and AI metrics from all AEI versions, MCP v4, and Microsoft.
    Base data is cached; emp/wage are overlaid per geo parameter.
    When selected_sources is provided, only those sources contribute to metrics.
    """
    base = _build_explorer_occ_base(selected_sources)
    if not base:
        return []

    eco = load_eco_raw()
    if eco is None:
        return base

    # Resolve geo columns with fallback to national
    emp_col = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"
    if emp_col not in eco.columns:
        emp_col = "emp_tot_nat_2025"
    if wage_col not in eco.columns:
        wage_col = "a_med_nat_2025"

    # Build occ→emp/wage lookup for this geo
    occ_emp_wage = eco.groupby("title_current").agg(
        _emp=(emp_col, "first"),
        _wage=(wage_col, "first"),
    ).reset_index()
    occ_to_emp: dict = dict(zip(occ_emp_wage["title_current"], occ_emp_wage["_emp"]))
    occ_to_wage: dict = dict(zip(occ_emp_wage["title_current"], occ_emp_wage["_wage"]))

    # Overlay emp/wage onto each base row
    result = []
    for occ_dict in base:
        row = dict(occ_dict)
        title = row["title_current"]
        row["emp"] = _safe_num(occ_to_emp.get(title))
        row["wage"] = _safe_num(occ_to_wage.get(title))
        result.append(row)

    return result


def get_occupation_tasks(title: str) -> Optional[dict]:
    """
    Returns task-level details for one occupation, cross-referenced with
    all AEI versions, MCP v4, and Microsoft by task_normalized.
    Tasks are sorted alphabetically by the `task` column.
    pct_norm values are stored as-is from the CSV (already in % form, 0-100 range).
    """
    if title in _explorer_task_cache:
        return _explorer_task_cache[title]

    eco = load_eco_raw()
    if eco is None:
        return None

    occ_tasks = eco[eco["title_current"] == title].copy()
    if occ_tasks.empty:
        return None

    lookup = _build_explorer_task_lookup()
    top_mcps_lookup = _build_top_mcps_lookup()

    # Sort by task name alphabetically
    occ_tasks = occ_tasks.sort_values("task")

    tasks_list = []
    seen_task_norm: set = set()

    for _, row in occ_tasks.iterrows():
        tn = row["task_normalized"]
        if tn in seen_task_norm:
            continue
        seen_task_norm.add(tn)

        sources = dict(lookup.get(tn, {}))

        # Avg/max across all sources
        auto_vals = [v["auto_aug"] for v in sources.values() if v.get("auto_aug") is not None]
        pct_vals  = [v["pct_norm"]  for v in sources.values() if v.get("pct_norm")  is not None]

        phys_val = row.get("physical")
        if phys_val is None or (isinstance(phys_val, float) and np.isnan(phys_val)):
            is_physical = None
        else:
            is_physical = bool(phys_val)

        tasks_list.append({
            "task":           row["task"],
            "task_normalized": tn,
            "dwa_title":      row.get("dwa_title"),
            "iwa_title":      row.get("iwa_title"),
            "gwa_title":      row.get("gwa_title"),
            "freq_mean":      _safe_num(row.get("freq_mean")),
            "importance":     _safe_num(row.get("importance")),
            "relevance":      _safe_num(row.get("relevance")),
            "physical":       is_physical,
            "sources":        sources,
            "avg_auto_aug":   round(sum(auto_vals) / len(auto_vals), 3) if auto_vals else None,
            "max_auto_aug":   round(max(auto_vals), 3)                  if auto_vals else None,
            "avg_pct_norm":   round(sum(pct_vals) / len(pct_vals), 4)  if pct_vals  else None,
            "max_pct_norm":   round(max(pct_vals), 4)                  if pct_vals  else None,
            "top_mcps":       top_mcps_lookup.get(tn, []),
        })

    result = {"title": title, "tasks": tasks_list}
    _explorer_task_cache[title] = result
    return result


def _build_explorer_task_lookup() -> dict:
    """
    Builds and caches a dict: task_normalized -> {source_name: {"auto_aug": float|None, "pct_norm": float|None}}
    Sources: all AEI versions (using auto_aug_mean), MCP v4 (auto_aug_mean), Microsoft (auto_aug_mean).
    """
    global _explorer_task_lookup_cache
    if _explorer_task_lookup_cache is not None:
        return _explorer_task_lookup_cache

    result: dict = {}

    for ds_name in AEI_EXPLORER_DATASETS:
        meta = DATASETS.get(ds_name, {})
        fpath = meta.get("file", "")
        if not Path(fpath).exists():
            continue
        df = pd.read_csv(fpath)
        if "task_normalized" not in df.columns:
            continue
        if "auto_aug_mean" not in df.columns:
            df["auto_aug_mean"] = np.nan
        if "pct_normalized" not in df.columns:
            df["pct_normalized"] = np.nan
        df["auto_aug_mean"] = pd.to_numeric(df["auto_aug_mean"], errors="coerce")
        df["pct_normalized"] = pd.to_numeric(df["pct_normalized"], errors="coerce")
        agg = df.groupby("task_normalized", sort=False).agg(
            auto_aug=("auto_aug_mean", "mean"),
            pct_norm=("pct_normalized", "mean"),
        ).reset_index()
        for _, row in agg.iterrows():
            tn = row["task_normalized"]
            if tn not in result:
                result[tn] = {}
            result[tn][ds_name] = {
                "auto_aug": _safe_num(row["auto_aug"]),
                "pct_norm": _safe_num(row["pct_norm"]),
            }

    # MCP Cumul. v4
    mcp_meta = DATASETS.get("MCP Cumul. v4", {})
    mcp_file = mcp_meta.get("file", "")
    if Path(mcp_file).exists():
        mcp = pd.read_csv(mcp_file)
        if "auto_aug_mean" not in mcp.columns:
            mcp["auto_aug_mean"] = np.nan
        if "pct_normalized" not in mcp.columns:
            mcp["pct_normalized"] = np.nan
        mcp["auto_aug_mean"] = pd.to_numeric(mcp["auto_aug_mean"], errors="coerce")
        mcp["pct_normalized"] = pd.to_numeric(mcp["pct_normalized"], errors="coerce")
        if "task_normalized" in mcp.columns:
            agg = mcp.groupby("task_normalized", sort=False).agg(
                auto_aug=("auto_aug_mean", "mean"),
                pct_norm=("pct_normalized", "mean"),
            ).reset_index()
            for _, row in agg.iterrows():
                tn = row["task_normalized"]
                if tn not in result:
                    result[tn] = {}
                result[tn]["MCP"] = {
                    "auto_aug": _safe_num(row["auto_aug"]),
                    "pct_norm": _safe_num(row["pct_norm"]),
                }

    # Microsoft
    ms_meta = DATASETS.get("Microsoft", {})
    ms_file = ms_meta.get("file", "")
    if Path(ms_file).exists():
        ms = pd.read_csv(ms_file)
        if "auto_aug_mean" not in ms.columns:
            ms["auto_aug_mean"] = np.nan
        if "pct_normalized" not in ms.columns:
            ms["pct_normalized"] = np.nan
        ms["auto_aug_mean"] = pd.to_numeric(ms["auto_aug_mean"], errors="coerce")
        ms["pct_normalized"] = pd.to_numeric(ms["pct_normalized"], errors="coerce")
        if "task_normalized" in ms.columns:
            agg = ms.groupby("task_normalized", sort=False).agg(
                auto_aug=("auto_aug_mean", "mean"),
                pct_norm=("pct_normalized", "mean"),
            ).reset_index()
            for _, row in agg.iterrows():
                tn = row["task_normalized"]
                if tn not in result:
                    result[tn] = {}
                result[tn]["Microsoft"] = {
                    "auto_aug": _safe_num(row["auto_aug"]),
                    "pct_norm": _safe_num(row["pct_norm"]),
                }

    _explorer_task_lookup_cache = result
    return result


def _build_top_mcps_lookup() -> dict:
    """
    Builds and caches a dict: task_normalized -> list of {title, rating, url}.
    Source: MCP Cumul. v4 `top_mcps` (pipe-delimited "Name (rating)") and `top_mcp_urls` (pipe-delimited URLs).
    Returns up to 5 entries per task.
    """
    global _top_mcps_cache
    if _top_mcps_cache is not None:
        return _top_mcps_cache

    result: dict = {}
    mcp_meta = DATASETS.get("MCP Cumul. v4", {})
    mcp_file = mcp_meta.get("file", "")
    if not Path(mcp_file).exists():
        _top_mcps_cache = result
        return result

    mcp = pd.read_csv(mcp_file)
    if "top_mcps" not in mcp.columns or "top_mcp_urls" not in mcp.columns:
        _top_mcps_cache = result
        return result
    if "task_normalized" not in mcp.columns:
        _top_mcps_cache = result
        return result

    for _, row in mcp.iterrows():
        tn = row.get("task_normalized")
        if pd.isna(tn) or tn in result:
            continue
        raw_mcps = row.get("top_mcps", "")
        raw_urls = row.get("top_mcp_urls", "")
        if pd.isna(raw_mcps) or not raw_mcps:
            continue

        titles_raw = [s.strip() for s in str(raw_mcps).split("||")]
        urls_raw = [s.strip() for s in str(raw_urls).split("||")] if not pd.isna(raw_urls) else []

        entries: list[dict] = []
        for i, title_str in enumerate(titles_raw[:5]):
            # Parse "Name (rating)" format
            m = re.match(r"^(.+?)\s*\((\d+(?:\.\d+)?)\)\s*$", title_str)
            if m:
                name = m.group(1).strip()
                rating = float(m.group(2))
            else:
                name = title_str
                rating = None
            url = urls_raw[i] if i < len(urls_raw) else None
            entries.append({"title": name, "rating": rating, "url": url})
        result[tn] = entries

    _top_mcps_cache = result
    return result


def _compute_task_metrics(task_norms: list, lookup: dict, selected_sources: Optional[frozenset] = None) -> dict:
    """
    Given a list of unique task_normalized values and the lookup, compute 10 metrics:
    - auto_avg_with_vals: avg of (per-task avg-across-sources), only tasks with >=1 source
    - auto_max_with_vals: avg of (per-task max-across-sources), only tasks with >=1 source
    - auto_avg_all:       sum of (per-task avg, nulls=0) / total_n_tasks
    - auto_max_all:       sum of (per-task max, nulls=0) / total_n_tasks
    - pct_avg_with_vals, pct_max_with_vals, pct_avg_all, pct_max_all  (same pattern)
    - sum_pct_avg:        sum of per-task pct_avg (only tasks with pct values)
    - sum_pct_max:        sum of per-task pct_max (only tasks with pct values)

    When selected_sources is provided (a frozenset of source names), only those
    sources contribute to the metrics. When None, all sources are used.
    """
    total_n = len(task_norms)
    auto_avgs_with: list = []
    auto_maxs_with: list = []
    auto_avgs_sum = 0.0
    auto_maxs_sum = 0.0

    pct_avgs_with: list = []
    pct_maxs_with: list = []
    pct_avgs_sum = 0.0
    pct_maxs_sum = 0.0

    for tn in task_norms:
        all_sources = lookup.get(tn, {})
        if selected_sources is not None:
            sources = {k: v for k, v in all_sources.items() if k in selected_sources}
        else:
            sources = all_sources

        auto_vals = [v["auto_aug"] for v in sources.values() if v.get("auto_aug") is not None]
        if auto_vals:
            t_avg = sum(auto_vals) / len(auto_vals)
            t_max = max(auto_vals)
            auto_avgs_with.append(t_avg)
            auto_maxs_with.append(t_max)
            auto_avgs_sum += t_avg
            auto_maxs_sum += t_max

        pct_vals = [v["pct_norm"] for v in sources.values() if v.get("pct_norm") is not None]
        if pct_vals:
            t_pct_avg = sum(pct_vals) / len(pct_vals)
            t_pct_max = max(pct_vals)
            pct_avgs_with.append(t_pct_avg)
            pct_maxs_with.append(t_pct_max)
            pct_avgs_sum += t_pct_avg
            pct_maxs_sum += t_pct_max

    n_with_auto = len(auto_avgs_with)
    n_with_pct  = len(pct_avgs_with)

    def _r3(v): return round(v, 3) if v is not None else None
    def _r4(v): return round(v, 4) if v is not None else None

    return {
        "auto_avg_with_vals": _r3(sum(auto_avgs_with) / n_with_auto) if n_with_auto else None,
        "auto_max_with_vals": _r3(sum(auto_maxs_with) / n_with_auto) if n_with_auto else None,
        "auto_avg_all":       _r3(auto_avgs_sum / total_n) if total_n else None,
        "auto_max_all":       _r3(auto_maxs_sum / total_n) if total_n else None,
        "pct_avg_with_vals":  _r4(sum(pct_avgs_with) / n_with_pct)  if n_with_pct  else None,
        "pct_max_with_vals":  _r4(sum(pct_maxs_with) / n_with_pct)  if n_with_pct  else None,
        "pct_avg_all":        _r4(pct_avgs_sum / total_n) if total_n else None,
        "pct_max_all":        _r4(pct_maxs_sum / total_n) if total_n else None,
        "sum_pct_avg":        _r4(pct_avgs_sum) if pct_avgs_with else None,
        "sum_pct_max":        _r4(pct_maxs_sum) if pct_maxs_with else None,
    }


def _build_explorer_groups_base(selected_sources: Optional[frozenset] = None) -> dict:
    """
    Build and cache the geo-independent base data for explorer groups.
    Returns dict with 'major', 'minor', 'broad' keys, each a list of dicts
    containing hierarchy, metrics, task counts, physical counts, parent info, dws, n_occs,
    and an '_occs' field with the set of occupation titles in each group (for geo overlay).
    Cache is keyed by selected_sources (frozenset or None).
    """
    global _explorer_groups_base_cache
    cache_key = selected_sources
    if cache_key in _explorer_groups_base_cache:
        return _explorer_groups_base_cache[cache_key]

    eco = load_eco_raw()
    if eco is None:
        return {"major": [], "minor": [], "broad": []}

    lookup = _build_explorer_task_lookup()

    # Basic occ stats (geo-independent)
    group_agg_dict = {
        "major":    ("major_occ_category", "first"),
        "minor":    ("minor_occ_category", "first"),
        "broad":    ("broad_occ",          "first"),
    }
    if "dws_star_rating" in eco.columns:
        group_agg_dict["dws_star_rating"] = ("dws_star_rating", "first")
    if "job_zone" in eco.columns:
        group_agg_dict["job_zone"] = ("job_zone", "first")
    occ_basic = eco.groupby("title_current").agg(**group_agg_dict).reset_index()

    # Build occ->dws mapping for group averaging
    occ_to_dws: dict = {}
    if "dws_star_rating" in occ_basic.columns:
        for _, r in occ_basic.iterrows():
            v = _safe_num(r.get("dws_star_rating"))
            if v is not None:
                occ_to_dws[r["title_current"]] = v

    # Build occ->job_zone mapping for group averaging
    occ_to_jz: dict = {}
    if "job_zone" in occ_basic.columns:
        for _, r in occ_basic.iterrows():
            v = _safe_num(r.get("job_zone"))
            if v is not None:
                occ_to_jz[r["title_current"]] = v

    # Unique (title, task_norm) pairs with physical flag
    task_pairs = eco[["title_current", "task_normalized", "physical"]].drop_duplicates(
        subset=["title_current", "task_normalized"]
    ).copy()

    def _phys_bool(v):
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        return bool(v)

    task_pairs["physical_bool"] = task_pairs["physical"].apply(_phys_bool)

    # Build occ→level mappings
    occ_to_major = dict(zip(occ_basic["title_current"], occ_basic["major"]))
    occ_to_minor = dict(zip(occ_basic["title_current"], occ_basic["minor"]))
    occ_to_broad = dict(zip(occ_basic["title_current"], occ_basic["broad"]))

    result: dict = {}

    level_configs = [
        ("major", occ_to_major, None,       None),
        ("minor", occ_to_minor, occ_to_major, None),
        ("broad", occ_to_broad, occ_to_minor, occ_to_major),
    ]

    for level_key, occ_to_level, occ_to_parent, occ_to_grandparent in level_configs:
        from collections import defaultdict
        level_to_occs: dict = defaultdict(set)
        for title in occ_basic["title_current"]:
            lv = occ_to_level.get(title) or "Unknown"
            level_to_occs[lv].add(title)

        groups_data = []
        for group_name in sorted(level_to_occs.keys()):
            occs = level_to_occs[group_name]

            # Unique task_norms for this group
            group_tasks = task_pairs[task_pairs["title_current"].isin(occs)]
            unique_task_norms = group_tasks["task_normalized"].unique().tolist()
            n_tasks = len(unique_task_norms)

            # Physical: count unique task_norms that are physical
            phys_by_task = group_tasks.groupby("task_normalized")["physical_bool"].any()
            n_phys = int(phys_by_task.sum())

            metrics = _compute_task_metrics(unique_task_norms, lookup, selected_sources)

            # Parent info (take mode from occs)
            parent_name = None
            grandparent_name = None
            if occ_to_parent is not None:
                from collections import Counter
                parents = [occ_to_parent.get(t) for t in occs if occ_to_parent.get(t)]
                if parents:
                    parent_name = Counter(parents).most_common(1)[0][0]
            if occ_to_grandparent is not None:
                from collections import Counter
                grandparents = [occ_to_grandparent.get(t) for t in occs if occ_to_grandparent.get(t)]
                if grandparents:
                    grandparent_name = Counter(grandparents).most_common(1)[0][0]

            # DWS star rating: average across occs in this group
            dws_vals = [occ_to_dws[t] for t in occs if t in occ_to_dws]
            group_dws = round(sum(dws_vals) / len(dws_vals), 1) if dws_vals else None

            # Job zone: average across occs in this group
            jz_vals = [occ_to_jz[t] for t in occs if t in occ_to_jz]
            group_jz = round(sum(jz_vals) / len(jz_vals), 1) if jz_vals else None

            row = {
                "name":    group_name,
                "parent":  parent_name,
                "grandparent": grandparent_name,
                "dws_star_rating": group_dws,
                "job_zone": group_jz,
                "n_occs":  len(occs),
                "n_tasks": n_tasks,
                "n_physical_tasks": n_phys,
                "pct_physical": round(n_phys / n_tasks, 4) if n_tasks else None,
                "_occs": occs,   # preserved for geo overlay
                **metrics,
            }
            groups_data.append(row)

        result[level_key] = groups_data

    _explorer_groups_base_cache[cache_key] = result
    return result


def get_explorer_groups(geo: str = "nat", selected_sources: Optional[frozenset] = None) -> dict:
    """
    Returns pre-computed aggregations for major/minor/broad levels.
    Each group's metrics are computed from unique task_norms across all occupations
    in that group (NOT averaged from occ-level values).
    Each row also includes parent hierarchy fields.
    Base data is cached; emp/wage are overlaid per geo parameter.
    When selected_sources is provided, only those sources contribute to metrics.
    """
    base = _build_explorer_groups_base(selected_sources)
    if not base:
        return {"major": [], "minor": [], "broad": []}

    eco = load_eco_raw()
    if eco is None:
        return base

    # Resolve geo columns with fallback to national
    emp_col = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"
    if emp_col not in eco.columns:
        emp_col = "emp_tot_nat_2025"
    if wage_col not in eco.columns:
        wage_col = "a_med_nat_2025"

    # Build occ→emp/wage lookup for this geo
    occ_agg = eco.groupby("title_current").agg(
        _emp=(emp_col, "first"),
        _wage=(wage_col, "first"),
    ).reset_index()
    occ_to_emp: dict = dict(zip(occ_agg["title_current"], occ_agg["_emp"]))
    occ_to_wage: dict = dict(zip(occ_agg["title_current"], occ_agg["_wage"]))

    result: dict = {}
    for level_key in ("major", "minor", "broad"):
        groups_data = []
        for base_row in base.get(level_key, []):
            row = dict(base_row)
            occs = row.pop("_occs")

            # Compute emp (sum) and wage (emp-weighted avg) for this geo
            total_emp = sum((_safe_num(occ_to_emp.get(t)) or 0) for t in occs)
            wage_sum = 0.0
            wage_emp = 0.0
            for t in occs:
                e = _safe_num(occ_to_emp.get(t)) or 0
                w = _safe_num(occ_to_wage.get(t))
                if w is not None and e > 0:
                    wage_sum += w * e
                    wage_emp += e

            row["emp"] = round(total_emp, 0) if total_emp else None
            row["wage"] = round(wage_sum / wage_emp, 0) if wage_emp else None
            groups_data.append(row)

        result[level_key] = groups_data

    return result


def get_wa_explorer_data(geo: str = "nat", selected_sources: Optional[frozenset] = None) -> list:
    """
    Returns WA explorer data: list of rows for GWA, IWA, DWA levels.
    Each row includes: level, name, parent, gwa, emp, wage, n_occs, n_tasks, metrics.
    emp uses the same allocation logic as the WA backend (emp_occ / n_unique_tasks_in_occ).
    AI metrics are computed from tasks deduplicated at each level (task_norm x activity).
    Results are cached per (geo, selected_sources).
    """
    wa_cache_key = (geo, selected_sources)
    if wa_cache_key in _wa_explorer_geo_cache:
        return _wa_explorer_geo_cache[wa_cache_key]

    eco = load_eco_raw()
    if eco is None:
        return []

    lookup = _build_explorer_task_lookup()

    # Resolve geo columns with fallback to national
    emp_col = f"emp_tot_{geo}_2025"
    wage_col = f"a_med_{geo}_2025"
    if emp_col not in eco.columns:
        emp_col = "emp_tot_nat_2025"
    if wage_col not in eco.columns:
        wage_col = "a_med_nat_2025"

    needed_cols = [
        "title_current", "task_normalized", "task",
        "dwa_title", "iwa_title", "gwa_title",
        "physical", emp_col, wage_col,
        "freq_mean", "relevance", "importance",
    ]
    avail_cols = [c for c in needed_cols if c in eco.columns]
    df = eco[avail_cols].copy()

    # Compute emp weights with both methods (freq and value) so frontend can toggle
    task_dedup = df.drop_duplicates(subset=["title_current", "task_normalized"]).copy()
    task_dedup["_freq_w"] = task_dedup["freq_mean"].fillna(0.0) if "freq_mean" in task_dedup.columns else 0.0
    task_dedup["_value_w"] = (
        task_dedup["freq_mean"].fillna(0.0)
        * task_dedup["relevance"].fillna(0.0)
        * task_dedup["importance"].fillna(0.0)
    ) if all(c in task_dedup.columns for c in ["freq_mean", "relevance", "importance"]) else 0.0
    task_dedup["_freq_sum"] = task_dedup.groupby("title_current")["_freq_w"].transform("sum")
    task_dedup["_value_sum"] = task_dedup.groupby("title_current")["_value_w"].transform("sum")
    task_dedup["_freq_frac"] = (task_dedup["_freq_w"] / task_dedup["_freq_sum"].replace(0, np.nan)).fillna(0.0)
    task_dedup["_value_frac"] = (task_dedup["_value_w"] / task_dedup["_value_sum"].replace(0, np.nan)).fillna(0.0)
    frac_lookup = task_dedup[["title_current", "task_normalized", "_freq_frac", "_value_frac"]].copy()
    df = df.merge(frac_lookup, on=["title_current", "task_normalized"], how="left")

    if emp_col in df.columns:
        df["emp_per_task_freq"]  = (df["_freq_frac"]  * df[emp_col].fillna(0)).fillna(0.0)
        df["emp_per_task_value"] = (df["_value_frac"] * df[emp_col].fillna(0)).fillna(0.0)
    else:
        df["emp_per_task_freq"]  = 0.0
        df["emp_per_task_value"] = 0.0

    def _phys_bool(v):
        if v is None:
            return False
        if isinstance(v, float) and np.isnan(v):
            return False
        return bool(v)

    df["physical_bool"] = df["physical"].apply(_phys_bool)

    rows_out: list = []

    level_specs = [
        ("gwa", "gwa_title", None,        "gwa_title"),
        ("iwa", "iwa_title", "gwa_title", "gwa_title"),
        ("dwa", "dwa_title", "iwa_title", "gwa_title"),
    ]

    for level_key, act_col, parent_col, gwa_col in level_specs:
        if act_col not in df.columns:
            continue

        level_df = df[df[act_col].notna()].copy()
        if level_df.empty:
            continue

        # Unique (task_norm, act) pairs for deduplication at this level
        unique_pairs = level_df[[act_col, "task_normalized"]].drop_duplicates()

        for act_name, act_df in sorted(level_df.groupby(act_col), key=lambda x: x[0]):
            # emp: sum emp_per_task across unique (occ, task_norm) combos for BOTH methods
            occ_task_dedup = act_df.drop_duplicates(subset=["title_current", "task_normalized"])
            total_emp_freq  = occ_task_dedup["emp_per_task_freq"].fillna(0).sum()
            total_emp_value = occ_task_dedup["emp_per_task_value"].fillna(0).sum()

            # wage: emp-weighted avg for BOTH methods
            wage_sum_f = 0.0; wage_emp_f = 0.0
            wage_sum_v = 0.0; wage_emp_v = 0.0
            for _, r in occ_task_dedup.iterrows():
                ef = _safe_num(r.get("emp_per_task_freq")) or 0
                ev = _safe_num(r.get("emp_per_task_value")) or 0
                w = _safe_num(r.get(wage_col))
                if w is not None:
                    if ef > 0: wage_sum_f += w * ef; wage_emp_f += ef
                    if ev > 0: wage_sum_v += w * ev; wage_emp_v += ev

            n_occs = int(act_df["title_current"].nunique())

            # Unique task_norms at this activity level
            unique_task_norms = unique_pairs[unique_pairs[act_col] == act_name]["task_normalized"].unique().tolist()
            n_tasks = len(unique_task_norms)

            # Physical: unique tasks that are physical
            phys_by_task = act_df.drop_duplicates(subset=["task_normalized"]).set_index("task_normalized")["physical_bool"]
            n_phys = int(phys_by_task.sum())

            metrics = _compute_task_metrics(unique_task_norms, lookup, selected_sources)

            # Parent / gwa info
            parent_name = None
            gwa_name    = None
            if parent_col and parent_col in act_df.columns:
                from collections import Counter
                parents = [v for v in act_df[parent_col].dropna() if v]
                parent_name = Counter(parents).most_common(1)[0][0] if parents else None
            if gwa_col and gwa_col in act_df.columns:
                from collections import Counter
                gwas = [v for v in act_df[gwa_col].dropna() if v]
                gwa_name = Counter(gwas).most_common(1)[0][0] if gwas else None
            elif level_key == "gwa":
                gwa_name = act_name

            rows_out.append({
                "level":   level_key,
                "name":    str(act_name),
                "parent":  parent_name,
                "gwa":     gwa_name,
                "emp_freq":  round(float(total_emp_freq), 1) if total_emp_freq else None,
                "emp_value": round(float(total_emp_value), 1) if total_emp_value else None,
                "wage_freq":  round(wage_sum_f / wage_emp_f, 0) if wage_emp_f else None,
                "wage_value": round(wage_sum_v / wage_emp_v, 0) if wage_emp_v else None,
                "n_occs":  n_occs,
                "n_tasks": n_tasks,
                "n_physical_tasks": n_phys,
                "pct_physical": round(n_phys / n_tasks, 4) if n_tasks else None,
                **metrics,
            })

    _wa_explorer_geo_cache[wa_cache_key] = rows_out
    return rows_out


def get_wa_tasks_for_activity(level: str, name: str, geo: str = "nat") -> list:
    """
    Returns task-level details for a specific WA activity (gwa/iwa/dwa).
    Tasks are deduplicated by task_normalized, with emp summed across all occupations.
    Accepts a geo parameter for geography-specific emp/wage data.
    """
    cache_key = (level, name, geo)
    if cache_key in _wa_cache:
        return _wa_cache[cache_key]

    eco = load_eco_raw()
    if eco is None:
        return []

    lookup = _build_explorer_task_lookup()
    top_mcps_lookup = _build_top_mcps_lookup()

    act_col_map = {"gwa": "gwa_title", "iwa": "iwa_title", "dwa": "dwa_title"}
    act_col = act_col_map.get(level)
    if not act_col or act_col not in eco.columns:
        return []

    # Resolve geo columns with fallback to national
    emp_col_name = f"emp_tot_{geo}_2025"
    wage_col_name = f"a_med_{geo}_2025"
    if emp_col_name not in eco.columns:
        emp_col_name = "emp_tot_nat_2025"
    if wage_col_name not in eco.columns:
        wage_col_name = "a_med_nat_2025"

    # Filter to this activity
    act_df = eco[eco[act_col] == name].copy()
    if act_df.empty:
        return []

    # emp allocation — weighted by freq or freq*rel*imp
    task_dd = eco.drop_duplicates(subset=["title_current", "task_normalized"]).copy()
    task_dd["_freq_w"] = task_dd["freq_mean"].fillna(0.0) if "freq_mean" in task_dd.columns else 0.0
    task_dd["_value_w"] = (
        task_dd["freq_mean"].fillna(0.0)
        * task_dd["relevance"].fillna(0.0)
        * task_dd["importance"].fillna(0.0)
    ) if all(c in task_dd.columns for c in ["freq_mean", "relevance", "importance"]) else 0.0
    task_dd["_freq_sum"]  = task_dd.groupby("title_current")["_freq_w"].transform("sum")
    task_dd["_value_sum"] = task_dd.groupby("title_current")["_value_w"].transform("sum")
    task_dd["_freq_frac"]  = (task_dd["_freq_w"]  / task_dd["_freq_sum"].replace(0, np.nan)).fillna(0.0)
    task_dd["_value_frac"] = (task_dd["_value_w"] / task_dd["_value_sum"].replace(0, np.nan)).fillna(0.0)
    frac_lkp = task_dd[["title_current", "task_normalized", "_freq_frac", "_value_frac"]].copy()
    act_df = act_df.merge(frac_lkp, on=["title_current", "task_normalized"], how="left")
    if emp_col_name in act_df.columns:
        act_df["emp_per_task_freq"]  = (act_df["_freq_frac"]  * act_df[emp_col_name].fillna(0)).fillna(0.0)
        act_df["emp_per_task_value"] = (act_df["_value_frac"] * act_df[emp_col_name].fillna(0)).fillna(0.0)
    else:
        act_df["emp_per_task_freq"]  = 0.0
        act_df["emp_per_task_value"] = 0.0

    def _phys_bool(v):
        if v is None: return False
        if isinstance(v, float) and np.isnan(v): return False
        return bool(v)

    tasks_out = []
    seen_norms = set()

    for tn, grp in act_df.groupby("task_normalized"):
        if tn in seen_norms:
            continue
        seen_norms.add(tn)

        # Use first row for task text, hierarchy, physical
        first = grp.iloc[0]
        deduped = grp.drop_duplicates(subset=["title_current"])
        emp_freq  = float(deduped["emp_per_task_freq"].fillna(0).sum())
        emp_value = float(deduped["emp_per_task_value"].fillna(0).sum())

        # wage: emp-weighted avg for both methods
        wage_sum_f = 0.0; wage_emp_f = 0.0
        wage_sum_v = 0.0; wage_emp_v = 0.0
        for _, r in deduped.iterrows():
            ef = _safe_num(r.get("emp_per_task_freq")) or 0
            ev = _safe_num(r.get("emp_per_task_value")) or 0
            w = _safe_num(r.get(wage_col_name))
            if w is not None:
                if ef > 0: wage_sum_f += w * ef; wage_emp_f += ef
                if ev > 0: wage_sum_v += w * ev; wage_emp_v += ev

        sources = lookup.get(tn, {})
        auto_vals = [v["auto_aug"] for v in sources.values() if v.get("auto_aug") is not None]
        pct_vals  = [v["pct_norm"]  for v in sources.values() if v.get("pct_norm")  is not None]

        phys_val = first.get("physical")
        is_physical = None
        if phys_val is not None and not (isinstance(phys_val, float) and np.isnan(phys_val)):
            is_physical = bool(phys_val)

        tasks_out.append({
            "task":                str(first.get("task", tn)),
            "task_normalized":     tn,
            "dwa_title":           first.get("dwa_title"),
            "iwa_title":           first.get("iwa_title"),
            "gwa_title":           first.get("gwa_title"),
            "physical":            is_physical,
            "emp_freq":            round(emp_freq, 2) if emp_freq else None,
            "emp_value":           round(emp_value, 2) if emp_value else None,
            "wage_freq":           round(wage_sum_f / wage_emp_f, 0) if wage_emp_f else None,
            "wage_value":          round(wage_sum_v / wage_emp_v, 0) if wage_emp_v else None,
            "freq_mean":           _safe_num(first.get("freq_mean")),
            "importance":          _safe_num(first.get("importance")),
            "relevance":           _safe_num(first.get("relevance")),
            "title_current":       str(first.get("title_current", "")) or None,
            "broad_occ":           first.get("broad_occ") if not (isinstance(first.get("broad_occ"), float) and np.isnan(first.get("broad_occ"))) else None,
            "minor_occ_category":  first.get("minor_occ_category") if not (isinstance(first.get("minor_occ_category"), float) and np.isnan(first.get("minor_occ_category"))) else None,
            "major_occ_category":  first.get("major_occ_category") if not (isinstance(first.get("major_occ_category"), float) and np.isnan(first.get("major_occ_category"))) else None,
            "sources":             dict(sources),
            "avg_auto_aug":        round(sum(auto_vals) / len(auto_vals), 3) if auto_vals else None,
            "max_auto_aug":        round(max(auto_vals), 3) if auto_vals else None,
            "avg_pct_norm":        round(sum(pct_vals) / len(pct_vals), 4) if pct_vals else None,
            "max_pct_norm":        round(max(pct_vals), 4) if pct_vals else None,
            "top_mcps":            top_mcps_lookup.get(tn, []),
        })

    tasks_out.sort(key=lambda t: t["task"])
    _wa_cache[cache_key] = tasks_out
    return tasks_out


def get_all_tasks(geo: str = "nat") -> list:
    """
    Returns all unique tasks from eco_2025 with their AI metrics from the task lookup.
    Each task row includes: task, task_normalized, dwa_title, iwa_title, gwa_title,
    physical, n_occs (unique occ count), avg_auto_aug, max_auto_aug, avg_pct_norm, max_pct_norm,
    plus the sources dict. emp/wage are computed for the given geography.
    Results are cached per geo.
    """
    if geo in _all_tasks_geo_cache:
        return _all_tasks_geo_cache[geo]

    eco = load_eco_raw()
    if eco is None:
        return []

    lookup = _build_explorer_task_lookup()

    # Resolve geo columns with fallback to national
    emp_col_name = f"emp_tot_{geo}_2025"
    wage_col_name = f"a_med_{geo}_2025"
    if emp_col_name not in eco.columns:
        emp_col_name = "emp_tot_nat_2025"
    if wage_col_name not in eco.columns:
        wage_col_name = "a_med_nat_2025"

    # Get unique tasks with their metadata (use first occurrence for text/hierarchy)
    task_cols = ["task_normalized", "task", "dwa_title", "iwa_title", "gwa_title", "physical"]
    avail = [c for c in task_cols if c in eco.columns]
    task_meta = eco[avail].drop_duplicates(subset=["task_normalized"]).copy()

    # Count unique occs per task
    occ_counts = (
        eco.groupby("task_normalized")["title_current"]
        .nunique()
        .reset_index()
        .rename(columns={"title_current": "n_occs"})
    )
    task_meta = task_meta.merge(occ_counts, on="task_normalized", how="left")

    # Compute emp/wage allocation per task:
    # For each occ, n_unique_tasks = count of unique task_norms in that occ.
    # Each task gets emp_occ / n_unique_tasks. Sum across all occs sharing the task.
    has_emp  = emp_col_name in eco.columns
    has_wage = wage_col_name in eco.columns

    # Build per-occ n_unique_tasks count
    occ_task_counts = (
        eco.drop_duplicates(subset=["title_current", "task_normalized"])
        .groupby("title_current")["task_normalized"]
        .count()
        .rename("n_unique_tasks")
    )
    eco_occ = eco.drop_duplicates(subset=["title_current"]).set_index("title_current")

    # Build per (occ, task) contribution rows
    occ_task_pairs = eco.drop_duplicates(subset=["title_current", "task_normalized"])[
        ["title_current", "task_normalized"]
    ].copy()
    occ_task_pairs = occ_task_pairs.join(occ_task_counts, on="title_current")

    # Attach emp/wage from occ-level data
    for col in [c for c in [emp_col_name, wage_col_name] if c in eco_occ.columns]:
        occ_task_pairs[col] = occ_task_pairs["title_current"].map(eco_occ[col])

    if has_emp:
        occ_task_pairs["emp_contrib"] = (
            occ_task_pairs[emp_col_name].fillna(0) / occ_task_pairs["n_unique_tasks"].replace(0, 1)
        )

    # Sum emp contributions by task
    task_emp = None
    if has_emp:
        task_emp = occ_task_pairs.groupby("task_normalized").agg(
            emp=("emp_contrib", "sum"),
        ).reset_index()

    # Employment-weighted wage: sum(emp_contrib * wage) / sum(emp_contrib)
    task_wage = None
    if has_emp and has_wage:
        occ_task_pairs["wage_contrib"] = (
            occ_task_pairs["emp_contrib"] * occ_task_pairs[wage_col_name].fillna(0)
        )
        wage_agg = occ_task_pairs.groupby("task_normalized").agg(
            wage_sum=("wage_contrib", "sum"),
            emp_sum=("emp_contrib", "sum"),
        )
        wage_agg["wage"] = np.where(
            wage_agg["emp_sum"] > 0,
            wage_agg["wage_sum"] / wage_agg["emp_sum"],
            np.nan,
        )
        task_wage = wage_agg[["wage"]].reset_index()

    # Merge emp/wage into task_meta
    if task_emp is not None:
        task_meta = task_meta.merge(task_emp, on="task_normalized", how="left")
    if task_wage is not None:
        task_meta = task_meta.merge(task_wage, on="task_normalized", how="left")

    def _phys_bool(v):
        if v is None: return False
        if isinstance(v, float) and np.isnan(v): return False
        return bool(v)

    def _nan_to_none(v):
        if v is None: return None
        try:
            if np.isnan(float(v)): return None
        except Exception:
            pass
        return float(v)

    result = []
    for _, row in task_meta.sort_values("task").iterrows():
        tn = row["task_normalized"]
        sources = dict(lookup.get(tn, {}))
        auto_vals = [v["auto_aug"] for v in sources.values() if v.get("auto_aug") is not None]
        pct_vals  = [v["pct_norm"]  for v in sources.values() if v.get("pct_norm")  is not None]

        phys_val = row.get("physical")
        is_physical = None
        if phys_val is not None and not (isinstance(phys_val, float) and np.isnan(phys_val)):
            is_physical = bool(phys_val)

        result.append({
            "task":           str(row.get("task", tn)),
            "task_normalized": tn,
            "dwa_title":      row.get("dwa_title"),
            "iwa_title":      row.get("iwa_title"),
            "gwa_title":      row.get("gwa_title"),
            "physical":       is_physical,
            "n_occs":         int(row.get("n_occs", 0)),
            "emp":            _nan_to_none(row.get("emp")),
            "wage":           _nan_to_none(row.get("wage")),
            "sources":        sources,
            "avg_auto_aug":   round(sum(auto_vals) / len(auto_vals), 3) if auto_vals else None,
            "max_auto_aug":   round(max(auto_vals), 3)                  if auto_vals else None,
            "avg_pct_norm":   round(sum(pct_vals) / len(pct_vals), 4)  if pct_vals  else None,
            "max_pct_norm":   round(max(pct_vals), 4)                  if pct_vals  else None,
        })

    _all_tasks_geo_cache[geo] = result
    return result


# ── All eco task rows (one row per task×occ in eco_2025) ────────────────────


def get_all_eco_task_rows(geo: str = "nat", selected_sources: Optional[frozenset] = None) -> list:
    """
    Returns every row from eco_2025 (~23,850 rows), each being a unique
    (task, occupation, DWA/IWA/GWA) combination. Includes the occupation
    hierarchy columns and raw (undivided) emp/wage numbers, plus AI metrics
    from the explorer task lookup.
    Results are cached per (geo, selected_sources).
    """
    eco_cache_key = (geo, selected_sources)
    if eco_cache_key in _all_eco_tasks_geo_cache:
        return _all_eco_tasks_geo_cache[eco_cache_key]

    eco = load_eco_raw()
    if eco is None:
        return []

    lookup = _build_explorer_task_lookup()
    top_mcps_lookup = _build_top_mcps_lookup()

    # Resolve geo columns with fallback to national
    emp_col_name = f"emp_tot_{geo}_2025"
    wage_col_name = f"a_med_{geo}_2025"
    if emp_col_name not in eco.columns:
        emp_col_name = "emp_tot_nat_2025"
    if wage_col_name not in eco.columns:
        wage_col_name = "a_med_nat_2025"

    # Pre-compute weighted emp allocation (freq and value methods)
    task_dedup = eco.drop_duplicates(subset=["title_current", "task_normalized"]).copy()
    task_dedup["_freq_w"] = task_dedup["freq_mean"].fillna(0.0) if "freq_mean" in task_dedup.columns else 0.0
    task_dedup["_value_w"] = (
        task_dedup["freq_mean"].fillna(0.0)
        * task_dedup["relevance"].fillna(0.0)
        * task_dedup["importance"].fillna(0.0)
    ) if all(c in task_dedup.columns for c in ["freq_mean", "relevance", "importance"]) else 0.0
    task_dedup["_freq_sum"] = task_dedup.groupby("title_current")["_freq_w"].transform("sum")
    task_dedup["_value_sum"] = task_dedup.groupby("title_current")["_value_w"].transform("sum")
    task_dedup["_freq_frac"] = (task_dedup["_freq_w"] / task_dedup["_freq_sum"].replace(0, np.nan)).fillna(0.0)
    task_dedup["_value_frac"] = (task_dedup["_value_w"] / task_dedup["_value_sum"].replace(0, np.nan)).fillna(0.0)
    frac_lookup = task_dedup.set_index(["title_current", "task_normalized"])[["_freq_frac", "_value_frac"]].to_dict("index")

    # Pre-compute AI metrics per task_normalized to avoid repeated dict lookups
    task_metrics: dict[str, dict] = {}
    for tn, all_sources in lookup.items():
        if selected_sources is not None:
            sources = {k: v for k, v in all_sources.items() if k in selected_sources}
        else:
            sources = all_sources
        auto_vals = [v["auto_aug"] for v in sources.values() if v.get("auto_aug") is not None]
        pct_vals = [v["pct_norm"] for v in sources.values() if v.get("pct_norm") is not None]
        task_metrics[tn] = {
            "sources": dict(sources),
            "avg_auto_aug": round(sum(auto_vals) / len(auto_vals), 3) if auto_vals else None,
            "max_auto_aug": round(max(auto_vals), 3) if auto_vals else None,
            "avg_pct_norm": round(sum(pct_vals) / len(pct_vals), 4) if pct_vals else None,
            "max_pct_norm": round(max(pct_vals), 4) if pct_vals else None,
        }

    def _safe_str(v) -> str | None:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return str(v)

    def _safe_float(v) -> float | None:
        if v is None:
            return None
        try:
            f = float(v)
            if np.isnan(f) or np.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def _safe_bool(v) -> bool | None:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return bool(v)

    result = []
    for _, row in eco.iterrows():
        tn = row.get("task_normalized", "")
        metrics = task_metrics.get(tn, {})
        occ = row.get("title_current", "")

        # Weighted emp allocation for this task*occ
        fracs = frac_lookup.get((occ, tn), {"_freq_frac": 0.0, "_value_frac": 0.0})
        ff = fracs["_freq_frac"]
        vf = fracs["_value_frac"]
        raw_emp = _safe_float(row.get(emp_col_name)) or 0.0

        result.append({
            "task": str(row.get("task", tn)),
            "task_normalized": str(tn),
            "title_current": str(occ),
            "broad_occ": _safe_str(row.get("broad_occ")),
            "minor_occ_category": _safe_str(row.get("minor_occ_category")),
            "major_occ_category": _safe_str(row.get("major_occ_category")),
            "dwa_title": _safe_str(row.get("dwa_title")),
            "iwa_title": _safe_str(row.get("iwa_title")),
            "gwa_title": _safe_str(row.get("gwa_title")),
            "physical": _safe_bool(row.get("physical")),
            "emp": _safe_float(row.get(emp_col_name)),
            "wage": _safe_float(row.get(wage_col_name)),
            "emp_freq": round(ff * raw_emp, 2) if ff and raw_emp else None,
            "emp_value": round(vf * raw_emp, 2) if vf and raw_emp else None,
            "freq_mean":  _safe_float(row.get("freq_mean")),
            "importance": _safe_float(row.get("importance")),
            "relevance":  _safe_float(row.get("relevance")),
            "sources": metrics.get("sources", {}),
            "avg_auto_aug": metrics.get("avg_auto_aug"),
            "max_auto_aug": metrics.get("max_auto_aug"),
            "avg_pct_norm": metrics.get("avg_pct_norm"),
            "max_pct_norm": metrics.get("max_pct_norm"),
            "top_mcps": top_mcps_lookup.get(tn, []),
        })

    _all_eco_tasks_geo_cache[eco_cache_key] = result
    return result


# ── Task Changes (dataset comparison) ─────────────────────────────────────────

def _build_eco2015_baseline_set() -> set:
    """
    Build set of (task_normalized, title_current) from eco_2015 crosswalked to 2019 SOC.
    Used to determine "not_in_baseline" status for cross-family comparisons.
    """
    global _eco2015_baseline_set_cache
    if _eco2015_baseline_set_cache is not None:
        return _eco2015_baseline_set_cache

    eco15 = load_eco2015_raw()
    crosswalk = load_crosswalk()
    if eco15 is None or crosswalk is None:
        _eco2015_baseline_set_cache = set()
        return _eco2015_baseline_set_cache

    # eco_2015 has soc_code_2010 — join with crosswalk to get title_current
    merged = eco15[["task_normalized", "soc_code_2010"]].drop_duplicates().merge(
        crosswalk[["O*NET-SOC 2010 Code", "O*NET-SOC 2019 Title"]],
        left_on="soc_code_2010", right_on="O*NET-SOC 2010 Code",
        how="inner",
    )
    result = set(zip(merged["task_normalized"], merged["O*NET-SOC 2019 Title"]))
    _eco2015_baseline_set_cache = result
    return result


def _prepare_dataset_for_comparison(ds_name: str) -> Optional[pd.DataFrame]:
    """
    Load a dataset and normalize to (task_normalized, title_current) pairs
    with auto_aug_mean and pct_normalized values (averaged across duplicates).
    AEI datasets are crosswalked to 2019 SOC.
    """
    meta = DATASETS.get(ds_name)
    if meta is None or not Path(meta["file"]).exists():
        return None

    df = pd.read_csv(meta["file"])
    if "task_normalized" not in df.columns:
        return None

    for col in ("auto_aug_mean", "pct_normalized"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    is_aei = meta["is_aei"]

    if is_aei:
        # Crosswalk AEI data (2010 SOC) to title_current (2019 SOC)
        crosswalk = load_crosswalk()
        if crosswalk is None:
            return None

        # Add soc_code_2010 if not present
        if "soc_code_2010" not in df.columns:
            return None

        # Join with crosswalk
        df = df.merge(
            crosswalk[["O*NET-SOC 2010 Code", "O*NET-SOC 2019 Title"]],
            left_on="soc_code_2010", right_on="O*NET-SOC 2010 Code",
            how="inner",
        )
        title_col = "O*NET-SOC 2019 Title"
    else:
        if "title_current" not in df.columns:
            return None
        title_col = "title_current"

    # Group by (title_current, task_normalized), average the scores
    agg = (
        df.groupby([title_col, "task_normalized"], sort=False)
        .agg(auto_aug_mean=("auto_aug_mean", "mean"), pct_normalized=("pct_normalized", "mean"))
        .reset_index()
    )
    if title_col != "title_current":
        agg = agg.rename(columns={title_col: "title_current"})

    return agg


def compute_task_changes(from_dataset: str, to_dataset: str, geo: str = "nat") -> list[dict]:
    """
    Compare two datasets at the task level.
    Returns list of dicts with status, deltas, and metadata for each task-occ pair.
    emp/wage are returned for the requested geography.
    """
    cache_key = (from_dataset, to_dataset, geo)
    if cache_key in _task_changes_cache:
        return _task_changes_cache[cache_key]

    from_df = _prepare_dataset_for_comparison(from_dataset)
    to_df = _prepare_dataset_for_comparison(to_dataset)

    if from_df is None and to_df is None:
        return []

    # Build eco baseline sets for "not_in_baseline" checks
    eco25 = load_eco_raw()
    eco25_set: set = set()
    if eco25 is not None:
        eco25_set = set(zip(eco25["task_normalized"], eco25["title_current"]))

    eco15_set = _build_eco2015_baseline_set()

    # Determine which baseline each dataset uses
    from_meta = DATASETS.get(from_dataset, {})
    to_meta = DATASETS.get(to_dataset, {})
    from_is_aei = from_meta.get("is_aei", False)
    to_is_aei = to_meta.get("is_aei", False)
    from_baseline_set = eco15_set if from_is_aei else eco25_set
    to_baseline_set = eco15_set if to_is_aei else eco25_set

    # Full outer join
    if from_df is not None and to_df is not None:
        merged = from_df.merge(
            to_df,
            on=["task_normalized", "title_current"],
            how="outer",
            suffixes=("_from", "_to"),
        )
    elif from_df is not None:
        merged = from_df.rename(columns={"auto_aug_mean": "auto_aug_mean_from", "pct_normalized": "pct_normalized_from"})
        merged["auto_aug_mean_to"] = np.nan
        merged["pct_normalized_to"] = np.nan
    else:
        assert to_df is not None
        merged = to_df.rename(columns={"auto_aug_mean": "auto_aug_mean_to", "pct_normalized": "pct_normalized_to"})
        merged["auto_aug_mean_from"] = np.nan
        merged["pct_normalized_from"] = np.nan

    # Resolve geo columns with fallback to national
    geo_emp_col = f"emp_tot_{geo}_2025"
    geo_wage_col = f"a_med_{geo}_2025"
    if eco25 is not None:
        if geo_emp_col not in eco25.columns:
            geo_emp_col = "emp_tot_nat_2025"
        if geo_wage_col not in eco25.columns:
            geo_wage_col = "a_med_nat_2025"

    # Enrich with eco_2025 metadata (one row per (task_normalized, title_current))
    eco_meta: Optional[pd.DataFrame] = None
    if eco25 is not None:
        agg_cols = {
            "task": "first",
            "broad_occ": "first",
            "minor_occ_category": "first",
            "major_occ_category": "first",
            "dwa_title": "first",
            "iwa_title": "first",
            "gwa_title": "first",
            "physical": "first",
            "freq_mean": "first",
            "importance": "first",
            "relevance": "first",
        }
        if geo_emp_col in eco25.columns:
            agg_cols[geo_emp_col] = "first"
        if geo_wage_col in eco25.columns:
            agg_cols[geo_wage_col] = "first"
        eco_meta = (
            eco25.groupby(["task_normalized", "title_current"], sort=False)
            .agg(agg_cols)
            .reset_index()
        )
        merged = merged.merge(eco_meta, on=["task_normalized", "title_current"], how="left")

    # Get explorer lookups for source breakdown and top MCPs
    task_lookup = _build_explorer_task_lookup()
    top_mcps_lookup = _build_top_mcps_lookup()

    # Build result rows
    result: list[dict] = []
    for _, row in merged.iterrows():
        tn = row["task_normalized"]
        tc = row["title_current"]
        from_aug = _safe_num(row.get("auto_aug_mean_from"))
        to_aug = _safe_num(row.get("auto_aug_mean_to"))
        from_pct = _safe_num(row.get("pct_normalized_from"))
        to_pct = _safe_num(row.get("pct_normalized_to"))

        has_from = from_aug is not None or from_pct is not None
        has_to = to_aug is not None or to_pct is not None

        # More robust presence check: was this task-occ in the from/to dataset?
        # A row has "from" data if auto_aug_mean_from or pct_normalized_from is not NaN
        from_val = row.get("auto_aug_mean_from")
        to_val = row.get("auto_aug_mean_to")
        has_from = pd.notna(from_val) if from_val is not None else False
        has_to = pd.notna(to_val) if to_val is not None else False
        # Also check pct as fallback (a dataset might have pct but not auto_aug)
        if not has_from:
            fpct = row.get("pct_normalized_from")
            has_from = pd.notna(fpct) if fpct is not None else False
        if not has_to:
            tpct = row.get("pct_normalized_to")
            has_to = pd.notna(tpct) if tpct is not None else False

        pair = (tn, tc)

        if has_from and has_to:
            # Both datasets rated this task-occ — compare auto_aug values
            if from_aug is None and to_aug is None:
                status = "unchanged"
            elif from_aug is None or to_aug is None:
                status = "changed"  # one has score, other doesn't
            elif abs(to_aug - from_aug) > 1e-6:
                status = "changed"
            else:
                status = "unchanged"
        elif has_to and not has_from:
            # Only in "to" — check against "from" baseline
            if pair in from_baseline_set:
                status = "new"
            else:
                status = "not_in_baseline"
        elif has_from and not has_to:
            # Only in "from" — check against "to" baseline
            if pair in to_baseline_set:
                status = "removed"
            else:
                status = "not_in_baseline"
        else:
            continue  # no data on either side (shouldn't happen)

        # Compute deltas
        delta_aug = None
        if from_aug is not None and to_aug is not None:
            delta_aug = round(to_aug - from_aug, 4)
        delta_pct = None
        if from_pct is not None and to_pct is not None:
            delta_pct = round(to_pct - from_pct, 4)

        # Source breakdown from explorer lookup
        task_sources = task_lookup.get(tn, {})
        source_autos = [v["auto_aug"] for v in task_sources.values() if v.get("auto_aug") is not None]
        source_pcts = [v["pct_norm"] for v in task_sources.values() if v.get("pct_norm") is not None]

        result.append({
            "task": row.get("task") if pd.notna(row.get("task")) else tn,
            "task_normalized": tn,
            "title_current": tc,
            "broad_occ": row.get("broad_occ") if pd.notna(row.get("broad_occ", np.nan)) else None,
            "minor_occ_category": row.get("minor_occ_category") if pd.notna(row.get("minor_occ_category", np.nan)) else None,
            "major_occ_category": row.get("major_occ_category") if pd.notna(row.get("major_occ_category", np.nan)) else None,
            "dwa_title": row.get("dwa_title") if pd.notna(row.get("dwa_title", np.nan)) else None,
            "iwa_title": row.get("iwa_title") if pd.notna(row.get("iwa_title", np.nan)) else None,
            "gwa_title": row.get("gwa_title") if pd.notna(row.get("gwa_title", np.nan)) else None,
            "physical": bool(row.get("physical")) if pd.notna(row.get("physical", np.nan)) else None,
            "freq_mean": _safe_num(row.get("freq_mean")),
            "importance": _safe_num(row.get("importance")),
            "relevance": _safe_num(row.get("relevance")),
            "emp": _safe_num(row.get(geo_emp_col)),
            "wage": _safe_num(row.get(geo_wage_col)),
            "status": status,
            "from_auto_aug": from_aug,
            "to_auto_aug": to_aug,
            "delta_auto_aug": delta_aug,
            "from_pct": from_pct,
            "to_pct": to_pct,
            "delta_pct": delta_pct,
            "sources": task_sources,
            "avg_auto_aug": round(sum(source_autos) / len(source_autos), 4) if source_autos else None,
            "max_auto_aug": round(max(source_autos), 4) if source_autos else None,
            "avg_pct_norm": round(sum(source_pcts) / len(source_pcts), 4) if source_pcts else None,
            "max_pct_norm": round(max(source_pcts), 4) if source_pcts else None,
            "top_mcps": top_mcps_lookup.get(tn, []),
        })

    _task_changes_cache[cache_key] = result
    return result
