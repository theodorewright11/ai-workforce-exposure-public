"""Tasks under one work activity — the DWA drill-down in Explore.

All that survives of v1's `occupation_report.py` (1,597 lines). The rest --
the SKA loader, the SKA similarity matrix, the four-gate risk score, the
software/tech-commodity sections and the v1 occupation report itself -- is
retired with SKA (PRD §4). The v2 occupation card lives in `occupation.py`.

Extracted mechanically as the transitive closure of `get_wa_task_list`, so
nothing here is carried along "just in case".
"""
from __future__ import annotations

import dashboard.api  # noqa: F401  — sys.path bootstrap

import math
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from config import DATA_DIR, DATASETS
from compute import _build_top_mcps_lookup, _safe_num, compute_work_activities, load_eco_raw


PRIMARY_DATASET: str = "AEI Both + Micro 2026-05-31"


AUTO_HIGH: float = 4.0


AUTO_MID: float = 2.5


N_TASK_MCPS: int = 5


MCP_TITLES_DESC_FILE    = DATA_DIR / "mcp_titles_desc.csv"


_mcp_titles_desc_cache: Optional[dict[str, str]] = None


_eco_wa_stats_cache: dict[str, dict[str, dict[str, dict]]] = {}


def _color_bucket_auto(score: Optional[float]) -> str:
    """Map auto_aug score → color bucket. Three neutral framings."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "none"
    if score > AUTO_HIGH:
        return "high"     # automation level > 4
    if score > AUTO_MID:
        return "mid"      # 2.5 < level <= 4
    return "low"


def _round_or_none(v: Optional[float], n: int = 2) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), n)


def _mcp_titles_desc_lookup() -> dict[str, str]:
    """title (lowercased / titlecased) → text_for_llm description."""
    global _mcp_titles_desc_cache
    if _mcp_titles_desc_cache is not None:
        return _mcp_titles_desc_cache
    out: dict[str, str] = {}
    if MCP_TITLES_DESC_FILE.exists():
        df = pd.read_csv(MCP_TITLES_DESC_FILE)
        if "title" in df.columns and "text_for_llm" in df.columns:
            for _, row in df.iterrows():
                t = str(row["title"]).strip()
                d = row.get("text_for_llm")
                if not t or pd.isna(d):
                    continue
                out[t] = str(d)
                # also index by lowercase for case-insensitive lookup
                out[t.lower()] = str(d)
    _mcp_titles_desc_cache = out
    return out


_primary_tasklevel_cache: Optional[dict] = None


def _primary_task_level_lookup() -> dict:
    """{task_normalized: {auto, pct, freq}} — All-Confirmed values aggregated to
    the task level (mean across occupations). Used by the WA task-list drill where
    tasks are pooled across occupations."""
    global _primary_tasklevel_cache
    if _primary_tasklevel_cache is not None:
        return _primary_tasklevel_cache
    out: dict = {}
    fpath = DATASETS.get(PRIMARY_DATASET, {}).get("file", "")
    if Path(fpath).exists():
        df = pd.read_csv(fpath, low_memory=False)
        for c in ("auto_aug_mean", "pct_normalized", "freq_mean"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "task_normalized" in df.columns:
            g = df.groupby("task_normalized").agg(
                auto=("auto_aug_mean", "mean"),
                pct=("pct_normalized", "mean"),
                freq=("freq_mean", "mean"),
            )
            for tn, r in g.iterrows():
                out[tn] = {"auto": _safe_num(r["auto"]), "pct": _safe_num(r["pct"]), "freq": _safe_num(r["freq"])}
    _primary_tasklevel_cache = out
    return out


def _auto_label(score: Optional[float]) -> str:
    """Interpretive label for a 0–5 auto-aug value, so the number reads in place."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "no usage seen"
    if score >= 4.5:
        return "most automated usage seen"
    if score >= 3.5:
        return "automated usage seen"
    if score >= 2.5:
        return "mixed usage seen"
    if score >= 1.5:
        return "augmentative usage seen"
    return "low automation usage seen"


def _eco_wa_stats(geo: str) -> dict[str, dict[str, dict]]:
    """Economy-wide stats per WA at gwa/iwa/dwa, computed against PRIMARY_DATASET.

    Returns: {level: {wa_name: {pct_tasks_affected, workers_affected, wages_affected,
                                 auto_aug_mean, rank_pct, rank_workers, rank_wages,
                                 rank_auto, total}}}.

    Cached per geo. Workers/wages depend on geo; pct/auto_aug do not, but we store
    them under the same key for simplicity.
    """
    if geo in _eco_wa_stats_cache:
        return _eco_wa_stats_cache[geo]

    settings = {
        "selected_datasets": [PRIMARY_DATASET],
        "combine_method":    "Average",
        "method":            "freq",
        "use_auto_aug":      True,
        "physical_mode":     "all",
        "geo":               geo,
        "agg_level":         "occupation",
        "sort_by":           "Workers Affected",
        "top_n":             9999,
        "search_query":      "",
        "context_size":      3,
    }
    wa_result = compute_work_activities(settings)
    # PRIMARY is is_aei=False → comes back as mcp_group
    group = (wa_result or {}).get("mcp_group") or (wa_result or {}).get("aei_group") or {}

    # Per-WA auto_aug_mean from the dataset CSV (one pass, level-agnostic)
    auto_by_level: dict[str, dict[str, float]] = {"gwa": {}, "iwa": {}, "dwa": {}}
    fpath = DATASETS.get(PRIMARY_DATASET, {}).get("file", "")
    if Path(fpath).exists():
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if "auto_aug_mean" in df.columns:
                df["auto_aug_mean"] = pd.to_numeric(df["auto_aug_mean"], errors="coerce")
                for level_key, col in (("gwa", "gwa_title"), ("iwa", "iwa_title"), ("dwa", "dwa_title")):
                    if col not in df.columns:
                        continue
                    sub = df[df[col].notna() & df["auto_aug_mean"].notna()].copy()
                    # Dedup by (task_normalized, wa) so a task counted once per WA
                    if "task_normalized" in sub.columns:
                        sub = sub.drop_duplicates(subset=["task_normalized", col])
                    auto_by_level[level_key] = sub.groupby(col)["auto_aug_mean"].mean().to_dict()
        except Exception:
            pass  # leave auto empty; eco_stats just won't have auto

    out: dict[str, dict[str, dict]] = {"gwa": {}, "iwa": {}, "dwa": {}}
    for level_key in ("gwa", "iwa", "dwa"):
        rows = group.get(level_key) or []
        if not rows:
            continue
        rec_df = pd.DataFrame(rows)
        if rec_df.empty or "category" not in rec_df.columns:
            continue
        # Attach auto and compute per-metric ranks (1 = highest)
        rec_df["auto_aug_mean"] = rec_df["category"].map(auto_by_level.get(level_key, {})).astype(float)
        for metric in ("pct_tasks_affected", "workers_affected", "wages_affected", "auto_aug_mean"):
            if metric not in rec_df.columns:
                rec_df[metric] = np.nan
            rec_df[f"rank_{metric.split('_')[0]}"] = (
                rec_df[metric].rank(ascending=False, method="min", na_option="bottom").astype("Int64")
            )
        total = int(len(rec_df))
        for _, r in rec_df.iterrows():
            name = str(r["category"])
            out[level_key][name] = {
                "pct_tasks_affected": _round_or_none(r.get("pct_tasks_affected"), 1),
                "workers_affected":   _round_or_none(r.get("workers_affected"), 0),
                "wages_affected":     _round_or_none(r.get("wages_affected"), 0),
                "auto_aug_mean":      _round_or_none(r.get("auto_aug_mean"), 2),
                "rank_pct":           int(r["rank_pct"])     if pd.notna(r["rank_pct"])     else None,
                "rank_workers":       int(r["rank_workers"]) if pd.notna(r["rank_workers"]) else None,
                "rank_wages":         int(r["rank_wages"])   if pd.notna(r["rank_wages"])   else None,
                "rank_auto":          int(r["rank_auto"])    if pd.notna(r["rank_auto"])    else None,
                "total":              total,
            }

    _eco_wa_stats_cache[geo] = out
    return out


def get_wa_task_list(level: str, name: str) -> list[dict]:
    """Tasks under one work activity (gwa/iwa/dwa), pooled across occupations.
    Same card shape as the My-Occupation tasks: All-Confirmed automation level,
    usage-vs-median, GWA/IWA/DWA detail, top MCP tools. Centrality is the mean
    freq×imp×rel across the occupations sharing the task, ranked within the list.
    Sorted by automation level desc."""
    eco = load_eco_raw()
    col = {"gwa": "gwa_title", "iwa": "iwa_title", "dwa": "dwa_title"}.get(level)
    if eco is None or col is None or col not in eco.columns:
        return []
    sub = eco[eco[col] == name].copy()
    if sub.empty:
        return []

    tl = _primary_task_level_lookup()
    wa_stats = _eco_wa_stats("nat")
    top_mcps_lookup = _build_top_mcps_lookup()
    desc_lookup = _mcp_titles_desc_lookup()

    for c in ("freq_mean", "importance", "relevance"):
        sub[c] = pd.to_numeric(sub.get(c), errors="coerce").fillna(0.0)
    sub["_cent"] = sub["freq_mean"] * sub["importance"] * sub["relevance"]
    cent_by_tn = sub.groupby("task_normalized")["_cent"].mean()
    reps = sub.drop_duplicates("task_normalized").set_index("task_normalized")

    usage_vals: list[float] = []
    for tn in cent_by_tn.index:
        t = tl.get(tn)
        if t and t.get("pct") and t.get("freq") and t["freq"] > 0:
            usage_vals.append(t["pct"] / t["freq"])
    umed = float(np.median(usage_vals)) if usage_vals else 0.0

    def _wad(lvl: str, nm) -> Optional[dict]:
        if nm is None or (isinstance(nm, float) and pd.isna(nm)) or not str(nm):
            return None
        rec = wa_stats.get(lvl, {}).get(str(nm)) or {}
        return {"name": str(nm), "auto": rec.get("auto_aug_mean"), "rank_pct": rec.get("rank_pct"), "total": rec.get("total")}

    rows: list[dict] = []
    for tn in cent_by_tn.index:
        t = tl.get(tn, {})
        auto = t.get("auto")
        pct, freq = t.get("pct"), t.get("freq")
        umult = (pct / freq) / umed if (pct and freq and freq > 0 and umed > 0) else None
        rep = reps.loc[tn]
        mcps: list[dict] = []
        for m in top_mcps_lookup.get(tn, [])[:N_TASK_MCPS]:
            ttl = (m.get("title") or "").strip()
            mcps.append({"title": ttl, "rating": m.get("rating"), "url": m.get("url"),
                         "description": desc_lookup.get(ttl) or desc_lookup.get(ttl.lower())})
        rows.append({
            "task": rep.get("task"),
            "task_normalized": tn,
            "centrality": _round_or_none(float(cent_by_tn.loc[tn]), 1),
            "auto": _round_or_none(auto, 1),
            "auto_label": _auto_label(auto),
            "color_bucket": _color_bucket_auto(auto),
            "usage_mult": _round_or_none(umult, 1),
            "gwa": _wad("gwa", rep.get("gwa_title")),
            "iwa": _wad("iwa", rep.get("iwa_title")),
            "dwa": _wad("dwa", rep.get("dwa_title")),
            "top_mcps": mcps,
        })
    # centrality rank within this WA's task list
    by_cent = sorted(rows, key=lambda r: (r["centrality"] is None, -(r["centrality"] or 0.0)))
    for i, r in enumerate(by_cent, start=1):
        r["centrality_rank"] = i
    rows.sort(key=lambda r: (r["auto"] is None, -(r["auto"] or 0.0)))
    return rows
