"""
run_main_figures.py — regenerate every MAIN-BODY paper figure, in paper order.

Run from anywhere:
    python paper_figures/run_main_figures.py

Figures are written to paper_figures/figures/ (committed). Intermediate CSVs and
working copies land in paper_figures/results/ (gitignored). Section headers mirror
the paper's Results section (§6.2–6.8). See MAIN_FIGURES.md for the rendered set.

Requires the raw datasets in ../data/ (gitignored — see README for how to obtain
them). Each figure is run independently; if one fails the rest still proceed and a
summary of failures is printed at the end.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper_figures/
sys.path.insert(0, str(HERE))                    # so `import lib.*` resolves
sys.path.insert(0, str(HERE.parent))             # repo root, so `backend.*` resolves

# Builders print unicode (en-dashes, arrows); force UTF-8 so a cp1252 console
# (default on Windows) doesn't crash the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from lib.builders import part1, part2, part3      # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Paper Results order (§6.2 → §6.8). Each entry: (label, callable, extra_args).
SECTIONS: list[tuple[str, list[tuple[str, object, tuple]]]] = [
    ("6.2  Overall Measures", [
        ("External benchmark convergence (major + occupation)", part1.build_convergence, ()),
        ("AI economic exposure across data configurations", part1.build_overview, ()),
    ]),
    ("6.3  Trends", [
        ("All Confirmed vs Ceiling over time + data tables", part1.build_temporal, ()),
    ]),
    ("6.4  Major Occupational Categories", [
        ("Major categories — % tasks exposed (Confirmed | Variant A | Variant B)",
         part2.build_major_categories_pct, ()),
        ("Major categories — workers and wages", part2.build_major_categories_wkrs_wages, ()),
    ]),
    ("6.5  General Work Activities", [
        ("GWA — % tasks exposed (Confirmed | Variant A | Variant B)", part2.build_gwa_pct, ()),
    ]),
    ("6.6  Skills, Knowledge, Abilities", [
        ("SKA capability vs workforce need (skills + knowledge/abilities)",
         part2.build_ska_levels, ()),
    ]),
    ("6.7  Agentic AI", [
        ("Agentic confirmed vs ceiling — major categories", part3.build_agentic_ceiling_major, ()),
        ("Agentic confirmed vs ceiling — general work activities", part3.build_agentic_ceiling_gwa, ()),
    ]),
    ("6.8  Other Areas of Interest", [
        ("Job zone violin — full economy vs non-physical", part2.build_job_zone_violin, ()),
        ("Tech commodities where AI has reach", part3.build_tech_commodities, ()),
        ("High exposure × negative employment projection (focused set)",
         part3.build_risk_score_5f_workers, ()),
        ("U.S. states clustered on AI exposure", part3.build_state_clusters_map, ()),
        ("AI usage intensity by sector", part3.build_intensity_anchor_fulleco, ()),
    ]),
]


def main() -> None:
    failures: list[tuple[str, str]] = []
    for header, items in SECTIONS:
        print("\n" + "=" * 78)
        print(f"  {header}")
        print("=" * 78)
        for label, fn, args in items:
            print(f"\n  -> {label}")
            try:
                fn(RESULTS, FIGURES, *args)
            except Exception as exc:  # noqa: BLE001 — report and continue
                failures.append((label, f"{type(exc).__name__}: {exc}"))
                print(f"    !! FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 78)
    if failures:
        print(f"  MAIN-BODY FIGURES: {len(failures)} failure(s):")
        for label, err in failures:
            print(f"    - {label}: {err}")
    else:
        print("  MAIN-BODY FIGURES: all figures regenerated into paper_figures/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()
