"""
run_main_figures.py — regenerate every MAIN-BODY paper figure, in paper order.

Run from anywhere:
    python paper_figures/run_main_figures.py

Figures are written to paper_figures/figures/ (committed). Intermediate CSVs and
working copies land in paper_figures/results/ (gitignored). See MAIN_FIGURES.md
for the rendered set and paper_figures/lib/figure_data.py for the shared data
layer every figure is built from.

Exposure is weighted by work time: each (task, occupation) row carries
`time_per_day` (estimated hours per day, normalised so an occupation's tasks sum
to a 7-hour workday), so an exposure percentage is the share of the workday AI
reaches, employment-weighted at group level. Usage intensity is the deliberate
exception — it divides by employment alone.

Requires the datasets in ../data/. Each figure runs independently; if one fails
the rest still proceed and a summary of failures is printed at the end.
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

from lib.builders import (  # noqa: E402
    adoption, agentic, focused, jobzone, occupation, trend, verbs,
)

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Main-body order (mirrors MAIN_FIGURES.md).
SECTIONS: list[tuple[str, list[tuple[str, object]]]] = [
    ("Occupational structure", [
        ("Major categories — phys/non-phys stacked, with workers and wages",
         occupation.build_major_stacked),
        ("General work activities — phys/non-phys stacked, with workers and wages",
         occupation.build_gwa_stacked),
    ]),
    ("Job zones", [
        ("Job-zone violins + per-zone usage columns", jobzone.build_job_zone_usage),
    ]),
    ("Verb families", [
        ("Verb-family overview — All Confirmed", verbs.build_verb_family_overview),
        ("The verb family each major treats most unlike the economy",
         verbs.build_verb_family_exemplars),
    ]),
    ("Agentic AI", [
        ("Intermediate work activities where MCP tooling runs ahead of confirmed use",
         agentic.build_agentic_tooling),
    ]),
    ("Actual AI usage", [
        ("Major-category adoption ×median", adoption.build_major_adoption),
        ("GWA adoption ×median", adoption.build_gwa_adoption),
    ]),
    ("Trends", [
        ("Phys / non-phys / aggregate trend", trend.build_trend_phys),
    ]),
    ("Focused set", [
        ("Focused set + usage ×median", focused.build_focused_set_usage),
    ]),
]


def main() -> None:
    failures: list[tuple[str, str]] = []
    for header, items in SECTIONS:
        print("\n" + "=" * 78)
        print(f"  {header}")
        print("=" * 78)
        for label, fn in items:
            print(f"\n  -> {label}")
            try:
                fn(RESULTS, FIGURES)
            except Exception as exc:  # noqa: BLE001 — report and continue
                failures.append((label, f"{type(exc).__name__}: {exc}"))
                print(f"    !! FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 78)
    if failures:
        print(f"  MAIN-BODY FIGURES: {len(failures)} failure(s):")
        for label, err in failures:
            print(f"    - {label}: {err}")
        sys.exit(1)
    print("  MAIN-BODY FIGURES: all figures regenerated into paper_figures/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()
