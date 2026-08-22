"""
run_supplemental_figures.py — regenerate every SUPPLEMENTAL (appendix) paper
figure, in supplementary-materials order.

Run from anywhere:
    python paper_figures/run_supplemental_figures.py

Figures are written to paper_figures/figures/ (committed). Intermediate CSVs and
working copies land in paper_figures/results/ (gitignored). See
SUPPLEMENTAL_FIGURES.md for the rendered set.

Same weighting as the main body — work time (`time_per_day`, normalised to a
7-hour workday), employment-weighted at group level. Usage is the exception by
design: Σ debiased pct ÷ Σ employment, employment alone.

Requires the datasets in ../data/. Each figure runs independently; failures are
summarized at the end and do not stop the run.
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

from lib.builders import benchmarks, drivers, verbs  # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Supplementary Materials figure order (mirrors SUPPLEMENTAL_FIGURES.md).
SECTIONS: list[tuple[str, list[tuple[str, object]]]] = [
    ("Verb Families — Non-Physical Tasks Only", [
        ("Verb-family overview — All Confirmed, non-physical",
         verbs.build_verb_family_overview_nonphys),
    ]),
    ("Actual AI Usage Inside Three Majors", [
        ("Top occupations + top tasks, 3 majors (6 figures)",
         drivers.build_usage_drivers_majors),
    ]),
    ("Actual AI Usage Inside the Four Leading Work Activities", [
        ("Top detailed activities per GWA (4 figures)", drivers.build_usage_drivers_gwa),
    ]),
    ("Convergence Against External Benchmarks", [
        ("Full convergence matrix — major level", benchmarks.build_convergence_major),
        ("Full convergence matrix — occupation level", benchmarks.build_convergence_occ),
    ]),
    ("Where We and Eloundou Disagree", [
        ("Eloundou z-score divergence by major", benchmarks.build_eloundou_divergence),
    ]),
]


def main() -> None:
    failures: list[tuple[str, str]] = []
    for header, items in SECTIONS:
        print("\n" + "=" * 78)
        print(f"  Supplementary: {header}")
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
        print(f"  SUPPLEMENTAL FIGURES: {len(failures)} failure(s):")
        for label, err in failures:
            print(f"    - {label}: {err}")
        sys.exit(1)
    print("  SUPPLEMENTAL FIGURES: all figures regenerated into paper_figures/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()
