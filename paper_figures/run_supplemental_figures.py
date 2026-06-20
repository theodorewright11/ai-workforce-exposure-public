"""
run_supplemental_figures.py — regenerate every SUPPLEMENTAL (appendix) paper
figure, in supplementary-materials order.

Run from anywhere:
    python paper_figures/run_supplemental_figures.py

Figures are written to paper_figures/figures/ (committed). Intermediate CSVs and
working copies land in paper_figures/results/ (gitignored). Section headers mirror
the Supplementary Materials figure sections. See SUPPLEMENTAL_FIGURES.md for the
rendered set.

Requires the raw datasets in ../data/ (gitignored — see README). Each figure runs
independently; failures are summarized at the end and do not stop the run.
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

from lib.builders import appendix                 # noqa: E402

RESULTS = HERE / "results"
FIGURES = HERE / "figures"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def _convergence_all(results: Path, figures: Path) -> None:
    """Full convergence matrix, one figure per SOC level (Major/Minor/Broad/Occ)."""
    for lvl_key, lvl_title in [("major", "Major level"), ("minor", "Minor level"),
                               ("broad", "Broad level"), ("occupation", "Occupation level")]:
        short = "occ" if lvl_key == "occupation" else lvl_key
        appendix.build_convergence_full(
            results, figures,
            levels=[(lvl_key, lvl_title)],
            out_name=f"convergence_full_{short}.png",
            csv_name=f"spearman_combined_full_{short}.csv",
        )


# Supplementary Materials figure order (mirrors paper_figures results.md appendix).
SECTIONS: list[tuple[str, list[tuple[str, object]]]] = [
    ("Full Convergence Matrix (Major / Minor / Broad / Occupation)", [
        ("Full convergence matrices — all four SOC levels", _convergence_all),
    ]),
    ("Aggregate Economy — Overview Without Auto-Aug", [
        ("Exposure across configs with auto-aug weighting off", appendix.build_overview_no_autoaug),
    ]),
    ("Trend Line — Physical vs Non-Physical Tasks", [
        ("Temporal trend restricted to non-physical tasks", appendix.build_temporal_trend_nonphys),
    ]),
    ("Major Occupational Category Trends and 2-Year Projection", [
        ("Major-category tasks + workers trend", appendix.build_major_categories_trend),
    ]),
    ("Where We and Eloundou Disagree by Major Occupational Category", [
        ("Eloundou z-score divergence by major", appendix.build_eloundou_divergence_major),
    ]),
    ("Knowledge and Abilities Full Elements", [
        ("Full element-level SKA (knowledge + abilities)", appendix.build_ska_full),
    ]),
    ("Generalized Work Activities — Workers and Wages", [
        ("GWA workers/wages counterpart to the main-body GWA chart", appendix.build_gwa_wkrs_wages),
    ]),
    ("State Rankings", [
        ("State clusters — each panel ranked independently", appendix.build_state_clusters_each_ranked),
        ("State clusters — combined rank sum across both panels", appendix.build_state_clusters_combined_ranked),
    ]),
    ("Actual AI Usage", [
        ("Underadoption gap by major occupational category", appendix.build_underadoption_gap),
        ("Within-major intensity drivers (Life/Phys/Soc, Arts/Design, Comp/Math)",
         appendix.build_intensity_drivers),
    ]),
    ("Framework Capability and Adoption Correlations", [
        ("Capability vs adoption properties across all occupations",
         appendix.build_capability_vs_adoption_all_occs),
        ("Adoption frictions vs exposure within non-physical occupations",
         appendix.build_adoption_friction_scatter),
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
    else:
        print("  SUPPLEMENTAL FIGURES: all figures regenerated into paper_figures/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()
