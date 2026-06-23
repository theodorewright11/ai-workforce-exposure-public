"""dashboard.api — FastAPI backend for the public AI Workforce Exposure dashboard.

Importing this package puts the repo's `backend/` and `paper_figures/` dirs on
sys.path so the dashboard can reuse the existing compute engine and figure-lib
helpers without copying them:

  - repo root        → `from backend...`, `from lib...` (package-style, if needed)
  - <root>/backend   → `from config import ...`, `from compute import ...` (flat,
                       the style backend/compute.py itself uses internally)
  - <root>/paper_figures → `from lib.config import ...`, `from lib.exploratory...`

This mirrors the path setup in paper_figures/lib/config.py and run_main_figures.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
BACKEND_DIR = ROOT / "backend"
PAPER_FIGURES_DIR = ROOT / "paper_figures"

for _p in (str(ROOT), str(BACKEND_DIR), str(PAPER_FIGURES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
