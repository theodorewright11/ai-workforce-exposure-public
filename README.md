# AI Workforce Exposure — Public Release

Code, figures, and (soon) an interactive dashboard for the paper **"Mapping AI
Exposure Across the U.S. Workforce: Evidence from Millions of AI Conversations"**
(Wright, Schwarze, Boyd, 2026).

The project measures how current AI capability maps onto the U.S. occupational task
structure — combining real-world AI usage data (Anthropic's Claude, Microsoft's
Copilot), an MCP-server capability pipeline, occupational structure from O*NET, and
employment/wage data from BLS — to estimate exposure across tasks, workers, wages,
skills, work activities, and more.

---

## Repository layout

```
paper_figures/                 Regenerate every figure in the paper + supplement
├── run_main_figures.py        → all MAIN-BODY figures, in paper order (§6.2–6.8)
├── run_supplemental_figures.py→ all SUPPLEMENTAL figures, in supplement order
├── MAIN_FIGURES.md            Rendered main-body figures (headers + images inline)
├── SUPPLEMENTAL_FIGURES.md    Rendered supplemental figures
├── figures/                   Committed PNGs (the rendered output)
└── lib/                       Supporting code (figure builders + helpers)
backend/                       Shared compute engine (also powers the dashboard)
dashboard/                     Interactive dashboard (FastAPI + Next.js) — see dashboard/README.md
data/                          Datasets (NOT committed — see "Data" below)
requirements.txt
```

The `paper_figures/lib/` folder holds everything the two scripts need: the figure
builders (`lib/builders/`), the SKA / intensity / risk-score / state-cluster helpers
(`lib/exploratory/`), and the styling + config infrastructure. `backend/` is the
shared compute engine — the same code that powers the (forthcoming) dashboard.

---

## Regenerating the figures

```bash
python -m venv venv && source venv/Scripts/activate   # or your env of choice
pip install -r requirements.txt

# 1. obtain the datasets (see "Data" below) into ./data/
# 2. regenerate:
python paper_figures/run_main_figures.py          # main-body figures
python paper_figures/run_supplemental_figures.py  # supplemental figures
```

Both scripts write PNGs into `paper_figures/figures/`. Each figure runs
independently; if one fails (e.g. a missing dataset) the rest still proceed and a
summary of failures prints at the end.

---

## Data

The underlying datasets are **not committed to this repository** — they total
several hundred MB and are derived, public-source data. The committed PNGs in
`paper_figures/figures/` are the rendered results; the raw data is only needed to
*regenerate* them.

> **Data access:** _[TBD before public release — raw datasets will be hosted
> externally (e.g. Zenodo / Hugging Face) with a download step here.]_ Original
> source data and the construction pipeline are described in the paper's
> Supplementary Materials.

Once obtained, datasets go in `./data/` (raw `final_*.csv`) and
`./data/reference/` (O*NET SKA + external-index reference files). The figure
scripts read from these locations.

---

## Citation

Wright, T., Schwarze, A. C., & Boyd, Z. M. (2026). *Mapping AI Exposure Across the
U.S. Workforce: Evidence from Millions of AI Conversations.*

Built for the Utah Office of AI Policy as part of a research project measuring AI's
workforce impact.
