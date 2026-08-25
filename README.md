# **Dashboard Link**: https://aiworkforceexposure.com

# AI Workforce Exposure — Public Release

> **Work in progress.** Code, figures, and the dashboard are still being finalized
> ahead of the public release. Expect rough edges.

Code, figures, and an interactive dashboard for the paper **"Mapping AI Exposure
Across the U.S. Workforce: Evidence from Millions of AI Conversations"** (Wright,
Schwarze, Boyd, 2026).

The project measures how current AI capability maps onto the U.S. workforce. It
combines real-world AI usage (Anthropic's Claude, Microsoft's Copilot), an
MCP-server capability pipeline, occupation structure from O*NET, and employment
and wage data from BLS, to estimate exposure across tasks, workers, wages, skills,
work activities, and more.

A project of Utah's Office of AI Policy (OAIP), supported by the BYU Department of
Mathematics.

## Links

- Paper: TBA
- Dashboard: https://ai-workforce-exposure-public.vercel.app/
- Main paper and dashboard repository (this repo): https://github.com/theodorewright11/ai-workforce-exposure-public
- Dataset-construction repository : https://github.com/theodorewright11/ai-workforce-exposure-dataset-construction-public
- Final datasets (HuggingFace): https://huggingface.co/datasets/theodorewright11/ai-workforce-exposure-datasets-public
- MCP → O\*NET classification repository: https://github.com/theodorewright11/mcp-onet-task-classification-public
- MPC datasets (HuggingFace): https://huggingface.co/datasets/theodorewright11/mcp-onet-task-classification-public

---

## What's here

```
paper_figures/   Regenerate every figure in the paper + supplement
  lib/figure_data.py   Dataset pins + work-time exposure / usage helpers
  lib/builders/        One module per figure family
  tests/               Unit tests for the helpers and layout invariants
dashboard/       Interactive dashboard (FastAPI backend + Next.js frontend)
backend/         Shared compute engine (powers both the figures and the dashboard)
data/            Datasets (committed; see "Data" below)
```

**The dashboard** has three pages:

- **My Occupation** — look up any of the 923 occupations and see what share of its
  workday current AI reaches, what share it doesn't, and which kinds of work those
  are, down to the individual task.
- **Economy at a Glance** — the whole picture in six blocks: the headline share and
  its trend, the eight verb families, occupational categories, work activities, the
  31 occupations that are both heavily exposed and projected to shed employment,
  and states ranked by the exposure of their job mix.
- **Explore the Data** — the research surface: all five configurations, every
  aggregation level, geography, trends and projections.

Every number the dashboard shows is produced by the same code as the corresponding
paper figure, so the two cannot drift. How to run it and how the code is organized
are in [`dashboard/README.md`](dashboard/README.md).

**The figures** regenerate from `paper_figures/`:

```bash
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
python paper_figures/run_main_figures.py          # 10 main-body figures
python paper_figures/run_supplemental_figures.py  # 14 supplemental figures
```

PNGs land in `paper_figures/figures/` (committed); the CSV behind every chart
lands in `paper_figures/results/` (regenerable, not committed). Each figure runs
independently; if one fails the rest still proceed. See
[`paper_figures/MAIN_FIGURES.md`](paper_figures/MAIN_FIGURES.md) and
[`paper_figures/SUPPLEMENTAL_FIGURES.md`](paper_figures/SUPPLEMENTAL_FIGURES.md)
for the rendered set.

Exposure in the figures is weighted by **work time**. Each (task, occupation)
row carries `time_per_day` — estimated hours per day, normalised so an
occupation's tasks sum to a 7-hour workday — so an exposure percentage is the
share of the workday AI reaches, employment-weighted at group level. Actual-usage
intensity is denominated by employment alone. The shared data layer is
`paper_figures/lib/figure_data.py`; tests are in `paper_figures/tests/`.

---

## Data

The datasets live in `./data/` (raw `final_*.csv`) and `./data/reference/` (O*NET
SKA and external-index reference files). They are derived from public sources;
source links and the construction pipeline are described in the paper's
Supplementary Materials. They are included in the repository so the figure scripts
and the dashboard backend have everything they need at build time.

---

## Citation

Wright, T., Schwarze, A. C., & Boyd, Z. M. (2026). *Mapping AI Exposure Across the
U.S. Workforce: Evidence from Millions of AI Conversations.*
