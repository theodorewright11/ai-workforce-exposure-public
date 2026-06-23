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

---

## What's here

```
paper_figures/   Regenerate every figure in the paper + supplement
dashboard/       Interactive dashboard (FastAPI backend + Next.js frontend)
backend/         Shared compute engine (powers both the figures and the dashboard)
data/            Datasets (not committed; see "Data" below)
```

**The dashboard** lets you look up any occupation, explore exposure across the SOC
and work-activity hierarchies, and see where AI is actually being used. How to run
it and how the dashboard code is organized are in
[`dashboard/README.md`](dashboard/README.md).

**The figures** regenerate from `paper_figures/`:

```bash
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
# obtain the datasets into ./data/ (see "Data" below), then:
python paper_figures/run_main_figures.py
python paper_figures/run_supplemental_figures.py
```

Each figure runs independently; if one fails (for example a missing dataset) the
rest still proceed.

---

## Data

The underlying datasets are **not committed to this repository**. They total
several hundred MB and are derived from public sources. The committed PNGs in
`paper_figures/figures/` are the rendered results; the raw data is only needed to
regenerate them or to run the dashboard backend.

> **Data access:** _[TBD before public release. Raw datasets will be hosted
> externally, with a download step here.]_ Source links and the construction
> pipeline are described in the paper's Supplementary Materials.

Once obtained, datasets go in `./data/` (raw `final_*.csv`) and `./data/reference/`
(O*NET SKA and external-index reference files).

---

## Citation

Wright, T., Schwarze, A. C., & Boyd, Z. M. (2026). *Mapping AI Exposure Across the
U.S. Workforce: Evidence from Millions of AI Conversations.*
