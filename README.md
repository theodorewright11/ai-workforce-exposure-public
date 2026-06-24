# **Dashboard Link**: https://ai-workforce-exposure-public.vercel.app/

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

The datasets live in `./data/` (raw `final_*.csv`) and `./data/reference/` (O*NET
SKA and external-index reference files). They are derived from public sources;
source links and the construction pipeline are described in the paper's
Supplementary Materials. They are included in the repository so the figure scripts
and the dashboard backend have everything they need at build time.

---

## Citation

Wright, T., Schwarze, A. C., & Boyd, Z. M. (2026). *Mapping AI Exposure Across the
U.S. Workforce: Evidence from Millions of AI Conversations.*
