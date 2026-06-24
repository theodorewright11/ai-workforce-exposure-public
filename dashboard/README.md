# *****Dashboard Link**: https://ai-workforce-exposure-public.vercel.app/

# AI Workforce Exposure — Interactive Dashboard

The interactive companion to the paper *Mapping AI Exposure Across the U.S.
Workforce*. A deliberately simple, read-only view of the paper's findings:

- **My Occupation** — pick an occupation, see its AI exposure: headline % tasks /
  workers / wages exposed, per-source task breakdown (Claude Conv / API / Copilot /
  MCP), work activities, skills-knowledge-abilities gap, similar occupations, and
  the software/MCP tools it relies on.
- **Explore the Data** — three views, each drillable through the SOC / work-activity
  hierarchy:
  1. **Occupation exposure** (Major → Minor → Broad → Occupation)
  2. **Work-activity exposure** (GWA → IWA → DWA)
  3. **Actual AI usage** — debiased usage intensity (× the median reference)

All exposure views are driven by the paper's **five data configurations**
(All Confirmed, Conversational Confirmed, Agentic Confirmed, All Sources Ceiling,
Agentic Ceiling), with an optional **trend over time** + **2-year projection**.
Frequency weighting and the auto-augmentation multiplier are always on (the paper's
primary settings); there is no A/B comparison or dataset cascade — researchers who
want that can use the data directly.

---

## Architecture

```
dashboard/
├── api/         FastAPI backend — thin layer over ../backend/compute.py + ../paper_figures/lib
│   ├── main.py              endpoints (config, exposure, children, trend, usage, occupation-report)
│   ├── usage_intensity.py   debiased usage-intensity (Actual Usage tab)
│   ├── occupation_report.py occupation page payload (ported from the internal dashboard)
│   └── requirements.txt
└── frontend/    Next.js 14 + React + TypeScript
    └── src/app/{occupation,data}   the two pages
```

The backend adds **no new computation** — it reuses the same engine that generates
the paper figures (`backend/compute.py`, `paper_figures/lib/`). See the paper's
Supplementary Materials for the methodology.

---

## Running locally

You need the datasets in `../data/` (see the repo root `README.md` for access).

**Backend** (from the repo root):

```bash
# install deps into the repo venv (pandas/numpy already there for the figures)
.venv/Scripts/python -m pip install -r dashboard/api/requirements.txt

# run the API (http://localhost:8000)
.venv/Scripts/python -m uvicorn dashboard.api.main:app --reload --port 8000
```

**Frontend** (separate terminal):

```bash
cd dashboard/frontend
npm install
npm run dev            # http://localhost:3000
```

The frontend reads `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`).

---

## Data it uses

Everything is already in `../data/` for the figure reproduction, plus two files the
occupation page needs: the AEI conversation snapshots (`final_aei_v1…v5.csv`) for the
per-source breakdown, and `mcp_titles_desc.csv` for MCP server descriptions. The
SKA/tech reference files live in `../data/reference/`.
