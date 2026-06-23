# Backend image for the public dashboard (FastAPI). Mirrors the internal
# dashboard's Docker setup. Build context = repo root.
#
# The app reuses backend/compute.py and paper_figures/lib, so both are copied;
# the path bootstrap in dashboard/api/__init__.py puts them on sys.path.
#
# DATA: this image expects the CSVs at /app/data. Two ways to provide them:
#   (a) commit data/ to the repo (it gets copied by the line below), or
#   (b) attach a Railway volume mounted at /app/data and remove the data COPY.
FROM python:3.12-slim

WORKDIR /app

COPY backend/ ./backend/
COPY paper_figures/lib/ ./paper_figures/lib/
COPY dashboard/ ./dashboard/
COPY data/ ./data/

RUN pip install --no-cache-dir -r dashboard/api/requirements.txt

# $PORT is provided by Railway.
CMD uvicorn dashboard.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
