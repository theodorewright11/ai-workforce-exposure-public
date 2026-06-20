"""
paper_config.py — Visual palette and formatting constants for paper charts.

Defines a consistent visual language for all paper figures. Based on the
workforce_meeting_v2 readable style (big text, fill space), adapted for
publication use. Colors are muted/washed-out to match the correlation
heatmap aesthetic.
"""
from __future__ import annotations

import plotly.graph_objects as go

from lib.utils import COLORS, FONT_FAMILY

# ── Canvas dimensions ────────────────────────────────────────────────────
PAPER_W: int = 1400
PAPER_H: int = 787

# ── Print geometry ───────────────────────────────────────────────────────
# Every paper figure is placed at this column width in the manuscript.
# Print pt for a font of `f` px on a `W`-px-wide canvas:
#   pt = f × PAPER_COL_INCHES × PT_PER_INCH / W
# Inverting to size fonts to a pt target:
#   px = pt × W / (PAPER_COL_INCHES × PT_PER_INCH)
PAPER_COL_INCHES: float = 6.5
PT_PER_INCH: float = 72.0

# ── Standardized font ladder (print pt @ 6.5" column) ────────────────────
# Single source of truth. Every paper figure resolves its chrome from this
# via paper_fonts(width). Ratios: title 1.22× ticks · panel/axis 1.11× ticks.
FONT_PT_LADDER: dict[str, float] = {
    "title":          11.0,   # Chart title
    "panel_title":    10.0,   # Subplot / facet title
    "axis_title":     10.0,   # Axis title
    "tick":            9.0,   # Tick labels
    "legend":          9.0,   # Legend items
    "in_chart_floor":  8.0,   # Hard floor — data labels, cell text, annotations
}

# Hierarchy invariants — enforced at import so an invalid ladder crashes
# the chart script instead of shipping a malformed figure.
assert (
    FONT_PT_LADDER["title"]
    >= FONT_PT_LADDER["panel_title"]
    >= FONT_PT_LADDER["axis_title"]
    >= FONT_PT_LADDER["tick"]
    >= FONT_PT_LADDER["in_chart_floor"]
), "FONT_PT_LADDER violates hierarchy: title ≥ panel ≥ axis ≥ tick ≥ floor"
assert FONT_PT_LADDER["legend"] == FONT_PT_LADDER["tick"], (
    "FONT_PT_LADDER: legend must equal tick size"
)
assert FONT_PT_LADDER["in_chart_floor"] >= 8.0, (
    "FONT_PT_LADDER: in_chart_floor cannot drop below 8 pt (paper readability rule)"
)


def paper_fonts(canvas_width_px: int = PAPER_W) -> dict[str, int]:
    """Resolve the standardized pt ladder into pixel sizes for a given canvas width.

    Use this in every paper figure so the printed pt size stays constant
    regardless of canvas width. Sizes are rounded to integer px.
    """
    px_per_pt = canvas_width_px / (PAPER_COL_INCHES * PT_PER_INCH)
    return {role: max(1, round(pt * px_per_pt)) for role, pt in FONT_PT_LADDER.items()}


def paper_floor_px(canvas_width_px: int = PAPER_W) -> int:
    """Minimum legal font size in px for in-chart text on a given canvas.

    Any data label, cell value, annotation, or inside-bar text must be ≥ this.
    """
    return paper_fonts(canvas_width_px)["in_chart_floor"]


# ── Typography (px) — derived from the ladder at default PAPER_W ─────────
# These names are kept for back-compat with existing chart scripts. New
# code should call paper_fonts(width) directly so off-1400 canvases scale.
_PX = paper_fonts(PAPER_W)
TITLE_FS: int = _PX["title"]
SUBTITLE_FS: int = _PX["tick"]      # subtitles removed from paper, kept for legacy callers
INSIDE_FS: int = _PX["tick"]        # Primary values inside bars
OUTSIDE_FS: int = _PX["tick"]       # Secondary info outside bars / data labels
TICK_FS: int = _PX["tick"]          # Axis tick labels
LABEL_FS: int = _PX["axis_title"]   # Axis titles
LEGEND_FS: int = _PX["legend"]      # Legend items
ANNOT_FS: int = _PX["in_chart_floor"]   # Footnotes — at the floor
HEATMAP_TEXT_FS: int = _PX["tick"]      # Correlation values inside heatmap cells
TABLE_HEADER_FS: int = _PX["tick"]      # Table column headers
TABLE_CELL_FS: int = _PX["in_chart_floor"]   # Table cell text — at the floor

# ── Five-config colors ───────────────────────────────────────────────────
CONFIG_COLORS: dict[str, str] = {
    "all_confirmed":      "#3a5f83",
    "all_ceiling":        "#4a7c6f",
    "human_conversation": "#c05621",
    "agentic_confirmed":  "#7b5ea7",
    "agentic_ceiling":    "#2e8b8b",
}


# ── Paper-internal dataset overrides ─────────────────────────────────────
# Some paper static charts need a different dataset than the analysis-wide
# ANALYSIS_CONFIGS value. Keeping the override here — instead of repointing
# ANALYSIS_CONFIGS — means exploratory/claude_lab/extcompare scripts that
# consume the same config keys stay on the "natural" file (e.g. the eco_2015
# AEI family) and don't silently shift baselines.
#
# Current overrides:
#   - agentic_confirmed: use the eco_2025-rebased AEI API file so paper
#     static charts compare cleanly against agentic_ceiling (= MCP + API,
#     also eco_2025). Trend series (ANALYSIS_CONFIG_SERIES) stays on the
#     natural eco_2015 family.
#
# Used by paper builders that iterate CONFIG_ORDER / OVERVIEW_CONFIG_ORDER
# and by direct lookups (e.g. part_3 _agentic_ceiling_top10).
PAPER_CONFIG_DATASET_OVERRIDES: dict[str, str] = {
    "agentic_confirmed": "AEI API 2025 2026-02-12",
}


def paper_dataset_for(config_key: str) -> str:
    """Resolve a config key to its dataset name, applying paper-internal
    overrides. Falls back to ANALYSIS_CONFIGS[key] when no override exists."""
    from lib.config import ANALYSIS_CONFIGS
    return PAPER_CONFIG_DATASET_OVERRIDES.get(config_key, ANALYSIS_CONFIGS[config_key])

# ── Three-metric colors (muted, cohesive) ────────────────────────────────
# Tasks = blue · Workers = gold · Wages = green-teal (money association).
METRIC_COLORS: dict[str, str] = {
    "tasks":   "#4a7a94",   # Muted slate blue
    "workers": "#c4a76c",   # Warm tan / gold
    "wages":   "#6a9e8f",   # Muted sage teal (money green)
}

# Lighter shades for ceiling lines (and other "secondary" overlays).
METRIC_COLORS_LIGHT: dict[str, str] = {
    "tasks":   "#90b3c4",   # Lighter slate blue
    "workers": "#dcc395",   # Lighter gold
    "wages":   "#9bc1b3",   # Lighter sage
}

# ── Heatmap scale ────────────────────────────────────────────────────────
HEATMAP_LOW: str = "#f0e6d3"
HEATMAP_HIGH: str = "#0d2b45"

# ── Trend line colors — match the aggregate (overview) palette so the
# narrative reads consistently. all_ceiling lightened for legibility on
# overlapping line charts.
TREND_COLORS: dict[str, str] = {
    "all_confirmed": "#3a5f83",   # CONFIG_COLORS slate blue
    "all_ceiling":   "#7eaa9d",   # CONFIG_COLORS teal green, lightened
}

# ── Full palette (consolidated reference) ────────────────────────────────
PAPER_PALETTE: dict[str, str] = {
    **CONFIG_COLORS,
    **METRIC_COLORS,
    "text":       COLORS["text"],
    "text_dark":  "#0a0a0a",      # Darker text for heatmap cells
    "muted":      COLORS["muted"],
    "neutral":    COLORS["neutral"],
    "grid":       COLORS["grid"],
    "surface":    "#ffffff",
    "page":       COLORS["bg_page"],
    "positive":   COLORS["positive"],
    "negative":   COLORS["negative"],
    # Table accent colors
    "row_highlight": "#e8f0f7",   # Light blue for start/end rows
    "cell_pos":      "#e6f4ea",   # Light green for positive deltas
    "row_ref":       "#f5f5f0",   # Light cream for reference row
}


# ── Paper-specific formatters ────────────────────────────────────────────

def fmt_wages(val: float) -> str:
    """Format wages with T/B/M/K units."""
    sign = "-" if val < 0 else ""
    av = abs(val)
    if av >= 1e12:
        return f"{sign}${av / 1e12:.1f}T"
    if av >= 1e9:
        return f"{sign}${av / 1e9:.1f}B"
    if av >= 1e6:
        return f"{sign}${av / 1e6:.1f}M"
    if av >= 1e3:
        return f"{sign}${av / 1e3:.0f}K"
    return f"{sign}${av:.0f}"


def fmt_workers(val: float) -> str:
    """Format workers with M/K units."""
    sign = "-" if val < 0 else ""
    av = abs(val)
    if av >= 1e6:
        return f"{sign}{av / 1e6:.1f}M"
    if av >= 1e3:
        return f"{sign}{av / 1e3:.0f}K"
    return f"{sign}{int(av)}"


def fmt_date(iso: str) -> str:
    """Format '2025-03-06' → 'March 6, 2025'."""
    from datetime import datetime
    dt = datetime.strptime(iso, "%Y-%m-%d")
    return dt.strftime("%B %d, %Y").replace(" 0", " ")


# ── Figure styling ───────────────────────────────────────────────────────

def style_paper_figure(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    width: int = PAPER_W,
    height: int = PAPER_H,
    margin: dict | None = None,
) -> go.Figure:
    """Apply consistent paper styling to a Plotly figure.

    Font sizes are derived from the standardized pt ladder via paper_fonts(width),
    so off-1400 canvases (e.g. convergence matrices) print at the same pt size as
    everything else. Subtitles are intentionally ignored — paper figures carry no
    in-image subtitle; that text belongs in the figure caption.
    """
    if subtitle:
        # Subtitles are not rendered on paper figures. Caller can keep passing
        # one for back-compat; we silently drop it.
        pass

    px = paper_fonts(width)
    m = margin or dict(l=20, r=60, t=90, b=70)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=px["title"], color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left",
        ),
        font=dict(family=FONT_FAMILY, color=PAPER_PALETTE["text"]),
        plot_bgcolor=PAPER_PALETTE["surface"],
        paper_bgcolor=PAPER_PALETTE["surface"],
        width=width,
        height=height,
        margin=m,
        legend=dict(
            font=dict(size=px["legend"], family=FONT_FAMILY),
            orientation="h",
            yanchor="top", y=-0.08, xanchor="left", x=0,
        ),
    )

    fig.update_xaxes(
        gridcolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title_font=dict(size=px["axis_title"], family=FONT_FAMILY),
    )
    fig.update_yaxes(
        gridcolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        title_font=dict(size=px["axis_title"], family=FONT_FAMILY),
    )

    return fig
