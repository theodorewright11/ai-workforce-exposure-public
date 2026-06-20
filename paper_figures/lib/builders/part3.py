"""
Part 3 — Action: What To Do About It

Audience scaffolding will be reintroduced as more charts come in.

  1. Conv → Confirmed → Ceiling gap by major occ category
  2. Tech commodities composite (top-25)
  3. Risk score 5f — Occupations Most At Risk (SKA-gated focused set)
  4. U.S. states clustered on AI exposure (choropleth map)
  5. AI intensity vs. median-rank anchor (full eco_2025)

Run from project root:
    venv/Scripts/python -m lib.builders.part3
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib.config import (
    REFERENCE_DIR,
    ANALYSIS_CONFIGS,
    ANALYSIS_CONFIG_LABELS,
    ROOT,
    ensure_results_dir,
    get_pct_tasks_affected,
)
from lib.utils import FONT_FAMILY, save_csv, save_figure
from lib.paper_config import (
    PAPER_W, PAPER_H,
    ANNOT_FS, LABEL_FS, TICK_FS, INSIDE_FS,
    METRIC_COLORS, METRIC_COLORS_LIGHT, PAPER_PALETTE,
    paper_fonts, paper_dataset_for,
    style_paper_figure, fmt_wages, fmt_workers,
)
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
ANALYSIS_DATA_DIR = REFERENCE_DIR
TECH_SKILLS_FILE = ANALYSIS_DATA_DIR / "technology_skills_v30.1.csv"

PRIMARY_KEY = "all_confirmed"
PRIMARY_DATASET = ANALYSIS_CONFIGS[PRIMARY_KEY]
PRIMARY_LABEL = ANALYSIS_CONFIG_LABELS[PRIMARY_KEY]

# Intensity figures (intensity_anchor_fulleco in Part 3 + intensity_drivers
# and underadoption_gap in the appendix) use an AEI-only, eco_2025-rebased
# pool instead of PRIMARY_DATASET. Equal 3-source debias (Claude/Copilot/
# ChatGPT GWA priors) still applies — the bias prior is dataset-agnostic.
_INTENSITY_DATASET = "AEI Both 2025 2026-02-12"

# Tasks blue + workers green blend, light → dark (used by tech_commodities)
BLEND_LIGHT = "#cdd9d4"
BLEND_DARK = "#2a4f56"
TASKS_LIGHT = "#cfe0ec"
TASKS_DARK = "#2c4f6b"


# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────

def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


def _run_config(dataset_name: str, agg_level: str) -> pd.DataFrame:
    """Run get_group_data for one dataset and return a category dataframe."""
    from backend.compute import get_group_data
    config = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": "all",
        "geo": "nat",
        "agg_level": agg_level,
        "sort_by": "% Tasks Affected",
        "top_n": 9999,
        "search_query": "",
        "context_size": 3,
    }
    data = get_group_data(config)
    assert data is not None, f"No data for {dataset_name} @ {agg_level}"
    df: pd.DataFrame = data["df"]
    return df.rename(columns={data["group_col"]: "category"})


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Tech commodities composite (reuse skills_landscape pipeline)
# ─────────────────────────────────────────────────────────────────────────

def _structural_for_tech() -> pd.DataFrame:
    """Per-occ structural data for tech-skills join."""
    from backend.compute import get_explorer_occupations
    rows = [
        {
            "title_current": o["title_current"],
            "emp": o.get("emp") or 0,
            "wage": o.get("wage") or 0,
            "major": o.get("major", ""),
        }
        for o in get_explorer_occupations()
    ]
    return pd.DataFrame(rows)


def build_tech_commodities(results: Path, figures: Path) -> None:
    """Top 25 software commodities selected by total employment usage
    (each (occ, tool) entry contributes the occ's emp once — same occ counts
    multiple times across its tools), then ranked by mean % tasks affected.

    No AI data enters the selection — only the ordering and bar length.
    Bar length = mean % tasks affected (all_confirmed). Color = workers using.
    """
    assert TECH_SKILLS_FILE.exists(), f"Tech skills file not found: {TECH_SKILLS_FILE}"
    pct = get_pct_tasks_affected(PRIMARY_DATASET)
    structural = _structural_for_tech()

    tech = pd.read_csv(TECH_SKILLS_FILE)
    tech.columns = [c.strip() for c in tech.columns]
    tech = tech.merge(
        structural.rename(columns={"title_current": "Title"}),
        on="Title", how="left",
    )
    pct_merge = pct.rename("pct").reset_index()
    pct_merge.columns = ["Title", "pct"]
    tech = tech.merge(pct_merge, on="Title", how="left")
    tech["pct"] = tech["pct"].fillna(0.0)
    tech["emp"] = tech["emp"].fillna(0.0)

    # Selection: top 25 commodities by Σ emp across entries (same occ
    # counts once per tool it lists under the commodity — by design).
    agg = (
        tech.groupby("Commodity Title")
        .agg(
            workers_using=("emp", "sum"),
            mean_pct_affected=("pct", "mean"),
            n_occs=("Title", "nunique"),
            n_entries=("Title", "size"),
        )
        .reset_index()
    )
    top = agg.sort_values("workers_using", ascending=False).head(25).copy()
    # Display order: by % tasks affected (the ranking dimension).
    top = top.sort_values("mean_pct_affected", ascending=False)
    save_csv(top, results / "tech_commodities_top25.csv", float_format="%.3f")

    plot = top.sort_values("mean_pct_affected", ascending=True)  # plotly bottom-up
    # Y-axis: strip the trailing "software" word from each commodity
    # name (all 25 end with it — the title and axis title carry context).
    plot["display_name"] = (
        plot["Commodity Title"].str.replace(r"\s+software$", "", regex=True)
    )

    # ── Layout / fonts: pull every size from paper_fonts(W) so print pt
    # stays on the standard ladder (no `-2` adjustments, no hardcoded px).
    W = PAPER_W
    px = paper_fonts(W)

    wk_vals = plot["workers_using"].to_numpy(dtype=float)
    wk_min, wk_max = float(wk_vals.min()), float(wk_vals.max())

    # Color scale anchored on the workers METRIC_COLORS gold. Light-to-dark
    # so the eye reads color as "more workers using this commodity".
    WORKERS_LIGHT = "#f1e6cc"
    WORKERS_DARK = METRIC_COLORS["workers"]

    pct_max = float(plot["mean_pct_affected"].max())

    MARGIN_L, MARGIN_R = 20, 140

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot["mean_pct_affected"],
        y=plot["display_name"],
        orientation="h",
        marker=dict(
            color=wk_vals,
            colorscale=[[0, WORKERS_LIGHT], [1, WORKERS_DARK]],
            cmin=wk_min, cmax=wk_max,
            showscale=False,                # legend drawn manually below
            line=dict(width=0),
        ),
        text=[f"{p:.0f}%" for p in plot["mean_pct_affected"]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["text"], family=FONT_FAMILY),
        constraintext="none",
        cliponaxis=False,
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>% tasks affected: %{x:.1f}%<extra></extra>",
    ))

    # Bottom legend: single centered annotation. Inline color swatches
    # interpolating WORKERS_LIGHT → WORKERS_DARK give a gradient-like
    # cue without needing separate shapes. We use 7 swatches for a
    # smoother light-to-dark transition.
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    rgb_l = _hex_to_rgb(WORKERS_LIGHT)
    rgb_d = _hex_to_rgb(WORKERS_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"Workers Using&nbsp;&nbsp;{fmt_workers(wk_min)}&nbsp;"
        f"{swatch_html}&nbsp;{fmt_workers(wk_max)}"
    )
    # Note: xref="paper" x=0.5 centers on the *rendered* plot area, but
    # Plotly auto-expands the left margin to fit long y-tick labels, so
    # the rendered plot area is shifted right of the PNG center. The
    # tuned x value below lands the legend centered on the PNG.
    fig.add_annotation(
        x=0.14, y=-0.17,
        xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    # Right-side annotation per bar: just "N occs" (workers is encoded
    # in bar color and read off the bottom legend).
    label_x = pct_max * 1.04
    for i, occ in enumerate(plot["n_occs"]):
        fig.add_annotation(
            x=label_x, y=i,
            xref="x", yref="y",
            text=f"{int(occ)} occs",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=px["in_chart_floor"],
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

    # Force every text element to honor the 8 pt floor — no auto-shrink.
    fig.update_layout(
        uniformtext=dict(minsize=px["in_chart_floor"], mode="show"),
        bargap=0.22,
    )

    style_paper_figure(
        fig,
        "AI Exposure of the 25 Most-Used Software Commodities",
        height=1280, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=110, b=190),
    )
    x_top = pct_max * 1.15
    fig.update_xaxes(
        title=dict(text="Software Use Exposed",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showticklabels=True, ticksuffix="%",
        range=[0, x_top],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )
    # Force every commodity name to render — Plotly auto-thins category
    # ticks when there are many rows; dtick=1 keeps all 25.
    fig.update_yaxes(
        title=dict(text="O*NET Software Commodity",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickmode="linear", tick0=0, dtick=1,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )

    save_figure(fig, results / "figures" / "tech_commodities.png", scale=2)
    _copy_fig(results, figures, "tech_commodities.png")
    print("  -> tech_commodities.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Agentic Confirmed → Agentic Ceiling MCP extension
# (top 10 major occ + top 10 GWA, % tasks | workers | wages)
# ─────────────────────────────────────────────────────────────────────────

# Local copy of the part_2 GWA loader — keeps part_3 independent of cross-part
# import paths. The agentic_confirmed dataset resolved via paper_dataset_for()
# (eco_2025-rebased AEI API) and agentic_ceiling are both is_aei=False, so both
# come back as "mcp_group" with matching baselines.
def _get_wa_data(dataset_name: str, level: str = "gwa") -> pd.DataFrame:
    from backend.compute import compute_work_activities
    settings = {
        "selected_datasets": [dataset_name],
        "combine_method": "Average",
        "method": "freq",
        "use_auto_aug": True,
        "physical_mode": "all",
        "geo": "nat",
        "sort_by": "workers_affected",
        "top_n": 9999,
    }
    result = compute_work_activities(settings)
    group = result.get("mcp_group") or result.get("aei_group")
    if group is None:
        return pd.DataFrame()
    rows = group.get(level, [])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _axis_max_and_ticks(max_val: float) -> tuple[float, list[float]]:
    """Tight axis range + clean tick values. Mirrors the part_2 helper."""
    import math
    if max_val <= 0:
        return 1.0, [0.0]
    range_max = max_val * 1.05
    raw_step = range_max / 3.0
    magnitude = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / magnitude
    if norm < 1.5:
        step = 1.0 * magnitude
    elif norm < 2.25:
        step = 2.0 * magnitude
    elif norm < 3.0:
        step = 2.5 * magnitude
    elif norm < 7.0:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude
    n_max = int(range_max / step)
    ticks = [float(step * i) for i in range(n_max + 1)]
    return float(range_max), ticks


def _strip_zero_decimal(s: str) -> str:
    for unit in ("M", "B", "K", "T"):
        s = s.replace(f".0{unit}", unit)
    return s


def _balanced_wrap(label: str, max_chars: int = 22) -> str:
    if len(label) <= max_chars:
        return label
    candidates: list[tuple[int, int, str, str]] = []
    for i in range(1, len(label)):
        if label[i - 1] in (",", " "):
            line1 = label[:i].rstrip(", ").rstrip()
            line2 = label[i:].lstrip()
            if not line1 or not line2:
                continue
            if label[i - 1] == ",":
                line1 = line1 + ","
            candidates.append((max(len(line1), len(line2)), i, line1, line2))
    if not candidates:
        return label
    candidates.sort(key=lambda t: (t[0], t[1]))
    _, _, line1, line2 = candidates[0]
    return f"{line1}<br>{line2}"


def _wrap_major_label(label: str, max_chars: int = 22) -> str:
    return _balanced_wrap(label.replace(" Occupations", ""), max_chars)


def _wrap_gwa_label(label: str, max_chars: int = 32) -> str:
    return _balanced_wrap(label, max_chars)


def _agentic_ceiling_top10(level: str) -> pd.DataFrame:
    """Load agentic_confirmed + agentic_ceiling at the requested level
    (major or gwa), merge on category, return the top 10 sorted by
    agentic_confirmed % tasks descending. Each row carries pct_conf,
    pct_ceil, and pct_gap = pct_ceil − pct_conf so callers can re-rank
    by gap for the second panel.

    Both configs are is_aei=False on the eco_2025 baseline, so subtraction
    is clean (no mixed task universe / WA mapping)."""
    if level == "major":
        conf = _run_config(paper_dataset_for("agentic_confirmed"), "major")
        ceil = _run_config(paper_dataset_for("agentic_ceiling"), "major")
    else:
        conf = _get_wa_data(paper_dataset_for("agentic_confirmed"), level)
        ceil = _get_wa_data(paper_dataset_for("agentic_ceiling"), level)

    keep = ["category", "pct_tasks_affected"]
    df = (
        conf[keep].rename(columns={"pct_tasks_affected": "pct_conf"})
        .merge(
            ceil[keep].rename(columns={"pct_tasks_affected": "pct_ceil"}),
            on="category", how="inner",
        )
    )
    df["pct_gap"] = (df["pct_ceil"] - df["pct_conf"]).clip(lower=0)
    top10 = df.sort_values("pct_conf", ascending=False).head(10).reset_index(drop=True)
    return top10


def _build_agentic_ceiling_panels(
    top10: pd.DataFrame,
    title: str,
    y_axis_title: str,
    wrap_fn,
    out_name: str,
    results: Path,
    figures: Path,
    *,
    row_height: int = 110,
    hspacing: float = 0.30,
) -> None:
    """Two-panel horizontal bar chart, both panels showing % tasks
    exposed for the same 10 categories but ranked differently:

      Panel 1: agentic_confirmed % tasks, sorted by this value desc
               ("where agentic AI is most prominently used today")
      Panel 2: ceiling gap (= agentic_ceiling − agentic_confirmed),
               same 10 categories, re-sorted by gap desc
               ("where the biggest untapped agentic tooling lives")

    Each panel uses its own y-axis ordering — `shared_yaxes=False` —
    because the rank differs between panels. The y-labels appear on
    both panels' left edges so readers can match categories by name.
    """
    # Panel 1: sorted by pct_conf desc; reverse for plotly bottom-up.
    df1 = top10.iloc[::-1].reset_index(drop=True)
    cats1_r = [wrap_fn(c) for c in df1["category"].tolist()]
    vals1   = df1["pct_conf"].tolist()

    # Panel 2: same 10 cats, re-sorted by gap desc; reverse for plotly.
    df2 = top10.sort_values("pct_gap", ascending=False).reset_index(drop=True)
    df2 = df2.iloc[::-1].reset_index(drop=True)
    cats2_r = [wrap_fn(c) for c in df2["category"].tolist()]
    vals2   = df2["pct_gap"].tolist()

    n_cats = len(cats1_r)

    CANVAS_W = 2000
    px = paper_fonts(CANVAS_W)
    TITLE_FS_  = px["title"]
    PANEL_FS_  = px["panel_title"]
    AXIS_FS_   = px["axis_title"]
    TICK_FS_   = px["tick"]
    BAR_FS_    = px["in_chart_floor"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Agentic Usage",
            "Unused Agentic Tooling",
        ],
        horizontal_spacing=hspacing,   # GWA-level labels are longer phrases
                                        # than major-cat labels and need
                                        # more horizontal_spacing before
                                        # panel-2 y-labels stop overlapping
                                        # back into panel 1's plot area.
        shared_yaxes=False,         # each panel has its own ordering
    )

    inside_font  = dict(size=BAR_FS_, color="white",                    family=FONT_FAMILY)
    outside_font = dict(size=BAR_FS_, color=PAPER_PALETTE["text"],     family=FONT_FAMILY)

    panels = [
        {
            "col": 1,
            "cats": cats1_r,
            "vals": vals1,
            "color": METRIC_COLORS["tasks"],
            "axis_title": "Tasks Exposed",
            "hover_name": "Agentic Confirmed",
        },
        {
            "col": 2,
            "cats": cats2_r,
            "vals": vals2,
            "color": METRIC_COLORS_LIGHT["tasks"],
            "axis_title": "Tasks (Ceiling − Confirmed)",
            "hover_name": "Ceiling Gap",
        },
    ]

    for p in panels:
        vals = p["vals"]
        vmax = max(vals) if vals else 1.0
        # Inside (white) when bar wide enough to legibly hold "XX.X%";
        # otherwise outside in dark text past the bar end. 0.30 catches
        # the GWA panel-2 short bars (~12–15% on a 47% panel max) and
        # places their labels outside cleanly.
        threshold = 0.30 * vmax
        positions = ["inside" if v >= threshold else "outside" for v in vals]
        labels    = [f"{v:.1f}%" for v in vals]

        fig.add_trace(go.Bar(
            y=p["cats"], x=vals, orientation="h",
            marker=dict(color=p["color"], line=dict(width=0)),
            text=labels,
            textposition=positions,
            insidetextanchor="end",
            insidetextfont=inside_font,
            outsidetextfont=outside_font,
            textangle=0,
            showlegend=False,
            cliponaxis=False, constraintext="none",
            hovertemplate=(
                "<b>%{y}</b><br>" + p["hover_name"] + ": %{x:.1f}%<extra></extra>"
            ),
        ), row=1, col=p["col"])

    bottom_margin = 150
    height = max(PAPER_H + 200, n_cats * row_height + 360)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=TITLE_FS_, color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left",
            y=0.99, yanchor="top",
        ),
        font=dict(family=FONT_FAMILY, color=PAPER_PALETTE["text"]),
        plot_bgcolor=PAPER_PALETTE["surface"],
        paper_bgcolor=PAPER_PALETTE["surface"],
        width=CANVAS_W,
        height=height,
        margin=dict(l=110, r=110, t=140, b=bottom_margin),
        bargap=0.22,
        showlegend=False,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        showline=True, linecolor=PAPER_PALETTE["grid"],
        zeroline=True, zerolinecolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=TICK_FS_, family=FONT_FAMILY),
        tickangle=0,
        ticklabelstandoff=8,
    )
    fig.update_yaxes(
        showgrid=False, showline=False,
        tickfont=dict(size=TICK_FS_, family=FONT_FAMILY),
        automargin=True,
        ticklabelstandoff=2,
        ticks="",
        ticklen=0,
    )

    # Per-panel axis range with ~20% padding past the max so the outside
    # value labels at the longest bar's end print without clipping.
    for p in panels:
        vmax = max(p["vals"]) if p["vals"] else 0.0
        _rng, ticks = _axis_max_and_ticks(vmax)
        padded_max = vmax * 1.20 if vmax > 0 else _rng
        fig.update_xaxes(
            range=[0, padded_max], tickvals=ticks,
            ticktext=[f"{int(v)}%" for v in ticks],
            title=dict(text=p["axis_title"],
                       font=dict(size=AXIS_FS_, family=FONT_FAMILY)),
            row=1, col=p["col"],
        )
    fig.update_yaxes(
        title=dict(text=y_axis_title,
                   font=dict(size=AXIS_FS_, family=FONT_FAMILY), standoff=20),
        row=1, col=1,
    )

    # Subplot titles render via fig.layout.annotations — restyle to the
    # panel-title pt from the paper font ladder.
    for ann in fig.layout.annotations:
        ann.font = dict(size=PANEL_FS_, color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    save_figure(fig, results / "figures" / out_name, scale=2)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")


def build_agentic_ceiling_major(results: Path, figures: Path) -> None:
    """Two-panel chart for the top 10 major occ categories ranked by
    agentic_confirmed % tasks. Left panel: current agentic use. Right
    panel: the ceiling gap (untapped MCP tooling), same 10 categories
    re-sorted by gap desc."""
    top10 = _agentic_ceiling_top10("major")
    save_csv(top10, results / "agentic_ceiling_major.csv", float_format="%.3f")
    _build_agentic_ceiling_panels(
        top10,
        title="Agentic Confirmed vs. Agentic Ceiling Gap — Top 10 Major Occupational Categories",
        y_axis_title="Major Occupational Category",
        wrap_fn=_wrap_major_label,
        out_name="agentic_ceiling_major.png",
        results=results,
        figures=figures,
        row_height=110,
        hspacing=0.30,
    )


def build_agentic_ceiling_gwa(results: Path, figures: Path) -> None:
    """Top 10 GWAs ranked by agentic_confirmed % tasks exposed. Same
    two-panel structure as the major chart."""
    top10 = _agentic_ceiling_top10("gwa")
    save_csv(top10, results / "agentic_ceiling_gwa.csv", float_format="%.3f")
    _build_agentic_ceiling_panels(
        top10,
        title="Agentic Confirmed vs. Agentic Ceiling Gap — Top 10 General Work Activities",
        y_axis_title="O*NET General Work Activity",
        wrap_fn=_wrap_gwa_label,
        out_name="agentic_ceiling_gwa.png",
        results=results,
        figures=figures,
        row_height=110,
        hspacing=0.42,
    )


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: AI intensity vs. median-rank anchor (chart 15 from
# exploratory/pct_norm_vs_eco v3) — major occ categories ranked by
# Σ pct (rated, bias-corrected) / Σ (freq × emp) over FULL eco_2025.
# Bars colored by pct_tasks_affected (darker = higher).
# Imports at function level so part_3 can still run if exploratory/ is
# absent (folder is gitignored).
# ─────────────────────────────────────────────────────────────────────────

def build_intensity_anchor_fulleco(results: Path, figures: Path) -> None:
    try:
        from lib.exploratory.intensity import (
            BIAS_VARIANTS, compute_bias_ratios,
        )
        from lib.exploratory.intensity_v3 import (
            compute_v3_intensity,
            compute_major_full_eco_denominator,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/audit_pct_norm_eco not available ({exc})")
        return

    # Intensity-figure dataset: AEI Conv + AEI API pooled onto eco_2025
    # (final_aei_all_usage_2025_2026-02-12.csv). Drops Microsoft from the
    # numerator while keeping the equal 3-source bias correction below
    # (the bias prior is GWA-level and applies regardless of dataset).
    base = compute_v3_intensity(
        "aei_all_eco2025", compute_bias_ratios(BIAS_VARIANTS["equal"])
    ).copy()
    full_den = compute_major_full_eco_denominator()
    base["den_full"] = base["category"].map(full_den).fillna(0.0)
    base["ratio_full"] = np.where(
        base["den_full"] > 0, base["num"] / base["den_full"], 0.0
    )
    total_full = base["ratio_full"].sum()
    base["ratio_full_pct"] = (
        base["ratio_full"] / total_full * 100.0 if total_full > 0 else 0.0
    )
    # Pull pct_tasks_affected from the same dashboard pipeline so the
    # colorbar values are consistent with the intensity dataset (also
    # AEI-only, eco_2025-rebased). NOTE: this intentionally diverges from
    # the Part 2 major_categories chart's colorbar (which uses PRIMARY_DATASET
    # = AEI Both + Micro) — the intensity figure series stays on the
    # no-Microsoft dataset end-to-end.
    major_df = _run_config(_INTENSITY_DATASET, "major")
    pct_aff = major_df.set_index("category")["pct_tasks_affected"]
    base["pct_tasks_affected"] = base["category"].map(pct_aff).fillna(0.0)

    # Anchor major: Office and Administrative Support — a near-median major
    # category used as the x=1 reference for the lift axis.
    anchor_major = "Office and Administrative Support Occupations"
    anchor_val = base.loc[base["category"] == anchor_major, "ratio_full_pct"].iloc[0]
    assert anchor_val > 0, f"Anchor value for {anchor_major} must be > 0"
    base["lift"] = base["ratio_full_pct"] / anchor_val
    median_lift = float(base["lift"].median())

    # Debiased usage-mass share, the bias-corrected counterpart to raw_pct.
    # raw_pct sums to 100% across the 22 majors (Σ pct_normalized is already a
    # share of total economy usage); the debiased adj_pct sums to >100 because
    # the GWA bias correction rescales the mass, so we renormalize to 100% to
    # keep the two columns directly comparable as shares.
    num_total = float(base["num"].sum())
    base["debias_pct"] = np.where(
        num_total > 0, base["num"] / num_total * 100.0, 0.0
    )

    out = base[["category", "ratio_full_pct", "lift", "pct_tasks_affected",
                "raw_pct", "debias_pct"]].copy()
    out["anchor_value"] = anchor_val
    out["median_lift"] = median_lift
    save_csv(
        out.sort_values("lift", ascending=False),
        results / "intensity_anchor_fulleco.csv",
        float_format="%.4f",
    )

    plot_df = base.sort_values("lift", ascending=True).reset_index(drop=True)
    # Strip the redundant " Occupations" suffix from y-tick display labels.
    plot_df["display_category"] = (
        plot_df["category"].str.replace(r"\s*Occupations\s*$", "", regex=True)
    )
    cvals = plot_df["pct_tasks_affected"].to_numpy(dtype=float)
    cmin, cmax = float(cvals.min()), float(cvals.max())

    W = PAPER_W
    px = paper_fonts(W)

    # Per-bar inside/outside text decision — only the few very wide bars
    # get the value label inside in white (matches part_2 major_categories
    # style); everything else keeps the dark label outside so the number
    # stays legible. Threshold at 20% of x_top keeps just the top three
    # (Life/Phys/Sci, Computer/Math, Arts) inside; Architecture (4.11×) and
    # below read as dark outside labels.
    x_top = max(plot_df["lift"]) * 1.04
    INSIDE_THRESHOLD = x_top * 0.20
    pos = [
        "inside" if v >= INSIDE_THRESHOLD else "outside"
        for v in plot_df["lift"]
    ]
    # Axis headroom beyond the longest bar so the top bars' inside labels
    # aren't crammed against the right edge and the bar ends clear BOTH
    # right-margin columns (Σ debias pct sits in the plot-area tail and can't
    # shift further right). 18% squishes the bars slightly to open that tail
    # so the inner column has clean whitespace to land in.
    x_axis_max = max(plot_df["lift"]) * 1.22
    inside_font  = dict(size=px["tick"], color="white",                family=FONT_FAMILY)
    outside_font = dict(size=px["tick"], color=PAPER_PALETTE["text"], family=FONT_FAMILY)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["display_category"], x=plot_df["lift"], orientation="h",
        marker=dict(
            color=cvals,
            colorscale=[[0, TASKS_LIGHT], [1, TASKS_DARK]],
            cmin=cmin, cmax=cmax,
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{v:.2f}x" for v in plot_df["lift"]],
        textposition=pos,
        insidetextanchor="end",
        insidetextfont=inside_font,
        outsidetextfont=outside_font,
        constraintext="none",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>lift: %{x:.2f}x<extra></extra>",
        showlegend=False,
    ))

    # Median reference line — anchor is set so it sits at x = 1
    fig.add_vline(
        x=1.0, line_dash="dash",
        line_color=PAPER_PALETTE["negative"], line_width=1.5,
    )
    fig.add_annotation(
        x=1.0, y=1.005,
        xref="x", yref="paper",
        text="median",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=px["in_chart_floor"], color=PAPER_PALETTE["negative"], family=FONT_FAMILY),
    )

    # Two right-margin numeric columns in the whitespace beyond the bars:
    # the bias-corrected Σ debias pct (inner) and the un-debiased Σ raw pct
    # (outer, the same quantity the appendix intensity_drivers charts print
    # beside each bar). Both are usage-mass shares that sum to 100% across the
    # 22 majors — raw_pct from Σ pct_normalized, debias_pct from Σ adj_pct
    # renormalized to 100% (debias reallocates mass, so the comparable view is
    # the redistributed share, matching the audit pipeline's "renormed to 100%"
    # convention). Printing them side by side exposes how the GWA debias
    # reweights mass: Computer & Math falls 43.7% -> 28.2%, Arts rises
    # 8.7% -> 12.5%. The bars keep their bias-corrected lift labels. Fixed
    # %.1f keeps a constant decimal width so right-anchoring lines the decimals
    # into clean columns; two-line headers keep each column narrow. Both sit
    # in the right whitespace the original chart already had (the squished
    # plot-area tail past the bars + the original ~92 px margin), so the total
    # right whitespace matches the single-column original — the outer Σ raw pct
    # stays at its original 1.06 position and the inner Σ debias pct lands in
    # the gridline-free tail just left of it.
    COL_X_DEBIAS = 0.93  # paper x — right edge of the inner (debias) column
    COL_X_RAW = 1.06     # paper x — right edge of the outer (raw) column
    for col_x, header, field in (
        (COL_X_DEBIAS, "Σ debias<br>pct", "debias_pct"),
        (COL_X_RAW, "Σ raw<br>pct", "raw_pct"),
    ):
        fig.add_annotation(
            x=col_x, y=1.005, xref="paper", yref="paper",
            text=header, showarrow=False,
            xanchor="right", yanchor="bottom", align="right",
            font=dict(size=px["in_chart_floor"], color=PAPER_PALETTE["text"], family=FONT_FAMILY),
        )
        for cat, val in zip(plot_df["display_category"], plot_df[field]):
            fig.add_annotation(
                x=col_x, y=cat, xref="paper", yref="y",
                text=f"{val:.1f}%", showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(size=px["tick"], color=PAPER_PALETTE["text"], family=FONT_FAMILY),
            )

    # Bottom legend: HTML-swatch gradient matching tech_commodities style.
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    rgb_l = _hex_to_rgb(TASKS_LIGHT)
    rgb_d = _hex_to_rgb(TASKS_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"Tasks Exposed&nbsp;&nbsp;{cmin:.0f}%&nbsp;"
        f"{swatch_html}&nbsp;{cmax:.0f}%"
    )
    # Compact layout: 22 majors × 38 px/row + 90/170 margins ≈ 1096 px
    # (~5.1" at PAPER_W=1400). Matches the risk_score_5f shortening pattern
    # (commit 4600633) at a slightly looser pitch since major labels are
    # single-line after stripping the " Occupations" suffix. Bottom margin
    # bumped to 170 to clear the gradient legend below the x-axis title.
    n = len(plot_df)
    MARGIN_T, MARGIN_B = 90, 170
    chart_h = n * 38 + MARGIN_T + MARGIN_B
    plot_h_px = chart_h - MARGIN_T - MARGIN_B

    # Bottom legend — pixel-anchored ~50 px above the canvas bottom so it
    # never clips when chart_h changes. Centered on the PNG (paper x=0.14
    # matches the part_3 tech_commodities layout's PNG-centered legend
    # given the same l=20 margin).
    legend_y = -(MARGIN_B - 50) / plot_h_px
    fig.add_annotation(
        x=0.14, y=legend_y,
        xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    style_paper_figure(
        fig,
        "Actual AI Usage as a Multiple of Median Usage",
        subtitle=(
            "Σ pct usage ÷ Σ (freq × employment) for equalization — debiased to a "
            "Claude / Copilot / ChatGPT GWA-distribution blend (work-related ChatGPT chats)."
        ),
        height=chart_h, width=W,
        # r kept at the original single-column width — both numeric columns
        # (Σ debias pct, Σ raw pct) fit in the existing right whitespace
        # (squished plot-area tail + this margin), so the chart's right edge
        # matches the original. Σ raw pct keeps its small edge clearance so the
        # trailing "%" never clips.
        margin=dict(l=20, r=92, t=MARGIN_T, b=MARGIN_B),
    )
    fig.update_layout(bargap=0.15)
    fig.update_xaxes(
        title=dict(text="Usage Relative to Median (×)", font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        range=[0, x_axis_max],
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )
    # tickmode="array" pins every category label — plotly auto-thins
    # categorical ticks at this tight pitch otherwise.
    y_labels = list(plot_df["display_category"])
    fig.update_yaxes(
        title=dict(text="Major Occupational Category", font=dict(size=px["axis_title"], family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        tickmode="array", tickvals=y_labels, ticktext=y_labels,
    )

    save_figure(fig, results / "figures" / "intensity_anchor_fulleco.png", scale=2)
    _copy_fig(results, figures, "intensity_anchor_fulleco.png")
    print("  -> intensity_anchor_fulleco.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Risk Score Audit — Section 5f (SKA-gated focused 43)
# Pulls audit's flag-frame + focused-set builders, renders in paper style.
# Skips gracefully if exploratory/risk_score_audit isn't available.
# ─────────────────────────────────────────────────────────────────────────

def build_risk_score_5f(results: Path, figures: Path) -> None:
    try:
        from lib.exploratory.risk_score import (
            _load_flag_df, _build_focused_set,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/audit_risk_score not available ({exc})")
        return

    flags_df = _load_flag_df()
    sub = _build_focused_set(flags_df)
    s5f = sub[sub["ska_gated"] == 1].copy()
    s5f["abs_emp"] = s5f["emp_proj_pct"].abs()
    s5f = s5f.sort_values("abs_emp", ascending=True).reset_index(drop=True)

    save_csv(
        s5f.sort_values("abs_emp", ascending=False)[
            ["title_current", "major_short", "job_zone", "emp_proj_pct",
             "pct", "ska_pct", "pct_delta", "workers_affected", "wages_affected"]
        ],
        results / "risk_score_5f.csv",
        float_format="%.3f",
    )

    pct_min_f = float(s5f["pct"].min())
    pct_max_f = float(s5f["pct"].max())

    # Composition counts for the caption / prose (printed to stdout, also
    # saved alongside the per-occ CSV so the numbers are reproducible).
    zone_counts = (
        s5f["job_zone"].astype(int).value_counts().sort_index()
    )
    major_counts = s5f["major_short"].value_counts()
    print(f"  -> risk_score_5f composition (n={len(s5f)}):")
    print("       by job zone:")
    for z, c in zone_counts.items():
        print(f"         zone {z}: {c}")
    print("       by major occ category:")
    for m, c in major_counts.items():
        print(f"         {m}: {c}")
    counts_rows = (
        [{"group": "job_zone", "label": f"zone {int(z)}", "count": int(c)}
         for z, c in zone_counts.items()]
        + [{"group": "major", "label": m, "count": int(c)}
           for m, c in major_counts.items()]
    )
    save_csv(pd.DataFrame(counts_rows), results / "risk_score_5f_counts.csv")

    W = PAPER_W + 280
    px = paper_fonts(W)
    floor_px = px["in_chart_floor"]
    tick_px = px["tick"]
    axis_px = px["axis_title"]

    def _truncate_title(s: str, max_len: int = 50) -> str:
        # Keep each y-label on a single line: if the title overruns max_len,
        # cut at the last comma or space inside the window and append "…".
        if len(s) <= max_len:
            return s
        breakers = [i for i in range(max_len) if s[i] in ", "]
        cut = max(breakers) if breakers else max_len - 1
        return s[:cut].rstrip(" ,") + "…"

    y_labels = [_truncate_title(t) for t in s5f["title_current"]]

    # Plotly's colorbar.x interpretation in v6 doesn't line up cleanly
    # with plot-area paper coords, so we hand-draw the legend the same way
    # build_tech_commodities() does — full positioning control for
    # label-left + gradient-right, centered on the whole PNG.
    MARGIN_L, MARGIN_R = 520, 120
    MARGIN_T, MARGIN_B = 90, 150
    plot_area_px = W - MARGIN_L - MARGIN_R
    canvas_center_paper = (W / 2 - MARGIN_L) / plot_area_px

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y_labels, x=s5f["abs_emp"], orientation="h",
        marker=dict(
            color=s5f["pct"].values,
            colorscale=[[0, TASKS_LIGHT], [1, TASKS_DARK]],
            cmin=pct_min_f, cmax=pct_max_f,
            showscale=False,                # legend drawn manually below
            line=dict(width=0),
        ),
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>emp proj: -%{x:.1f}%<extra></extra>",
    ))

    # Bottom legend: single centered annotation with inline HTML swatches,
    # mirroring the build_tech_commodities() pattern. All on one line:
    #   "Tasks Exposed   51% [■■■■■■■] 87%"
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))
    rgb_l = _hex_to_rgb(TASKS_LIGHT)
    rgb_d = _hex_to_rgb(TASKS_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"Tasks Exposed&nbsp;&nbsp;{pct_min_f:.0f}%&nbsp;"
        f"{swatch_html}&nbsp;{pct_max_f:.0f}%"
    )
    # xref="paper" x=0.5 centers on the rendered plot area, not on the PNG.
    # With a large explicit left margin (and Plotly's tendency to expand
    # it further to fit long y-tick labels), the plot area sits right of
    # the PNG center. The x below was empirically tuned to land the legend
    # centered on the PNG for this margin pair; retune if margins change.
    fig.add_annotation(
        x=0.10, y=-0.08,
        xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=floor_px,
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    n = len(s5f)
    # All labels are single-line (truncated). y-tick drops to the 8 pt
    # floor and per-row to 32 so the 38-row chart fits a single page —
    # height lands ~1450 px → ~6.7" at the 1400 px / 6.5" canvas.
    height = max(600, n * 32 + MARGIN_T + MARGIN_B)

    # Tight x range so bars fill more of the chart width. Tasks column
    # sits right after the longest bar; x_top is set so the tasks text
    # ends close to the right edge of the plot area.
    x_max = float(s5f["abs_emp"].max())
    # "87% tasks" text width in x units (rough: 9 chars × 0.55 × floor_px).
    tasks_text_w_px = int(0.55 * floor_px * len("87% tasks"))
    tasks_label_x = x_max * 1.04
    # x_top: tasks_label_x + text width (converted to x units) + small pad.
    # Iterative solve since x_per_px depends on x_top — one pass is enough
    # because the relationship is monotone and the pad absorbs the error.
    approx_x_per_px = (x_max * 1.20) / max(plot_area_px, 1)
    x_top = tasks_label_x + tasks_text_w_px * approx_x_per_px + 1.5

    # In-bar emp_proj label: only when the bar is wide enough to hold the
    # text inside without overflowing into the bar's left edge.
    # Approx text width: 7 chars (e.g. "-36.1%") × 0.55 × floor_px.
    inside_text_w_px = int(0.55 * floor_px * 7)
    inside_pad_px = 18
    needed_px = inside_text_w_px + inside_pad_px
    x_per_px = x_top / max(plot_area_px, 1)
    inside_threshold = needed_px * x_per_px

    # Per-bar dynamic text color for the inside label: white on darker
    # bars, dark on lighter ones.
    pct_mid = (pct_min_f + pct_max_f) / 2

    for i, row in s5f.iterrows():
        # Tasks number — always at fixed x, left-anchored, neutral color.
        fig.add_annotation(
            x=tasks_label_x, y=y_labels[i],
            xref="x", yref="y",
            text=f"{row['pct']:.0f}% tasks",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=floor_px,
                      color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
        )

        # Proj number — inside the bar (right-aligned at bar end) if it
        # fits, else just outside the bar end (left-anchored).
        proj_text = f"{row['emp_proj_pct']:+.1f}%"
        if row["abs_emp"] >= inside_threshold:
            text_color = "white" if row["pct"] >= pct_mid else PAPER_PALETTE["text_dark"]
            fig.add_annotation(
                x=row["abs_emp"], y=y_labels[i],
                xref="x", yref="y",
                text=proj_text,
                showarrow=False,
                xanchor="right", yanchor="middle",
                xshift=-8,
                font=dict(size=floor_px, color=text_color, family=FONT_FAMILY),
            )
        else:
            fig.add_annotation(
                x=row["abs_emp"], y=y_labels[i],
                xref="x", yref="y",
                text=proj_text,
                showarrow=False,
                xanchor="left", yanchor="middle",
                xshift=4,
                font=dict(size=floor_px,
                          color=PAPER_PALETTE["neutral"], family=FONT_FAMILY),
            )

    style_paper_figure(
        fig,
        "Occupations with High AI Exposure and Negative Employment Projection",
        height=height, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    # X-axis ticks (and gridlines) stop at 40%, even though x_top extends
    # further to leave room for the "% tasks" annotation column.
    fig.update_xaxes(
        title=dict(text="BLS Projected Employment 2024–2034 (%)",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"], ticksuffix="%",
        range=[0, x_top],
        tickmode="array", tickvals=[0, 10, 20, 30, 40],
        tickfont=dict(size=tick_px, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        title=dict(text="Occupation", font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickfont=dict(size=floor_px, family=FONT_FAMILY),
        # Force every category label — plotly auto-thins categorical
        # ticks at this tight pitch otherwise.
        tickmode="array", tickvals=y_labels, ticktext=y_labels,
    )
    fig.update_layout(bargap=0.15)

    save_figure(fig, results / "figures" / "risk_score_5f.png", scale=2)
    _copy_fig(results, figures, "risk_score_5f.png")
    print("  -> risk_score_5f.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 (variant): risk_score_5f_workers — two-panel rank-by-tasks view.
# Same SKA-gated focused set as risk_score_5f, but flipped framing:
#   Panel A: x = % tasks exposed (in-bar: tasks %)
#   Panel B: x = workers exposed (in-bar: formatted count)
#   Right column: emp_proj_pct, one straight vertical line
#   Y-order:  shared, descending by % tasks exposed
#   Color:    |emp_proj_pct| — darker = larger employment decline
# Replaces risk_score_5f in main() — see results.md / README.md.
# ─────────────────────────────────────────────────────────────────────────

EMP_LIGHT = "#efd9c2"   # at-risk gradient: light tan → deep burgundy
EMP_DARK = "#7a2e1f"


def build_risk_score_5f_workers(results: Path, figures: Path) -> None:
    try:
        from lib.exploratory.risk_score import (
            _load_flag_df, _build_focused_set,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/audit_risk_score not available ({exc})")
        return

    flags_df = _load_flag_df()
    sub = _build_focused_set(flags_df)
    s5f = sub[sub["ska_gated"] == 1].copy()

    # Sort by % tasks exposed DESCENDING — largest at top of each panel.
    # Plotly horizontal bars render bottom-up, so pass ascending=True.
    s = s5f.sort_values("pct", ascending=True).reset_index(drop=True)

    save_csv(
        s5f.sort_values("pct", ascending=False)[
            ["title_current", "major_short", "job_zone", "emp_proj_pct",
             "pct", "workers_affected", "wages_affected"]
        ],
        results / "risk_score_5f_workers.csv",
        float_format="%.3f",
    )

    def _truncate_title(t: str, max_len: int = 50) -> str:
        if len(t) <= max_len:
            return t
        breakers = [i for i in range(max_len) if t[i] in ", "]
        cut = max(breakers) if breakers else max_len - 1
        return t[:cut].rstrip(" ,") + "…"

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    def _format_workers(w: float) -> str:
        if w >= 1_000_000:
            return f"{w/1_000_000:.1f}M"
        if w >= 1_000:
            return f"{w/1_000:.0f}K"
        return f"{w:.0f}"

    y_labels = [_truncate_title(t) for t in s["title_current"]]

    abs_emp = s["emp_proj_pct"].abs()
    cmin = float(abs_emp.min())
    cmax = float(abs_emp.max())
    abs_mid = (cmin + cmax) / 2

    W = PAPER_W + 380
    px = paper_fonts(W)
    floor_px = px["in_chart_floor"]
    tick_px = px["tick"]
    axis_px = px["axis_title"]
    panel_px = px["panel_title"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["% Tasks Exposed", "Workers Exposed"],
        shared_yaxes=True,
        horizontal_spacing=0.04,
        column_widths=[0.4, 0.6],
    )

    common_marker = dict(
        color=abs_emp.values,
        colorscale=[[0, EMP_LIGHT], [1, EMP_DARK]],
        cmin=cmin, cmax=cmax,
        showscale=False,
        line=dict(width=0),
    )

    fig.add_trace(go.Bar(
        y=y_labels, x=s["pct"], orientation="h",
        marker=common_marker, showlegend=False,
        hovertemplate=(
            "<b>%{y}</b><br>tasks exposed: %{x:.1f}%"
            "<br>emp proj: %{customdata:+.1f}%<extra></extra>"
        ),
        customdata=s["emp_proj_pct"].values,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=y_labels, x=s["workers_affected"], orientation="h",
        marker=common_marker, showlegend=False,
        hovertemplate=(
            "<b>%{y}</b><br>workers exposed: %{x:,.0f}"
            "<br>emp proj: %{customdata:+.1f}%<extra></extra>"
        ),
        customdata=s["emp_proj_pct"].values,
    ), row=1, col=2)

    # Right-column emp_proj line + Panel B x-range padding.
    MARGIN_L, MARGIN_R = 380, 110
    MARGIN_T, MARGIN_B = 150, 160
    plot_area_px = W - MARGIN_L - MARGIN_R
    pct_max = float(s["pct"].max())
    wrk_max = float(s["workers_affected"].max())
    inside_threshold_pct = 0.20 * pct_max
    # 0.20 so Shipping/Receiving (~19% of wrk_max) lands outside its bar.
    inside_threshold_wrk = 0.20 * wrk_max
    proj_col_x = wrk_max * 1.18
    x_top_b = wrk_max * 1.34

    for i, row in s.iterrows():
        is_dark = abs(row["emp_proj_pct"]) >= abs_mid
        text_color_inside = "white" if is_dark else PAPER_PALETTE["text_dark"]

        # Panel A in-bar: tasks-exposed %.
        pct_text = f"{row['pct']:.0f}%"
        if row["pct"] >= inside_threshold_pct:
            fig.add_annotation(
                x=row["pct"], y=y_labels[i], xref="x1", yref="y1",
                text=pct_text, showarrow=False,
                xanchor="right", yanchor="middle", xshift=-6,
                font=dict(size=floor_px, color=text_color_inside, family=FONT_FAMILY),
            )
        else:
            fig.add_annotation(
                x=row["pct"], y=y_labels[i], xref="x1", yref="y1",
                text=pct_text, showarrow=False,
                xanchor="left", yanchor="middle", xshift=4,
                font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                          family=FONT_FAMILY),
            )

        # Panel B in-bar: formatted workers count.
        w = row["workers_affected"]
        w_text = _format_workers(w)
        if w >= inside_threshold_wrk:
            fig.add_annotation(
                x=w, y=y_labels[i], xref="x2", yref="y2",
                text=w_text, showarrow=False,
                xanchor="right", yanchor="middle", xshift=-6,
                font=dict(size=floor_px, color=text_color_inside, family=FONT_FAMILY),
            )
        else:
            fig.add_annotation(
                x=w, y=y_labels[i], xref="x2", yref="y2",
                text=w_text, showarrow=False,
                xanchor="left", yanchor="middle", xshift=4,
                font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                          family=FONT_FAMILY),
            )

        # Right column: emp_proj_pct at fixed x past Panel B's bars.
        fig.add_annotation(
            x=proj_col_x, y=y_labels[i], xref="x2", yref="y2",
            text=f"{row['emp_proj_pct']:+.1f}%", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                      family=FONT_FAMILY),
        )

    # Right-column header.
    fig.add_annotation(
        x=proj_col_x, y=1.0, xref="x2", yref="y2 domain",
        text="Emp Proj", showarrow=False,
        xanchor="left", yanchor="bottom", yshift=4,
        font=dict(size=floor_px, color=PAPER_PALETTE["neutral"],
                  family=FONT_FAMILY),
    )

    n = len(s)
    height = max(620, n * 32 + MARGIN_T + MARGIN_B)

    style_paper_figure(
        fig,
        "High AI Exposure × Negative Employment Projection — Tasks vs. Workers",
        height=height, width=W,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )

    # Subplot titles: scale + push up for whitespace.
    for ann in fig.layout.annotations[:2]:
        ann.font = dict(size=panel_px, color=PAPER_PALETTE["text"], family=FONT_FAMILY)
        ann.yshift = 12

    fig.update_xaxes(
        title=dict(text="% Tasks Exposed",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"], ticksuffix="%",
        tickfont=dict(size=tick_px, family=FONT_FAMILY),
        row=1, col=1,
    )
    fig.update_xaxes(
        title=dict(text="Workers Exposed",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        tickfont=dict(size=tick_px, family=FONT_FAMILY),
        range=[0, x_top_b],
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text="Occupation",
                   font=dict(size=axis_px, family=FONT_FAMILY)),
        showgrid=False, showline=False,
        tickfont=dict(size=floor_px, family=FONT_FAMILY),
        tickmode="array", tickvals=y_labels, ticktext=y_labels,
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, showline=False, showticklabels=False,
        row=1, col=2,
    )

    # Bottom legend — single HTML block, xanchor="center", empirically
    # tuned x. Same pattern as build_risk_score_5f(). x=0.104 calibrated
    # to land legend center at PNG center (measured via PIL on rendered
    # PNG — offset within 1 px of canvas center).
    rgb_l = _hex_to_rgb(EMP_LIGHT)
    rgb_d = _hex_to_rgb(EMP_DARK)
    N_SWATCH = 7
    swatch_html = ""
    for i in range(N_SWATCH):
        t = i / (N_SWATCH - 1)
        c = tuple(int(rgb_l[k] + (rgb_d[k] - rgb_l[k]) * t) for k in range(3))
        swatch_html += f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>■</span>"
    legend_text = (
        f"BLS Emp Proj 2024–2034 (more negative → darker)&nbsp;&nbsp;"
        f"-{cmin:.0f}%&nbsp;{swatch_html}&nbsp;-{cmax:.0f}%"
    )
    fig.add_annotation(
        x=0.104, y=-0.08,
        xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=floor_px, color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    fig.update_layout(bargap=0.15)

    save_figure(fig, results / "figures" / "risk_score_5f_workers.png", scale=2)
    _copy_fig(results, figures, "risk_score_5f_workers.png")
    print("  -> risk_score_5f_workers.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 5: State Exposure vs. Most-At-Risk Concentration
# Two-panel horizontal bar (% emp exposed | % emp in "Most At Risk" set).
# Computation runs through deepdive_state_signal (gitignored exploratory);
# skips gracefully if that folder isn't present.
# ─────────────────────────────────────────────────────────────────────────

def _name_to_postal() -> dict[str, str]:
    return {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL",
        "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
        "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
        "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
        "Wisconsin": "WI", "Wyoming": "WY",
    }


def build_state_clusters_map(results: Path, figures: Path) -> None:
    """Matplotlib version of the state clusters map.

    Uses a US states geojson + matplotlib so we can render true diagonal-
    stripe hatching on the states where Ward and K-means disagree. Each
    polygon's fill is its Ward cluster color; disagreement states get a
    diagonal hatch overlay in the K-means cluster's color. AK and HI render
    in inset axes below the contiguous map; DC renders as a labeled marker
    next to MD because its polygon is too small to see at this scale.
    """
    try:
        from lib.exploratory.state_clusters import (
            compute_clusters, OUTLIER_CLUSTER_ID,
            _load_state_features, CLUSTER_FEATURES, OUTLIER_GEOS,
            _pick_k_from_linkage, K_MIN, K_MAX,
        )
    except ImportError as exc:
        print(f"  -> SKIPPED: exploratory/deepdive_state_clusters not available ({exc})")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import geopandas as gpd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, fcluster

    # Pin k=3 so the cluster count stays locked even when upstream feature
    # values shift (e.g. when the SKA-gated focused set drives a new
    # focused_share_pct distribution). Same pin used by the appendix
    # state_clusters_each_ranked chart, so the two stay in sync.
    K_PIN = 3
    import lib.exploratory.state_clusters as _dsc_mod
    _orig_k_min, _orig_k_max = _dsc_mod.K_MIN, _dsc_mod.K_MAX
    _dsc_mod.K_MIN = _dsc_mod.K_MAX = K_PIN
    try:
        pkg = compute_clusters()
    finally:
        _dsc_mod.K_MIN, _dsc_mod.K_MAX = _orig_k_min, _orig_k_max
    state_df = pkg["state_df"]
    cluster_names = pkg["cluster_names"]
    cluster_color = pkg["cluster_color"]
    order = pkg["order"]

    # Persist the per-state cluster assignments alongside the figure so the
    # CSV can never drift out of sync with the rendered map again (the dense-
    # prose tables are transcribed from this CSV). Same columns/format the
    # appendix state_clusters_each_ranked.csv uses, since both come from the
    # identical compute_clusters() call.
    save_csv(
        state_df[["geo", "cluster", "cluster_name",
                  "pct_emp_wtd", "focused_share_pct"]],
        results / "state_clusters_map.csv",
        float_format="%.3f",
    )

    # Recompute K-means disagreement on the same input clustering went into.
    raw = _load_state_features()
    sub_in = raw[~raw["geo"].isin(OUTLIER_GEOS)].copy().reset_index(drop=True)
    Xz = StandardScaler().fit_transform(sub_in[CLUSTER_FEATURES].to_numpy(dtype=float))
    Z = linkage(Xz, method="ward")
    k_, _ = _pick_k_from_linkage(Z, K_PIN, K_PIN)
    ward_lab = fcluster(Z, t=k_, criterion="maxclust")
    km_lab = KMeans(n_clusters=k_, n_init=20, random_state=42).fit_predict(Xz) + 1
    aligned = np.zeros_like(km_lab)
    for w in np.unique(ward_lab):
        idx = ward_lab == w
        majority = pd.Series(km_lab[idx]).mode().iloc[0]
        aligned[km_lab == majority] = w
    for i, v in enumerate(aligned):
        if v == 0:
            aligned[i] = km_lab[i] + 100
    sub_in["ward_lab"] = ward_lab
    sub_in["km_lab"] = aligned
    disagree_lookup = {
        r["geo"].upper(): int(r["km_lab"])
        for _, r in sub_in.iterrows()
        if r["ward_lab"] != r["km_lab"]
    }

    # Per-state assignments (postal → ward cluster) for fill.
    state_cluster = {
        r["geo"].upper(): int(r["cluster"])
        for _, r in state_df.iterrows()
    }

    geojson_path = REFERENCE_DIR / "us_states_v2.geojson"
    assert geojson_path.exists(), f"Missing geojson: {geojson_path}"
    gdf_raw = gpd.read_file(geojson_path)
    n2p = _name_to_postal()
    gdf_raw["postal"] = gdf_raw["NAME"].map(n2p)
    gdf_raw = gdf_raw.dropna(subset=["postal"])

    # Contiguous: Albers Equal-Area Conic for the standard US look.
    contiguous = (
        gdf_raw[~gdf_raw["postal"].isin({"AK", "HI"})]
        .to_crs("EPSG:5070")
        .copy()
    )
    # AK and HI: project each to its own local CRS so insets render at a
    # reasonable scale and aspect ratio (EPSG:5070's bounds are designed
    # for the contiguous US — AK lands far off-projection and HI gets
    # mangled).
    ak = gdf_raw[gdf_raw["postal"] == "AK"].to_crs("EPSG:3338").copy()
    hi = gdf_raw[gdf_raw["postal"] == "HI"].to_crs("EPSG:26961").copy()

    # ── Figure layout: matches Plotly version's 1400×940 canvas at 6.5"
    # column width → 6.5 × 4.37 in. Top ~78% = map (with AK/HI insets in
    # the contiguous map's lower-left corner), bottom ~17% = legend
    # (centered horizontally below the map), with a thin title strip on
    # top. Font sizes are the paper ladder's print pt values, since
    # matplotlib's `fontsize=N` is already pt.
    FIG_W_IN = 6.5
    FIG_H_IN = 5.20  # taller than 1400×940 aspect to give legend more room
    DPI = 300
    TITLE_PT = 11
    LEGEND_PT = 9
    INSET_LABEL_PT = 7

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    # Main map fills the top portion; legend below.
    ax_main = fig.add_axes([0.01, 0.28, 0.98, 0.68])
    # AK / HI insets overlay the lower-left of the contiguous map (where
    # the Pacific would be) — same convention as Plotly's albers usa.
    ax_ak = fig.add_axes([0.02, 0.27, 0.16, 0.18])
    ax_hi = fig.add_axes([0.18, 0.28, 0.10, 0.11])
    ax_main.set_axis_off()
    ax_ak.set_axis_off()
    ax_hi.set_axis_off()
    ax_main.set_aspect("equal")
    ax_ak.set_aspect("equal")
    ax_hi.set_aspect("equal")

    BORDER_COLOR = "white"
    BORDER_W = 0.5
    # Matplotlib's default hatch line width is 1.0. Dial down slightly so
    # the diagonal stripes read lighter against small state polygons.
    plt.rcParams["hatch.linewidth"] = 0.7

    def _draw(gdf_part, ax):
        for _, row in gdf_part.iterrows():
            postal = row["postal"]
            cid = state_cluster.get(postal)
            if cid is None:
                continue
            ward_fill = cluster_color.get(int(cid), "#cccccc")
            # Base polygon — ward color fill, thin white border.
            gpd.GeoSeries([row.geometry]).plot(
                ax=ax, color=ward_fill, edgecolor=BORDER_COLOR,
                linewidth=BORDER_W,
            )
            # Overlay: if this state is a Ward/K-means disagreement, draw
            # diagonal stripes in the K-means cluster's color on top.
            if postal in disagree_lookup:
                km_cid = disagree_lookup[postal]
                km_fill = cluster_color.get(km_cid, "#777777")
                gpd.GeoSeries([row.geometry]).plot(
                    ax=ax, facecolor="none",
                    edgecolor=km_fill,
                    hatch="////",
                    linewidth=BORDER_W,
                )

    _draw(contiguous, ax_main)
    _draw(ak, ax_ak)
    _draw(hi, ax_hi)

    # DC is too tiny to see — draw as a labeled marker east of MD.
    if "DC" in state_cluster:
        dc_fill = cluster_color.get(int(state_cluster["DC"]), "#cccccc")
        # Find MD polygon centroid as the anchor, then offset east.
        md_geom = contiguous[contiguous["postal"] == "MD"].geometry.iloc[0]
        cx, cy = md_geom.centroid.x, md_geom.centroid.y
        offset_x = 200_000   # 200 km east, in Albers meters
        ax_main.plot(cx + offset_x, cy, marker="o", markersize=8,
                     markerfacecolor=dc_fill, markeredgecolor="black",
                     markeredgewidth=0.8, linestyle="none")
        ax_main.annotate(
            "DC", xy=(cx + offset_x, cy),
            xytext=(8, 0), textcoords="offset points",
            fontsize=7, va="center", color=PAPER_PALETTE["text"],
        )

    # ── Title (top-left, same convention as the Plotly version) ─────────
    fig.text(
        0.012, 0.965,
        "U.S. States Clustered on Workforce Exposure",
        fontsize=TITLE_PT, fontweight="bold",
        color=PAPER_PALETTE["text"], family="sans-serif",
        ha="left", va="top",
    )

    # ── Legend (right-side block) ──────────────────────────────────────
    # Cluster names are long mouthful strings like "Mid Workforce Exposed /
    # Highest Emp Share in High AI Exp & <0 Emp Proj Occs". Shorten /
    # wrap them for the legend so they don't truncate or push the map
    # margin.
    import textwrap

    def _short_cluster_label(label: str) -> str:
        # Outlier label inlines per-state values — drop those, the map
        # has them via DC's labeled marker.
        if "(outlier" in label:
            short = label.split(" (outlier")[0] + " (outlier)"
        else:
            # Replace the verbose "Emp Share in High AI Exp & <0 Emp Proj
            # Occs" with a shorter phrase that still keeps the meaning.
            short = label.replace(
                "Emp Share in High AI Exp & <0 Emp Proj Occs",
                "Focused-Set Share",
            )
        return "\n".join(textwrap.wrap(short, width=34))

    legend_handles = []
    for cid in order:
        legend_handles.append(
            Patch(facecolor=cluster_color[cid], edgecolor="white",
                  label=_short_cluster_label(cluster_names[cid]))
        )
    # Add disagreement entry — neutral gray swatch with hatch overlay,
    # mirroring the per-state stripe treatment so the reader knows
    # what the diagonal stripes mean.
    legend_handles.append(
        Patch(facecolor="#cccccc", edgecolor="#555555", hatch="////",
              label="Ward / K-means disagreement\n(stripe color = K-means)")
    )
    # Legend goes BELOW the map, centered horizontally — matches the
    # Plotly version's vertical legend at x=0.5, xanchor=center, y=0.18.
    # 6 entries (5 clusters + 1 disagreement); vertical orientation
    # would stack too tall, so use 2 columns to fit the bottom strip.
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.25),
        frameon=False, fontsize=LEGEND_PT,
        title=None,
        ncol=2,
        handlelength=1.6, handleheight=1.1,
        labelspacing=0.6, columnspacing=1.8,
    )
    for txt in leg.get_texts():
        txt.set_color(PAPER_PALETTE["text"])

    # AK / HI labels above their insets so the reader knows what they are.
    ax_ak.set_title("AK", fontsize=INSET_LABEL_PT,
                    color=PAPER_PALETTE["text"], loc="center", pad=2)
    ax_hi.set_title("HI", fontsize=INSET_LABEL_PT,
                    color=PAPER_PALETTE["text"], loc="center", pad=2)

    out_path = results / "figures" / "state_clusters_map.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches=None,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    shutil.copy(out_path, figures / "state_clusters_map.png")
    print("  -> state_clusters_map.png")




# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    results = ensure_results_dir(HERE)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)

    print("=" * 64)
    print("Part 3: Action — What To Do About It")
    print("=" * 64)

    print("\n[1/6] Agentic Confirmed vs. Agentic Ceiling — major occupational categories")
    build_agentic_ceiling_major(results, figures)

    print("\n[2/6] Agentic Confirmed vs. Agentic Ceiling — general work activities")
    build_agentic_ceiling_gwa(results, figures)

    print("\n[3/6] Tech commodities composite")
    build_tech_commodities(results, figures)

    print("\n[4/6] Risk score 5f workers — rank-by-tasks two-panel view")
    build_risk_score_5f_workers(results, figures)

    print("\n[5/6] U.S. states clustered on AI exposure (map)")
    build_state_clusters_map(results, figures)

    print("\n[6/6] AI intensity vs. median-rank anchor (full eco_2025)")
    build_intensity_anchor_fulleco(results, figures)

    print("\n" + "=" * 64)
    print("Part 3 complete — figures in results/figures/ and figures/")
    print("=" * 64)


if __name__ == "__main__":
    main()
