"""
utils.py — Shared chart styling, formatting, and output helpers for analysis scripts.

All question scripts should use these helpers for consistent output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# ── Color palette ─────────────────────────────────────────────────────────────
# Matches the dashboard frontend exactly (see frontend/src/app/globals.css
# and frontend/src/components/TrendsView.tsx).

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

COLORS = {
    "primary": "#3a5f83",       # Slate blue — Group A / main data series
    "secondary": "#4a7c6f",     # Teal green — Group B / comparison series
    "accent": "#c05621",        # Orange-brown — highlights / search match
    "positive": "#166534",      # Green — increases / gains
    "negative": "#991b1b",      # Red — decreases / losses
    "neutral": "#5a5a5a",       # Secondary text gray
    "muted": "#9b9b9b",         # Muted text gray
    "brand": "#1a6b5a",         # Brand teal
    "utah": "#c05621",          # Orange — Utah-specific
    "national": "#3a5f83",      # Slate blue — National
    "aei": "#3a5f83",           # Slate blue — AEI family (Group A color)
    "mcp": "#4a7c6f",           # Teal green — MCP family (Group B color)
    "microsoft": "#c05621",     # Orange — Microsoft
    "bg": "#ffffff",            # White surface background
    "bg_page": "#f7f7f4",       # Cream off-white page background
    "text": "#1a1a1a",          # Primary text
    "grid": "#e4e4de",          # Border / gridline color
    "border": "#e4e4de",        # Standard border
}

# Categorical palette for multi-series charts — matches TrendsView.tsx PALETTE
CATEGORY_PALETTE = [
    "#3a5f83", "#4a7c6f", "#c05621", "#7b5ea7",
    "#2d7d9a", "#6b8e23", "#b8860b", "#8b4513",
    "#4682b4", "#2e8b57", "#cd853f", "#708090",
    "#5b4e99", "#2d7a55", "#c45c29", "#3d6b9e",
]


# ── Figure styling ────────────────────────────────────────────────────────────

def style_figure(
    fig: go.Figure,
    title: str,
    *,
    subtitle: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    width: int = 1200,
    height: int = 700,
    source_text: str = "Source: AEA Dashboard — Utah OAIP",
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent professional styling to a Plotly figure.

    Call this on every figure before saving. It handles layout, fonts,
    colors, margins, and source attribution.
    """
    title_text = title
    if subtitle:
        title_text += f"<br><span style='font-size:13px;color:{COLORS['neutral']}'>{subtitle}</span>"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color=COLORS["text"], family=FONT_FAMILY),
            x=0.01,
            xanchor="left",
        ),
        font=dict(
            family=FONT_FAMILY,
            size=12,
            color=COLORS["text"],
        ),
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["bg"],
        width=width,
        height=height,
        margin=dict(l=60, r=40, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=COLORS["neutral"]),
        ) if show_legend else dict(visible=False),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=COLORS["neutral"])) if x_title else None,
            gridcolor=COLORS["grid"],
            showline=True,
            linewidth=1,
            linecolor=COLORS["grid"],
            tickfont=dict(size=11, color=COLORS["neutral"]),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=COLORS["neutral"])) if y_title else None,
            gridcolor=COLORS["grid"],
            showline=True,
            linewidth=1,
            linecolor=COLORS["grid"],
            tickfont=dict(size=11, color=COLORS["neutral"]),
        ),
    )

    # Source attribution in bottom-right
    fig.add_annotation(
        text=source_text,
        xref="paper", yref="paper",
        x=1.0, y=-0.18,
        showarrow=False,
        font=dict(size=10, color=COLORS["muted"], family=FONT_FAMILY),
        xanchor="right",
    )

    return fig


def _format_bar_label(val: float) -> str:
    """Format a number for bar labels — clean, abbreviated, no clutter."""
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.1f}B" if abs_val < 10_000_000_000 else f"${val / 1_000_000_000:.0f}B"
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M" if abs_val < 10_000_000 else f"{val / 1_000_000:.0f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.0f}K"
    if abs_val >= 1:
        return f"{val:.0f}"
    return f"{val:.1f}"


def make_horizontal_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    *,
    subtitle: Optional[str] = None,
    x_title: Optional[str] = None,
    color: str = COLORS["primary"],
    highlight_categories: Optional[list[str]] = None,
    highlight_color: str = COLORS["accent"],
    top_n: Optional[int] = None,
    value_format: Optional[str] = None,
    **style_kwargs,
) -> go.Figure:
    """Create a styled horizontal bar chart.

    Args:
        df: DataFrame with category and value columns.
        category_col: Column name for category labels.
        value_col: Column name for bar values.
        title: Chart title.
        subtitle: Optional subtitle line.
        x_title: X-axis label.
        color: Default bar color.
        highlight_categories: Categories to highlight in a different color.
        highlight_color: Color for highlighted categories.
        top_n: If set, take only the top N rows (assumes df is pre-sorted).
        value_format: Optional format string override (e.g., "%.1f%%"). If None,
            uses smart abbreviation (1.2M, 450K, etc.).
    """
    plot_df = df.head(top_n) if top_n else df

    # Build colors per bar
    colors = []
    for cat in plot_df[category_col]:
        if highlight_categories and cat in highlight_categories:
            colors.append(highlight_color)
        else:
            colors.append(color)

    # Format bar labels
    if value_format:
        text_labels = [value_format % v for v in plot_df[value_col]]
    else:
        text_labels = [_format_bar_label(v) for v in plot_df[value_col]]

    fig = go.Figure(go.Bar(
        x=plot_df[value_col],
        y=plot_df[category_col],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, color=COLORS["neutral"], family=FONT_FAMILY),
        cliponaxis=False,
    ))

    # Reverse y-axis so rank 1 is on top
    fig.update_yaxes(autorange="reversed")

    style_figure(
        fig, title,
        subtitle=subtitle,
        x_title=x_title,
        y_title=None,
        show_legend=False,
        **style_kwargs,
    )

    # Extra polish: more left margin for long labels, hide x-axis gridlines
    fig.update_layout(
        margin=dict(l=20, r=80),
        xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11)),
        bargap=0.25,
    )

    return fig


def make_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    *,
    subtitle: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    palette: Optional[list[str]] = None,
    **style_kwargs,
) -> go.Figure:
    """Create a styled multi-line chart for trends.

    Args:
        df: Long-format DataFrame.
        x_col: Column for x-axis (usually date).
        y_col: Column for y-axis (metric value).
        color_col: Column that defines each line (category name).
        title: Chart title.
    """
    palette = palette or CATEGORY_PALETTE
    categories = df[color_col].unique()

    fig = go.Figure()
    for i, cat in enumerate(categories):
        cat_df = df[df[color_col] == cat].sort_values(x_col)
        fig.add_trace(go.Scatter(
            x=cat_df[x_col],
            y=cat_df[y_col],
            mode="lines+markers",
            name=str(cat),
            line=dict(color=palette[i % len(palette)], width=3),
            marker=dict(size=5),
        ))

    style_figure(
        fig, title,
        subtitle=subtitle,
        x_title=x_title,
        y_title=y_title,
        **style_kwargs,
    )

    return fig


# ── Output helpers ────────────────────────────────────────────────────────────

def save_figure(
    fig: go.Figure,
    path: Path,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: int = 3,
) -> None:
    """Save a Plotly figure as a high-res PNG.

    Args:
        fig: The styled Plotly figure.
        path: Output file path (.png).
        width: Override width (uses figure's layout width if None).
        height: Override height (uses figure's layout height if None).
        scale: Resolution multiplier (3 = 300 DPI equivalent at default size).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(
        fig, str(path),
        width=width or fig.layout.width or 1200,
        height=height or fig.layout.height or 700,
        scale=scale,
    )


def save_csv(df: pd.DataFrame, path: Path, *, float_format: str = "%.2f") -> None:
    """Save a DataFrame as CSV with consistent formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False, float_format=float_format)


# ── Formatting helpers ────────────────────────────────────────────────────────

def format_workers(n: float) -> str:
    """Format worker counts with adaptive units (e.g., '1.2M', '450K')."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:.0f}"


def format_wages(n: float) -> str:
    """Format wage dollars with adaptive units (e.g., '$1.2B', '$450M')."""
    if abs(n) >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    elif abs(n) >= 1_000_000:
        return f"${n / 1_000_000:.0f}M"
    elif abs(n) >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:.0f}"


def format_pct(n: float) -> str:
    """Format a percentage value (e.g., 45.3 -> '45.3%')."""
    return f"{n:.1f}%"


def describe_config(config: dict) -> str:
    """Return a human-readable one-liner describing a compute config.

    Useful for figure subtitles and report annotations.
    Example: 'AEI v4 + MCP v4 + Microsoft | Average | Freq | National | All tasks | Auto-aug On'
    """
    datasets = ", ".join(config.get("selected_datasets", []))
    combine = config.get("combine_method", "Average")
    method = "Time" if config.get("method") == "freq" else "Value"
    geo = "National" if config.get("geo") == "nat" else "Utah"
    phys = {
        "all": "All tasks",
        "exclude": "Excl. physical",
        "only": "Physical only",
    }.get(config.get("physical_mode", "all"), "All tasks")
    aug = "Auto-aug On" if config.get("use_auto_aug") else "Auto-aug Off"

    parts = [datasets, combine, method, geo, phys, aug]
    return " | ".join(parts)


# ── PDF generation ───────────────────────────────────────────────────────────

def generate_pdf(md_path: Path, pdf_path: Path) -> None:
    """Convert a markdown report (with inline images) to a styled PDF.

    Uses markdown + xhtml2pdf (pure Python, no native dependencies).
    Image paths in the markdown are resolved relative to the markdown file.

    Args:
        md_path: Path to the source markdown file.
        pdf_path: Path for the output PDF file.
    """
    import re

    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError:
        print(f"  [skip PDF] install markdown + xhtml2pdf for PDF export")
        return

    md_text = md_path.read_text(encoding="utf-8")
    md_dir = md_path.parent

    # Convert relative image paths to absolute file paths for xhtml2pdf
    def _resolve_img(match: re.Match) -> str:
        alt = match.group(1)
        src = match.group(2)
        abs_src = (md_dir / src).resolve()
        # xhtml2pdf needs plain file paths, not file:// URIs
        return f"![{alt}]({abs_src})"

    md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _resolve_img, md_text)

    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    # Wrap in styled HTML matching the dashboard theme
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: Helvetica, Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: {COLORS['text']};
        padding: 24px;
    }}
    h1 {{
        font-size: 20pt;
        font-weight: bold;
        color: {COLORS['text']};
        border-bottom: 2px solid {COLORS['border']};
        padding-bottom: 6px;
        margin-top: 24px;
    }}
    h2 {{
        font-size: 15pt;
        font-weight: bold;
        color: {COLORS['primary']};
        border-bottom: 1px solid {COLORS['border']};
        padding-bottom: 4px;
        margin-top: 20px;
    }}
    h3 {{
        font-size: 12pt;
        font-weight: bold;
        color: {COLORS['secondary']};
        margin-top: 16px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 9pt;
    }}
    th {{
        background-color: {COLORS['bg_page']};
        border: 1px solid {COLORS['border']};
        padding: 6px 8px;
        text-align: left;
        font-weight: bold;
    }}
    td {{
        border: 1px solid {COLORS['border']};
        padding: 4px 8px;
    }}
    img {{
        max-width: 100%;
        margin: 12px 0;
    }}
    code {{
        background-color: {COLORS['bg_page']};
        padding: 1px 4px;
        font-size: 9pt;
    }}
    hr {{
        border: none;
        border-top: 1px solid {COLORS['border']};
        margin: 16px 0;
    }}
    strong {{
        font-weight: bold;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(pdf_path), "wb") as f:
        pisa.CreatePDF(html_doc, dest=f)
