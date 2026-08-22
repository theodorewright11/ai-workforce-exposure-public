"""Actual-AI-usage driver charts — who inside a category is doing the using.

Ten supplement figures, all on one renderer so they read as one family:

  6 — top-10 occupations and top-10 tasks inside each of three majors
      (Life/Phys/Soc Sci, Arts & Design, Computer & Math).
  4 — top-10 DWAs inside each of the four GWAs that lead the GWA
      adoption ranking. No counterpart in the earlier figure set.

Every bar is a usage rate expressed as a multiple of a median, and the
number beside it is the sum of RAW usage pct. What changed from the paper
version:

  - 2026-05-31 snapshots via `figure_data.intensity_pairs` (AEI Conv + API
    pooled on eco_2025, no Microsoft — the equal 3-source debias assumes a
    Claude-only numerator).
  - The usage denominator is EMPLOYMENT alone, not freq x emp. See
    `figure_data.intensity_pairs` for why usage is not re-weighted by time.
  - Occupation shading is work time exposed, not freq-weighted % tasks.
  - No debiased-share column. The debias is a GWA-level prior on the
    numerator and belongs there; printing it as a percentage reads as an
    observation when it is a correction.

Anchoring follows the folder rule — anchor on the whole of the thing the
chart decomposes — with one deliberate exception, documented on
`dwa_usage_frame`.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from lib import figure_data
from lib.utils import FONT_FAMILY, save_csv, save_figure
from lib.paper_config import PAPER_W, PAPER_PALETTE, style_paper_figure, paper_fonts


# ─────────────────────────────────────────────────────────────────────────
# Shared renderer for all ten driver charts
# ─────────────────────────────────────────────────────────────────────────


def _copy_fig(results: Path, figures: Path, name: str) -> None:
    shutil.copy(results / "figures" / name, figures / name)


# ─────────────────────────────────────────────────────────────────────────
# Within-major intensity drivers — decomposes the three high-lift bars of
# the major-adoption chart (Life/Phys/Sci, Arts, Comp/Math)
# into the top-10 occupations and top-10 tasks driving each major's lift.
#
# Per occ (or per task) ratio = Σ debiased adj_pct / Σ (freq × emp), then
# normalized by the within-major median ratio so the dashed median line
# sits at x=1 and lifts read directly as "× the major's median row."
# ─────────────────────────────────────────────────────────────────────────

TARGET_MAJORS_DRIVERS: list[tuple[str, str, str]] = [
    ("Life, Physical, and Social Science Occupations",
     "life_phys_soc_sci",
     "Life, Physical & Social Science"),
    ("Arts, Design, Entertainment, Sports, and Media Occupations",
     "arts_design_ent",
     "Arts, Design & Entertainment"),
    ("Computer and Mathematical Occupations",
     "comp_math",
     "Computer and Mathematical"),
]


_ORPHAN_STARTS = {"and", "or", "but", "nor", "yet", "&", "the", "a", "an", "of",
                  "to", "in", "on", "for", "with", "by", "at", "from"}


def _balance_two_lines(words: list[str], width: int) -> str | None:
    """Find the word-boundary split that minimizes the longer of two
    lines, with both lines ≤ width. Splits that would start line 2 with
    an orphan word (conjunction / preposition / article) get a small
    char-budget penalty so balanced splits without orphans are preferred
    even when slightly less even. On a true tie, the later split wins
    (more text on line 1). Returns "<br>"-joined string, or None if no
    valid 2-line split exists."""
    n = len(words)
    if n == 0:
        return ""
    if n == 1:
        return words[0] if len(words[0]) <= width else None
    best_split = None
    best_score: tuple[int, int] | None = None
    for split in range(1, n):
        line1 = " ".join(words[:split])
        line2 = " ".join(words[split:])
        if len(line1) > width or len(line2) > width:
            continue
        this_max = max(len(line1), len(line2))
        starts_orphan = (
            words[split].lower().rstrip(",.;:!?") in _ORPHAN_STARTS
        )
        # Penalty adds 5 chars to the comparison max — small enough that
        # a strongly-balanced orphan split still wins over a heavily
        # lopsided non-orphan split, large enough to tip ties.
        adjusted_max = this_max + (5 if starts_orphan else 0)
        score = (adjusted_max, -split)  # later split wins on tie
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    if best_split is None:
        return None
    return f"{' '.join(words[:best_split])}<br>{' '.join(words[best_split:])}"


def _wrap_driver_label(s: str, width: int, max_lines: int = 2) -> str:
    """Balanced word-wrap for y-tick labels. When text wraps to two
    lines, the split is chosen to minimize the longer line so the two
    lines are roughly the same length. Past max_lines, the trailing
    word gets an ellipsis. max_lines=2 keeps every row at uniform
    height in plotly bar charts.
    """
    import textwrap
    words = str(s).split()
    if not words:
        return ""
    full = " ".join(words)
    if len(full) <= width:
        return full

    if max_lines == 2:
        # Try the full text; if it doesn't fit two balanced lines, drop
        # trailing words one at a time and try again. The first success
        # is the longest text we can show with balanced 2-line wrap.
        truncated = False
        trial = list(words)
        while trial:
            result = _balance_two_lines(trial, width)
            if result is not None:
                if truncated:
                    # Apply ellipsis to the last visible word on line 2.
                    line1, _, line2 = result.partition("<br>")
                    line2 = line2.rstrip(",.;:") + "…"
                    return f"{line1}<br>{line2}"
                return result
            trial = trial[:-1]
            truncated = True
        return ""

    # Fallback for max_lines != 2 (not used in current charts).
    lines = textwrap.wrap(full, width=width, break_long_words=False,
                          break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        if len(last) > width - 1:
            last = last[: width - 1].rstrip()
        lines[-1] = last.rstrip(",.;:") + "…"
    return "<br>".join(lines)


def _render_intensity_driver_chart(
    plot_df: pd.DataFrame,
    results: Path,
    figures: Path,
    *,
    level: str,            # "occ" or "task"
    slug: str,
    major_short: str,
    label_col: str,
    color_col: str,
    color_label: str,
    color_fmt: str,
    label_wrap: int,
    margin_left: int | None = None,
    margin_pad: int = 0,
    x_mult: float | None = None,
    level_word: str | None = None,
    title: str | None = None,
    out_name: str | None = None,
) -> None:
    """Render one within-major intensity-driver chart, mirroring the
    main-body intensity_anchor_fulleco style: horizontal bars, dashed
    median reference at x=1, TASKS_LIGHT→TASKS_DARK color, HTML-swatch
    bottom legend, raw-pct + lift label outside each bar.

    `plot_df` must already be sorted ascending by lift so Plotly draws
    the largest bar at the top.

    `level` stays the geometry preset ("occ" or "task" — row pitch, right
    margin, x headroom and tick format). `level_word`, `title` and
    `out_name` are optional overrides so a caller can reuse the same
    renderer for a different row type: the DWA charts render work
    activities on the "task" geometry with level_word="Activities".
    """
    out_name = out_name or f"intensity_drivers_{level}_{slug}.png"

    display_labels = plot_df[label_col].astype(str).map(
        lambda s: _wrap_driver_label(s, label_wrap)
    )
    # Auto-fit margin_left to the actual longest rendered line so the
    # y-axis title sits flush against the PNG left edge regardless of
    # what occupations/tasks happen to be in a given major. Uses a
    # per-character width table calibrated against 9pt sans-serif
    # rendered at PAPER_W=1400 — narrow chars ('i','l','t') are ~5–7px,
    # wide chars ('M','W','m','w') are ~17–19px, defaults ~12.
    if margin_left is None:
        _CHAR_W = {
            'i': 6, 'l': 6, 'j': 7, 't': 9, 'f': 9, 'r': 9,
            '.': 6, ',': 6, ';': 6, ':': 6, "'": 5, ' ': 6,
            '(': 8, ')': 8, '-': 9, '/': 8, '!': 6, '|': 6,
            '0': 14, '1': 9, '2': 14, '3': 14, '4': 14, '5': 14,
            '6': 14, '7': 13, '8': 14, '9': 14,
            'I': 8, 'J': 10,
            'M': 20, 'W': 21, 'm': 19, 'w': 18,
        }
        DEFAULT_W = 15  # most upper/lowercase letters at 9pt sans-serif
        def _est(line: str) -> float:
            return sum(_CHAR_W.get(c, DEFAULT_W) for c in line)
        max_line_px = max(
            (max(_est(line) for line in lbl.split("<br>"))
             for lbl in display_labels),
            default=120.0,
        )
        # +70 covers the vertical y-axis title (~30px wide at 10pt), the
        # standoff between title and labels (~18px), and the axis-to-
        # label gap (~12px), plus a ~10px safety buffer so the title
        # doesn't overlap the longest label (e.g. the life_phys_soc_sci
        # chart's "Anthropologists and Archeologists" pushed right up
        # against "Occupations" at the prior +50 setting).
        #
        # `margin_pad` is the escape hatch for label sets the per-character
        # table under-counts — it leans on default-width characters, so a
        # label with few narrow 'i'/'l's and no wide 'M'/'W's renders wider
        # than estimated and the rotated axis title collides with it. Adding
        # to the auto-fit is preferable to the hardcoded per-slug pixel
        # override this function used to need, which had to be re-tuned
        # whenever the top-10 changed.
        margin_left = int(max_line_px + 70 + margin_pad)
    cvals = plot_df[color_col].to_numpy(dtype=float)
    cmin, cmax = float(np.nanmin(cvals)), float(np.nanmax(cvals))
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
        cmin, cmax = (cmin if np.isfinite(cmin) else 0.0,
                      (cmin if np.isfinite(cmin) else 0.0) + 1.0)

    W = PAPER_W
    px = paper_fonts(W)

    TASKS_LIGHT = "#cfe0ec"
    TASKS_DARK = "#2c4f6b"

    text_labels = [
        f"{lift:.2f}×   ({raw:.3f}% raw pct)"
        for lift, raw in zip(plot_df["lift"], plot_df["raw_pct"])
    ]

    level_word = level_word or ("Occupations" if level == "occ" else "Tasks")
    # One line — shortened from the main-body "Actual Equalized AI Usage
    # as a Multiple of Median Usage" so the major name fits on the same
    # line at 11pt across PAPER_W=1400 (cap ≈ 85 chars). "Occupations"
    # is 6 chars longer than "Tasks" so the prefix has to be terse.
    title = title or (
        f"AI Usage as Multiple of Median — Top {level_word} in {major_short}"
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=display_labels, x=plot_df["lift"], orientation="h",
        marker=dict(
            color=cvals,
            colorscale=[[0, TASKS_LIGHT], [1, TASKS_DARK]],
            cmin=cmin, cmax=cmax,
            showscale=False,
            line=dict(width=0),
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(size=px["tick"], color=PAPER_PALETTE["text"],
                      family=FONT_FAMILY),
        cliponaxis=False,
        showlegend=False,
    ))

    fig.add_vline(
        x=1.0, line_dash="dash",
        line_color=PAPER_PALETTE["negative"], line_width=1.5,
    )
    # Median label sits just above the plot, right of the dashed line,
    # on both occ and task charts (1-line title leaves enough top space).
    fig.add_annotation(
        x=1.0, y=1.0, xref="x", yref="paper",
        text="median", showarrow=False,
        xanchor="left", yanchor="bottom",
        xshift=2, yshift=6,
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["negative"], family=FONT_FAMILY),
    )

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
        f"{color_label}&nbsp;&nbsp;{color_fmt.format(cmin)}&nbsp;"
        f"{swatch_html}&nbsp;{color_fmt.format(cmax)}"
    )
    # Compact layout: per-row pitch tightened so 10-row charts land
    # ~3.7–4.3" tall. Task charts get more room because labels are
    # 2-line wrapped (`_wrap_driver_label(width=52, max_lines=2)`); occ
    # charts also have 2-line wraps at width=36 (e.g. "Fine Artists,
    # Including Painters, Sculptors, and Illustrators"), so occ pitch
    # bumped from 50→58 to give 2-line occ labels room without crowding
    # adjacent rows. margin_b unified to 170 across all intensity charts
    # (intensity_anchor_fulleco + underadoption_gap + drivers) so the
    # bottom gradient legend lands the same 120 px below the plot bottom
    # and 50 px above the canvas bottom on every one.
    n_rows = len(plot_df)
    row_pitch = 60 if level == "task" else 58
    margin_top = 90
    margin_bottom = 170
    margin_right = 90 if level == "task" else 110
    height = n_rows * row_pitch + margin_top + margin_bottom
    style_paper_figure(
        fig, title,
        height=height, width=W,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
    )
    fig.update_layout(bargap=0.15)

    # Legend centered on the PNG itself, not on the plot area. xref="paper"
    # is plot-area-relative — to land on PNG midpoint W/2 in pixel terms,
    # solve: PNG_center = margin_left + x_paper × plot_width. With
    # yaxis automargin=False below, margin_left is exactly what we pass
    # here, so the formula is exact.
    plot_w_px = float(W - margin_left - margin_right)
    legend_x = (float(W) / 2.0 - float(margin_left)) / plot_w_px
    plot_h_px = float(height - margin_top - margin_bottom)
    legend_y = -(margin_bottom - 50) / plot_h_px  # matches underadoption_gap spacing
    fig.add_annotation(
        x=legend_x, y=legend_y, xref="paper", yref="paper",
        text=legend_text, showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=px["in_chart_floor"],
                  color=PAPER_PALETTE["text"], family=FONT_FAMILY),
    )

    # x_top sized so the longest bar's outside text ("X.XX×   (Y.YYY raw
    # pct)") extends into the trimmed margin_right and lands near, but
    # inside, the PNG right edge — minimises right-side whitespace. Task
    # charts have a wider margin_left (for long wrapped task labels), so
    # the plot is narrower → needs a larger x_top multiplier to keep the
    # max-lift bar from pushing its outside text past x=W.
    # Multipliers bumped (task 1.78→1.92, occ 1.38→1.48) so the wider
    # "X.XX×   (Y.YYY% raw pct)" outside labels — one char wider since the
    # raw pct gained a "%" — clear the PNG right edge without clipping.
    # The outside label needs a fixed number of PIXELS, so the fraction of
    # the axis it occupies grows as the plot narrows — which means any
    # `margin_pad` has to be paid for here too, or the widest bar's label
    # runs off the canvas. Callers that pad the left margin pass a matching
    # x_mult; `test_v2_driver_labels_fit_the_canvas` pins that they clear.
    x_mult = x_mult or (1.92 if level == "task" else 1.48)
    x_top = float(plot_df["lift"].max()) * x_mult
    # Task chart lift values reach 7,000×+ so plotly's auto-tick spacing
    # crams 5–6 ticks into a narrow plot area. Constrain to ~4 ticks and
    # use the SI suffix format (1k, 2k, …) for compact readability.
    use_si = level == "task" and float(plot_df["lift"].max()) >= 1000.0
    fig.update_xaxes(
        title=dict(text="Usage Relative to Median (×)",
                   font=dict(size=px["axis_title"], family=FONT_FAMILY),
                   standoff=18),
        showgrid=True, gridcolor=PAPER_PALETTE["grid"],
        range=[0, x_top],
        tickangle=0,
        nticks=4 if level == "task" else 6,
        tickformat="~s" if use_si else ",.0f",
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
    )
    # automargin=False pins margin_left to exactly what we pass to
    # style_paper_figure() so the PNG-centered legend math above is
    # exact. Margins below at the call sites are sized to fit the
    # longest wrapped label without plotly needing to expand.
    # tickmode="array" pins every category label — at the tight 50–60 px
    # row pitch, plotly auto-thins categorical ticks otherwise.
    fig.update_yaxes(
        title=dict(text=level_word,
                   font=dict(size=px["axis_title"], family=FONT_FAMILY),
                   standoff=12),
        showgrid=False, showline=False,
        tickfont=dict(size=px["tick"], family=FONT_FAMILY),
        automargin=False,
        tickmode="array", tickvals=list(display_labels), ticktext=list(display_labels),
    )

    save_figure(fig, results / "figures" / out_name, scale=2)
    _copy_fig(results, figures, out_name)
    print(f"  -> {out_name}")



# The three majors that lead the major-adoption chart, opened up.
TARGET_MAJORS = TARGET_MAJORS_DRIVERS

# How many GWAs get a DWA chart, and how many rows each chart carries.
N_TOP_GWAS = 4
TOP_N_ROWS = 10

# Shading legend for the two charts whose colour is work time exposed.
EXPOSED_LABEL = "Work Time Exposed"
EXPOSED_FMT = "{:.0f}%"

# Extra left margin on the wide-label charts. The renderer fits the margin
# from a per-character width table, which under-counts label sets that lean
# on default-width characters — Computer & Mathematical's task labels put
# the rotated "Tasks" axis title on top of "Conduct research to extend
# mathematical knowledge in". Padding the auto-fit is preferable to the
# hardcoded per-slug pixel override the paper version carries, which has to
# be re-tuned whenever a top-10 changes.
WIDE_LABEL_PAD = 40
# Paying for WIDE_LABEL_PAD on the x axis. The outside label is a fixed
# number of pixels, so 40 px off the plot width has to come back as axis
# headroom or the widest bar's label runs off the canvas. Pinned by
# test_driver_labels_fit_the_canvas rather than left as a magic number:
# if a future top-10 shifts the geometry, the test fails loudly.
WIDE_LABEL_X_MULT = 2.08


def _slug(text: str) -> str:
    """Filename-safe slug from an activity title."""
    out = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    assert out, f"empty slug for {text!r}"
    return out


# A GWA title has to fit a chart title, which caps at ~85 characters at 11 pt
# across PAPER_W. `figure_data.short_gwa_label` is too terse for prose ("Working
# w/ Public") because it is sized for a y-axis label column charged twice.
# This applies the same maps one pass shallower.
_GWA_TITLE_MAX = 30


def gwa_title_label(gwa: str) -> str:
    """Readable GWA label for a chart title — shorter than the full O*NET
    name, longer than the axis-column abbreviation."""
    from lib.figure_data import _GWA_SHORT_LABELS

    label = _GWA_SHORT_LABELS.get(gwa, gwa)
    label = figure_data._GWA_SHORT_PASS2.get(label, label)
    assert len(label) <= _GWA_TITLE_MAX, f"GWA title label too long: {label!r}"
    return label


# ─────────────────────────────────────────────────────────────────────────
# Frames
# ─────────────────────────────────────────────────────────────────────────

def _task_text() -> pd.Series:
    """task_normalized → the original punctuated, capitalized statement, for
    human-readable y-tick labels. Duplicates across the file are punctuation
    and capitalization variants of the same task."""
    text = pd.read_csv(
        figure_data.INTENSITY_FILE, usecols=["task_normalized", "task"]
    ).dropna(subset=["task_normalized", "task"])
    return text.drop_duplicates(subset=["task_normalized"]).set_index(
        "task_normalized"
    )["task"]


def dwa_to_gwa() -> pd.Series:
    """dwa_title -> gwa_title.

    DWA sits under exactly one GWA in eco_2025 — 2,083 detailed activities
    across 37 general ones, no DWA in two — so restricting a DWA set to a
    GWA is unambiguous and needs no /n split of its own. Asserted here
    rather than assumed, because the whole four-chart set is built on it.
    """
    from backend.compute import load_eco_raw

    eco = load_eco_raw()
    nest = (
        eco[["dwa_title", "gwa_title"]]
        .dropna()
        .drop_duplicates()
    )
    fanout = nest.groupby("dwa_title")["gwa_title"].nunique()
    assert (fanout == 1).all(), (
        f"{int((fanout > 1).sum())} DWAs map to more than one GWA"
    )
    return nest.set_index("dwa_title")["gwa_title"]


def _usage_frame(
    num: pd.Series,
    den: pd.Series,
    raw: pd.Series,
    index_name: str,
) -> pd.DataFrame:
    """Ratio-of-sums usage rate per group, anchored on the lower-middle
    entry, with the group's summed RAW usage pct alongside.

    Never a mean of per-task rates: `intensity = adj_pct / emp` is a rate,
    so summing it over a group's rows measures how many rows the group has
    as much as how intensively it is used.
    """
    frame = pd.DataFrame({"num": num})
    frame["den"] = den.reindex(frame.index)
    frame["raw_pct"] = raw.reindex(frame.index).fillna(0.0)
    frame = frame[frame["den"] > 0].copy()
    assert not frame.empty, "no group survived the positive-denominator filter"
    frame["ratio"] = frame["num"] / frame["den"]
    frame["lift"] = figure_data.anchor_lower_median(frame["ratio"])
    frame.index.name = index_name
    return frame.reset_index()


def occ_usage_frame(major: str) -> pd.DataFrame:
    """Occupations inside one major: usage lift, raw pct, work time exposed.

    Anchored on the median occupation IN THIS MAJOR — the chart decomposes
    the major, so the major is what 1.00x means. Only occupations with at
    least one rated pair enter the anchor population; the rest have no
    observed usage at all and entering them as zeros would drag the median
    down and inflate every bar.
    """
    pairs = figure_data.intensity_pairs()
    sub = pairs[pairs["major_occ_category"] == major]
    assert not sub.empty, f"no rated usage pairs in {major!r}"

    emp = figure_data.pair_level_emp("title_current")
    frame = _usage_frame(
        num=sub.groupby("title_current")["adj_pct"].sum(),
        den=emp,
        raw=sub.groupby("title_current")["pct_normalized"].sum(),
        index_name="title_current",
    )

    exposure = figure_data.occ_exposure(figure_data.PRIMARY_DATASET).set_index(
        "title_current"
    )["p"]
    frame["pct_exposed"] = (
        frame["title_current"].map(exposure).astype(float) * 100.0
    )
    return frame.dropna(subset=["pct_exposed"])


def task_usage_frame(major: str) -> pd.DataFrame:
    """Tasks inside one major: usage lift, raw pct, mean auto-aug.

    A task is pooled across the occupations in the major that hold it, on
    both sides of the ratio, so the rate is that task's usage per worker
    doing it anywhere in the major.
    """
    pairs = figure_data.intensity_pairs()
    sub = pairs[pairs["major_occ_category"] == major]
    assert not sub.empty, f"no rated usage pairs in {major!r}"

    eco = figure_data.eco_pairs()
    eco_major = eco[eco["major_occ_category"] == major]
    frame = _usage_frame(
        num=sub.groupby("task_normalized")["adj_pct"].sum(),
        den=eco_major.groupby("task_normalized")["eco_weight"].sum(),
        raw=sub.groupby("task_normalized")["pct_normalized"].sum(),
        index_name="task_normalized",
    )

    frame["auto_aug"] = frame["task_normalized"].map(
        sub.groupby("task_normalized")["auto_aug_mean"].mean()
    )
    frame["task_display"] = (
        frame["task_normalized"].map(_task_text()).fillna(frame["task_normalized"])
    )
    return frame.dropna(subset=["auto_aug"])


def gwa_usage_ranking() -> pd.DataFrame:
    """The GWA adoption ranking — the same construction
    `adoption.build_gwa_adoption` plots, recomputed here so the four DWA
    charts pick their GWAs off the published ranking rather than a variant
    of it."""
    rows = figure_data.intensity_gwa_rows()
    den = figure_data.eco_gwa_weight_split()
    frame = _usage_frame(
        num=rows.groupby("gwa_title")["adj_pct_split"].sum().reindex(
            den.index
        ).fillna(0.0),
        den=den,
        raw=rows.groupby("gwa_title")["raw_pct_split"].sum(),
        index_name="gwa_title",
    )
    return frame.sort_values("lift", ascending=False).reset_index(drop=True)


def dwa_usage_frame() -> pd.DataFrame:
    """Every DWA in the economy: usage lift, raw pct, work time exposed, and
    the GWA it sits under.

    Both sides are /n-split across a pair's distinct DWAs — a task sits in
    several, so joining pair-level usage onto DWA rows without dividing
    counts the same usage two or three times.

    THE ANCHOR IS ECONOMY-WIDE, which departs from the folder's usual rule
    of anchoring on the thing the chart decomposes. Four separate charts
    each anchored inside their own GWA would put an unrelated activity at
    1.00x on each one, and no bar on any of them could be compared with a
    bar on another. Anchored on the median DWA in the economy, all four sit
    on one scale and read directly against the GWA adoption chart's own
    x-medians. The population is DWAs with at least one rated pair.
    """
    num_rows = figure_data.intensity_act_rows("dwa_title")
    den_rows = figure_data.eco_act_split_rows("dwa_title")

    frame = _usage_frame(
        num=num_rows.groupby("dwa_title")["adj_pct_split"].sum(),
        den=den_rows.groupby("dwa_title")["eco_weight_split"].sum(),
        raw=num_rows.groupby("dwa_title")["raw_pct_split"].sum(),
        index_name="dwa_title",
    )

    frame["gwa_title"] = frame["dwa_title"].map(dwa_to_gwa())

    exposure = figure_data.act_exposure(
        figure_data.PRIMARY_DATASET, "dwa_title"
    ).set_index("category")["pct"]
    frame["pct_exposed"] = frame["dwa_title"].map(exposure).astype(float)
    return frame.dropna(subset=["gwa_title", "pct_exposed"])


# ─────────────────────────────────────────────────────────────────────────
# Chart 1-6 — occupations and tasks inside three majors
# ─────────────────────────────────────────────────────────────────────────

def build_usage_drivers_majors(results: Path, figures: Path) -> None:
    for major_full, slug, major_short in TARGET_MAJORS:
        occ = occ_usage_frame(major_full)
        top_occ = occ.nlargest(TOP_N_ROWS, "lift")
        save_csv(
            top_occ.sort_values("lift", ascending=False),
            results / f"usage_drivers_occ_{slug}.csv",
            float_format="%.4f",
        )
        name = f"usage_drivers_occ_{slug}.png"
        _render_intensity_driver_chart(
            top_occ.sort_values("lift", ascending=True),
            results, figures,
            level="occ", slug=slug, major_short=major_short,
            label_col="title_current", color_col="pct_exposed",
            color_label=EXPOSED_LABEL, color_fmt=EXPOSED_FMT,
            label_wrap=36, out_name=name,
        )

        task = task_usage_frame(major_full)
        top_task = task.nlargest(TOP_N_ROWS, "lift")
        save_csv(
            top_task.sort_values("lift", ascending=False),
            results / f"usage_drivers_task_{slug}.csv",
            float_format="%.4f",
        )
        name = f"usage_drivers_task_{slug}.png"
        _render_intensity_driver_chart(
            top_task.sort_values("lift", ascending=True),
            results, figures,
            level="task", slug=slug, major_short=major_short,
            label_col="task_display", color_col="auto_aug",
            color_label="Auto-Aug (1–5)", color_fmt="{:.2f}",
            label_wrap=52, margin_pad=WIDE_LABEL_PAD, x_mult=WIDE_LABEL_X_MULT, out_name=name,
        )


# ─────────────────────────────────────────────────────────────────────────
# Chart 7-10 — the DWAs inside the four leading GWAs
# ─────────────────────────────────────────────────────────────────────────

def build_usage_drivers_gwa(results: Path, figures: Path) -> None:
    """One chart per leading GWA: the ten detailed activities inside it with
    the highest AI usage per worker.

    No minimum-evidence gate. Usage intensity is a rate, so a DWA holding
    very little work can rank high on it — but the raw pct printed beside
    every bar IS the evidence, the same way the agentic figure prints the
    catalogue share rather than silently filtering. A reader can discount a
    row; a dropped row cannot be recovered.
    """
    ranking = gwa_usage_ranking()
    top_gwas = ranking.head(N_TOP_GWAS)
    assert len(top_gwas) == N_TOP_GWAS, f"only {len(top_gwas)} GWAs ranked"
    save_csv(ranking, results / "usage_gwa_ranking.csv", float_format="%.4f")

    dwa = dwa_usage_frame()
    for rank, row in enumerate(top_gwas.itertuples(), start=1):
        gwa = row.gwa_title
        sub = dwa[dwa["gwa_title"] == gwa]
        assert not sub.empty, f"no rated DWAs under {gwa!r}"
        top = sub.nlargest(TOP_N_ROWS, "lift")

        slug = _slug(gwa)
        save_csv(
            top.sort_values("lift", ascending=False),
            results / f"usage_dwa_{slug}.csv",
            float_format="%.4f",
        )
        short = gwa_title_label(gwa)
        name = f"usage_dwa_{slug}.png"
        _render_intensity_driver_chart(
            top.sort_values("lift", ascending=True),
            results, figures,
            level="task", slug=slug, major_short=short,
            label_col="dwa_title", color_col="pct_exposed",
            color_label=EXPOSED_LABEL, color_fmt=EXPOSED_FMT,
            label_wrap=52, margin_pad=WIDE_LABEL_PAD, x_mult=WIDE_LABEL_X_MULT,
            level_word="Activities",
            title=(
                "AI Usage as Multiple of Median — "
                f"Top Activities in {short}"
            ),
            out_name=name,
        )
        print(f"     GWA {rank}/{N_TOP_GWAS}: {gwa} ({row.lift:.2f}x)")
