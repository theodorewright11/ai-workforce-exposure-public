/**
 * JS color constants matching the CSS custom properties in globals.css.
 *
 * These exist because chart libraries like Plotly operate on raw data props
 * (e.g. `marker.color`) and cannot resolve CSS custom properties at runtime.
 *
 * When changing group colors:
 *   1. Update --group-a / --group-b in globals.css
 *   2. Update GROUP_A_COLOR / GROUP_B_COLOR below to match
 *
 * In Phase 2 (Recharts migration), these constants can be removed since
 * Recharts renders into the DOM where CSS variables resolve naturally.
 */
export const GROUP_A_COLOR = "#3a5f83";
export const GROUP_B_COLOR = "#4a7c6f";

// Metric colors (paper-consistent): tasks = steel blue, workers = warm sand,
// wages = sage green. Used by the Data page metric panels.
export const METRIC_COLORS = {
  pct_tasks_affected: "#3a5f83",
  workers_affected: "#c2a15c",
  wages_affected: "#5b9279",
} as const;

export const INTENSITY_COLOR = "#3a5f83";

// Distinct hues for trend lines (top-N categories).
export const CATEGORY_PALETTE = [
  "#3a5f83", "#c2a15c", "#5b9279", "#9e6b8e", "#6b8fb5",
  "#b0894a", "#7aa583", "#8a6d9e", "#c47b6a", "#5f8a8a",
];

