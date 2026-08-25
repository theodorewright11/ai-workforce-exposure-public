/**
 * api.ts — client for the public dashboard backend (dashboard/api/main.py).
 * Read-only: 5-config exposure/usage/trend views + the occupation report.
 */
import type {
  ConfigResponse,
  ExposureResponse,
  UsageResponse,
  TrendResponse,
  OccupationReport,
  OccReportTitlesResponse,
} from "./types";

// Strip any trailing slash(es) so a value like "https://host/" doesn't produce
// a double-slash path ("//api/config") that the backend 404s on.
export const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export async function fetchConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) throw new Error(`/api/config failed: ${res.status}`);
  return res.json();
}

// ── Data page ───────────────────────────────────────────────────────────────

export type ExposureKind = "occ" | "wa";

export function fetchExposure(
  config: string, level: string, geo: string, kind: ExposureKind,
): Promise<ExposureResponse> {
  return postJSON("/api/exposure", { config, level, geo, kind });
}

export function fetchExposureChildren(
  config: string, level: string, geo: string, kind: ExposureKind, parent: string,
): Promise<ExposureResponse> {
  return postJSON("/api/exposure/children", { config, level, geo, kind, parent });
}

export function fetchTrend(
  config: string, level: string, geo: string, kind: ExposureKind,
): Promise<TrendResponse> {
  return postJSON("/api/trend", { config, level, geo, kind });
}

export function fetchUsage(
  level: string, parentLevel?: string, parent?: string,
): Promise<UsageResponse> {
  return postJSON("/api/usage", {
    level,
    parent_level: parentLevel ?? null,
    parent: parent ?? null,
  });
}

// Drill straight to occupations from a SOC level (occ kind).
export function fetchExposureToOcc(
  config: string, level: string, geo: string, parent: string,
): Promise<ExposureResponse> {
  return postJSON("/api/exposure/children", { config, level, geo, kind: "occ", parent, to_level: "occupation" });
}

// Tasks under one work activity (for the DWA drill-down task list).
export interface WaTask {
  task: string; task_normalized: string; centrality: number | null; centrality_rank: number;
  auto: number | null; auto_label: string; color_bucket: string; usage_mult: number | null;
  gwa: { name: string; auto: number | null; rank_pct: number | null; total: number | null } | null;
  iwa: { name: string; auto: number | null; rank_pct: number | null; total: number | null } | null;
  dwa: { name: string; auto: number | null; rank_pct: number | null; total: number | null } | null;
  top_mcps: { title: string; rating: number | null; url: string | null; description: string | null }[];
}
export function fetchWaTasks(level: string, name: string): Promise<{ tasks: WaTask[] }> {
  return postJSON("/api/wa-tasks", { level, name });
}

// ── Occupation page (copied report) ───────────────────────────────────────────

export async function fetchOccupationReportTitles(): Promise<OccReportTitlesResponse> {
  const res = await fetch(`${API_BASE}/api/occupation-report/titles`);
  if (!res.ok) throw new Error(`/api/occupation-report/titles failed: ${res.status}`);
  return res.json();
}

export async function fetchOccupationReport(
  title: string, geo: string = "nat",
): Promise<OccupationReport> {
  const url = `${API_BASE}/api/occupation-report?title=${encodeURIComponent(title)}&geo=${encodeURIComponent(geo)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`/api/occupation-report failed: ${res.status}`);
  return res.json();
}

/* ── Economy at a Glance ───────────────────────────────────────────────────
 * The page takes no controls, so its whole payload is one request.
 */
export interface TrendPoint { date: string; pct: number }
export interface TrendSeriesBlock { key: string; label: string; points: TrendPoint[] }
export interface EconomyTrend {
  series: TrendSeriesBlock[];
  headline_pct: number; headline_unexposed: number;
  first_pct: number; first_date: string; latest_date: string; change_pp: number;
}
export interface EconomyFamily {
  family: string; label: string; short: string;
  pct_exposed: number; pct_unexposed: number; usage_x: number;
  share_of_day: number; n_tasks: number;
}
export interface EconomyGroup {
  category: string; pct_exposed: number; pct_unexposed: number;
  workers_exposed: number; wages_exposed: number; usage_x: number;
}
export interface FocusedRow {
  title: string; major: string; job_zone: number | null;
  pct_exposed: number; pct_unexposed: number; emp_proj_pct: number;
  usage_x: number; workers_exposed: number;
}
export interface StateRow {
  geo: string; state: string; rank: number;
  pct_exposed: number; pct_unexposed: number;
  employment: number; workers_exposed: number;
}
export interface EconomyResponse {
  trend: EconomyTrend;
  families: EconomyFamily[];
  majors: EconomyGroup[];
  gwas: EconomyGroup[];
  focused: { exposure_min: number; count: number; total_workers: number; rows: FocusedRow[] };
  states: { total: number; top: StateRow[]; bottom: StateRow[] };
  dataset: string;
}

export async function fetchEconomy(geo: string = "nat"): Promise<EconomyResponse> {
  const res = await fetch(`${API_BASE}/api/economy?geo=${encodeURIComponent(geo)}`);
  if (!res.ok) throw new Error(`/api/economy failed: ${res.status}`);
  return res.json();
}

/* ── Verb families (Explore tab 3) ─────────────────────────────────────────
 * Verb family is not a level in GWA → IWA → DWA; it re-buckets the DWA level,
 * so it has its own endpoint and its own drill path (family → DWA).
 */
export interface FamilyRowApi {
  category: string; family?: string; short?: string;
  pct_exposed: number; pct_unexposed: number; usage_x: number;
  share_of_day?: number; n_tasks: number;
}
export interface FamiliesResponse {
  rows: FamilyRowApi[]; parent: string | null; child_level: string | null;
}

export async function fetchFamilies(
  config: string = "all_confirmed", geo: string = "nat", parent?: string,
): Promise<FamiliesResponse> {
  const q = new URLSearchParams({ config, geo });
  if (parent) q.set("parent", parent);
  const res = await fetch(`${API_BASE}/api/families?${q.toString()}`);
  if (!res.ok) throw new Error(`/api/families failed: ${res.status}`);
  return res.json();
}
