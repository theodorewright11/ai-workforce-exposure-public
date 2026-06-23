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
