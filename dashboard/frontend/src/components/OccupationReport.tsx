"use client";

import { useEffect, useMemo, useState } from "react";
import type { ConfigResponse } from "@/lib/types";
import { fetchOccupationReport, fetchOccupationReportTitles } from "@/lib/api";
import { fmtPct, fmtCount, fmtWages, ordinal } from "@/lib/format";

/* ──────────────────────────────────────────────────────────────────────────
   Types (the report payload we consume)
   ────────────────────────────────────────────────────────────────────────── */
interface Gates { pct: number; ska: number; growth: number; emp_decline: number; count: number; emp_proj: number | null; }
interface Intensity { occ_intensity_rank?: number; occ_intensity_total?: number; major_intensity_pct?: number; occ_intensity_pct?: number; }
interface Headline {
  title: string; major: string | null; minor: string | null; broad: string | null;
  job_zone: number | null; n_tasks: number | null;
  emp: number | null; wage: number | null;
  pct_tasks_affected: number | null; workers_affected: number | null; wages_affected: number | null;
  gates: Gates; intensity: Intensity;
}
interface RankBlock { pct: number; workers: number; wages: number; total: number; }
interface GroupRanks { economy: RankBlock; major: RankBlock; minor: RankBlock; broad: RankBlock; }
interface TrendPt { date: string; pct_tasks_affected: number; }
interface WaDetail { name: string; auto: number | null; rank_pct: number | null; total: number | null; }
interface Mcp { title: string; rating: number | null; url: string | null; description: string | null; }
interface Task {
  task: string; task_normalized: string; importance: number; freq_mean: number; relevance: number;
  centrality: number | null; physical: boolean; auto: number | null; auto_label: string;
  color_bucket: string; usage_mult: number | null;
  gwa: WaDetail | null; iwa: WaDetail | null; dwa: WaDetail | null; top_mcps: Mcp[]; rank: number;
}
interface SkaRow { element: string; importance: number | null; level: number | null; occ_score: number | null; ai_top10: number | null; pct_of_need: number | null; }
interface Ska { summary: { overall_pct?: number }; rows: { skills: SkaRow[]; abilities: SkaRow[]; knowledge: SkaRow[] }; }
interface RankWindow { rank: number | null; total: number; window: { title: string; value: number | null; rank: number; is_occ: boolean }[]; }
interface MajorRanking { major?: string; pct?: RankWindow; adoption?: RankWindow; }
interface Tech { software: string; commodity: string; commodity_avg_pct: number | null; }
interface Report {
  title: string; geo: string;
  headline: Headline; tasks: Task[]; group_ranks: GroupRanks;
  trend: TrendPt[]; ska: Ska; major_ranking: MajorRanking; tech: Tech[];
}

/* ──────────────────────────────────────────────────────────────────────────
   Palette + small helpers
   ────────────────────────────────────────────────────────────────────────── */
const BUCKET = {
  high: { label: "Automated usage seen",       color: "#a8824a" },
  mid:  { label: "Augmentative usage seen",    color: "#c9b27e" },
  low:  { label: "Low automation usage seen",  color: "#8ea9bf" },
  none: { label: "No usage seen",              color: "#c9ccce" },
} as const;
type BucketKey = keyof typeof BUCKET;

// Blue gradient for the 0–4 gate count (light → dark = more exposed).
const GATE_BLUES = ["#dbe6f0", "#aec6dd", "#7da3c4", "#4f7da8", "#2f5f86"];
const ZONE_LABEL: Record<number, string> = {
  1: "little/no prep", 2: "some prep", 3: "medium prep", 4: "considerable prep", 5: "extensive prep",
};
const GATE_DEFS: { key: keyof Gates; label: string }[] = [
  { key: "pct",         label: "Over 50% of tasks exposed" },
  { key: "growth",      label: "Exposure growing faster than average" },
  { key: "ska",         label: "AI reaches the occupation's core skills" },
  { key: "emp_decline", label: "Employment projected to decline (BLS 2025–34)" },
];

function bucketKey(b: string): BucketKey {
  return (["high", "mid", "low", "none"].includes(b) ? b : "none") as BucketKey;
}

/* ──────────────────────────────────────────────────────────────────────────
   Component
   ────────────────────────────────────────────────────────────────────────── */
export default function OccupationReport({ config }: { config: ConfigResponse }) {
  const [titles, setTitles] = useState<string[]>([]);
  const [title, setTitle] = useState<string>("");
  const [geo, setGeo] = useState("nat");
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);

  useEffect(() => {
    fetchOccupationReportTitles().then((d) => {
      setTitles(d.titles);
      if (d.titles.length) setTitle(d.titles.find((t) => t === "Computer Programmers") ?? d.titles[0]);
    });
  }, []);

  useEffect(() => {
    if (!title) return;
    setLoading(true);
    fetchOccupationReport(title, geo)
      .then((r) => setReport(r as unknown as Report))
      .finally(() => setLoading(false));
  }, [title, geo]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return titles.filter((t) => t.toLowerCase().includes(q)).slice(0, 12);
  }, [query, titles]);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "24px 24px 64px" }}>
      {/* Picker + geo */}
      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 22 }}>
        <div style={{ position: "relative", flex: 1, minWidth: 260 }}>
          <label style={lblStyle}>Occupation</label>
          <input
            value={pickerOpen ? query : title}
            onChange={(e) => { setQuery(e.target.value); setPickerOpen(true); }}
            onFocus={() => { setQuery(""); setPickerOpen(true); }}
            onBlur={() => setTimeout(() => setPickerOpen(false), 150)}
            placeholder="Search occupations…"
            style={{ ...inputStyle, width: "100%" }}
          />
          {pickerOpen && matches.length > 0 && (
            <div style={{
              position: "absolute", top: "100%", left: 0, right: 0, zIndex: 30, marginTop: 3,
              background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 8,
              boxShadow: "0 6px 20px rgba(0,0,0,0.12)", maxHeight: 320, overflowY: "auto",
            }}>
              {matches.map((m) => (
                <div key={m} onMouseDown={() => { setTitle(m); setPickerOpen(false); }}
                  style={{ padding: "8px 12px", fontSize: 13, cursor: "pointer", color: "var(--text-secondary)" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--brand-light)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
                  {m}
                </div>
              ))}
            </div>
          )}
        </div>
        <div>
          <label style={lblStyle}>Geography</label>
          <select value={geo} onChange={(e) => setGeo(e.target.value)} style={inputStyle}>
            {Object.entries(config.geo_options).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
        </div>
      </div>

      {loading && !report && <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</div>}
      {report && (
        <>
          <Overview report={report} />
          <RankingInMajor mr={report.major_ranking} />
          <TasksSection tasks={report.tasks} />
          <SkaSection ska={report.ska} />
          <SoftwareSection tech={report.tech} />
        </>
      )}
    </div>
  );
}

/* ── Overview card ─────────────────────────────────────────────────────────── */
function Overview({ report }: { report: Report }) {
  const h = report.headline;
  const [whyOpen, setWhyOpen] = useState(false);
  const gates = h.gates;
  const first = report.trend[0]?.pct_tasks_affected;
  const pct = h.pct_tasks_affected ?? 0;
  const zone = h.job_zone ? Math.round(h.job_zone) : null;

  return (
    <div style={cardStyle}>
      {/* header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.2 }}>{h.title}</div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
            <span><strong>Major:</strong> {h.major}</span>{"  ·  "}
            <span><strong>Minor:</strong> {h.minor}</span>{"  ·  "}
            <span><strong>Broad:</strong> {h.broad}</span>
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            {zone && <Chip>Job Zone {zone} · {ZONE_LABEL[zone]}</Chip>}
            <Chip>{h.n_tasks ?? 0} tasks</Chip>
          </div>
        </div>
        {/* 4-gate exposure tier */}
        <div style={{ textAlign: "right" }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 9,
            background: GATE_BLUES[gates.count], color: gates.count >= 3 ? "#fff" : "var(--text-primary)",
            fontWeight: 700, fontSize: 14,
          }}>
            {gates.count} of 4 exposure signals
          </div>
          <div>
            <button onClick={() => setWhyOpen((v) => !v)} style={whyBtn}>{whyOpen ? "Hide" : "Why?"}</button>
          </div>
          {whyOpen && (
            <div style={{ marginTop: 8, textAlign: "left", background: "var(--bg-sidebar)", borderRadius: 8, padding: "10px 12px", width: 300 }}>
              {GATE_DEFS.map((g) => {
                const on = gates[g.key] === 1;
                return (
                  <div key={g.key} style={{ display: "flex", gap: 8, fontSize: 12, padding: "3px 0", color: on ? "var(--text-primary)" : "var(--text-muted)" }}>
                    <span style={{ color: on ? "#2f5f86" : "var(--text-muted)", fontWeight: 700 }}>{on ? "●" : "○"}</span>
                    <span>{g.label}{g.key === "emp_decline" && gates.emp_proj != null ? ` (${gates.emp_proj}%)` : ""}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* KPI row */}
      <div style={{ display: "flex", gap: 16, marginTop: 20, flexWrap: "wrap", alignItems: "stretch" }}>
        {/* Tasks exposed — prominent */}
        <div style={{ ...kpiStyle, flex: "1 1 220px", background: "var(--brand-light)", borderColor: "var(--brand-border)" }}>
          <div style={kpiLabel}>% Tasks Exposed</div>
          <div style={{ fontSize: 38, fontWeight: 800, color: "var(--brand)", lineHeight: 1.05 }}>{fmtPct(pct)}</div>
          {first != null && first < pct && (
            <div style={{ fontSize: 11.5, color: "var(--text-secondary)" }}>up from {fmtPct(first)} a year ago</div>
          )}
          <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginTop: 4 }}>
            rank {ordinal(report.group_ranks.economy.pct)} of {report.group_ranks.economy.total} occupations
          </div>
        </div>
        <Kpi label="Workers Exposed" value={fmtCount(h.workers_affected ?? 0)}
          sub={`rank ${ordinal(report.group_ranks.economy.workers)} of ${report.group_ranks.economy.total}`} />
        <Kpi label="Wages Exposed" value={fmtWages(h.wages_affected ?? 0)}
          sub={`rank ${ordinal(report.group_ranks.economy.wages)} of ${report.group_ranks.economy.total}`} />
        <Kpi label="AI Adoption Rank"
          value={h.intensity.occ_intensity_rank ? `#${h.intensity.occ_intensity_rank}` : "—"}
          sub={`of ${h.intensity.occ_intensity_total ?? "—"} — relative to how much this work is done`} />
      </div>
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 10 }}>
        AI adoption rank reflects how much workers here already use AI relative to their task load, ranked across the economy (not a forecast).
      </div>
    </div>
  );
}

/* ── Ranking within major ───────────────────────────────────────────────────── */
function RankingInMajor({ mr }: { mr: MajorRanking }) {
  if (!mr.major || !mr.pct) return null;
  return (
    <Section title="Where it ranks in its major category"
      subtitle={`Its place among the ${mr.pct.total} occupations in ${mr.major}, shown with its 5 nearest neighbours on each metric. Left: share of tasks exposed. Right: how much AI is actually used (adoption).`}>
      <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
        <RankWindowChart title={`% Tasks Exposed — rank ${mr.pct.rank} of ${mr.pct.total}`}
          win={mr.pct.window} fmt={(v) => fmtPct(v)} color="#3a5f83" />
        {mr.adoption && (
          <RankWindowChart title={`AI Adoption — rank ${mr.adoption.rank} of ${mr.adoption.total}`}
            win={mr.adoption.window} fmt={() => ""} color="#4a7c6f" showRank />
        )}
      </div>
    </Section>
  );
}

function RankWindowChart({ title, win, fmt, color, showRank }: {
  title: string; win: RankWindow["window"]; fmt: (v: number) => string; color: string; showRank?: boolean;
}) {
  const max = Math.max(1, ...win.map((w) => w.value ?? 0));
  return (
    <div style={{ flex: 1, minWidth: 280 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {win.map((w) => (
          <div key={w.title}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 1,
              color: w.is_occ ? "var(--brand)" : "var(--text-secondary)", fontWeight: w.is_occ ? 700 : 400 }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "75%" }} title={w.title}>
                #{w.rank} {w.title}
              </span>
              <span style={{ fontWeight: 600 }}>{showRank ? "" : fmt(w.value ?? 0)}</span>
            </div>
            <div style={{ height: 9, background: "var(--bg-sidebar)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${((w.value ?? 0) / max) * 100}%`, height: "100%",
                background: w.is_occ ? "var(--brand)" : color, opacity: w.is_occ ? 1 : 0.7 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Tasks ──────────────────────────────────────────────────────────────────── */
function TasksSection({ tasks }: { tasks: Task[] }) {
  // centrality rank within occ (1 = most central)
  const centralityRank = useMemo(() => {
    const order = [...tasks].sort((a, b) => (b.centrality ?? 0) - (a.centrality ?? 0));
    const m = new Map<string, number>();
    order.forEach((t, i) => m.set(t.task_normalized, i + 1));
    return m;
  }, [tasks]);

  const groups: { key: BucketKey; items: Task[] }[] = (["high", "mid", "low", "none"] as BucketKey[])
    .map((k) => ({ key: k, items: tasks.filter((t) => bucketKey(t.color_bucket) === k) }))
    .filter((g) => g.items.length);

  return (
    <Section title="Tasks"
      subtitle="Every task in this occupation, grouped by how AI is being used on it (from All Confirmed usage). Auto value is the 0–5 automation–augmentation score; Centrality ranks the task by how core it is to the job (frequency × importance × relevance); Usage shows how much AI is used on it vs. the occupation's median task. Expand a row for its work activities and the AI tools that target it.">
      {groups.map((g) => (
        <TaskBucket key={g.key} bucket={g.key} items={g.items} centralityRank={centralityRank} total={tasks.length} />
      ))}
    </Section>
  );
}

function TaskBucket({ bucket, items, centralityRank, total }: {
  bucket: BucketKey; items: Task[]; centralityRank: Map<string, number>; total: number;
}) {
  const [open, setOpen] = useState(bucket === "high" || bucket === "mid");
  const meta = BUCKET[bucket];
  return (
    <div style={{ marginBottom: 10 }}>
      <button onClick={() => setOpen((v) => !v)} style={{ ...bucketHeader, borderLeft: `4px solid ${meta.color}` }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{meta.label}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{items.length} · {open ? "−" : "+"}</span>
      </button>
      {open && (
        <div>
          {/* column header */}
          <div style={{ ...taskRow, fontSize: 10.5, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em", cursor: "default" }}>
            <span style={{ flex: 1 }}>Task</span>
            <span style={taskCol}>Centrality</span>
            <span style={taskCol}>Usage</span>
            <span style={{ ...taskCol, width: 150 }}>Auto</span>
          </div>
          {items.map((t) => <TaskItem key={t.task_normalized} t={t} centrality={centralityRank.get(t.task_normalized)} total={total} />)}
        </div>
      )}
    </div>
  );
}

function TaskItem({ t, centrality, total }: { t: Task; centrality?: number; total: number }) {
  const [open, setOpen] = useState(false);
  const meta = BUCKET[bucketKey(t.color_bucket)];
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <div style={{ ...taskRow }} onClick={() => setOpen((v) => !v)}>
        <span style={{ flex: 1, fontSize: 12.5 }}>
          <span style={{ opacity: 0.4, marginRight: 6 }}>{open ? "▾" : "▸"}</span>{t.task}
        </span>
        <span style={taskCol}>{centrality ? `#${centrality}` : "—"}<span style={{ color: "var(--text-muted)" }}>/{total}</span></span>
        <span style={taskCol}>{t.usage_mult != null ? `${t.usage_mult}×` : "—"}</span>
        <span style={{ ...taskCol, width: 150, textAlign: "right" }}>
          <span style={{ fontWeight: 700, color: meta.color }}>{t.auto ?? "—"}</span>
          <span style={{ color: "var(--text-muted)", marginLeft: 6, fontSize: 10.5 }}>{t.auto_label}</span>
        </span>
      </div>
      {open && (
        <div style={{ padding: "8px 14px 14px 26px", background: "var(--bg-sidebar)", fontSize: 12 }}>
          {/* work activities */}
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: t.top_mcps.length ? 12 : 0 }}>
            {([["GWA", t.gwa], ["IWA", t.iwa], ["DWA", t.dwa]] as [string, WaDetail | null][]).map(([lvl, w]) =>
              w ? (
                <div key={lvl} style={{ minWidth: 200 }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>{lvl}</div>
                  <div style={{ color: "var(--text-secondary)" }}>{w.name}</div>
                  <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
                    auto {w.auto ?? "—"} (avg over rated tasks) · ranks #{w.rank_pct ?? "—"} of {w.total ?? "—"} by tasks exposed
                  </div>
                </div>
              ) : null
            )}
          </div>
          {t.top_mcps.length > 0 && (
            <div>
              <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>
                AI tools that target this task
              </div>
              {t.top_mcps.map((m, i) => (
                <div key={i} style={{ marginBottom: 5 }}>
                  <a href={m.url ?? "#"} target="_blank" rel="noreferrer" style={{ color: "var(--brand)", fontWeight: 600, fontSize: 12 }}>
                    {m.title}
                  </a>
                  {m.rating != null && <span style={{ color: "var(--text-muted)", marginLeft: 6 }}>★ {m.rating}</span>}
                  {m.description && (
                    <div style={{ color: "var(--text-muted)", fontSize: 11, lineHeight: 1.4 }}>
                      {m.description.length > 160 ? m.description.slice(0, 160) + "…" : m.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Skills, Knowledge & Abilities ──────────────────────────────────────────── */
function SkaSection({ ska }: { ska: Ska }) {
  const rows = useMemo(() => {
    const all: (SkaRow & { kind: string })[] = [
      ...ska.rows.skills.map((r) => ({ ...r, kind: "Skill" })),
      ...ska.rows.knowledge.map((r) => ({ ...r, kind: "Knowledge" })),
      ...ska.rows.abilities.map((r) => ({ ...r, kind: "Ability" })),
    ].filter((r) => r.pct_of_need != null && (r.importance ?? 0) >= 3);
    // centrality rank by occ_score (level × importance)
    const byScore = [...all].sort((a, b) => (b.occ_score ?? 0) - (a.occ_score ?? 0));
    const cmap = new Map<string, number>();
    byScore.forEach((r, i) => cmap.set(r.element, i + 1));
    const withRank = all.map((r) => ({ ...r, centrality: cmap.get(r.element) ?? 0, mult: (r.pct_of_need ?? 0) / 100 }));
    withRank.sort((a, b) => b.mult - a.mult);
    return withRank;
  }, [ska]);

  const wins = rows.filter((r) => r.mult >= 1);
  const behind = rows.filter((r) => r.mult < 1);

  return (
    <Section title="Skills, Knowledge & Abilities"
      subtitle="The skills, knowledge, and abilities that matter to this job (importance ≥ 3). AI capability is shown as a multiple of the occupation's need — 1.0× means AI matches what the job requires, above that means AI already reaches further. Centrality ranks how core each one is to the occupation.">
      <SkaQuadrant label="AI does this well" rows={wins} good />
      <SkaQuadrant label="AI is still catching up" rows={behind} />
    </Section>
  );
}

function SkaQuadrant({ label, rows, good }: { label: string; rows: (SkaRow & { centrality: number; mult: number; kind: string })[]; good?: boolean }) {
  const [open, setOpen] = useState(true);
  if (!rows.length) return null;
  return (
    <div style={{ marginBottom: 10 }}>
      <button onClick={() => setOpen((v) => !v)} style={{ ...bucketHeader, borderLeft: `4px solid ${good ? "#3a5f83" : "#8ea9bf"}` }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{label}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{rows.length} · {open ? "−" : "+"}</span>
      </button>
      {open && (
        <div>
          <div style={{ ...taskRow, fontSize: 10.5, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", cursor: "default" }}>
            <span style={{ flex: 1 }}>Element</span>
            <span style={taskCol}>Centrality</span>
            <span style={{ ...taskCol, width: 120 }}>AI capability</span>
          </div>
          {rows.map((r) => (
            <div key={r.kind + r.element} style={{ ...taskRow, cursor: "default", borderBottom: "1px solid var(--border)" }}>
              <span style={{ flex: 1, fontSize: 12.5 }}>
                {r.element} <span style={{ color: "var(--text-muted)", fontSize: 10.5 }}>· {r.kind}</span>
              </span>
              <span style={taskCol}>#{r.centrality}</span>
              <span style={{ ...taskCol, width: 120, fontWeight: 700, color: r.mult >= 1 ? "#3a5f83" : "#8a6f5a" }}>
                {r.mult.toFixed(1)}×
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Software commodities ───────────────────────────────────────────────────── */
function SoftwareSection({ tech }: { tech: Tech[] }) {
  const commodities = useMemo(() => {
    const m = new Map<string, { commodity: string; count: number; pct: number }>();
    for (const t of tech) {
      const cur = m.get(t.commodity);
      if (cur) cur.count += 1;
      else m.set(t.commodity, { commodity: t.commodity, count: 1, pct: t.commodity_avg_pct ?? 0 });
    }
    return Array.from(m.values()).sort((a, b) => b.pct - a.pct);
  }, [tech]);

  const high = commodities.filter((c) => c.pct >= 50);
  const low = commodities.filter((c) => c.pct < 50);

  return (
    <Section title="Software & tools"
      subtitle="The software categories this occupation relies on, ranked by how exposed that category is economy-wide (average % of tasks exposed across occupations that use it). Tools count is how many of this job's tools fall in that category.">
      <SoftTier label="AI shows strong capability here (≥ 50% exposed)" rows={high} good />
      <SoftTier label="Lower exposure (< 50%)" rows={low} />
    </Section>
  );
}

function SoftTier({ label, rows, good }: { label: string; rows: { commodity: string; count: number; pct: number }[]; good?: boolean }) {
  const [open, setOpen] = useState(!!good);
  if (!rows.length) return null;
  const max = Math.max(1, ...rows.map((r) => r.pct));
  return (
    <div style={{ marginBottom: 10 }}>
      <button onClick={() => setOpen((v) => !v)} style={{ ...bucketHeader, borderLeft: `4px solid ${good ? "#3a5f83" : "#8ea9bf"}` }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{label}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{rows.length} · {open ? "−" : "+"}</span>
      </button>
      {open && (
        <div style={{ display: "flex", flexDirection: "column", gap: 5, padding: "6px 0" }}>
          {rows.map((r) => (
            <div key={r.commodity}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 1 }}>
                <span style={{ color: "var(--text-secondary)" }}>{r.commodity} <span style={{ color: "var(--text-muted)", fontSize: 10.5 }}>· {r.count} tool{r.count > 1 ? "s" : ""}</span></span>
                <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{fmtPct(r.pct)}</span>
              </div>
              <div style={{ height: 9, background: "var(--bg-sidebar)", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${(r.pct / max) * 100}%`, height: "100%", background: good ? "#3a5f83" : "#8ea9bf", opacity: 0.8 }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── shared bits ────────────────────────────────────────────────────────────── */
function Section({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div style={cardStyle}>
      <button onClick={() => setOpen((v) => !v)} style={{ width: "100%", background: "none", border: "none", cursor: "pointer", textAlign: "left", padding: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 17, fontWeight: 700, color: "var(--text-primary)" }}>{title}</span>
          <span style={{ color: "var(--text-muted)", fontSize: 18 }}>{open ? "−" : "+"}</span>
        </div>
      </button>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.55, maxWidth: 880 }}>{subtitle}</div>
      {open && <div style={{ marginTop: 14 }}>{children}</div>}
    </div>
  );
}
function Kpi({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div style={{ ...kpiStyle, flex: "1 1 160px" }}>
      <div style={kpiLabel}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.1 }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>{sub}</div>
    </div>
  );
}
function Chip({ children }: { children: React.ReactNode }) {
  return <span style={{ fontSize: 11, padding: "3px 9px", borderRadius: 20, background: "var(--bg-sidebar)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>{children}</span>;
}

const cardStyle: React.CSSProperties = { background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "20px 24px", marginBottom: 18 };
const kpiStyle: React.CSSProperties = { background: "var(--bg-sidebar)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 2 };
const kpiLabel: React.CSSProperties = { fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" };
const lblStyle: React.CSSProperties = { display: "block", fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 };
const inputStyle: React.CSSProperties = { padding: "8px 11px", fontSize: 13, borderRadius: 7, border: "1px solid var(--border)", background: "var(--bg-surface)", color: "var(--text-primary)" };
const bucketHeader: React.CSSProperties = { width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 12px", background: "var(--bg-sidebar)", border: "1px solid var(--border)", borderRadius: 7, cursor: "pointer" };
const taskRow: React.CSSProperties = { display: "flex", alignItems: "center", gap: 10, padding: "8px 6px", cursor: "pointer" };
const taskCol: React.CSSProperties = { width: 80, textAlign: "right", fontSize: 12, flexShrink: 0 };
const whyBtn: React.CSSProperties = { background: "none", border: "none", color: "var(--brand)", cursor: "pointer", fontSize: 11.5, marginTop: 4, textDecoration: "underline", padding: 0 };
