"use client";

import { useEffect, useMemo, useState } from "react";
import type { ConfigResponse } from "@/lib/types";
import { fetchOccupationReport, fetchOccupationReportTitles } from "@/lib/api";
import { fmtPct, fmtCount, fmtWages, ordinal } from "@/lib/format";

/* ── Types ──────────────────────────────────────────────────────────────────── */
interface Gates { pct: number; ska: number; growth: number; emp_decline: number; count: number; emp_proj: number | null; growth_rank: number | null; ska_rank: number | null; total: number; }
interface Intensity { occ_intensity_rank?: number; occ_intensity_total?: number; occ_intensity_x_median?: number; }
interface Headline {
  title: string; major: string | null; minor: string | null; broad: string | null;
  job_zone: number | null; n_tasks: number | null; emp: number | null; wage: number | null;
  pct_tasks_affected: number | null; workers_affected: number | null; wages_affected: number | null;
  gates: Gates; intensity: Intensity;
}
interface RankBlock { pct: number; workers: number; wages: number; total: number; }
interface GroupRanks { economy: RankBlock; }
interface TrendPt { date: string; pct_tasks_affected: number; }
interface WaDetail { name: string; auto: number | null; rank_pct: number | null; total: number | null; }
interface Mcp { title: string; rating: number | null; url: string | null; description: string | null; }
interface Task {
  task: string; task_normalized: string; centrality: number | null; physical: boolean;
  auto: number | null; auto_label: string; color_bucket: string; usage_mult: number | null;
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
interface HierEntry { title: string; major: string | null; minor: string | null; broad: string | null; }

/* ── Palette / labels ───────────────────────────────────────────────────────── */
const BUCKET = {
  high: { label: "High automation seen",     color: "#a8824a" },
  mid:  { label: "Moderate automation seen", color: "#c9b27e" },
  low:  { label: "Low automation seen",      color: "#8ea9bf" },
  none: { label: "No usage seen",            color: "#c9ccce" },
} as const;
type BucketKey = keyof typeof BUCKET;
const GATE_BLUES = ["#dbe6f0", "#aec6dd", "#7da3c4", "#4f7da8", "#2f5f86"];
const ZONE_LABEL: Record<number, string> = { 1: "little/no prep", 2: "some prep", 3: "medium prep", 4: "considerable prep", 5: "extensive prep" };
const MONTHS = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"];

function bucketKey(b: string): BucketKey { return (["high", "mid", "low", "none"].includes(b) ? b : "none") as BucketKey; }
function monthYear(iso?: string): string {
  if (!iso) return "";
  const [y, m] = iso.split("-");
  return `${MONTHS[Number(m) - 1] ?? m} ${y}`;
}

/* ── Component ──────────────────────────────────────────────────────────────── */
export default function OccupationReport({ config }: { config: ConfigResponse }) {
  const [titles, setTitles] = useState<string[]>([]);
  const [hier, setHier] = useState<HierEntry[]>([]);
  const [title, setTitle] = useState<string>("");
  const [geo, setGeo] = useState("nat");
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"search" | "browse">("search");

  useEffect(() => {
    fetchOccupationReportTitles().then((d) => {
      setTitles(d.titles);
      setHier((d.hierarchy as HierEntry[]) ?? []);
      if (d.titles.length) setTitle(d.titles.find((t) => t === "Computer Programmers") ?? d.titles[0]);
    });
  }, []);
  useEffect(() => {
    if (!title) return;
    setLoading(true);
    fetchOccupationReport(title, geo).then((r) => setReport(r as unknown as Report)).finally(() => setLoading(false));
  }, [title, geo]);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "24px 24px 64px" }}>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 20 }}>
        <div style={{ flex: 1, minWidth: 280 }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
            <span style={lblStyle}>Occupation</span>
            <button onClick={() => setMode("search")} style={miniTab(mode === "search")}>Search</button>
            <button onClick={() => setMode("browse")} style={miniTab(mode === "browse")}>Browse by category</button>
          </div>
          {mode === "search"
            ? <SearchPicker titles={titles} current={title} onPick={setTitle} />
            : <BrowsePicker hier={hier} onPick={setTitle} />}
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

/* ── Pickers ────────────────────────────────────────────────────────────────── */
function SearchPicker({ titles, current, onPick }: { titles: string[]; current: string; onPick: (t: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? titles.filter((t) => t.toLowerCase().includes(q)).slice(0, 12) : [];
  }, [query, titles]);
  return (
    <div style={{ position: "relative" }}>
      <input value={open ? query : current}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => { setQuery(""); setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Search occupations…" style={{ ...inputStyle, width: "100%" }} />
      {open && matches.length > 0 && (
        <div style={dropdown}>
          {matches.map((m) => (
            <div key={m} onMouseDown={() => { onPick(m); setOpen(false); }} style={dropItem}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--brand-light)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>{m}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function BrowsePicker({ hier, onPick }: { hier: HierEntry[]; onPick: (t: string) => void }) {
  const [major, setMajor] = useState("");
  const [minor, setMinor] = useState("");
  const [broad, setBroad] = useState("");
  const majors = useMemo(() => Array.from(new Set(hier.map((h) => h.major).filter(Boolean))).sort() as string[], [hier]);
  const minors = useMemo(() => Array.from(new Set(hier.filter((h) => h.major === major).map((h) => h.minor).filter(Boolean))).sort() as string[], [hier, major]);
  const broads = useMemo(() => Array.from(new Set(hier.filter((h) => h.minor === minor).map((h) => h.broad).filter(Boolean))).sort() as string[], [hier, minor]);
  const occs = useMemo(() => hier.filter((h) => h.broad === broad).map((h) => h.title).sort(), [hier, broad]);
  const sel: React.CSSProperties = { ...inputStyle, width: "100%", marginBottom: 6 };
  return (
    <div>
      <select value={major} onChange={(e) => { setMajor(e.target.value); setMinor(""); setBroad(""); }} style={sel}>
        <option value="">Major category…</option>{majors.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
      {major && <select value={minor} onChange={(e) => { setMinor(e.target.value); setBroad(""); }} style={sel}>
        <option value="">Minor category…</option>{minors.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>}
      {minor && <select value={broad} onChange={(e) => setBroad(e.target.value)} style={sel}>
        <option value="">Broad occupation…</option>{broads.map((b) => <option key={b} value={b}>{b}</option>)}
      </select>}
      {broad && <select onChange={(e) => e.target.value && onPick(e.target.value)} style={sel} defaultValue="">
        <option value="">Occupation…</option>{occs.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>}
    </div>
  );
}

/* ── Overview ───────────────────────────────────────────────────────────────── */
function Overview({ report }: { report: Report }) {
  const h = report.headline;
  const g = h.gates;
  const econ = report.group_ranks.economy;
  const pct = h.pct_tasks_affected ?? 0;
  const firstPt = report.trend[0];
  const pctFirst = firstPt?.pct_tasks_affected;
  const when = monthYear(firstPt?.date);
  const zone = h.job_zone ? Math.round(h.job_zone) : null;
  const nRated = report.tasks.filter((t) => bucketKey(t.color_bucket) !== "none").length;
  const xMed = h.intensity.occ_intensity_x_median;

  const workersFirst = pctFirst != null && h.emp != null ? (pctFirst / 100) * h.emp : null;
  const wagesFirst = workersFirst != null && h.wage != null ? workersFirst * h.wage : null;

  const signals = [
    { on: g.pct, label: "Tasks exposed", val: fmtPct(pct), need: "needs > 50%" },
    { on: g.growth, label: "Exposure growth", val: g.growth_rank != null ? `#${g.growth_rank} of ${g.total}` : "—", need: "needs above median" },
    { on: g.ska, label: "AI reach into skills, knowledge & abilities", val: g.ska_rank != null ? `#${g.ska_rank} of ${g.total}` : "—", need: "needs above median" },
    { on: g.emp_decline, label: "Employment outlook", val: g.emp_proj != null ? `${g.emp_proj}%` : "—", need: "needs projected decline" },
  ];

  return (
    <div className="report-card" style={cardStyle}>
      <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.2 }}>{h.title}</div>
      {/* line 1: hierarchy */}
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 5 }}>
        <strong>Major:</strong> {h.major}{"  ·  "}<strong>Minor:</strong> {h.minor}{"  ·  "}<strong>Broad:</strong> {h.broad}
      </div>
      {/* line 2: pills */}
      <div style={{ marginTop: 9, display: "flex", gap: 8, flexWrap: "wrap" }}>
        {zone && <Chip>Job Zone {zone} · {ZONE_LABEL[zone]}</Chip>}
        <Chip>{nRated}/{h.n_tasks ?? 0} tasks rated</Chip>
        {h.wage != null && <Chip>Median wage {fmtWages(h.wage)}</Chip>}
        {h.emp != null && <Chip>{fmtCount(h.emp)} employed</Chip>}
      </div>
      {/* line 3: exposure-flag strip — wraps on narrow screens */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 16, alignItems: "stretch" }}>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", padding: "8px 18px", background: GATE_BLUES[g.count], color: g.count >= 3 ? "#fff" : "var(--text-primary)", borderRadius: 9, minWidth: 92 }}>
          <span style={{ fontWeight: 800, fontSize: 19, lineHeight: 1 }}>{g.count} of 4</span>
          <span style={{ fontSize: 9.5, opacity: 0.9, textTransform: "uppercase", letterSpacing: "0.05em", marginTop: 3 }}>Exposure flags</span>
        </div>
        {signals.map((s, i) => (
          <div key={i} style={{ flex: "1 1 160px", minWidth: 150, padding: "8px 12px", border: "1px solid var(--border)", borderRadius: 9, display: "flex", flexDirection: "column", gap: 2 }}>
            <span style={{ fontSize: 11, color: s.on ? "var(--text-primary)" : "var(--text-muted)", fontWeight: s.on ? 600 : 400, lineHeight: 1.3 }}>
              <span style={{ color: s.on ? "#2f5f86" : "var(--text-muted)", marginRight: 5 }}>{s.on ? "●" : "○"}</span>{s.label}
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: s.on ? "#2f5f86" : "var(--text-muted)" }}>{s.val}</span>
            <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{s.need}</span>
          </div>
        ))}
      </div>

      {/* KPI row — all same size, Tasks highlighted */}
      <div style={{ display: "flex", gap: 14, marginTop: 18, flexWrap: "wrap" }}>
        <Kpi highlight label="% Tasks Exposed" value={fmtPct(pct)}
          up={pctFirst != null && pctFirst < pct ? `from ${fmtPct(pctFirst)} · ${when}` : undefined}
          sub={`rank ${ordinal(econ.pct)} of ${econ.total} occupations`} />
        <Kpi label="Workers Exposed" value={fmtCount(h.workers_affected ?? 0)}
          up={workersFirst != null && workersFirst < (h.workers_affected ?? 0) ? `from ${fmtCount(workersFirst)} · ${when}` : undefined}
          sub={`rank ${ordinal(econ.workers)} of ${econ.total} occupations`} />
        <Kpi label="Wages Exposed" value={fmtWages(h.wages_affected ?? 0)}
          up={wagesFirst != null && wagesFirst < (h.wages_affected ?? 0) ? `from ${fmtWages(wagesFirst)} · ${when}` : undefined}
          sub={`rank ${ordinal(econ.wages)} of ${econ.total} occupations`} />
        <Kpi label="Actual Usage Rank" value={xMed != null ? `${xMed}×` : "—"}
          sub={h.intensity.occ_intensity_rank ? `rank ${ordinal(h.intensity.occ_intensity_rank)} of ${h.intensity.occ_intensity_total} occupations` : "—"} />
      </div>
    </div>
  );
}

/* ── Ranking within major ───────────────────────────────────────────────────── */
function RankingInMajor({ mr }: { mr: MajorRanking }) {
  if (!mr.major || !mr.pct) return null;
  return (
    <Section title="Where it ranks in its major category"
      subtitle={`Its place among the ${mr.pct.total} occupations in ${mr.major}, with its 5 nearest neighbours on each metric. Left: share of tasks exposed. Right: actual AI usage, as a multiple of the median occupation's usage relative to task need.`}>
      <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
        <RankWindowChart title={`% Tasks Exposed — rank ${mr.pct.rank} of ${mr.pct.total}`} win={mr.pct.window} fmt={(v) => fmtPct(v)} color="#3a5f83" />
        {mr.adoption && <RankWindowChart title={`Actual Usage — rank ${mr.adoption.rank} of ${mr.adoption.total}`} win={mr.adoption.window} fmt={(v) => `${v.toFixed(1)}×`} color="#4a7c6f" />}
      </div>
    </Section>
  );
}
function RankWindowChart({ title, win, fmt, color }: { title: string; win: RankWindow["window"]; fmt: (v: number) => string; color: string }) {
  const max = Math.max(1, ...win.map((w) => w.value ?? 0));
  return (
    <div style={{ flex: 1, minWidth: 280 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 8 }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {win.map((w) => (
          <div key={w.title}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 1, color: w.is_occ ? "var(--brand)" : "var(--text-secondary)", fontWeight: w.is_occ ? 700 : 400 }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "72%" }} title={w.title}>#{w.rank} {w.title}</span>
              <span style={{ fontWeight: 600 }}>{fmt(w.value ?? 0)}</span>
            </div>
            <div style={{ height: 9, background: "var(--bg-sidebar)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${((w.value ?? 0) / max) * 100}%`, height: "100%", background: w.is_occ ? "var(--brand)" : color, opacity: w.is_occ ? 1 : 0.7 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Tasks ──────────────────────────────────────────────────────────────────── */
function TasksSection({ tasks }: { tasks: Task[] }) {
  const centralityRank = useMemo(() => {
    const order = [...tasks].sort((a, b) => (b.centrality ?? 0) - (a.centrality ?? 0));
    const m = new Map<string, number>();
    order.forEach((t, i) => m.set(t.task_normalized, i + 1));
    return m;
  }, [tasks]);
  const groups = (["high", "mid", "low", "none"] as BucketKey[])
    .map((k) => ({ key: k, items: tasks.filter((t) => bucketKey(t.color_bucket) === k) })).filter((g) => g.items.length);
  return (
    <Section title="Tasks"
      subtitle="Every task in this occupation, grouped by its automation level — 1 = little or no automation, 5 = full or near-full automation seen. Centrality ranks how core the task is to the job (frequency × importance × relevance); Usage is how much AI is used on it as a multiple of the occupation's median task usage. Expand a row for its work activities and the AI tools (MCP) that target it.">
      {groups.map((g) => <TaskBucket key={g.key} bucket={g.key} items={g.items} centralityRank={centralityRank} total={tasks.length} />)}
    </Section>
  );
}
function TaskBucket({ bucket, items, centralityRank, total }: { bucket: BucketKey; items: Task[]; centralityRank: Map<string, number>; total: number }) {
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
          <div style={{ ...taskRow, cursor: "default", fontSize: 10.5, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em" }}>
            <span style={{ flex: 1, minWidth: 0 }}>Task</span>
            <span style={taskCol}>Centrality</span>
            <span style={taskCol}>Usage</span>
            <span style={{ ...taskCol, width: 92, lineHeight: 1.15 }}>Automation<br />level</span>
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
      <div style={taskRow} onClick={() => setOpen((v) => !v)}>
        <span style={{ flex: 1, minWidth: 0, fontSize: 12.5, wordBreak: "break-word" }}><span style={{ opacity: 0.4, marginRight: 6 }}>{open ? "▾" : "▸"}</span>{t.task}</span>
        <span style={taskCol}>{centrality ? `#${centrality}` : "—"}<span style={{ color: "var(--text-muted)" }}>/{total}</span></span>
        <span style={taskCol}>{t.usage_mult != null ? `${t.usage_mult}×` : "—"}</span>
        <span style={{ ...taskCol, width: 92, fontSize: 15, fontWeight: 700, color: meta.color }}>{t.auto != null ? `${t.auto}/5` : "—"}</span>
      </div>
      {open && (
        <div style={{ padding: "10px 14px 14px 26px", background: "var(--bg-sidebar)", fontSize: 12 }}>
          <SubHead>Work Activities</SubHead>
          {([["General Work Activity", t.gwa], ["Intermediate Work Activity", t.iwa], ["Detailed Work Activity", t.dwa]] as [string, WaDetail | null][]).map(([lvl, w]) =>
            w ? (
              <div key={lvl} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>{lvl}</div>
                <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{w.name}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>
                  Automation level {w.auto != null ? `${w.auto}/5` : "—"} (avg over AI-exposed tasks) · Ranks #{w.rank_pct ?? "—"} of {w.total ?? "—"} by tasks exposed
                </div>
              </div>
            ) : null)}
          {t.top_mcps.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <SubHead>AI tools (MCP) that target this task</SubHead>
              {t.top_mcps.map((m, i) => (
                <div key={i} style={{ marginBottom: 5 }}>
                  <a href={m.url ?? "#"} target="_blank" rel="noreferrer" style={{ color: "var(--brand)", fontWeight: 600, fontSize: 12 }}>{m.title}</a>
                  {m.rating != null && <span style={{ color: "var(--text-muted)", marginLeft: 6 }}>{m.rating}/5 automation level</span>}
                  {m.description && <div style={{ color: "var(--text-muted)", fontSize: 11, lineHeight: 1.4 }}>{m.description.length > 160 ? m.description.slice(0, 160) + "…" : m.description}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── SKA (split by type) ────────────────────────────────────────────────────── */
function SkaSection({ ska }: { ska: Ska }) {
  return (
    <Section title="Skills, Knowledge & Abilities"
      subtitle="The skills, knowledge, and abilities that matter to this job (importance ≥ 3). AI capability is the average of the top-10 AI-capability scores among the occupations most exposed on that element — shown as a multiple of this occupation's need, where 1.0× means AI matches what the job requires and above that means AI already reaches further. Centrality ranks how core each one is to the occupation. This maps task-level exposure onto skills/knowledge/abilities, so read it as directional rather than exact.">
      <SkaType label="Skills" rows={ska.rows.skills} />
      <SkaType label="Knowledge" rows={ska.rows.knowledge} />
      <SkaType label="Abilities" rows={ska.rows.abilities} />
    </Section>
  );
}
function SkaType({ label, rows }: { label: string; rows: SkaRow[] }) {
  const prepped = useMemo(() => {
    const r = rows.filter((x) => x.pct_of_need != null && (x.importance ?? 0) >= 3);
    const byScore = [...r].sort((a, b) => (b.occ_score ?? 0) - (a.occ_score ?? 0));
    const cmap = new Map<string, number>();
    byScore.forEach((x, i) => cmap.set(x.element, i + 1));
    return r.map((x) => ({ ...x, centrality: cmap.get(x.element) ?? 0, mult: (x.pct_of_need ?? 0) / 100 })).sort((a, b) => b.mult - a.mult);
  }, [rows]);
  if (!prepped.length) return null;
  const wins = prepped.filter((r) => r.mult >= 1);
  const behind = prepped.filter((r) => r.mult < 1);
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 }}>{label}</div>
      <SkaQuadrant label="AI does this well" rows={wins} good />
      <SkaQuadrant label="AI is still catching up" rows={behind} />
    </div>
  );
}
function SkaQuadrant({ label, rows, good }: { label: string; rows: (SkaRow & { centrality: number; mult: number })[]; good?: boolean }) {
  const [open, setOpen] = useState(false);
  if (!rows.length) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <button onClick={() => setOpen((v) => !v)} style={{ ...bucketHeader, borderLeft: `4px solid ${good ? "#3a5f83" : "#8ea9bf"}` }}>
        <span style={{ fontWeight: 600, fontSize: 12.5 }}>{label}</span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{rows.length} · {open ? "−" : "+"}</span>
      </button>
      {open && (
        <div style={{ ...taskRow, cursor: "default", fontSize: 10.5, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em" }}>
          <span style={{ flex: 1, minWidth: 0 }}>Element</span>
          <span style={taskCol}>Centrality</span>
          <span style={{ ...taskCol, width: 92, lineHeight: 1.15 }}>AI<br />capability</span>
        </div>
      )}
      {open && rows.map((r) => (
        <div key={r.element} style={{ ...taskRow, cursor: "default", borderBottom: "1px solid var(--border)" }}>
          <span style={{ flex: 1, minWidth: 0, fontSize: 12.5, wordBreak: "break-word" }}>{r.element}</span>
          <span style={taskCol}>#{r.centrality}</span>
          <span style={{ ...taskCol, width: 92, fontWeight: 700, color: r.mult >= 1 ? "#3a5f83" : "#8a6f5a" }}>{r.mult.toFixed(1)}×</span>
        </div>
      ))}
    </div>
  );
}

/* ── Software ───────────────────────────────────────────────────────────────── */
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
  return (
    <Section title="Software & tools"
      subtitle="The software categories this occupation relies on, ranked by how exposed that category is economy-wide (average % of tasks exposed across occupations that use it). Tools is how many of this job's tools fall in that category.">
      <SoftTier label="AI shows strong capability (≥ 50% exposed)" rows={commodities.filter((c) => c.pct >= 50)} good />
      <SoftTier label="Lower capability (< 50%)" rows={commodities.filter((c) => c.pct < 50)} />
    </Section>
  );
}
function SoftTier({ label, rows, good }: { label: string; rows: { commodity: string; count: number; pct: number }[]; good?: boolean }) {
  const [open, setOpen] = useState(false);
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
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, fontSize: 12, marginBottom: 1 }}>
                <span style={{ color: "var(--text-secondary)", minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.commodity} <span style={{ color: "var(--text-muted)", fontSize: 10.5 }}>· {r.count} tool{r.count > 1 ? "s" : ""}</span></span>
                <span style={{ fontWeight: 600, color: "var(--text-primary)", flexShrink: 0 }}>{fmtPct(r.pct)}</span>
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

/* ── shared ─────────────────────────────────────────────────────────────────── */
function Section({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);   // panels default collapsed
  return (
    <div className="report-card" style={cardStyle}>
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
function Kpi({ label, value, sub, up, highlight }: { label: string; value: string; sub: string; up?: string; highlight?: boolean }) {
  return (
    <div style={{ ...kpiStyle, flex: "1 1 170px", ...(highlight ? { background: "var(--brand-light)", borderColor: "var(--brand-border)" } : {}) }}>
      <div style={kpiLabel}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: highlight ? 800 : 700, color: highlight ? "var(--brand)" : "var(--text-primary)", lineHeight: 1.05 }}>{value}</div>
      {up && <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>up {up}</div>}
      <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 3 }}>{sub}</div>
    </div>
  );
}
function Chip({ children }: { children: React.ReactNode }) {
  return <span style={{ fontSize: 11, padding: "3px 9px", borderRadius: 20, background: "var(--bg-sidebar)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}>{children}</span>;
}
function SubHead({ children }: { children: React.ReactNode }) {
  return <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.03em", marginBottom: 5 }}>{children}</div>;
}

const cardStyle: React.CSSProperties = { background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 12, padding: "20px 24px", marginBottom: 18 };
const kpiStyle: React.CSSProperties = { background: "var(--bg-sidebar)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 2 };
const kpiLabel: React.CSSProperties = { fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" };
const lblStyle: React.CSSProperties = { fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" };
const inputStyle: React.CSSProperties = { padding: "8px 11px", fontSize: 13, borderRadius: 7, border: "1px solid var(--border)", background: "var(--bg-surface)", color: "var(--text-primary)" };
const bucketHeader: React.CSSProperties = { width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 12px", background: "var(--bg-sidebar)", border: "1px solid var(--border)", borderRadius: 7, cursor: "pointer" };
const taskRow: React.CSSProperties = { display: "flex", alignItems: "center", gap: 10, padding: "8px 6px", cursor: "pointer" };
const taskCol: React.CSSProperties = { width: 64, textAlign: "right", fontSize: 12, flexShrink: 0 };
const dropdown: React.CSSProperties = { position: "absolute", top: "100%", left: 0, right: 0, zIndex: 30, marginTop: 3, background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 8, boxShadow: "0 6px 20px rgba(0,0,0,0.12)", maxHeight: 320, overflowY: "auto" };
const dropItem: React.CSSProperties = { padding: "8px 12px", fontSize: 13, cursor: "pointer", color: "var(--text-secondary)" };
function miniTab(active: boolean): React.CSSProperties {
  return { fontSize: 11, padding: "1px 8px", borderRadius: 5, cursor: "pointer", border: "1px solid var(--border)", background: active ? "var(--brand-light)" : "transparent", color: active ? "var(--brand)" : "var(--text-muted)", fontWeight: active ? 600 : 400 };
}
