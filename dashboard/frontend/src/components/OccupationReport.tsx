"use client";

/* My Occupation — the v2 card (PRD §3.1).
 *
 * One control (which job). Four headline numbers. Then the verb families,
 * each showing the work AI has been observed doing AND the work it hasn't,
 * with the usage signal beside it. Expand a family for its tasks; expand a
 * task for its work-activity hierarchy and the AI tools aimed at it.
 *
 * Language rule: we observe AI being used on a task and how completely it
 * did the work. We do NOT observe what AI is capable of, and exposure is not
 * what AI is doing today — it is what AI would do or assist if every worker
 * used it as observed. So labels say "exposed", never "can do", "AI-capable"
 * or "observed doing".
 */

import { useEffect, useMemo, useState } from "react";
import { fetchOccupationReport, fetchOccupationReportTitles } from "@/lib/api";

/* ── Types ─────────────────────────────────────────────────────────────── */

interface Headline {
  title: string;
  major: string | null; minor: string | null; broad: string | null;
  job_zone: number | null;
  pct_exposed: number; pct_unexposed: number;
  pct_rank: number; total_occupations: number;
  usage_x: number; usage_rank: number | null; usage_of: number;
  pct_first: number | null; change_pp: number | null;
  first_date: string; latest_date: string;
  employment: number; median_wage: number | null; workers_exposed: number;
}

interface FamilyRow {
  family: string; label: string; short: string;
  pct_exposed: number; pct_unexposed: number;
  usage_x: number; usage_share: number; share_of_day: number; n_tasks: number;
}

interface Mcp { title: string; rating: number | null; url: string | null; description: string | null }

interface TaskRow {
  task: string;
  activities: { general: string[]; intermediate: string[]; detailed: string[] };
  top_mcps: Mcp[];
  pct_exposed: number; pct_unexposed: number;
  usage_x: number; usage_share: number; auto_aug: number | null;
}

interface Card { headline: Headline; families: FamilyRow[]; tasks: Record<string, TaskRow[]> }
interface HierEntry { title: string; broad: string; minor: string; major: string }

/* ── Palette ────────────────────────────────────────────────────────────
 * Single hue per measure, intensity carries magnitude. Deliberately not
 * red/green: high exposure is bad news to a worker and good news to an
 * employer, and the chart must not assert either.
 */
const OBSERVED = "#3a5f83";
const NOT_OBSERVED = "#dfe4e8";
const USAGE = "#b0894a";

const nf = new Intl.NumberFormat("en-US");
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
const monthYear = (d?: string) =>
  d ? new Date(d + "T00:00:00Z").toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" }) : "";

/* ── Page ──────────────────────────────────────────────────────────────── */

export default function OccupationReport() {
  const [titles, setTitles] = useState<string[]>([]);
  const [hier, setHier] = useState<HierEntry[]>([]);
  const [title, setTitle] = useState("");
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    fetchOccupationReportTitles().then((d) => {
      setTitles(d.titles);
      setHier((d.hierarchy as HierEntry[]) ?? []);
      if (d.titles.length) setTitle(d.titles.find((t) => t === "Computer Programmers") ?? d.titles[0]);
    });
  }, []);

  useEffect(() => {
    if (!title) return;
    setLoading(true); setStale(false);
    fetchOccupationReport(title, "nat")
      .then((r) => {
        const c = r as unknown as Card;
        // The v2 card and the v1 report share this route, so a frontend that
        // deploys ahead of the backend gets the old shape. Detect it rather
        // than letting `families.map` throw a white screen.
        if (!c || !Array.isArray(c.families) || !c.headline) { setCard(null); setStale(true); return; }
        setCard(c); setStale(false);
      })
      .catch(() => { setCard(null); setStale(true); })
      .finally(() => setLoading(false));
  }, [title]);

  return (
    <div className="page-shell" style={{ maxWidth: 880, margin: "0 auto", padding: "28px 24px 72px" }}>
      <OccupationPicker titles={titles} hier={hier} current={title} onPick={setTitle} />
      {loading && !card && <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</div>}
      {!loading && stale && <BackendMismatch />}
      {card && (
        <>
          <Headlines h={card.headline} />
          <Families families={card.families} tasks={card.tasks} />
          <Footnote />
        </>
      )}
    </div>
  );
}

function BackendMismatch() {
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "18px 20px",
      background: "var(--bg-surface)", fontSize: 13.5, lineHeight: 1.6, color: "var(--text-secondary)" }}>
      <strong style={{ color: "var(--text-primary)" }}>This page is updating.</strong>{" "}
      The API is still serving the previous version of this report. It should resolve on its own
      once the backend finishes deploying — try again in a few minutes.
    </div>
  );
}

/* ── The one control ───────────────────────────────────────────────────── */

function OccupationPicker({ titles, hier, current, onPick }: {
  titles: string[]; hier: HierEntry[]; current: string; onPick: (t: string) => void;
}) {
  const [mode, setMode] = useState<"search" | "browse">("search");
  const major = hier.find((h) => h.title === current)?.major;
  return (
    <div style={{ marginBottom: 26 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 8 }}>
        <label style={{ fontSize: 12, fontWeight: 600, letterSpacing: "0.06em",
          textTransform: "uppercase", color: "var(--text-muted)" }}>What do you do?</label>
        <button onClick={() => setMode(mode === "search" ? "browse" : "search")}
          style={{ fontSize: 12, fontWeight: 600, color: "var(--brand)", background: "none",
            border: "none", cursor: "pointer", padding: 0 }}>
          {mode === "search" ? "Don't know the title? Browse by category →" : "← Back to search"}
        </button>
      </div>
      {mode === "search"
        ? <SearchPicker titles={titles} current={current} onPick={onPick} />
        : <BrowsePicker hier={hier} onPick={(t) => { onPick(t); setMode("search"); }} />}
      {major && mode === "search" && (
        <div style={{ fontSize: 12.5, color: "var(--text-muted)", marginTop: 7 }}>{major}</div>
      )}
    </div>
  );
}

function SearchPicker({ titles, current, onPick }: {
  titles: string[]; current: string; onPick: (t: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const starts = titles.filter((t) => t.toLowerCase().startsWith(q));
    const rest = titles.filter((t) => !t.toLowerCase().startsWith(q) && t.toLowerCase().includes(q));
    return [...starts, ...rest].slice(0, 10);
  }, [query, titles]);
  return (
    <div style={{ position: "relative" }}>
      <input value={open ? query : current}
        onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => { setQuery(""); setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Search 923 occupations…"
        style={{ width: "100%", fontSize: 19, fontWeight: 500, padding: "13px 16px",
          border: "1px solid var(--border)", borderRadius: 10, background: "var(--bg-surface)",
          color: "var(--text-primary)", outline: "none" }} />
      {open && matches.length > 0 && (
        <div style={{ position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 20,
          background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10,
          boxShadow: "0 8px 24px rgba(0,0,0,0.10)", maxHeight: 320, overflowY: "auto" }}>
          {matches.map((m) => (
            <div key={m} onMouseDown={() => { onPick(m); setOpen(false); }}
              style={{ padding: "10px 16px", fontSize: 14, cursor: "pointer", color: "var(--text-primary)" }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--brand-light)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>{m}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/* Narrow down by O*NET's occupational hierarchy for anyone who does not know
 * the official title of their job — which is most people. */
function BrowsePicker({ hier, onPick }: { hier: HierEntry[]; onPick: (t: string) => void }) {
  const [major, setMajor] = useState("");
  const [minor, setMinor] = useState("");
  const [broad, setBroad] = useState("");
  const uniq = (xs: (string | undefined)[]) => Array.from(new Set(xs.filter(Boolean) as string[])).sort();
  const majors = useMemo(() => uniq(hier.map((h) => h.major)), [hier]);
  const minors = useMemo(() => uniq(hier.filter((h) => h.major === major).map((h) => h.minor)), [hier, major]);
  const broads = useMemo(() => uniq(hier.filter((h) => h.minor === minor).map((h) => h.broad)), [hier, minor]);
  const occs = useMemo(() => uniq(hier.filter((h) => h.broad === broad).map((h) => h.title)), [hier, broad]);
  const sel: React.CSSProperties = {
    width: "100%", fontSize: 14, padding: "10px 12px", marginBottom: 8,
    border: "1px solid var(--border)", borderRadius: 8,
    background: "var(--bg-surface)", color: "var(--text-primary)",
  };
  return (
    <div>
      <Step n={1} label="Field">
        <select value={major} onChange={(e) => { setMajor(e.target.value); setMinor(""); setBroad(""); }} style={sel}>
          <option value="">Choose a field…</option>
          {majors.map((m) => <option key={m} value={m}>{m.replace(" Occupations", "")}</option>)}
        </select>
      </Step>
      {major && (
        <Step n={2} label="Group">
          <select value={minor} onChange={(e) => { setMinor(e.target.value); setBroad(""); }} style={sel}>
            <option value="">Choose a group…</option>
            {minors.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </Step>
      )}
      {minor && (
        <Step n={3} label="Kind of role">
          <select value={broad} onChange={(e) => setBroad(e.target.value)} style={sel}>
            <option value="">Choose a kind of role…</option>
            {broads.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </Step>
      )}
      {broad && (
        <Step n={4} label="Occupation">
          <select onChange={(e) => e.target.value && onPick(e.target.value)} style={sel} defaultValue="">
            <option value="">Choose an occupation…</option>
            {occs.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </Step>
      )}
    </div>
  );
}

function Step({ n, label, children }: { n: number; label: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase",
        color: "var(--text-muted)", marginBottom: 4 }}>{n} · {label}</div>
      {children}
    </div>
  );
}

/* ── Four numbers ──────────────────────────────────────────────────────── */

function Headlines({ h }: { h: Headline }) {
  return (
    <>
      <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text-primary)", marginBottom: 3 }}>{h.title}</div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 14 }}>
        {[h.major, h.minor, h.broad].filter(Boolean).join(" · ")}
        {h.job_zone ? ` · Job Zone ${h.job_zone}` : ""}
        {h.median_wage ? ` · median wage $${nf.format(Math.round(h.median_wage))}` : ""}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(178px, 1fr))", gap: 1,
        background: "var(--border)", border: "1px solid var(--border)", borderRadius: 12,
        overflow: "hidden", marginBottom: 12 }}>
        <Stat value={`${h.pct_exposed}%`} label="of work time exposed to AI"
          sub={`${h.pct_unexposed}% not exposed · rank ${h.pct_rank} of ${h.total_occupations}`} accent />
        <Stat value={h.usage_x > 0 ? `${h.usage_x}×` : "—"} label="AI use vs. the median occupation"
          sub={h.usage_rank ? `rank ${h.usage_rank} of ${h.usage_of} occupations with observed use` : "no use observed"} />
        <Stat value={h.change_pp == null ? "—" : `${(h.change_pp ?? 0) > 0 ? "+" : ""}${h.change_pp} pp`}
          label={`change since ${monthYear(h.first_date)}`}
          sub={h.pct_first != null ? `was ${h.pct_first}%` : ""} />
        <Stat value={nf.format(Math.round(h.employment))} label="people do this job"
          sub={`the exposed share is ≈ ${nf.format(h.workers_exposed)} workers' worth of work time`} />
      </div>
      <Caption>
        <strong>Exposure is not the share of work AI is doing today.</strong> It is the share of
        this job&rsquo;s work time AI would do or assist if every worker used it on the tasks
        people have already brought to AI, at the level of automation they brought them at. AI
        use is real usage per worker, compared with the median occupation. Workers&rsquo; worth
        of work time is hours restated as people, not a count of jobs at risk.
      </Caption>
    </>
  );
}

function Stat({ value, label, sub, accent }: { value: string; label: string; sub: string; accent?: boolean }) {
  return (
    <div style={{ background: "var(--bg-surface)", padding: "18px 18px 16px" }}>
      <div style={{ fontSize: 29, fontWeight: 680, lineHeight: 1.05, letterSpacing: "-0.025em",
        color: accent ? OBSERVED : "var(--text-primary)" }}>{value}</div>
      <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 6, lineHeight: 1.35 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4, lineHeight: 1.35 }}>{sub}</div>}
    </div>
  );
}

/* ── Verb families ─────────────────────────────────────────────────────── */

function Families({ families, tasks }: { families: FamilyRow[]; tasks: Record<string, TaskRow[]> }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <section style={{ marginTop: 30 }}>
      <h2 style={{ fontSize: 17, fontWeight: 700, color: "var(--text-primary)", marginBottom: 10 }}>
        What kind of work is exposed
      </h2>
      <div className="mobile-only" style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
        Swipe the table sideways for AI use →
      </div>
      <div className="fam-scroll" style={{ border: "1px solid var(--border)", borderRadius: 12 }}>
        <div className="fam-inner">
        <div className="fam-grid" style={{ alignItems: "end",
          padding: "9px 14px", background: "var(--brand-light)", fontSize: 10, fontWeight: 600,
          letterSpacing: "0.04em", textTransform: "uppercase", lineHeight: 1.3, color: "var(--text-muted)" }}>
          <div />
          <div>Kind of work</div>
          <div>Work time exposed vs. not</div>
          <div style={{ textAlign: "right" }}>Work time exposed</div>
          <div style={{ textAlign: "right" }}>AI use vs. this job&rsquo;s median</div>
          <div style={{ textAlign: "right" }}>Share of this job&rsquo;s AI use</div>
        </div>
        {families.map((f) => (
          <FamilyRowView key={f.family} f={f}
            open={open === f.family} onToggle={() => setOpen(open === f.family ? null : f.family)}
            tasks={tasks[f.family] ?? []} />
        ))}
        </div>
      </div>
      <Caption>
        This job&rsquo;s tasks, grouped by the kind of action they are. The bar and the first
        number are the group&rsquo;s exposure. The last two numbers are real AI use: compared
        with the median group in this job (1× is a typical part of the job), and as a share of
        all AI use seen in this job. Click a row to see its tasks.
      </Caption>
    </section>
  );
}

function FamilyRowView({ f, open, onToggle, tasks }: {
  f: FamilyRow; open: boolean; onToggle: () => void; tasks: TaskRow[];
}) {
  return (
    <div style={{ borderTop: "1px solid var(--border)", background: "var(--bg-surface)" }}>
      <div onClick={onToggle} role="button" tabIndex={0} className="fam-grid"
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onToggle()}
        style={{ padding: "13px 14px", cursor: "pointer", alignItems: "center" }}>
        <Caret open={open} />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{f.label.split(" / ")[0]}</div>
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", marginTop: 2 }}>
            {f.n_tasks} task{f.n_tasks === 1 ? "" : "s"} · {f.share_of_day}% of the day
          </div>
        </div>
        <div style={{ height: 20, borderRadius: 4, overflow: "hidden", display: "flex", background: NOT_OBSERVED }}>
          <div style={{ width: `${f.pct_exposed}%`, background: OBSERVED, transition: "width .25s" }} />
        </div>
        <Num v={`${f.pct_exposed}%`} c={OBSERVED} />
        <Num v={`${f.usage_x}×`} c={USAGE} />
        <Num v={`${f.usage_share}%`} c={USAGE} />
      </div>
      {open && <TaskList tasks={tasks} />}
    </div>
  );
}

function Num({ v, c }: { v: string; c: string }) {
  return <div style={{ textAlign: "right", fontSize: 13.5, fontWeight: 650, color: c }}>{v}</div>;
}

function Caret({ open }: { open: boolean }) {
  // Drawn, not typed: phones render the triangle character as a colour emoji.
  return (
    <svg aria-hidden width="9" height="9" viewBox="0 0 10 10" style={{ display: "block", flexShrink: 0,
      transform: open ? "rotate(90deg)" : "none", transition: "transform .15s" }}>
      <path d="M2 0.5 L9 5 L2 9.5 Z" fill="var(--text-primary)" />
    </svg>
  );
}

/* ── Tasks inside a family ─────────────────────────────────────────────── */

function TaskList({ tasks }: { tasks: TaskRow[] }) {
  return (
    <div className="fam-tasks" style={{ padding: "4px 14px 14px", background: "var(--brand-light)" }}>
      <div className="task-grid" style={{ alignItems: "end",
        fontSize: 10, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase",
        lineHeight: 1.3, color: "var(--text-muted)", padding: "8px 0 4px" }}>
        <div />
        <div>Tasks in this occupation ({tasks.length})</div>
        <div style={{ textAlign: "right" }}>Work time exposed</div>
        <div style={{ textAlign: "right" }}>AI use vs. this job&rsquo;s median task</div>
      </div>
      {tasks.length === 0 && <div style={{ fontSize: 12.5, color: "var(--text-muted)" }}>No tasks listed.</div>}
      {tasks.map((t) => <TaskItem key={t.task} t={t} />)}
    </div>
  );
}

function TaskItem({ t }: { t: TaskRow }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <div onClick={() => setOpen(!open)} role="button" tabIndex={0} className="task-grid"
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setOpen(!open)}
        style={{ alignItems: "baseline", padding: "9px 0", cursor: "pointer" }}>
        <Caret open={open} />
        <div style={{ fontSize: 12.5, color: "var(--text-primary)", lineHeight: 1.45 }}>{sentence(t.task)}</div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: OBSERVED, textAlign: "right" }}>{t.pct_exposed}%</div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: USAGE, textAlign: "right" }}>{t.usage_x}×</div>
      </div>
      {open && (
        <div style={{ padding: "2px 0 14px 26px" }}>
          <ActivityLevel label="General work activity" items={t.activities.general} />
          <ActivityLevel label="Intermediate work activity" items={t.activities.intermediate} />
          <ActivityLevel label="Detailed work activity" items={t.activities.detailed} />
          {t.top_mcps.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <SubLabel>AI tools built for this task ({t.top_mcps.length})</SubLabel>
              {t.top_mcps.map((m) => (
                <div key={m.title} style={{ marginBottom: 6 }}>
                  <div style={{ fontSize: 12, color: "var(--text-primary)" }}>
                    {m.url
                      ? <a href={m.url} target="_blank" rel="noopener noreferrer"
                          style={{ color: "var(--brand)", textDecoration: "none" }}>{m.title}</a>
                      : m.title}
                    {m.rating != null && (
                      <span style={{ color: "var(--text-muted)", marginLeft: 6 }}>· match {m.rating}/5</span>
                    )}
                  </div>
                  {m.description && (
                    <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.45, marginTop: 1 }}>
                      {m.description.length > 190 ? m.description.slice(0, 190) + "…" : m.description}
                    </div>
                  )}
                </div>
              ))}
              <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.5, marginTop: 6 }}>
                MCP servers our classification pipeline matched to this task, with how well each
                matched. A tool existing is not evidence it is in use here.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ActivityLevel({ label, items }: { label: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div style={{ marginBottom: 7 }}>
      <SubLabel>{label}</SubLabel>
      {items.map((a) => (
        <div key={a} style={{ fontSize: 11.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>{a}</div>
      ))}
    </div>
  );
}

function SubLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 9.5, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase",
      color: "var(--text-muted)", marginBottom: 2 }}>{children}</div>
  );
}

/* ── Chrome ────────────────────────────────────────────────────────────── */

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.65, marginTop: 10, maxWidth: 780 }}>
      {children}
    </p>
  );
}

function Footnote() {
  return (
    <p style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.65, marginTop: 26,
      borderTop: "1px solid var(--border)", paddingTop: 16 }}>
      Exposure counts only tasks people have brought to AI often enough to pass a minimum
      threshold. It assumes every worker uses AI on those tasks, so it is an upper bound on what
      is happening today. It is not a measure of what AI is capable of, and not a forecast of job
      loss. Physical work is under-covered: the underlying record is digital AI use.
    </p>
  );
}
