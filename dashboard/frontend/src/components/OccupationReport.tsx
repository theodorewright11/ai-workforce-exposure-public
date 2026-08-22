"use client";

/* My Occupation — the v2 card (PRD §3.1).
 *
 * One control (which job). Four headline numbers. Then the verb families,
 * each showing what AI reaches AND what it doesn't, with the usage signal
 * beside it. Expand a family for the actual tasks.
 *
 * Every exposure bar draws its own complement — that is the contrast rule
 * made visual, and it is why nothing here is a lone number.
 */

import { useEffect, useMemo, useState } from "react";
import { fetchOccupationReport, fetchOccupationReportTitles } from "@/lib/api";

/* ── Types ─────────────────────────────────────────────────────────────── */

interface Headline {
  title: string;
  major: string | null;
  minor: string | null;
  broad: string | null;
  job_zone: number | null;
  pct_exposed: number;
  pct_unexposed: number;
  pct_rank: number;
  total_occupations: number;
  usage_x: number;
  usage_rank: number | null;
  usage_of: number;
  pct_first: number | null;
  change_pp: number | null;
  first_date: string;
  latest_date: string;
  employment: number;
  median_wage: number | null;
  workers_exposed: number;
}

interface FamilyRow {
  family: string;
  label: string;
  short: string;
  pct_exposed: number;
  pct_unexposed: number;
  usage_x: number;
  usage_share: number;
  share_of_day: number;
  n_tasks: number;
}

interface TaskRow {
  task: string;
  activities: string[];
  pct_exposed: number;
  pct_unexposed: number;
  usage_x: number;
  usage_share: number;
  auto_aug: number | null;
}

interface Card {
  headline: Headline;
  families: FamilyRow[];
  tasks: Record<string, TaskRow[]>;
}

interface HierEntry { title: string; broad: string; minor: string; major: string }

/* ── Palette ───────────────────────────────────────────────────────────────
 * Single hue per measure, intensity carries magnitude. Deliberately not
 * red/green: high exposure is bad news to a worker and good news to an
 * employer, and the chart must not assert either.
 */
const EXPOSED = "#3a5f83";        // steel blue — what AI reaches
const UNEXPOSED = "#dfe4e8";      // pale grey — what it doesn't
const USAGE = "#b0894a";          // warm sand — observed AI use

const nf = new Intl.NumberFormat("en-US");
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function monthYear(d?: string) {
  if (!d) return "";
  const dt = new Date(d + "T00:00:00Z");
  return dt.toLocaleString("en-US", { month: "long", year: "numeric", timeZone: "UTC" });
}

/* ── Page ──────────────────────────────────────────────────────────────── */

export default function OccupationReport() {
  const [titles, setTitles] = useState<string[]>([]);
  const [hier, setHier] = useState<HierEntry[]>([]);
  const [title, setTitle] = useState("");
  const [card, setCard] = useState<Card | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchOccupationReportTitles().then((d) => {
      setTitles(d.titles);
      setHier((d.hierarchy as HierEntry[]) ?? []);
      if (d.titles.length) {
        setTitle(d.titles.find((t) => t === "Computer Programmers") ?? d.titles[0]);
      }
    });
  }, []);

  useEffect(() => {
    if (!title) return;
    setLoading(true);
    fetchOccupationReport(title, "nat")
      .then((r) => setCard(r as unknown as Card))
      .finally(() => setLoading(false));
  }, [title]);

  return (
    <div style={{ maxWidth: 880, margin: "0 auto", padding: "28px 24px 72px" }}>
      <OccupationPicker titles={titles} hier={hier} current={title} onPick={setTitle} />
      {loading && !card && <div style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</div>}
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

/* ── The one control ───────────────────────────────────────────────────── */

function OccupationPicker({
  titles, hier, current, onPick,
}: { titles: string[]; hier: HierEntry[]; current: string; onPick: (t: string) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const starts = titles.filter((t) => t.toLowerCase().startsWith(q));
    const rest = titles.filter((t) => !t.toLowerCase().startsWith(q) && t.toLowerCase().includes(q));
    return [...starts, ...rest].slice(0, 10);
  }, [query, titles]);

  const major = hier.find((h) => h.title === current)?.major;

  return (
    <div style={{ marginBottom: 28 }}>
      <label style={{ display: "block", fontSize: 12, fontWeight: 600, letterSpacing: "0.06em",
        textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 8 }}>
        What do you do?
      </label>
      <div style={{ position: "relative" }}>
        <input
          value={open ? query : current}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => { setQuery(""); setOpen(true); }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search 923 occupations…"
          style={{
            width: "100%", fontSize: 19, fontWeight: 500, padding: "13px 16px",
            border: "1px solid var(--border)", borderRadius: 10,
            background: "var(--bg-surface)", color: "var(--text-primary)", outline: "none",
          }}
        />
        {open && matches.length > 0 && (
          <div style={{
            position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 20,
            background: "var(--bg-surface)", border: "1px solid var(--border)",
            borderRadius: 10, boxShadow: "0 8px 24px rgba(0,0,0,0.10)",
            maxHeight: 320, overflowY: "auto",
          }}>
            {matches.map((m) => (
              <div key={m} onMouseDown={() => { onPick(m); setOpen(false); }}
                style={{ padding: "10px 16px", fontSize: 14, cursor: "pointer", color: "var(--text-primary)" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--brand-light)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
                {m}
              </div>
            ))}
          </div>
        )}
      </div>
      {major && !open && (
        <div style={{ fontSize: 12.5, color: "var(--text-muted)", marginTop: 7 }}>{major}</div>
      )}
    </div>
  );
}

/* ── Four numbers ──────────────────────────────────────────────────────── */

function Headlines({ h }: { h: Headline }) {
  const rising = (h.change_pp ?? 0) > 0;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
      gap: 1, background: "var(--border)", border: "1px solid var(--border)",
      borderRadius: 12, overflow: "hidden", marginBottom: 30,
    }}>
      <Stat
        value={`${h.pct_exposed}%`}
        label="of work time AI can do"
        sub={`${h.pct_unexposed}% it can't · rank ${h.pct_rank} of ${h.total_occupations}`}
        accent
      />
      <Stat
        value={h.usage_x > 0 ? `${h.usage_x}×` : "—"}
        label="AI use vs. the median job"
        sub={h.usage_rank ? `rank ${h.usage_rank} of ${h.usage_of}` : "no usage observed"}
      />
      <Stat
        value={h.change_pp == null ? "—" : `${rising ? "+" : ""}${h.change_pp} pp`}
        label={`since ${monthYear(h.first_date)}`}
        sub={h.pct_first != null ? `was ${h.pct_first}%` : ""}
      />
      <Stat
        value={nf.format(Math.round(h.employment))}
        label="people do this job"
        sub={`≈ ${nf.format(h.workers_exposed)} FTE of it is exposed`}
      />
    </div>
  );
}

function Stat({ value, label, sub, accent }: {
  value: string; label: string; sub: string; accent?: boolean;
}) {
  return (
    <div style={{ background: "var(--bg-surface)", padding: "18px 18px 16px" }}>
      <div style={{
        fontSize: 30, fontWeight: 680, lineHeight: 1.05, letterSpacing: "-0.025em",
        color: accent ? EXPOSED : "var(--text-primary)",
      }}>{value}</div>
      <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 6, lineHeight: 1.35 }}>{label}</div>
      {sub && <div style={{ fontSize: 11.5, color: "var(--text-muted)", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

/* ── Verb families ─────────────────────────────────────────────────────── */

function Families({ families, tasks }: { families: FamilyRow[]; tasks: Record<string, TaskRow[]> }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <section>
      <h2 style={{ fontSize: 17, fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
        What kind of work is exposed
      </h2>
      <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.55, marginBottom: 16 }}>
        This job&rsquo;s tasks grouped by the kind of work they are. The filled part of each bar is
        the share of that work AI has been observed doing; the rest is what it hasn&rsquo;t.
        Click a row for the tasks behind it.
      </p>
      <div style={{ border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden" }}>
        {families.map((f, i) => (
          <FamilyRowView
            key={f.family} f={f} first={i === 0}
            open={open === f.family}
            onToggle={() => setOpen(open === f.family ? null : f.family)}
            tasks={tasks[f.family] ?? []}
          />
        ))}
      </div>
    </section>
  );
}

function FamilyRowView({ f, first, open, onToggle, tasks }: {
  f: FamilyRow; first: boolean; open: boolean; onToggle: () => void; tasks: TaskRow[];
}) {
  return (
    <div style={{ borderTop: first ? "none" : "1px solid var(--border)", background: "var(--bg-surface)" }}>
      <div onClick={onToggle} style={{ padding: "14px 16px", cursor: "pointer", display: "grid",
        gridTemplateColumns: "150px 1fr 88px", gap: 14, alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{f.label.split(" / ")[0]}</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
            {f.n_tasks} task{f.n_tasks === 1 ? "" : "s"} · {f.share_of_day}% of the day
          </div>
        </div>

        {/* the bar draws its own complement */}
        <div>
          <div style={{ height: 22, borderRadius: 4, overflow: "hidden", display: "flex", background: UNEXPOSED }}>
            <div style={{ width: `${f.pct_exposed}%`, background: EXPOSED, transition: "width .25s" }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginTop: 4 }}>
            <span style={{ color: EXPOSED, fontWeight: 600 }}>{f.pct_exposed}% AI can do</span>
            <span style={{ color: "var(--text-muted)" }}>{f.pct_unexposed}% it can&rsquo;t</span>
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 15, fontWeight: 650, color: f.usage_share > 0 ? USAGE : "var(--text-muted)" }}>
            {f.usage_share > 0 ? `${f.usage_share}%` : "0%"}
          </div>
          <div style={{ fontSize: 10.5, color: "var(--text-muted)", lineHeight: 1.25, marginTop: 2 }}>
            of this job&rsquo;s AI use
          </div>
        </div>
      </div>

      {open && (
        <div style={{ padding: "2px 16px 14px", background: "var(--brand-light)" }}>
          {tasks.length === 0 && (
            <div style={{ fontSize: 12.5, color: "var(--text-muted)", padding: "8px 0" }}>No tasks listed.</div>
          )}
          {tasks.map((t) => (
            <div key={t.task} style={{ padding: "9px 0", borderTop: "1px solid var(--border)" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 64px 60px", gap: 12, alignItems: "baseline" }}>
                <div style={{ fontSize: 12.5, color: "var(--text-primary)", lineHeight: 1.45 }}>
                  {sentence(t.task)}
                </div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: EXPOSED, textAlign: "right" }}>
                  {t.pct_exposed}%
                </div>
                <div style={{ fontSize: 12.5, color: t.usage_share > 0 ? USAGE : "var(--text-muted)", textAlign: "right" }}>
                  {t.usage_share}%
                </div>
              </div>
              {t.activities.length > 0 && (
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
                  {t.activities.slice(0, 2).join(" · ")}
                  {t.activities.length > 2 ? ` · +${t.activities.length - 2} more` : ""}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Framing (PRD §7) ──────────────────────────────────────────────────── */

function Footnote() {
  return (
    <p style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.6, marginTop: 24 }}>
      Exposure measures task-level overlap with observed AI capability, weighted by how much
      of the workday each task takes. It is not a forecast of job loss, and it is an upper
      bound — it compresses how often AI is used with how completely it does the work.
      Physical work is under-covered by construction: the underlying data is digital AI use.
    </p>
  );
}
