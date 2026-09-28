"use client";

/* Economy at a Glance — six blocks, zero controls (PRD §3.2).
 *
 * Each block mirrors a paper figure and carries a contrast: exposed vs. not,
 * exposure vs. usage, or now vs. then. A block showing a lone number does not
 * ship. Every exposure bar draws its own unexposed remainder.
 */

import { useEffect, useState } from "react";
import {
  fetchEconomy,
  type EconomyResponse, type EconomyFamily, type EconomyGroup,
  type FocusedRow, type StateRow, type EconomyTrend,
} from "@/lib/api";

const EXPOSED = "#3a5f83";   // work AI has been observed doing
const UNEXPOSED = "#dfe4e8"; // work it has not
const USAGE = "#b0894a";

const nf = new Intl.NumberFormat("en-US");
const fmtM = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : nf.format(Math.round(n));
const fmtB = (n: number) =>
  n >= 1e12 ? `$${(n / 1e12).toFixed(2)}T` : n >= 1e9 ? `$${Math.round(n / 1e9)}B` : `$${fmtM(n)}`;

export default function EconomyPage() {
  const [data, setData] = useState<EconomyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetchEconomy("nat").then(setData).catch((e) => setErr((e as Error).message));
  }, []);

  if (err) return <Shell><p style={{ color: "#b91c1c", fontSize: 13 }}>Backend error: {err}</p></Shell>;
  if (!data) return <Shell><p style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading…</p></Shell>;

  return (
    <Shell>
      <Headline trend={data.trend} />
      <TrendBlock trend={data.trend} />
      <FamilyBlock families={data.families} />
      <GroupBlock
        title="Which fields" rows={data.majors}
        blurb="The 22 major occupational categories, most-exposed first. Each row, left to right: a bar whose filled part is the share of that category's work time AI has been observed doing; that share as a number; the number of workers whose full working time the exposed share adds up to; and actual AI use as a multiple of the median category. The workers figure is an amount of work time restated as people, not a count of jobs at risk."
      />
      <GroupBlock
        title="Which activities" rows={data.gwas}
        blurb="The 15 most-exposed of O*NET's 37 general work activities, read the same way as the panel above. This is the standard taxonomy, cut by domain of activity; the verb families above cut the same work by kind of action instead."
      />
      <FocusedBlock focused={data.focused} />
      <StatesBlock states={data.states} />
      <Footnote />
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="page-shell" style={{ maxWidth: 940, margin: "0 auto", padding: "28px 24px 72px" }}>{children}</div>;
}

/* ── 1 · Headline ──────────────────────────────────────────────────────── */

function Headline({ trend }: { trend: EconomyTrend }) {
  return (
    <div style={{ marginBottom: 34 }}>
      <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 10 }}>
        Share of the U.S. workday exposed to AI
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 16, flexWrap: "wrap" }}>
        <div style={{ fontSize: 62, fontWeight: 700, lineHeight: 1, letterSpacing: "-0.03em", color: EXPOSED }}>
          {trend.headline_pct}%
        </div>
        <div style={{ fontSize: 15, color: "var(--text-secondary)" }}>
          {trend.headline_unexposed}% of the workday is not
        </div>
      </div>
      <div style={{ fontSize: 13.5, color: "var(--text-secondary)", marginTop: 10 }}>
        Up <strong style={{ color: "var(--text-primary)" }}>{trend.change_pp} points</strong> from{" "}
        {trend.first_pct}% in {trend.first_date}.
      </div>
      <div style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.55, marginTop: 8, maxWidth: 700 }}>
        <strong>This is not the share of work AI is doing today.</strong> It is the share of work
        time AI would do or assist if every worker used it on the tasks people have already
        brought to AI, at the level of automation they brought them at.
      </div>
      <div style={{ height: 26, borderRadius: 5, overflow: "hidden", display: "flex", background: UNEXPOSED, marginTop: 16 }}>
        <div style={{ width: `${trend.headline_pct}%`, background: EXPOSED }} />
      </div>
    </div>
  );
}

/* ── 2 · Trend ─────────────────────────────────────────────────────────── */

function TrendBlock({ trend }: { trend: EconomyTrend }) {
  const all = trend.series.flatMap((s) => s.points.map((p) => p.pct));
  const max = Math.max(...all, 60);
  const W = 620, H = 190, PADL = 40, PADR = 22, PADB = 26, PADT = 8;
  const dates = trend.series[0]?.points.map((p) => p.date) ?? [];
  const x = (i: number) => PADL + (i / Math.max(dates.length - 1, 1)) * (W - PADL - PADR);
  const y = (v: number) => PADT + (1 - v / max) * (H - PADT - PADB);
  const colors: Record<string, string> = { all: EXPOSED, exclude: "#7aa5c4", only: "#b8c4cc" };

  return (
    <Block title="How fast it is moving"
      blurb="Five snapshots of the same measure, August 2025 to May 2026. Three lines: all work; non-physical work only; and physical work only. Physical work is under-covered by construction — the underlying record is digital AI use — so it is drawn separately rather than hidden inside the aggregate.">
      <div style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", minWidth: 480, height: "auto" }} role="img"
          aria-label="Exposure trend, August 2025 to May 2026">
          {[0, 20, 40, 60].filter((g) => g <= max).map((g) => (
            <g key={g}>
              <line x1={PADL} x2={W - PADR} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth={1} />
              <text x={PADL - 7} y={y(g) + 4} textAnchor="end" fontSize={10} fill="var(--text-muted)">{g}%</text>
            </g>
          ))}
          {trend.series.map((s) => (
            <g key={s.key}>
              <polyline fill="none" stroke={colors[s.key] ?? EXPOSED} strokeWidth={2.5}
                strokeLinejoin="round" strokeLinecap="round"
                points={s.points.map((p, i) => `${x(i)},${y(p.pct)}`).join(" ")} />
              {s.points.map((p, i) => <circle key={i} cx={x(i)} cy={y(p.pct)} r={3} fill={colors[s.key] ?? EXPOSED} />)}
              <text x={x(s.points.length - 1) - 4} y={y(s.points[s.points.length - 1].pct) - 9}
                textAnchor="end" fontSize={11} fontWeight={600} fill={colors[s.key] ?? EXPOSED}>
                {s.label} {s.points[s.points.length - 1].pct}%
              </text>
            </g>
          ))}
          {dates.map((d, i) => (
            <text key={d} x={x(i)} y={H - 8} textAnchor="middle" fontSize={9.5} fill="var(--text-muted)">
              {d.slice(2, 7)}
            </text>
          ))}
        </svg>
      </div>
    </Block>
  );
}

/* ── 3 · Verb families ─────────────────────────────────────────────────── */

function FamilyBlock({ families }: { families: EconomyFamily[] }) {
  return (
    <Block title="What kind of work"
      blurb="AI exposure and actual usage across the eight verb families — every task in the economy grouped by the kind of action it is. Each row, left to right: the family and the share of the workday it accounts for; a bar whose filled part is the share of that family's work time AI has been observed doing and whose remainder is the work it has not; that share as a number; and actual AI use on the family as a multiple of the median family. Exposure varies far more between kinds of work than between fields.">
      <div style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
        {families.map((f, i) => (
          <div key={f.family} className="eco-fam-grid" style={{
            alignItems: "center",
            padding: "12px 14px", background: "var(--bg-surface)",
            borderTop: i ? "1px solid var(--border)" : "none",
          }}>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)" }}>{f.short}</div>
              <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{f.share_of_day}% of the workday</div>
            </div>
            <div>
              <div style={{ height: 18, borderRadius: 3, overflow: "hidden", display: "flex", background: UNEXPOSED }}>
                <div style={{ width: `${f.pct_exposed}%`, background: EXPOSED }} />
              </div>
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: EXPOSED, textAlign: "right" }}>{f.pct_exposed}%</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: USAGE, textAlign: "right" }}>{f.usage_x}×</div>
          </div>
        ))}
      </div>
      <Legend />
    </Block>
  );
}

function Legend() {
  return (
    <div style={{ display: "flex", gap: 18, fontSize: 11, color: "var(--text-muted)", marginTop: 9, flexWrap: "wrap" }}>
      <span><Swatch c={EXPOSED} /> exposed share: what AI would do or assist if everyone used it as observed</span>
      <span><Swatch c={UNEXPOSED} /> the rest</span>
      <span><Swatch c={USAGE} /> actual AI use, vs. the median family</span>
    </div>
  );
}
function Swatch({ c }: { c: string }) {
  return <span style={{ display: "inline-block", width: 9, height: 9, background: c, borderRadius: 2, marginRight: 5 }} />;
}

/* ── 4 / 5 · Majors and GWAs ───────────────────────────────────────────── */

function GroupBlock({ title, rows, blurb }: { title: string; rows: EconomyGroup[]; blurb: string }) {
  return (
    <Block title={title} blurb={blurb}>
      <div style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
        {rows.map((r, i) => (
          <div key={r.category} className="eco-group-grid" style={{
            alignItems: "center",
            padding: "10px 14px", background: "var(--bg-surface)",
            borderTop: i ? "1px solid var(--border)" : "none",
          }}>
            <div className="row-name" style={{ fontSize: 13, color: "var(--text-primary)", overflow: "hidden",
              textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.category}>{r.category}</div>
            <div style={{ height: 15, borderRadius: 3, overflow: "hidden", display: "flex", background: UNEXPOSED }}>
              <div style={{ width: `${r.pct_exposed}%`, background: EXPOSED }} />
            </div>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: EXPOSED, textAlign: "right" }}>{r.pct_exposed}%</div>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", textAlign: "right" }}>{fmtM(r.workers_exposed)}</div>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: USAGE, textAlign: "right" }}>{r.usage_x}×</div>
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "2px 18px", fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
        <span>bar &amp; % — exposed vs. not (if everyone used AI as observed)</span><span>then workers&rsquo; worth of exposed work time</span><span>then AI use vs. median</span>
      </div>
    </Block>
  );
}

/* ── 6 · The 31 ────────────────────────────────────────────────────────── */

function FocusedBlock({ focused }: { focused: EconomyResponse["focused"] }) {
  const [all, setAll] = useState(false);
  const rows = all ? focused.rows : focused.rows.slice(0, 12);
  return (
    <Block title={`The ${focused.count}`}
      blurb={`Occupations where AI already reaches at least ${focused.exposure_min}% of the workday AND the BLS projects employment to fall through 2034. Two gates, nothing else. The exposed work across them adds up to the full working time of ${fmtM(focused.total_workers)} workers.`}>
      <div style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
        <div className="eco-focus-grid" style={{ padding: "8px 14px",
          background: "var(--brand-light)", fontSize: 10.5, fontWeight: 600, letterSpacing: "0.04em",
          textTransform: "uppercase", color: "var(--text-muted)" }}>
          <div>Occupation</div><div style={{ textAlign: "right" }}>Exposed</div>
          <div style={{ textAlign: "right" }}>BLS 25–34</div><div style={{ textAlign: "right" }}>Use</div>
        </div>
        {rows.map((r: FocusedRow) => (
          <div key={r.title} className="eco-focus-grid" style={{
            alignItems: "center", padding: "9px 14px", background: "var(--bg-surface)",
            borderTop: "1px solid var(--border)" }}>
            <div style={{ minWidth: 0 }}>
              <div className="wrap-mobile" style={{ fontSize: 12.5, color: "var(--text-primary)", overflow: "hidden",
                textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.title}>{r.title}</div>
              <div style={{ fontSize: 10.5, color: "var(--text-muted)" }}>{r.major} · {fmtM(r.workers_exposed)} workers&rsquo; worth</div>
            </div>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: EXPOSED, textAlign: "right" }}>{r.pct_exposed}%</div>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", textAlign: "right" }}>{r.emp_proj_pct}%</div>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: USAGE, textAlign: "right" }}>{r.usage_x}×</div>
          </div>
        ))}
      </div>
      {focused.rows.length > 12 && (
        <button onClick={() => setAll(!all)} style={{
          marginTop: 10, fontSize: 12.5, fontWeight: 600, color: "var(--brand)", background: "none",
          border: "none", cursor: "pointer", padding: 0,
        }}>{all ? "Show fewer" : `Show all ${focused.count}`}</button>
      )}
    </Block>
  );
}

/* ── 7 · States ────────────────────────────────────────────────────────── */

function StatesBlock({ states }: { states: EconomyResponse["states"] }) {
  const max = Math.max(...states.top.map((s) => s.pct_exposed));
  const Row = (s: StateRow) => (
    <div key={s.geo} style={{ display: "grid", gridTemplateColumns: "24px 1fr 78px 52px", gap: 10,
      alignItems: "center", padding: "8px 14px", background: "var(--bg-surface)",
      borderTop: "1px solid var(--border)" }}>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{s.rank}</div>
      <div style={{ fontSize: 12.5, color: "var(--text-primary)" }}>{s.state}</div>
      <div style={{ height: 13, borderRadius: 3, overflow: "hidden", display: "flex", background: UNEXPOSED }}>
        <div style={{ width: `${(s.pct_exposed / max) * 100}%`, background: EXPOSED }} />
      </div>
      <div style={{ fontSize: 12.5, fontWeight: 600, color: EXPOSED, textAlign: "right" }}>{s.pct_exposed}%</div>
    </div>
  );
  return (
    <Block title="Where"
      blurb="States ranked by the exposure of their job mix: each state's occupations weighted by how many people it employs in them. A single occupation's exposure is identical in every state — what differs is which jobs the state has — so this is a statement about the mix, not about the work.">
      <div style={{ border: "1px solid var(--border)", borderRadius: 10, overflow: "hidden" }}>
        {states.top.map(Row)}
        <div style={{ padding: "7px 14px", background: "var(--brand-light)", fontSize: 10.5,
          fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--text-muted)",
          borderTop: "1px solid var(--border)" }}>
          Lowest of {states.total}
        </div>
        {states.bottom.map(Row)}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
        Bars are scaled to the highest state, not to 100%, so the spread is visible.
      </div>
    </Block>
  );
}

/* ── Chrome ────────────────────────────────────────────────────────────── */

function Block({ title, blurb, children }: { title: string; blurb: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 38 }}>
      <h2 style={{ fontSize: 17, fontWeight: 700, color: "var(--text-primary)", marginBottom: 5 }}>{title}</h2>
      <p style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.55, marginBottom: 14, maxWidth: 700 }}>{blurb}</p>
      {children}
    </section>
  );
}

function Footnote() {
  return (
    <p style={{ fontSize: 11.5, color: "var(--text-muted)", lineHeight: 1.6, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
      Exposure is the share of work time AI would do or assist if every worker used it on the
      tasks people have already brought to AI often enough to pass a minimum threshold, at the
      level of automation they brought them at. It is not the share of work AI is doing today,
      which makes it an upper bound, and it is not a forecast of job loss. Figures are U.S.-only
      and from past snapshots.
    </p>
  );
}
