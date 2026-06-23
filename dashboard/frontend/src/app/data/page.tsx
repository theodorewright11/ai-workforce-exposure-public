"use client";

import { useEffect, useMemo, useState } from "react";
import type { ConfigResponse, ExposureResponse, UsageResponse, TrendResponse, MetricKey } from "@/lib/types";
import {
  fetchConfig, fetchExposure, fetchExposureChildren, fetchExposureToOcc,
  fetchTrend, fetchUsage, fetchWaTasks, type ExposureKind, type WaTask,
} from "@/lib/api";
import { METRIC_COLORS, INTENSITY_COLOR, CATEGORY_PALETTE } from "@/lib/theme";
import { fmtPct, fmtCount, fmtWages, fmtIntensity } from "@/lib/format";
import MetricBars, { type BarDatum } from "@/components/data/MetricBars";
import TrendChart, { type TrendSeries } from "@/components/data/TrendChart";

type Tab = "occ" | "wa" | "usage";
type RankBy = "current" | "abs" | "pct";

const OCC_CHILD: Record<string, string | null> = { major: "minor", minor: "broad", broad: "occupation", occupation: null };
const WA_CHILD: Record<string, string | null> = { gwa: "iwa", iwa: "dwa", dwa: null };
const USAGE_CHILD: Record<string, string | null> = { major: "minor", minor: "broad", broad: "occupation", occupation: "task", task: null };

const METRICS: { key: MetricKey; title: string }[] = [
  { key: "pct_tasks_affected", title: "% Tasks Exposed" },
  { key: "workers_affected", title: "Workers Exposed" },
  { key: "wages_affected", title: "Wages Exposed" },
];
const TABS: { key: Tab; label: string }[] = [
  { key: "occ", label: "Occupation Exposure" },
  { key: "wa", label: "Work-Activity Exposure" },
  { key: "usage", label: "Actual AI Usage" },
];

function metricFmt(k: MetricKey) { return k === "pct_tasks_affected" ? fmtPct : k === "workers_affected" ? fmtCount : fmtWages; }
function mVal(r: { pct_tasks_affected: number; workers_affected: number; wages_affected: number }, k: MetricKey) { return r[k]; }
function mRank(r: { rank_pct: number; rank_workers: number; rank_wages: number }, k: MetricKey) {
  return k === "pct_tasks_affected" ? r.rank_pct : k === "workers_affected" ? r.rank_workers : r.rank_wages;
}
function olsFit(pts: { x: number; y: number }[]) {
  const n = pts.length; if (n < 2) return null;
  const sx = pts.reduce((s, p) => s + p.x, 0), sy = pts.reduce((s, p) => s + p.y, 0);
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0), sxy = pts.reduce((s, p) => s + p.x * p.y, 0);
  const d = n * sxx - sx * sx; if (!d) return null;
  const b = (n * sxy - sx * sy) / d; return { b, a: (sy - b * sx) / n };
}
function dnum(d: string) { return new Date(d + "T00:00:00Z").getTime() / 86400000; }

interface PathEntry { level: string; category: string; jumped?: boolean; }

export default function DataPage() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>("occ");
  const [configKey, setConfigKey] = useState("all_confirmed");
  const [occLevel, setOccLevel] = useState("major");
  const [waLevel, setWaLevel] = useState("gwa");
  const [usageLevel, setUsageLevel] = useState("major");
  const [geo, setGeo] = useState("nat");
  const [trend, setTrend] = useState(false);
  const [ols, setOls] = useState(false);
  const [rankBy, setRankBy] = useState<RankBy>("current");
  const [sortMetric, setSortMetric] = useState<MetricKey>("pct_tasks_affected");
  const [topN, setTopN] = useState(20);
  const [showMode, setShowMode] = useState<"topn" | "range">("topn");
  const [range, setRange] = useState<[number, number]>([1, 25]);
  const [search, setSearch] = useState("");
  const [jumpToOcc, setJumpToOcc] = useState(false);

  const sliceShow = <T,>(rows: T[]): T[] => showMode === "range" ? rows.slice(range[0] - 1, range[1]) : rows.slice(0, topN);

  const [path, setPath] = useState<PathEntry[]>([]);
  const [expData, setExpData] = useState<ExposureResponse | null>(null);
  const [usageData, setUsageData] = useState<UsageResponse | null>(null);
  const [trendData, setTrendData] = useState<TrendResponse | null>(null);
  const [waTasks, setWaTasks] = useState<{ name: string; tasks: WaTask[] } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { fetchConfig().then((c) => { setConfig(c); setConfigKey(c.default_config); }).catch((e) => setErr(e.message)); }, []);

  const baseLevel = tab === "occ" ? occLevel : tab === "wa" ? waLevel : usageLevel;
  const childMap = tab === "occ" ? OCC_CHILD : tab === "wa" ? WA_CHILD : USAGE_CHILD;
  const lastEntry = path.length ? path[path.length - 1] : null;
  const curLevel = lastEntry ? (lastEntry.jumped ? "occupation" : (childMap[lastEntry.level] ?? baseLevel)) : baseLevel;
  const kind: ExposureKind = tab === "wa" ? "wa" : "occ";
  const canJump = jumpToOcc && (tab === "occ" || tab === "usage") && curLevel !== "occupation" && curLevel !== "task";
  const canDrillCur = lastEntry?.jumped ? false : (
    (tab === "wa" && curLevel === "dwa") || canJump || (childMap[curLevel] != null)
  );

  useEffect(() => { setPath([]); setWaTasks(null); }, [tab, occLevel, waLevel, usageLevel, configKey, geo]);
  useEffect(() => { if (tab === "usage") setTrend(false); }, [tab]);

  useEffect(() => {
    if (!config) return;
    let cancel = false; setLoading(true); setWaTasks(null);
    const parent = lastEntry;
    const run = async () => {
      try {
        if (tab === "usage") {
          const d = parent ? await fetchUsage(curLevel, parent.level, parent.category) : await fetchUsage(curLevel);
          if (!cancel) { setUsageData(d); setExpData(null); setTrendData(null); }
        } else if (trend && !parent) {
          const d = await fetchTrend(configKey, curLevel, geo, kind);
          if (!cancel) { setTrendData(d); setExpData(null); setUsageData(null); }
        } else if (parent?.jumped) {
          const d = await fetchExposureToOcc(configKey, parent.level, geo, parent.category);
          if (!cancel) { setExpData(d); setUsageData(null); setTrendData(null); }
        } else {
          const d = parent
            ? await fetchExposureChildren(configKey, parent.level, geo, kind, parent.category)
            : await fetchExposure(configKey, curLevel, geo, kind);
          if (!cancel) { setExpData(d); setUsageData(null); setTrendData(null); }
        }
      } catch (e) { if (!cancel) setErr((e as Error).message); }
      finally { if (!cancel) setLoading(false); }
    };
    run();
    return () => { cancel = true; };
  }, [config, tab, configKey, curLevel, geo, kind, trend, path]); // eslint-disable-line

  const displayRows = useMemo(() => {
    if (!expData) return [];
    let rows = expData.rows.slice();
    if (search.trim()) { const q = search.toLowerCase(); rows = rows.filter((r) => r.category.toLowerCase().includes(q)); }
    else rows = sliceShow(rows.sort((a, b) => mVal(b, sortMetric) - mVal(a, sortMetric)));
    return rows;
  }, [expData, search, sortMetric, topN, showMode, range]); // eslint-disable-line

  const usageRows = useMemo(() => {
    if (!usageData) return [];
    let rows = usageData.rows.slice();
    if (search.trim()) { const q = search.toLowerCase(); rows = rows.filter((r) => r.category.toLowerCase().includes(q)); }
    else rows = sliceShow(rows);
    return rows;
  }, [usageData, search, topN, showMode, range]); // eslint-disable-line

  // ── trend: per-category start/end/Δ/proj, ranked by rankBy ──────────────────
  const trendTable = useMemo(() => {
    if (!trendData) return [];
    const cats = new Map<string, { date: string; value: number }[]>();
    for (const dp of trendData.data_points) for (const r of dp.rows) {
      (cats.get(r.category) ?? cats.set(r.category, []).get(r.category)!).push({ date: dp.date, value: mVal(r, sortMetric) });
    }
    const rows = Array.from(cats.entries()).map(([category, pts]) => {
      pts.sort((a, b) => dnum(a.date) - dnum(b.date));
      const start = pts[0]?.value ?? 0, end = pts[pts.length - 1]?.value ?? 0;
      const abs = end - start, pctChg = start ? (abs / start) * 100 : 0;
      const d0 = dnum(pts[0].date);
      const f = olsFit(pts.map((p) => ({ x: dnum(p.date) - d0, y: p.value })));
      const proj = f ? f.a + f.b * (dnum(pts[pts.length - 1].date) - d0 + 730) : null;
      return { category, pts, start, end, abs, pctChg, proj };
    });
    const key = rankBy === "abs" ? (r: typeof rows[0]) => r.abs : rankBy === "pct" ? (r: typeof rows[0]) => r.pctChg : (r: typeof rows[0]) => r.end;
    rows.sort((a, b) => key(b) - key(a));
    return sliceShow(rows);
  }, [trendData, sortMetric, rankBy, topN, showMode, range]); // eslint-disable-line

  const trendSeries: TrendSeries[] = useMemo(
    () => trendTable.map((r, i) => ({ category: r.category, points: r.pts, color: CATEGORY_PALETTE[i % CATEGORY_PALETTE.length] })),
    [trendTable]
  );
  const [activeLine, setActiveLine] = useState<string | null>(null);

  if (err) return <Centered><span style={{ color: "#b91c1c" }}>Backend error: {err}</span></Centered>;
  if (!config) return <Centered><Spinner /></Centered>;

  const levelOptions = tab === "occ" ? config.occ_levels : tab === "wa" ? config.wa_levels : config.usage_levels;
  const setBaseLevel = tab === "occ" ? setOccLevel : tab === "wa" ? setWaLevel : setUsageLevel;
  const showMax = expData?.total_categories ?? usageData?.rows.length ?? 0;
  const total = expData
    ? { workers_affected: expData.total_workers, wages_affected: expData.total_wages, pct_tasks_affected: 0 }
    : { workers_affected: 0, wages_affected: 0, pct_tasks_affected: 0 };

  const drill = (category: string) => {
    if (tab === "wa" && curLevel === "dwa") {
      setLoading(true);
      fetchWaTasks("dwa", category).then((d) => setWaTasks({ name: category, tasks: d.tasks })).finally(() => setLoading(false));
      return;
    }
    if (canJump) { setPath((p) => [...p, { level: curLevel, category, jumped: true }]); return; }
    if (childMap[curLevel]) setPath((p) => [...p, { level: curLevel, category }]);
  };
  const mFmt = metricFmt(sortMetric);

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "22px 24px 60px" }}>
      <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--border)", marginBottom: 18 }}>
        {TABS.map((t) => <button key={t.key} onClick={() => setTab(t.key)} style={tabStyle(tab === t.key)}>{t.label}</button>)}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "flex-end", marginBottom: 18 }}>
        {tab !== "usage" && <Field label="Data configuration"><Select value={configKey} onChange={setConfigKey} options={config.configs.map((c) => ({ value: c.key, label: c.label }))} /></Field>}
        <Field label={tab === "usage" ? "Hierarchy level" : "Level"}><Select value={baseLevel} onChange={setBaseLevel} options={Object.entries(levelOptions).map(([label, value]) => ({ value, label }))} /></Field>
        {tab !== "usage" && <Field label="Geography"><Select value={geo} onChange={setGeo} options={Object.entries(config.geo_options).map(([value, label]) => ({ value, label }))} /></Field>}
        {tab !== "usage" && <Field label="Sort / metric"><Select value={sortMetric} onChange={(v) => setSortMetric(v as MetricKey)} options={METRICS.map((m) => ({ value: m.key, label: m.title }))} /></Field>}
        <Field label={`Show${showMax ? ` (of ${showMax})` : ""}`}>
          <ShowControl topN={topN} setTopN={(n) => { setTopN(n); setShowMode("topn"); }} mode={showMode} setMode={setShowMode} range={range} setRange={setRange} max={showMax} />
        </Field>
        <Field label="Search"><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="filter…" style={{ ...inputStyle, width: 140 }} /></Field>
        {(tab === "occ" || tab === "usage") && !trend && <Toggle on={jumpToOcc} onClick={() => setJumpToOcc((v) => !v)}>Drill-down → occupations</Toggle>}
        {tab !== "usage" && <Toggle on={trend} onClick={() => setTrend((v) => !v)} disabled={path.length > 0}>Trend over time</Toggle>}
        {tab !== "usage" && trend && <Toggle on={ols} onClick={() => setOls((v) => !v)}>2-yr projection</Toggle>}
        {tab !== "usage" && trend && (
          <Field label="Rank by">
            <div style={{ display: "flex", gap: 4 }}>
              {([["current", "Current"], ["abs", "Abs change"], ["pct", "% change"]] as [RankBy, string][]).map(([k, l]) =>
                <button key={k} onClick={() => setRankBy(k)} style={segBtn(rankBy === k)}>{l}</button>)}
            </div>
          </Field>
        )}
      </div>

      {/* Breadcrumb */}
      {(path.length > 0 || waTasks) && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 14, fontSize: 12, flexWrap: "wrap" }}>
          <button onClick={() => { setPath([]); setWaTasks(null); }} style={crumbStyle}>All</button>
          {path.map((p, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ color: "var(--text-muted)" }}>›</span>
              <button onClick={() => { setPath(path.slice(0, i + 1)); setWaTasks(null); }} style={crumbStyle}>{p.category}</button>
            </span>
          ))}
          {waTasks && <><span style={{ color: "var(--text-muted)" }}>›</span><span style={{ fontWeight: 600 }}>{waTasks.name} — tasks</span></>}
        </div>
      )}

      {loading && <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 10 }}>Loading…</div>}

      {/* Content */}
      {waTasks ? (
        <WaTaskList tasks={waTasks.tasks} />
      ) : tab === "usage" ? (
        <UsagePanel rows={usageRows} canDrill={canDrillCur} onDrill={drill} />
      ) : trend && path.length === 0 ? (
        trendSeries.length ? (
          <div>
            <TrendChart series={trendSeries} yLabel={METRICS.find((m) => m.key === sortMetric)!.title} format={mFmt} ols={ols} activeCat={activeLine} onHover={setActiveLine} />
            <TrendTable rows={trendTable} colors={CATEGORY_PALETTE} fmt={mFmt} ols={ols} active={activeLine} onHover={setActiveLine} rankBy={rankBy} />
          </div>
        ) : <Empty />
      ) : (
        <div style={{ display: "flex", gap: 26 }}>
          {METRICS.map((m) => {
            const bars: BarDatum[] = displayRows.map((r) => ({
              category: r.category, value: mVal(r, m.key), rank: mRank(r, m.key),
              total: total[m.key], display: metricFmt(m.key)(mVal(r, m.key)),
            }));
            return (
              <MetricBars key={m.key} title={m.title} rows={bars} color={METRIC_COLORS[m.key]}
                totalCategories={expData?.total_categories ?? bars.length}
                canDrill={canDrillCur && !search.trim()} onDrill={drill}
                showShare={m.key !== "pct_tasks_affected"} />
            );
          })}
        </div>
      )}

      <p style={{ fontSize: 11.5, color: "var(--text-muted)", marginTop: 26, lineHeight: 1.65, maxWidth: 760 }}>
        {tab === "usage" ? (
          <>Relative ranking of where AI is currently being used. Read each bar as <strong>X× the median</strong> usage relative to task need at this level. We correct as best we can for AI user-base bias, and divide by task frequency and employment so more-common work doesn&rsquo;t simply show up more. Absolute reach can&rsquo;t be inferred — only the relative ordering.</>
        ) : (
          <>Ranking of what current AI capability exposes in the workforce, as informed by actual AI usage — where we most expect to see transformation going forward. This is agnostic to what that change looks like: high exposure does not mean these jobs go away.</>
        )}
      </p>
    </div>
  );
}

/* ── WA task list (DWA drill) ───────────────────────────────────────────────── */
const BUCKET_COLOR: Record<string, string> = { high: "#a8824a", mid: "#c9b27e", low: "#8ea9bf", none: "#c9ccce" };
function WaTaskList({ tasks }: { tasks: WaTask[] }) {
  if (!tasks.length) return <Empty />;
  return (
    <div style={{ maxWidth: 900 }}>
      <div style={{ display: "flex", padding: "6px", fontSize: 10.5, color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase" }}>
        <span style={{ flex: 1 }}>Task</span><span style={{ width: 90, textAlign: "right" }}>Centrality</span>
        <span style={{ width: 80, textAlign: "right" }}>Usage</span>
        <span style={{ width: 110, textAlign: "right", lineHeight: 1.15 }}>Automation<br />level</span>
      </div>
      {tasks.map((t) => <WaTaskItem key={t.task_normalized} t={t} n={tasks.length} />)}
    </div>
  );
}
function WaTaskItem({ t, n }: { t: WaTask; n: number }) {
  const [open, setOpen] = useState(false);
  const color = BUCKET_COLOR[t.color_bucket] ?? "#c9ccce";
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 6px", cursor: "pointer" }} onClick={() => setOpen((v) => !v)}>
        <span style={{ flex: 1, fontSize: 12.5 }}><span style={{ opacity: 0.4, marginRight: 6 }}>{open ? "▾" : "▸"}</span>{t.task}</span>
        <span style={{ width: 90, textAlign: "right", fontSize: 12 }}>#{t.centrality_rank}<span style={{ color: "var(--text-muted)" }}>/{n}</span></span>
        <span style={{ width: 80, textAlign: "right", fontSize: 12 }}>{t.usage_mult != null ? `${t.usage_mult}×` : "—"}</span>
        <span style={{ width: 110, textAlign: "right", fontSize: 15, fontWeight: 700, color }}>{t.auto != null ? `${t.auto}/5` : "—"}</span>
      </div>
      {open && (
        <div style={{ padding: "10px 14px 14px 26px", background: "var(--bg-sidebar)", fontSize: 12 }}>
          {([["General Work Activity", t.gwa], ["Intermediate Work Activity", t.iwa], ["Detailed Work Activity", t.dwa]] as [string, WaTask["gwa"]][]).map(([lvl, w]) =>
            w ? (
              <div key={lvl} style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase" }}>{lvl}</div>
                <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>{w.name}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 11 }}>Automation {w.auto != null ? `${w.auto}/5` : "—"} (avg over AI-exposed tasks) · Ranks #{w.rank_pct ?? "—"} of {w.total ?? "—"} by tasks exposed</div>
              </div>
            ) : null)}
          {t.top_mcps.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 5 }}>AI tools (MCP) that target this task</div>
              {t.top_mcps.map((m, i) => (
                <div key={i} style={{ marginBottom: 5 }}>
                  <a href={m.url ?? "#"} target="_blank" rel="noreferrer" style={{ color: "var(--brand)", fontWeight: 600, fontSize: 12 }}>{m.title}</a>
                  {m.rating != null && <span style={{ color: "var(--text-muted)", marginLeft: 6 }}>{m.rating}/5</span>}
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

/* ── trend table ────────────────────────────────────────────────────────────── */
interface TRow { category: string; start: number; end: number; abs: number; pctChg: number; proj: number | null; }
function TrendTable({ rows, colors, fmt, ols, active, onHover, rankBy }: {
  rows: TRow[]; colors: string[]; fmt: (v: number) => string; ols: boolean; active: string | null; onHover: (c: string | null) => void; rankBy: RankBy;
}) {
  const bold = (col: "end" | "abs" | "pct"): React.CSSProperties =>
    (rankBy === "current" && col === "end") || (rankBy === "abs" && col === "abs") || (rankBy === "pct" && col === "pct")
      ? { fontWeight: 700, color: "var(--text-primary)" } : {};
  return (
    <div style={{ marginTop: 18, overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ color: "var(--text-muted)", textAlign: "right" }}>
            <th style={{ textAlign: "left", padding: "6px 8px" }}>Category</th>
            <th style={thR}>Start</th><th style={{ ...thR, ...bold("end") }}>End</th>
            <th style={{ ...thR, ...bold("abs") }}>Abs Δ</th><th style={{ ...thR, ...bold("pct") }}>% Δ</th>
            {ols && <th style={thR}>2-yr proj.</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.category} onMouseEnter={() => onHover(r.category)} onMouseLeave={() => onHover(null)}
              style={{ borderTop: "1px solid var(--border)", background: active === r.category ? "var(--brand-light)" : "transparent", cursor: "default" }}>
              <td style={{ padding: "6px 8px", display: "flex", alignItems: "center", gap: 7 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: colors[i % colors.length], flexShrink: 0 }} />{r.category}
              </td>
              <td style={tdR}>{fmt(r.start)}</td><td style={{ ...tdR, ...bold("end") }}>{fmt(r.end)}</td>
              <td style={{ ...tdR, ...bold("abs"), color: r.abs >= 0 ? "#2f6f4f" : "#a33" }}>{r.abs >= 0 ? "+" : ""}{fmt(r.abs)}</td>
              <td style={{ ...tdR, ...bold("pct"), color: r.pctChg >= 0 ? "#2f6f4f" : "#a33" }}>{r.pctChg >= 0 ? "+" : ""}{r.pctChg.toFixed(0)}%</td>
              {ols && <td style={tdR}>{r.proj != null ? fmt(r.proj) : "—"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── usage panel ────────────────────────────────────────────────────────────── */
function UsagePanel({ rows, canDrill, onDrill }: { rows: UsageResponse["rows"]; canDrill: boolean; onDrill: (c: string) => void }) {
  if (!rows.length) return <Empty />;
  const max = Math.max(1, ...rows.map((r) => r.intensity));
  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.04em" }}>AI Usage Intensity (× median reference)</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rows.map((r) => (
          <div key={r.category} onClick={() => canDrill && onDrill(r.category)} style={{ cursor: canDrill ? "pointer" : "default" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 2, color: "var(--text-secondary)" }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "62%" }} title={r.category}>{canDrill && <span style={{ opacity: 0.5, marginRight: 4 }}>▸</span>}{r.category}</span>
              <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{fmtIntensity(r.intensity)}<span style={{ color: "var(--text-muted)", fontWeight: 400, marginLeft: 22 }}>{r.raw_pct.toFixed(3)}% raw</span></span>
            </div>
            <div style={{ height: 11, background: "var(--bg-sidebar)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${(r.intensity / max) * 100}%`, height: "100%", background: INTENSITY_COLOR, opacity: 0.85 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── controls ───────────────────────────────────────────────────────────────── */
function ShowControl({ topN, setTopN, mode, setMode, range, setRange, max }: {
  topN: number; setTopN: (n: number) => void; mode: "topn" | "range";
  setMode: (m: "topn" | "range") => void; range: [number, number]; setRange: (r: [number, number]) => void; max: number;
}) {
  const presets = [10, 25, 50];
  const hi = Math.max(2, max || 100);
  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
      {presets.map((p) => <button key={p} onClick={() => setTopN(p)} style={segBtn(mode === "topn" && topN === p)}>{p}</button>)}
      <button onClick={() => setMode("range")} style={segBtn(mode === "range")}>Range</button>
      {mode === "range" && <DualRange min={1} max={hi} lo={Math.min(range[0], hi)} hiVal={Math.min(range[1], hi)} onChange={setRange} />}
    </div>
  );
}
function DualRange({ min, max, lo, hiVal, onChange }: { min: number; max: number; lo: number; hiVal: number; onChange: (r: [number, number]) => void }) {
  const pctL = ((lo - min) / (max - min || 1)) * 100;
  const pctH = ((hiVal - min) / (max - min || 1)) * 100;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <div style={{ position: "relative", width: 180, height: 22 }}>
        <div style={{ position: "absolute", top: 10, left: 0, right: 0, height: 3, background: "var(--border)", borderRadius: 2 }} />
        <div style={{ position: "absolute", top: 10, height: 3, background: "var(--brand)", borderRadius: 2, left: `${pctL}%`, right: `${100 - pctH}%` }} />
        <input className="dual-range" type="range" min={min} max={max} value={lo} onChange={(e) => onChange([Math.min(Number(e.target.value), hiVal - 1), hiVal])} />
        <input className="dual-range" type="range" min={min} max={max} value={hiVal} onChange={(e) => onChange([lo, Math.max(Number(e.target.value), lo + 1)])} />
      </div>
      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>ranks {lo}–{hiVal}</span>
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div style={{ display: "flex", flexDirection: "column", gap: 4 }}><span style={{ fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>{children}</div>;
}
function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>{options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select>;
}
function Toggle({ on, onClick, disabled, children }: { on: boolean; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return <button onClick={onClick} disabled={disabled} style={{ ...segBtn(on), padding: "7px 13px", opacity: disabled ? 0.4 : 1, cursor: disabled ? "not-allowed" : "pointer", marginBottom: 1 }}>{children}</button>;
}
function Centered({ children }: { children: React.ReactNode }) { return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 120px)" }}>{children}</div>; }
function Spinner() { return <><div style={{ width: 36, height: 36, borderRadius: "50%", border: "3px solid var(--brand)", borderTopColor: "transparent", animation: "spin 0.7s linear infinite" }} /><style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style></>; }
function Empty() { return <div style={{ fontSize: 13, color: "var(--text-muted)", padding: "30px 0" }}>No data for this selection.</div>; }

const inputStyle: React.CSSProperties = { padding: "6px 10px", fontSize: 13, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg-surface)", color: "var(--text-primary)" };
const crumbStyle: React.CSSProperties = { background: "none", border: "none", color: "var(--brand)", cursor: "pointer", fontSize: 12, padding: 0, textDecoration: "underline" };
const thR: React.CSSProperties = { textAlign: "right", padding: "6px 8px", fontWeight: 600 };
const tdR: React.CSSProperties = { textAlign: "right", padding: "6px 8px", color: "var(--text-secondary)" };
function tabStyle(active: boolean): React.CSSProperties {
  return { padding: "9px 16px", fontSize: 13.5, fontWeight: active ? 600 : 400, color: active ? "var(--brand)" : "var(--text-secondary)", background: "none", border: "none", cursor: "pointer", borderBottom: active ? "2px solid var(--brand)" : "2px solid transparent", marginBottom: -1 };
}
function segBtn(active: boolean): React.CSSProperties {
  return { fontSize: 12, padding: "6px 11px", borderRadius: 6, cursor: "pointer", border: "1px solid var(--border)", background: active ? "var(--brand-light)" : "var(--bg-surface)", color: active ? "var(--brand)" : "var(--text-secondary)", fontWeight: active ? 600 : 400 };
}
