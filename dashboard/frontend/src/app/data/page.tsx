"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  ConfigResponse, ExposureResponse, UsageResponse, TrendResponse, MetricKey,
} from "@/lib/types";
import {
  fetchConfig, fetchExposure, fetchExposureChildren, fetchTrend, fetchUsage,
  type ExposureKind,
} from "@/lib/api";
import { METRIC_COLORS, INTENSITY_COLOR, CATEGORY_PALETTE } from "@/lib/theme";
import { fmtPct, fmtCount, fmtWages, fmtIntensity } from "@/lib/format";
import MetricBars, { type BarDatum } from "@/components/data/MetricBars";
import TrendChart, { type TrendSeries } from "@/components/data/TrendChart";

type Tab = "occ" | "wa" | "usage";

// next-level-down per hierarchy (null = leaf, no drill)
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

function metricFmt(k: MetricKey) {
  return k === "pct_tasks_affected" ? fmtPct : k === "workers_affected" ? fmtCount : fmtWages;
}
function metricVal(r: { pct_tasks_affected: number; workers_affected: number; wages_affected: number }, k: MetricKey) {
  return r[k];
}
function metricRank(r: { rank_pct: number; rank_workers: number; rank_wages: number }, k: MetricKey) {
  return k === "pct_tasks_affected" ? r.rank_pct : k === "workers_affected" ? r.rank_workers : r.rank_wages;
}

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
  const [sortMetric, setSortMetric] = useState<MetricKey>("pct_tasks_affected");
  const [topN, setTopN] = useState(20);
  const [search, setSearch] = useState("");

  // hierarchy drill path: each entry is the category we drilled into, at `level`
  const [path, setPath] = useState<{ level: string; category: string }[]>([]);

  const [expData, setExpData] = useState<ExposureResponse | null>(null);
  const [usageData, setUsageData] = useState<UsageResponse | null>(null);
  const [trendData, setTrendData] = useState<TrendResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchConfig().then((c) => { setConfig(c); setConfigKey(c.default_config); }).catch((e) => setErr(e.message));
  }, []);

  const baseLevel = tab === "occ" ? occLevel : tab === "wa" ? waLevel : usageLevel;
  const childMap = tab === "occ" ? OCC_CHILD : tab === "wa" ? WA_CHILD : USAGE_CHILD;
  // current display level: base, or the child of the deepest drilled level
  const curLevel = path.length ? (childMap[path[path.length - 1].level] ?? baseLevel) : baseLevel;
  const kind: ExposureKind = tab === "wa" ? "wa" : "occ";
  const canDrillCur = childMap[curLevel] != null;

  // reset drill + trend when switching tab or base level
  useEffect(() => { setPath([]); }, [tab, occLevel, waLevel, usageLevel, configKey, geo]);
  useEffect(() => { if (tab === "usage") setTrend(false); }, [tab]);

  // fetch displayed data
  useEffect(() => {
    if (!config) return;
    let cancel = false;
    setLoading(true);
    const parent = path.length ? path[path.length - 1] : null;

    const run = async () => {
      try {
        if (tab === "usage") {
          const d = parent
            ? await fetchUsage(curLevel, parent.level, parent.category)
            : await fetchUsage(curLevel);
          if (!cancel) { setUsageData(d); setExpData(null); setTrendData(null); }
        } else if (trend && !parent) {
          const d = await fetchTrend(configKey, curLevel, geo, kind);
          if (!cancel) { setTrendData(d); setExpData(null); setUsageData(null); }
        } else {
          const d = parent
            ? await fetchExposureChildren(configKey, parent.level, geo, kind, parent.category)
            : await fetchExposure(configKey, curLevel, geo, kind);
          if (!cancel) { setExpData(d); setUsageData(null); setTrendData(null); }
        }
      } catch (e) {
        if (!cancel) setErr((e as Error).message);
      } finally {
        if (!cancel) setLoading(false);
      }
    };
    run();
    return () => { cancel = true; };
  }, [config, tab, configKey, curLevel, geo, kind, trend, path]);

  // ── derive displayed rows ────────────────────────────────────────────────
  const displayRows = useMemo(() => {
    if (!expData) return [];
    let rows = expData.rows.slice();
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((r) => r.category.toLowerCase().includes(q));
    } else {
      rows = rows.sort((a, b) => metricVal(b, sortMetric) - metricVal(a, sortMetric)).slice(0, topN);
    }
    return rows;
  }, [expData, search, sortMetric, topN]);

  const usageRows = useMemo(() => {
    if (!usageData) return [];
    let rows = usageData.rows.slice();
    if (search.trim()) {
      const q = search.toLowerCase();
      rows = rows.filter((r) => r.category.toLowerCase().includes(q));
    } else {
      rows = rows.slice(0, topN);
    }
    return rows;
  }, [usageData, search, topN]);

  const totals = expData
    ? { workers_affected: expData.total_workers, wages_affected: expData.total_wages,
        pct_tasks_affected: expData.total_categories } // pct share denom n/a; use category count
    : { workers_affected: 0, wages_affected: 0, pct_tasks_affected: 0 };

  // trend series (top categories)
  const trendSeries: TrendSeries[] = useMemo(() => {
    if (!trendData) return [];
    const cats = trendData.top_categories.slice(0, 8);
    const byCat: Record<string, { date: string; value: number }[]> = {};
    for (const dp of trendData.data_points) {
      for (const r of dp.rows) {
        if (!cats.includes(r.category)) continue;
        (byCat[r.category] ??= []).push({ date: dp.date, value: metricVal(r, sortMetric) });
      }
    }
    return cats
      .filter((c) => byCat[c]?.length)
      .map((c, i) => ({ category: c, points: byCat[c], color: CATEGORY_PALETTE[i % CATEGORY_PALETTE.length] }));
  }, [trendData, sortMetric]);

  if (err) return <Centered><span style={{ color: "#b91c1c" }}>Backend error: {err}</span></Centered>;
  if (!config) return <Centered><Spinner /></Centered>;

  const levelOptions = tab === "occ" ? config.occ_levels : tab === "wa" ? config.wa_levels : config.usage_levels;
  const setBaseLevel = tab === "occ" ? setOccLevel : tab === "wa" ? setWaLevel : setUsageLevel;
  const drill = (category: string) => { if (canDrillCur) setPath((p) => [...p, { level: curLevel, category }]); };

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "22px 24px 60px" }}>
      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--border)", marginBottom: 18 }}>
        {TABS.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)} style={tabStyle(tab === t.key)}>{t.label}</button>
        ))}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, alignItems: "flex-end", marginBottom: 22 }}>
        {tab !== "usage" && (
          <Field label="Data configuration">
            <Select value={configKey} onChange={setConfigKey}
              options={config.configs.map((c) => ({ value: c.key, label: c.label }))} />
          </Field>
        )}
        <Field label={tab === "usage" ? "Hierarchy level" : "Level"}>
          <Select value={baseLevel} onChange={setBaseLevel}
            options={Object.entries(levelOptions).map(([label, value]) => ({ value, label }))} />
        </Field>
        {tab !== "usage" && (
          <Field label="Geography">
            <Select value={geo} onChange={setGeo}
              options={Object.entries(config.geo_options).map(([value, label]) => ({ value, label }))} />
          </Field>
        )}
        <Field label="Sort / metric">
          <Select value={sortMetric} onChange={(v) => setSortMetric(v as MetricKey)}
            options={METRICS.map((m) => ({ value: m.key, label: m.title }))} />
        </Field>
        <Field label="Show">
          <Select value={String(topN)} onChange={(v) => setTopN(Number(v))}
            options={[10, 20, 30].map((n) => ({ value: String(n), label: `Top ${n}` }))} />
        </Field>
        <Field label="Search">
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="filter…"
            style={{ ...inputStyle, width: 150 }} />
        </Field>
        {tab !== "usage" && (
          <label style={toggleLabel}>
            <input type="checkbox" checked={trend} onChange={(e) => setTrend(e.target.checked)} disabled={path.length > 0} />
            Trend over time
          </label>
        )}
        {tab !== "usage" && trend && (
          <label style={toggleLabel}>
            <input type="checkbox" checked={ols} onChange={(e) => setOls(e.target.checked)} />
            2-yr projection
          </label>
        )}
      </div>

      {/* Breadcrumb (drill path) */}
      {path.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 14, fontSize: 12 }}>
          <button onClick={() => setPath([])} style={crumbStyle}>All</button>
          {path.map((p, i) => (
            <span key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ color: "var(--text-muted)" }}>›</span>
              <button onClick={() => setPath(path.slice(0, i + 1))} style={crumbStyle}>{p.category}</button>
            </span>
          ))}
          <span style={{ color: "var(--text-muted)" }}>›</span>
          <span style={{ fontWeight: 600 }}>{labelForLevel(levelOptions, curLevel, tab)}</span>
        </div>
      )}

      {loading && <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 10 }}>Loading…</div>}

      {/* Content */}
      {tab === "usage" ? (
        <UsagePanel rows={usageRows} canDrill={canDrillCur} onDrill={drill} />
      ) : trend && path.length === 0 ? (
        trendSeries.length ? (
          <TrendChart series={trendSeries} yLabel={METRICS.find((m) => m.key === sortMetric)!.title}
            format={metricFmt(sortMetric)} ols={ols} />
        ) : <Empty />
      ) : (
        <div style={{ display: "flex", gap: 26 }}>
          {METRICS.map((m) => {
            const bars: BarDatum[] = displayRows.map((r) => ({
              category: r.category,
              value: metricVal(r, m.key),
              rank: metricRank(r, m.key),
              total: totals[m.key],
              display: metricFmt(m.key)(metricVal(r, m.key)),
            }));
            return (
              <MetricBars key={m.key} title={m.title} rows={bars} color={METRIC_COLORS[m.key]}
                totalCategories={expData?.total_categories ?? bars.length}
                canDrill={canDrillCur && !search.trim()} onDrill={drill} />
            );
          })}
        </div>
      )}

      {/* Footnote */}
      <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 26, lineHeight: 1.6, maxWidth: 720 }}>
        Exposure is task-level overlap with current AI capability (an upper bound), not a job-loss forecast.
        All views use frequency weighting with the auto-augmentation multiplier on.
        {tab === "usage" && " Actual usage shows debiased AI-usage intensity (÷ frequency×employment), × the median reference — AEI Conv+API on eco-2025, no Microsoft."}
      </p>
    </div>
  );
}

// ── Usage panel ──────────────────────────────────────────────────────────────
function UsagePanel({ rows, canDrill, onDrill }: {
  rows: UsageResponse["rows"]; canDrill: boolean; onDrill: (c: string) => void;
}) {
  if (!rows.length) return <Empty />;
  const max = Math.max(1, ...rows.map((r) => r.intensity));
  return (
    <div style={{ maxWidth: 720 }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.04em" }}>
        AI Usage Intensity (× median reference)
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rows.map((r) => (
          <div key={r.category} onClick={() => canDrill && onDrill(r.category)} style={{ cursor: canDrill ? "pointer" : "default" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, marginBottom: 2, color: "var(--text-secondary)" }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "62%" }} title={r.category}>
                {canDrill && <span style={{ opacity: 0.5, marginRight: 4 }}>▸</span>}{r.category}
              </span>
              <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                {fmtIntensity(r.intensity)}<span style={{ color: "var(--text-muted)", fontWeight: 400, marginLeft: 6 }}>{r.raw_pct.toFixed(3)}% raw</span>
              </span>
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

// ── small UI helpers ─────────────────────────────────────────────────────────
function labelForLevel(opts: Record<string, string>, level: string, tab: Tab): string {
  const found = Object.entries(opts).find(([, v]) => v === level);
  if (found) return found[0];
  return level.charAt(0).toUpperCase() + level.slice(1);
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 10.5, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
      {children}
    </div>
  );
}
function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}
function Centered({ children }: { children: React.ReactNode }) {
  return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 120px)" }}>{children}</div>;
}
function Spinner() {
  return <>
    <div style={{ width: 36, height: 36, borderRadius: "50%", border: "3px solid var(--brand)", borderTopColor: "transparent", animation: "spin 0.7s linear infinite" }} />
    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
  </>;
}
function Empty() {
  return <div style={{ fontSize: 13, color: "var(--text-muted)", padding: "30px 0" }}>No data for this selection.</div>;
}

const inputStyle: React.CSSProperties = {
  padding: "6px 10px", fontSize: 13, borderRadius: 6,
  border: "1px solid var(--border)", background: "var(--bg-surface)", color: "var(--text-primary)",
};
const toggleLabel: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "var(--text-secondary)", cursor: "pointer", paddingBottom: 6,
};
const crumbStyle: React.CSSProperties = {
  background: "none", border: "none", color: "var(--brand)", cursor: "pointer", fontSize: 12, padding: 0, textDecoration: "underline",
};
function tabStyle(active: boolean): React.CSSProperties {
  return {
    padding: "9px 16px", fontSize: 13.5, fontWeight: active ? 600 : 400,
    color: active ? "var(--brand)" : "var(--text-secondary)",
    background: "none", border: "none", cursor: "pointer",
    borderBottom: active ? "2px solid var(--brand)" : "2px solid transparent", marginBottom: -1,
  };
}
