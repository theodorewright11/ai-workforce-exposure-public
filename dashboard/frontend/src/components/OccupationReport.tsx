"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import type {
  ColorBucket,
  ConfigResponse,
  OccupationReport,
  OccReportTask,
  OccReportWaRow,
  OccReportSkaRow,
  OccReportSimilar,
  OccReportHierarchyEntry,
  OccReportSectorChainEntry,
  OccReportTrendPoint,
} from "@/lib/types";
import {
  fetchOccupationReport,
  fetchOccupationReportTitles,
} from "@/lib/api";

type PickerMode = "search" | "browse";
type WaLevel = "gwa" | "iwa" | "dwa";

interface Props {
  config: ConfigResponse;
}

/* ── Color tokens ─────────────────────────────────────────────────────────── */

const BUCKET_BG: Record<ColorBucket, string> = {
  high: "rgba(184, 96, 60, 0.10)",
  mid:  "rgba(214, 165, 96, 0.10)",
  low:  "rgba(110, 138, 156, 0.10)",
  none: "rgba(228, 228, 222, 0.30)",
};
const BUCKET_BORDER: Record<ColorBucket, string> = {
  high: "rgba(184, 96, 60, 0.40)",
  mid:  "rgba(214, 165, 96, 0.40)",
  low:  "rgba(110, 138, 156, 0.40)",
  none: "rgba(155, 155, 155, 0.30)",
};
const BUCKET_DOT: Record<ColorBucket, string> = {
  high: "#c87a5b",
  mid:  "#d6a560",
  low:  "#7a99ab",
  none: "#cfcfc6",
};
const BUCKET_FG: Record<ColorBucket, string> = {
  high: "#a35135",
  mid:  "#a07a3c",
  low:  "#52768a",
  none: "#8a8a82",
};
const BUCKET_LABEL: Record<ColorBucket, string> = {
  high: "More automated usage seen",
  mid:  "More augmentative",
  low:  "Less automated usage seen",
  none: "No data",
};
const BUCKET_TIER_HEADING: Record<ColorBucket, string> = {
  high: "AI does this well",
  mid:  "AI helps with this",
  low:  "Mostly still you",
  none: "No data",
};

const TIER_COLORS: Record<string, { bg: string; fg: string; label: string; short: string }> = {
  high:     { bg: "rgba(184, 96, 60, 0.12)",  fg: "#b8603c", label: "High Exposure",     short: "High" },
  mod_high: { bg: "rgba(214, 165, 96, 0.14)", fg: "#c08a45", label: "Mod-High Exposure", short: "Mod-High" },
  mod_low:  { bg: "rgba(160, 160, 130, 0.14)", fg: "#7a8862", label: "Mod-Low Exposure", short: "Mod-Low" },
  low:      { bg: "rgba(110, 138, 156, 0.14)", fg: "#5e7e92", label: "Low Exposure",     short: "Low" },
};

const SOURCE_META = {
  aei_conv:  { label: "Claude Conv", short: "Claude Conv", color: "#3a5f83" },
  aei_api:   { label: "Claude API",  short: "Claude API",  color: "#6e4d7e" },
  microsoft: { label: "Copilot",     short: "Copilot",     color: "#8a4225" },
  mcp:       { label: "MCP",         short: "MCP",         color: "#4a7c6f" },
} as const;

/* ── Formatters ───────────────────────────────────────────────────────────── */

function fmtNumber(v: number | null | undefined, decimals = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}
function fmtAuto(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(2);
}
function fmtWage(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
}
function fmtPctOfNeed(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(0)}%`;
}
function fmtPct(v: number | null | undefined, dec = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(dec)}%`;
}
function fmtRank(rank: number | null | undefined, total: number): string {
  if (rank === null || rank === undefined) return "—";
  return `#${rank} of ${total}`;
}
function fmtRankShort(rank: number | null | undefined, total: number): string {
  if (rank === null || rank === undefined) return "—";
  return `#${rank}/${total}`;
}

/* ── Job zone / outlook interpretations ───────────────────────────────────── */

const JOB_ZONE_INTERP: Record<number, string> = {
  1: "Little to no preparation",
  2: "Some preparation",
  3: "Medium preparation",
  4: "Considerable preparation",
  5: "Extensive preparation",
};

const OUTLOOK_INTERP: Record<number, string> = {
  0: "Limited outlook, low wages",
  1: "Strong outlook, low wages",
  2: "Limited outlook, high wages",
  3: "Moderate outlook, low–mod wages",
  4: "Good outlook, high wages",
  5: "Strongest outlook, high wages",
};

/* ── Primitives ───────────────────────────────────────────────────────────── */

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", padding: "4px 10px",
      borderRadius: 999, background: "var(--brand-light)", color: "var(--brand)",
      fontSize: 12, fontWeight: 500, border: "1px solid var(--brand-border)",
      whiteSpace: "nowrap",
    }}>{children}</span>
  );
}

function MiniBar({
  value, max = 5, color = "var(--brand)", height = 6, bg = "rgba(0,0,0,0.06)",
}: { value: number; max?: number; color?: string; height?: number; bg?: string }) {
  const pct = Math.max(0, Math.min(1, value / max));
  return (
    <div style={{ background: bg, height, borderRadius: height / 2, overflow: "hidden", width: "100%" }}>
      <div style={{
        width: `${pct * 100}%`, height: "100%", background: color,
        borderRadius: height / 2, transition: "width 0.4s ease",
      }} />
    </div>
  );
}

function TierDot({ bucket, size = 8 }: { bucket: ColorBucket; size?: number }) {
  return (
    <span style={{
      display: "inline-block", width: size, height: size, borderRadius: "50%",
      background: BUCKET_DOT[bucket], flexShrink: 0,
    }} />
  );
}

function Sparkline({
  points, w = 220, h = 48, color = "var(--brand)", showLabels = false, fill = true,
}: {
  points: OccReportTrendPoint[]; w?: number; h?: number; color?: string;
  showLabels?: boolean; fill?: boolean;
}) {
  const valid = points.filter((p): p is OccReportTrendPoint & { pct_tasks_affected: number } =>
    p.pct_tasks_affected !== null && p.pct_tasks_affected !== undefined);
  if (valid.length < 2) return null;
  const vals = valid.map((p) => p.pct_tasks_affected);
  const minV = Math.min(...vals) * 0.8;
  const maxV = Math.max(...vals) * 1.05;
  const range = (maxV - minV) || 1;
  const pad = 6;
  const xs = valid.map((_, i) => pad + (i / (valid.length - 1)) * (w - 2 * pad));
  const ys = vals.map((v) => h - pad - ((v - minV) / range) * (h - 2 * pad));
  const path = xs.map((x, i) => (i === 0 ? `M ${x},${ys[i]}` : `L ${x},${ys[i]}`)).join(" ");
  const area = `${path} L ${xs[xs.length - 1]},${h} L ${xs[0]},${h} Z`;
  return (
    <svg width={w} height={h + (showLabels ? 18 : 0)} viewBox={`0 0 ${w} ${h + (showLabels ? 18 : 0)}`}>
      {fill && <path d={area} fill={color} opacity={0.10} />}
      <path d={path} stroke={color} strokeWidth={2} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      {xs.map((x, i) => (
        <g key={i}>
          <circle cx={x} cy={ys[i]} r={i === xs.length - 1 ? 4 : 2.5} fill={color} />
          {i === xs.length - 1 && <circle cx={x} cy={ys[i]} r={7} fill={color} opacity={0.18} />}
          {showLabels && (
            <text x={x} y={h + 13} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
              {valid[i].date.slice(0, 7)}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

function RiskGauge({
  score, tier, size = 108,
}: { score: number; tier: "high" | "mod_high" | "mod_low" | "low"; size?: number }) {
  const r = size / 2 - 8;
  const cx = size / 2;
  const cy = size / 2;
  const total = 10;
  const startAngle = -210;
  const endAngle = 30;
  const arcAngle = endAngle - startAngle;
  const fillAngle = startAngle + (Math.max(0, Math.min(score, total)) / total) * arcAngle;
  const polar = (a: number): [number, number] => [
    cx + r * Math.cos((a * Math.PI) / 180),
    cy + r * Math.sin((a * Math.PI) / 180),
  ];
  const arc = (a1: number, a2: number, color: string, w = 8) => {
    const [x1, y1] = polar(a1);
    const [x2, y2] = polar(a2);
    const large = a2 - a1 > 180 ? 1 : 0;
    return (
      <path
        d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
        stroke={color} strokeWidth={w} fill="none" strokeLinecap="round"
      />
    );
  };
  const tierColor = TIER_COLORS[tier].fg;
  return (
    <svg width={size} height={size * 0.85}>
      {arc(startAngle, endAngle, "#eeeeea", 8)}
      {arc(startAngle, fillAngle, tierColor, 8)}
      <text x={cx} y={cy + 4} textAnchor="middle" fontSize={size * 0.3} fontWeight={700} fill="var(--text-primary)">
        {score}
      </text>
      <text x={cx} y={cy + size * 0.22} textAnchor="middle" fontSize={11} fill="var(--text-muted)" letterSpacing="0.04em">
        / 10
      </text>
    </svg>
  );
}

interface SourceCellInput {
  aei_conv_max?: number | null;
  aei_api_max?: number | null;
  microsoft?: number | null;
  mcp?: number | null;
}

function SourceMiniBars({ t, showLabels = true }: { t: SourceCellInput; showLabels?: boolean }) {
  const sources = [
    { key: "aei_conv", label: SOURCE_META.aei_conv.short,  v: t.aei_conv_max ?? null, c: SOURCE_META.aei_conv.color },
    { key: "aei_api",  label: SOURCE_META.aei_api.short,   v: t.aei_api_max  ?? null, c: SOURCE_META.aei_api.color },
    { key: "ms",       label: SOURCE_META.microsoft.short, v: t.microsoft    ?? null, c: SOURCE_META.microsoft.color },
    { key: "mcp",      label: SOURCE_META.mcp.short,       v: t.mcp          ?? null, c: SOURCE_META.mcp.color },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
      {sources.map((s) => (
        <div key={s.key} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 4,
          }}>
            {showLabels && (
              <p style={{
                fontSize: 9, color: "var(--text-muted)",
                letterSpacing: "0.04em", textTransform: "uppercase", margin: 0,
                whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
              }}>{s.label}</p>
            )}
            <span style={{
              fontSize: 11, fontWeight: 600,
              color: s.v != null ? "var(--text-primary)" : "var(--text-muted)",
              fontVariantNumeric: "tabular-nums", marginLeft: "auto",
            }}>
              {s.v != null ? s.v.toFixed(1) : "—"}
            </span>
          </div>
          <MiniBar value={s.v ?? 0} max={5} color={s.c} height={5} />
        </div>
      ))}
    </div>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <span style={{
      display: "inline-block", transform: open ? "rotate(90deg)" : "rotate(0deg)",
      transition: "transform 0.15s", color: "var(--text-muted)", fontSize: 11,
      width: 12, textAlign: "center",
    }}>▶</span>
  );
}

/* Card primitive — section-level collapsible wrapper */
function Card({
  span = 12, title, sub, defaultOpen = true, children, headerExtra,
}: {
  span?: number; title: string; sub?: React.ReactNode;
  defaultOpen?: boolean; children: React.ReactNode;
  headerExtra?: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{
      gridColumn: `span ${span}`,
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: 12, padding: 20,
    }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          width: "100%", display: "flex", alignItems: "flex-start",
          justifyContent: "space-between", gap: 12,
          background: "transparent", border: "none", padding: 0, cursor: "pointer",
          textAlign: "left", marginBottom: open ? 14 : 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: 10, flex: 1, minWidth: 0 }}>
          <span style={{ paddingTop: 4 }}><Chevron open={open} /></span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h3 style={{
              fontSize: 15, fontWeight: 700, color: "var(--text-primary)",
              marginBottom: 4, marginTop: 0,
            }}>{title}</h3>
            {sub && (
              <p style={{
                fontSize: 11, color: "var(--text-muted)",
                lineHeight: 1.5, margin: 0,
              }}>{sub}</p>
            )}
          </div>
        </div>
        {headerExtra && <div onClick={(e) => e.stopPropagation()}>{headerExtra}</div>}
      </button>
      {open && children}
    </div>
  );
}

/* TierGroup — collapsible sub-section grouped by color bucket */
function TierGroup({
  bucket, count, defaultOpen = true, children,
}: {
  bucket: ColorBucket; count: number; defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: 8 }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: 8,
          padding: "8px 12px", background: BUCKET_BG[bucket],
          border: `1px solid ${BUCKET_BORDER[bucket]}`,
          borderRadius: 8, cursor: "pointer", textAlign: "left",
        }}
      >
        <Chevron open={open} />
        <TierDot bucket={bucket} />
        <span style={{
          fontSize: 12, fontWeight: 600, color: BUCKET_FG[bucket],
          letterSpacing: "0.02em",
        }}>
          {BUCKET_TIER_HEADING[bucket]}
        </span>
        <span style={{
          marginLeft: "auto", fontSize: 11, color: "var(--text-muted)",
          fontVariantNumeric: "tabular-nums",
        }}>
          {count} {count === 1 ? "row" : "rows"}
        </span>
      </button>
      {open && <div style={{ marginTop: 6 }}>{children}</div>}
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────────────── */

export default function OccupationReport({ config }: Props) {
  const [titles, setTitles] = useState<string[]>([]);
  const [hierarchy, setHierarchy] = useState<OccReportHierarchyEntry[]>([]);
  const [selectedTitle, setSelectedTitle] = useState<string>("");
  const [pickerMode, setPickerMode] = useState<PickerMode>("search");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
  const [geo, setGeo] = useState<string>("nat");
  const [report, setReport] = useState<OccupationReport | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [waLevel, setWaLevel] = useState<WaLevel>("gwa");
  const [showRiskFlags, setShowRiskFlags] = useState<boolean>(false);

  useEffect(() => {
    fetchOccupationReportTitles()
      .then((d) => {
        setTitles(d.titles);
        setHierarchy(d.hierarchy);
      })
      .catch((e) => setError(e.message));
  }, []);

  const loadReport = useCallback((title: string, g: string) => {
    if (!title) return;
    setLoading(true);
    setError(null);
    fetchOccupationReport(title, g)
      .then((r) => setReport(r))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedTitle) loadReport(selectedTitle, geo);
  }, [selectedTitle, geo, loadReport]);

  const filteredTitles = useMemo(() => {
    if (!searchQuery) return titles.slice(0, 12);
    const q = searchQuery.toLowerCase();
    return titles.filter((t) => t.toLowerCase().includes(q)).slice(0, 12);
  }, [searchQuery, titles]);

  const handleSelect = (t: string) => {
    setSelectedTitle(t);
    setSearchQuery(t);
    setShowSuggestions(false);
  };

  const geoName = config.geo_options[geo] ?? geo;

  return (
    <div style={{ maxWidth: 1240, margin: "0 auto", padding: "32px 24px 80px" }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{
          fontSize: 28, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8,
        }}>
          My Occupation Report
        </h1>
        <p style={{
          fontSize: 14, color: "var(--text-secondary)", marginBottom: 0, lineHeight: 1.5,
        }}>
          Pick your occupation and see, in one place, where AI already does the work, where you still
          have an advantage, what tasks to delegate, and how your role compares to similar ones. All numbers
          are drawn from the dashboard&apos;s <strong>all-confirmed</strong> dataset:
          measured AI usage across Anthropic Claude conversations, AEI API/agentic tool-use, and Microsoft
          Copilot.
        </p>
      </header>

      <Picker
        pickerMode={pickerMode}
        setPickerMode={setPickerMode}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        showSuggestions={showSuggestions}
        setShowSuggestions={setShowSuggestions}
        filteredTitles={filteredTitles}
        onSelect={handleSelect}
        selectedTitle={selectedTitle}
        hierarchy={hierarchy}
        geo={geo}
        setGeo={setGeo}
        geoOptions={config.geo_options}
      />

      {error && (
        <div style={{
          marginTop: 24, padding: "14px 18px", borderRadius: 8,
          background: "rgba(184, 96, 60, 0.10)", color: "#8a4225",
          border: "1px solid rgba(184, 96, 60, 0.40)",
        }}>
          {error}
        </div>
      )}

      {!selectedTitle && !loading && (
        <div style={{
          marginTop: 32, padding: "48px 24px", borderRadius: 12,
          background: "var(--bg-sidebar)", border: "1px solid var(--border)",
          textAlign: "center", color: "var(--text-secondary)",
        }}>
          <p style={{ fontSize: 15, marginBottom: 8 }}>Search for your occupation above to begin.</p>
          <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
            Try: <em>Registered Nurses, Software Developers, Customer Service Representatives, Lawyers, …</em>
          </p>
        </div>
      )}

      {loading && (
        <div style={{ marginTop: 48, textAlign: "center" }}>
          <div style={{
            display: "inline-block", width: 36, height: 36, borderRadius: "50%",
            border: "3px solid var(--brand)", borderTopColor: "transparent",
            animation: "spin 0.7s linear infinite",
          }} />
          <p style={{ marginTop: 12, fontSize: 13, color: "var(--text-muted)" }}>
            Building your report…
          </p>
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
      )}

      {!loading && report && (
        <ReportBody
          report={report}
          waLevel={waLevel}
          setWaLevel={setWaLevel}
          showRiskFlags={showRiskFlags}
          setShowRiskFlags={setShowRiskFlags}
          geoName={geoName}
        />
      )}
    </div>
  );
}

/* ── Picker (Search + Browse tabs) ────────────────────────────────────────── */

interface PickerProps {
  pickerMode: PickerMode;
  setPickerMode: (m: PickerMode) => void;
  searchQuery: string;
  setSearchQuery: (s: string) => void;
  showSuggestions: boolean;
  setShowSuggestions: (b: boolean) => void;
  filteredTitles: string[];
  onSelect: (t: string) => void;
  selectedTitle: string;
  hierarchy: OccReportHierarchyEntry[];
  geo: string;
  setGeo: (g: string) => void;
  geoOptions: Record<string, string>;
}

function Picker(props: PickerProps) {
  const { pickerMode, setPickerMode, geo, setGeo, geoOptions } = props;
  return (
    <div style={{
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: 10, padding: 14,
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: 12, gap: 12,
      }}>
        <div style={{ display: "flex", gap: 4 }}>
          <PickerTab label="Search"
                     active={pickerMode === "search"}
                     onClick={() => setPickerMode("search")} />
          <PickerTab label="Browse by category"
                     active={pickerMode === "browse"}
                     onClick={() => setPickerMode("browse")} />
        </div>
        <select
          value={geo}
          onChange={(e) => setGeo(e.target.value)}
          style={{
            padding: "8px 14px", border: "1px solid var(--border)",
            borderRadius: 8, fontSize: 13, cursor: "pointer",
            background: "var(--bg-surface)", color: "var(--text-primary)",
            minWidth: 180,
          }}
        >
          {Object.entries(geoOptions).map(([code, name]) => (
            <option key={code} value={code}>{name}</option>
          ))}
        </select>
      </div>

      {pickerMode === "search" ? <SearchPanel {...props} /> : <BrowsePanel {...props} />}
    </div>
  );
}

function PickerTab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "7px 14px", borderRadius: 6,
        fontSize: 13, fontWeight: active ? 600 : 500,
        background: active ? "var(--brand-light)" : "transparent",
        color: active ? "var(--brand)" : "var(--text-secondary)",
        border: "1px solid",
        borderColor: active ? "var(--brand-light)" : "transparent",
        cursor: "pointer",
        transition: "all 0.13s",
      }}
    >
      {label}
    </button>
  );
}

function SearchPanel({
  searchQuery, setSearchQuery, showSuggestions, setShowSuggestions,
  filteredTitles, onSelect,
}: PickerProps) {
  return (
    <div style={{ position: "relative" }}>
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => { setSearchQuery(e.target.value); setShowSuggestions(true); }}
        onFocus={() => setShowSuggestions(true)}
        onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
        placeholder="Search for your occupation (e.g. Registered Nurses)…"
        style={{
          width: "100%", padding: "10px 14px",
          border: "1px solid var(--border)", borderRadius: 8,
          fontSize: 14, outline: "none",
          transition: "border-color 0.15s",
        }}
      />
      {showSuggestions && filteredTitles.length > 0 && (
        <div style={{
          position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 20,
          maxHeight: 320, overflowY: "auto",
          background: "var(--bg-surface)", border: "1px solid var(--border)",
          borderRadius: 8, boxShadow: "0 6px 20px rgba(0,0,0,0.10)",
        }}>
          {filteredTitles.map((t) => (
            <button
              key={t}
              onMouseDown={() => onSelect(t)}
              style={{
                display: "block", width: "100%", textAlign: "left",
                padding: "9px 14px", fontSize: 13,
                background: "transparent", border: "none", cursor: "pointer",
                color: "var(--text-primary)",
                borderBottom: "1px solid var(--border-light)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-base)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              {t}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function BrowsePanel({ hierarchy, onSelect, selectedTitle }: PickerProps) {
  const [major, setMajor] = useState<string>("");
  const [minor, setMinor] = useState<string>("");
  const [broad, setBroad] = useState<string>("");

  const majors = useMemo(() => {
    const s = new Set<string>();
    hierarchy.forEach((h) => h.major && s.add(h.major));
    return Array.from(s).sort();
  }, [hierarchy]);

  const minors = useMemo(() => {
    if (!major) return [];
    const s = new Set<string>();
    hierarchy.forEach((h) => h.major === major && h.minor && s.add(h.minor));
    return Array.from(s).sort();
  }, [hierarchy, major]);

  const broads = useMemo(() => {
    if (!minor) return [];
    const s = new Set<string>();
    hierarchy.forEach((h) => h.minor === minor && h.broad && s.add(h.broad));
    return Array.from(s).sort();
  }, [hierarchy, minor]);

  const occs = useMemo(() => {
    if (!broad) return [];
    return hierarchy.filter((h) => h.broad === broad).map((h) => h.title).sort();
  }, [hierarchy, broad]);

  const onMajor = (m: string) => { setMajor(m); setMinor(""); setBroad(""); };
  const onMinor = (m: string) => { setMinor(m); setBroad(""); };

  useEffect(() => {
    if (!selectedTitle || !hierarchy.length) return;
    const found = hierarchy.find((h) => h.title === selectedTitle);
    if (!found) return;
    if (found.major && found.major !== major) setMajor(found.major);
    if (found.minor && found.minor !== minor) setMinor(found.minor);
    if (found.broad && found.broad !== broad) setBroad(found.broad);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTitle, hierarchy.length]);

  return (
    <div>
      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 10,
      }}>
        <BrowseSelect label="Major" value={major} options={majors}
                      placeholder="Pick a major category…" onChange={onMajor} />
        <BrowseSelect label="Minor" value={minor} options={minors}
                      placeholder={major ? "Pick a minor…" : "Pick a major first"}
                      onChange={onMinor} disabled={!major} />
        <BrowseSelect label="Broad" value={broad} options={broads}
                      placeholder={minor ? "Pick a broad…" : "Pick a minor first"}
                      onChange={setBroad} disabled={!minor} />
      </div>

      {broad ? (
        <div style={{
          maxHeight: 280, overflowY: "auto",
          border: "1px solid var(--border-light)", borderRadius: 8,
          background: "var(--bg-base)",
        }}>
          {occs.length === 0 && (
            <p style={{ padding: 14, fontSize: 13, color: "var(--text-muted)" }}>
              No occupations under this broad category.
            </p>
          )}
          {occs.map((t) => {
            const isSelected = t === selectedTitle;
            return (
              <button
                key={t}
                onClick={() => onSelect(t)}
                style={{
                  display: "block", width: "100%", textAlign: "left",
                  padding: "10px 14px", fontSize: 13,
                  background: isSelected ? "var(--brand-light)" : "transparent",
                  color: isSelected ? "var(--brand)" : "var(--text-primary)",
                  border: "none", cursor: "pointer",
                  borderBottom: "1px solid var(--border-light)",
                  fontWeight: isSelected ? 600 : 400,
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) e.currentTarget.style.background = "var(--bg-surface)";
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) e.currentTarget.style.background = "transparent";
                }}
              >
                {t}
              </button>
            );
          })}
        </div>
      ) : (
        <p style={{ fontSize: 12, color: "var(--text-muted)", padding: "12px 4px 4px" }}>
          Pick a Major → Minor → Broad to see the occupations under that branch.
        </p>
      )}
    </div>
  );
}

function BrowseSelect({
  label, value, options, placeholder, onChange, disabled,
}: {
  label: string; value: string; options: string[]; placeholder: string;
  onChange: (v: string) => void; disabled?: boolean;
}) {
  return (
    <div>
      <p style={{
        fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase",
        letterSpacing: "0.04em", marginBottom: 4,
      }}>{label}</p>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: "100%", padding: "9px 12px",
          border: "1px solid var(--border)", borderRadius: 8,
          fontSize: 13, cursor: disabled ? "not-allowed" : "pointer",
          background: disabled ? "var(--bg-base)" : "var(--bg-surface)",
          color: value ? "var(--text-primary)" : "var(--text-muted)",
          opacity: disabled ? 0.6 : 1,
        }}
      >
        <option value="">{placeholder}</option>
        {options.map((o) => (<option key={o} value={o}>{o}</option>))}
      </select>
    </div>
  );
}

/* ── Report body ──────────────────────────────────────────────────────────── */

function ReportBody({
  report, waLevel, setWaLevel, showRiskFlags, setShowRiskFlags, geoName,
}: {
  report: OccupationReport;
  waLevel: WaLevel;
  setWaLevel: (l: WaLevel) => void;
  showRiskFlags: boolean;
  setShowRiskFlags: (b: boolean) => void;
  geoName: string;
}) {
  return (
    <div style={{ marginTop: 28, display: "flex", flexDirection: "column", gap: 16 }}>
      <Hero
        report={report}
        showRiskFlags={showRiskFlags}
        setShowRiskFlags={setShowRiskFlags}
        geoName={geoName}
      />

      <KpiRow report={report} />

      <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 16 }}>
        <Card span={4} title="Where you rank"
              sub="Where THIS occupation sits within each scope on % tasks affected (lower number = more exposed). Workers / wages ranks shown beneath each bar.">
          <RankBars report={report} />
        </Card>

        <Card span={4} title="Sector chain"
              sub="How THIS occupation's major / minor / broad CATEGORIES rank against all categories at the same level — and the categories' aggregate stats.">
          <SectorChain report={report} />
        </Card>

        <Card span={4} title="Software commodities"
              sub="O*NET commodity categories this occupation's tools fall into. Count = # of distinct softwares in this occ. Rank = where this commodity sits economy-wide on average AI exposure.">
          <TechCommodities items={report.tech} />
        </Card>

        <Card span={12} title="Tasks AI can help with"
              sub={
                <>
                  Each task is rated by all four data sources (AEI Conv = Claude conversational,
                  AEI API = agentic tool-use, Microsoft Copilot, MCP servers). The colored bar reflects
                  the max across AEI Conv, AEI API, and Microsoft. <strong>AEI API</strong> specifically
                  captures agentic AI capability: where it&apos;s high, the work can be done by AI
                  tools acting on its own (file edits, API calls, browsing) rather than just chat. Click
                  a task to see the top MCP servers that match it.
                </>
              }>
          <PaletteLegend />
          <TasksByTier tasks={report.tasks} />
        </Card>

        <Card span={12} title="Where AI leads, where you lead"
              sub="Skills + Knowledge + Abilities (importance ≥ 3 only). “AI capability” is the top-10-occupation average for that element. Above 100% of need = AI leads. Sorted with biggest AI lead at top.">
          <SkaSection report={report} />
        </Card>

        <Card span={12} title="Your work activities"
              sub="The same per-source AI ratings, rolled up to the categories your tasks fall into. Each WA also shows its economy-wide stats and rank among all GWAs / IWAs / DWAs.">
          <WaSection report={report} waLevel={waLevel} setWaLevel={setWaLevel} />
        </Card>

        <Card span={12} title="Similar occupations"
              sub="Closest match by Skills + Knowledge + Abilities profile (L1 distance over importance×level vectors). Useful for seeing whether occupations with similar skill demands face similar AI exposure.">
          <SimilarTable report={report} />
        </Card>
      </div>

      <PaletteFooter primaryDataset={report.primary_dataset} />
    </div>
  );
}

/* ── Hero ─────────────────────────────────────────────────────────────────── */

function Hero({
  report, showRiskFlags, setShowRiskFlags, geoName,
}: {
  report: OccupationReport;
  showRiskFlags: boolean;
  setShowRiskFlags: (b: boolean) => void;
  geoName: string;
}) {
  const h = report.headline;
  const tier = h.risk.tier;
  const tierStyle = TIER_COLORS[tier] ?? TIER_COLORS.low;
  const jzInterp = h.job_zone ? JOB_ZONE_INTERP[Math.round(h.job_zone)] : null;
  const olInterp = h.dws_star_rating ? OUTLOOK_INTERP[Math.round(h.dws_star_rating)] : null;
  const flagsRaised = Object.values(h.risk.flags).reduce((a, b) => a + b, 0);
  return (
    <section style={{
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: 12, padding: 28,
    }}>
      <div style={{
        display: "grid", gridTemplateColumns: "1fr auto", gap: 28,
        alignItems: "center",
      }}>
        <div>
          <p style={{
            fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.12em",
            textTransform: "uppercase", fontWeight: 600, marginBottom: 6, marginTop: 0,
          }}>
            Occupation report · {geoName}
          </p>
          <h2 style={{
            fontSize: 30, fontWeight: 700, letterSpacing: "-0.02em",
            marginBottom: 6, marginTop: 0, color: "var(--text-primary)",
          }}>
            {h.title}
          </h2>
          <p style={{
            fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5,
            marginBottom: 0, marginTop: 0,
          }}>
            {[h.broad, h.minor, h.major].filter(Boolean).join(" · ")}
          </p>
          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            {h.job_zone !== null && h.job_zone !== undefined && (
              <Pill>Job Zone {h.job_zone.toFixed(0)}{jzInterp ? ` · ${jzInterp}` : ""}</Pill>
            )}
            {h.dws_star_rating !== null && h.dws_star_rating !== undefined && (
              <Pill>Outlook {h.dws_star_rating.toFixed(0)}/5{olInterp ? ` · ${olInterp}` : ""}</Pill>
            )}
            {h.n_tasks !== null && h.n_tasks !== undefined && (
              <Pill>{h.n_tasks} O*NET tasks</Pill>
            )}
            {h.emp !== null && h.emp !== undefined && (
              <Pill>{fmtNumber(h.emp)} workers</Pill>
            )}
          </div>
        </div>
        <div style={{
          display: "flex", gap: 20, alignItems: "center",
          padding: "18px 22px", borderRadius: 14,
          background: tierStyle.bg, border: `1px solid ${tierStyle.fg}33`,
        }}>
          <RiskGauge score={h.risk.score} tier={tier} size={108} />
          <div>
            <p style={{
              fontSize: 11, color: "var(--text-secondary)", letterSpacing: "0.06em",
              textTransform: "uppercase", fontWeight: 600, margin: 0,
            }}>
              Exposure tier
            </p>
            <p style={{
              fontSize: 24, fontWeight: 700, color: tierStyle.fg, lineHeight: 1.1,
              margin: "2px 0 0",
            }}>
              {tierStyle.label}
            </p>
            <p style={{
              fontSize: 11, color: "var(--text-secondary)", marginTop: 6, lineHeight: 1.4,
              marginBottom: 6,
            }}>
              {flagsRaised} of 8 flags raised
            </p>
            <button
              onClick={() => setShowRiskFlags(!showRiskFlags)}
              style={{
                fontSize: 11, padding: "4px 10px", borderRadius: 6,
                background: "var(--bg-surface)", border: "1px solid var(--border)",
                color: "var(--text-secondary)", cursor: "pointer",
              }}
            >
              {showRiskFlags ? "Hide" : "Why?"}
            </button>
          </div>
        </div>
      </div>

      {showRiskFlags && <RiskFlagsTable risk={h.risk} />}
    </section>
  );
}

const RISK_FLAG_LABELS: Record<string, string> = {
  flag1_pct:        "Pct tasks affected > 50%",
  flag2_ska:        "SKA percentage > median",
  flag3_pct_trend:  "Pct trend rising fast",
  flag4_ska_trend:  "SKA gap rising fast",
  flag5_job_zone:   "Job zone 1–3",
  flag6_outlook:    "Outlook 2–3",
  flag7_n_software: "n_software > median",
  flag8_auto_aug:   "Auto-aug > median",
};

function RiskFlagsTable({ risk }: { risk: OccupationReport["headline"]["risk"] }) {
  return (
    <div style={{
      marginTop: 18, padding: 14, borderRadius: 8,
      background: "var(--bg-base)", border: "1px solid var(--border-light)",
    }}>
      <p style={{
        fontSize: 11, color: "var(--text-muted)", marginBottom: 8, lineHeight: 1.4,
      }}>
        Exposure score is built from 8 binary flags weighted 1× or 2×. High Exposure requires a score of 8+
        AND pct_tasks_affected ≥ 33%; otherwise it caps at Mod-High.
      </p>
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 6,
      }}>
        {Object.entries(RISK_FLAG_LABELS).map(([key, lbl]) => {
          const v = (risk.flags as unknown as Record<string, number>)[key] ?? 0;
          return (
            <div key={key} style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "5px 10px", borderRadius: 6,
              background: v ? "rgba(184, 96, 60, 0.08)" : "transparent",
              border: "1px solid var(--border-light)",
              fontSize: 12,
            }}>
              <span style={{ color: v ? "var(--text-primary)" : "var(--text-muted)" }}>{lbl}</span>
              <span style={{ fontWeight: 600, color: v ? "#8a4225" : "var(--text-muted)" }}>
                {v ? "✓" : "—"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── KPI row ──────────────────────────────────────────────────────────────── */

function KpiRow({ report }: { report: OccupationReport }) {
  const h = report.headline;
  const trendValid = report.trend.filter(
    (p) => p.pct_tasks_affected !== null && p.pct_tasks_affected !== undefined,
  );
  const firstTrend = trendValid[0];
  const subTrend = firstTrend
    ? `Up from ${(firstTrend.pct_tasks_affected as number).toFixed(0)}% in ${firstTrend.date.slice(0, 7)}`
    : undefined;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1.2fr 1fr 1fr 1fr", gap: 12,
    }}>
      <KpiCard
        accent="var(--brand)"
        big
        label="% of weighted tasks affected"
        value={h.pct_tasks_affected !== null && h.pct_tasks_affected !== undefined
          ? `${h.pct_tasks_affected}%`
          : "—"}
        sub={subTrend}
        chart={trendValid.length >= 2
          ? <Sparkline points={report.trend} w={220} h={48} color="var(--brand)" />
          : undefined}
      />
      <KpiCard
        label="Workers affected"
        value={fmtNumber(h.workers_affected)}
        sub={`of ${fmtNumber(h.emp)} employed`}
        rank={report.group_ranks.economy.workers !== undefined
          ? `#${report.group_ranks.economy.workers} / ${report.group_ranks.economy.total} occupations`
          : undefined}
      />
      <KpiCard
        label="Wages affected"
        value={fmtWage(h.wages_affected)}
        sub={`Median ${fmtWage(h.wage)}/yr`}
        rank={report.group_ranks.economy.wages !== undefined
          ? `#${report.group_ranks.economy.wages} / ${report.group_ranks.economy.total}`
          : undefined}
      />
      <KpiCard
        label="AI adoption rank"
        value={h.intensity.occ_intensity_rank !== null && h.intensity.occ_intensity_rank !== undefined
          ? `#${h.intensity.occ_intensity_rank}`
          : "—"}
        sub={h.intensity.occ_intensity_total !== null && h.intensity.occ_intensity_total !== undefined
          ? `of ${fmtNumber(h.intensity.occ_intensity_total)} occupations · how much workers in this occupation already use AI relative to their task load`
          : undefined}
        rank={h.intensity.major_intensity_rank !== null && h.intensity.major_intensity_rank !== undefined
          ? `#${h.intensity.major_intensity_rank} / ${h.intensity.major_intensity_total} in major`
          : undefined}
      />
    </div>
  );
}

function KpiCard({
  label, value, sub, rank, accent, chart, big,
}: {
  label: string; value: string; sub?: string; rank?: string;
  accent?: string; chart?: React.ReactNode; big?: boolean;
}) {
  return (
    <div style={{
      background: "var(--bg-surface)", border: "1px solid var(--border)",
      borderRadius: 12, padding: 18, position: "relative", overflow: "hidden",
    }}>
      {accent && (
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0,
          width: 3, background: accent,
        }} />
      )}
      <p style={{
        fontSize: 11, color: "var(--text-muted)", letterSpacing: "0.06em",
        textTransform: "uppercase", fontWeight: 600, marginBottom: 8, marginTop: 0,
      }}>
        {label}
      </p>
      <p style={{
        fontSize: big ? 38 : 26, fontWeight: 700, lineHeight: 1,
        letterSpacing: "-0.02em",
        color: accent || "var(--text-primary)",
        margin: 0, fontVariantNumeric: "tabular-nums",
      }}>
        {value}
      </p>
      {sub && (
        <p style={{
          fontSize: 12, color: "var(--text-secondary)", marginTop: 6, marginBottom: 0,
        }}>
          {sub}
        </p>
      )}
      {rank && (
        <p style={{
          fontSize: 11, color: "var(--text-muted)", marginTop: 4, marginBottom: 0,
        }}>
          {rank}
        </p>
      )}
      {chart && <div style={{ marginTop: 8 }}>{chart}</div>}
    </div>
  );
}

/* ── Where you rank ──────────────────────────────────────────────────────── */

function RankBars({ report }: { report: OccupationReport }) {
  const r = report.group_ranks;
  const intensity = report.headline.intensity;
  const scopes: Array<{ name: string; r: typeof r.economy | undefined | null }> = [
    { name: "Economy", r: r.economy },
    { name: "Major",   r: r.major },
    { name: "Minor",   r: r.minor },
    { name: "Broad",   r: r.broad },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {scopes.map(({ name, r: ranks }) => {
          if (!ranks || ranks.pct === undefined) return null;
          const pos = ((ranks.total - ranks.pct) / ranks.total) * 100;
          return (
            <div key={name}>
              <div style={{
                display: "flex", justifyContent: "space-between", marginBottom: 4,
              }}>
                <span style={{ fontSize: 12, color: "var(--text-secondary)", fontWeight: 500 }}>
                  {name}
                </span>
                <span style={{
                  fontSize: 12, color: "var(--text-primary)", fontWeight: 600,
                  fontVariantNumeric: "tabular-nums",
                }}>
                  #{ranks.pct} <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                    / {ranks.total}
                  </span>
                </span>
              </div>
              <div style={{
                height: 6, background: "rgba(0,0,0,0.05)", borderRadius: 3,
                overflow: "hidden", position: "relative",
              }}>
                <div style={{
                  position: "absolute", inset: 0, width: `${pos}%`,
                  background: "linear-gradient(90deg, var(--brand), #c87a5b)",
                }} />
              </div>
              <div style={{
                display: "flex", gap: 12, marginTop: 4,
                fontSize: 10, color: "var(--text-muted)",
                fontVariantNumeric: "tabular-nums",
              }}>
                {ranks.workers !== undefined && <span>workers #{ranks.workers}</span>}
                {ranks.wages !== undefined && <span>wages #{ranks.wages}</span>}
              </div>
            </div>
          );
        })}
      </div>
      {intensity.occ_intensity_rank !== null && intensity.occ_intensity_rank !== undefined && (
        <div style={{
          padding: "10px 12px", borderRadius: 8,
          background: "var(--bg-base)", border: "1px solid var(--border-light)",
        }}>
          <p style={{
            fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em",
            color: "var(--text-muted)", marginBottom: 6, marginTop: 0, fontWeight: 600,
          }}>
            Per-task intensity
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, fontSize: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-secondary)" }}>Occupation</span>
              <span style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                {fmtRankShort(intensity.occ_intensity_rank, intensity.occ_intensity_total ?? 0)}
              </span>
            </div>
            {intensity.major_intensity_rank !== null && intensity.major_intensity_rank !== undefined && (
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-secondary)" }}>Major</span>
                <span style={{ fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                  {fmtRankShort(intensity.major_intensity_rank, intensity.major_intensity_total ?? 0)}
                </span>
              </div>
            )}
          </div>
          <p style={{
            fontSize: 10, color: "var(--text-muted)", marginTop: 6, marginBottom: 0, lineHeight: 1.4,
          }}>
            Σ pct ÷ Σ freq×emp, bias-corrected.
          </p>
        </div>
      )}
    </div>
  );
}

/* ── Sector chain (major / minor / broad) ────────────────────────────────── */

function SectorChain({ report }: { report: OccupationReport }) {
  const chain = report.sector_chain;
  const entries: Array<{ key: string; label: string; entry: OccReportSectorChainEntry | null }> = [
    { key: "major", label: "Major", entry: chain?.major ?? null },
    { key: "minor", label: "Minor", entry: chain?.minor ?? null },
    { key: "broad", label: "Broad", entry: chain?.broad ?? null },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {entries.map(({ key, label, entry }) => {
        if (!entry) return null;
        return (
          <div key={key} style={{
            padding: "10px 12px", borderRadius: 8,
            background: "var(--bg-base)", border: "1px solid var(--border-light)",
          }}>
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "baseline",
              marginBottom: 6, gap: 8,
            }}>
              <p style={{
                fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase",
                letterSpacing: "0.06em", fontWeight: 600, margin: 0,
              }}>
                {label}
              </p>
              <p style={{
                fontSize: 11, color: "var(--brand)", fontWeight: 600, margin: 0,
                fontVariantNumeric: "tabular-nums",
              }}>
                {fmtRankShort(entry.rank_pct, entry.total)}
              </p>
            </div>
            <p style={{
              fontSize: 13, fontWeight: 600, color: "var(--text-primary)",
              marginBottom: 6, marginTop: 0, lineHeight: 1.3,
            }}>
              {entry.name}
            </p>
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6, fontSize: 11,
              fontVariantNumeric: "tabular-nums",
            }}>
              <SectorChainStat label="% Tasks"
                               value={fmtPct(entry.pct_tasks_affected)}
                               rank={entry.rank_pct} total={entry.total} />
              <SectorChainStat label="Workers"
                               value={fmtNumber(entry.workers_affected)}
                               rank={entry.rank_workers} total={entry.total} />
              <SectorChainStat label="Wages"
                               value={fmtWage(entry.wages_affected)}
                               rank={entry.rank_wages} total={entry.total} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SectorChainStat({
  label, value, rank, total,
}: {
  label: string; value: string;
  rank?: number | null; total: number;
}) {
  return (
    <div>
      <p style={{
        fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase",
        letterSpacing: "0.04em", margin: 0,
      }}>{label}</p>
      <p style={{
        fontSize: 12, fontWeight: 600, color: "var(--text-primary)", margin: "2px 0 0",
      }}>{value}</p>
      <p style={{ fontSize: 9, color: "var(--text-muted)", margin: 0 }}>
        {fmtRankShort(rank, total)}
      </p>
    </div>
  );
}

/* ── Tools list ───────────────────────────────────────────────────────────── */

interface CommodityRow {
  commodity: string;
  count: number;
  rank: number | null;
  total: number;
}

function TechCommodities({ items }: { items: OccupationReport["tech"] }) {
  const grouped = useMemo<CommodityRow[]>(() => {
    const byCommodity = new Map<string, CommodityRow>();
    for (const t of items) {
      const existing = byCommodity.get(t.commodity);
      if (existing) {
        existing.count += 1;
      } else {
        byCommodity.set(t.commodity, {
          commodity: t.commodity,
          count: 1,
          rank: t.commodity_rank ?? null,
          total: t.commodity_total,
        });
      }
    }
    return Array.from(byCommodity.values()).sort((a, b) => {
      if (a.rank == null && b.rank == null) return 0;
      if (a.rank == null) return 1;
      if (b.rank == null) return -1;
      return a.rank - b.rank;
    });
  }, [items]);

  if (!grouped.length) return (
    <p style={{ fontSize: 12, color: "var(--text-muted)" }}>No software commodities listed.</p>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", maxHeight: 420, overflowY: "auto" }}>
      {grouped.map((c, i) => (
        <div key={`${c.commodity}-${i}`} style={{
          display: "grid", gridTemplateColumns: "1fr 70px", gap: 8, alignItems: "center",
          padding: "8px 0",
          borderTop: i === 0 ? "none" : "1px dotted var(--border-light)",
        }}>
          <div style={{ minWidth: 0 }}>
            <p style={{
              fontSize: 12, fontWeight: 500, margin: 0, color: "var(--text-primary)",
              lineHeight: 1.35,
            }}>{c.commodity}</p>
            <p style={{
              fontSize: 10, color: "var(--text-muted)", margin: "2px 0 0",
            }}>
              {c.count} {c.count === 1 ? "tool" : "tools"} in this occupation
            </p>
          </div>
          <p style={{
            fontSize: 12, fontVariantNumeric: "tabular-nums", textAlign: "right",
            color: "#a35135", fontWeight: 600, margin: 0,
          }}>
            {c.rank ? `#${c.rank}` : "—"}
            {c.total ? <span style={{
              color: "var(--text-muted)", fontWeight: 400,
            }}>/{c.total}</span> : null}
          </p>
        </div>
      ))}
    </div>
  );
}

/* ── Tasks (grouped by tier) ─────────────────────────────────────────────── */

interface TaskWithMeta extends OccReportTask {
  value_weight: number;
  value_rank: number;
  value_total: number;
}

const TASK_GRID = "1fr 320px 80px 70px 60px 24px";

function TasksByTier({ tasks }: { tasks: OccReportTask[] }) {
  // Compute freq × imp × rel ("value") weight + rank-within-occupation once
  // for ALL tasks before splitting by tier, so ranks reflect the whole occ.
  const tasksWithMeta = useMemo<TaskWithMeta[]>(() => {
    const withWeight = tasks.map((t) => ({
      ...t,
      value_weight: (t.freq_mean ?? 0) * (t.importance ?? 0) * (t.relevance ?? 0),
    }));
    const sorted = [...withWeight].sort((a, b) => b.value_weight - a.value_weight);
    const rankByKey = new Map<string, number>();
    sorted.forEach((t, i) => rankByKey.set(t.task_normalized, i + 1));
    const total = sorted.length;
    return withWeight.map((t) => ({
      ...t,
      value_rank: rankByKey.get(t.task_normalized) ?? 0,
      value_total: total,
    }));
  }, [tasks]);

  const buckets: ColorBucket[] = ["high", "mid", "low", "none"];
  const grouped: Record<ColorBucket, TaskWithMeta[]> = { high: [], mid: [], low: [], none: [] };
  for (const t of tasksWithMeta) grouped[t.color_bucket].push(t);

  return (
    <div style={{ marginTop: 10 }}>
      <TaskHeaderRow />
      {buckets.map((b) => {
        const rows = grouped[b];
        if (!rows.length) return null;
        return (
          <TierGroup key={b} bucket={b} count={rows.length} defaultOpen={b !== "low" && b !== "none"}>
            <TaskRowsCompact tasks={rows} />
          </TierGroup>
        );
      })}
    </div>
  );
}

function TaskHeaderRow() {
  const labelStyle: React.CSSProperties = {
    fontSize: 9, color: "var(--text-muted)", letterSpacing: "0.06em",
    textTransform: "uppercase", fontWeight: 600, margin: 0,
  };
  return (
    <div style={{
      display: "grid", gridTemplateColumns: TASK_GRID, gap: 14,
      alignItems: "center", padding: "0 4px 8px",
      borderBottom: "1px solid var(--border-light)", marginBottom: 8,
    }}>
      <p style={labelStyle}>Task</p>
      <p style={{ ...labelStyle, textAlign: "left" }}>Per-source AI score (0–5)</p>
      <p style={{ ...labelStyle, textAlign: "right" }}>Freq×Imp×Rel</p>
      <p style={{ ...labelStyle, textAlign: "right" }}>Rank in occ</p>
      <p style={{ ...labelStyle, textAlign: "right" }}>Max AI</p>
      <span />
    </div>
  );
}

function TaskRowsCompact({ tasks }: { tasks: TaskWithMeta[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      {tasks.map((t) => {
        const isOpen = open === t.task_normalized;
        const max = Math.max(
          t.aei_conv_max ?? 0, t.aei_api_max ?? 0,
          t.microsoft ?? 0, t.mcp ?? 0,
        );
        const hasMcps = t.top_mcps.length > 0;
        // Higher value rank = higher in occ. Map rank→intensity 1.0 (top) to 0.35 (bottom).
        const intensity = t.value_total > 1
          ? 1 - (t.value_rank - 1) / (t.value_total - 1)
          : 1;
        const rankOpacity = 0.35 + intensity * 0.65;
        const rankWeight = intensity > 0.66 ? 700 : intensity > 0.33 ? 600 : 400;
        return (
          <div
            key={t.task_normalized}
            onClick={() => hasMcps && setOpen(isOpen ? null : t.task_normalized)}
            style={{
              borderTop: "1px solid var(--border-light)",
              padding: "10px 4px",
              cursor: hasMcps ? "pointer" : "default",
            }}
          >
            <div style={{
              display: "grid", gridTemplateColumns: TASK_GRID, gap: 14,
              alignItems: "center",
            }}>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
                <span style={{ marginTop: 5 }}><TierDot bucket={t.color_bucket} /></span>
                <p style={{
                  fontSize: 13, lineHeight: 1.45, margin: 0, color: "var(--text-primary)",
                }}>
                  {t.task}
                </p>
              </div>
              <SourceMiniBars t={t} />
              <p style={{
                fontSize: 13, textAlign: "right", margin: 0,
                color: "var(--text-primary)", fontWeight: 500,
                fontVariantNumeric: "tabular-nums",
              }}>
                {t.value_weight > 0 ? t.value_weight.toFixed(1) : "—"}
              </p>
              <p style={{
                fontSize: 13, textAlign: "right", margin: 0,
                color: "var(--text-primary)", opacity: rankOpacity, fontWeight: rankWeight,
                fontVariantNumeric: "tabular-nums",
              }}>
                {t.value_rank}<span style={{ color: "var(--text-muted)", fontWeight: 400 }}>/{t.value_total}</span>
              </p>
              <p style={{
                fontSize: 16, fontWeight: 700, textAlign: "right",
                color: BUCKET_FG[t.color_bucket],
                fontVariantNumeric: "tabular-nums", margin: 0,
              }}>
                {max > 0 ? max.toFixed(1) : "—"}
              </p>
              <span style={{
                textAlign: "right",
                color: hasMcps ? "var(--text-muted)" : "transparent",
                fontSize: 11, transition: "transform 0.15s",
                transform: isOpen ? "rotate(90deg)" : "rotate(0deg)",
                display: "inline-block",
              }}>
                {hasMcps ? "▶" : ""}
              </span>
            </div>
            {isOpen && hasMcps && (
              <div style={{
                marginTop: 10, padding: "10px 12px", background: "#fafaf6",
                border: "1px solid var(--border-light)", borderRadius: 8,
              }}>
                <p style={{
                  fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.08em",
                  textTransform: "uppercase", marginBottom: 6, marginTop: 0,
                }}>
                  Top MCP servers
                </p>
                {t.top_mcps.map((m, i) => (
                  <div key={i} style={{
                    fontSize: 12, padding: "4px 0", color: "var(--text-secondary)",
                  }}>
                    <strong>
                      {m.url ? (
                        <a href={m.url} target="_blank" rel="noreferrer" style={{ color: "var(--brand)" }}>
                          {m.title}
                        </a>
                      ) : (
                        <span style={{ color: "var(--brand)" }}>{m.title}</span>
                      )}
                    </strong>
                    {m.rating !== null && m.rating !== undefined && (
                      <span style={{ color: "var(--text-muted)" }}> · {m.rating}★</span>
                    )}
                    {m.description && <> — {m.description}</>}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── SKA section ─────────────────────────────────────────────────────────── */

function SkaSection({ report }: { report: OccupationReport }) {
  const s = report.ska.summary;
  return (
    <div>
      <div style={{
        marginBottom: 14, padding: "12px 16px", background: "var(--bg-base)",
        border: "1px solid var(--border-light)", borderRadius: 8,
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
        gap: 12,
      }}>
        <SkaSummaryStat label="Overall"   pct={s.overall_pct} />
        <SkaSummaryStat label="Skills"    pct={s.skills_pct} />
        <SkaSummaryStat label="Abilities" pct={s.abilities_pct} />
        <SkaSummaryStat label="Knowledge" pct={s.knowledge_pct} />
      </div>

      <SkaSubsection title="Skills" rows={report.ska.rows.skills} defaultOpen />
      <SkaSubsection title="Knowledge" rows={report.ska.rows.knowledge} defaultOpen={false} />
      <SkaSubsection title="Abilities" rows={report.ska.rows.abilities} defaultOpen={false} />
    </div>
  );
}

function SkaSummaryStat({ label, pct }: { label: string; pct?: number | null }) {
  if (pct === null || pct === undefined) return null;
  const bucket: ColorBucket = pct >= 100 ? "high" : pct >= 66 ? "mid" : "low";
  return (
    <div>
      <p style={{
        fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase",
        letterSpacing: "0.04em", marginBottom: 4, marginTop: 0,
      }}>{label}</p>
      <p style={{
        fontSize: 20, fontWeight: 700, color: BUCKET_FG[bucket], margin: 0,
        fontVariantNumeric: "tabular-nums",
      }}>
        {pct.toFixed(0)}%
      </p>
      <p style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2, marginBottom: 0 }}>
        AI capability vs. occ requirement
      </p>
    </div>
  );
}

function SkaSubsection({
  title, rows, defaultOpen,
}: { title: string; rows: OccReportSkaRow[]; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!rows.length) return null;

  const buckets: ColorBucket[] = ["high", "mid", "low", "none"];
  const grouped: Record<ColorBucket, OccReportSkaRow[]> = { high: [], mid: [], low: [], none: [] };
  for (const r of rows) grouped[r.color_bucket].push(r);

  return (
    <div style={{ marginTop: 14 }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          width: "100%", display: "flex", alignItems: "center", gap: 8,
          padding: "8px 12px", background: "var(--bg-base)",
          border: "1px solid var(--border-light)", borderRadius: 8,
          cursor: "pointer", textAlign: "left",
        }}
      >
        <Chevron open={open} />
        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
          {title}
        </span>
        <span style={{
          marginLeft: "auto", fontSize: 11, color: "var(--text-muted)",
          fontVariantNumeric: "tabular-nums",
        }}>
          {rows.length} elements
        </span>
      </button>
      {open && (
        <div style={{ marginTop: 8 }}>
          {buckets.map((b) => {
            const sub = grouped[b];
            if (!sub.length) return null;
            return (
              <TierGroup key={b} bucket={b} count={sub.length} defaultOpen={b !== "low" && b !== "none"}>
                <SkaTable rows={sub} />
              </TierGroup>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SkaTable({ rows }: { rows: OccReportSkaRow[] }) {
  return (
    <div style={{
      overflowX: "auto", border: "1px solid var(--border-light)", borderRadius: 8,
    }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "var(--bg-base)", textAlign: "left" }}>
            <Th>Element</Th>
            <Th align="right">Importance</Th>
            <Th align="right">Level</Th>
            <Th align="right">Your score</Th>
            <Th align="right">AI top-10</Th>
            <Th align="right">AI as % of need</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--border-light)" }}>
              <Td>
                <span style={{
                  display: "inline-block", width: 8, height: 8, borderRadius: "50%",
                  background: BUCKET_DOT[r.color_bucket], marginRight: 8,
                }} />
                {r.element}
              </Td>
              <Td align="right">{fmtNumber(r.importance, 1)}</Td>
              <Td align="right">{fmtNumber(r.level, 1)}</Td>
              <Td align="right">{fmtNumber(r.occ_score, 1)}</Td>
              <Td align="right">{fmtNumber(r.ai_top10, 1)}</Td>
              <Td align="right" style={{ fontWeight: 600, color: BUCKET_FG[r.color_bucket] }}>
                {fmtPctOfNeed(r.pct_of_need)}
              </Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Work activities ─────────────────────────────────────────────────────── */

function WaSection({
  report, waLevel, setWaLevel,
}: {
  report: OccupationReport; waLevel: WaLevel; setWaLevel: (l: WaLevel) => void;
}) {
  const rows = report.work_activities[waLevel];
  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        {(["gwa", "iwa", "dwa"] as const).map((lvl) => (
          <button
            key={lvl}
            onClick={() => setWaLevel(lvl)}
            style={{
              padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
              cursor: "pointer", border: "1px solid var(--border)",
              background: waLevel === lvl ? "var(--brand-light)" : "var(--bg-surface)",
              color: waLevel === lvl ? "var(--brand)" : "var(--text-secondary)",
            }}
          >
            {lvl.toUpperCase()}
          </button>
        ))}
      </div>
      <WaByTier rows={rows} level={waLevel} />
    </div>
  );
}

function WaByTier({ rows, level }: { rows: OccReportWaRow[]; level: WaLevel }) {
  const buckets: ColorBucket[] = ["high", "mid", "low", "none"];
  const grouped: Record<ColorBucket, OccReportWaRow[]> = { high: [], mid: [], low: [], none: [] };
  for (const r of rows) grouped[r.color_bucket].push(r);
  if (!rows.length) {
    return (
      <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
        No {level.toUpperCase()} activities.
      </p>
    );
  }
  return (
    <div>
      {buckets.map((b) => {
        const sub = grouped[b];
        if (!sub.length) return null;
        return (
          <TierGroup key={b} bucket={b} count={sub.length} defaultOpen={b !== "low" && b !== "none"}>
            <WaList rows={sub} />
          </TierGroup>
        );
      })}
    </div>
  );
}

function WaList({ rows }: { rows: OccReportWaRow[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {rows.map((r, i) => (<WaCard key={`${r.name}-${i}`} row={r} />))}
    </div>
  );
}

function WaCard({ row }: { row: OccReportWaRow }) {
  const eco = row.eco_stats;
  return (
    <div style={{
      padding: "12px 14px", border: "1px solid var(--border-light)", borderRadius: 8,
      background: "var(--bg-surface)",
    }}>
      <div style={{
        display: "flex", alignItems: "baseline", gap: 8, marginBottom: 12,
      }}>
        <span style={{ alignSelf: "center" }}><TierDot bucket={row.color_bucket} /></span>
        <h4 style={{
          fontSize: 14, fontWeight: 600, margin: 0, color: "var(--text-primary)",
          flex: 1, minWidth: 0, lineHeight: 1.35,
        }}>
          {row.name}
        </h4>
        <span style={{
          fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums",
          whiteSpace: "nowrap",
        }}>
          {row.n_tasks} {row.n_tasks === 1 ? "task" : "tasks"}
        </span>
      </div>
      <SourceMiniBars t={row} />
      {eco && (
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12,
          paddingTop: 10, borderTop: "1px dotted var(--border-light)",
        }}>
          <EcoPill label="Eco %"   value={fmtPct(eco.pct_tasks_affected)}    rank={eco.rank_pct}     total={eco.total} />
          <EcoPill label="Workers" value={fmtNumber(eco.workers_affected)}   rank={eco.rank_workers} total={eco.total} />
          <EcoPill label="Wages"   value={fmtWage(eco.wages_affected)}       rank={eco.rank_wages}   total={eco.total} />
          <EcoPill label="Auto"    value={fmtAuto(eco.auto_aug_mean)}        rank={eco.rank_auto}    total={eco.total} />
        </div>
      )}
    </div>
  );
}

function EcoPill({
  label, value, rank, total,
}: { label: string; value: string; rank?: number | null; total: number }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "baseline", gap: 6,
      padding: "4px 10px", background: "var(--bg-base)",
      border: "1px solid var(--border-light)", borderRadius: 6,
      fontSize: 12,
    }}>
      <span style={{
        color: "var(--text-muted)", textTransform: "uppercase",
        fontSize: 9, fontWeight: 600, letterSpacing: "0.04em",
      }}>{label}</span>
      <span style={{
        fontWeight: 600, fontVariantNumeric: "tabular-nums",
        color: "var(--text-primary)",
      }}>{value}</span>
      {rank != null && (
        <span style={{
          fontSize: 10, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums",
        }}>
          #{rank}/{total}
        </span>
      )}
    </div>
  );
}

/* ── Similar occupations ─────────────────────────────────────────────────── */

const SIMILAR_FLAG_KEYS: Array<keyof OccupationReport["headline"]["risk"]["flags"]> = [
  "flag1_pct", "flag2_ska", "flag3_pct_trend", "flag4_ska_trend",
  "flag5_job_zone", "flag6_outlook", "flag7_n_software", "flag8_auto_aug",
];

function ExposureProfile({ risk }: { risk: OccReportSimilar["risk"] }) {
  if (!risk) {
    return <span style={{ color: "var(--text-muted)", fontSize: 12 }}>—</span>;
  }
  const tierStyle = TIER_COLORS[risk.tier] ?? TIER_COLORS.low;
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 4,
        padding: "2px 8px", borderRadius: 999,
        background: tierStyle.bg, color: tierStyle.fg,
        border: `1px solid ${tierStyle.fg}33`,
        fontSize: 10, fontWeight: 600, letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      }}>
        {tierStyle.short} · {risk.score}/10
      </span>
      <div style={{ display: "flex", gap: 3 }} title={`${risk.score} of 10 (8 weighted exposure flags)`}>
        {SIMILAR_FLAG_KEYS.map((k) => {
          const lit = (risk.flags as unknown as Record<string, number>)[k] ?? 0;
          return (
            <span key={k} style={{
              width: 7, height: 7, borderRadius: "50%",
              background: lit ? tierStyle.fg : "var(--border-light)",
            }} />
          );
        })}
      </div>
    </div>
  );
}

function SimilarTable({ report }: { report: OccupationReport }) {
  if (!report.similar.length) return (
    <p style={{ fontSize: 12, color: "var(--text-muted)" }}>No similar occupations found.</p>
  );
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "var(--bg-base)", textAlign: "left" }}>
            <Th>Occupation</Th>
            <Th>Sector</Th>
            <Th>Exposure</Th>
            <Th align="right">% Tasks Affected</Th>
            <Th align="right">Median Wage</Th>
            <Th align="right">Job Zone</Th>
            <Th align="right">Outlook</Th>
            <Th align="right">SKA distance</Th>
          </tr>
        </thead>
        <tbody>
          {report.similar.map((o, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--border-light)" }}>
              <Td>{o.title}</Td>
              <Td style={{ color: "var(--text-secondary)", fontSize: 12 }}>{o.major}</Td>
              <Td><ExposureProfile risk={o.risk} /></Td>
              <Td align="right">{fmtPct(o.pct_tasks_affected, 0)}</Td>
              <Td align="right">{fmtWage(o.wage)}</Td>
              <Td align="right">{o.job_zone ?? "—"}</Td>
              <Td align="right">{o.dws_star_rating ?? "—"}</Td>
              <Td align="right" style={{ color: "var(--text-muted)" }}>{fmtNumber(o.distance, 1)}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Palette legend / footer ─────────────────────────────────────────────── */

function PaletteLegend() {
  return (
    <div style={{
      display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap",
      marginBottom: 8,
    }}>
      {(["high", "mid", "low"] as ColorBucket[]).map((b) => (
        <div key={b} style={{
          display: "flex", alignItems: "center", gap: 6,
          fontSize: 11, color: "var(--text-secondary)",
        }}>
          <span style={{
            display: "inline-block", width: 12, height: 12, borderRadius: 3,
            background: BUCKET_BG[b], border: `1px solid ${BUCKET_BORDER[b]}`,
          }} />
          {BUCKET_LABEL[b]}
        </div>
      ))}
    </div>
  );
}

function PaletteFooter({ primaryDataset }: { primaryDataset: string }) {
  return (
    <p style={{
      fontSize: 11, color: "var(--text-muted)", textAlign: "center",
      marginTop: 12, lineHeight: 1.5,
    }}>
      Colors are tied to demonstrated AI usage levels: tasks with higher max auto-aug scores (≥4 across AEI
      Conv, AEI API, and Microsoft) show as &ldquo;more automated usage seen&rdquo;; 2.5–4 as &ldquo;more
      augmentative&rdquo;; below 2.5 as &ldquo;less automated usage seen.&rdquo; SKA elements use the same
      three-tier framing on AI capability as a percentage of the occupation&apos;s requirement (≥100% / 66–100% /
      &lt;66%). Source: <strong>{primaryDataset}</strong> for headline metrics
      and SKA; per-source auto-aug from the explorer task lookup.
    </p>
  );
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */

function Th({ children, align = "left" }: { children?: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th style={{
      padding: "10px 12px", fontSize: 11, fontWeight: 600,
      color: "var(--text-muted)", textTransform: "uppercase",
      letterSpacing: "0.04em", textAlign: align,
    }}>
      {children}
    </th>
  );
}

function Td({
  children, align = "left", style,
}: { children?: React.ReactNode; align?: "left" | "right"; style?: React.CSSProperties }) {
  return (
    <td style={{
      padding: "10px 12px", color: "var(--text-primary)",
      textAlign: align, verticalAlign: "top",
      ...(style ?? {}),
    }}>
      {children}
    </td>
  );
}
