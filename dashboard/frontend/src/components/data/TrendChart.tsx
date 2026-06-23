"use client";

import { useState } from "react";

export interface TrendSeries {
  category: string;
  points: { date: string; value: number }[];
  color: string;
}

interface TrendChartProps {
  series: TrendSeries[];
  yLabel: string;
  format: (v: number) => string;
  ols: boolean;            // extend each line with a 2-yr linear OLS projection
  activeCat?: string | null;
  onHover?: (cat: string | null) => void;
}

const W = 1000, H = 380, PAD_L = 84, PAD_R = 24, PAD_T = 18, PAD_B = 46;
const PROJ_DAYS = 730;
const MONTHS = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"];

function days(d: string): number { return new Date(d + "T00:00:00Z").getTime() / 86400000; }
function label(d: string): string { const [y, m] = d.split("-"); return `${MONTHS[Number(m) - 1] ?? m} ${y}`; }

function fit(pts: { x: number; y: number }[]): { b: number; a: number } {
  const n = pts.length;
  const sx = pts.reduce((s, p) => s + p.x, 0), sy = pts.reduce((s, p) => s + p.y, 0);
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0), sxy = pts.reduce((s, p) => s + p.x * p.y, 0);
  const denom = n * sxx - sx * sx;
  const b = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  return { b, a: (sy - b * sx) / n };
}

export default function TrendChart({ series, yLabel, format, ols, activeCat, onHover }: TrendChartProps) {
  const [hover, setHover] = useState<string | null>(null);
  const active = activeCat ?? hover;
  if (series.length === 0) return null;

  const allDates = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.date)))).sort();
  const d0 = days(allDates[0]);
  const dLast = days(allDates[allDates.length - 1]);
  const dMax = ols ? dLast + PROJ_DAYS : dLast;

  // tight y-range: min/max of all shown values (+ projection ends), padded a touch
  let vMin = Infinity, vMax = -Infinity;
  for (const s of series) {
    for (const p of s.points) { vMin = Math.min(vMin, p.value); vMax = Math.max(vMax, p.value); }
    if (ols && s.points.length >= 2) {
      const { b, a } = fit(s.points.map((p) => ({ x: days(p.date) - d0, y: p.value })));
      const ve = a + b * (dMax - d0); vMin = Math.min(vMin, ve); vMax = Math.max(vMax, ve);
    }
  }
  if (!isFinite(vMin)) { vMin = 0; vMax = 1; }
  const pad = (vMax - vMin) * 0.08 || Math.max(1, vMax * 0.05);
  const lo = vMin - pad, hi = vMax + pad;

  const xOf = (d: number) => PAD_L + ((d - d0) / (dMax - d0 || 1)) * (W - PAD_L - PAD_R);
  const yOf = (v: number) => PAD_T + (1 - (v - lo) / (hi - lo || 1)) * (H - PAD_T - PAD_B);
  const gridY = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {gridY.map((v, i) => (
        <g key={i}>
          <line x1={PAD_L} y1={yOf(v)} x2={W - PAD_R} y2={yOf(v)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PAD_L - 12} y={yOf(v) + 3} textAnchor="end" fontSize={11} fill="var(--text-muted)">{format(v)}</text>
        </g>
      ))}
      {allDates.map((d) => (
        <text key={d} x={xOf(days(d))} y={H - PAD_B + 18} textAnchor="middle" fontSize={10} fill="var(--text-muted)">{label(d)}</text>
      ))}
      <text x={20} y={H / 2} fontSize={11} fill="var(--text-secondary)" transform={`rotate(-90 20 ${H / 2})`} textAnchor="middle">{yLabel}</text>

      {series.map((s) => {
        const dim = active && active !== s.category;
        const pts = s.points.slice().sort((p, q) => days(p.date) - days(q.date));
        const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(days(p.date))} ${yOf(p.value)}`).join(" ");
        let proj = "";
        if (ols && pts.length >= 2) {
          const { b, a } = fit(pts.map((p) => ({ x: days(p.date) - d0, y: p.value })));
          const last = pts[pts.length - 1];
          proj = `M ${xOf(days(last.date))} ${yOf(last.value)} L ${xOf(dMax)} ${yOf(a + b * (dMax - d0))}`;
        }
        return (
          <g key={s.category} onMouseEnter={() => { setHover(s.category); onHover?.(s.category); }}
             onMouseLeave={() => { setHover(null); onHover?.(null); }}
             style={{ cursor: "pointer", opacity: dim ? 0.22 : 1, transition: "opacity 0.13s" }}>
            <path d={path} fill="none" stroke={s.color} strokeWidth={active === s.category ? 3 : 1.9} />
            {proj && <path d={proj} fill="none" stroke={s.color} strokeWidth={1.5} strokeDasharray="4 3" />}
            {pts.map((p) => <circle key={p.date} cx={xOf(days(p.date))} cy={yOf(p.value)} r={2.8} fill={s.color} />)}
          </g>
        );
      })}
    </svg>
  );
}
