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
}

const W = 760, H = 360, PAD_L = 64, PAD_R = 130, PAD_T = 16, PAD_B = 44;
const PROJ_DAYS = 730;

function days(d: string): number {
  return new Date(d + "T00:00:00Z").getTime() / 86400000;
}

/** Simple OLS slope/intercept of value over days-since-first. */
function fit(pts: { x: number; y: number }[]): { b: number; a: number } {
  const n = pts.length;
  const sx = pts.reduce((s, p) => s + p.x, 0);
  const sy = pts.reduce((s, p) => s + p.y, 0);
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0);
  const sxy = pts.reduce((s, p) => s + p.x * p.y, 0);
  const denom = n * sxx - sx * sx;
  const b = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / n;
  return { b, a };
}

export default function TrendChart({ series, yLabel, format, ols }: TrendChartProps) {
  const [active, setActive] = useState<string | null>(null);
  if (series.length === 0) return null;

  const allDates = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.date)))).sort();
  const d0 = days(allDates[0]);
  const dLast = days(allDates[allDates.length - 1]);
  const dMax = ols ? dLast + PROJ_DAYS : dLast;

  let vMax = 0;
  for (const s of series) {
    for (const p of s.points) vMax = Math.max(vMax, p.value);
    if (ols && s.points.length >= 2) {
      const { b, a } = fit(s.points.map((p) => ({ x: days(p.date) - d0, y: p.value })));
      vMax = Math.max(vMax, a + b * (dMax - d0));
    }
  }
  vMax = vMax * 1.08 || 1;

  const xOf = (d: number) => PAD_L + ((d - d0) / (dMax - d0 || 1)) * (W - PAD_L - PAD_R);
  const yOf = (v: number) => PAD_T + (1 - v / vMax) * (H - PAD_T - PAD_B);

  // y gridlines
  const ticks = 5;
  const gridY = Array.from({ length: ticks + 1 }, (_, i) => (vMax * i) / ticks);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 900 }}>
      {/* gridlines + y labels */}
      {gridY.map((v, i) => (
        <g key={i}>
          <line x1={PAD_L} y1={yOf(v)} x2={W - PAD_R} y2={yOf(v)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PAD_L - 8} y={yOf(v) + 3} textAnchor="end" fontSize={10} fill="var(--text-muted)">
            {format(v)}
          </text>
        </g>
      ))}
      {/* x labels */}
      {allDates.map((d) => (
        <text key={d} x={xOf(days(d))} y={H - PAD_B + 16} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
          {d.slice(0, 7)}
        </text>
      ))}
      <text x={14} y={H / 2} fontSize={10} fill="var(--text-secondary)"
        transform={`rotate(-90 14 ${H / 2})`} textAnchor="middle">{yLabel}</text>

      {/* lines */}
      {series.map((s) => {
        const dim = active && active !== s.category;
        const pts = s.points.slice().sort((p, q) => days(p.date) - days(q.date));
        const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(days(p.date))} ${yOf(p.value)}`).join(" ");
        let proj = "";
        let projEnd: { x: number; y: number; v: number } | null = null;
        if (ols && pts.length >= 2) {
          const { b, a } = fit(pts.map((p) => ({ x: days(p.date) - d0, y: p.value })));
          const last = pts[pts.length - 1];
          const vEnd = a + b * (dMax - d0);
          proj = `M ${xOf(days(last.date))} ${yOf(last.value)} L ${xOf(dMax)} ${yOf(vEnd)}`;
          projEnd = { x: xOf(dMax), y: yOf(vEnd), v: vEnd };
        }
        return (
          <g key={s.category}
             onMouseEnter={() => setActive(s.category)}
             onMouseLeave={() => setActive(null)}
             style={{ cursor: "pointer", opacity: dim ? 0.25 : 1, transition: "opacity 0.13s" }}>
            <path d={path} fill="none" stroke={s.color} strokeWidth={active === s.category ? 2.6 : 1.8} />
            {proj && <path d={proj} fill="none" stroke={s.color} strokeWidth={1.4} strokeDasharray="4 3" />}
            {pts.map((p) => (
              <circle key={p.date} cx={xOf(days(p.date))} cy={yOf(p.value)} r={2.6} fill={s.color} />
            ))}
            {projEnd && (
              <text x={projEnd.x + 4} y={projEnd.y + 3} fontSize={9} fill={s.color}>
                {format(projEnd.v)}
              </text>
            )}
          </g>
        );
      })}

      {/* legend */}
      {series.map((s, i) => (
        <g key={s.category}
           onMouseEnter={() => setActive(s.category)}
           onMouseLeave={() => setActive(null)}
           style={{ cursor: "pointer" }}
           transform={`translate(${W - PAD_R + 8}, ${PAD_T + 4 + i * 15})`}>
          <rect width={9} height={9} rx={2} fill={s.color} y={-7} />
          <text x={13} fontSize={9.5} fill="var(--text-secondary)">
            {s.category.length > 22 ? s.category.slice(0, 21) + "…" : s.category}
          </text>
        </g>
      ))}
    </svg>
  );
}
