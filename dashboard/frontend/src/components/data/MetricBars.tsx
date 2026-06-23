"use client";

import { useState } from "react";
import { ordinal } from "@/lib/format";

export interface BarDatum {
  category: string;
  value: number;
  rank: number;
  total: number;          // economy share denominator (sum across all categories)
  display: string;        // formatted value label
}

interface MetricBarsProps {
  title: string;
  rows: BarDatum[];        // already ordered by the page's sort metric
  color: string;          // bar fill
  totalCategories: number; // economy size for "rank/N"
  canDrill: boolean;
  onDrill?: (category: string) => void;
  searchMatch?: string;   // category to highlight
  showShare?: boolean;    // show "% of economy" in tooltip (meaningless for a ratio metric)
}

/** One metric column: horizontal bars with hover tooltip (rank + economy share)
 *  and optional click-to-drill. Bars scale to the max value shown. */
export default function MetricBars({
  title, rows, color, totalCategories, canDrill, onDrill, searchMatch, showShare = true,
}: MetricBarsProps) {
  const [hover, setHover] = useState<string | null>(null);
  const max = Math.max(1, ...rows.map((r) => r.value));

  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{
        fontSize: 12, fontWeight: 600, color: "var(--text-secondary)",
        marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.04em",
      }}>
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {rows.map((r) => {
          const pctOfMax = (r.value / max) * 100;
          const share = r.total > 0 ? (r.value / r.total) * 100 : 0;
          const isHover = hover === r.category;
          const isMatch = searchMatch && r.category === searchMatch;
          return (
            <div
              key={r.category}
              onMouseEnter={() => setHover(r.category)}
              onMouseLeave={() => setHover(null)}
              onClick={() => canDrill && onDrill?.(r.category)}
              style={{
                position: "relative",
                cursor: canDrill ? "pointer" : "default",
              }}
            >
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: 11, marginBottom: 2,
                color: isMatch ? "var(--brand)" : "var(--text-secondary)",
                fontWeight: isMatch ? 600 : 400,
              }}>
                <span style={{
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  maxWidth: "70%",
                }} title={r.category}>
                  {canDrill && <span style={{ opacity: 0.5, marginRight: 4 }}>▸</span>}
                  {r.category}
                </span>
                <span style={{ fontWeight: 600, color: "var(--text-primary)", flexShrink: 0 }}>
                  {r.display}
                </span>
              </div>
              <div style={{ height: 10, background: "var(--bg-sidebar)", borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  width: `${pctOfMax}%`, height: "100%",
                  background: isMatch ? "var(--brand)" : color,
                  opacity: isHover ? 1 : 0.85,
                  transition: "width 0.3s, opacity 0.13s",
                }} />
              </div>
              {isHover && (
                <div style={{
                  position: "absolute", top: "100%", left: 0, zIndex: 20,
                  marginTop: 3, padding: "6px 9px",
                  background: "var(--bg-surface)", border: "1px solid var(--border)",
                  borderRadius: 6, boxShadow: "0 4px 14px rgba(0,0,0,0.12)",
                  fontSize: 10.5, lineHeight: 1.5, color: "var(--text-secondary)",
                  whiteSpace: "nowrap",
                }}>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{r.display}</div>
                  <div>Rank {ordinal(r.rank)} of {totalCategories}</div>
                  {showShare && <div>{share.toFixed(1)}% of economy total</div>}
                  {canDrill && <div style={{ color: "var(--brand)", marginTop: 2 }}>click to drill down →</div>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
