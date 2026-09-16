"use client";

/* Placeholder for the custom domain (see middleware.ts): the live occupation
 * card, grayed out and inert, under a "coming soon" notice. Fixed and above
 * the nav so the shared layout chrome stays hidden and nothing scrolls.
 */

import OccupationReport from "@/components/OccupationReport";

export default function ComingSoonPage() {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, overflow: "hidden", background: "var(--bg-base)" }}>
      <div
        aria-hidden
        style={{ filter: "grayscale(1) blur(1.5px)", opacity: 0.45, pointerEvents: "none", userSelect: "none" }}
      >
        <OccupationReport />
      </div>

      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", padding: 16, background: "rgba(255,255,255,0.35)" }}>
        <div style={{ background: "#fff", border: "1px solid #e2e5e9", borderRadius: 12, boxShadow: "0 12px 40px rgba(0,0,0,0.12)", padding: "36px 32px", maxWidth: 480, textAlign: "center" }}>
          <div style={{ fontSize: 12, letterSpacing: "0.12em", textTransform: "uppercase", color: "#6b7280", fontWeight: 600 }}>
            AI Workforce Exposure
          </div>
          <h1 style={{ fontSize: 34, fontWeight: 700, margin: "10px 0 12px", color: "#1f2937" }}>Coming soon</h1>
          <p style={{ fontSize: 15, lineHeight: 1.6, color: "#4b5563", margin: 0 }}>
            An interactive look at how AI is being used across U.S. occupations, from the{" "}
            <a href="https://commerce.utah.gov/ai/" target="_blank" rel="noreferrer" style={{ color: "#3a5f83", fontWeight: 600, textDecoration: "underline" }}>
              Utah Office of Artificial Intelligence Policy
            </a>.
          </p>
        </div>
      </div>
    </div>
  );
}
