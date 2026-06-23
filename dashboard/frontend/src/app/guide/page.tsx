"use client";

/* Guide — About the project, how to read the charts, and a link to the paper.
   Placeholder copy for now; we'll fill it in once the rest is settled. */

const SECTION: React.CSSProperties = {
  background: "var(--bg-surface)", border: "1px solid var(--border)",
  borderRadius: 10, padding: "20px 24px", marginBottom: 18,
};
const H2: React.CSSProperties = { fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 };
const P: React.CSSProperties = { fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: 10 };

export default function GuidePage() {
  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: "28px 24px 60px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--text-primary)", marginBottom: 6 }}>
        Guide
      </h1>
      <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 22 }}>
        What this is, how to read the charts, and where to find the paper. <em>(Placeholder — copy to be finalized.)</em>
      </p>

      <div style={SECTION}>
        <div style={H2}>About</div>
        <p style={P}>
          This dashboard is the interactive companion to <strong>Mapping AI Exposure Across the
          U.S. Workforce: Evidence from Millions of AI Conversations</strong>. It measures how
          current AI capability maps onto the U.S. occupational task structure, combining real-world
          AI usage (Anthropic&rsquo;s Claude, Microsoft&rsquo;s Copilot), an MCP-server capability
          pipeline, occupation structure from O*NET, and BLS employment &amp; wage data.
        </p>
        <p style={P}>[Placeholder: short project description, who it&rsquo;s for, and the OAIP attribution.]</p>
      </div>

      <div style={SECTION}>
        <div style={H2}>How to read the charts</div>
        <p style={P}>
          <strong>Exposure</strong> is task-level overlap with what AI can currently do — an upper
          bound, not a forecast of job loss. <strong>% Tasks Exposed</strong> is a ratio of totals;
          <strong> Workers</strong> and <strong>Wages Exposed</strong> translate that into people
          and dollars. <strong>Actual AI usage</strong> shows where AI is currently being used,
          relative to the median, corrected for user-base bias and task size.
        </p>
        <p style={P}>[Placeholder: per-view reading notes — configurations, hierarchy drill-down, trends, the usage intensity multiplier.]</p>
      </div>

      <div style={SECTION}>
        <div style={H2}>The paper</div>
        <p style={P}>
          Full methodology, data construction, and results are in the paper and its Supplementary
          Materials.
        </p>
        <p style={P}>
          <a href="#" style={{ color: "var(--brand)", fontWeight: 600 }}>
            Read the paper →
          </a>{" "}
          <span style={{ color: "var(--text-muted)" }}>[link TBD before public release]</span>
        </p>
      </div>
    </div>
  );
}
