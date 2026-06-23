"use client";

/* Guide — About the project, how to read the charts, and a link to the paper. */

const SECTION: React.CSSProperties = {
  background: "var(--bg-surface)", border: "1px solid var(--border)",
  borderRadius: 10, padding: "20px 24px", marginBottom: 18,
};
const H2: React.CSSProperties = { fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 };
const P: React.CSSProperties = { fontSize: 13.5, color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: 10 };

export default function GuidePage() {
  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: "28px 24px 60px" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--text-primary)", marginBottom: 18 }}>Guide</h1>

      <div style={SECTION}>
        <div style={H2}>About</div>
        <p style={P}>
          This dashboard is the interactive companion to the paper <strong>Mapping AI Exposure
          Across the U.S. Workforce: Evidence from Millions of AI Conversations</strong>. It measures
          how current AI capability maps onto the U.S. workforce, combining real-world AI usage
          (Anthropic&rsquo;s Claude, Microsoft&rsquo;s Copilot), agentic AI MCP-server data, workforce
          data from O*NET, and BLS employment &amp; wage data.
        </p>
        <p style={{ ...P, marginBottom: 0, color: "var(--text-muted)", fontSize: 12.5 }}>
          A project of Utah&rsquo;s Office of AI Policy (OAIP), supported by the BYU Department of Mathematics.
        </p>
      </div>

      <div style={SECTION}>
        <div style={H2}>How to read the charts</div>
        <p style={{ ...P, marginBottom: 0, color: "var(--text-muted)" }}>TBA</p>
      </div>

      <div style={SECTION}>
        <div style={H2}>The paper</div>
        <p style={P}>
          Full methodology, data construction, and results are in the paper and its Supplementary
          Materials.
        </p>
        <p style={{ ...P, marginBottom: 0 }}>
          <strong style={{ color: "var(--brand)" }}>Read the paper →</strong>{" "}
          <span style={{ color: "var(--text-muted)" }}>Coming soon</span>
        </p>
      </div>
    </div>
  );
}
