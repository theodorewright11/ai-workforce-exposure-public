"use client";

const GITHUB_ICON = (
  <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
  </svg>
);

const LINK_STYLE: React.CSSProperties = {
  color: "var(--text-muted)",
  textDecoration: "none",
  fontSize: 11,
  display: "inline-flex",
  alignItems: "center",
  gap: 4,
  transition: "color 0.15s",
};

export default function Footer() {
  return (
    <footer
      style={{
        borderTop: "2px solid var(--border)",
        padding: "18px 24px",
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px 20px",
        fontSize: 11,
        color: "var(--text-muted)",
        backgroundColor: "#f2f2ef",
      }}
    >
      <span>Source: 2025 O*NET &middot; 2025 BLS OEWS &middot; Anthropic Economic Index &middot; Microsoft Copilot &middot; MCP Server Classification</span>
      <span style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <a
          href="https://github.com/theodorewright11/ai-workforce-exposure-public"
          target="_blank"
          rel="noopener noreferrer"
          style={LINK_STYLE}
        >
          {GITHUB_ICON} Dashboard &amp; Paper Figures GitHub
        </a>
        <a
          href="https://github.com/theodorewright11/mcp-onet-task-classification-public"
          target="_blank"
          rel="noopener noreferrer"
          style={LINK_STYLE}
        >
          {GITHUB_ICON} MCP Data GitHub
        </a>
        <a
          href="https://github.com/theodorewright11/ai-workforce-exposure-dataset-construction-public"
          target="_blank"
          rel="noopener noreferrer"
          style={LINK_STYLE}
        >
          {GITHUB_ICON} Dataset Construction GitHub
        </a>
        <a href="mailto:theodorewrightwork@gmail.com" style={LINK_STYLE}>
          <svg width="13" height="13" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
            <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
          </svg>
          Contact
        </a>
      </span>
    </footer>
  );
}
