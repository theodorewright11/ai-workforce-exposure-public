"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/occupation", label: "My Occupation" },
  { href: "/data",       label: "Explore the Data" },
  { href: "/guide",      label: "Guide" },
];

export default function Navigation() {
  const pathname = usePathname();
  return (
    <nav className="nav-bar" style={{
      position: "fixed", top: 0, left: 0, right: 0, height: "var(--nav-height)", zIndex: 50,
      background: "linear-gradient(180deg, var(--bg-surface) 0%, #f8faf9 100%)",
      borderTop: "3px solid var(--brand)", borderBottom: "1px solid var(--border)",
      display: "flex", alignItems: "center", padding: "0 24px",
    }}>
      {/* left: brand (truncates) */}
      <Link href="/occupation" style={{ textDecoration: "none", flex: "1 1 0", minWidth: 0, display: "flex", alignItems: "center", gap: 12 }}>
        <div className="nav-brand-accent" style={{ width: 3, height: 34, borderRadius: 2, backgroundColor: "var(--brand)", opacity: 0.75, flexShrink: 0 }} />
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", minWidth: 0 }}>
          <span className="nav-brand-title" style={{ fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em", lineHeight: 1.2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            Mapping AI Exposure Across the U.S. Workforce
          </span>
          <span className="nav-brand-sub" style={{ fontWeight: 500, color: "var(--text-muted)", letterSpacing: "0.02em", lineHeight: 1.3, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            Evidence from Millions of AI Conversations
          </span>
        </div>
      </Link>

      {/* center: links */}
      <div className="nav-links" style={{ display: "flex", alignItems: "center", gap: 8, flex: "0 0 auto" }}>
        {NAV_LINKS.map(({ href, label }) => {
          const active = pathname === href || pathname.startsWith(href);
          return (
            <Link key={href} href={href} className="nav-link" style={{
              borderRadius: 7, fontWeight: active ? 600 : 450,
              color: active ? "var(--brand)" : "var(--text-secondary)",
              backgroundColor: active ? "var(--brand-light)" : "transparent",
              textDecoration: "none", transition: "all 0.13s", whiteSpace: "nowrap",
              borderBottom: active ? "2px solid var(--brand)" : "2px solid transparent",
            }}>{label}</Link>
          );
        })}
      </div>

      {/* right: spacer to keep links centered */}
      <div style={{ flex: "1 1 0" }} />
    </nav>
  );
}
