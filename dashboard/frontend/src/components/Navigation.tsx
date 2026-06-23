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
    <nav
      style={{
        position: "fixed",
        top: 0, left: 0, right: 0,
        height: "var(--nav-height)",
        zIndex: 50,
        background: "linear-gradient(180deg, var(--bg-surface) 0%, #f8faf9 100%)",
        borderTop: "3px solid var(--brand)",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        padding: "0 32px",
      }}
    >
      {/* Brand — full paper title */}
      <Link href="/occupation" style={{ textDecoration: "none", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 3, height: 34, borderRadius: 2, backgroundColor: "var(--brand)", opacity: 0.75 }} />
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em", lineHeight: 1.2 }}>
              Mapping AI Exposure Across the U.S. Workforce
            </span>
            <span style={{ fontSize: 11, fontWeight: 500, color: "var(--text-muted)", letterSpacing: "0.02em", lineHeight: 1.3 }}>
              Evidence from Millions of AI Conversations
            </span>
          </div>
        </div>
      </Link>

      {/* Links — pushed to the right, spaced out for a site feel */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: "auto" }}>
        {NAV_LINKS.map(({ href, label }) => {
          const active = pathname === href || pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              style={{
                padding: "8px 18px",
                borderRadius: 7,
                fontSize: 14,
                fontWeight: active ? 600 : 450,
                color: active ? "var(--brand)" : "var(--text-secondary)",
                backgroundColor: active ? "var(--brand-light)" : "transparent",
                textDecoration: "none",
                transition: "all 0.13s",
                whiteSpace: "nowrap",
                borderBottom: active ? "2px solid var(--brand)" : "2px solid transparent",
              }}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
