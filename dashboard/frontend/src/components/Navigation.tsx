"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/occupation", label: "My Occupation" },
  { href: "/data",       label: "Explore the Data" },
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
        padding: "0 20px",
      }}
    >
      {/* Brand */}
      <Link href="/occupation" style={{ textDecoration: "none" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginRight: 28, flexShrink: 0 }}>
          <div style={{ width: 3, height: 28, borderRadius: 2, backgroundColor: "var(--brand)", opacity: 0.7 }} />
          <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em", lineHeight: 1.25 }}>
              AI Workforce Exposure
            </span>
            <span style={{ fontSize: 10, fontWeight: 500, color: "var(--text-muted)", letterSpacing: "0.05em", textTransform: "uppercase", lineHeight: 1.3 }}>
              Mapping AI Across the U.S. Workforce
            </span>
          </div>
        </div>
      </Link>

      {/* Links */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        {NAV_LINKS.map(({ href, label }) => {
          const active = pathname === href || pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              style={{
                padding: "6px 14px",
                borderRadius: 6,
                fontSize: 13,
                fontWeight: active ? 600 : 400,
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
