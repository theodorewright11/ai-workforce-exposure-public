import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "AI Workforce Exposure",
  description:
    "Explore AI exposure across U.S. occupations, work activities, and actual usage — companion to Mapping AI Exposure Across the U.S. Workforce.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.className}>
      <body className="antialiased" style={{ backgroundColor: "var(--bg-base)", color: "var(--text-primary)" }}>
        <Navigation />
        {/* Content sits below the fixed nav */}
        <div style={{ paddingTop: "var(--nav-height)", minHeight: "calc(100vh - 60px)" }}>
          <div style={{ background: "#fbf3e2", borderBottom: "1px solid #efe3c4", color: "#7a5e22", textAlign: "center", padding: "10px 16px", lineHeight: 1.5 }}>
            <div style={{ fontSize: 12.5 }}>
              Interactive companion dashboard to the paper{" "}
              <strong style={{ color: "#5a4416" }}>
                &ldquo;Mapping AI Exposure Across the U.S. Workforce: Evidence from Millions of AI Conversations&rdquo;
              </strong>
            </div>
            <div style={{ fontSize: 12.5, marginTop: 2 }}>
              Full paper coming soon. Follow updates on{" "}
              <a href="https://github.com/theodorewright11/ai-workforce-exposure-public" target="_blank" rel="noreferrer" style={{ color: "#8a5a1a", fontWeight: 600, textDecoration: "underline" }}>GitHub</a>{" "}
              or the{" "}
              <a href="https://commerce.utah.gov/ai/" target="_blank" rel="noreferrer" style={{ color: "#8a5a1a", fontWeight: 600, textDecoration: "underline" }}>OAIP website</a>.
            </div>
            <div style={{ fontSize: 11, marginTop: 3, opacity: 0.8 }}>
              Work in progress, might be some minor bugs. Best viewed on a computer.
            </div>
          </div>
          {children}
        </div>
        <Footer />
      </body>
    </html>
  );
}
