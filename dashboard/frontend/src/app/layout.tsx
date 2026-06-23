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
          {children}
        </div>
        <Footer />
      </body>
    </html>
  );
}
