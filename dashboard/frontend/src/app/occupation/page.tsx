"use client";

import { useEffect, useState } from "react";
import type { ConfigResponse } from "@/lib/types";
import { fetchConfig } from "@/lib/api";
import OccupationReport from "@/components/OccupationReport";

export default function MyOccupationPage() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [error,  setError]  = useState<string | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig).catch((e) => setError(e.message));
  }, []);

  if (error) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 56px)" }}>
      <p style={{ color: "#b91c1c" }}>Backend error: {error}</p>
    </div>
  );

  if (!config) return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "calc(100vh - 56px)", gap: 16 }}>
      <div style={{ width: 36, height: 36, borderRadius: "50%", border: "3px solid var(--brand)", borderTopColor: "transparent", animation: "spin 0.7s linear infinite" }} />
      <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Loading…</p>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  return <OccupationReport config={config} />;
}
