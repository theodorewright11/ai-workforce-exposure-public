"use client";

/* My Occupation — the landing page.
 *
 * One control, so there is no config to fetch before the page can render:
 * the card owns its own occupation list and loads straight into a default.
 * (PRD §1.3, control budget.)
 */

import OccupationReport from "@/components/OccupationReport";

export default function MyOccupationPage() {
  return <OccupationReport />;
}
