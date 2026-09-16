import { NextRequest, NextResponse } from "next/server";

// The custom domain shows only the placeholder until launch; the
// *.vercel.app URL keeps serving the full dashboard.
const PLACEHOLDER_HOSTS = new Set(["aiworkforceexposure.com", "www.aiworkforceexposure.com"]);

export function middleware(req: NextRequest) {
  const host = (req.headers.get("host") ?? "").split(":")[0].toLowerCase();
  if (!PLACEHOLDER_HOSTS.has(host) || req.nextUrl.pathname === "/coming-soon") return;
  const url = req.nextUrl.clone();
  url.pathname = "/coming-soon";
  url.search = "";
  return NextResponse.rewrite(url);
}

export const config = {
  matcher: ["/((?!_next/|favicon.ico).*)"],
};
