/**
 * API route: POST /api/report
 * Proxies an in-app bug report to the FastAPI backend, which files a GitHub issue.
 */

import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const clientIp = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";

  const upstream = await fetch(`${BACKEND_URL}/report-issue`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-forwarded-for": clientIp },
    body: JSON.stringify(body),
  });

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
