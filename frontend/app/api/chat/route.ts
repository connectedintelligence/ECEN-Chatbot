/**
 * API route: POST /api/chat
 * Proxies the request to the FastAPI backend and streams the response back.
 * This avoids CORS issues and lets the frontend talk to /api/chat directly.
 */

import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

/**
 * The FastAPI backend loads ML models on startup (~60-90s after a cold start
 * or deploy). Until it binds its port, fetch fails with ECONNREFUSED. Retry
 * instead of surfacing an error to the user.
 */
async function fetchBackendWithRetry(body: unknown, clientIp: string): Promise<Response> {
  const maxAttempts = 30;
  for (let attempt = 1; ; attempt++) {
    try {
      return await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // Forward the real client IP (set by Cloud Run on the outer request)
          // so the backend's per-IP rate limiting sees users, not localhost.
          "x-forwarded-for": clientIp,
        },
        body: JSON.stringify(body),
      });
    } catch (err) {
      if (attempt >= maxAttempts) throw err;
      await new Promise(r => setTimeout(r, 3000));
    }
  }
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const clientIp = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";

  const upstream = await fetchBackendWithRetry(body, clientIp);

  if (!upstream.ok) {
    const err = await upstream.text();
    return new Response(err, { status: upstream.status });
  }

  // Stream the SSE response straight through
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
