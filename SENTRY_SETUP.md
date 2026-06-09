# Sentry → GitHub Issue → Codex Fix

This connects runtime errors (from the chat UI **and** the FastAPI backend) to the
agentic fix loop. The chain:

```
User hits an error in the UI / backend throws
        │
        ▼  Sentry SDK (frontend @sentry/nextjs, backend sentry-sdk[fastapi])
   Sentry captures the error + stack trace, groups duplicates into one issue
        │
        ▼  Sentry Alert rule: "Create a new GitHub issue" (label: sentry)
   A GitHub issue appears in Aa-Rho-Hi/ECEN-Chatbot with the stack trace
        │
        ▼  .github/workflows/codex.yml  (triggers on the `sentry` label)
   Codex verifies against the code → opens a fix PR  codex/fix-issue-<n>
        │
        ▼
   You review + merge.   ← human gate, nothing auto-merges
```

Sentry's free Developer plan is enough for this. (Seer Autofix, Sentry's own
PR-writing AI, needs a paid plan — we deliberately use your Codex loop instead.)

## What's already wired in the code
- **Backend:** `backend/sentry_init.py` + an early `init_sentry()` call in
  `backend/main.py`. No-op unless `SENTRY_DSN` is set. `sentry-sdk[fastapi]` added
  to `backend/requirements.txt`.
- **Frontend:** `@sentry/nextjs` added to `package.json`; `instrumentation.ts`,
  `instrumentation-client.ts`, `sentry.server.config.ts`, `sentry.edge.config.ts`,
  `app/global-error.tsx`, and `next.config.js` wrapped with `withSentryConfig`.
  No-op unless `NEXT_PUBLIC_SENTRY_DSN` is set.
- **Workflow:** `codex.yml` now also triggers on issues labeled `sentry`.

## Setup steps (Sentry side)

1. **Create two Sentry projects** at https://sentry.io (free): one **Next.js**
   (frontend) and one **Python/FastAPI** (backend). Copy each project's **DSN**.

2. **Set the DSNs as env vars** (NOT committed — `.env` is git-ignored):
   - Backend `.env`: `SENTRY_DSN=<python-project-dsn>`
   - Frontend env: `NEXT_PUBLIC_SENTRY_DSN=<nextjs-project-dsn>` (set in your
     host's env / `.env.local` for Next.js).

3. **Install deps** (once):
   - Backend: `pip install -r backend/requirements.txt`
   - Frontend: `cd frontend && npm install`
   - Reliable alternative for the frontend files: run the official wizard, which
     matches your exact SDK version and sets up source-map upload —
     `npx @sentry/wizard@latest -i nextjs` (it will reconcile the files above).

4. **Connect the Sentry ↔ GitHub integration:** Sentry → Settings → Integrations →
   **GitHub** → install on `Aa-Rho-Hi/ECEN-Chatbot`. (Required so Sentry can open
   issues in the repo.)

5. **Create the alert rule that opens a labeled GitHub issue:**
   Sentry → Alerts → Create Alert → **Issues** →
   - When: **A new issue is created** (keep it to *new* issues to avoid noise).
   - Optional filter: level = error, or "event count > N in 1h" to suppress one-offs.
   - Then: **Create a new GitHub issue** →
     - Integration: your GitHub install
     - Repository: `ECEN-Chatbot`
     - **Labels: `sentry`**  ← this is what triggers `codex.yml`
   - Save. (Make sure the `sentry` label exists in the repo, or let Sentry create it.)

6. **Test:** throw a deliberate error (e.g. a temporary route that raises, or a
   thrown error in the UI). Within a minute Sentry creates a `sentry`-labeled
   GitHub issue, `codex.yml` fires, and Codex opens a fix PR for your review.

## Notes
- **Noise control:** Sentry groups repeats into one issue, so you get one GitHub
  issue per distinct bug, not per occurrence. Tighten the alert filter if needed.
- **Privacy:** backend init sets `send_default_pii=False`, so user questions
  aren't sent to Sentry. Adjust in `backend/sentry_init.py` if you want more/less.
- **Bidirectional sync (optional):** the Sentry GitHub integration can resolve the
  Sentry issue when you close/merge the GitHub issue/PR.
- Everything is gated on DSNs, so with no DSN set the app behaves exactly as before.
