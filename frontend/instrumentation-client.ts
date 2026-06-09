// Sentry init for the browser. Captures unhandled client-side errors in the
// chat UI. Uses the public DSN (safe to expose). No-op unless the DSN is set.
import * as Sentry from "@sentry/nextjs";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    environment: process.env.NEXT_PUBLIC_SENTRY_ENVIRONMENT || process.env.NODE_ENV,
    tracesSampleRate: Number(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE || 0),
  });
}

// Note: Sentry.captureRouterTransitionStart (navigation-tracing hook) exists only
// in newer @sentry/nextjs (v8.42+/v9). Omitted here for version compatibility;
// re-add `export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;`
// if you upgrade and want client-side navigation spans.
