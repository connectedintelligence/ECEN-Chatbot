/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    BACKEND_URL: process.env.BACKEND_URL || "http://localhost:8000",
  },
};

// Wrap with Sentry so build-time source maps upload and errors are tied to
// releases. The wrapper is harmless when SENTRY_* env vars are unset.
const { withSentryConfig } = require("@sentry/nextjs");

module.exports = withSentryConfig(nextConfig, {
  org: process.env.SENTRY_ORG,
  project: process.env.SENTRY_PROJECT,
  // Auth token (CI) enables source map upload: SENTRY_AUTH_TOKEN.
  silent: !process.env.CI,
  widenClientFileUpload: true,
  disableLogger: true,
});
