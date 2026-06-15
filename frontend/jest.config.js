// Plain JS config so Jest needs no ts-node to parse it (a .ts config requires
// ts-node, which isn't a declared dependency — see issue #15 post-mortem).
const nextJest = require("next/jest.js");

const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jest-environment-jsdom",
  // NOTE: the option is `setupFilesAfterEnv`, NOT `setupFilesAfterFramework`.
  // The wrong key silently skips jest.setup.ts, so @testing-library/jest-dom
  // matchers (toBeDisabled, toBeInTheDocument, ...) never load and any
  // component test fails. This was the root cause that blocked the #15 fix.
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
};

module.exports = createJestConfig(config);
