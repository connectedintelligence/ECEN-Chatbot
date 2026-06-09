# Agentic Issue → Fix → PR Workflow (OpenAI Codex)

This repo uses the official [OpenAI Codex GitHub Action](https://github.com/openai/codex-action)
to turn bug reports into reviewed fixes. **No fix is ever merged automatically —
a human approves and merges every PR.**

## The loop

```
Tester finds a bug
      │
      ▼
Opens a GitHub Issue (Bug report template) with "@codex" in the body
      │
      ▼  (.github/workflows/codex.yml — job 1 "codex")
Codex VERIFIES the issue against the code  [has OPENAI_API_KEY, contents: read only]
      │
      ├─ not a real bug ─▶ makes no edits, emits an explanation
      │
      └─ confirmed ─▶ edits files in the workspace, emits a diff (artifact)
      │
      ▼  (.github/workflows/codex.yml — job 2 "open_pr")  [write access, NO api key]
Applies the diff → opens PR  codex/fix-issue-<n>  → comments result on the issue
      │
      ▼
┌─────────────────────────────┐
│  HUMAN reviews + clicks Merge │  ◀── the only merge gate
└─────────────────────────────┘
```

The two jobs are deliberately separated: job 1 holds the `OPENAI_API_KEY` but
can't push; job 2 can push but never sees the key. This is OpenAI's recommended
pattern so a malicious issue body can't both reach the key and write code.

## One-time setup

1. **Connect Codex to GitHub / install the Codex GitHub app** on
   `Aa-Rho-Hi/ECEN-Chatbot` (https://github.com/apps/codex, or from the Codex
   settings at https://chatgpt.com/codex). This lets Codex operate on the repo.

2. **Add the API key secret.** Repo → Settings → Secrets and variables → Actions →
   New repository secret:
   - Name: `OPENAI_API_KEY`
   - Value: your OpenAI API key from https://platform.openai.com/api-keys
   - The key's OpenAI account needs available credit/billing enabled.

3. **Protect `main` so nothing merges without you.** Repo → Settings → Branches →
   add a rule for `main`:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1)
   - ✅ Do not allow bypassing the above settings

## Files in this setup
| File | Role |
|---|---|
| `.github/workflows/codex.yml` | Triggers on issues / `@codex`; verifies + opens fix PRs |
| `.github/workflows/codex-review.yml` | Codex PR review (read-only comment) |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Structured bug report that feeds the loop |
| `AGENTS.md` | Project context + verify notes Codex reads automatically |

## Good to know
- **Trigger phrase is `@codex`** (configurable in the `if:` block of `codex.yml`).
- **Sandbox has no network.** Codex can read code and run `py_compile`, but
  `npm install` / DB connections won't work in CI — it validates by reasoning.
- **The review workflow won't run on Codex's own PRs.** GitHub doesn't re-trigger
  workflows for PRs opened with the default `GITHUB_TOKEN`. It runs on human PRs;
  to also review Codex PRs, open them with a Personal Access Token instead.
- **Cost:** runs bill your OpenAI key per invocation; only issues containing
  `@codex` trigger a run.

## Security notes
- `.env` and all secrets are git-ignored — never commit them.
- Each workflow grants only the GitHub permissions in its `permissions:` block,
  and branch protection prevents merging to `main` without your approval.

---
_Switched from the Claude Code Action to Codex on 2026-06-08 to use an OpenAI
API key. The Claude workflow files were removed; see git history if you want to
switch back._
