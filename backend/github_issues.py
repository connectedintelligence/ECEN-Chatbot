"""
github_issues.py — create GitHub issues from in-app bug reports.

Lets non-technical testers report problems straight from the chat UI; the backend
files the issue on their behalf so they never need a GitHub account. The Codex
triage workflow then picks the issue up.

Env vars:
  GH_ISSUE_TOKEN — a token with `issues: write` on the repo (fine-grained PAT or
                   a GitHub App installation token). REQUIRED to enable reporting.
  GH_REPO        — "owner/repo", e.g. "Aa-Rho-Hi/ECEN-Chatbot".
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class ReportingDisabled(RuntimeError):
    """Raised when GH_ISSUE_TOKEN / GH_REPO aren't configured."""


def reporting_enabled() -> bool:
    return bool(os.getenv("GH_ISSUE_TOKEN") and os.getenv("GH_REPO"))


def create_issue(title: str, body: str, labels: list[str] | None = None) -> dict:
    """Create a GitHub issue. Returns {number, html_url}. Raises on failure."""
    token = os.getenv("GH_ISSUE_TOKEN", "").strip()
    repo = os.getenv("GH_REPO", "").strip()
    if not token or not repo:
        raise ReportingDisabled("GH_ISSUE_TOKEN or GH_REPO not set")

    resp = httpx.post(
        f"{GITHUB_API}/repos/{repo}/issues",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"title": title, "body": body, "labels": labels or []},
        timeout=10.0,
    )
    if resp.status_code >= 300:
        log.error("GitHub issue creation failed: %s %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
    data = resp.json()
    return {"number": data["number"], "html_url": data["html_url"]}
