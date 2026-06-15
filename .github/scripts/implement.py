#!/usr/bin/env python3
"""
implement.py — Called by codex.yml to generate and apply a fix.
Reads the issue + all prior triage comments, calls the OpenAI API,
writes the fix AND a regression test, then the workflow gates on both
test suites passing before opening a PR.
"""

import json
import os
import sys
import urllib.request
from openai import OpenAI

SOURCE_FILES = [
    # Backend
    "backend/generator.py",
    "backend/retriever.py",
    "backend/main.py",
    "backend/graph_retriever.py",
    # Crawler
    "crawler/crawler.py",
    "crawler/chunker.py",
    "crawler/ingest.py",
    # Frontend
    "frontend/components/ChatUI.tsx",
    "frontend/lib/parseAnswer.ts",
    "frontend/app/api/chat/route.ts",
]

TEST_FILES = [
    # Existing tests — Codex must not break these
    "tests/test_generator_prompt.py",
    "frontend/__tests__/parseAnswer.test.ts",
]


def read_files(paths: list[str]) -> str:
    parts = []
    for path in paths:
        try:
            with open(path) as f:
                parts.append(f"=== {path} ===\n{f.read()}")
        except FileNotFoundError:
            pass
    return "\n\n".join(parts)


def fetch_issue_comments(repo: str, issue_number: str, token: str) -> str:
    """Fetch all comments on the issue so the triage plan is visible to the model."""
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            comments = json.loads(resp.read())
        parts = []
        for c in comments:
            author = c.get("user", {}).get("login", "unknown")
            body = c.get("body", "").strip()
            parts.append(f"[{author}]: {body}")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"Warning: could not fetch issue comments: {e}")
        return ""


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    issue_number = os.environ.get("ISSUE_NUMBER", "?")
    issue_title  = os.environ.get("ISSUE_TITLE", "")
    issue_body   = os.environ.get("ISSUE_BODY", "")
    comment_body = os.environ.get("COMMENT_BODY", "")
    repo         = os.environ.get("GITHUB_REPOSITORY", "")
    token        = os.environ.get("GITHUB_TOKEN", "")

    files_content = read_files(SOURCE_FILES)
    tests_content = read_files(TEST_FILES)

    # Fetch triage comments so the detailed fix plan reaches the model
    prior_comments = ""
    if repo and token and issue_number != "?":
        prior_comments = fetch_issue_comments(repo, issue_number, token)

    triage_section = (
        f"\nPrior issue comments (including triage plan — follow this plan):\n{prior_comments}\n"
        if prior_comments else ""
    )

    prompt = f"""You are a senior software engineer fixing a bug in the TAMU ECE RAG chatbot \
(FastAPI/Python backend + Next.js/TypeScript frontend). You write tests first, then the fix.

Issue #{issue_number}: {issue_title}

Issue body:
{issue_body}
{triage_section}
Approval comment:
{comment_body}

Current source files (backend + frontend):
{files_content}

Existing test files (you must not break these, and you must extend them):
{tests_content}

## Your job — act like a proper developer + tester

1. UNDERSTAND the root cause from the triage plan above.
2. WRITE or EXTEND the relevant test file so it has a test that would FAIL before your fix \
   and PASS after it. This is a regression test — it proves the bug is fixed and guards against \
   future regressions. Follow the existing test style.
3. WRITE the fix across all affected files (backend AND frontend if needed).
4. VERIFY mentally that your fix makes the new test pass and does not break the existing ones.

## Rules
- Follow the triage plan — it has already identified the root cause and all affected files.
- Fix ALL affected files. Never fix only the backend when the frontend is also broken, or vice versa.
- Keep every Python file py_compile-clean.
- Keep TypeScript/TSX/TS files syntactically valid.
- Tests go in: tests/test_<module>.py (backend) or frontend/__tests__/<subject>.test.ts (frontend).
- Do not change unrelated behaviour or unrelated tests.

## Output format
Output a JSON object with EXACTLY this structure (no markdown, no code fences):
{{
  "summary": "One sentence: root cause + what the fix does + what the new test checks",
  "files": [
    {{
      "path": "tests/test_generator_prompt.py",
      "content": "...complete file including new regression test..."
    }},
    {{
      "path": "backend/generator.py",
      "content": "...complete fixed file..."
    }}
  ]
}}

Include EVERY file that changes — source files AND test files. Only omit files that are truly unchanged."""

    print(f"Calling OpenAI API for issue #{issue_number}…")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=16000,
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    print(f"Response received ({len(raw)} chars)")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\n{raw[:500]}")
        sys.exit(1)

    print(f"Summary: {result.get('summary', '')}")

    files = result.get("files", [])
    print(f"Files to write: {[f['path'] for f in files]}")

    allowed_prefixes = ("backend/", "crawler/", "frontend/", "tests/")

    for fc in files:
        path    = fc.get("path", "").strip()
        content = fc.get("content", "")
        if not path or not content:
            continue
        if not any(path.startswith(d) for d in allowed_prefixes):
            print(f"Skipping disallowed path: {path}")
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        print(f"Written: {path}")

    if not files:
        print("No files changed.")


if __name__ == "__main__":
    main()
