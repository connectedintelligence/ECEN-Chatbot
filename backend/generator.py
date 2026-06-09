
"""
generator.py — Calls the TAMU LLM API to generate answers from retrieved context.

TAMU's AI gateway is OpenAI-compatible, so we use the openai SDK
pointed at their base URL. Supports both streaming and non-streaming.
"""

from __future__ import annotations

import os
import logging
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

log = logging.getLogger(__name__)

TAMU_API_URL = os.getenv("OPENAI_BASE_URL", "https://chat-api.tamu.ai/openai")
TAMU_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAMU_MODEL = os.getenv("OPENAI_MODEL", "protected.gpt-5")

SYSTEM_PROMPT = (
    "You are the official assistant for the TAMU Department of Electrical and Computer Engineering. "
    "You have two modes depending on the question type:\n\n"

    "1. FACTUAL questions (faculty, programs, deadlines, requirements, contacts, tuition, admission criteria): "
    "Answer STRICTLY from the provided context. Do not add outside information. "
    "If the answer is not in the context, say: I don't have those details — check the sources below.\n\n"

    "2. ADVISORY or TREND questions (career advice, industry demand, what to study, which area is growing, "
    "how to succeed, what employers want): "
    "First ground your answer in what the department offers (from context). "
    "Then supplement with your general knowledge about industry trends and career guidance — "
    "clearly framing it as general perspective, not department policy. "
    "Be concrete, practical, and helpful. Provide detailed explanations and examples where applicable.\n\n"

    "Always: write in a clear, organized tone. Use headers and bullet points. "
    "When context contains a list (people, degrees, programs, research areas, courses), include EVERY item — never truncate or summarize a list. "
    "For course recommendations, strictly respect the level requested: "
    "if the question asks for 'undergraduate courses', list ONLY courses numbered below 500 (e.g. ECEN 314, ECEN 420); "
    "if the question asks for 'graduate courses', list ONLY courses numbered 500 and above; "
    "if no level is specified, list both but clearly separate them under 'Undergraduate' and 'Graduate' headings. "
    "When asked about a topic (e.g. 'control systems', 'machine learning'), always address BOTH: "
    "(1) relevant courses at the requested level, AND "
    "(2) faculty who research that area — if both are in the context. "
    "Do not include URLs in your answer."
)

HUMAN_TEMPLATE = """
Context:
{context}

Question: {question}

Instructions: If the context contains a list of any kind (people, degrees, programs, courses, research areas, etc.), reproduce ALL items completely — do not skip, summarize, or stop early.

Answer:"""


MAX_CHUNK_CHARS = 4000  # increased to avoid truncating list-style pages (e.g. faculty directories)

def _build_context(chunks: list[dict]) -> str:
    parts = []
    i = 0
    for c in chunks:
        # Citation-only chunks carry no text (they exist purely to appear in the
        # Sources list); skip them so they don't pollute the LLM context.
        if not (c.get("text") or "").strip():
            continue
        i += 1
        # Synthetic graph rosters (complete faculty list, area roster) are
        # deterministic and must never be truncated — chopping them would
        # reintroduce the very mid-name cutoff we built them to prevent.
        if c.get("section") == "graph":
            text = c["text"]
        else:
            text = c["text"][:MAX_CHUNK_CHARS]
        # Prepend title so LLM knows the source context
        parts.append(f"[{i}] {c['title']}\n{text}")
    return "\n\n---\n\n".join(parts)


def _get_client() -> AsyncOpenAI:
    # Long read timeout: protected.gpt-5 can be slow to first token.
    return AsyncOpenAI(
        api_key=TAMU_API_KEY,
        base_url=TAMU_API_URL,
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=15.0, pool=15.0),
    )


# Output token budget. protected.gpt-5 is a *reasoning* model: it spends hidden
# tokens thinking before any visible answer, so the cap must be generous or the
# visible answer gets truncated mid-sentence. We also detect a "length" stop and
# continue automatically (see generate()).
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8000"))
MAX_CONTINUATIONS = int(os.getenv("LLM_MAX_CONTINUATIONS", "3"))
# Low temperature → more deterministic, exact answers grounded in the context.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))


async def _stream_once(messages: list[dict]) -> tuple[str, str]:
    """One streamed completion. Returns (text, finish_reason).

    finish_reason is "stop" on a complete answer, "length" if the model hit the
    token cap (i.e. the answer was truncated and should be continued).
    """
    import httpx
    import json as _json

    payload = {
        "model": TAMU_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }

    # Reasoning models (protected.gpt-5) can be slow to first token, and SSE
    # streams sit idle between tokens — so keep a short connect timeout but a
    # long read timeout to avoid httpx.ReadTimeout mid-generation.
    timeout = httpx.Timeout(connect=15.0, read=300.0, write=15.0, pool=15.0)
    parts: list[str] = []
    finish_reason = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{TAMU_API_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {TAMU_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        ) as resp:
            log.info("TAMU raw status: %s", resp.status_code)
            async for line in resp.aiter_lines():
                line = line.strip()
                log.debug("RAW SSE line: %r", line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("[DONE]", ""):
                    continue
                try:
                    obj = _json.loads(data)
                    choice = obj.get("choices", [{}])[0]
                    text = choice.get("delta", {}).get("content", "")
                    if text:
                        parts.append(text)
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr
                except Exception:
                    pass

    return "".join(parts), finish_reason


async def generate(question: str, chunks: list[dict]) -> str:
    """Non-streaming generation via raw httpx SSE (TAMU API requires streaming).

    Auto-continues when the model stops because it hit the token cap
    (finish_reason == "length") so long lists are never cut off mid-answer.
    """
    context = _build_context(chunks)
    user_msg = HUMAN_TEMPLATE.format(context=context, question=question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    full = ""
    for attempt in range(MAX_CONTINUATIONS + 1):
        text, finish_reason = await _stream_once(messages)
        full += text
        log.info("generate() pass %d: +%d chars, finish_reason=%r",
                 attempt, len(text), finish_reason)
        if finish_reason != "length":
            break
        # Truncated by the token cap — ask the model to keep going from exactly
        # where it stopped, feeding back what it has written so far.
        messages = messages + [
            {"role": "assistant", "content": full},
            {"role": "user", "content":
                "Continue exactly where you left off. Do not repeat anything "
                "already written; just finish the answer, completing any list."},
        ]

    result = full.strip()
    log.info("generate() collected %d chars total: %r", len(result), result[:120])
    return result


def _parse_sse_content(raw: str) -> str:
    """Extract concatenated delta content from a raw SSE string."""
    import json
    content = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                content.append(text)
        except Exception:
            pass
    return "".join(content).strip()


async def generate_stream(question: str, chunks: list[dict]) -> AsyncIterator[str]:
    """Streaming generation. Yields real LLM delta tokens as they arrive.

    Auto-continues on finish_reason == "length" so long answers don't cut off.
    """
    client = _get_client()
    context = _build_context(chunks)
    user_msg = HUMAN_TEMPLATE.format(context=context, question=question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(MAX_CONTINUATIONS + 1):
        stream = await client.chat.completions.create(
            model=TAMU_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        passage = ""
        finish_reason = ""
        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta.content
            if delta:
                passage += delta
                yield delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if finish_reason != "length":
            break
        messages = messages + [
            {"role": "assistant", "content": passage},
            {"role": "user", "content":
                "Continue exactly where you left off. Do not repeat anything "
                "already written; just finish the answer, completing any list."},
        ]
