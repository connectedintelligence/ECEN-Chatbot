"""
graph_retriever.py — Retrieves context from the knowledge graph for a given query.

At query time:
  1. Detect query intent (faculty lookup, research area, degree, relationship)
  2. Traverse the graph for relevant nodes
  3. Return formatted context strings to pass to the LLM alongside vector results
"""

from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger(__name__)

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "graph.json")

_graph: dict | None = None


def _load_graph() -> dict:
    global _graph
    if _graph is None:
        if not os.path.exists(GRAPH_PATH):
            log.warning("graph.json not found — run graph_builder.py first")
            return {"nodes": {"faculty": {}, "research_areas": {}, "degree_programs": {}, "research_centers": {}}, "edges": {}}
        with open(GRAPH_PATH) as f:
            _graph = json.load(f)
        log.info(f"Graph loaded: {len(_graph['nodes']['faculty'])} faculty, "
                 f"{len(_graph['nodes']['research_areas'])} research areas")
    return _graph


# ── Intent detection ─────────────────────────────────────────────────────────

def _detect_intents(query: str) -> list[str]:
    """Detect what kinds of graph traversal are needed."""
    q = query.lower()
    intents = []

    if any(w in q for w in ["faculty", "professor", "who", "researcher", "staff", "teach"]):
        intents.append("faculty_lookup")
    if any(w in q for w in ["research area", "research group", "work on", "specialize", "focus"]):
        intents.append("research_area")
    if any(w in q for w in ["degree", "program", "major", "ms ", "phd", "bachelor", "master", "certificate", "online"]):
        intents.append("degree_lookup")
    if any(w in q for w in ["both", "and", "across", "combine", "intersection", "also"]):
        intents.append("relationship")
    if any(w in q for w in ["email", "office", "phone", "contact", "reach"]):
        intents.append("contact_lookup")
    if any(w in q for w in ["list", "all", "every", "members", "who are"]):
        intents.append("list_all")

    return intents or ["general"]


# ── Graph traversal helpers ───────────────────────────────────────────────────

# Short-form / abbreviation → canonical research area name.
# Needed because _find_research_areas skips tokens shorter than 4 chars ("ai",
# "ml") and because colloquial words like "control" / "controls" don't literally
# appear in any TAMU ECE area name.
_AREA_ALIASES: dict[str, str] = {
    # AI / ML
    "ai":                                     "Artificial Intelligence and Machine Learning",
    "ml":                                     "Artificial Intelligence and Machine Learning",
    "deep learning":                          "Artificial Intelligence and Machine Learning",
    "neural network":                         "Artificial Intelligence and Machine Learning",
    "neural networks":                        "Artificial Intelligence and Machine Learning",
    # Control systems → closest area
    "control":                                "Computer Engineering and Systems",
    "controls":                               "Computer Engineering and Systems",
    "control systems":                        "Computer Engineering and Systems",
    "control theory":                         "Computer Engineering and Systems",
    "cyber-physical":                         "Computer Engineering and Systems",
    "cyberphysical":                          "Computer Engineering and Systems",
    "stochastic control":                     "Computer Engineering and Systems",
    "reinforcement learning":                 "Artificial Intelligence and Machine Learning",
    "rl":                                     "Artificial Intelligence and Machine Learning",
    # Security shorthand
    "cybersecurity":                          "Security",
    "cyber security":                         "Security",
}


def _find_research_areas(query: str, graph: dict) -> list[str]:
    """Return research area names mentioned in the query.

    Checks both keyword overlap with area names (words > 3 chars) and the
    _AREA_ALIASES table for abbreviations / colloquial terms like 'AI' or
    'control' that don't literally appear in any area name.
    """
    q = query.lower()
    matched: set[str] = set()
    for area in graph["nodes"]["research_areas"]:
        keywords = [w.lower() for w in area.split() if len(w) > 3]
        if any(kw in q for kw in keywords):
            matched.add(area)
    for alias, area in _AREA_ALIASES.items():
        if alias in q and area in graph["nodes"]["research_areas"]:
            matched.add(area)
    return list(matched)


def _find_faculty_by_name(query: str, graph: dict) -> list[str]:
    """Return faculty names mentioned in the query."""
    matched = []
    for name in graph["nodes"]["faculty"]:
        parts = name.lower().split()
        if any(p in query.lower() for p in parts if len(p) > 3):
            matched.append(name)
    return matched


def _format_faculty(name: str, node: dict) -> str:
    lines = [f"**{name}**"]
    if node.get("titles"):
        lines.append("  " + "; ".join(node["titles"][:3]))
    if node.get("office"):
        lines.append(f"  Office: {node['office']}")
    if node.get("phone"):
        lines.append(f"  Phone: {node['phone']}")
    if node.get("email"):
        lines.append(f"  Email: {node['email']}")
    if node.get("research_areas"):
        lines.append(f"  Research Areas: {', '.join(node['research_areas'])}")
    return "\n".join(lines)


def _format_research_area(name: str, node: dict, graph: dict) -> str:
    lines = [f"**{name}**"]
    if node.get("description"):
        lines.append(f"  {node['description']}")

    # Find group leader
    leaders = [e["faculty"] for e in graph["edges"].get("faculty_group_leader", []) if e["research_area"] == name]
    if leaders:
        lines.append(f"  Group Leader: {', '.join(leaders)}")

    # Faculty members
    members = [f for f in node.get("faculty", []) if f not in leaders]
    if members:
        lines.append(f"  Faculty ({len(members)}): {', '.join(members)}")

    return "\n".join(lines)


def _format_degree(name: str, node: dict) -> str:
    parts = [f"**{name}**"]
    parts.append(f"  Level: {node.get('level', '').title()}")
    parts.append(f"  Type: {node.get('type', '').title()}")
    if node.get("short"):
        parts.append(f"  Short: {node['short']}")
    return "\n".join(parts)


# ── Main graph query function ─────────────────────────────────────────────────

def graph_query(query: str) -> str | None:
    """
    Query the knowledge graph for the given user question.
    Returns a formatted context string, or None if graph has no relevant info.
    """
    graph = _load_graph()
    if not graph["nodes"]["faculty"]:
        return None

    intents = _detect_intents(query)
    sections = []

    # ── Research area lookup ──────────────────────────────────────────────────
    matched_areas = _find_research_areas(query, graph)

    if "research_area" in intents or matched_areas:
        if matched_areas:
            area_sections = []
            for area in matched_areas:
                node = graph["nodes"]["research_areas"].get(area)
                if node:
                    area_sections.append(_format_research_area(area, node, graph))
            if area_sections:
                sections.append("Research Areas:\n" + "\n\n".join(area_sections))
        elif "list_all" in intents or "general" in intents:
            # List all research areas briefly
            all_areas = list(graph["nodes"]["research_areas"].keys())
            sections.append("Research Areas offered by TAMU ECE:\n" + "\n".join(f"- {a}" for a in all_areas))

    # ── Faculty lookup ────────────────────────────────────────────────────────
    if "faculty_lookup" in intents or "contact_lookup" in intents:
        matched_faculty = _find_faculty_by_name(query, graph)

        if matched_faculty:
            fac_sections = []
            for name in matched_faculty:
                node = graph["nodes"]["faculty"].get(name)
                if node:
                    fac_sections.append(_format_faculty(name, node))
            if fac_sections:
                sections.append("Faculty:\n" + "\n\n".join(fac_sections))
        elif matched_areas:
            # Return faculty from matched research areas
            for area in matched_areas:
                node = graph["nodes"]["research_areas"].get(area, {})
                faculty_names = node.get("faculty", [])[:10]
                if faculty_names:
                    fac_details = []
                    for name in faculty_names:
                        fnode = graph["nodes"]["faculty"].get(name)
                        if fnode:
                            fac_details.append(_format_faculty(name, fnode))
                    if fac_details:
                        sections.append(f"Faculty in {area}:\n" + "\n\n".join(fac_details))

    # ── Relationship query (faculty in multiple areas) ────────────────────────
    if "relationship" in intents and len(matched_areas) >= 2:
        # Find faculty that appear in ALL matched areas
        area_faculty_sets = [
            set(graph["nodes"]["research_areas"].get(a, {}).get("faculty", []))
            for a in matched_areas
        ]
        common = area_faculty_sets[0]
        for s in area_faculty_sets[1:]:
            common &= s

        if common:
            sections.append(
                f"Faculty working across {' and '.join(matched_areas)}:\n" +
                "\n".join(f"- {name}" for name in sorted(common))
            )
        else:
            sections.append(
                f"No faculty found with membership in all of: {', '.join(matched_areas)}. "
                f"They may work in adjacent areas."
            )

    # ── Degree lookup ─────────────────────────────────────────────────────────
    if "degree_lookup" in intents:
        q = query.lower()
        # Filter degrees by query keywords
        relevant = []
        for name, node in graph["nodes"]["degree_programs"].items():
            name_lower = name.lower()
            short_lower = node.get("short", "").lower()
            if (any(w in name_lower for w in q.split() if len(w) > 3) or
                    any(w in short_lower for w in q.split() if len(w) > 2) or
                    node.get("level", "") in q or
                    node.get("type", "") in q):
                relevant.append(node)

        if not relevant:
            relevant = list(graph["nodes"]["degree_programs"].values())

        undergrad = [d for d in relevant if d["level"] == "undergraduate"]
        grad = [d for d in relevant if d["level"] == "graduate" and d["type"] == "degree"]
        online = [d for d in relevant if d["type"] == "online"]
        certs = [d for d in relevant if d["type"] == "certificate"]

        deg_lines = []
        if undergrad:
            deg_lines.append("Undergraduate:\n" + "\n".join(f"  - {d['name']}" for d in undergrad))
        if grad:
            deg_lines.append("Graduate:\n" + "\n".join(f"  - {d['name']}" for d in grad))
        if online:
            deg_lines.append("Online:\n" + "\n".join(f"  - {d['name']}" for d in online))
        if certs:
            deg_lines.append("Certificates:\n" + "\n".join(f"  - {d['name']}" for d in certs))

        if deg_lines:
            sections.append("Degree Programs:\n" + "\n".join(deg_lines))

    if not sections:
        return None

    return "--- Knowledge Graph Context ---\n\n" + "\n\n".join(sections)


# ── Layered "faculty by research area" roster ────────────────────────────────
# Title/role words that are never part of a person's actual name. A list whose
# every token is one of these (e.g. "Regents Professor") is a scraping artifact,
# not a faculty member, and is dropped.
_TITLE_TOKENS = {
    "professor", "professors", "regents", "distinguished", "chair", "dean",
    "director", "fellow", "associate", "assistant", "interim", "endowed",
    "head", "emeritus", "lecturer", "member", "affiliated", "co", "senior",
}


def _name_key(name: str) -> frozenset:
    """Order-independent identity key for a person, ignoring punctuation,
    title words, and single-letter initials. 'Tian, Chao' and 'Chao Tian'
    collide; 'Regents Professor' collapses to the empty set (→ not a person)."""
    toks = [t for t in re.findall(r"[a-z]+", name.lower())
            if len(t) > 1 and t not in _TITLE_TOKENS]
    return frozenset(toks)


def _display_from_title(title: str) -> str:
    """'Tian, Chao | Texas A&M University Engineering' → 'Chao Tian'."""
    base = title.split("|")[0].strip()
    if "," in base:
        last, first = [p.strip() for p in base.split(",", 1)]
        return f"{first} {last}".strip()
    return base


# ── Global "list ALL faculty" roster ─────────────────────────────────────────
# A faculty word + an enumeration word, with NO specific research area or person
# named, means the user wants the entire department roster. Retrieval can't serve
# this (only top-k chunks reach the LLM, and 600-token chunks split lists mid-
# name → the "P.R." / "Scott L." truncation). The graph holds every faculty node,
# so we answer it deterministically and completely.
_FACULTY_WORDS = ("faculty", "professor", "professors", "instructor", "instructors")
_ENUMERATE_WORDS = ("list", "all", "every", "complete", "entire", "full",
                    "who are", "name the", "everyone", "names of")


def research_area_names() -> list[str]:
    """Canonical research-area names from the graph (for the LLM router)."""
    graph = _load_graph()
    return list(graph["nodes"]["research_areas"].keys())


def is_full_faculty_query(query: str) -> bool:
    """True for 'list all faculty' style questions with no specific area/name."""
    q = (query or "").lower()
    if not any(w in q for w in _FACULTY_WORDS):
        return False
    if not any(w in q for w in _ENUMERATE_WORDS):
        return False
    graph = _load_graph()
    # If the user named a specific research area or person, it's not a global list.
    if _find_research_areas(q, graph):
        return False
    if _find_faculty_by_name(q, graph):
        return False
    return True


def build_full_faculty_roster() -> str | None:
    """Complete department faculty roster straight from the graph.

    Returns a grouped-by-research-area listing (every member, never truncated)
    plus a complete alphabetical list of all faculty, so the LLM can reproduce
    the entire roster without relying on retrieved chunks.
    """
    graph = _load_graph()
    fac = graph["nodes"]["faculty"]
    areas = graph["nodes"]["research_areas"]
    if not fac:
        return None

    leaders_by_area: dict[str, list[str]] = {}
    for e in graph["edges"].get("faculty_group_leader", []):
        bucket = leaders_by_area.setdefault(e["research_area"], [])
        if e["faculty"] not in bucket:   # de-dupe duplicate leader edges
            bucket.append(e["faculty"])

    lines = [
        "--- Complete TAMU ECE Faculty Roster (from knowledge graph) ---",
        "",
        f"This is the authoritative, COMPLETE list of all {len(fac)} faculty "
        "members on record, organized by research area. "
        "Do NOT add a disclaimer about the list being incomplete — it is complete.",
        "",
        "HOW TO ANSWER: Open with one short sentence summarizing the roster "
        f"(e.g. \"TAMU ECE has {len(fac)} faculty members across "
        f"{len([a for a in areas if areas[a].get('faculty')])} research areas\"). "
        "Then present the faculty grouped under bold research-area headings, "
        "exactly as organized below. Include EVERY name — do not drop anyone — "
        "but you do not need to repeat the separate alphabetical list; the "
        "grouped sections are the answer. Note briefly that some faculty appear "
        "in more than one area. Keep it clean and scannable, not a wall of text.",
        "",
        "By research area:",
    ]
    for area, node in areas.items():
        members = [m for m in node.get("faculty", []) if _name_key(m)]
        if not members:
            continue
        leaders = leaders_by_area.get(area, [])
        header = f"\n{area}"
        if leaders:
            header += f" (Group Leader: {', '.join(leaders)})"
        lines.append(header + ":")
        lines += [f"  - {m}" for m in members]

    all_names = sorted(fac.keys(), key=lambda n: n.split()[-1].lower())
    lines.append("")
    lines.append(f"Complete alphabetical list (all {len(all_names)} faculty):")
    lines += [f"- {n}" for n in all_names]

    return "\n".join(lines)


def faculty_roster_sources() -> list[dict]:
    """The pages the full faculty roster is actually drawn from — the per-area
    research pages that list each group's members. Used as citations so the
    answer cites only relevant sources, not the whole retrieval set."""
    graph = _load_graph()
    areas = graph["nodes"]["research_areas"]
    out: list[dict] = []
    seen: set[str] = set()
    for area, node in areas.items():
        url = node.get("url")
        if not url or url in seen or not node.get("faculty"):
            continue
        seen.add(url)
        out.append({"url": url, "title": f"{area} — Faculty", "section": "research"})
    return out


def build_area_roster(topic_words: list[str], precise_chunks: list[dict]) -> str | None:
    """Layered roster for 'which professors research <area>' queries.

    Tier 1 (precise): faculty whose own profile literally names the topic —
      derived from `precise_chunks` (exact-phrase people matches).
    Tier 2 (broader): every graph research-area whose membership overlaps the
      precise tier — so the umbrella groups the core researchers belong to are
      surfaced without a hand-maintained topic→area synonym map. Falls back to
      keyword-matched areas when the precise tier is empty.

    Returns a formatted context string, or None if nothing matched.
    """
    graph = _load_graph()
    topic_label = " ".join(topic_words)

    # ── Tier 1: precise, de-duplicated by person, mapped to graph display name.
    fac_nodes = graph["nodes"]["faculty"]
    graph_key_to_name = {_name_key(g): g for g in fac_nodes}

    precise_display: list[str] = []
    precise_keys: set[frozenset] = set()
    for c in precise_chunks:
        disp = _display_from_title(c.get("title", ""))
        key = _name_key(disp)
        if not key or key in precise_keys:
            continue
        precise_keys.add(key)
        precise_display.append(graph_key_to_name.get(key, disp))

    # ── Tier 2: broader umbrella areas.
    # Use only areas with STRONG overlap with the precise tier — most of the
    # core researchers must belong — so we surface the genuine umbrella groups
    # (e.g. Communications and Networks) and not every area a prolific professor
    # happens to touch. Threshold: at least 60% of the core, floor of 2.
    areas = graph["nodes"]["research_areas"]
    broader: dict[str, list[str]] = {}
    if precise_keys:
        threshold = max(2, round(0.6 * len(precise_keys)))
        scored = []
        for area, node in areas.items():
            members = [m for m in node.get("faculty", []) if _name_key(m)]  # drops title-only junk
            overlap = sum(1 for m in members if _name_key(m) in precise_keys)
            if overlap >= threshold:
                scored.append((overlap, area, members))
        # Most-relevant umbrella first.
        for _, area, members in sorted(scored, key=lambda x: -x[0]):
            broader[area] = members
    if not broader:  # no precise tier, or no area cleared the bar → keyword match
        for area in _find_research_areas(topic_label, graph):
            broader[area] = [m for m in areas[area].get("faculty", []) if _name_key(m)]

    if not precise_display and not broader:
        return None

    lines = [f'--- Faculty by Research Area: "{topic_label}" ---', ""]
    lines.append(
        f'TAMU ECE has no research area named exactly "{topic_label}"; it spans the '
        f'broader umbrella groups listed below. Present BOTH tiers in the answer.'
        if broader and precise_display else ""
    )
    if precise_display:
        lines.append("")
        lines.append(
            f'Core researchers (their faculty profile explicitly lists "{topic_label}") '
            f'— {len(precise_display)}:'
        )
        lines += [f"- {n}" for n in precise_display]
    if broader:
        lines.append("")
        lines.append("Broader related research groups:")
        for area, members in broader.items():
            lines.append(f"- {area} ({len(members)}): {', '.join(members)}")
    return "\n".join(l for l in lines if l is not None)
