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
    "machine learning":                       "Artificial Intelligence and Machine Learning",
    "artificial intelligence":                "Artificial Intelligence and Machine Learning",
    # Security shorthand
    "cybersecurity":                          "Security",
    "cyber security":                         "Security",
    # Energy & power
    "power":                                  "Energy and Power",
    "power system":                           "Energy and Power",
    "power systems":                          "Energy and Power",
    "power electronics":                      "Energy and Power",
    "energy":                                 "Energy and Power",
    "smart grid":                             "Energy and Power",
    "renewable energy":                       "Energy and Power",
    # Communications & networks
    "communications":                         "Communications and Networks",
    "communication":                          "Communications and Networks",
    "networks":                               "Communications and Networks",
    "networking":                             "Communications and Networks",
    "wireless":                               "Communications and Networks",
    "information theory":                     "Information Science and Learning Systems",
    # Devices / nano / EM / analog / chip
    "nanotechnology":                         "Device Science and Nanotechnology",
    "nanoelectronics":                        "Device Science and Nanotechnology",
    "semiconductor":                          "Device Science and Nanotechnology",
    "electromagnetics":                       "Electromagnetics and Microwaves",
    "microwave":                              "Electromagnetics and Microwaves",
    "microwaves":                             "Electromagnetics and Microwaves",
    "antenna":                                "Electromagnetics and Microwaves",
    "analog":                                 "Analog and Mixed Signals",
    "mixed-signal":                           "Analog and Mixed Signals",
    "vlsi":                                   "Computer Engineering and Systems",
    "embedded":                               "Computer Engineering and Systems",
    "computer architecture":                  "Computer Engineering and Systems",
    "chip manufacturing":                     "Chip Manufacturing",
    "semiconductor manufacturing":            "Chip Manufacturing",
    # Biomedical
    "biomedical":                             "Biomedical Imaging, Sensing and Genomic Signal Processing",
    "bioinformatics":                         "Biomedical Imaging, Sensing and Genomic Signal Processing",
    "genomic":                                "Biomedical Imaging, Sensing and Genomic Signal Processing",
    "medical imaging":                        "Biomedical Imaging, Sensing and Genomic Signal Processing",
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


# Score (from retriever._people_by_area signal weights) at or above which a
# faculty member's own profile shows their work CENTERS on the topic, not just
# touches it. Below this they're listed as "also active". Tunable via env.
CORE_SIGNAL_THRESHOLD = float(os.getenv("AREA_CORE_THRESHOLD", "4.0"))

# Areas whose research-area page tags no "Group Leader," but that have a verified
# departmental lead by TITLE (from the person's own profile). Value: (name, role,
# email). Verified on the department site: Krishna Narayanan's profile lists
# "Associate Head for AI" with email krn@tamu.edu. (The graph node's email is
# wrong — a crawl merge mixed in Rajendran's — so the correct email is pinned
# here rather than read from the polluted node.) We surface these labeled by
# their real title — NOT as the research-area "group leader," which the site
# doesn't designate.
_AREA_CONTACT_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "Artificial Intelligence and Machine Learning": (
        "Krishna Narayanan", "Associate Head for AI", "krn@tamu.edu"),
}


def build_area_roster(topic_words: list[str], precise_chunks: list[dict]) -> str | None:
    """Graded roster for 'which professors research <area>' queries.

    `precise_chunks` come from retriever._people_by_area, each carrying a
    `signal_score` (how strongly the profile reflects the topic) and
    `signal_hits` (the specific sub-fields matched). We de-duplicate by person,
    then grade into two tiers by signal strength:

      • Primary  — score >= CORE_SIGNAL_THRESHOLD: work centers on the topic.
      • Also active — cleared the precision floor but lower signal.

    This replaces the old binary split (exact-umbrella-phrase = "core",
    everyone else dumped into "broader related groups"), which collapsed the
    core tier to one person. The department's own research-area roster is still
    appended for completeness, but as a clearly-labeled third section rather than
    the primary answer. Returns a formatted context string, or None.
    """
    graph = _load_graph()
    topic_label = " ".join(topic_words)
    fac_nodes = graph["nodes"]["faculty"]
    graph_key_to_name = {_name_key(g): g for g in fac_nodes}

    # ── De-duplicate precise chunks by person, unioning their matched sub-fields
    #    across ALL their profile chunks. The grade is then the SUM of the
    #    distinct sub-field weights — so a researcher whose signals are split
    #    across chunks (e.g. "artificial intelligence" in one, "deep learning" in
    #    another) is credited for both, not graded by a single best chunk.
    by_person: dict[frozenset, dict] = {}
    for c in precise_chunks:
        disp = _display_from_title(c.get("title", ""))
        key = _name_key(disp)
        if not key:
            continue
        hits = c.get("signal_hits", {}) or {}
        if not isinstance(hits, dict):  # tolerate the old list-of-terms shape
            hits = {t: 2.0 for t in hits}
        if key not in by_person:
            by_person[key] = {
                "name": graph_key_to_name.get(key, disp),
                "hits": dict(hits),
            }
        else:
            for term, w in hits.items():
                by_person[key]["hits"][term] = max(by_person[key]["hits"].get(term, 0.0), w)

    # ── Department research-area grouping (authoritative). Used as BOTH a recall
    #    FLOOR and the completeness roster. A faculty member the department lists
    #    in this area but whose profile uses different vocabulary (e.g. "learning
    #    and game theory" instead of "machine learning") scores 0 on signal terms
    #    and would otherwise vanish — so we add every area member the signals
    #    missed at score 0, guaranteeing we never drop someone the department
    #    itself groups here (the Shakkottai case).
    areas = graph["nodes"]["research_areas"]
    matched_areas = _find_research_areas(topic_label, graph)
    dept_rosters: dict[str, list[str]] = {}
    for area in matched_areas:
        members = [m for m in areas[area].get("faculty", []) if _name_key(m)]
        if not members:
            continue
        dept_rosters[area] = members
        for m in members:
            k = _name_key(m)
            if k and k not in by_person:
                by_person[k] = {"name": graph_key_to_name.get(k, m),
                                "hits": {}, "score": 0.0, "dept_listed": True}

    for p in by_person.values():
        if "score" not in p:
            p["score"] = sum(p["hits"].values())
    people = sorted(by_person.values(), key=lambda p: p["score"], reverse=True)
    primary = [p for p in people if p["score"] >= CORE_SIGNAL_THRESHOLD]
    also_active = [p for p in people if p["score"] < CORE_SIGNAL_THRESHOLD]
    # If nobody cleared the core bar (e.g. a small/sparse area) but we do have
    # graded people, promote the strongest few so the answer still names a core.
    if not primary and people:
        primary = people[: min(5, len(people))]
        also_active = people[len(primary):]

    # ── Suggested first contact: the area's group leader(s) + email. A small
    #    topic → area → leader → contact hop so "who should I reach out to about
    #    X" yields an actionable name, not just a list.
    # Only the PRIMARY area's group leader — never a tangential area that merely
    # shares a keyword (e.g. an AI query also matches "Information Science and
    # Learning Systems" via 'learning', whose leader is Chao Tian; we must not
    # offer him as the AI contact). If the primary area has no designated leader
    # on the department site, show no contact rather than inventing one.
    contacts: list[str] = []
    primary_area = _best_area_for(topic_label)
    if primary_area:
        for e in graph["edges"].get("faculty_group_leader", []):
            if e.get("research_area") != primary_area:
                continue
            leader = e.get("faculty")
            email = (fac_nodes.get(leader, {}) or {}).get("email")
            contacts.append(f"{leader} (leads {primary_area})" + (f" — {email}" if email else ""))
        # If the area has no designated group leader but has a verified lead by
        # title (e.g. AI/ML → Narayanan, "Associate Head for AI"), use that.
        if not contacts and primary_area in _AREA_CONTACT_OVERRIDES:
            name, role, email = _AREA_CONTACT_OVERRIDES[primary_area]
            contacts.append(f"{name} ({role})" + (f" — {email}" if email else ""))

    if not people and not dept_rosters:
        return None

    def _fmt(p: dict) -> str:
        hits = sorted(p["hits"])
        if hits:
            return f"- {p['name']} [{', '.join(hits)}]"
        tag = " (department-listed in the area)" if p.get("dept_listed") else ""
        return f"- {p['name']}{tag}"

    lines = [f'--- Faculty by Research Area: "{topic_label}" ---', ""]
    lines.append(
        f'Faculty work on "{topic_label}" at different depths. Below they are '
        "graded by how strongly their own profile reflects this work (matched "
        "sub-fields in brackets). Faculty the department officially lists in the "
        "area but whose profiles use other wording are included too, marked "
        "department-listed, so no genuine area member is dropped.")
    lines.append("")
    lines.append(
        "HOW TO ANSWER: Lead with the PRIMARY researchers as the core answer — "
        "name them and, in a few words each, their specific focus from the "
        "bracketed sub-fields. Then list those ALSO ACTIVE (including the "
        "department-listed faculty) more briefly. Include EVERY name provided — "
        "do not drop the department-listed ones — but do NOT collapse the two "
        "tiers into one undifferentiated list, and never invent a focus not "
        "shown. If a suggested contact is given, offer it as a good first point "
        "of contact.")

    if contacts:
        lines.append("")
        lines.append("Suggested first contact:")
        lines += [f"- {c}" for c in contacts]
    if primary:
        lines.append("")
        lines.append(f"Primary researchers (work centers on {topic_label} — {len(primary)}):")
        lines += [_fmt(p) for p in primary]
    if also_active:
        lines.append("")
        lines.append(f"Also active in {topic_label} (related/applied or department-listed — {len(also_active)}):")
        lines += [_fmt(p) for p in also_active]
    if dept_rosters:
        lines.append("")
        lines.append(
            "Department's official research-area roster (the full area grouping "
            "on the website):")
        for area, members in dept_rosters.items():
            lines.append(f"- {area} ({len(members)}): {', '.join(members)}")
    return "\n".join(lines)


# ── Cross-area intersection ("faculty who work on BOTH X and Y") ──────────────
# The single-area roster path mashes "AI and power systems" into one topic and
# matches only the first area (AI), returning the AI list and silently dropping
# the "power systems" constraint. This handler detects a genuine two-area query
# and returns the faculty the department lists in BOTH areas' rosters.
_INTERSECTION_RE = re.compile(
    r"\b(both|intersection|combin\w+|overlap\w*|across|bridg\w+|as well as)\b", re.I)


def _best_area_for(phrase: str) -> str | None:
    """Map a free-text phrase to the single best-matching research area."""
    graph = _load_graph()
    areas = graph["nodes"]["research_areas"]
    p = phrase.lower()
    for alias in sorted(_AREA_ALIASES, key=len, reverse=True):
        if alias in p and _AREA_ALIASES[alias] in areas:
            return _AREA_ALIASES[alias]
    ptoks = set(re.findall(r"[a-z]+", p))
    best, best_score = None, 0
    for a in areas:
        atoks = {w for w in a.lower().split() if len(w) > 3}
        score = len(ptoks & atoks)
        if score > best_score:
            best, best_score = a, score
    return best


def build_intersection_roster(question: str) -> str | None:
    """Faculty in the graph rosters of BOTH areas named in a two-area query.

    Returns a formatted context string, or None if the query isn't a clean
    two-distinct-area intersection (so the caller falls back to the normal
    single-area roster).
    """
    graph = _load_graph()
    areas_node = graph["nodes"]["research_areas"]
    q = (question or "").lower()
    sides = re.split(r"\band\b|&|\+|,|/| vs\.? | versus ", q)
    mapped: list[str] = []
    for s in sides:
        a = _best_area_for(s)
        if a and a not in mapped:
            mapped.append(a)
    if len(mapped) < 2:
        return None
    mapped = mapped[:2]
    a1, a2 = mapped
    m1 = {_name_key(m): m for m in areas_node[a1].get("faculty", []) if _name_key(m)}
    m2 = {_name_key(m): m for m in areas_node[a2].get("faculty", []) if _name_key(m)}
    common = set(m1) & set(m2)

    lines = [f'--- Faculty working across "{a1}" AND "{a2}" ---', ""]
    if common:
        names = sorted(m1[k] for k in common)
        lines.append(
            f"These faculty are listed by the department in BOTH {a1} and {a2} "
            f"— i.e. they genuinely work across the two areas ({len(names)}):")
        lines += [f"- {n}" for n in names]
        lines.append("")
        lines.append(
            "HOW TO ANSWER: This is a 'works on BOTH' question — present ONLY these "
            "cross-area faculty as the answer. Do NOT list everyone from just one of "
            "the two areas; that would ignore the second half of the question.")
    else:
        lines.append(
            f"No faculty are listed in BOTH {a1} and {a2}. State that honestly "
            "rather than listing one area's faculty as if they answered the question. "
            "You may add that researchers in each area sometimes collaborate.")
        lines.append("")
        lines.append(f"{a1} faculty: {', '.join(sorted(m1.values()))}")
        lines.append(f"{a2} faculty: {', '.join(sorted(m2.values()))}")
    return "\n".join(lines)


# ── Authoritative complete degree-program list ───────────────────────────────
# "What programs does ECE offer?" was answered from a crawled degrees page that
# can be stale (it dropped the Microelectronics MS, the certificates, and the
# minor even though the graph has them). Serve the complete list straight from
# the graph so newer programs are never missed — same pattern as the faculty
# roster.
def is_degree_list_query(query: str) -> bool:
    """True for 'what degrees/programs does ECE offer' style questions."""
    q = (query or "").lower()
    if any(w in q for w in ("research", "course", "faculty", "professor",
                            "scholarship", "deadline", "requirement", "apply",
                            "admission", "tuition", "gre")):
        return False
    has_degree_word = any(w in q for w in (
        "program", "degree", "major", "ms", "m.s", "phd", "ph.d", "master",
        "doctoral", "doctorate", "certificate", "minor", "bachelor"))
    has_ask = any(w in q for w in (
        "offer", "available", "what", "which", "list", "all", "have",
        "provide", "kind", "type"))
    return has_degree_word and has_ask


def build_degree_roster() -> str | None:
    """Complete department degree-program list from the graph (never truncated)."""
    graph = _load_graph()
    degs = list(graph["nodes"]["degree_programs"].values())
    if not degs:
        return None
    groups = [
        ("Undergraduate degrees", lambda d: d["level"] == "undergraduate" and d["type"] == "degree"),
        ("Undergraduate minor", lambda d: d["type"] == "minor"),
        ("Graduate degrees (on campus)", lambda d: d["level"] == "graduate" and d["type"] == "degree"),
        ("Online graduate degrees", lambda d: d["type"] == "online"),
        ("Graduate certificates", lambda d: d["type"] == "certificate"),
    ]
    lines = [
        "--- Complete TAMU ECE Degree Programs (from knowledge graph) ---",
        "",
        f"This is the authoritative, COMPLETE list of all {len(degs)} degree "
        "programs, certificates, and minors the department offers. Do NOT omit "
        "any of them and do NOT add a disclaimer about the list being incomplete.",
        "",
        "HOW TO ANSWER: present these grouped under the headings below, including "
        "EVERY program in each group. If the user asked only about a subset (e.g. "
        "'MS and PhD programs'), lead with that subset, but still include the "
        "Microelectronics MS, the certificates, and the minor where relevant — "
        "never silently drop them.",
    ]
    for heading, pred in groups:
        items = [d for d in degs if pred(d)]
        if items:
            lines.append("")
            lines.append(f"{heading}:")
            lines += [f"- {d['name']}" for d in items]
    return "\n".join(lines)
