"""
graph_builder.py — Builds a lightweight knowledge graph from the Postgres
(pgvector) chunk store `ecen_docs`. (Migrated off Qdrant.)

Nodes:
  Faculty        — name, title, office, email, phone, url
  ResearchArea   — name, description, url
  DegreeProgram  — name, level (undergrad/grad), type (online/cert/degree), url
  ResearchCenter — name, url

Edges:
  Faculty      -[MEMBER_OF]->    ResearchArea
  Faculty      -[GROUP_LEADER]-> ResearchArea
  DegreeProgram -[LEVEL]->       {undergraduate, graduate}

Run (after a fresh crawl + ingest so ecen_docs is up to date):
  python3 graph_builder.py          # builds graph, backs up + saves graph.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from types import SimpleNamespace

import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Same DSN the backend uses — Dockerized pgvector on host port 5433, DB `ecen`.
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
TABLE = os.getenv("PG_TABLE", "ecen_docs")
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "graph.json")
# Below this faculty count a build is treated as degenerate (DB empty/unreachable
# or schema drift): we write to graph.json.new and keep the good graph.json.
MIN_SANE_FACULTY = 30

# ── Known research areas (canonical names) ────────────────────────────────────
RESEARCH_AREAS = [
    "Analog and Mixed Signals",
    "Artificial Intelligence and Machine Learning",
    "Biomedical Imaging, Sensing and Genomic Signal Processing",
    "Chip Manufacturing",
    "Communications and Networks",
    "Computer Engineering and Systems",
    "Device Science and Nanotechnology",
    "Electromagnetics and Microwaves",
    "Energy and Power",
    "Information Science and Learning Systems",
    "Security",
]

RESEARCH_AREA_URLS = {
    "Analog and Mixed Signals": "https://engineering.tamu.edu/electrical/research/analog-mixed-signals.html",
    "Artificial Intelligence and Machine Learning": "https://engineering.tamu.edu/electrical/research/artificial-intelligence-and-machine-learning.html",
    "Biomedical Imaging, Sensing and Genomic Signal Processing": "https://engineering.tamu.edu/electrical/research/biomedical-imaging-sensing-genomic-signal-processing.html",
    "Chip Manufacturing": "https://engineering.tamu.edu/electrical/research/chip-manufacturing.html",
    "Communications and Networks": "https://engineering.tamu.edu/electrical/research/communications-and-networks.html",
    "Computer Engineering and Systems": "https://engineering.tamu.edu/electrical/research/computer-engineering-systems.html",
    "Device Science and Nanotechnology": "https://engineering.tamu.edu/electrical/research/device-science-and-nanotechnology.html",
    "Electromagnetics and Microwaves": "https://engineering.tamu.edu/electrical/research/electromagnetics-microwaves.html",
    "Energy and Power": "https://engineering.tamu.edu/electrical/research/energy-and-power.html",
    "Information Science and Learning Systems": "https://engineering.tamu.edu/electrical/research/information-science-and-systems.html",
    "Security": "https://engineering.tamu.edu/electrical/research/security.html",
}

# ── Manual overrides ──────────────────────────────────────────────────────────
# The graph is built from website research-area pages, which don't always reflect
# a faculty member's full expertise. Add corrections here so they survive rebuilds.
# Format: faculty name (must match graph key exactly) → list of extra research areas to add.
GRAPH_OVERRIDES: dict[str, list[str]] = {
    # P.R. Kumar: pioneer in stochastic control, reinforcement learning for control,
    # cyber-physical systems, and networked control — clearly AI × control.
    # The website lists him under Computer Engineering and Systems / Info Science
    # but not under AI/ML.
    "P.R. Kumar": ["Artificial Intelligence and Machine Learning"],
    # Krishna Narayanan: title is literally "Associate Head for AI"; works on
    # coding/information theory and machine learning. The website files him under
    # Communications and Networks / Computer Engineering and Systems / Information
    # Science and Learning Systems, but not under the AI/ML page, so the graph
    # missed him for AI/ML rosters.
    "Krishna Narayanan": ["Artificial Intelligence and Machine Learning"],
}

# ── Known degree programs ─────────────────────────────────────────────────────
DEGREE_PROGRAMS = [
    {"name": "Bachelor of Science in Electrical Engineering", "level": "undergraduate", "type": "degree", "short": "BS EE"},
    {"name": "Bachelor of Science in Computer Engineering", "level": "undergraduate", "type": "degree", "short": "BS CE"},
    {"name": "Minor in Electrical Engineering", "level": "undergraduate", "type": "minor", "short": "Minor EE"},
    {"name": "Master of Science in Electrical Engineering", "level": "graduate", "type": "degree", "short": "MS EE"},
    {"name": "Master of Science in Computer Engineering", "level": "graduate", "type": "degree", "short": "MS CE"},
    {"name": "Master of Science in Microelectronics and Semiconductors", "level": "graduate", "type": "degree", "short": "MS MESC"},
    {"name": "Doctor of Philosophy in Electrical Engineering", "level": "graduate", "type": "degree", "short": "PhD EE"},
    {"name": "Doctor of Philosophy in Computer Engineering", "level": "graduate", "type": "degree", "short": "PhD CE"},
    {"name": "Online Master of Science in Electrical Engineering", "level": "graduate", "type": "online", "short": "Online MS EE"},
    {"name": "Online Master of Science in Computer Engineering", "level": "graduate", "type": "online", "short": "Online MS CE"},
    {"name": "Online Doctor of Philosophy in Electrical Engineering", "level": "graduate", "type": "online", "short": "Online PhD EE"},
    {"name": "Online Doctor of Philosophy in Computer Engineering", "level": "graduate", "type": "online", "short": "Online PhD CE"},
    {"name": "Analog and Mixed-Signal Integrated Circuit Design Certificate", "level": "graduate", "type": "certificate", "short": "Analog IC Cert"},
    {"name": "Digital Integrated Circuit Design Certificate", "level": "graduate", "type": "certificate", "short": "Digital IC Cert"},
    {"name": "Electromagnetic Fields and Microwave Circuit Design Certificate", "level": "graduate", "type": "certificate", "short": "EM Cert"},
    {"name": "Semiconductor Manufacturing Certificate", "level": "graduate", "type": "certificate", "short": "Semiconductor Cert"},
]


def _fetch_all_chunks() -> list:
    """Load every chunk from Postgres `ecen_docs`.

    Returns a list of objects shaped like the old Qdrant points — each has a
    `.payload` dict with url/title/section/text/chunk_id — so the rest of the
    builder is unchanged.
    """
    conn = psycopg2.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT chunk_id, url, title, section, text FROM {TABLE};"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    points = [
        SimpleNamespace(payload={
            "chunk_id": r[0], "url": r[1], "title": r[2],
            "section": r[3], "text": r[4],
        })
        for r in rows
    ]
    return points


# Words that appear in navigation/section headings — not faculty names
_NON_FACULTY_WORDS = {
    "our", "research", "areas", "centers", "manufacturing", "chip", "view",
    "more", "news", "program", "programs", "faculty", "members", "leader",
    "undergraduate", "graduate", "department", "electrical", "computer",
    "engineering", "university", "texas", "college", "information", "science",
    "learning", "systems", "communications", "networks", "security", "energy",
    "power", "analog", "mixed", "signals", "device", "nanotechnology",
    "electromagnetics", "microwaves", "biomedical", "imaging", "sensing",
    "genomic", "signal", "processing", "artificial", "intelligence", "machine",
    # Sub-area / topic headings that scrape as two-capitalized-word lines and
    # were being mistaken for people (e.g. "Power Electronics").
    "electronics", "circuits", "circuit", "photonics", "optics", "controls",
    "robotics", "semiconductor", "semiconductors", "microelectronics",
    "wireless", "sensors", "hardware", "software", "quantum", "nano", "vlsi",
    "design", "integrated", "fields", "biology", "data", "cyber", "physical",
    "smart", "grid", "grids", "embedded", "high", "performance", "low",
    "computing", "computational", "optical", "rf", "microwave", "antenna",
}

# Title/role words — a line built only from these is a job title, not a name
# (e.g. "Regents Professor", "Distinguished Professor"). Used to keep titles
# from being mistaken for faculty names when they appear before any person.
_TITLE_ONLY_WORDS = {
    "professor", "regents", "distinguished", "associate", "assistant",
    "chair", "dean", "director", "fellow", "affiliated", "co-director",
    "interim", "endowed", "head", "member", "emeritus", "lecturer", "senior",
}

def _clean_name(line: str) -> str:
    """Strip quoted nicknames and collapse whitespace, e.g.
    'Jeyavijayan "JV" Rajendran' → 'Jeyavijayan Rajendran'. Without this, a name
    with a quoted nickname fails name detection, so the next person's title/email
    lines get merged into the PREVIOUS faculty entry (this is what put
    Rajendran's email on Narayanan's node)."""
    line = re.sub(r'[\"“”\'‘’][^\"“”\'‘’]*[\"“”\'‘’]', "", line)  # drop "JV" nicknames
    return re.sub(r"\s+", " ", line).strip()


def _is_likely_faculty_name(line: str) -> bool:
    """Return True if line looks like a person's name, not a heading or nav item."""
    line = _clean_name(line)
    # Must match capitalized word pattern
    if not re.match(r"^[A-Z][a-zA-Z\.\-]+([\s][A-Z][a-zA-Z\.\-]+){1,4}$", line):
        return False
    # Must have at least 2 words
    words = line.split()
    if len(words) < 2:
        return False
    lower_words = {w.lower().rstrip(".") for w in words}
    # Must not be all navigation/section words
    if lower_words.issubset(_NON_FACULTY_WORDS):
        return False
    # Must not be all title/role words ("Regents Professor", "Distinguished Professor")
    if lower_words.issubset(_TITLE_ONLY_WORDS):
        return False
    # Must not end with a single initial only (e.g. "Scott L." — incomplete name)
    if re.match(r"^.*\s[A-Z]\.$", line) and len(words) == 2:
        return False
    return True


def _parse_faculty_from_chunk(text: str, url: str) -> list[dict]:
    """
    Extract faculty entries from a research area page chunk.
    Each entry: name, title, office, phone, email, profile_url, is_group_leader
    """
    faculty = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    current = {}
    is_group_leader_section = False

    for line in lines:
        # Detect section markers
        if "Group Leader" in line:
            is_group_leader_section = True
            continue
        if "Faculty Members" in line:
            is_group_leader_section = False
            continue

        # Office
        if line.startswith("Office:") or re.match(r"^WEB \d", line):
            if current:
                current["office"] = line.replace("Office:", "").strip()
            continue

        # Phone
        if re.match(r"^\d{3}-\d{3}-\d{4}", line) or line.startswith("Phone:"):
            if current:
                current["phone"] = line.replace("Phone:", "").strip()
            continue

        # Email
        if "@tamu.edu" in line or "@ece.tamu.edu" in line:
            if current:
                current["email"] = re.sub(r"^Email:\s*", "", line).strip()
            continue

        # Title lines
        title_keywords = ["Professor", "Associate", "Assistant", "Director", "Dean",
                          "Chair", "Fellow", "Affiliated", "Co-Director", "Interim",
                          "Regents", "Distinguished", "Endowed", "Head", "Member"]
        if any(kw in line for kw in title_keywords):
            # A title line is never a name. Attach it to the current person if
            # we have one; otherwise drop it (e.g. a stray "Regents Professor"
            # heading before any faculty entry) so it can't be misread as a name.
            if current:
                current.setdefault("titles", []).append(line)
            continue

        # Faculty name detection
        if _is_likely_faculty_name(line):
            if current and current.get("name"):
                faculty.append(current)
            current = {
                "name": _clean_name(line),
                "is_group_leader": is_group_leader_section,
                "source_url": url,
                "titles": [],
            }

    if current and current.get("name"):
        faculty.append(current)

    return faculty


def build_graph() -> dict:
    """Build the full knowledge graph from the Postgres chunk store."""
    all_points = _fetch_all_chunks()
    log.info(f"Loaded {len(all_points)} chunks from Postgres ({TABLE})")

    graph = {
        "nodes": {
            "faculty": {},       # name -> {name, titles, office, phone, email, source_url}
            "research_areas": {},  # name -> {name, url, description}
            "degree_programs": {},  # name -> {name, level, type, short}
            "research_centers": {},  # name -> {name, url}
        },
        "edges": {
            "faculty_member_of": [],    # {faculty, research_area}
            "faculty_group_leader": [],  # {faculty, research_area}
            "degree_level": [],          # {degree, level}
        }
    }

    # ── Seed known nodes ──────────────────────────────────────────────────────
    for area in RESEARCH_AREAS:
        graph["nodes"]["research_areas"][area] = {
            "name": area,
            "url": RESEARCH_AREA_URLS.get(area, ""),
            "description": "",
            "faculty": [],
        }

    for deg in DEGREE_PROGRAMS:
        graph["nodes"]["degree_programs"][deg["name"]] = deg
        graph["edges"]["degree_level"].append({"degree": deg["name"], "level": deg["level"], "type": deg["type"]})

    # ── Extract faculty from research area pages ───────────────────────────────
    research_area_chunks = [
        p for p in all_points
        if "/research/" in p.payload.get("url", "")
        and p.payload.get("section") == "research"
        and "profiles" not in p.payload.get("url", "")
    ]

    for point in research_area_chunks:
        url = point.payload["url"]
        text = point.payload["text"]

        # Match URL to research area
        matched_area = None
        for area, area_url in RESEARCH_AREA_URLS.items():
            if area_url == url:
                matched_area = area
                break
        if not matched_area:
            continue

        # Add description from first chunk
        if "::0" in point.payload["chunk_id"] and not graph["nodes"]["research_areas"][matched_area]["description"]:
            # First ~300 chars as description
            desc_lines = [l for l in text.splitlines() if l.strip() and len(l) > 40]
            if desc_lines:
                graph["nodes"]["research_areas"][matched_area]["description"] = desc_lines[0][:400]

        # Parse faculty
        faculty_list = _parse_faculty_from_chunk(text, url)
        for f in faculty_list:
            name = f["name"]
            if name not in graph["nodes"]["faculty"]:
                graph["nodes"]["faculty"][name] = {
                    "name": name,
                    "titles": f.get("titles", []),
                    "office": f.get("office", ""),
                    "phone": f.get("phone", ""),
                    "email": f.get("email", ""),
                    "source_url": f.get("source_url", ""),
                    "research_areas": [],
                }
            else:
                # Merge contact info
                existing = graph["nodes"]["faculty"][name]
                if f.get("office") and not existing["office"]:
                    existing["office"] = f["office"]
                if f.get("email") and not existing["email"]:
                    existing["email"] = f["email"]
                if f.get("phone") and not existing["phone"]:
                    existing["phone"] = f["phone"]

            # Add research area to faculty
            if matched_area not in graph["nodes"]["faculty"][name]["research_areas"]:
                graph["nodes"]["faculty"][name]["research_areas"].append(matched_area)

            # Add faculty to research area
            if name not in graph["nodes"]["research_areas"][matched_area]["faculty"]:
                graph["nodes"]["research_areas"][matched_area]["faculty"].append(name)

            # Add edge
            if f.get("is_group_leader"):
                graph["edges"]["faculty_group_leader"].append({"faculty": name, "research_area": matched_area})
            else:
                graph["edges"]["faculty_member_of"].append({"faculty": name, "research_area": matched_area})

    # ── Apply manual overrides ────────────────────────────────────────────────
    for faculty_name, extra_areas in GRAPH_OVERRIDES.items():
        if faculty_name not in graph["nodes"]["faculty"]:
            log.warning("GRAPH_OVERRIDES: faculty '%s' not found in graph — skipping", faculty_name)
            continue
        for area in extra_areas:
            if area not in graph["nodes"]["research_areas"]:
                log.warning("GRAPH_OVERRIDES: area '%s' not recognised — skipping", area)
                continue
            f_node = graph["nodes"]["faculty"][faculty_name]
            if area not in f_node["research_areas"]:
                f_node["research_areas"].append(area)
                log.info("GRAPH_OVERRIDES: added '%s' → '%s'", faculty_name, area)
            a_node = graph["nodes"]["research_areas"][area]
            if faculty_name not in a_node["faculty"]:
                a_node["faculty"].append(faculty_name)
            graph["edges"]["faculty_member_of"].append({"faculty": faculty_name, "research_area": area})

    faculty_count = len(graph["nodes"]["faculty"])
    area_count = len(graph["nodes"]["research_areas"])
    degree_count = len(graph["nodes"]["degree_programs"])
    edge_count = len(graph["edges"]["faculty_member_of"]) + len(graph["edges"]["faculty_group_leader"])

    log.info(f"Graph built: {faculty_count} faculty, {area_count} research areas, {degree_count} degrees, {edge_count} edges")
    log.info("Faculty parsed: %s", ", ".join(sorted(graph["nodes"]["faculty"])))

    # Safety: a near-empty result means the DB was unreachable/empty or the
    # schema drifted — don't clobber the known-good graph. Write to .new instead.
    if faculty_count < MIN_SANE_FACULTY:
        out = GRAPH_PATH + ".new"
        with open(out, "w") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        log.warning(
            "Only %d faculty parsed (< %d). Refusing to overwrite %s. "
            "Wrote %s for inspection — check the DB has fresh ecen_docs data.",
            faculty_count, MIN_SANE_FACULTY, GRAPH_PATH, out,
        )
        return graph

    # Back up the current graph before overwriting, so a bad rebuild is reversible.
    if os.path.exists(GRAPH_PATH):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{GRAPH_PATH}.{stamp}.bak"
        shutil.copy2(GRAPH_PATH, backup)
        log.info("Backed up existing graph to %s", backup)

    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    log.info(f"Graph saved to {GRAPH_PATH}")

    return graph


if __name__ == "__main__":
    build_graph()
