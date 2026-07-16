"""
freehire.dev API realtime provider.

Free, open-source IT job aggregator (2.9M+ live postings). No API key required.
Crawls directly from Workday, Greenhouse, Lever, Ashby, iCIMS, etc.

Responsibility:
  * Fetch IT job listings from freehire.dev API.
  * Normalise into the canonical realtime schema.
  * Extract US state codes for accurate location analytics.
"""

import time
import threading
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_FREEHIRE_URL = "https://freehire.dev/api/v1/jobs/search"
_COOLDOWN_SECONDS = 300  # 5 minutes
_last_call_ts = 0.0
_lock = threading.Lock()

# IT categories on freehire
_IT_CATEGORIES = [
    "backend", "frontend", "fullstack", "devops", "sre",
    "data_engineering", "data_science", "data_analytics",
    "ml_ai", "ai_engineering", "security", "mobile", "qa", "architecture",
]

# Title-based IT filter to catch non-IT jobs that slip through categories
_IT_TITLE_INCLUDE = {
    "software", "engineer", "developer", "programmer", "coding",
    "devops", "sre", "site reliability",
    "data engineer", "data scientist", "data architect",
    "machine learning", "ml engineer", "ml ops", "ml platform",
    "ai engineer", "ai platform", "ai architect", "ai research",
    "platform engineer", "infrastructure", "cloud",
    "security engineer", "cyber", "application security",
    "full stack", "fullstack", "backend", "frontend",
    "architect", "solutions architect", "tech lead", "technical lead",
    "systems engineer", "network engineer", "database engineer",
    "qa engineer", "test engineer", "automation engineer",
    "data analytics", "data platform", "data infrastructure",
    "reliability engineer", "production engineer",
    "blockchain", "embedded", "firmware", "hardware engineer",
    "research scientist", "applied scientist",
}

_NON_IT_TITLE_EXCLUDE = {
    "security officer", "unarmed security", "security guard",
    "security screener", "transportation security",
    "finance", "accounting", "human resources", "hr ",
    "marketing", "sales ", "sales representative", "sales manager",
    "revenue", "asset management",
    "total rewards", "compensation", "benefits analyst",
    "business operations", "gtm operations", "rev ops",
    "executive assistant", "administrative",
    "supply chain", "logistics", "warehouse",
    "legal", "paralegal", "compliance officer",
    "recruiter", "talent acquisition",
}

_TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "react", "node.js", "angular", "vue",
    "aws", "azure", "gcp", "cloud", "docker", "kubernetes",
    "sql", "postgresql", "mysql", "mongodb", "redis",
    "pytorch", "tensorflow", "machine learning", "deep learning",
    "ai", "llm", "langchain", "nlp",
    "spark", "kafka", "airflow", "hadoop",
    "terraform", "ansible", "ci/cd", "jenkins",
    "linux", "git", "api", "rest",
}


def _is_it_job(raw):
    """Check if a job is genuinely IT/technical using title + skills."""
    title = (raw.get("title") or "").lower()
    skills = [s.lower() for s in (raw.get("skills") or [])]

    # Reject if title matches a non-IT pattern
    for kw in _NON_IT_TITLE_EXCLUDE:
        if kw in title:
            return False

    # Accept if title matches an IT pattern
    for kw in _IT_TITLE_INCLUDE:
        if kw in title:
            return True

    # Accept if has 2+ technical skills (catches roles like "ML Risk Analyst")
    tech_count = sum(1 for s in skills if s in _TECH_SKILLS)
    if tech_count >= 2:
        return True

    return False

_SENIORITY_KEYWORDS = {
    "senior": "Senior", "lead": "Senior", "principal": "Senior", "staff": "Senior",
    "junior": "Junior", "entry": "Junior", "intern": "Junior", "graduate": "Junior",
    "manager": "Manager", "director": "Manager", "head": "Manager", "vp": "Manager",
}

_SKILL_NORMALISE = {
    "python": "Python", "java": "Java", "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "react": "React", "nodejs": "Node.js", "node": "Node.js",
    "aws": "AWS", "azure": "Azure", "gcp": "GCP",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "sql": "SQL", "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
    "machine-learning": "Machine Learning", "ml": "Machine Learning",
    "deep-learning": "Deep Learning", "nlp": "NLP",
    "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "ai": "AI", "llm": "LLM", "langchain": "LangChain",
    "go": "Go", "golang": "Go", "rust": "Rust",
    "graphql": "GraphQL", "redis": "Redis", "kafka": "Kafka",
    "spark": "Spark", "airflow": "Airflow",
    "terraform": "Terraform", "ansible": "Ansible",
    "ci-cd": "CI/CD", "cicd": "CI/CD",
    "git": "Git", "linux": "Linux",
}

_US_CITY_STATE = {
    "san francisco": "CA", "mountain view": "CA", "palo alto": "CA",
    "menlo park": "CA", "los gatos": "CA", "sunnyvale": "CA",
    "san jose": "CA", "oakland": "CA", "berkeley": "CA",
    "los angeles": "CA", "san diego": "CA", "irvine": "CA", "santa monica": "CA",
    "seattle": "WA", "redmond": "WA", "bellevue": "WA",
    "new york": "NY", "brooklyn": "NY", "manhattan": "NY",
    "austin": "TX", "dallas": "TX", "houston": "TX", "san antonio": "TX",
    "chicago": "IL", "boston": "MA", "cambridge": "MA",
    "denver": "CO", "boulder": "CO",
    "portland": "OR", "washington": "DC",
    "philadelphia": "PA", "pittsburgh": "PA",
    "atlanta": "GA", "miami": "FL", "orlando": "FL",
    "detroit": "MI", "minneapolis": "MN",
    "raleigh": "NC", "charlotte": "NC",
    "salt lake city": "UT", "phoenix": "AZ",
    "nashville": "TN", "richmond": "VA",
}

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}


def _seniority_from_title(title):
    t = (title or "").lower()
    for kw, level in _SENIORITY_KEYWORDS.items():
        if kw in t:
            return level
    return "Mid"


def _normalise_skill(raw_skill):
    s = raw_skill.lower()
    if s in _SKILL_NORMALISE:
        return _SKILL_NORMALISE[s]
    return raw_skill.replace("-", " ").title()


def _extract_state(raw):
    """
    Extract US state code from a freehire job dict.
    Priority: location string (e.g. 'Minneapolis, MN') -> cities -> countries.
    """
    # 1. Try location string first (most reliable when present)
    loc = raw.get("location") or ""
    parts = [p.strip() for p in loc.split(",")]
    # "Minneapolis, MN" -> parts[-1] = "MN"; "Remote (U.S. Only)" no match
    for p in reversed(parts):
        p_upper = p.upper().rstrip(".")
        if p_upper in _US_STATES:
            return p_upper

    # 2. Try city lookup
    cities = raw.get("cities") or []
    if cities:
        city_lower = cities[0].strip().lower()
        if city_lower in _US_CITY_STATE:
            return _US_CITY_STATE[city_lower]

    # 3. Fallback to country code
    countries = raw.get("countries") or []
    if countries:
        return countries[0].upper()[:20]
    return "US"


def _normalise(raw):
    loc = raw.get("location") or "Remote"
    state = _extract_state(raw)
    title = raw.get("title") or "Unknown"

    posted_raw = raw.get("posted_at", "")
    try:
        posted_date = datetime.fromisoformat(posted_raw.replace("Z", "+00:00")).date().isoformat()
    except (ValueError, TypeError):
        posted_date = datetime.now().date().isoformat()

    skills = [_normalise_skill(s) for s in (raw.get("skills") or [])]

    return {
        "title": title,
        "company": raw.get("company") or "Unknown",
        "location": loc,
        "state": state,
        "skills": skills[:8],
        "experience": _seniority_from_title(title),
        "salary": None,
        "posted_date": posted_date,
        "source": "freehire/api",
        "url": raw.get("url") or "",
    }


def can_call(now=None):
    now = now or time.time()
    with _lock:
        return (now - _last_call_ts) >= _COOLDOWN_SECONDS


def _set_cooldown():
    global _last_call_ts
    with _lock:
        _last_call_ts = time.time()


def fetch_jobs(max_results=30):
    """
    Pull a fresh batch of IT jobs from freehire.dev.
    Returns a dict with 'jobs' list, or dict with 'cooldown', or None on failure.
    """
    if not can_call():
        logger.info("freehire on cooldown; skipping live call.")
        return {"cooldown": True, "source": "freehire/api", "jobs": []}

    headers = {
        "User-Agent": "ITJobMarket/1.0 (Student Project)",
    }

    params = [("countries", "us"), ("limit", str(min(max_results, 100)))]
    # freehire uses repeated params for OR semantics
    for cat in _IT_CATEGORIES:
        params.append(("category", cat))

    jobs = []
    try:
        resp = requests.get(_FREEHIRE_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        raw_jobs = body.get("data", []) if isinstance(body, dict) else []

        for raw in raw_jobs:
            if _is_it_job(raw):
                jobs.append(_normalise(raw))
            if len(jobs) >= max_results:
                break

    except Exception as e:
        logger.warning("freehire live call failed: %s", e)
        return None

    if not jobs:
        logger.info("freehire returned 0 IT jobs.")
        return None

    _set_cooldown()
    return {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "freehire/api",
        "jobs": jobs,
    }
