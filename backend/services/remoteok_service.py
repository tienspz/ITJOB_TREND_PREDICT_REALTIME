"""
RemoteOK API realtime provider.

Replaces the legacy RapidAPI JSearch provider. The RemoteOK API is free
and does not require an API key.

Responsibility:
  * Fetch job listings from RemoteOK API.
  * Filter for IT-related jobs (using tags/title).
  * Normalise into the canonical realtime schema.
  * Feed discovered links to the offline scraper queue.
"""

import os
import time
import threading
import logging
from datetime import datetime

import requests

from backend import config

logger = logging.getLogger(__name__)

_REMOTEOK_URL = "https://remoteok.com/api"
_COOLDOWN_SECONDS = 600  # 10 minutes (RemoteOK requests minimal polling)
_last_call_ts = 0.0
_lock = threading.Lock()

# Title & tag patterns to identify genuine IT jobs on RemoteOK
_IT_TITLE_INCLUDE = {
    "software", "engineer", "developer", "programmer",
    "devops", "sre", "site reliability",
    "data engineer", "data scientist", "data architect",
    "machine learning", "ml engineer", "ml ops",
    "ai engineer", "ai scientist", "ai research",
    "platform engineer", "infrastructure", "cloud",
    "security engineer", "cyber security", "application security",
    "full stack", "fullstack", "backend", "frontend",
    "software architect", "solutions architect", "tech lead", "technical lead",
    "systems engineer", "network engineer", "database",
    "qa engineer", "test engineer", "automation engineer",
    "reliability engineer", "production engineer",
    "blockchain", "embedded", "firmware", "hardware",
    "data analyst", "business intelligence", "bi ",
    "research scientist", "applied scientist",
    "product manager", "technical product",
    "ux ", "ui ", "interface designer", "web designer", "product designer",
    "graduate data scientist", "data scientist",
}

_NON_IT_TITLE_EXCLUDE = {
    "data entry", "data entry clerk", "data entry specialist",
    "pool technician", "handyperson", "procurement",
    "english teacher", "teacher", "tutor",
    "sales executive", "sales representative", "sales manager",
    "customer success", "customer service", "customer support",
    "human resources", "hr ", "recruiter", "talent acquisition",
    "estimator", "driver", "delivery",
    "administrative", "administrator", "executive assistant",
    "receptionist", "office assistant",
    "warehouse", "logistics", "supply chain",
    "legal", "paralegal", "compliance",
    "marketing", "content creator", "social media",
    "architectural", "drafter", "drafting",
    "job fair", "general applicant", "future openings",
    "virtual assistant", "video editor", "creative strategist",
    "vehicle imager", "the role", "hiring operations",
    "client success",
    "video editor", "video production",
    "off season", "sports",
}


def _is_it_job(job):
    title = (job.get("position") or "").lower()
    tags = [str(t).lower() for t in job.get("tags", [])]

    # Reject non-IT title patterns first
    for kw in _NON_IT_TITLE_EXCLUDE:
        if kw in title:
            return False

    # Accept if title matches an IT pattern
    for kw in _IT_TITLE_INCLUDE:
        if kw in title:
            return True

    # Accept if tags contain tech keywords
    _TECH_TAGS = {"python", "javascript", "react", "node.js", "aws", "docker",
                  "kubernetes", "devops", "backend", "frontend", "full stack",
                  "data science", "machine learning", "ai", "deep learning",
                  "software", "engineering", "cloud", "security", "dev"}
    for tag in tags:
        if tag in _TECH_TAGS:
            return True

    return False

def _seniority_from_title(title):
    t = (title or "").lower()
    if any(k in t for k in ("senior", "lead", "principal", "staff")):
        return "Senior"
    if any(k in t for k in ("junior", "entry", "intern", "graduate")):
        return "Junior"
    if any(k in t for k in ("manager", "director", "head", "vp")):
        return "Manager"
    return "Mid"

def _extract_skills(job):
    tags = job.get("tags", [])
    title = (job.get("position") or "").lower()
    
    # Common mappings
    skill_map = {
        "python": "Python", "java": "Java", "javascript": "JavaScript", "js": "JavaScript",
        "react": "React", "node": "Node.js", "aws": "AWS", "azure": "Azure", "gcp": "GCP",
        "docker": "Docker", "kubernetes": "Kubernetes", "sql": "SQL", "machine learning": "Machine Learning",
        "ai": "AI", "llm": "LLM", "pytorch": "PyTorch", "tensorflow": "TensorFlow", "go": "Go", "golang": "Go"
    }
    
    found_skills = []
    seen = set()
    
    for tag in tags:
        tag_l = str(tag).lower()
        if tag_l in skill_map and skill_map[tag_l] not in seen:
            found_skills.append(skill_map[tag_l])
            seen.add(skill_map[tag_l])
            
    # Fallback checking title
    for k, v in skill_map.items():
        if k in title and v not in seen:
            found_skills.append(v)
            seen.add(v)
            
    return found_skills[:8]

def _normalise(raw):
    """Map one RemoteOK result into the canonical realtime schema."""
    loc = raw.get("location") or "Remote"
    state_parsed = loc.split(",")[-1].strip() if "," in loc else loc
    state = state_parsed[:20] if state_parsed else "Remote"
    
    salary_min = raw.get("salary_min")
    salary_max = raw.get("salary_max")
    salary = None
    if salary_min and salary_max:
        salary = (salary_min + salary_max) / 2
    elif salary_min:
        salary = salary_min
    elif salary_max:
        salary = salary_max
        
    date_str = raw.get("date", "")
    try:
        posted_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date().isoformat()
    except (ValueError, TypeError):
        posted_date = datetime.now().date().isoformat()
        
    return {
        "title": raw.get("position") or "Unknown",
        "company": raw.get("company") or "Unknown",
        "location": loc,
        "state": state,
        "skills": _extract_skills(raw),
        "experience": _seniority_from_title(raw.get("position")),
        "salary": salary,
        "posted_date": posted_date,
        "source": "remoteok/api",
        "url": raw.get("url") or raw.get("apply_url") or "",
    }

def can_call(now=None):
    """True if the cooldown has elapsed."""
    now = now or time.time()
    with _lock:
        return (now - _last_call_ts) >= _COOLDOWN_SECONDS

def _set_cooldown():
    global _last_call_ts
    with _lock:
        _last_call_ts = time.time()

def fetch_jobs(max_results=30):
    """
    Pull a fresh batch of IT jobs from RemoteOK.
    Returns a list of normalised job dicts, or None on failure.
    Returns a dict with 'cooldown' key if cooldown is active.
    """
    if not can_call():
        logger.info("RemoteOK on cooldown; skipping live call.")
        return {"cooldown": True, "source": "remoteok/api", "jobs": []}

    headers = {
        "User-Agent": "ITJobMarket/1.0 (Student Project)",
    }

    jobs = []
    try:
        resp = requests.get(_REMOTEOK_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Skip the first element as it contains TOS/legal info
        if data and isinstance(data, list):
            data = data[1:]
            
        for raw in data:
            if _is_it_job(raw):
                jobs.append(_normalise(raw))
            if len(jobs) >= max_results:
                break
                
    except Exception as e:
        logger.warning("RemoteOK live call failed: %s", e)
        return None

    if not jobs:
        logger.info("RemoteOK returned 0 IT jobs (filtered out all results).")
        return None

    _set_cooldown()
    return {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "remoteok/api",
        "jobs": jobs,
    }

def add_to_queue(links):
    """
    Persist freshly discovered job links into the scraper's pending queue.
    """
    if not links:
        return 0
    import pandas as pd
    new_rows = [{"job_link": lnk} for lnk in links if lnk]
    if not new_rows:
        return 0
    try:
        if os.path.exists(config.PENDING_LINKS_FILE):
            df = pd.read_csv(config.PENDING_LINKS_FILE)
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df[~df["job_link"].astype(str).duplicated(keep="last")]
        else:
            df = pd.DataFrame(new_rows)
        df.to_csv(config.PENDING_LINKS_FILE, index=False)
        return len(new_rows)
    except Exception as e:
        logger.warning("add_to_queue failed: %s", e)
        return 0
