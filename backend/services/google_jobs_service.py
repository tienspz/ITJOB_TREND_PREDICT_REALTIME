"""
Google Jobs realtime provider.

Uses SerpAPI's Google Jobs endpoint if a SERPAPI_KEY is present in the
environment. The project currently has no SerpAPI budget, so when the key is
absent this module is a clean no-op (returns None) and trend_service falls
back to freehire_service / remoteok_service / cache.

The interface mirrors the other providers so they are
interchangeable from trend_service's point of view.
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

_SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
_SERPAPI_URL = "https://serpapi.com/search.json"


def is_configured():
    return bool(_SERPAPI_KEY)


def fetch_jobs(max_results=30):
    """Pull Google Jobs via SerpAPI. Returns None if unavailable/failed."""
    if not is_configured():
        logger.info("No SERPAPI_KEY configured; Google Jobs provider disabled.")
        return None

    try:
        params = {
            "engine": "google_jobs",
            "q": "IT software engineer",
            "api_key": _SERPAPI_KEY,
        }
        resp = requests.get(_SERPAPI_URL, params=params, timeout=8)
        resp.raise_for_status()
        raw_jobs = resp.json().get("jobs_results", []) or []
    except Exception as e:
        logger.warning("SerpAPI Google Jobs call failed: %s", e)
        return None

    jobs = []
    for raw in raw_jobs[:max_results]:
        title = raw.get("title", "Unknown")
        loc = raw.get("location", "Remote")
        desc = raw.get("description", "") + " " + " ".join(raw.get("detected_extensions", {}).get("skills", []))
        jobs.append({
            "title": title,
            "company": raw.get("company_name", "Unknown"),
            "location": loc,
            "state": loc.split(",")[-1].strip()[:20] if "," in loc else loc,
            "skills": _extract_skills(desc + " " + title),
            "experience": _seniority(title),
            "salary": None,
            "posted_date": raw.get("detected_extensions", {}).get("posted_at", ""),
            "source": "google_jobs/serpapi",
            "url": "",
        })

    if not jobs:
        return None
    return {"source": "google_jobs/serpapi", "jobs": jobs}


def _seniority(title):
    t = (title or "").lower()
    if any(k in t for k in ("senior", "lead", "principal")):
        return "Senior"
    if any(k in t for k in ("junior", "entry", "intern")):
        return "Junior"
    return "Mid"


def _extract_skills(text):
    text_l = (text or "").lower()
    found = []
    for kw in ("python", "java", "javascript", "react", "aws", "azure", "docker",
              "kubernetes", "sql", "spark", "pytorch", "tensorflow", "machine learning"):
        if kw in text_l:
            found.append(kw.title() if kw[0].isalpha() else kw)
    return found[:6]
