"""
Realtime history accumulator.

The Kaggle dataset is a single-week snapshot (first_seen: 2024-01-12..17), so
genuine longitudinal trends cannot come from it. This module builds the
project's OWN time series: every realtime fetch appends one snapshot row to
data/realtime_history.csv (deduplicated per hour). The longer the system
runs, the more real trend data it owns.

Row schema:
    timestamp        ISO datetime of the snapshot (hour precision)
    source           live | cache | snapshot
    total_jobs       job count in the payload
    top_skill        most frequent skill at that moment
    top_skill_share  its share (%)
    domain_counts    "Software Engineering:5|Data Science:3|..." pairs
"""

import csv
import logging
import os
from collections import Counter
from datetime import datetime

from backend import config

logger = logging.getLogger(__name__)

HISTORY_FILE = os.path.join(config.DATA_DIR, "realtime_history.csv")
_FIELDS = ["timestamp", "source", "total_jobs", "top_skill", "top_skill_share", "domain_counts"]

# Map realtime job titles onto the 5 historical IT domains for comparability
_DOMAIN_KW = {
    "Data Science": ["data", "ai", "machine learning", "nlp", "analytics"],
    "DevOps/SRE": ["cloud", "devops", "sre", "system", "network", "infrastructure", "platform"],
    "Cybersecurity": ["security", "cyber"],
    "QA/Testing": ["test", "qa", "quality"],
}


def _job_domain(title):
    t = str(title or "").lower()
    for domain, kws in _DOMAIN_KW.items():
        if any(k in t for k in kws):
            return domain
    return "Software Engineering"


def _last_recorded_hour():
    if not os.path.exists(HISTORY_FILE):
        return None
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            last = None
            for row in csv.DictReader(f):
                last = row
            if last:
                return last["timestamp"][:13]  # YYYY-MM-DDTHH
    except (OSError, KeyError):
        pass
    return None


def record_snapshot(report):
    """Append one history row per hour from a realtime report bundle.

    Only 'live' and 'cache' payloads are recorded — the built-in demo
    snapshot would pollute the series with constant fake numbers.
    """
    source = report.get("source", "")
    if source not in ("live", "cache"):
        return False

    now = datetime.now()
    hour_key = now.strftime("%Y-%m-%dT%H")
    if _last_recorded_hour() == hour_key:
        return False  # already recorded this hour

    jobs = report.get("jobs", []) or []
    top_skills = report.get("top_skills_7d") or report.get("top_skills") or []
    top = top_skills[0] if top_skills else {}

    domains = Counter(_job_domain(j.get("title")) for j in jobs)
    domain_str = "|".join(f"{d}:{c}" for d, c in domains.most_common())

    row = {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "total_jobs": report.get("total_jobs", len(jobs)),
        "top_skill": top.get("skill", ""),
        "top_skill_share": top.get("share", ""),
        "domain_counts": domain_str,
    }

    try:
        new_file = not os.path.exists(HISTORY_FILE)
        with open(HISTORY_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Recorded realtime history snapshot (%s, %s jobs)", source, row["total_jobs"])
        return True
    except OSError as e:
        logger.warning("Could not record history snapshot: %s", e)
        return False


def load_history():
    """Return the accumulated snapshots as a list of dicts (oldest first)."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []
