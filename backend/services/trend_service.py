"""
Trend orchestration service.

Single entry point for all realtime intelligence. Implements the provider
chain and the analytics the dashboard needs:

    Provider chain:  freehire.dev  ->  RemoteOK  ->  Google Jobs  ->  cache  ->  snapshot

Analytics computed from the (always-available) job list:
  * Top skills (24h / 7d)
  * Top hiring locations
  * Top companies
  * Top emerging roles
  * Hiring velocity (postings per day)
  * Historical (Kaggle) vs Realtime skill comparison + auto conclusion

The result is a dict the realtime router serialises straight to JSON.
Nothing here raises to the caller — the demo always gets usable data.
"""

import os
import logging
from collections import Counter
from datetime import datetime, timedelta

from backend import config
from backend.services import freehire_service, remoteok_service, google_jobs_service, realtime_cache
from backend.services import history_service

logger = logging.getLogger(__name__)

# Skills whose growth vs the historical Kaggle baseline is "interesting".
# Used to phrase the auto-generated conclusion.
_EMERGING_SKILLS = {
    "AI", "LLM", "Langchain", "PyTorch", "TensorFlow",
    "Machine Learning", "Deep Learning", "NLP",
}


# ---------------------------------------------------------------------------
# Provider chain
# ---------------------------------------------------------------------------
def _get_payload():
    """
    Resolve the freshest job payload.

    Returns (payload, source_label). source_label is one of:
      'live'      - a genuine API call succeeded this request
      'cache'     - a recent (<1h) live snapshot from disk
      'snapshot'  - built-in realistic snapshot (demo fallback)
    """
    if freehire_service.can_call():
        live = freehire_service.fetch_jobs()
        if live and live.get("jobs"):
            realtime_cache.save_cache(live)
            return live, "live"

    if remoteok_service.can_call():
        live = remoteok_service.fetch_jobs()
        if live and live.get("jobs"):
            realtime_cache.save_cache(live)
            links = [j.get("url") for j in live.get("jobs", []) if j.get("url")]
            remoteok_service.add_to_queue(links)
            return live, "live"

    if google_jobs_service.is_configured():
        live = google_jobs_service.fetch_jobs()
        if live:
            realtime_cache.save_cache(live)
            return live, "live"

    cached = realtime_cache.load_cache()
    if cached:
        return cached, "cache"

    return realtime_cache.get_snapshot(), "snapshot"


# ---------------------------------------------------------------------------
# Analytics primitives
# ---------------------------------------------------------------------------
def _top_skills(jobs, top_n=10):
    c = Counter()
    for j in jobs:
        for s in j.get("skills", []) or []:
            c[s] += 1
    total = sum(c.values()) or 1
    return [{"skill": s, "count": n, "share": round(n / total * 100, 1)} for s, n in c.most_common(top_n)]


def _top_by_field(jobs, field, top_n=10):
    c = Counter(j.get(field) for j in jobs if j.get(field))
    total = sum(c.values()) or 1
    return [{field: k, "count": n, "share": round(n / total * 100, 1)} for k, n in c.most_common(top_n)]


def _hiring_velocity(jobs):
    """Postings per day over the last 7 days."""
    today = datetime.now().date()
    buckets = Counter()
    for j in jobs:
        d = j.get("posted_date", "")
        try:
            posted = datetime.fromisoformat(d).date()
        except (ValueError, TypeError):
            continue
        days_ago = (today - posted).days
        if 0 <= days_ago <= 6:
            buckets[days_ago] += 1
    series = []
    for back in range(6, -1, -1):
        day = today - timedelta(days=back)
        series.append({"date": day.isoformat(), "count": buckets.get(back, 0)})
    return series


# ---------------------------------------------------------------------------
# Historical baseline (computed once, cached on the function attribute)
# ---------------------------------------------------------------------------
_HIST_CACHE = {"skills": None}


def _historical_skill_baseline():
    """
    Skill frequency from the Kaggle-derived dataset. Used as the
    'Historical Intelligence' side of the comparison.

    Computed lazily and cached; falls back to a documented distribution if
    the processed CSV is not present.
    """
    if _HIST_CACHE["skills"] is not None:
        return _HIST_CACHE["skills"]

    try:
        import pandas as pd
        if os.path.exists(config.PROCESSED_DATA_FILE):
            df = pd.read_csv(config.PROCESSED_DATA_FILE, usecols=lambda c: c.startswith("skill_"))
            totals = {c.replace("skill_", ""): int(df[c].sum()) for c in df.columns}
            grand = sum(totals.values()) or 1
            baseline = {k: round(v / grand * 100, 1) for k, v in totals.items()}
            _HIST_CACHE["skills"] = baseline
            return baseline
    except Exception as e:
        logger.info("Historical baseline from CSV failed (%s); using fallback.", e)

    # Fallback distribution documented from the Kaggle IT subset.
    _HIST_CACHE["skills"] = {
        "programming": 42.0, "database": 18.5, "framework": 12.0,
        "cloud": 10.5, "devops": 8.0, "ai_ml": 5.0,
        "data_engineering": 3.0, "security": 1.0,
    }
    return _HIST_CACHE["skills"]


# Canonical realtime skill token -> historical bucket (for comparison).
_SKILL_TO_BUCKET = {
    "Python": "programming", "Java": "programming", "Javascript": "programming",
    "JavaScript": "programming", "Go": "programming", "Sql": "database", "SQL": "database",
    "Postgresql": "database", "Mongodb": "database", "React": "framework", "Node.Js": "framework",
    "Spring": "framework", "Graphql": "framework", "Aws": "cloud", "Azure": "cloud", "Gcp": "cloud",
    "Docker": "devops", "Kubernetes": "devops", "Terraform": "devops", "Ci/Cd": "devops",
    "Pytorch": "ai_ml", "Tensorflow": "ai_ml", "Machine Learning": "ai_ml", "Deep Learning": "ai_ml",
    "AI": "ai_ml", "Nlp": "ai_ml", "Llm": "ai_ml", "Langchain": "ai_ml",
    "Spark": "data_engineering", "Airflow": "data_engineering", "Kafka": "data_engineering",
    "Cybersecurity": "security",
}


def _compare_skills(realtime_skills):
    """
    Build a side-by-side Historical vs Realtime comparison and auto-phrase
    a one-line conclusion the dashboard can show directly.
    """
    hist = _historical_skill_baseline()

    # Map realtime skill names to historical buckets, then aggregate shares.
    bucket_shares = {}
    for row in realtime_skills:
        bucket = _SKILL_TO_BUCKET.get(row["skill"], row["skill"].lower().replace(" ", "_"))
        bucket_shares[bucket] = bucket_shares.get(bucket, 0.0) + row["share"]

    all_tokens = set(hist) | set(bucket_shares)

    rows = []
    for tok in all_tokens:
        h = hist.get(tok, 0.0)
        r = bucket_shares.get(tok, 0.0)
        delta = round(r - h, 1)
        rows.append({
            "skill": tok,
            "historical_share": h,
            "realtime_share": r,
            "delta": delta,
        })
    rows.sort(key=lambda x: x["delta"], reverse=True)

    # Auto conclusion: pick the biggest positive mover from the AI bucket.
    conclusion = "Phân bố kỹ năng realtime gần với baseline lịch sử."
    gainers = [r for r in rows if r["delta"] > 1.5]
    if gainers:
        top_gain = gainers[0]
        if top_gain["skill"] in ("ai_ml", "AI", "LLM", "Langchain"):
            conclusion = (
                f"Nhu cầu kỹ năng AI/GenAI tăng {top_gain['delta']:.1f} điểm phần trăm so với dataset lịch sử — "
                "dấu hiệu rõ của làn sóng AI Engineering."
            )
        else:
            label = top_gain["skill"].replace("_", " ").title()
            conclusion = (
                f"Nhu cầu kỹ năng '{label}' tăng {top_gain['delta']:.1f} điểm phần trăm so với baseline lịch sử."
            )
    return rows, conclusion


# ---------------------------------------------------------------------------
# Public aggregation
# ---------------------------------------------------------------------------
def build_realtime_report():
    """Full realtime intelligence bundle for the dashboard."""
    payload, source = _get_payload()
    jobs = payload.get("jobs", [])

    skills = _top_skills(jobs)
    comparison, conclusion = _compare_skills(skills)

    # 24h vs 7d skill splits
    today = datetime.now().date()
    jobs_24h = [j for j in jobs if _days_ago(j.get("posted_date"), today) <= 1]
    jobs_7d = [j for j in jobs if _days_ago(j.get("posted_date"), today) <= 7]

    report = {
        "fetched_at": payload.get("fetched_at", datetime.now().isoformat(timespec="seconds")),
        "source": source,
        "raw_source": payload.get("source", source),
        "total_jobs": len(jobs),
        "jobs": jobs,
        "top_skills": skills,
        "top_skills_24h": _top_skills(jobs_24h),
        "top_skills_7d": _top_skills(jobs_7d),
        "top_locations": _top_by_field(jobs, "state"),
        "top_companies": _top_by_field(jobs, "company"),
        "top_roles": _top_by_field(jobs, "title", top_n=8),
        "emerging_roles": [
            r for r in _top_by_field(jobs, "title", top_n=8)
            if any(k in r["title"].lower() for k in ("ai", "ml", "machine", "data", "cloud", "devops"))
        ][:5],
        "hiring_velocity": _hiring_velocity(jobs),
        "comparison": {
            "rows": comparison[:10],
            "conclusion": conclusion,
            "historical_source": "Kaggle 1.3M IT Jobs (2024)",
        },
        "meta": {
            "is_live": source == "live",
            "note": _source_note(source),
        },
    }

    # Accumulate the project's own longitudinal series (1 row/hour max).
    try:
        history_service.record_snapshot(report)
    except Exception as e:  # never let history-keeping break the dashboard
        logger.warning("history snapshot failed: %s", e)

    return report


def _source_note(source):
    return {
        "live": "Dữ liệu realtime trực tiếp từ freehire.dev (2.9M IT jobs, crawl trực tiếp từ ATS).",
        "cache": "Dùng snapshot cache (call API thành công gần đây) — tránh rate-limit.",
        "snapshot": "API rate-limit/không khả dụng — dùng snapshot minh hoạ thực tế cho demo.",
    }.get(source, "")


def _days_ago(date_str, today):
    try:
        return (today - datetime.fromisoformat(date_str).date()).days
    except (ValueError, TypeError):
        return 999


# ---------------------------------------------------------------------------
# Lightweight end-user convenience views (used by /api/trending etc.)
# ---------------------------------------------------------------------------
def get_trending():
    r = build_realtime_report()
    return {
        "top_skills": r["top_skills"],
        "top_jobs": r["top_roles"],
        "top_locations": r["top_locations"],
        "source": r["source"],
        "fetched_at": r["fetched_at"],
    }


def get_jobs_feed():
    r = build_realtime_report()
    return {"jobs": r["jobs"], "total": r["total_jobs"], "source": r["source"], "fetched_at": r["fetched_at"]}


def get_skills_view():
    r = build_realtime_report()
    return {
        "top_skills": r["top_skills"],
        "top_skills_24h": r["top_skills_24h"],
        "top_skills_7d": r["top_skills_7d"],
        "comparison": r["comparison"],
        "source": r["source"],
    }


def get_trends_view():
    r = build_realtime_report()
    return {
        "hiring_velocity": r["hiring_velocity"],
        "comparison": r["comparison"],
        "emerging_roles": r["emerging_roles"],
        "source": r["source"],
    }


def get_locations_view():
    r = build_realtime_report()
    return {"top_locations": r["top_locations"], "source": r["source"]}


def get_companies_view():
    r = build_realtime_report()
    return {"top_companies": r["top_companies"], "source": r["source"]}
