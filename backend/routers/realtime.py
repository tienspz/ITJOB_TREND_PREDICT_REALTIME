"""
Realtime router — live market intelligence endpoints.

This is the lecturer-mandated realtime pipeline surface. Every endpoint
delegates to trend_service, which implements the provider chain
(freehire.dev -> RemoteOK -> cache -> snapshot) so the dashboard
ALWAYS returns usable data even when APIs are rate-limited.
"""

import numpy as np
from flask import Blueprint, jsonify

from backend.services import trend_service

bp = Blueprint("realtime", __name__)


@bp.route("/api/realtime/jobs", methods=["GET"])
def realtime_jobs():
    """Normalised realtime job feed."""
    return jsonify(trend_service.get_jobs_feed())


@bp.route("/api/realtime/skills", methods=["GET"])
def realtime_skills():
    """Top skills (24h / 7d) + Historical vs Realtime comparison."""
    return jsonify(trend_service.get_skills_view())


@bp.route("/api/realtime/trends", methods=["GET"])
def realtime_trends():
    """Hiring velocity + emerging roles + historical comparison."""
    return jsonify(trend_service.get_trends_view())


@bp.route("/api/realtime/locations", methods=["GET"])
def realtime_locations():
    return jsonify(trend_service.get_locations_view())


@bp.route("/api/realtime/companies", methods=["GET"])
def realtime_companies():
    return jsonify(trend_service.get_companies_view())


@bp.route("/api/realtime/report", methods=["GET"])
def realtime_report():
    """The full bundle (used by the realtime dashboard tab)."""
    return jsonify(trend_service.build_realtime_report())


# ---------------------------------------------------------------------------
# Convenience endpoints from the project brief
# ---------------------------------------------------------------------------
@bp.route("/api/trending", methods=["GET"])
def trending():
    """Compact {top_skills, top_jobs, top_locations} view."""
    return jsonify(trend_service.get_trending())


@bp.route("/api/realtime-jobs", methods=["GET"])
def realtime_jobs_brief():
    """Brief alias: pull freehire.dev + RemoteOK, normalise."""
    return jsonify(trend_service.get_jobs_feed())
