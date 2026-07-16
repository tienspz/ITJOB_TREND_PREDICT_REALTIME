"""
Core router — health check, metadata, chart discovery, figure serving.
"""

import os
from flask import Blueprint, jsonify, send_from_directory

from backend import config
from backend import model_registry

bp = Blueprint("core", __name__)


@bp.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": model_registry.health(),
    })


@bp.route("/api/meta", methods=["GET"])
def get_metadata():
    return jsonify(model_registry.all_meta())


@bp.route("/api/charts", methods=["GET"])
def charts():
    """
    Return the set of generated PNG report figures plus their metadata.
    Used by the dashboard to know which images exist without hard-coding.
    """
    figures = []
    if os.path.isdir(config.REPORTS_DIR):
        for fn in sorted(os.listdir(config.REPORTS_DIR)):
            if fn.lower().endswith(".png"):
                figures.append({"filename": fn, "url": f"/reports/figures/{fn}"})

    # Merge in figure_metadata.json if present (titles, metrics, etc.)
    meta_by_name = {}
    meta_path = os.path.join(config.REPORTS_DIR, "figure_metadata.json")
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                for row in json.load(f):
                    meta_by_name[row.get("filename")] = row
        except (ValueError, OSError):
            pass

    for fig in figures:
        fig.update(meta_by_name.get(fig["filename"], {}))

    return jsonify({"charts": figures})


@bp.route("/reports/figures/<path:filename>", methods=["GET"])
def serve_figure(filename):
    return send_from_directory(config.REPORTS_DIR, filename)
