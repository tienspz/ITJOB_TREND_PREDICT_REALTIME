"""
Core router — health check, metadata, chart discovery, figure serving.
"""

import os
from flask import Blueprint, jsonify, send_from_directory

from backend import config
from backend import model_registry

bp = Blueprint("core", __name__)

# Dataset-level stats used by the static dashboard (previously rendered
# server-side with Jinja2). Computed once from the processed CSV and cached.
_dataset_stats_cache = None


def _dataset_stats():
    global _dataset_stats_cache
    if _dataset_stats_cache is None:
        stats = {}
        if os.path.exists(config.PROCESSED_DATA_FILE):
            try:
                import pandas as pd
                skill_cols = ["skill_programming", "skill_cloud", "skill_ai_ml", "skill_database"]
                df = pd.read_csv(config.PROCESSED_DATA_FILE, usecols=["it_domain"] + skill_cols)
                stats = {
                    "data_rows": int(len(df)),
                    "top_domain": str(df["it_domain"].value_counts().index[0]),
                    "top_skill": df[skill_cols].sum().idxmax().replace("skill_", "").replace("_", " ").title(),
                }
            except Exception:
                stats = {}
        _dataset_stats_cache = stats
    return _dataset_stats_cache


@bp.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": model_registry.health(),
    })


@bp.route("/api/meta", methods=["GET"])
def get_metadata():
    meta = model_registry.all_meta()
    meta["salary_meta"] = {**(meta.get("salary_meta") or {}), **_dataset_stats()}
    return jsonify(meta)


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
