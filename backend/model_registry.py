"""
Model registry — single source of truth for loaded ML models.

server.py loads the .joblib artifacts once at startup and registers them
here; routers read them via get_*() helpers. This avoids passing Flask app
context around and keeps the routers thin.
"""

import os
import joblib

from backend import config


class Registry:
    salary_model = None
    demand_model = None
    cluster_model = None
    salary_meta = {}
    demand_meta = {}
    cluster_meta = {}


def load_all(logger=None):
    """Load every model artifact into RAM. Safe to call once at startup."""
    def _log(msg):
        if logger:
            logger.info(msg)

    pairs = [
        ("salary", "best_salary_model.joblib", "salary_model_meta.joblib"),
        ("demand", "demand_model.joblib", "demand_meta.joblib"),
        ("cluster", "cluster_model.joblib", "cluster_meta.joblib"),
    ]
    for kind, model_file, meta_file in pairs:
        try:
            model = joblib.load(os.path.join(config.MODELS_DIR, model_file))
            meta = {}
            try:
                meta = joblib.load(os.path.join(config.MODELS_DIR, meta_file))
            except Exception:
                pass
            setattr(Registry, f"{kind}_model", model)
            setattr(Registry, f"{kind}_meta", meta if isinstance(meta, dict) else {})
            _log(f"{kind} model loaded.")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load {kind} model: {e}")


def get_salary_model():
    return Registry.salary_model


def get_demand_model():
    return Registry.demand_model


def get_cluster_model():
    return Registry.cluster_model


def get_salary_meta():
    return Registry.salary_meta


def get_demand_meta():
    return Registry.demand_meta


def get_cluster_meta():
    return Registry.cluster_meta


def _json_safe(obj):
    """Make a metadata dict JSON-serialisable.

    The cluster_meta stores cluster_descriptions with BOTH int and str keys
    (e.g. 0 and '0'), which Python's json can't sort. Normalise all dict keys
    to strings so jsonify() works.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def all_meta():
    return {
        "salary_meta": _json_safe(Registry.salary_meta),
        "demand_meta": _json_safe(Registry.demand_meta),
        "cluster_meta": _json_safe(Registry.cluster_meta),
    }


def health():
    return {
        "salary": Registry.salary_model is not None,
        "demand": Registry.demand_model is not None,
        "cluster": Registry.cluster_model is not None,
    }
