"""
Prediction router — salary / demand / cluster / CV upload / simple predict.

All input validation is shared via the same schema helpers so behaviour
matches the original monolithic server.py exactly (frontend stays green).
"""

import os
import pandas as pd
from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from backend import config
from backend import model_registry
from backend.services.cv_parser import parse_cv_and_predict

bp = Blueprint("predict", __name__)


# ---------------------------------------------------------------------------
# Input validation helpers (preserved verbatim from the original server.py)
# ---------------------------------------------------------------------------
def _normalize_category_list(values):
    if not isinstance(values, (list, tuple, set)):
        return []
    return [str(v) for v in values if v is not None]


def _parse_int(value, default, min_value=None, max_value=None):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _parse_float(value, default, min_value=None, max_value=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _validate_category(value, valid_values, default):
    if value is None:
        return default
    value_str = str(value)
    if value_str in valid_values:
        return value_str
    for candidate in valid_values:
        if candidate.lower() == value_str.lower():
            return candidate
    return default


DEFAULT_INPUTS = {
    "num_skills": 5, "skill_diversity": 3,
    "skill_programming": 1, "skill_cloud": 0, "skill_ai_ml": 0,
    "skill_database": 0, "skill_devops": 0, "skill_framework": 1,
    "skill_data_engineering": 0, "skill_security": 0, "skill_soft_skills": 1,
    "salary_annual": 100000.0,
    "seniority_level": "Mid", "job_type": "Remote",
    "state": "CA", "it_domain": "Software Engineering",
}

FEATURE_VALIDATION = {
    "num_skills": {"type": "int", "min": 0, "max": 30},
    "skill_diversity": {"type": "int", "min": 0, "max": 10},
    "skill_programming": {"type": "int", "min": 0, "max": 1},
    "skill_cloud": {"type": "int", "min": 0, "max": 1},
    "skill_ai_ml": {"type": "int", "min": 0, "max": 1},
    "skill_database": {"type": "int", "min": 0, "max": 1},
    "skill_devops": {"type": "int", "min": 0, "max": 1},
    "skill_framework": {"type": "int", "min": 0, "max": 1},
    "skill_data_engineering": {"type": "int", "min": 0, "max": 1},
    "skill_security": {"type": "int", "min": 0, "max": 1},
    "skill_soft_skills": {"type": "int", "min": 0, "max": 1},
    "salary_annual": {"type": "float", "min": 0, "max": 1000000},
    "seniority_level": {"type": "category"},
    "job_type": {"type": "category"},
    "state": {"type": "category"},
    "it_domain": {"type": "category"},
}

SALARY_INPUT_SCHEMA = {k: v for k, v in FEATURE_VALIDATION.items() if k != "salary_annual"}
DEMAND_INPUT_SCHEMA = {k: v for k, v in FEATURE_VALIDATION.items()
                       if k in {"it_domain", "state", "seniority_level", "job_type"}}
CLUSTER_INPUT_SCHEMA = {k: v for k, v in FEATURE_VALIDATION.items() if k != "salary_annual"}


def validate_input(data, schema, meta_source=None):
    validated = {}
    for field, rules in schema.items():
        default = DEFAULT_INPUTS.get(field)
        if rules["type"] == "int":
            validated[field] = _parse_int(data.get(field, default), default,
                                          rules.get("min"), rules.get("max"))
        elif rules["type"] == "float":
            validated[field] = _parse_float(data.get(field, default), default,
                                            rules.get("min"), rules.get("max"))
        elif rules["type"] == "category":
            valid_values = _normalize_category_list(meta_source.get(field)) if meta_source else []
            if not valid_values:
                valid_values = [default] if default is not None else []
            validated[field] = _validate_category(data.get(field, default), valid_values, default)
        else:
            validated[field] = data.get(field, default)
    return validated


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------
@bp.route("/api/predict_salary", methods=["POST"])
def predict_salary():
    salary_model = model_registry.get_salary_model()
    if salary_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400
    try:
        features = validate_input(data, SALARY_INPUT_SCHEMA, model_registry.get_salary_meta())
        features['domain_seniority'] = features['it_domain'] + '_' + features['seniority_level']
        features['state_seniority'] = features['state'] + '_' + features['seniority_level']
        df = pd.DataFrame([features])
        prediction = salary_model.predict(df)[0]
        return jsonify({
            "predicted_salary": round(float(prediction), 2),
            "currency": "USD", "period": "Annual",
        })
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during salary prediction. Please check your inputs."}), 400


@bp.route("/api/predict", methods=["POST"])
def predict_simple():
    """
    Simplified salary prediction per the project brief.

    Body: {"experience": 3, "skill": "Python", "location": "Ho Chi Minh"}
    Returns: {"salary": <number>}

    Maps the friendly fields onto the model's feature space, using sensible
    defaults so a one-field form still produces a prediction.
    """
    salary_model = model_registry.get_salary_model()
    if salary_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json or {}
    exp = _parse_int(data.get("experience"), 3, min_value=0, max_value=30)
    skill = str(data.get("skill") or "Python")
    location = str(data.get("location") or "")

    seniority = "Junior" if exp <= 2 else ("Senior" if exp >= 6 else "Mid")

    features = dict(DEFAULT_INPUTS)
    features.update({
        "seniority_level": seniority,
        "skill_programming": 1,
        "skill_cloud": 1 if skill.lower() in ("aws", "azure", "gcp", "cloud") else 0,
        "skill_ai_ml": 1 if skill.lower() in ("python", "pytorch", "tensorflow", "machine learning", "ai") else 0,
        "skill_database": 1 if skill.lower() in ("sql", "mysql", "postgres", "mongodb") else 0,
        "num_skills": 5 + min(exp, 10),
    })

    # Try to honour the requested location/state if the model knows it.
    meta = model_registry.get_salary_meta()
    known_states = _normalize_category_list(meta.get("state"))
    if known_states:
        features["state"] = _validate_category(location, known_states, known_states[0])
    features['domain_seniority'] = features['it_domain'] + '_' + features['seniority_level']
    features['state_seniority'] = features['state'] + '_' + features['seniority_level']
    try:
        pred = float(salary_model.predict(pd.DataFrame([features]))[0])
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"predict_simple error: {e}")
        return jsonify({"error": "Prediction failed"}), 400
    return jsonify({
        "salary": round(pred, 2),
        "currency": "USD", "period": "Annual",
        "mapped": {"seniority_level": seniority, "state": features["state"]},
    })


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
@bp.route("/api/predict_demand", methods=["POST"])
def predict_demand():
    demand_model = model_registry.get_demand_model()
    if demand_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400
    try:
        features = validate_input(data, DEMAND_INPUT_SCHEMA, model_registry.get_demand_meta())
        df = pd.DataFrame([features])
        prediction = demand_model.predict(df)[0]
        prediction = max(0, min(100, prediction))
        return jsonify({
            "demand_score": round(float(prediction), 2),
            "max_score": 100,
            "interpretation": "High" if prediction > 75 else "Medium" if prediction > 40 else "Low",
        })
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Demand Prediction error: {e}")
        return jsonify({"error": "An error occurred during demand prediction. Please check your inputs."}), 400


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------
@bp.route("/api/cluster", methods=["POST"])
def cluster_job():
    cluster_model = model_registry.get_cluster_model()
    if cluster_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400
    try:
        features = validate_input(data, CLUSTER_INPUT_SCHEMA, model_registry.get_salary_meta())
        features['domain_seniority'] = features['it_domain'] + '_' + features['seniority_level']
        features['state_seniority'] = features['state'] + '_' + features['seniority_level']
        df = pd.DataFrame([features])
        cluster_id = int(cluster_model.predict(df)[0])

        cluster_meta = model_registry.get_cluster_meta()
        descriptions = cluster_meta.get("cluster_descriptions", cluster_meta)
        description = descriptions.get(str(cluster_id)) or descriptions.get(cluster_id, "Unknown Cluster")

        return jsonify({"cluster_id": cluster_id, "description": description})
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Clustering error: {e}")
        return jsonify({"error": "An error occurred during clustering. Please check your inputs."}), 400


# ---------------------------------------------------------------------------
# CV upload
# ---------------------------------------------------------------------------
def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@bp.route("/api/upload_cv", methods=["POST"])
def upload_cv():
    if 'cv' not in request.files:
        return jsonify({"error": "No cv file part"}), 400

    file = request.files['cv']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not (file and _allowed_file(file.filename)):
        return jsonify({"error": "File type not allowed. Please upload PDF or image."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(filepath)
    ext = os.path.splitext(filename)[1].lower()

    try:
        salary_model = model_registry.get_salary_model()
        result = parse_cv_and_predict(filepath, ext, model=salary_model)
        _safe_remove(filepath)

        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        _safe_remove(filepath)
        from flask import current_app
        current_app.logger.error(f"CV parsing error: {e}")
        return jsonify({"error": "An error occurred during CV parsing. Please try another file."}), 500


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
