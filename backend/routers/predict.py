"""
Prediction router — salary / demand / cluster / CV upload / simple predict.

All input validation is shared via the same schema helpers so behaviour
matches the original monolithic server.py exactly (frontend stays green).
"""

import json
import os
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, send_from_directory
from sklearn.linear_model import LinearRegression
from werkzeug.utils import secure_filename

from backend import config
from backend import model_registry
from backend.services.cv_parser import parse_cv_and_predict, parse_cv_text_and_predict

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
    "years_experience": 3,
    "salary_annual": 100000.0,
    "seniority_level": "Mid", "job_type": "Remote",
    "state": "CA", "it_domain": "Software Engineering",
}

# skill_* features are keyword COUNTS per category (0..~10 in the training
# data), not binary flags — keep serving semantics identical to training.
FEATURE_VALIDATION = {
    "num_skills": {"type": "int", "min": 0, "max": 30},
    "skill_diversity": {"type": "int", "min": 0, "max": 10},
    "skill_programming": {"type": "int", "min": 0, "max": 10},
    "skill_cloud": {"type": "int", "min": 0, "max": 10},
    "skill_ai_ml": {"type": "int", "min": 0, "max": 10},
    "skill_database": {"type": "int", "min": 0, "max": 10},
    "skill_devops": {"type": "int", "min": 0, "max": 10},
    "skill_framework": {"type": "int", "min": 0, "max": 10},
    "skill_data_engineering": {"type": "int", "min": 0, "max": 10},
    "skill_security": {"type": "int", "min": 0, "max": 10},
    "skill_soft_skills": {"type": "int", "min": 0, "max": 10},
    "years_experience": {"type": "int", "min": 0, "max": 40},
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
        "years_experience": exp,
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
# Skills comparison
# ---------------------------------------------------------------------------
@bp.route("/api/compare_skills", methods=["POST"])
def compare_skills():
    salary_model = model_registry.get_salary_model()
    if salary_model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data or "skills" not in data or not isinstance(data["skills"], list):
        return jsonify({"error": "Provide a 'skills' array"}), 400

    skills_input = [str(s).strip() for s in data["skills"] if s]

    known_skill_groups = {
        "prog": ["Python", "Java", "JavaScript", "C++", "SQL", "TypeScript", "Go", "Rust"],
        "cloud": ["AWS", "Azure", "Docker", "Kubernetes", "CI/CD", "GCP", "Terraform"],
        "ai": ["Machine Learning", "PyTorch", "TensorFlow", "AI", "Data Science", "NLP", "LLM"],
        "db": ["PostgreSQL", "MongoDB", "MySQL", "Oracle", "Redis", "Cassandra"],
        "devops": ["Terraform", "Ansible", "CI/CD", "GitHub Actions", "Jenkins", "Helm"],
        "framework": ["React", "Node.js", "Spring", "Vue.js", "Angular", "Django", "Flask"],
        "dataeng": ["Spark", "Airflow", "DBT", "Hadoop", "Kafka", "Snowflake"],
        "sec": ["Cybersecurity", "Penetration Testing", "Cloud Security", "IAM", "Compliance", "Zero Trust"],
    }

    category_map = {}
    for cat, skills_list in known_skill_groups.items():
        for skill in skills_list:
            category_map[skill.lower()] = cat

    num_skills = len(skills_input)
    categories_found = set()
    skill_counts = {cat: 0 for cat in known_skill_groups}

    for skill in skills_input:
        cat = category_map.get(skill.lower())
        if cat:
            skill_counts[cat] += 1
            categories_found.add(cat)

    features = dict(DEFAULT_INPUTS)
    features.update({
        "num_skills": min(num_skills, 30),
        "years_experience": _parse_int(data.get("years_experience"),
                                       DEFAULT_INPUTS["years_experience"], 0, 40),
        "skill_diversity": len(categories_found),
        "skill_programming": skill_counts["prog"],
        "skill_cloud": skill_counts["cloud"],
        "skill_ai_ml": skill_counts["ai"],
        "skill_database": skill_counts["db"],
        "skill_devops": skill_counts["devops"],
        "skill_framework": skill_counts["framework"],
        "skill_data_engineering": skill_counts["dataeng"],
        "skill_security": skill_counts["sec"],
    })
    features['domain_seniority'] = features['it_domain'] + '_' + features['seniority_level']
    features['state_seniority'] = features['state'] + '_' + features['seniority_level']
    df = pd.DataFrame([features])

    try:
        pred_salary = float(salary_model.predict(df)[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    # Market percentile: rank the predicted salary against the full dataset.
    percentile = 50
    related_roles = []

    if os.path.exists(config.PROCESSED_DATA_FILE):
        try:
            df_all = pd.read_csv(config.PROCESSED_DATA_FILE, usecols=["salary_annual", "it_domain"])
            salaries = df_all["salary_annual"].dropna().to_numpy()
            if len(salaries) > 0:
                percentile = int(np.searchsorted(np.sort(salaries), pred_salary) / len(salaries) * 100)
            related_roles = df_all["it_domain"].value_counts().index[:5].tolist()
        except Exception:
            pass
    if not related_roles:
        related_roles = ["Software Engineering", "Data Science", "DevOps/SRE"]

    gap_suggestions = {
        "prog": "Learn Python or Java to boost salary",
        "cloud": "Add AWS or Azure for cloud roles",
        "ai": "Learn ML/AI for higher demand roles",
        "db": "Database skills like PostgreSQL increase value",
        "devops": "Docker/Kubernetes knowledge is highly valued",
        "framework": "Learning React or Node.js opens frontend roles",
        "dataeng": "Spark or Airflow for data engineering",
        "sec": "Cybersecurity skills are in high demand",
    }
    missing_cats = set(known_skill_groups.keys()) - categories_found
    skill_gaps = [gap_suggestions[cat] for cat in gap_suggestions if cat in missing_cats]

    return jsonify({
        "skills_analyzed": skills_input,
        "market_value": round(pred_salary, 2),
        "currency": "USD",
        "percentile": percentile,
        "in_demand": percentile > 60,
        "related_roles": related_roles[:5],
        "skill_gaps": skill_gaps[:3],
        "skill_coverage": {
            "total": num_skills,
            "categories": len(categories_found),
            "details": {cat: bool(v) for cat, v in skill_counts.items()}
        }
    })


# ---------------------------------------------------------------------------
# Historical trends — segment-anchored projection
#
# The Kaggle dataset is a single-week snapshot (first_seen 2024-01-12..17),
# so per-month history per segment does not exist. Instead of synthesising
# fake months, this endpoint anchors REAL segment statistics (median salary,
# posting share of the chosen domain+state) onto the BLS-projected monthly
# baseline (backup_trends.csv) and says so in the response.
# ---------------------------------------------------------------------------
_SEGMENT_CACHE = {"df": None}


def _segment_data():
    if _SEGMENT_CACHE["df"] is None and os.path.exists(config.PROCESSED_DATA_FILE):
        _SEGMENT_CACHE["df"] = pd.read_csv(
            config.PROCESSED_DATA_FILE,
            usecols=["it_domain", "state", "salary_annual"],
        )
    return _SEGMENT_CACHE["df"]


@bp.route("/api/historical_trends", methods=["POST"])
def historical_trends():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    domain = str(data.get("domain", "Software Engineering"))
    state = str(data.get("state", "CA"))
    months = min(_parse_int(data.get("months"), 12, min_value=1), 36)

    df = _segment_data()
    if df is None:
        return jsonify({"error": "Historical data not found"}), 503

    try:
        # REAL segment statistics from the Kaggle snapshot
        domain_mask = df["it_domain"].astype(str).str.contains(domain, case=False, na=False, regex=False)
        state_mask = df["state"].astype(str).str.contains(state, case=False, na=False, regex=False)
        segment = df[domain_mask & state_mask]
        segment_scope = "domain+state"
        if len(segment) < 10:
            segment = df[domain_mask]
            segment_scope = "domain"
        if len(segment) < 10:
            return jsonify({"error": "Insufficient data for trend analysis"}), 400

        seg_salaries = segment["salary_annual"].dropna()
        seg_median = float(seg_salaries.median()) if len(seg_salaries) else None
        seg_share = len(segment) / max(len(df), 1)
        if seg_median is None:
            return jsonify({"error": "No salary data for this segment"}), 400

        # BLS-projected monthly baseline
        baseline = pd.read_csv(config.BACKUP_TRENDS_FILE).tail(months)
        base_salary = baseline["avg_salary"].astype(float)
        base_jobs = baseline["job_count"].astype(float)

        # Anchor the segment's real median at the latest month, apply the
        # baseline growth shape; demand = baseline volume x real segment share
        salary_trend = (seg_median * base_salary / base_salary.iloc[-1]).round(2).tolist()
        demand_trend = (base_jobs * seg_share).round().astype(int).tolist()

        X = np.arange(len(salary_trend)).reshape(-1, 1)
        lr = LinearRegression().fit(X, np.array(salary_trend))
        growth_rate = float(lr.coef_[0])
        growth_pct = (growth_rate / max(np.mean(salary_trend), 1)) * 100

        return jsonify({
            "labels": baseline["month"].tolist(),
            "salary_trend": salary_trend,
            "demand_trend": demand_trend,
            "domain": domain,
            "state": state,
            "growth_rate": round(growth_rate, 2),
            "growth_percentage": round(growth_pct, 2),
            "data_points": len(salary_trend),
            "data_source": "segment_anchored_projection",
            "note": ("Chuỗi thời gian là CHIẾU từ baseline BLS (~3.5%/năm), neo trên số liệu "
                     "THẬT của phân khúc từ snapshot Kaggle 01/2024."),
            "segment": {
                "scope": segment_scope,
                "job_count": int(len(segment)),
                "share_pct": round(seg_share * 100, 2),
                "median_salary": round(seg_median, 2),
                "p25_salary": round(float(seg_salaries.quantile(0.25)), 2),
                "p75_salary": round(float(seg_salaries.quantile(0.75)), 2),
            },
        })

    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Trend analysis error: {e}")
        return jsonify({"error": f"Trend analysis failed: {e}"}), 400


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
@bp.route("/api/generate_report", methods=["POST"])
def generate_report():
    data = request.json or {}
    sections = data.get("sections", ["salary", "demand", "cluster"])

    salary_model = model_registry.get_salary_model()
    demand_model = model_registry.get_demand_model()
    cluster_model = model_registry.get_cluster_model()
    salary_meta = model_registry.get_salary_meta()

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "sections": {}
    }

    domains = _normalize_category_list(salary_meta.get("it_domain")) or \
        ["Software Engineering", "Data Science", "DevOps/SRE"]

    if "salary" in sections and salary_model is not None:
        salary_data = []
        for domain in domains:
            for level in ["Junior", "Mid", "Senior"]:
                f = dict(DEFAULT_INPUTS)
                f["it_domain"] = domain
                f["seniority_level"] = level
                f['domain_seniority'] = f['it_domain'] + '_' + f['seniority_level']
                f['state_seniority'] = f['state'] + '_' + f['seniority_level']
                try:
                    pred = float(salary_model.predict(pd.DataFrame([f]))[0])
                    salary_data.append({
                        "domain": domain,
                        "seniority": level,
                        "predicted_salary": round(pred, 2),
                    })
                except Exception:
                    pass
        report["sections"]["salary"] = {
            "model": salary_meta.get("model_type", "ML model"),
            "r2_score": salary_meta.get("r2_score"),
            "mae": salary_meta.get("mae"),
            "predictions": salary_data
        }

    if "demand" in sections and demand_model is not None:
        demand_meta = model_registry.get_demand_meta()
        demand_data = []
        for domain in domains:
            for state in ["CA", "TX", "NY", "WA", "Remote"]:
                features = {
                    "it_domain": domain,
                    "seniority_level": "Senior",
                    "state": state,
                    "job_type": "Remote",
                }
                try:
                    pred = float(demand_model.predict(pd.DataFrame([features]))[0])
                    pred = max(0, min(100, pred))
                    demand_data.append({
                        "domain": domain,
                        "state": state,
                        "demand_score": round(pred, 1),
                        "interpretation": "High" if pred > 75 else "Medium" if pred > 40 else "Low"
                    })
                except Exception:
                    pass
        report["sections"]["demand"] = {
            "model": "RandomForestRegressor (Demand)",
            "r2_score": demand_meta.get("r2_score") if demand_meta else None,
            "predictions": demand_data
        }

    if "cluster" in sections and cluster_model is not None:
        cluster_meta = model_registry.get_cluster_meta()
        report["sections"]["cluster"] = {
            "model": "KMeans + PCA",
            "n_clusters": cluster_meta.get("n_clusters"),
            "silhouette_score": cluster_meta.get("silhouette_score"),
            "descriptions": {str(k): v for k, v in cluster_meta.get("cluster_descriptions", {}).items()},
        }

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    report_filename = f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(config.REPORTS_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # REPORTS_DIR maps to /reports/figures/* (served by the core router)
    report_url = f"/reports/figures/{report_filename}"

    return jsonify({
        "status": "success",
        "sections": list(report["sections"].keys()),
        "report_url": report_url,
        "report": report
    })


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


@bp.route("/api/parse_cv_text", methods=["POST"])
def parse_cv_text():
    """Accept pre-extracted text (e.g. from client-side Tesseract.js OCR)
    and run the same skill-extraction + salary-prediction pipeline."""
    data = request.json
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "No text provided."}), 400

    try:
        salary_model = model_registry.get_salary_model()
        result = parse_cv_text_and_predict(data["text"], model=salary_model)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"CV text parsing error: {e}")
        return jsonify({"error": "An error occurred during CV text parsing."}), 500


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
