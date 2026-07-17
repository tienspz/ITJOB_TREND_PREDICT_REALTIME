"""
API smoke tests — run with:  python -m pytest tests/ -v

Uses the Flask test client so no server process is needed. Models must exist
in models/ (run retrain_all.py first if they don't).
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "healthy"
    assert all(body["models_loaded"].values()), "all 3 models should be loaded"


def test_meta(client):
    r = client.get("/api/meta")
    assert r.status_code == 200
    meta = r.get_json()
    sal = meta["salary_meta"]
    for key in ("r2_score", "mae", "cv_r2_mean", "cv_folds", "it_domain",
                "seniority_level", "state", "job_type", "data_rows"):
        assert key in sal, f"salary_meta missing {key}"
    assert meta["cluster_meta"].get("n_clusters")


def test_predict_salary(client):
    r = client.post("/api/predict_salary", json={
        "it_domain": "Software Engineering", "seniority_level": "Senior",
        "state": "CA", "job_type": "Remote",
        "years_experience": 8, "num_skills": 6,
        "skill_programming": 2, "skill_cloud": 1,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert 20000 <= body["predicted_salary"] <= 500000


def test_predict_salary_rejects_empty(client):
    r = client.post("/api/predict_salary", json=None,
                    content_type="application/json")
    assert r.status_code == 400


def test_predict_simple(client):
    r = client.post("/api/predict", json={"experience": 5, "skill": "Python", "location": "CA"})
    assert r.status_code == 200
    assert r.get_json()["salary"] > 0


def test_predict_demand(client):
    r = client.post("/api/predict_demand", json={
        "it_domain": "Data Science", "state": "NY",
        "seniority_level": "Mid", "job_type": "Remote",
    })
    assert r.status_code == 200
    body = r.get_json()
    assert 0 <= body["demand_score"] <= 100
    assert body["interpretation"] in ("High", "Medium", "Low")


def test_cluster(client):
    r = client.post("/api/cluster", json={
        "it_domain": "Data Science", "seniority_level": "Senior",
        "state": "CA", "job_type": "Remote", "skill_ai_ml": 2, "num_skills": 5,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body["cluster_id"], int)
    assert body["description"]


def test_compare_skills(client):
    r = client.post("/api/compare_skills", json={
        "skills": ["Python", "SQL", "AWS"], "years_experience": 4,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["market_value"] > 0
    assert 0 <= body["percentile"] <= 100
    assert len(body["related_roles"]) > 0


def test_compare_skills_requires_array(client):
    r = client.post("/api/compare_skills", json={"skills": "Python"})
    assert r.status_code == 400


def test_historical_trends(client):
    r = client.post("/api/historical_trends", json={
        "domain": "Software Engineering", "state": "CA", "months": 12,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["labels"]) == len(body["salary_trend"]) == len(body["demand_trend"])
    assert body["data_source"] == "segment_anchored_projection"
    assert body["segment"]["median_salary"] > 0


def test_generate_report(client):
    r = client.post("/api/generate_report", json={"sections": ["salary", "cluster"]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert "salary" in body["sections"] and "cluster" in body["sections"]


def test_realtime_trends_forecast(client):
    r = client.get("/api/realtime-trends")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "success"
    assert len(body["forecast"]) == 2
    assert body["metrics"]["model_used"] in ("polynomial_deg2", "holt_winters")
    for p in body["forecast"]:
        assert p["ci_low"] <= p["job_count"] <= p["ci_high"]


def test_upload_cv_rejects_missing_file(client):
    r = client.post("/api/upload_cv", data={})
    assert r.status_code == 400
