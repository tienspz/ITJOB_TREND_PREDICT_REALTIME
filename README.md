# IT Job Market Intelligence — Phân tích thị trường IT

Offline ML pipeline + Real-time job analytics dashboard.  
Predicts salary, demand, and skill trends. No paid API keys required — primary data from **freehire.dev** (free, 2.9M IT jobs) + **RemoteOK** fallback + **Kaggle LinkedIn dataset** (1.3M postings).

## Features

| Feature | Description |
|---------|-------------|
| **Salary Prediction** | XGBoost / tuned RandomForest — predicts annual salary from title, seniority, location, skills (R²=0.531, MAE=$23,088) |
| **Demand Scoring** | 0–100 hiring intensity score for (job_title, location) pairs (R²=0.673) |
| **Market Segmentation** | KMeans clustering into 5 job segments (e.g., "Senior High-Paying", "Junior Growing") |
| **Realtime Trends** | Live job fetching from freehire.dev + RemoteOK with cache, IT filter, trend visualization |
| **Skill Comparison** | Top skills bucket comparison between Kaggle baseline and live API data |
| **Trend Forecast** | 2-month forecast using polynomial regression on 30-month Kaggle-derived historical data |

## Quick Start

### Local (no Docker)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Preprocess Kaggle data (skip if models exist)
python Dataset/preprocess_kaggle.py    # → data/it_jobs_processed.csv

# 3. Train models (skip if .joblib files exist)
python retrain_all.py                   # → models/*.joblib

# 4. Run server
python -m backend.server                # → http://localhost:5000
```

### Docker

```bash
docker compose up --build
# → api_server on port 5000
# → ml_worker runs weekly retrain (auto_worker.py)
```

### Realtime trends (in-app)

Click **"Lấy xu hướng thị trường Real-time"** on the dashboard.  
Provider chain: `freehire.dev → RemoteOK → Google Jobs (needs key) → cache → snapshot`

## Project Structure

```
├── backend/
│   ├── server.py                  # Flask app (routes, templates)
│   ├── services/
│   │   ├── trend_service.py        # Orchestrates realtime pipeline
│   │   ├── freehire_service.py     # freehire.dev (primary, no key)
│   │   ├── remoteok_service.py     # RemoteOK (fallback)
│   │   ├── google_jobs_service.py  # Google Jobs (requires SERPAPI_KEY)
│   │   ├── realtime_cache.py       # 1-hour cache for API results
│   │   └── cv_parser.py            # CV parsing → job title
│   ├── routers/
│   │   ├── core.py                 # Health, meta, charts
│   │   ├── predict.py              # /predict, /cluster
│   │   └── realtime.py             # /realtime/*
│   ├── model_registry.py           # Loaded .joblib artifacts
│   └── config.py                   # Paths, constants
├── Dataset/
│   ├── linkedin_job_postings.csv  # Kaggle raw (1.3M rows)
│   ├── job_skills.csv
│   ├── job_summary.csv
│   └── preprocess_kaggle.py      # ETL → it_jobs_processed.csv
├── data/
│   ├── it_jobs_processed.csv     # 128K IT jobs, 19 features
│   ├── backup_trends.csv         # 30-month Kaggle-derived trends
│   └── realtime_cache.json       # Cached live data
├── models/
│   ├── best_salary_model.joblib  # XGBoost / tuned RF (R²=0.531)
│   ├── demand_model.joblib       # RandomForest (R²=0.673)
│   ├── cluster_model.pkl         # KMeans (K=5)
│   ├── scaler.joblib             # StandardScaler
│   ├── seniority_encoder.joblib  # LabelEncoder
│   └── skill_binarizer.joblib    # MultiLabelBinarizer
├── frontend/
│   ├── templates/index.html       # Jinja2 single-page dashboard
│   ├── static/js/app.js           # Chart.js + fetch calls
│   └── static/css/style.css
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── training_minimal.ipynb
│   ├── training.ipynb
│   └── analytics.ipynb
├── auto_worker.py                 # MLOps — weekly retrain
├── retrain_all.py                 # Full model pipeline
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── .env.example
└── LICENSE
```

## Environment

- `SERPAPI_KEY` — optional, for Google Jobs realtime provider
- All other data sources require **no API key** (Kaggle CSVs ship with repo; freehire.dev / RemoteOK are free)

## Tech Stack

- **Backend:** Python 3.10, Flask 3.x
- **ML:** scikit-learn, pandas, numpy
- **Frontend:** Bootstrap 5, Chart.js
- **Realtime:** freehire.dev (free, no key), RemoteOK (CORS-friendly)
- **Infra:** Docker, docker-compose
