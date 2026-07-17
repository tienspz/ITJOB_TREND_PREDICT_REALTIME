# IT Job Market Intelligence — Phân tích thị trường IT

Offline ML pipeline + Real-time job analytics dashboard.
Predicts salary, demand, and skill trends. No paid API keys required — primary data from **freehire.dev** (free, 2.9M IT jobs) + **RemoteOK** fallback + **Kaggle LinkedIn dataset** (1.3M postings).

## Features

| Feature | Description |
|---------|-------------|
| **Salary Prediction** | XGBoost vs tuned RandomForest, selected by CV on the train split; features include skills, seniority, location, **years of experience** (regex-extracted from postings) |
| **Demand Scoring** | 0–100 hiring intensity score for (domain, state, seniority, job_type) — 5-fold CV + isolated test |
| **Market Segmentation** | KMeans + PCA clustering into 5 job segments |
| **Realtime Trends** | Live job fetching (freehire.dev → RemoteOK → Google Jobs → cache), plus a self-accumulating time series (`data/realtime_history.csv`) |
| **Skill Comparison** | Top skills bucket comparison between Kaggle baseline and live API data |
| **Trend Forecast** | 2-month forecast — Polynomial vs Holt-Winters selected by rolling-origin backtest (MAPE), with 95% confidence interval |
| **Model Explainability** | Permutation importance on the isolated test set, shown on the dashboard |
| **CV Analysis** | PDF parsing → skills + seniority + years of experience → market-value prediction |

## ML Evaluation Protocol

All supervised models follow the standard protocol:

```
80% TRAIN  — algorithm comparison + hyperparameter tuning via k-fold CV
20% TEST   — ISOLATED: touched exactly once, to report final R²/MAE
```

Metrics live in `models/*_meta.joblib` and are served via `/api/meta`.

## Data Notes (đọc trước khi viết báo cáo)

- The Kaggle LinkedIn dump is a **single-week snapshot** (`first_seen` 2024-01-12..17) —
  it cannot provide month-over-month history. Long-horizon series in
  `data/backup_trends.csv` is a **BLS-growth projection** anchored on that snapshot,
  and the API/UI label it as such.
- Genuine longitudinal data is **self-accumulated**: every successful realtime fetch
  appends an hourly snapshot to `data/realtime_history.csv` (plus a daily snapshot from
  `auto_worker.py`). The longer the system runs, the more real trend data it owns.
- Salary present in ~25% of postings (range midpoints preferred, hourly rates annualised
  ×2080h); years-of-experience extractable from ~45% (median-imputed during training).
- Market scope: **US** (LinkedIn US postings).

## Quick Start

### Local (no Docker)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Preprocess Kaggle data (skip if data/it_jobs_processed.csv exists)
python Dataset/preprocess_kaggle.py    # → data/it_jobs_processed.csv

# 3. Train models (skip if .joblib files exist)
python retrain_all.py                   # → models/*.joblib + feature_importance.png

# 4. Run server
python -m backend.server                # → http://localhost:5000

# 5. Tests
python -m pytest tests/ -v
```

### Docker

```bash
docker compose up --build
# → api_server on port 5000
# → ml_worker runs weekly retrain + daily realtime snapshot (auto_worker.py)
```

### Deploy (Netlify + Render)

- Frontend: Netlify, publish dir `frontend`; `/api/*` and `/reports/*` proxied per `netlify.toml`
- Backend: Render (Docker) via `render.yaml`, health check `/api/health`
- After the first Render deploy, put the real backend URL into `netlify.toml`

## Project Structure

```
├── backend/
│   ├── server.py                  # Flask app + /api/realtime-trends forecast
│   ├── services/
│   │   ├── trend_service.py        # Orchestrates realtime pipeline
│   │   ├── history_service.py      # Self-accumulating realtime time series
│   │   ├── freehire_service.py     # freehire.dev (primary, no key)
│   │   ├── remoteok_service.py     # RemoteOK (fallback)
│   │   ├── google_jobs_service.py  # Google Jobs (requires SERPAPI_KEY)
│   │   ├── realtime_cache.py       # 1-hour cache for API results
│   │   └── cv_parser.py            # CV parsing → skills/seniority/YoE → salary
│   ├── routers/
│   │   ├── core.py                 # Health, meta (+dataset stats), charts
│   │   ├── predict.py              # predict/cluster/compare_skills/trends/report
│   │   └── realtime.py             # /api/realtime/*
│   ├── model_registry.py           # Loaded .joblib artifacts
│   └── config.py                   # Paths, constants
├── Dataset/
│   ├── linkedin_job_postings.csv  # Kaggle raw (1.3M rows, NOT committed)
│   ├── job_skills.csv
│   ├── job_summary.csv
│   └── preprocess_kaggle.py      # ETL → it_jobs_processed.csv
├── data/
│   ├── it_jobs_processed.csv     # 128K IT jobs (salary ranges + YoE)
│   ├── backup_trends.csv         # 30-month BLS-projected baseline
│   ├── realtime_history.csv      # Self-accumulated realtime series
│   └── realtime_cache.json       # Cached live data
├── models/                        # 3 models + meta (joblib, compress=3)
├── frontend/
│   ├── index.html                 # Static single-page dashboard (8 tabs)
│   ├── static/js/app.js           # Chart.js + fetch calls
│   └── static/css/style.css
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Raw Kaggle → processed CSV
│   ├── 02_model_training.ipynb       # 80/20 + CV + isolated test + importance
│   └── 03_realtime_forecast.ipynb    # Realtime pipeline + backtest forecast
├── tests/test_api.py              # pytest smoke suite (Flask test client)
├── auto_worker.py                 # MLOps — weekly retrain + daily snapshot
├── retrain_all.py                 # Full model pipeline
├── requirements.txt
├── Dockerfile / docker-compose.yml / render.yaml / netlify.toml
├── .env.example
└── LICENSE
```

## Environment

- `SERPAPI_KEY` — optional; register free at https://serpapi.com (100 searches/month)
  to enable the Google Jobs realtime provider
- `CORS_ORIGINS` — comma-separated allowed origins in production
- All other data sources require **no API key**

## Tech Stack

- **Backend:** Python 3.12+, Flask 3.x, waitress
- **ML:** scikit-learn 1.8, XGBoost, statsmodels (Holt-Winters), pandas, numpy
- **Frontend:** Bootstrap 5, Chart.js (static HTML — no server-side templating)
- **Infra:** Docker, docker-compose, Render (API), Netlify (frontend)
