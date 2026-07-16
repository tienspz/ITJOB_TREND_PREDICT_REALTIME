# IT Job Market Intelligence — Architecture

## System Overview
Two-pipeline system: **Historical Analysis** (Kaggle 1.3M LinkedIn dataset) for ML training, and **Real-time Pipeline** (free APIs) for live market intelligence.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Flask Backend (port 5000)                     │
│  ┌────────────┐  ┌──────────┐  ┌────────────┐  ┌────────────────┐  │
│  │ core_router │  │predict_   │  │realtime_    │  │ Legacy Route   │  │
│  │ /api/health │  │router     │  │router       │  │ /api/realtime- │  │
│  │ /api/meta   │  │ /predict  │  │ /realtime/* │  │ trends         │  │
│  │ /api/charts │  │ /cluster  │  │ /trending   │  │ (Kaggle+Realt) │  │
│  └────────────┘  └──────────┘  └────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────────────────────┐
          ▼                                               ▼
┌───────────────────────┐                  ┌──────────────────────────┐
│   ML Models (joblib)  │                  │  Realtime Provider Chain │
│  ┌─────────────────┐  │                  │                         │
│  │ Salary Predictor │  │                  │  1. freehire.dev ───────│
│  │ (RandomForest)   │  │                  │     (2.9M IT jobs, free)│
│  │ R²=0.531         │  │                  │  2. RemoteOK ───────────│
│  ├─────────────────┤  │                  │     (global remote, free)│
│  │ Demand Scorer    │  │                  │  3. Google Jobs ────────│
│  │ (RandomForest)   │  │                  │     (needs SERPAPI_KEY) │
│  │ 0-100 score      │  │                  │  4. Cache ──────────────│
│  ├─────────────────┤  │                  │     (<1h live snapshot) │
│  │ Cluster (KMeans) │  │                  │  5. Snapshot ───────────│
│  │ K=5 segments     │  │                  │     (built-in fallback) │
│  └─────────────────┘  │                  └──────────────────────────┘
└───────────────────────┘
```

## Pipeline 1: Historical (Kaggle 1.3M)

| Step | Script | Output |
|------|--------|--------|
| Raw CSVs | `Dataset/linkedin_job_postings.csv`, `job_skills.csv`, `job_summary.csv` | ~1.3M raw rows |
| ETL | `Dataset/preprocess_kaggle.py` | 128,307 IT jobs |
| Processed | `data/it_jobs_processed.csv` | 19 features incl. seniority, skills, domain |
| Retrain | `retrain_all.py` | 6 `.joblib` files in `models/` |

**Models trained:**
- **Salary Predictor** — XGBoost / tuned RandomForest (R²=0.531, MAE=$23,088)
- **Demand Scorer** — RandomForestRegressor (0–100 hiring intensity)
- **Job Cluster** — KMeans (K=5, PCA-reduced) for market segmentation

## Pipeline 2: Real-time

| Provider | API Key | Coverage | State Extraction |
|----------|---------|----------|-----------------|
| **freehire.dev** | None | 2.9M US IT jobs | City→state mapping |
| **RemoteOK** | None | Global remote jobs | Raw location string |
| Cache | — | Previous live call | — |
| Snapshot | — | 15 built-in jobs (jitter) | US state codes |

**IT Filtering:** Each provider has a title-based `_is_it_job()` filter:
1. Exclude non-IT patterns ("security officer", "data entry", "finance")
2. Include IT patterns ("engineer", "developer", "scientist")
3. Fallback: ≥2 tech skills in tags → accept

**Analytics computed:**
- Top skills (24h / 7d)
- Hiring velocity (postings/day over 7 days)
- Top locations, companies, roles
- Historical vs Realtime skill comparison (bucket-level)
- Polynomial trend forecast (degree 2, Kaggle baseline + API adjustment)

## Deployment

```
docker-compose.yml
├── api_server (port 5000)  →  python -m backend.server
└── ml_worker               →  python auto_worker.py (weekly retrain)
```

**Dockerfile:** `python:3.10-slim` → installs deps → copies source.

## Frontend

Bootstrap 5 dashboard served via Flask Jinja2 templates:
- `frontend/templates/index.html` — Single-page app
- `frontend/static/js/app.js` — Chart.js + API calls
- `frontend/static/css/style.css` — Custom styles

## Data Flow

```
User clicks "Lấy xu hướng thị trường Real-time"
  → trend_service.build_realtime_report()
  → freehire_service.fetch_jobs()  (or fallback chain)
  → realtime_cache.save_cache()
  → Analytics: top_skills, comparison, velocity...
  → JSON response to frontend

User clicks "Dự báo xu hướng" (Trend chart)
  → /api/realtime-trends
  → Read backup_trends.csv (Kaggle-derived, 30 months)
  → Get current API count, scale to match magnitude
  → Polynomial regression (degree 2) → forecast 2 months
```
