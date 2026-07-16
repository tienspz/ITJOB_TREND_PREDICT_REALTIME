"""
Central configuration for the backend.

All file paths are derived from a single BASE_DIR so the project stays
relocatable. Every router/service imports these constants instead of
recomputing paths locally.
"""

import os
from dotenv import load_dotenv

# Load .env (RAPIDAPI_KEY, etc.) once for the whole backend package.
load_dotenv()

# backend/  ->  project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Core data directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# Real-time cache + scraped links queue (used by the live pipeline)
REALTIME_CACHE_FILE = os.path.join(DATA_DIR, "realtime_cache.json")
PENDING_LINKS_FILE = os.path.join(DATA_DIR, "pending_links.csv")
BACKUP_TRENDS_FILE = os.path.join(DATA_DIR, "backup_trends.csv")

# Historical datasets (used by trend_service for the Historical vs Realtime compare)
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "it_jobs_processed.csv")

# File upload constraints
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Make sure runtime dirs exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
