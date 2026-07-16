FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Step 1: System dependencies (hiếm khi thay đổi → cache lâu dài)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Python dependencies (chỉ thay đổi khi requirements.txt đổi)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 3: Copy CHỈ source code (nhẹ, vài KB)
# Dataset, models, data được mount qua docker-compose volumes
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY auto_worker.py ./
COPY retrain_all.py ./

CMD ["python", "-m", "backend.server"]