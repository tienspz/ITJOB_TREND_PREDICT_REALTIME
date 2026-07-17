# Stage 1: Build dependencies
# Python 3.12 + the version ranges in requirements.txt resolve to the same
# numpy 2.x / scikit-learn 1.8.x line the .joblib models were trained with.
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/
COPY frontend/ ./frontend/
# Needed by the docker-compose ml_worker service (weekly retrain + snapshots)
COPY auto_worker.py retrain_all.py ./
EXPOSE 5000
CMD ["python", "-m", "backend.server"]
