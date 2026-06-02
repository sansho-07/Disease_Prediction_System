# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Compile-time deps for scipy / scikit-learn / shap / xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 — XGBoost/OpenMP runtime dep
# curl     — used by start.sh to poll /health before starting Flask
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as UID 1000 — create matching user
RUN useradd -m -u 1000 appuser

WORKDIR /home/appuser/app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project (owned by appuser)
COPY --chown=appuser:appuser . .

# Ensure SQLite DB directory is writable by appuser
RUN mkdir -p data && chown -R appuser:appuser data

USER appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_PORT=7860 \
    API_PORT=8000 \
    HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

EXPOSE 7860

CMD ["bash", "start.sh"]
