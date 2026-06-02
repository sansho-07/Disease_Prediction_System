#!/bin/bash
# start.sh — Container entrypoint for HF Spaces and Docker.
# Starts FastAPI (internal port 8000) then Flask (public port 7860).

set -e

API_PORT=${API_PORT:-8000}
FLASK_PORT=${FLASK_PORT:-7860}

echo "===== Application Startup at $(date -u '+%Y-%m-%d %H:%M:%S') ====="

# ── 1. Start FastAPI backend ───────────────────────────────────────────────────
echo "Starting FastAPI backend on port $API_PORT ..."
uvicorn app.api.prediction_api:app \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --workers 1 &
FASTAPI_PID=$!

# ── 2. Wait until FastAPI is healthy (up to 60s) ──────────────────────────────
echo "Waiting for FastAPI to be ready ..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "FastAPI is up (${i}s)."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: FastAPI did not start within 60 seconds. Aborting."
        kill "$FASTAPI_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ── 3. Start Flask frontend ────────────────────────────────────────────────────
echo "Starting Flask frontend on port $FLASK_PORT ..."
python web/app.py &
FLASK_PID=$!

echo "===== Both services running. Flask → :$FLASK_PORT  FastAPI → :$API_PORT ====="

# ── 4. Monitor — if either process exits, shut down cleanly ───────────────────
wait -n "$FASTAPI_PID" "$FLASK_PID"
echo "A service exited unexpectedly. Shutting down ..."
kill "$FASTAPI_PID" "$FLASK_PID" 2>/dev/null
exit 1
