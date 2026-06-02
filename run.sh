#!/usr/bin/env bash
# run.sh — Local development launcher.
# Starts FastAPI backend (port 8000) + Flask frontend (port 5001).
# NOTE: macOS AirPlay Receiver occupies port 5000 — use 5001 instead.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin"

# Guard: make sure venv exists
if [ ! -f "$VENV/python" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Create it with: python3.11 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "  🧬  MedAI Disease Prediction System"
echo "  ────────────────────────────────────"
echo ""

# ── 1. Kill anything already on our ports ─────────────────────────────────────
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null || true
lsof -ti tcp:5001 | xargs kill -9 2>/dev/null || true

# ── 2. Start FastAPI backend ───────────────────────────────────────────────────
echo "  ▶  Starting FastAPI backend on http://127.0.0.1:8000 ..."
"$VENV/uvicorn" app.api.prediction_api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --log-level warning &
API_PID=$!

# Wait until FastAPI is actually accepting requests
echo "  ⏳  Waiting for FastAPI ..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:8000/health" > /dev/null 2>&1; then
        echo "  ✅  FastAPI ready."
        break
    fi
    sleep 1
done

# ── 3. Start Flask frontend ────────────────────────────────────────────────────
echo "  ▶  Starting Flask frontend on http://127.0.0.1:5001 ..."
FLASK_PORT=5001 API_PORT=8000 "$VENV/python" web/app.py &
FLASK_PID=$!

echo ""
echo "  ✅  Both servers running."
echo "  🌐  Open your browser at:  http://127.0.0.1:5001"
echo "  📖  API docs at:           http://127.0.0.1:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# ── 4. Clean up on Ctrl+C ─────────────────────────────────────────────────────
trap "echo ''; echo '  Stopping servers...'; kill $API_PID $FLASK_PID 2>/dev/null; exit 0" INT TERM
wait
