#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Start both the FastAPI backend and Flask frontend
# Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/venv/bin"

echo ""
echo "  🧬  MedAI Disease Prediction System"
echo "  ────────────────────────────────────"
echo ""

# ── 1. Start FastAPI backend ──────────────────────────────────────────────────
echo "  ▶  Starting FastAPI backend on http://127.0.0.1:8000 ..."
"$VENV/uvicorn" app.api.prediction_api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload \
    --log-level warning &
API_PID=$!

# Give the API a moment to load models
sleep 3

# ── 2. Start Flask frontend ───────────────────────────────────────────────────
echo "  ▶  Starting Flask frontend on http://127.0.0.1:5000 ..."
"$VENV/python" web/app.py &
FLASK_PID=$!

echo ""
echo "  ✅  Both servers running."
echo "  🌐  Open your browser at:  http://127.0.0.1:5000"
echo "  📖  API docs at:           http://127.0.0.1:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

# ── 3. Wait and clean up on Ctrl+C ───────────────────────────────────────────
trap "echo ''; echo '  Stopping servers...'; kill $API_PID $FLASK_PID 2>/dev/null; exit 0" INT TERM
wait
