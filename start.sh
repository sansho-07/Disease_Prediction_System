#!/bin/bash
# start.sh — launches FastAPI backend + Flask frontend inside the container.
# Hugging Face Spaces exposes only port 7860, so Flask is the public entry point
# and proxies prediction requests to FastAPI on the internal port 8000.

set -e

API_PORT=${API_PORT:-8000}
FLASK_PORT=${FLASK_PORT:-7860}

echo "Starting FastAPI backend on port $API_PORT ..."
uvicorn app.api.prediction_api:app \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --workers 1 &
FASTAPI_PID=$!

# Wait until FastAPI is ready before starting Flask
echo "Waiting for FastAPI to be ready ..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$API_PORT/health" > /dev/null 2>&1; then
        echo "FastAPI is up."
        break
    fi
    sleep 1
done

echo "Starting Flask frontend on port $FLASK_PORT ..."
python web/app.py &
FLASK_PID=$!

# If either process dies, kill the other and exit so HF can restart the Space
wait -n $FASTAPI_PID $FLASK_PID
echo "A process exited. Shutting down ..."
kill $FASTAPI_PID $FLASK_PID 2>/dev/null
exit 1
