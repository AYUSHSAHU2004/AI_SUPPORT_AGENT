#!/bin/bash
# docker-entrypoint.sh — Starts the FastAPI server.
# SQLite DB and tables are created automatically on first startup by init_db().

set -e

echo "🚀 Starting Amazon Support Agent API on port ${FASTAPI_PORT:-8088}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${FASTAPI_PORT:-8088}" \
    --log-level "${LOG_LEVEL:-info}" \
    --workers 1