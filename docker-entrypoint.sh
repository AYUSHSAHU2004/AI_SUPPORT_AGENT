#!/bin/bash
# docker-entrypoint.sh — Starts the FastAPI server.
# SQLite DB and tables are created automatically on first startup by init_db().

set -e

# Uvicorn requires lowercase log levels; normalise whatever is in .env
LOG_LEVEL_LOWER=$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')

echo "🚀 Starting Amazon Support Agent API on port ${FASTAPI_PORT:-8088}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${FASTAPI_PORT:-8088}" \
    --log-level "$LOG_LEVEL_LOWER" \
    --workers 1