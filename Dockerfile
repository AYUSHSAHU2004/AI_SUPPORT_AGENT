# ── Stage 1: builder — install all Python dependencies ───────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools for compiled packages (chromadb, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime — lean final image ───────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source and entrypoint
COPY app/ ./app/
COPY docker-entrypoint.sh ./docker-entrypoint.sh

RUN chmod +x ./docker-entrypoint.sh

# Create data dirs (overridden by ./data volume mount at runtime)
RUN mkdir -p data/chroma data/golden_set

# Run as non-root for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8088

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')" \
    || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]