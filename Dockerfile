FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy application source code
COPY . .

# Set default port
ENV PORT=8000
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI server via uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
