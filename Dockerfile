# Enzyme kinetics HTTP service — runtime locked to Python 3.12
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENZYME_STORE_PATH=/data/enzymes.json

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY app ./app
COPY wsgi.py ./wsgi.py

# Persistent profile store, owned by the unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

# 2 workers + threads exercise the cross-process locking while staying small.
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--workers=2", "--threads=4", \
     "--worker-tmp-dir=/dev/shm", "--access-logfile=-", "--error-logfile=-", "wsgi:app"]
