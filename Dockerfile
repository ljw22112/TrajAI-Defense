# syntax=docker/dockerfile:1
# ---- 1. build the SPA -------------------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- 2. API + static SPA ------------------------------------------------------------
FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY server/requirements.txt server/requirements.txt
RUN pip install -r server/requirements.txt
COPY trajai/ trajai/
COPY server/ server/
COPY results/metrics.json results/metrics.json
COPY export/ export/
COPY --from=web /web/dist web/dist
RUN useradd --create-home app && chown -R app /srv
USER app
WORKDIR /srv/server
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health').status!=200)"
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]
