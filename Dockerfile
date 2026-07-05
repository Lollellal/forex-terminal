# Cloud-Deployment von backend/api (Trading OS API) — Phase 2 "PC muss nicht
# mehr an sein". Baut NUR die FastAPI-Schicht + ihre Domain-/Infrastructure-
# Abhängigkeiten, nicht das Legacy-Terminal (terminal_server.py/src/) und
# nicht mobile/ — die API braucht beides zur Laufzeit nicht (siehe
# backend/api/dependencies.py: jede Route geht ausschließlich über Supabase
# Postgres/Storage, keine lokalen Dateien).
#
# python:3.12-slim statt der lokal installierten 3.14, weil es ein stabiler,
# auf jedem Hosting-Anbieter verfügbarer Docker-Hub-Tag ist — der Code nutzt
# keine 3.13+-spezifische Syntax.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY backend/ backend/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# ${PORT:-8000}, weil Render/Railway den Port über die PORT-Env-Var vorgeben
# (Fly.io konfiguriert den Port stattdessen über fly.toml, der Fallback auf
# 8000 schadet dort nicht).
CMD ["sh", "-c", "uvicorn backend.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
