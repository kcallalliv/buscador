# Dockerfile
FROM python:3.10-slim

# Evitar prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Crear directorio
WORKDIR /app

# Dependencias del sistema (si BigQuery/Vertex lo requieren, suelen bastar ssl & ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copia requirements primero para cacheo
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código
COPY app /app/app
# (si tienes wsgi.py, descomenta la línea de abajo)
# COPY wsgi.py /app/wsgi.py

# Variable PORT para Cloud Run
ENV PORT=8080

# Comando: apunta a la app exportada en app/__init__.py
CMD exec gunicorn -b :${PORT} -w 2 "app:app"

