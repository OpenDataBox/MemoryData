# SimpleMem MCP Server
FROM python:3.11-slim

WORKDIR /app

# Install gosu so entrypoint can chown then drop to appuser
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy repository (MCP server lives at /app/MCP)
COPY . .
RUN chmod +x /app/scripts/docker-entrypoint.sh

WORKDIR /app/MCP

# Install MCP server dependencies only (self-contained; no root requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user; entrypoint runs as root, chowns data dir, then execs as appuser
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

# Data dir for LanceDB and users.db (override via env; mount volume in compose)
ENV DATA_DIR=/app/MCP/data

EXPOSE 8000

# Entrypoint ensures data dirs exist and are writable by appuser (bind mount or volume)
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
