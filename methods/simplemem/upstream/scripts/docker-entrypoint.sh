#!/bin/sh
# Ensure data dirs exist and are writable by appuser (handles bind mounts and root-owned volumes)
set -e
DATA_DIR="${DATA_DIR:-/app/MCP/data}"
LANCEDB_PATH="${LANCEDB_PATH:-$DATA_DIR/lancedb}"
mkdir -p "$DATA_DIR" "$LANCEDB_PATH"
chown -R appuser:appuser "$DATA_DIR"
exec gosu appuser "$@"
