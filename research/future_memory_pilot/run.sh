#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if command -v pwsh >/dev/null 2>&1; then
  exec pwsh -NoProfile -ExecutionPolicy Bypass -File "$script_dir/run_local.ps1" "$@"
fi
exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$script_dir/run_local.ps1" "$@"
