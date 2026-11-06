#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"
exec "$DIR/.venv/bin/python3" "$DIR/ui/server.py" 8050 0.0.0.0
