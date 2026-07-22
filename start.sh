#!/usr/bin/env bash
# Start Podcast Studio: launches the voice engine + the web app together.
# Stop both with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "No .env found — run ./setup.sh first."; exit 1; }
[ -x tts_local/.venv/bin/python ] || { echo "Voice engine not installed — run ./setup.sh first."; exit 1; }

echo "Starting local voice engine (port 5050)..."
tts_local/.venv/bin/python tts_local/app.py &
TTS_PID=$!
trap 'kill "$TTS_PID" 2>/dev/null' EXIT

echo "Starting Podcast Studio (port 3000)..."
npm start
