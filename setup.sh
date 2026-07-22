#!/usr/bin/env bash
# One-time setup for Podcast Studio (macOS / Linux).
# Installs Node deps + the Python voice engine, and creates your .env.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Podcast Studio setup =="

# --- prerequisites -----------------------------------------------------------
missing=()
command -v node   >/dev/null || missing+=("node   (https://nodejs.org  |  mac: brew install node)")
command -v ffmpeg >/dev/null || missing+=("ffmpeg (mac: brew install ffmpeg  |  linux: sudo apt install ffmpeg)")

# Python 3.10-3.12 for the voice engine (3.11 preferred)
PY=""
for c in python3.11 python3.12 python3.10 python3; do
  command -v "$c" >/dev/null && { PY="$c"; break; }
done
[ -n "$PY" ] || missing+=("python3 (mac: brew install python@3.11  |  linux: sudo apt install python3.11-venv)")

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo "Please install these first, then re-run ./setup.sh:"
  printf '  - %s\n' "${missing[@]}"
  exit 1
fi
echo "Using $($PY --version) for the voice engine."

# --- Node app ----------------------------------------------------------------
echo "Installing Node dependencies..."
npm install

[ -f .env ] || { cp .env.example .env; echo "Created .env (from .env.example)."; }

# --- Python voice engine (Chatterbox) ---------------------------------------
echo "Setting up the local voice engine (first time is a big download: PyTorch)..."
[ -d tts_local/.venv ] || "$PY" -m venv tts_local/.venv
tts_local/.venv/bin/pip install --upgrade pip -q
tts_local/.venv/bin/pip install -r tts_local/requirements.txt

echo ""
echo "== Setup complete =="
echo ""
echo "1. Get a FREE script-writing key at https://console.groq.com"
echo "   and paste it into .env as OPENAI_API_KEY=..."
echo "2. Run:  ./start.sh"
echo "3. Open: http://localhost:3000"
echo ""
echo "Note: the first podcast you generate downloads the voice model (~2 GB) and is slow;"
echo "after that it's much faster."
