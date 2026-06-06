#!/bin/bash
# Launches the Podcast Studio web app, but only after the local TTS
# service is accepting connections on port 5050. Used by the launchd
# agent com.podcaststudio.web so the Node server never comes up before
# its TTS backend.
set -e
cd /Users/edithlovesgod/podcast-studio

echo "[start-web] $(date) waiting for TTS on 127.0.0.1:5050..."
for i in $(seq 1 120); do
  if nc -z 127.0.0.1 5050 2>/dev/null; then
    echo "[start-web] $(date) TTS is up after ${i}s — starting node"
    break
  fi
  sleep 1
done

exec /opt/homebrew/bin/node server.js
