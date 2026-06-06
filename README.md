# Podcast Studio

> **Fastest free setup → use the Colab GPU engine.** See `colab/README_COLAB.md`.
> It runs the voices on a free cloud GPU (fast + cloning works) instead of your Mac.
> The rest of this README covers the all-local option, which is slower on a laptop.


A local AI podcast generator. Paste text (or an article URL) → an LLM writes a two-host
conversation → real voices speak each line → it stitches one downloadable MP3.

**Can run 100% free:** Groq (free) writes the script, and Chatterbox (free, local) does all
the voices *and* voice cloning right on your Mac. ElevenLabs is optional.

---

## Architecture (two small servers)

1. **Node app** (`server.js` + `public/`) — the UI, script writing, audio stitching. Port 3000.
2. **Local TTS service** (`tts_local/app.py`) — Chatterbox voice synthesis + cloning. Port 5050.

The Node app talks to the TTS service over localhost. You run both.

---

## One-time setup (Apple Silicon Mac)

### 0. Tools
```bash
# Homebrew (if you don't have it):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install node ffmpeg python@3.11
```
`ffmpeg` is required here (it stitches and normalizes the audio).

### 1. Node side
```bash
cd ~/Desktop/podcast-studio       # wherever you unzipped it
npm install
cp .env.example .env
open -e .env                      # paste your Groq key, save
```
In `.env`: set `OPENAI_API_KEY` to your Groq key (model + base URL are pre-filled for Groq).
Leave `ELEVENLABS_API_KEY` blank to stay fully free.

### 2. Local TTS side (Chatterbox)
```bash
cd ~/Desktop/podcast-studio/tts_local
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
First install is large (PyTorch + the model). The Chatterbox model (~1–2 GB) downloads the
first time you actually synthesize, not at install.

---

## Running it (two terminals)

**Terminal A — TTS service:**
```bash
cd ~/Desktop/podcast-studio/tts_local
source venv/bin/activate
python app.py
# leave running; first /tts call downloads the model and is slow, then it caches
```

**Terminal B — the app:**
```bash
cd ~/Desktop/podcast-studio
npm start
```
Open http://localhost:3000

---

## Using it

1. **Source & settings** — paste content (or fetch a URL), set hosts/tone/length. Write the script.
2. **Script** — edit lines, flip A/B speakers. Free to edit (no synthesis yet).
3. **Voices** — you'll see at least **"Chatterbox Default" (local)**. Assign it to one host.
   To make a second distinct voice, **clone** one: name it, choose a short clean audio clip of
   the target speaker (your own voice, or one you have permission for), click Clone — it appears
   in the list. Assign clones/Default to A and B. (ElevenLabs voices also show here if you set a key.)
4. **Generate episode** — synthesizes every line and stitches the MP3. Play or download.

### Getting two different voices, fully free
- Host A = "Chatterbox Default", Host B = a clone of your voice — done, two distinct voices.
- Or clone two different reference clips for two custom voices.

---

## Performance notes (Apple Silicon)
- Runs on the Mac's GPU via Metal (`mps`). First synthesis is slow (model load + download);
  later lines are faster.
- Chatterbox is English-first. For other languages, ElevenLabs (multilingual) is the better path.
- If `mps` ever errors, the service auto-falls back to CPU (slower but works).

## Going free on the script too
`.env` is pre-set for Groq (free). To use plain OpenAI, remove `OPENAI_BASE_URL` and set
`OPENAI_MODEL=gpt-4o-mini`. To run the LLM locally too, install Ollama and point
`OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`, `OPENAI_MODEL=llama3.1`.

## Troubleshooting
- **"Local TTS service unreachable"** → Terminal A isn't running, or still loading. Start `app.py`.
- **"ffmpeg not found"** → `brew install ffmpeg`.
- **Voices list only shows local** → that's expected with no ElevenLabs key. It's fine.
- **First generate takes minutes** → model download/warm-up. Subsequent runs are quick.

## Files
- `server.js` — Node backend (`/api/script`, `/api/voices`, `/api/clone`, `/api/podcast`)
- `public/index.html` — UI
- `tts_local/app.py` — Chatterbox TTS + cloning service
- `tts_local/requirements.txt` — Python deps
- `.env.example` — copy to `.env`
