"""
Local TTS service for Podcast Studio — free voice synthesis + cloning via Chatterbox.
Runs on your own machine. No API keys, nothing leaves your computer.

Endpoints:
  GET  /health           -> {ok, device, model_loaded}
  GET  /voices           -> {voices:[{id,name}]}   (built-in "default" + your clones)
  POST /clone            -> multipart {name, file} -> {id, name}   (saves a reference clip)
  POST /tts              -> json {text, voice_id}  -> audio/wav bytes
"""

import io
import os
import json
import uuid
import subprocess
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse
import soundfile as sf

BASE = Path(__file__).parent
VOICES_DIR = BASE / "voices"
VOICES_DIR.mkdir(exist_ok=True)
REGISTRY = VOICES_DIR / "registry.json"

app = FastAPI()

# --- registry helpers ---------------------------------------------------------

def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}

def save_registry(reg):
    REGISTRY.write_text(json.dumps(reg, indent=2))

# --- model (loaded lazily so the server starts fast) --------------------------

_model = None
_device = None
_model_lock = __import__("threading").Lock()

# Default generation params. NOTE (chatterbox-tts 0.1.7): none of generate()'s
# knobs meaningfully cut steady-state time — there's no inference/diffusion-steps
# param, and cfg_weight=0 (the usual "skip the CFG double-pass" speedup) crashes
# this build with a batch-shape mismatch. So these are exposed for quality/voice
# control, and cfg_weight is clamped to a positive value to avoid that crash.
DEFAULTS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}

def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model():
    global _model, _device
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # another thread loaded it while we waited
            return _model
        return _load_model()

def _enable_fast_attention():
    """Force output_attentions=False in the T3 decode loop so the llama backbone
    uses the fast SDPA kernel instead of eager attention. Requesting attention
    weights forces eager attention, which measured ~4x slower per decode step on
    mps (~476ms -> ~109ms/step). The weights are only consumed by the multilingual
    alignment_stream_analyzer, so we only disable them when there's no analyzer to
    feed — keeping this lossless. output_hidden_states stays on (logits need it)."""
    try:
        from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    except Exception as e:
        print(f"[tts] fast-attention patch skipped ({e})")
        return
    if getattr(T3HuggingfaceBackend, "_fast_attn_patched", False):
        return
    _orig_forward = T3HuggingfaceBackend.forward
    def _forward(self, *args, output_attentions=True, **kwargs):
        if getattr(self, "alignment_stream_analyzer", None) is None:
            output_attentions = False  # nobody reads them -> let SDPA run
        return _orig_forward(self, *args, output_attentions=output_attentions, **kwargs)
    T3HuggingfaceBackend.forward = _forward
    T3HuggingfaceBackend._fast_attn_patched = True
    print("[tts] fast-attention enabled (SDPA; output_attentions off for English model)")

def _load_model():
    global _model, _device
    import torch
    from chatterbox.tts import ChatterboxTTS

    _enable_fast_attention()
    _device = pick_device()
    # Checkpoints are saved for CUDA; make torch.load map tensors to our device
    # so loading works on Apple Silicon (mps) / CPU.
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device(_device))
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
    try:
        _model = ChatterboxTTS.from_pretrained(device=_device)
    except Exception as e:
        # fall back to CPU if the accelerator path fails
        print(f"[tts] {_device} load failed ({e}); falling back to cpu")
        _device = "cpu"
        _model = ChatterboxTTS.from_pretrained(device="cpu")
    finally:
        torch.load = _orig_load
    print(f"[tts] Chatterbox loaded on {_device}")
    return _model

def warm_up():
    """Load the model and run one tiny generation so the expensive first-call
    graph/kernel warmup on mps happens at startup, not on the user's first /tts.
    Runs in a background thread so the server still binds the port immediately."""
    try:
        model = get_model()
        model.generate("warm up.", cfg_weight=DEFAULTS["cfg_weight"])
        print("[tts] warm-up complete — first request will be fast")
    except Exception as e:
        print(f"[tts] warm-up skipped ({e})")

@app.on_event("startup")
def _startup():
    import threading
    threading.Thread(target=warm_up, daemon=True).start()

# --- ffmpeg helper to normalize uploaded samples to 24k mono wav --------------

def to_wav(src_bytes: bytes, out_path: Path):
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "24000", "-ac", "1", str(out_path)],
            input=src_bytes, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if p.returncode == 0 and out_path.exists():
            return
    # fallback: write raw bytes (Chatterbox can still load mp3/wav via its own loader)
    out_path.write_bytes(src_bytes)

# --- routes -------------------------------------------------------------------

@app.get("/")
def root():
    return {"service": "podcast-studio-tts", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {"ok": True, "device": _device, "model_loaded": _model is not None}

@app.get("/voices")
def voices():
    reg = load_registry()
    out = [{"id": "default", "name": "Chatterbox Default"}]
    for vid, meta in reg.items():
        out.append({"id": vid, "name": meta["name"]})
    return {"voices": out}

@app.post("/clone")
async def clone(name: str = Form("My Voice"), file: UploadFile = None):
    if file is None:
        raise HTTPException(400, "No audio file provided.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file.")
    vid = "v_" + uuid.uuid4().hex[:8]
    ref_path = VOICES_DIR / f"{vid}.wav"
    to_wav(data, ref_path)
    reg = load_registry()
    reg[vid] = {"name": name[:60], "file": ref_path.name}
    save_registry(reg)
    return {"id": vid, "name": reg[vid]["name"]}

@app.post("/tts")
async def tts(payload: dict):
    text = (payload.get("text") or "").strip()
    voice_id = payload.get("voice_id") or "default"
    if not text:
        raise HTTPException(400, "No text.")

    model = get_model()

    ref = None
    if voice_id != "default":
        reg = load_registry()
        meta = reg.get(voice_id)
        if not meta:
            raise HTTPException(404, f"Unknown voice {voice_id}")
        ref = str(VOICES_DIR / meta["file"])

    # Optional per-request generation params (quality/voice control).
    gen = dict(DEFAULTS)
    for k in DEFAULTS:
        if payload.get(k) is not None:
            gen[k] = float(payload[k])
    # cfg_weight must stay > 0: cfg_weight=0 crashes chatterbox-tts 0.1.7's
    # standard model (batch-shape mismatch in the T3 inference loop).
    if gen["cfg_weight"] <= 0:
        gen["cfg_weight"] = DEFAULTS["cfg_weight"]
    if ref:
        gen["audio_prompt_path"] = ref  # cloning path — must keep working

    try:
        wav = model.generate(text, **gen)
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {e}")

    audio = wav.squeeze().detach().cpu().numpy()
    # Release GPU memory after each synthesis so back-to-back requests (e.g. a
    # multi-host podcast rendering many segments in a row) don't accumulate on the
    # MPS allocator, slow to a crawl, and get the service SIGKILL'd. Additive only —
    # no change to output or behavior.
    try:
        del wav
        import gc
        gc.collect()
        if _device == "mps":
            import torch
            torch.mps.empty_cache()
    except Exception:
        pass
    buf = io.BytesIO()
    sf.write(buf, audio, model.sr, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TTS_PORT", "5050"))
    uvicorn.run(app, host="127.0.0.1", port=port)
