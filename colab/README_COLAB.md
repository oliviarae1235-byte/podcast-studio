# Cloud Voice Engine (free GPU via Google Colab)

This makes the voices **fast** and makes **cloning work**, all for free. The voice engine
runs on Colab's GPU; your Mac app just points at it.

## Steps

1. Go to https://colab.research.google.com → File → Upload notebook → pick
   `podcast_voice_engine.ipynb` (from this folder).
2. **Runtime → Change runtime type → T4 GPU → Save.** (Don't skip — this is the GPU.)
3. Run the 3 cells top to bottom (click ▶ on each, wait for each to finish).
4. The last cell prints:  `YOUR PUBLIC URL: https://xxxx.trycloudflare.com`
   Copy that whole URL.
5. On your Mac, open `.env` and set:
   ```
   LOCAL_TTS_URL=https://xxxx.trycloudflare.com
   ```
   (replace with your real URL). Save.
6. Restart the Mac app:  `npm start`  →  open http://localhost:3000
7. In Voices, click **Reload voices**. Generate is now fast.

## Important notes
- **Keep the Colab tab open** while you use the app. If you close it the URL dies.
- The URL **changes every time** you re-run the notebook. If you restart Colab, copy the new
  URL into `.env` again and restart `npm start`.
- Free Colab disconnects after idle (~90 min) or long sessions. Just re-run the cells; you'll
  get a fresh URL.
- The Mac app no longer needs its own local Python TTS service when using Colab. You can ignore
  `tts_local/` entirely while in cloud mode.

## Cloning voices (now that it's fast)
1. Record ~20-30s clips. On Mac: Voice Memos → record → share → Save to Files (.m4a is fine).
   Or download two different narrators from https://librivox.org (public domain).
2. In the app's Clone box: name it, choose the clip, click Clone.
3. It appears in the voice list (after Reload voices). Assign one to Host A, another to Host B.
4. Two distinct voices, free.
