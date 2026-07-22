// Podcast Studio — local AI podcast generator
// Pipeline: source text -> LLM dialogue script -> TTS per line -> stitched MP3
//
// Voices can come from three providers:
//   - "local"     : Chatterbox running on your machine (tts_local/app.py) — free, supports cloning
//   - "eleven"    : ElevenLabs API (optional; only if ELEVENLABS_API_KEY is set)
//   - "replicate" : the SAME Chatterbox model on a Replicate cloud GPU — much faster,
//                   supports cloning (optional; only if REPLICATE_API_TOKEN is set)
// Voice ids are prefixed: "local:<id>", "eleven:<id>", or "replicate:<id>".

import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, rm, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { randomUUID, createHmac, timingSafeEqual, scryptSync, randomBytes } from 'node:crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PORT = process.env.PORT || 3000;
const ELEVEN_KEY = process.env.ELEVENLABS_API_KEY || '';
const ELEVEN_MODEL = process.env.ELEVENLABS_MODEL || 'eleven_multilingual_v2';
const LOCAL_TTS_URL = process.env.LOCAL_TTS_URL || 'http://127.0.0.1:5050';
const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN || '';
const REPLICATE_MODEL = process.env.REPLICATE_MODEL || 'resemble-ai/chatterbox-multilingual';
// Replicate clones are stored on the Node side (the others live in their own services:
// local clones in tts_local/voices/registry.json, eleven voices in the ElevenLabs account).
const REPLICATE_VOICE_DIR = path.join(__dirname, 'voices_replicate');
const REPLICATE_REGISTRY = path.join(REPLICATE_VOICE_DIR, 'registry.json');
// Finished episodes are written here so long/batch jobs survive the browser
// closing — you can come back and download them. Served read-only at /output.
const OUTPUT_DIR = path.join(__dirname, 'output');
const LLM_PROVIDER = (process.env.LLM_PROVIDER || 'openai').toLowerCase();
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;

const app = express();
app.set('trust proxy', 1); // behind cloudflared: real client IP + https detection
app.use(express.json({ limit: '2mb' }));

// One canonical public URL: alias hosts (bare domain, www) redirect to it so
// sessions and Google Sign-In always live on a single origin. Local access
// (localhost) is never redirected. Unset CANONICAL_HOST -> no redirects.
const CANONICAL_HOST = process.env.CANONICAL_HOST || '';
app.use((req, res, next) => {
  if (!CANONICAL_HOST || req.hostname === CANONICAL_HOST || req.hostname === 'localhost' || req.hostname === '127.0.0.1') return next();
  res.redirect(301, `https://${CANONICAL_HOST}${req.originalUrl}`);
});

// ---------- accounts & access control ----------
// Hosted mode: when APP_PASSWORD and/or GOOGLE_CLIENT_ID is set in .env the app
// shows a sign-in / create-account page, and every episode, output file, and
// cloned voice belongs to the account that made it. A fresh laptop install
// leaves these unset -> no login step, nothing scoped, exactly like before.
const APP_PASSWORD = process.env.APP_PASSWORD || '';
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || '';
// Optional sign-up allowlist; empty = anyone may create an account.
const ALLOWED_EMAILS = (process.env.ALLOWED_EMAILS || '').split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
// Admin accounts also see pre-account ("legacy") episodes, files, and voices.
const ADMIN_EMAILS = (process.env.ADMIN_EMAILS || '').split(',').map((s) => s.trim().toLowerCase()).filter(Boolean);
const AUTH_ON = !!(APP_PASSWORD || GOOGLE_CLIENT_ID);
// Without a fixed SESSION_SECRET every restart logs everyone out.
const SESSION_SECRET = process.env.SESSION_SECRET || randomUUID();
const SESSION_DAYS = 30;
const AUTH_COOKIE = 'ps_auth';

// -- user + voice-ownership stores (data/*.json) --
const DATA_DIR = path.join(__dirname, 'data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');
const VOICE_OWNERS_FILE = path.join(DATA_DIR, 'voice_owners.json');
let users = null;        // [{id, email, name, passwordHash?, createdAt}]
let voiceOwners = null;  // { "local:v_abc123": userId, ... }
async function loadUsers() {
  if (!users) { try { users = JSON.parse(await readFile(USERS_FILE, 'utf8')); } catch { users = []; } }
  return users;
}
async function saveUsers() { await mkdir(DATA_DIR, { recursive: true }); await writeFile(USERS_FILE, JSON.stringify(users, null, 2)); }
async function loadVoiceOwners() {
  if (!voiceOwners) { try { voiceOwners = JSON.parse(await readFile(VOICE_OWNERS_FILE, 'utf8')); } catch { voiceOwners = {}; } }
  return voiceOwners;
}
async function saveVoiceOwners() { await mkdir(DATA_DIR, { recursive: true }); await writeFile(VOICE_OWNERS_FILE, JSON.stringify(voiceOwners, null, 2)); }

const normEmail = (e) => String(e || '').trim().toLowerCase();
const isAdmin = (u) => !!u && ADMIN_EMAILS.includes(u.email);
const emailAllowed = (e) => !ALLOWED_EMAILS.length || ALLOWED_EMAILS.includes(normEmail(e)) || ADMIN_EMAILS.includes(normEmail(e));
async function ensureUser(email, name) {
  await loadUsers();
  const em = normEmail(email);
  let u = users.find((x) => x.email === em);
  if (!u) {
    u = { id: randomUUID().slice(0, 8), email: em, name: String(name || em.split('@')[0]).slice(0, 60), createdAt: Date.now() };
    users.push(u);
    await saveUsers();
  }
  return u;
}
const hashPassword = (pw, salt = randomBytes(16).toString('hex')) => `${salt}:${scryptSync(pw, salt, 32).toString('hex')}`;
function checkPassword(pw, stored) {
  const [salt, hex] = String(stored || '').split(':');
  if (!salt || !hex) return false;
  const want = Buffer.from(hex, 'hex');
  const got = scryptSync(pw, salt, 32);
  return want.length === got.length && timingSafeEqual(want, got);
}

// Ownership rule for episodes/outputs/voices: with auth off nothing is scoped;
// items created before accounts existed (owner == null) belong to the admins.
const canSee = (user, owner) => !AUTH_ON || (owner ? owner === user?.id : isAdmin(user));

// -- sessions: "uid.exp.hmac" in an HttpOnly cookie --
const signSession = (uid, exp) => createHmac('sha256', SESSION_SECRET).update(`${uid}.${exp}`).digest('hex');
function sessionUid(req) {
  const raw = (req.headers.cookie || '').split(/;\s*/).find((c) => c.startsWith(AUTH_COOKIE + '='));
  if (!raw) return null;
  const [uid, exp, sig] = raw.slice(AUTH_COOKIE.length + 1).split('.');
  if (!uid || !exp || !sig || !/^\d+$/.test(exp) || Number(exp) < Date.now()) return null;
  const expect = signSession(uid, exp);
  return sig.length === expect.length && timingSafeEqual(Buffer.from(sig), Buffer.from(expect)) ? uid : null;
}
function setSession(req, res, uid) {
  const exp = Date.now() + SESSION_DAYS * 24 * 3600 * 1000;
  const secure = req.secure ? '; Secure' : '';
  res.setHeader('Set-Cookie', `${AUTH_COOKIE}=${uid}.${exp}.${signSession(uid, exp)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${SESSION_DAYS * 24 * 3600}${secure}`);
}

// brute-force brake: 8 bad passwords from one IP -> locked out for 10 minutes
const failedLogins = new Map();
const throttled = (ip) => { const f = failedLogins.get(ip); return f && f.count >= 8 && Date.now() - f.at < 10 * 60 * 1000; };
const recordFail = (ip) => { const f = failedLogins.get(ip) || { count: 0, at: 0 }; f.count++; f.at = Date.now(); failedLogins.set(ip, f); };

const AUTH_OPEN = new Set(['/login', '/login.html', '/api/login', '/api/signup', '/api/auth/google', '/api/auth/config', '/api/me', '/logout', '/favicon.ico']);
app.use(async (req, res, next) => {
  req.user = null;
  if (!AUTH_ON) return next();
  const uid = sessionUid(req);
  if (uid) { await loadUsers(); req.user = users.find((u) => u.id === uid) || null; }
  if (req.user || AUTH_OPEN.has(req.path)) return next();
  if (req.path.startsWith('/api/') || req.path.startsWith('/output/')) return res.status(401).json({ error: 'Not signed in.' });
  res.redirect('/login');
});

app.get('/login', (req, res) => {
  if (!AUTH_ON || req.user) return res.redirect('/');
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

app.get('/api/auth/config', (req, res) => res.json({ authOn: AUTH_ON, googleClientId: GOOGLE_CLIENT_ID || null, openSignup: !ALLOWED_EMAILS.length }));
app.get('/api/me', (req, res) => res.json({ user: req.user ? { email: req.user.email, name: req.user.name, admin: isAdmin(req.user) } : null }));

app.post('/api/signup', async (req, res) => {
  try {
    if (!AUTH_ON) return res.status(400).json({ error: 'Accounts are disabled on local installs.' });
    const ip = req.ip || 'unknown';
    if (throttled(ip)) return res.status(429).json({ error: 'Too many attempts. Try again in 10 minutes.' });
    const name = String(req.body?.name || '').trim().slice(0, 60);
    const email = normEmail(req.body?.email);
    const password = String(req.body?.password || '');
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return res.status(400).json({ error: 'Enter a valid email address.' });
    if (password.length < 8) return res.status(400).json({ error: 'Password must be at least 8 characters.' });
    if (!emailAllowed(email)) return res.status(403).json({ error: 'Sign-ups are restricted on this server.' });
    await loadUsers();
    if (users.some((u) => u.email === email)) { recordFail(ip); return res.status(409).json({ error: 'That email already has an account — sign in instead.' }); }
    const u = { id: randomUUID().slice(0, 8), email, name: name || email.split('@')[0], passwordHash: hashPassword(password), createdAt: Date.now() };
    users.push(u);
    await saveUsers();
    setSession(req, res, u.id);
    res.json({ ok: true, email });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.post('/api/login', async (req, res) => {
  try {
    const ip = req.ip || 'unknown';
    if (throttled(ip)) return res.status(429).json({ error: 'Too many attempts. Try again in 10 minutes.' });
    const email = normEmail(req.body?.email);
    const password = String(req.body?.password || '');
    if (!email || !password) return res.status(400).json({ error: 'Email and password required.' });
    await loadUsers();
    const u = users.find((x) => x.email === email);
    if (u?.passwordHash && checkPassword(password, u.passwordHash)) {
      failedLogins.delete(ip);
      setSession(req, res, u.id);
      return res.json({ ok: true, email });
    }
    // The owner's master password from .env works for admin emails even if that
    // account never set its own password (e.g. it was created via Google).
    if (APP_PASSWORD && ADMIN_EMAILS.includes(email)) {
      const given = Buffer.from(password);
      const want = Buffer.from(APP_PASSWORD);
      if (given.length === want.length && timingSafeEqual(given, want)) {
        const admin = await ensureUser(email);
        failedLogins.delete(ip);
        setSession(req, res, admin.id);
        return res.json({ ok: true, email });
      }
    }
    recordFail(ip);
    res.status(401).json({ error: 'Wrong email or password.' });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

// Google Sign-In: the login page posts the ID token (credential) from Google's
// button; Google's tokeninfo endpoint checks the signature, we check that the
// token was issued for OUR client id, then sign the account in (creating it on
// first use).
app.post('/api/auth/google', async (req, res) => {
  try {
    if (!GOOGLE_CLIENT_ID) return res.status(400).json({ error: 'Google sign-in is not configured.' });
    const credential = String(req.body?.credential || '');
    if (!credential) return res.status(400).json({ error: 'Missing Google credential.' });
    const r = await fetch(`https://oauth2.googleapis.com/tokeninfo?id_token=${encodeURIComponent(credential)}`);
    const info = await r.json().catch(() => ({}));
    if (!r.ok || info.aud !== GOOGLE_CLIENT_ID || info.email_verified !== 'true') {
      return res.status(401).json({ error: 'Google sign-in could not be verified.' });
    }
    if (!emailAllowed(info.email)) return res.status(403).json({ error: 'Sign-ups are restricted on this server.' });
    const u = await ensureUser(info.email, info.name);
    setSession(req, res, u.id);
    res.json({ ok: true, email: u.email });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.get('/logout', (req, res) => {
  res.setHeader('Set-Cookie', `${AUTH_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`);
  res.redirect('/login');
});

app.use(express.static(path.join(__dirname, 'public')));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });

// ---------- LLM script generation ----------

function buildScriptPrompt({ source, hostA, hostB, tone, minutes, language }) {
  const targetWords = Math.round((Number(minutes) || 4) * 140);
  return `You are a podcast script writer. Turn the SOURCE CONTENT below into a natural, engaging two-person podcast conversation.

Hosts:
- "${hostA}" (host A) — warm, curious, drives the conversation, asks the questions a listener would.
- "${hostB}" (host B) — the explainer, gives clear answers, occasional light humor.

Requirements:
- Language: ${language}.
- Tone/style: ${tone}.
- Target length: about ${targetWords} words total (roughly ${minutes} minutes spoken).
- Open with a short hook, not "Welcome to the podcast". End with a clean sign-off.
- Conversational: contractions, reactions, short back-and-forth. Avoid reading like an essay.
- Do NOT invent facts not supported by the source. If the source is thin, keep it shorter.
- No stage directions, no sound-effect notes, no markdown. Just spoken lines.

Return ONLY valid JSON, no preamble, no code fences:
{"title": "short episode title", "lines": [{"speaker": "A", "text": "..."}, {"speaker": "B", "text": "..."}]}

SOURCE CONTENT:
"""
${source.slice(0, 40000)}
"""`;
}

async function generateScript(params) {
  const prompt = buildScriptPrompt(params);
  if (LLM_PROVIDER === 'anthropic') {
    if (!ANTHROPIC_KEY) throw new Error('ANTHROPIC_API_KEY not set.');
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': ANTHROPIC_KEY, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({ model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514', max_tokens: 4000, messages: [{ role: 'user', content: prompt }] }),
    });
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text()}`);
    const data = await res.json();
    return parseScript((data.content || []).map((b) => b.text || '').join('\n'));
  }
  if (!OPENAI_KEY) throw new Error('OPENAI_API_KEY not set.');
  const base = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${OPENAI_KEY}` },
    body: JSON.stringify({ model: process.env.OPENAI_MODEL || 'gpt-4o-mini', messages: [{ role: 'user', content: prompt }], temperature: 0.8 }),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return parseScript(data.choices?.[0]?.message?.content || '');
}

function parseScript(raw) {
  let text = (raw || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '');
  const s = text.indexOf('{'), e = text.lastIndexOf('}');
  if (s !== -1 && e !== -1) text = text.slice(s, e + 1);
  const obj = JSON.parse(text);
  if (!Array.isArray(obj.lines) || !obj.lines.length) throw new Error('Script had no lines.');
  obj.lines = obj.lines
    .map((l) => ({ speaker: String(l.speaker || 'A').toUpperCase().startsWith('B') ? 'B' : 'A', text: String(l.text || '').trim() }))
    .filter((l) => l.text);
  obj.title = obj.title || 'Untitled Episode';
  return obj;
}

// ---------- voice providers ----------

async function elevenVoices() {
  if (!ELEVEN_KEY) return [];
  try {
    const r = await fetch('https://api.elevenlabs.io/v1/voices', { headers: { 'xi-api-key': ELEVEN_KEY } });
    if (!r.ok) return [];
    const d = await r.json();
    return (d.voices || []).map((v) => ({
      voice_id: `eleven:${v.voice_id}`, name: v.name, category: v.category || 'premade',
      labels: v.labels || {}, preview_url: v.preview_url,
    }));
  } catch { return []; }
}

async function localVoices() {
  try {
    const regPath = path.join(__dirname, 'tts_local', 'voices', 'registry.json');
    const reg = JSON.parse(await readFile(regPath, 'utf8'));
    return [
      { voice_id: 'local:default', name: 'Chatterbox Default', category: 'local · built-in', labels: {}, preview_url: null },
      ...Object.entries(reg).map(([id, meta]) => ({
        voice_id: `local:${id}`, name: meta.name || id, category: 'local · clone', labels: {}, preview_url: null,
      })),
    ];
  } catch { return []; }
}

// A voice is visible to a user if they cloned it; pre-account clones belong to
// the admins; the built-in default is visible to everyone. No-auth mode: all.
async function voiceVisible(user, voiceId) {
  if (!AUTH_ON || String(voiceId).endsWith(':default')) return true;
  await loadVoiceOwners();
  return canSee(user, voiceOwners[voiceId] || null);
}

// ---- Replicate (same Chatterbox model, cloud GPU) ----

async function loadReplicateReg() {
  try { return JSON.parse(await readFile(REPLICATE_REGISTRY, 'utf8')); }
  catch { return {}; } // missing file -> empty registry
}
async function saveReplicateReg(reg) {
  await mkdir(REPLICATE_VOICE_DIR, { recursive: true });
  await writeFile(REPLICATE_REGISTRY, JSON.stringify(reg, null, 2));
}

async function replicateVoices() {
  if (!REPLICATE_TOKEN) return [];
  const reg = await loadReplicateReg();
  const out = [{
    voice_id: 'replicate:default', name: 'Chatterbox (Replicate)',
    category: 'replicate · cloud', labels: {}, preview_url: null,
  }];
  for (const [id, meta] of Object.entries(reg)) {
    out.push({ voice_id: `replicate:${id}`, name: meta.name, category: 'replicate · clone', labels: {}, preview_url: null });
  }
  return out;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const replicateHeaders = () => ({ authorization: `Bearer ${REPLICATE_TOKEN}`, 'content-type': 'application/json' });

// Resolve (and cache) the model's latest version id. The version-based
// predictions endpoint works for community models; the model-level endpoint
// (/v1/models/.../predictions) is only for Replicate's official models.
let _replicateVersion = null;
async function replicateVersion() {
  if (_replicateVersion) return _replicateVersion;
  const r = await fetch(`https://api.replicate.com/v1/models/${REPLICATE_MODEL}`, { headers: replicateHeaders() });
  const m = await r.json();
  if (!r.ok || !m.latest_version?.id) throw new Error(`Replicate model lookup ${r.status}: ${m.detail || JSON.stringify(m)}`);
  _replicateVersion = m.latest_version.id;
  return _replicateVersion;
}

// Create a prediction on the model's latest version, then poll until terminal.
async function replicatePredict(input) {
  if (!REPLICATE_TOKEN) throw new Error('REPLICATE_API_TOKEN not set.');
  const headers = replicateHeaders();
  const version = await replicateVersion();
  const createRes = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST', headers, body: JSON.stringify({ version, input }),
  });
  let pred = await createRes.json();
  if (!createRes.ok) throw new Error(`Replicate create ${createRes.status}: ${pred.detail || JSON.stringify(pred)}`);

  const getUrl = pred.urls?.get || `https://api.replicate.com/v1/predictions/${pred.id}`;
  const deadline = Date.now() + 5 * 60 * 1000; // 5 min safety cap
  while (!['succeeded', 'failed', 'canceled'].includes(pred.status)) {
    if (Date.now() > deadline) throw new Error('Replicate prediction timed out.');
    await sleep(1500);
    const pollRes = await fetch(getUrl, { headers });
    pred = await pollRes.json();
    if (!pollRes.ok) throw new Error(`Replicate poll ${pollRes.status}: ${pred.detail || JSON.stringify(pred)}`);
  }
  if (pred.status !== 'succeeded') throw new Error(`Replicate ${pred.status}: ${pred.error || 'no output'}`);
  return pred.output;
}

// One line -> { buffer, ext }. id is "default" or a clone id from the registry.
async function replicateSynth(text, id) {
  // Field names per the live resemble-ai/chatterbox-multilingual schema:
  // text, language, cfg_weight, exaggeration, temperature, reference_audio.
  const input = {
    text,
    language: 'en',
    cfg_weight: 0.5,
    exaggeration: 0.5,
    temperature: 0.8,
  };
  if (id !== 'default') {
    const reg = await loadReplicateReg();
    const meta = reg[id];
    if (!meta) throw new Error(`Unknown Replicate voice ${id}`);
    if (meta.language) input.language = meta.language;
    // Send the stored reference clip inline as a data URI so cloning never
    // depends on a (possibly expiring) hosted URL.
    const bytes = await readFile(path.join(REPLICATE_VOICE_DIR, meta.file));
    input.reference_audio = `data:${meta.mime || 'audio/wav'};base64,${bytes.toString('base64')}`;
  }
  const output = await replicatePredict(input);
  // Chatterbox returns an audio file URL (sometimes wrapped in an array/object).
  const url = Array.isArray(output) ? output[0]
    : typeof output === 'string' ? output
    : output?.audio_out || output?.audio || output?.output;
  if (!url || typeof url !== 'string') throw new Error('Replicate returned no audio URL.');
  const dl = await fetch(url);
  if (!dl.ok) throw new Error(`Replicate output download ${dl.status}`);
  const ext = (path.extname(new URL(url).pathname).slice(1) || 'wav').toLowerCase();
  return { buffer: Buffer.from(await dl.arrayBuffer()), ext };
}

function splitVoice(v) {
  const i = String(v).indexOf(':');
  return { provider: v.slice(0, i), id: v.slice(i + 1) };
}

// returns { buffer, ext } for one line. An optional AbortSignal lets callers cap
// how long a single synthesis may take (used to detect a hung local engine).
async function synth(text, voiceRef, { signal } = {}) {
  const { provider, id } = splitVoice(voiceRef);
  if (provider === 'eleven') {
    if (!ELEVEN_KEY) throw new Error('ElevenLabs key not configured.');
    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${id}`, {
      method: 'POST',
      headers: { 'xi-api-key': ELEVEN_KEY, 'content-type': 'application/json', accept: 'audio/mpeg' },
      body: JSON.stringify({ text, model_id: ELEVEN_MODEL, voice_settings: { stability: 0.5, similarity_boost: 0.75, use_speaker_boost: true } }),
      signal,
    });
    if (!r.ok) throw new Error(`ElevenLabs: ${await r.text()}`);
    return { buffer: Buffer.from(await r.arrayBuffer()), ext: 'mp3' };
  }
  if (provider === 'replicate') {
    return replicateSynth(text, id);
  }
  // local Chatterbox
  const r = await fetch(`${LOCAL_TTS_URL}/tts`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, voice_id: id }),
    signal,
  });
  if (!r.ok) throw new Error(`Local TTS: ${await r.text()}`);
  return { buffer: Buffer.from(await r.arrayBuffer()), ext: 'wav' };
}

// ---------- audio stitching (normalizes mixed wav/mp3 -> one mp3) ----------

function ffmpegOk() {
  return new Promise((res) => { const p = spawn('ffmpeg', ['-version']); p.on('error', () => res(false)); p.on('close', (c) => res(c === 0)); });
}
function run(cmd, args) {
  return new Promise((res, rej) => { const p = spawn(cmd, args); let e = ''; p.stderr.on('data', (d) => (e += d)); p.on('close', (c) => (c === 0 ? res() : rej(new Error(e.slice(-400))))); });
}
const GAP_SEC = 0.4;   // consistent pause inserted between every segment
// Trim leading AND trailing silence (the second silenceremove runs on the reversed
// audio, then we reverse back). Chatterbox often leaves ragged 1-3s tails/heads;
// this removes them so the only pause between lines is our fixed GAP_SEC. A little
// head/tail (start_silence) is kept so we never clip the first/last phoneme.
const TRIM_FILTER = [
  'silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB:detection=peak',
  'areverse',
  'silenceremove=start_periods=1:start_silence=0.05:start_threshold=-50dB:detection=peak',
  'areverse',
].join(',');

async function stitch(segPaths, outPath) {
  const dir = path.dirname(segPaths[0] || outPath);
  // 1. trim silence off both ends of each segment and unify the format
  const norm = [];
  for (let i = 0; i < segPaths.length; i++) {
    const n = segPaths[i] + '.norm.wav';
    await run('ffmpeg', ['-y', '-i', segPaths[i], '-af', TRIM_FILTER, '-ar', '44100', '-ac', '1', n]);
    norm.push(n);
  }
  // 2. one short silent clip to drop between segments for a consistent gap
  const gap = path.join(dir, 'gap.wav');
  await run('ffmpeg', ['-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', String(GAP_SEC), gap]);
  // 3. interleave: seg, gap, seg, gap, … seg  (no gap before the first / after the last)
  const items = [];
  norm.forEach((p, i) => { if (i > 0) items.push(gap); items.push(p); });
  // Keep the concat list in the temp dir so no stray .txt lands next to the mp3.
  const list = path.join(dir, 'concat.txt');
  await writeFile(list, items.map((p) => `file '${p.replace(/'/g, "'\\''")}'`).join('\n'));
  await run('ffmpeg', ['-y', '-f', 'concat', '-safe', '0', '-i', list, '-c:a', 'libmp3lame', '-q:a', '4', outPath]);
}

// --- resilient per-line synthesis (handles the local engine hanging mid-job) ---

const CHUNK_SIZE = 7;                    // episodes are synthesized in batches of ~6-8 lines
const LINE_STALL_MS = 12 * 60 * 1000;    // a single line taking longer than this = a hung engine
const LINE_MAX_ATTEMPTS = 8;             // attempts per line before giving up on it (higher for sleep/wake tolerance)

// Restart the local TTS LaunchAgent and wait until it reports the model is loaded.
// Used to recover from a hung Chatterbox engine without any manual intervention.
async function restartTts() {
  console.log('[chunk] restarting local TTS service to clear a hang…');
  await new Promise((resolve) => {
    const p = spawn('launchctl', ['kickstart', '-k', `gui/${process.getuid()}/com.podcaststudio.tts`], { stdio: 'ignore' });
    p.on('error', () => resolve());
    p.on('close', () => resolve());
  });
  await sleep(15000); // let macOS reclaim GPU memory from the killed process before reloading the model
  const deadline = Date.now() + 3 * 60 * 1000;   // up to 3 min — model reload after sleep can be slow
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`${LOCAL_TTS_URL}/health`, { signal: AbortSignal.timeout(3000) });
      if (r.ok) { const d = await r.json(); if (d.model_loaded) { console.log('[chunk] TTS back up and warm.'); return; } }
    } catch {}
    await sleep(2000);
  }
  console.log('[chunk] TTS did not report ready within 3 min — retrying the line anyway.');
}

// Synthesize one line, aborting if it stalls past LINE_STALL_MS or if the job is cancelled.
function synthOnce(text, ref, cancelSignal) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('per-line stall timeout')), LINE_STALL_MS);
  const onCancel = () => ctrl.abort(new Error('Cancelled by user'));
  if (cancelSignal) cancelSignal.addEventListener('abort', onCancel, { once: true });
  return synth(text, ref, { signal: ctrl.signal }).finally(() => {
    clearTimeout(timer);
    if (cancelSignal) cancelSignal.removeEventListener('abort', onCancel);
  });
}

// One line with stall detection + recovery: if it hangs, restart the engine and
// retry; if it errors transiently, pause briefly and retry. Throws only after
// LINE_MAX_ATTEMPTS so one bad line doesn't silently drop, but a hang never wedges
// the whole job forever.
async function synthLine(text, ref, lineNo, cancelSignal) {
  for (let attempt = 1; attempt <= LINE_MAX_ATTEMPTS; attempt++) {
    if (cancelSignal?.aborted) throw new Error('Cancelled by user');
    try {
      return await synthOnce(text, ref, cancelSignal);
    } catch (e) {
      if (cancelSignal?.aborted) throw new Error('Cancelled by user');
      const stalled = e?.name === 'AbortError' || e?.code === 'ABORT_ERR' || e?.cause?.name === 'AbortError' || e?.cause?.code === 'ABORT_ERR' || /stall|timeout|abort/i.test(String(e?.message)) || /fetch failed/i.test(String(e?.message)) || /out of memory/i.test(String(e?.message));
      console.log(`[chunk] line ${lineNo} attempt ${attempt}/${LINE_MAX_ATTEMPTS} failed${stalled ? ' (stall)' : ''}: ${e?.message || e}`);
      if (attempt === LINE_MAX_ATTEMPTS) throw new Error(`stalled/failed after ${attempt} attempts: ${e?.message || e}`);
      if (stalled) await restartTts();   // hung engine -> reset it, then retry the line
      else await sleep(1500);            // transient error -> brief pause, then retry
    }
  }
}

// Clean up punctuation that trips Chatterbox into dropping words (smart quotes,
// dashes, ellipses, repeated/markdown punctuation) and guarantee the text ends on a
// terminator so the model knows where to stop (a common cause of dropped endings).
function normalizePunctuation(text) {
  let t = String(text || '')
    .replace(/[‘’ʼ]/g, "'")        // curly single quotes/apostrophes -> '
    .replace(/[“”]/g, '"')              // curly double quotes -> "
    .replace(/\s*[–—]\s*/g, ', ')        // en/em dash -> comma pause
    .replace(/…/g, '. ')                      // … -> period
    .replace(/&/g, ' and ')
    .replace(/[*_`#]+/g, '')                       // stray markdown
    .replace(/\s*\n+\s*/g, ' ')                    // newlines -> space
    .replace(/\.{2,}/g, '.')                       // .. -> .
    .replace(/([!?]){2,}/g, '$1')                  // !!! -> !,  ??? -> ?
    .replace(/,\s*([.!?;:])/g, '$1')               // ", ." -> "."
    .replace(/\s+([,.!?;:])/g, '$1')               // space before punctuation
    .replace(/\s{2,}/g, ' ')
    .trim();
  if (t && !/[.!?,;:"')\]]$/.test(t)) t += '.';     // ensure a terminal pause
  return t;
}

const MAX_PIECE_CHARS = 220;   // longer lines get split into ~sentence-sized pieces

// Greedily pack text into <=MAX_PIECE_CHARS pieces, preferring sentence boundaries,
// then clause (comma/semicolon) boundaries, then words — so no single synth call is
// long enough to provoke Chatterbox's mid-line word drops.
function packBy(parts, max) {
  const out = [];
  let cur = '';
  for (const raw of parts) {
    const p = raw.trim();
    if (!p) continue;
    if ((cur ? cur.length + 1 : 0) + p.length > max && cur) { out.push(cur); cur = ''; }
    cur = cur ? cur + ' ' + p : p;
  }
  if (cur) out.push(cur);
  return out;
}
function splitText(text, max = MAX_PIECE_CHARS) {
  const t = text.trim();
  if (t.length <= max) return [t];
  const sentences = t.match(/[^.!?]+[.!?]+["')\]]*|\S[^.!?]*$/g) || [t];
  const pieces = [];
  for (const s of sentences) {
    if (s.trim().length <= max) { pieces.push(s.trim()); continue; }
    const clauses = packBy(s.split(/(?<=[,;:])\s+/), max);     // try clause boundaries
    for (const c of clauses) {
      if (c.length <= max) pieces.push(c);
      else pieces.push(...packBy(c.split(/\s+/), max));        // last resort: by words
    }
  }
  return packBy(pieces, max);   // re-pack so short adjacent sentences ride together
}

// Synthesize every line and stitch ONE mp3 at outPath. Internally the lines are
// processed in chunks of ~CHUNK_SIZE so a long episode is a series of small batches
// rather than one marathon run; each line's text is punctuation-normalized and split
// into sentence-sized pieces (to avoid dropped words), and each piece is synthesized
// resiliently (a hung engine is detected and auto-restarted, then retried). All
// segments are stitched into a single file at the end — callers and the user only
// ever see one episode, never the chunks/pieces. onProgress(done,total) fires per line.
async function buildEpisode(lines, voiceA, voiceB, outPath, onProgress, cancelSignal) {
  const workDir = path.join(os.tmpdir(), 'podcast-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8));
  await mkdir(workDir, { recursive: true });
  try {
    const segs = [];
    const chunks = Math.ceil(lines.length / CHUNK_SIZE);
    for (let c = 0; c < lines.length; c += CHUNK_SIZE) {
      const chunk = lines.slice(c, c + CHUNK_SIZE);
      console.log(`[chunk] batch ${c / CHUNK_SIZE + 1}/${chunks}: lines ${c + 1}-${c + chunk.length} of ${lines.length}`);
      for (let j = 0; j < chunk.length; j++) {
        const i = c + j;
        const ref = lines[i].speaker === 'B' ? voiceB : voiceA;
        const pieces = splitText(normalizePunctuation(lines[i].text));
        for (let k = 0; k < pieces.length; k++) {
          const label = pieces.length > 1 ? `${i + 1}.${k + 1}/${pieces.length}` : `${i + 1}`;
          let out;
          if (cancelSignal?.aborted) throw new Error('Cancelled by user');
          try { out = await synthLine(pieces[k], ref, label, cancelSignal); }
          catch (err) { throw new Error(`Line ${i + 1}: ${err.message}`); }
          const p = path.join(workDir, `seg_${String(segs.length).padStart(4, '0')}.${out.ext}`);
          await writeFile(p, out.buffer);
          segs.push(p);
        }
        if (onProgress) onProgress(i + 1, lines.length);   // progress is per line, not per piece
      }
    }
    await stitch(segs, outPath);   // every chunk's segments -> one final mp3
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
}

// ---------- routes ----------

// Only the user's own local Chatterbox clones are shown. We deliberately skip the
// ElevenLabs and Replicate providers: ElevenLabs' /voices is an internet fetch on
// every page load (the slow part) and only contributed premade sample voices the
// user doesn't use. Local voices come from the localhost TTS service, so this is
// instant. The built-in "Chatterbox Default" sample is filtered out too — unless the
// user has no clones yet, in which case it's kept so the list is never empty.
app.get('/api/voices', async (req, res) => {
  const local = await localVoices();
  const visible = [];
  for (const v of local) if (await voiceVisible(req.user, v.voice_id)) visible.push(v);
  const clones = visible.filter((v) => v.voice_id !== 'local:default');
  res.json({ voices: clones.length ? clones : visible });
});

app.get('/api/voices/:id/audio', async (req, res) => {
  try {
    if (!(await voiceVisible(req.user, `local:${req.params.id}`))) return res.status(404).json({ error: 'Voice not found' });
    const regPath = path.join(__dirname, 'tts_local', 'voices', 'registry.json');
    const reg = JSON.parse(await readFile(regPath, 'utf8'));
    const meta = reg[req.params.id];
    if (!meta) return res.status(404).json({ error: 'Voice not found' });
    const filePath = path.join(__dirname, 'tts_local', 'voices', meta.file);
    res.sendFile(filePath);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.patch('/api/voices/:id', async (req, res) => {
  try {
    if (!(await voiceVisible(req.user, `local:${req.params.id}`))) return res.status(404).json({ error: 'Voice not found' });
    const { name } = req.body || {};
    if (!name || !name.trim()) return res.status(400).json({ error: 'Name required' });
    const regPath = path.join(__dirname, 'tts_local', 'voices', 'registry.json');
    const reg = JSON.parse(await readFile(regPath, 'utf8'));
    if (!reg[req.params.id]) return res.status(404).json({ error: 'Voice not found' });
    reg[req.params.id].name = name.trim().slice(0, 60);
    await writeFile(regPath, JSON.stringify(reg, null, 2));
    res.json({ ok: true, name: reg[req.params.id].name });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.delete('/api/voices/:id', async (req, res) => {
  try {
    if (!(await voiceVisible(req.user, `local:${req.params.id}`))) return res.status(404).json({ error: 'Voice not found' });
    const regPath = path.join(__dirname, 'tts_local', 'voices', 'registry.json');
    const reg = JSON.parse(await readFile(regPath, 'utf8'));
    const meta = reg[req.params.id];
    if (!meta) return res.status(404).json({ error: 'Voice not found' });
    const filePath = path.join(__dirname, 'tts_local', 'voices', meta.file);
    await rm(filePath, { force: true });
    delete reg[req.params.id];
    await writeFile(regPath, JSON.stringify(reg, null, 2));
    await loadVoiceOwners();
    delete voiceOwners[`local:${req.params.id}`];
    await saveVoiceOwners();
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// Clone defaults to the free local service. Pass provider=replicate to store a
// reference clip for the Replicate (cloud Chatterbox) provider instead.
app.post('/api/clone', upload.array('samples', 5), async (req, res) => {
  const provider = (req.body.provider || 'local').toLowerCase();
  if (provider === 'replicate') {
    try {
      if (!REPLICATE_TOKEN) return res.status(400).json({ error: 'REPLICATE_API_TOKEN not set.' });
      if (!req.files || !req.files.length) return res.status(400).json({ error: 'Upload at least one audio sample.' });
      const f = req.files[0]; // one reference clip, same as the local service
      const id = 'v_' + randomUUID().replace(/-/g, '').slice(0, 8);
      const ext = (path.extname(f.originalname || '').slice(1) || 'wav').toLowerCase();
      const fname = `${id}.${ext}`;
      await mkdir(REPLICATE_VOICE_DIR, { recursive: true });
      await writeFile(path.join(REPLICATE_VOICE_DIR, fname), f.buffer);
      const reg = await loadReplicateReg();
      reg[id] = {
        name: (req.body.name || 'My Voice').slice(0, 60),
        file: fname, mime: f.mimetype || 'audio/wav',
        language: (req.body.language || 'en'),
      };
      await saveReplicateReg(reg);
      if (AUTH_ON && req.user) {
        await loadVoiceOwners();
        voiceOwners[`replicate:${id}`] = req.user.id;
        await saveVoiceOwners();
      }
      return res.json({ voice_id: `replicate:${id}`, name: reg[id].name });
    } catch (e) {
      return res.status(500).json({ error: 'Could not store Replicate voice: ' + e.message });
    }
  }
  try {
    if (!req.files || !req.files.length) return res.status(400).json({ error: 'Upload at least one audio sample.' });
    const form = new FormData();
    form.append('name', (req.body.name || 'My Voice').slice(0, 60));
    // local service takes a single reference clip; use the first sample
    const f = req.files[0];
    form.append('file', new Blob([f.buffer], { type: f.mimetype || 'audio/wav' }), f.originalname || 'sample.wav');
    const r = await fetch(`${LOCAL_TTS_URL}/clone`, { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok) return res.status(r.status).json({ error: d.detail || JSON.stringify(d) });
    if (AUTH_ON && req.user) {
      await loadVoiceOwners();
      voiceOwners[`local:${d.id}`] = req.user.id;
      await saveVoiceOwners();
    }
    res.json({ voice_id: `local:${d.id}`, name: d.name });
  } catch (e) {
    res.status(500).json({ error: 'Local TTS service unreachable. Is tts_local/app.py running? ' + e.message });
  }
});

// Parse an already-written "Name: text" script verbatim — no rewriting.
// hostA/hostB are the names that map to speakers A and B (case-insensitive).
function parseVerbatim(source, hostA, hostB) {
  const lines = [];
  const a = (hostA || '').trim().toLowerCase();
  const b = (hostB || '').trim().toLowerCase();
  // match "Label: rest of line" — label is up to ~30 chars before the first colon
  const labelRe = /^\s*([^:\n]{1,30}?)\s*:\s*(.*)$/;
  let current = null; // {speaker, parts:[]}
  const flush = () => { if (current && current.parts.join(' ').trim()) lines.push({ speaker: current.speaker, text: current.parts.join(' ').replace(/\s+/g, ' ').trim() }); };

  for (const raw of source.split(/\r?\n/)) {
    const m = raw.match(labelRe);
    if (m) {
      const label = m[1].trim().toLowerCase();
      // only treat as a speaker label if it matches one of the two names
      let speaker = null;
      if (label === a) speaker = 'A';
      else if (label === b) speaker = 'B';
      if (speaker) {
        flush();
        current = { speaker, parts: m[2] ? [m[2]] : [] };
        continue;
      }
    }
    // continuation line (wrapped text) for the current speaker
    if (current && raw.trim()) current.parts.push(raw.trim());
  }
  flush();
  return lines;
}

app.post('/api/script', async (req, res) => {
  try {
    const { source, hostA = 'Alex', hostB = 'Sam', tone = 'Educational', minutes = 4, language = 'English', verbatim = false } = req.body || {};
    if (!source || source.trim().length < 20) return res.status(400).json({ error: 'Give me at least a couple sentences of source content.' });

    if (verbatim) {
      const linesV = parseVerbatim(source, hostA, hostB);
      if (!linesV.length) {
        return res.status(400).json({ error: `No lines matched "${hostA}:" or "${hostB}:". Check the host names match the labels in your script exactly.` });
      }
      return res.json({ title: 'My Script', lines: linesV });
    }

    res.json(await generateScript({ source, hostA, hostB, tone, minutes, language }));
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

// Legacy synchronous single-episode endpoint (kept for compatibility). The UI now
// drives generation through the background-job API below, which survives long runs.
app.post('/api/podcast', async (req, res) => {
  try {
    const { lines, voiceA, voiceB } = req.body || {};
    if (!Array.isArray(lines) || !lines.length) return res.status(400).json({ error: 'No script lines provided.' });
    if (!voiceA || !voiceB) return res.status(400).json({ error: 'Pick a voice for each host.' });
    if (!(await ffmpegOk())) return res.status(500).json({ error: 'ffmpeg not found. Install it: brew install ffmpeg' });

    const tmp = path.join(os.tmpdir(), `podcast-${Date.now()}.mp3`);
    try { await buildEpisode(lines, voiceA, voiceB, tmp); }
    catch (err) { return res.status(502).json({ error: err.message }); }
    const buf = await readFile(tmp);
    await rm(tmp, { force: true });
    res.setHeader('Content-Type', 'audio/mpeg');
    res.setHeader('Content-Disposition', 'attachment; filename="podcast.mp3"');
    res.send(buf);
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

// ---------- background jobs (long episodes + multi-audio batches) ----------
//
// A job is a batch of "audios", each its own named episode with its own script
// and voice pair. Audios are processed one at a time through a single global
// worker (the Mac can only synthesize one line at a time anyway), finished mp3s
// are written to output/, and the UI polls /api/jobs/:id for progress. Because
// the slow work happens server-side and the result lands on disk, closing the
// browser tab no longer loses the episode.

const jobs = new Map();        // jobId -> { id, audios:[audio], createdAt }
const workQueue = [];          // audio objects waiting to be synthesized
let workerBusy = false;
let currentAudio = null;
let currentAbortCtrl = null;

// Permanent history. Every episode ever attempted is recorded here — including its
// script lines + voices, so it can be re-generated later without retyping anything,
// and so nothing ever disappears from the History list (it's read straight off disk).
const EPISODES_FILE = path.join(OUTPUT_DIR, 'episodes.json');
let episodes = [];
let episodesLoaded = false;
async function loadEpisodes() {
  if (episodesLoaded) return;
  try {
    const parsed = JSON.parse(await readFile(EPISODES_FILE, 'utf8'));
    episodes = Array.isArray(parsed) ? parsed : [];
  } catch { episodes = []; }
  episodesLoaded = true;
  // Recover items that were in-flight when the server last stopped.
  // 'running' → 'error' (synthesis was cut off mid-stream).
  // 'queued'  → re-add to workQueue so they continue automatically.
  let needsSave = false;
  for (const e of episodes) {
    if (e.status === 'running') {
      // Re-queue to retry from the start — handles laptop sleep/wake and server restarts.
      e.status = 'queued'; e.line = 0; e.error = null; needsSave = true;
      workQueue.push(e);
    } else if (e.status === 'queued') {
      workQueue.push(e);
    }
  }
  if (needsSave) await saveEpisodes();
  if (workQueue.length) pumpWorker();
}
async function saveEpisodes() {
  await mkdir(OUTPUT_DIR, { recursive: true });
  await writeFile(EPISODES_FILE, JSON.stringify(episodes, null, 2));
}

function slugify(name) {
  return String(name || 'episode').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'episode';
}

// Pick an output filename that doesn't collide with an existing one.
async function uniqueOutPath(base) {
  await mkdir(OUTPUT_DIR, { recursive: true });
  const existing = new Set(await readdir(OUTPUT_DIR).catch(() => []));
  let file = `${base}.mp3`, n = 2;
  while (existing.has(file)) file = `${base}-${n++}.mp3`;
  return { file, full: path.join(OUTPUT_DIR, file) };
}

// Build one audio/episode record. It doubles as the live progress object AND the
// persisted history entry — the same object reference is pushed into both the job
// and the episodes manifest, so worker updates flow to both.
function makeEpisode(a, i = 0) {
  return {
    id: randomUUID().slice(0, 8),
    name: String(a.name || `Episode ${i + 1}`).slice(0, 80),
    lines: Array.isArray(a.lines) ? a.lines : [],
    voiceA: a.voiceA, voiceB: a.voiceB,
    owner: a.owner || null,   // account id in hosted mode; null = legacy/local
    status: 'queued', line: 0, total: Array.isArray(a.lines) ? a.lines.length : 0,
    file: null, error: null, createdAt: Date.now(),
  };
}

function audioSummary(a) {
  return { id: a.id, name: a.name, status: a.status, line: a.line, total: a.total, file: a.file, error: a.error, createdAt: a.createdAt };
}

// Keep the Mac awake while synthesizing so an overnight batch doesn't stall when
// the system would normally idle-sleep. caffeinate is spawned by the server itself
// the moment work starts and killed the instant the queue drains — you never run it
// by hand. `-w <pid>` ties it to this process so it can never outlive the server
// (no orphaned process pinning the machine awake forever if the server dies).
let caffeine = null;
function keepAwake(on) {
  if (on) {
    if (caffeine) return;
    try {
      caffeine = spawn('caffeinate', ['-i', '-m', '-s', '-w', String(process.pid)], { stdio: 'ignore' });
      caffeine.on('error', () => { caffeine = null; });
      caffeine.on('exit', () => { caffeine = null; });
      console.log('[caffeinate] keep-awake ON (synthesizing)');
    } catch { caffeine = null; }
  } else if (caffeine) {
    try { caffeine.kill(); } catch {}
    caffeine = null;
    console.log('[caffeinate] keep-awake OFF (queue idle)');
  }
}

function pumpWorker() {
  if (workerBusy || !workQueue.length) return;
  workerBusy = true;
  keepAwake(true);
  (async () => {
    while (workQueue.length) {
      const audio = workQueue.shift();
      if (audio.status === 'cancelled') continue;
      audio.status = 'running';
      await saveEpisodes();
      const ctrl = new AbortController();
      currentAudio = audio;
      currentAbortCtrl = ctrl;
      try {
        const { file, full } = await uniqueOutPath(slugify(audio.name));
        await buildEpisode(audio.lines, audio.voiceA, audio.voiceB, full, (done, total) => { audio.line = done; audio.total = total; }, ctrl.signal);
        audio.file = file;
        audio.status = 'done';
      } catch (e) {
        if (audio.cancelled) {
          audio.status = 'cancelled';
          audio.error = 'Cancelled';
        } else {
          // Never permanently fail — re-queue automatically after a short backoff.
          console.log(`[worker] "${audio.name}" failed (${e.message || e}), re-queuing in 20s`);
          audio.status = 'queued';
          audio.line = 0;
          audio.error = null;
          setTimeout(() => { workQueue.push(audio); pumpWorker(); }, 20 * 1000);
        }
      } finally {
        currentAudio = null;
        currentAbortCtrl = null;
      }
      await saveEpisodes();
    }
    keepAwake(false);
    workerBusy = false;
  })();
}

// Start a batch. body: { audios: [{ name, lines:[{speaker,text}], voiceA, voiceB }] }
app.post('/api/jobs', async (req, res) => {
  try {
    const { audios } = req.body || {};
    if (!Array.isArray(audios) || !audios.length) return res.status(400).json({ error: 'No audios provided.' });
    if (!(await ffmpegOk())) return res.status(500).json({ error: 'ffmpeg not found. Install it: brew install ffmpeg' });
    await loadEpisodes();

    const items = audios.map((a, i) => makeEpisode({ ...a, owner: req.user?.id || null }, i));
    for (const it of items) {
      if (!it.lines.length) return res.status(400).json({ error: `"${it.name}" has no script lines.` });
      if (!it.voiceA || !it.voiceB) return res.status(400).json({ error: `"${it.name}" is missing a Host A or Host B voice.` });
    }

    // Shared-server fairness: one Mac synthesizes for everyone, so a non-admin
    // account can have at most 4 episodes waiting/running at a time.
    if (AUTH_ON && !isAdmin(req.user)) {
      const active = episodes.filter((e) => e.owner === req.user.id && (e.status === 'queued' || e.status === 'running')).length;
      if (active + items.length > 4) return res.status(429).json({ error: 'Queue limit: you can have at most 4 episodes generating at once. Let some finish first.' });
    }

    const id = randomUUID().slice(0, 8);
    jobs.set(id, { id, audios: items, owner: req.user?.id || null, createdAt: Date.now() });
    episodes.push(...items);      // same object refs -> worker updates flow to history
    await saveEpisodes();
    workQueue.push(...items);
    pumpWorker();
    res.json({ jobId: id, audios: items.map(audioSummary) });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.get('/api/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job || !canSee(req.user, job.owner)) return res.status(404).json({ error: 'No such job.' });
  const audios = job.audios.map(audioSummary);
  res.json({ jobId: job.id, done: audios.every((a) => a.status === 'done' || a.status === 'error'), audios });
});

// List finished episodes on disk (so they're retrievable across restarts/tabs).
// Hosted mode: only files belonging to the signed-in account's episodes; files
// no episode references (pre-account strays) are admin-only.
app.get('/api/outputs', async (req, res) => {
  await loadEpisodes();
  await mkdir(OUTPUT_DIR, { recursive: true });
  const names = (await readdir(OUTPUT_DIR).catch(() => [])).filter((f) => f.toLowerCase().endsWith('.mp3'));
  const referenced = new Map(); // file -> owner
  for (const e of episodes) if (e.file) referenced.set(e.file, e.owner || null);
  const outputs = [];
  for (const f of names) {
    const owner = referenced.has(f) ? referenced.get(f) : null;
    if (!canSee(req.user, owner)) continue;
    const s = await stat(path.join(OUTPUT_DIR, f)).catch(() => null);
    if (s) outputs.push({ file: f, size: s.size, mtime: s.mtimeMs });
  }
  outputs.sort((a, b) => b.mtime - a.mtime);
  res.json({ outputs });
});

// Permanent history: every episode ever attempted, newest first, each with its
// name, date, status, and whether the mp3 is still on disk. `regenerable` is true
// when we saved enough (script lines + both voices) to re-run it. mp3s found in
// output/ that predate the manifest are included too (download-only, no script).
app.get('/api/history', async (req, res) => {
  await loadEpisodes();
  await mkdir(OUTPUT_DIR, { recursive: true });
  const onDisk = new Set((await readdir(OUTPUT_DIR).catch(() => [])).filter((f) => f.toLowerCase().endsWith('.mp3')));
  const referenced = new Set();
  const history = [];
  for (const e of episodes) {
    if (e.file) referenced.add(e.file);
    if (!canSee(req.user, e.owner)) continue;
    let size = null;
    const present = !!(e.file && onDisk.has(e.file));
    if (present) { const s = await stat(path.join(OUTPUT_DIR, e.file)).catch(() => null); if (s) size = s.size; }
    history.push({
      id: e.id, name: e.name, file: e.file, present, size,
      status: e.status, error: e.error, createdAt: e.createdAt,
      lines: Array.isArray(e.lines) ? e.lines.length : 0,
      line: e.line || 0, total: e.total || 0,
      queuePosition: workQueue.indexOf(e),
      regenerable: Array.isArray(e.lines) && e.lines.length > 0 && !!e.voiceA && !!e.voiceB,
    });
  }
  // mp3s on disk that no episode references (pre-account strays): admin-only.
  if (canSee(req.user, null)) {
    for (const f of onDisk) {
      if (referenced.has(f) || f === path.basename(EPISODES_FILE)) continue;
      const s = await stat(path.join(OUTPUT_DIR, f)).catch(() => null);
      history.push({ id: null, name: f.replace(/\.mp3$/i, ''), file: f, present: true, size: s ? s.size : null, status: 'done', error: null, createdAt: s ? s.mtimeMs : 0, lines: 0, regenerable: false });
    }
  }
  history.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
  res.json({ history });
});

// Re-run a past (or failed) episode using its saved script + voices — no retyping.
// Produces a fresh file/history entry; the original is left untouched.
app.post('/api/regenerate', async (req, res) => {
  try {
    await loadEpisodes();
    const { id } = req.body || {};
    const src = episodes.find((e) => e.id === id);
    if (!src || !canSee(req.user, src.owner)) return res.status(404).json({ error: 'No such episode in history.' });
    if (!Array.isArray(src.lines) || !src.lines.length || !src.voiceA || !src.voiceB) {
      return res.status(400).json({ error: 'This episode has no saved script/voices to regenerate from.' });
    }
    const item = makeEpisode({ name: src.name, lines: src.lines, voiceA: src.voiceA, voiceB: src.voiceB, owner: req.user?.id || src.owner || null });
    const jid = randomUUID().slice(0, 8);
    jobs.set(jid, { id: jid, audios: [item], owner: item.owner, createdAt: Date.now() });
    episodes.push(item);
    await saveEpisodes();
    workQueue.push(item);
    pumpWorker();
    res.json({ jobId: jid, audios: [audioSummary(item)] });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

app.post('/api/prioritize', async (req, res) => {
  const { id, position = 0 } = req.body || {};
  const idx = workQueue.findIndex((a) => a.id === id);
  if (idx === -1 || !canSee(req.user, workQueue[idx].owner)) return res.status(404).json({ error: 'Not found in queue.' });
  if (idx === position) return res.json({ ok: true });
  const [item] = workQueue.splice(idx, 1);
  workQueue.splice(Math.max(0, position), 0, item);
  res.json({ ok: true });
});

app.patch('/api/episode/:id', async (req, res) => {
  const { name } = req.body || {};
  if (!name || !name.trim()) return res.status(400).json({ error: 'Name required' });
  await loadEpisodes();
  const ep = episodes.find(e => e.id === req.params.id);
  if (!ep || !canSee(req.user, ep.owner)) return res.status(404).json({ error: 'Episode not found' });
  ep.name = name.trim().slice(0, 100);
  await saveEpisodes();
  res.json({ ok: true, name: ep.name });
});

app.post('/api/cancel', async (req, res) => {
  const { id } = req.body || {};
  // 1. Still waiting in the live queue
  const qIdx = workQueue.findIndex((a) => a.id === id);
  if (qIdx !== -1) {
    const audio = workQueue[qIdx];
    if (!canSee(req.user, audio.owner)) return res.status(404).json({ error: 'Not found or already done.' });
    audio.cancelled = true;
    audio.status = 'cancelled';
    audio.error = 'Cancelled';
    workQueue.splice(qIdx, 1);
    saveEpisodes();
    return res.json({ ok: true });
  }
  // 2. Currently being synthesized
  if (currentAudio?.id === id && currentAbortCtrl) {
    if (!canSee(req.user, currentAudio.owner)) return res.status(404).json({ error: 'Not found or already done.' });
    currentAudio.cancelled = true;
    currentAbortCtrl.abort(new Error('Cancelled by user'));
    return res.json({ ok: true });
  }
  // 3. Stuck as queued/running in the history DB (e.g. after a server restart)
  await loadEpisodes();
  const ep = episodes.find((e) => e.id === id);
  if (ep && !canSee(req.user, ep.owner)) return res.status(404).json({ error: 'Not found or already done.' });
  if (ep && (ep.status === 'queued' || ep.status === 'running')) {
    ep.status = 'cancelled';
    ep.error = 'Cancelled';
    await saveEpisodes();
    return res.json({ ok: true });
  }
  res.status(404).json({ error: 'Not found or already done.' });
});

// Finished mp3s. Hosted mode: you can only download files that belong to your
// episodes (unreferenced pre-account files are admin-only). Only .mp3 is served
// so the episodes.json manifest (scripts + metadata) is never downloadable.
app.get('/output/:file', async (req, res) => {
  const f = path.basename(req.params.file);
  if (!f.toLowerCase().endsWith('.mp3')) return res.status(404).json({ error: 'Not found' });
  if (AUTH_ON) {
    await loadEpisodes();
    const ep = episodes.find((e) => e.file === f);
    if (!canSee(req.user, ep ? ep.owner || null : null)) return res.status(404).json({ error: 'Not found' });
  }
  res.sendFile(path.join(OUTPUT_DIR, f), (err) => {
    if (err && !res.headersSent) res.status(404).json({ error: 'Not found' });
  });
});

app.post('/api/extract-url', async (req, res) => {
  try {
    const { url } = req.body || {};
    if (!url) return res.status(400).json({ error: 'No URL provided.' });
    const r = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0 PodcastStudio' } });
    const html = await r.text();
    const text = html.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ')
      .replace(/<[^>]+>/g, ' ').replace(/&[a-z]+;/gi, ' ').replace(/\s+/g, ' ').trim();
    res.json({ text: text.slice(0, 40000) });
  } catch (e) { res.status(500).json({ error: String(e.message || e) }); }
});

const server = app.listen(PORT, () => console.log(`\n  Podcast Studio:  http://localhost:${PORT}\n  Local TTS expected at: ${LOCAL_TTS_URL}\n`));
// Node's default requestTimeout is 5 min; local synthesis of a long episode runs
// far longer, so without this Node would abort the socket mid-generation. Jobs are
// async now, but the legacy /api/podcast route and any direct call still need it off.
server.requestTimeout = 0;
server.headersTimeout = 0;
server.timeout = 0;
