# 🎙️ MCI Digital Speech Biomarker System

**Turn an everyday speech recording into a structured cognitive-wellness biomarker profile — entirely on your own machine.**

Cognitive decline leaves measurable footprints in speech: slower speech rate, longer and more frequent pauses, reduced lexical diversity, and pronoun substitution ("that thing" instead of a concrete noun). This project extracts those signals from a `.wav` recording with a dual-track pipeline — **acoustic** (openSMILE eGeMAPS + pause/speech-rate analysis) and **linguistic** (Whisper transcription + lexical-diversity metrics) — and tracks them over time against each person's own baseline.

> ⚠️ **Non-diagnostic.** Outputs are *Cognitive Wellness Trend Reports* meant to support — never replace — formal assessment (MMSE/MoCA, biomarker testing). This is a research/demo tool, not a medical device.

---

## ✨ What you get

- **Web UI** — drag & drop a recording, get a biomarker report with trend charts
- **CLI** — `mci-analyze recording.wav` → structured JSON profile
- **Track A · Acoustic** — 88 eGeMAPS features (F₀, jitter, shimmer, HNR, loudness…), speech/articulation rate, pause frequency & duration
- **Track B · Linguistic** — Whisper transcript, TTR / MATTR / MTLD lexical diversity, pronoun-to-noun ratio, filler-word ratio
- **Personal baseline** — per-user longitudinal history; flags metrics deviating beyond ±2σ
- **Privacy-first** — everything runs locally; uploaded audio is deleted right after analysis (zero-retention)

## 🚀 Quickstart

### Prerequisites

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) — or plain `pip` + `venv`

### Install

```bash
git clone https://github.com/Xin-Zheng/mci-speech-biomarker.git
cd mci-speech-biomarker

uv venv
uv pip install -e ".[acoustic,asr,dev]"   # openSMILE eGeMAPS + Whisper ASR
```

> 💡 In a hurry? `uv pip install -e .` installs the light core only. The pipeline still works — it falls back to signal-based acoustics and lets you paste a transcript instead of running Whisper. Every report records which backends actually ran.

### Option 1 — Web UI (easiest way to play)

```bash
uv run mci-serve --port 8000
```

Open **http://127.0.0.1:8000** in your browser, then:

1. Drag & drop (or select) a speech recording (`.wav` works best; mp3/m4a/flac/ogg accepted)
2. Pick a **User ID** — sessions under the same ID build that person's baseline
3. Optionally paste a known transcript (skips Whisper — handy offline)
4. Click **Analyze speech** → read the report, baseline flags, and trend charts; download the JSON if you like

*No recording handy?* Record 30–60 seconds describing a picture (the classic "Cookie Theft" picture-description task works great), or just talk about your day. The UI's **Try demo transcript** button fills in sample text so you can explore the linguistic half with any short clip.

### Option 2 — Command line

```bash
# Full pipeline: VAD → acoustics + Whisper → JSON profile
uv run mci-analyze recording.wav --user-id demo --pretty

# Skip Whisper with a known transcript
uv run mci-analyze recording.wav --transcript "The boy is taking cookies from the jar..." --pretty

# Save the session to the longitudinal store and write the profile to a file
uv run mci-analyze recording.wav --user-id demo --save --out profile.json

# Useful flags: --whisper-model small|medium  --language yue (Cantonese mode)
#               --asr-backend qwen3|sensevoice|whisper
#               --diarization energy|pyannote  --acoustic-backend opensmile|signal
```

## 📊 Reading the report

| Metric | What it means | Direction of interest |
|---|---|---|
| Speech / articulation rate | Syllables per second (with / without pauses) | Lower = slower speech |
| Mean pause duration, pauses/min | Silent gaps ≥ 0.25 s between phrases | Higher = more/longer pauses |
| F₀, jitter, shimmer, HNR | Pitch and voice-quality measures (eGeMAPS) | Change vs own baseline |
| TTR / MATTR / MTLD | Lexical diversity of the transcript | Lower = narrower vocabulary |
| Pronoun-to-noun ratio | "it / that / thing" vs concrete nouns | Higher = more referential vagueness |
| Filler ratio | *um, uh, er…* per token | Higher = more disfluency |
| z-score / ⚠ flag | Deviation from the user's personal baseline | \|z\| ≥ 2σ gets flagged |

A sample output lives in [`examples/sample_profile.json`](examples/sample_profile.json). Baseline flags activate after ≥5 saved sessions — the spec recommends 30–90 days of regular recordings for a stable personal baseline.

## 🏗️ How it works

```
.wav ──▶ Layer 1 Ingestion ──▶ Layer 2 Dual-Track Extraction ──▶ Layer 3 Baseline ──▶ Layer 4 Report
              │                     ├─ Track A: acoustic            z-scores vs           JSON + Web UI
              │                     │   eGeMAPS, pauses, rate       personal >2σ flags
         VAD + diarization          └─ Track B: linguistic
         (pyannote*)                    Whisper*, TTR/MATTR/MTLD,
                                        pronoun:noun, fillers
```

`*` optional heavy backends with graceful fallbacks (energy VAD, signal acoustics). See [`SPEC.md`](SPEC.md) for the full design.

```
src/mci_biomarker/
├── schemas.py            # Biomarker profile — the JSON contract
├── pipeline.py           # wav → profile orchestrator
├── cli.py                # `mci-analyze`
├── audio/                # ingest (16 kHz mono) · VAD · diarization · eGeMAPS + pauses
├── linguistic/           # ASR chain (Qwen3/SenseVoice/Whisper) · EN + 粵語 metrics
├── baseline/             # JSONL history + z-score anomaly flags
└── web/                  # FastAPI app (`mci-serve`) + single-page UI
```

## 🇭🇰 Cantonese mode 粵語模式 (`--language yue`)

The acoustic track (eGeMAPS, pauses, speech rate) is language-independent and
works for Cantonese out of the box. Cantonese mode switches the speech-to-text
and linguistic tracks to Cantonese-native engines and metrics.

### Step 1 — Install the Cantonese extras

```bash
uv pip install -e ".[cantonese]"     # PyCantonese word segmentation (small)
uv pip install -e ".[asr-qwen]"      # ★ recommended ASR: Qwen3-ASR (Apache-2.0)
# Alternatives:
uv pip install -e ".[asr-sensevoice]"  # SenseVoice — lighter/faster, also native 粵語
uv pip install -e ".[asr]"             # faster-whisper only (see caveat below)
```

Install just **one** ASR extra. The first Cantonese run downloads its model
from Hugging Face (~1–4 GB, one time, then fully offline). On a modern laptop
CPU, Qwen3-ASR-0.6B transcribes ~1 minute of audio in ~15 seconds.

### Step 2a — Use it in the Web UI

1. Start the server: `uv run mci-serve --port 8000` → open http://127.0.0.1:8000
2. Set **Language 語言** to **「粵語 (Cantonese)」**
3. Drag & drop the Cantonese recording, set a User ID, click **Analyze speech**
4. The report shows `language: yue` and which ASR engine ran (`qwen3-asr`, …)

No recording yet? Click **Try demo transcript** with 粵語 selected — it fills a
Cantonese picture-description sample so you can explore the metrics with any clip.

### Step 2b — Use it from the CLI

```bash
# Audio only — Qwen3-ASR transcribes, full biomarker profile as JSON
uv run mci-analyze recording.wav --language yue --user-id demo --pretty

# You already have a transcript (skips ASR entirely — works with no ASR extra)
uv run mci-analyze recording.wav --language yue \
  --transcript "個媽媽喺廚房度洗碗，個男仔企喺張凳上面攞曲奇..." --pretty

# Save into the user's longitudinal baseline; force a specific ASR engine
uv run mci-analyze recording.wav --language yue --asr-backend qwen3 --save
```

### How it works & how to read Cantonese metrics

- **ASR chain (auto):** [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) →
  [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) → faster-whisper
  with the `yue` token (large-v3 generation only; English-only `*.en` models
  are auto-switched). Each engine fails soft to the next; if none is
  installed you still get the full acoustic report with a warning.
- **Why not Whisper alone?** Whisper often rewrites spoken Cantonese into
  standard written Chinese （佢→他、嘅→的、嘢→東西）, erasing exactly the
  colloquial pronouns and vague words our biomarkers measure. Qwen3-ASR and
  SenseVoice keep the spoken forms.
- **Metrics** run on segmented Cantonese words (PyCantonese; built-in
  max-match fallback): pronouns 我/你/佢/我哋 vs concrete nouns
  (pronoun:noun ↑ = more 「嗰個/嗰樣嘢」-style substitution), vague words
  嘢/嗰樣嘢， fillers 嗯/誒/即係， plus TTR/MATTR/MTLD. English code-switching
  (「我個friend…」) is kept as tokens. Simplified-character ASR output
  （个/系/时） is auto-normalized to traditional before scoring.
- **Baselines never mix languages** — English and Cantonese sessions for the
  same user form separate baselines.
- **Recording tips:** 1–3 minutes in a quiet room; describing a picture or
  recounting your day both work. Cantonese has 6 lexical tones, so read
  F₀ metrics against the speaker's own Cantonese baseline, not English norms.
- **Try real data:** the [HKCanCor corpus](https://github.com/fcbond/hkcancor)
  (CC BY) ships sample Cantonese recordings with gold transcripts — great for
  smoke-testing. We validated against its monologue sample: Qwen3-ASR
  transcript metrics (tokens, TTR, MATTR) closely matched gold-transcript
  metrics.

## 👥 Multiple people & patients

### Tell voices apart in one recording

Real assessments often include family members or clinicians. Multi-speaker
mode diarizes the recording — **who spoke when** — and produces **one
biomarker profile per person**, each saved under that patient.

- **Web UI:** choose *People in recording → Multiple people*, upload, click
  **🔍 Detect speakers**, assign each voice (duration + pitch shown, with a
  color timeline) to an existing or new patient, then **Analyze all speakers**.
- **Voiceprint auto-matching:** every saved analysis enrolls/refreshes the
  patient's voiceprint (a 41-dim MFCC+pitch centroid in
  `data/voiceprints.json` — derived features, not audio). On the next
  multi-speaker upload, detected voices are compared against enrolled
  patients; confident matches (conservative cosine threshold + margin) are
  **pre-selected automatically** with a "✓ Auto-matched voice · 98% similar"
  badge — you can always override. The CLI applies the same auto-assignment
  when no `--speaker-map` entry is given.
- **CLI:**
  ```bash
  uv run mci-analyze dialogue.wav --speakers all \
    --speaker-map "SPEAKER_00=chan-tai,SPEAKER_01=lee-mui" --save --pretty
  ```
  Unassigned voices get ids like `recording-speaker-01`.

Diarization backends (`--diarization auto`, tried in order):

1. **pyannote** (most accurate) — `uv pip install -e ".[diarize]"`, set
   `HF_TOKEN` after accepting the model terms on Hugging Face.
2. **cluster** (built-in fallback, no downloads) — MFCC + pitch voice
   embeddings with agglomerative clustering. Separates distinct voices
   (different pitch/timbre) well; very similar voices may merge — use
   pyannote for those.
3. **energy** — single-speaker assumption.

> ⚠️ In conversations, per-speaker *pause* metrics include turn-taking gaps
> while others were speaking — the report flags this; interpret accordingly.

### Patient reports

The **Patients 患者** tab lists everyone with session counts, last-session
date, and baseline status. Click a patient for their page: latest full
report, longitudinal trend charts, a session-by-session table (with
per-session JSON download), and editable display name/notes. Anyone with
saved sessions appears automatically; names/notes live in
`data/patients.json` alongside the git-ignored profile store.

## 🔒 Privacy

- **Local-first:** no audio or transcript ever leaves your machine (model downloads from Hugging Face on first use are the only network traffic).
- **Zero-retention audio:** web uploads are processed from a temp file and deleted immediately; only JSON profiles persist under `data/profiles/` (git-ignored). Voiceprints (`data/voiceprints.json`, also git-ignored) are 41-number acoustic averages used for patient auto-matching — speech cannot be reconstructed from them, but treat them as biometric data and get consent.
- **Low-confidence gating:** if Whisper confidence drops (e.g. strong dialect/accent), linguistic metrics are flagged and the report leans on acoustic markers instead.

## 🧪 Tests

```bash
uv run pytest -q
uv run ruff check src tests
```

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Whisper unavailable` warning, empty transcript | Install ASR extra: `uv pip install -e ".[asr]"` — or paste a transcript |
| `openSMILE unavailable`, `signal-fallback` backend | Install acoustics: `uv pip install -e ".[acoustic]"` |
| `pyannote … HF_TOKEN not set` | Expected without diarization setup; energy VAD is used instead |
| Empty transcript on a real recording | Try a larger model: `--whisper-model small` (and `--language` if not English) |
| Cantonese transcript empty or in 書面語 | Use `--language yue` with `.[asr-qwen]` installed; Whisper-only setups need large-v3+ |
| Port 8000 already in use | `uv run mci-serve --port 8001` |

## 🗺️ Roadmap — future improvements & candidate libraries

Organized by pipeline layer. Everything listed is open-source unless marked
💲 (paid API). **Now** = highest value/effort ratio; **Later** = research-grade.

### 🎙️ Capture, VAD & denoising

| Candidate | What it unlocks | When |
|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | Neural VAD — far fewer false alarms on TV/kitchen noise than energy VAD | Now |
| [noisereduce](https://github.com/timsainb/noisereduce) / RNNoise | Home-noise suppression before feature extraction | Next |
| Streaming capture (FunASR streaming / [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)) | Analyze while recording; on-device edge deployment | Later |

### 🗣️ Speaker diarization & identity

| Candidate | What it unlocks | When |
|---|---|---|
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) *(already wired, optional)* | Best-in-class diarization incl. overlapping speech | Now |
| [SpeechBrain](https://speechbrain.github.io/) ECAPA-TDNN embeddings | Replace MFCC clustering with neural voiceprints | Next |
| [3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker) (via FunASR) | Strong Chinese-speaker embeddings + diarization pipeline | Next |
| Neural voiceprint enrollment (ECAPA/CAM++ upgrade of current MFCC voiceprints) | More robust auto-recognition across microphones/rooms | Next |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) diarization | GPU-scale diarization for long recordings | Later |

### ✍️ ASR / transcription

| Candidate | What it unlocks | When |
|---|---|---|
| [Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ASR) | Word-level timestamps → precise pause/articulation alignment | Now |
| Qwen3-ASR-1.7B (vs current 0.6B) | Better Cantonese accuracy on elderly/noisy speech | Now |
| Fine-tuned Cantonese Whisper (e.g. [khleeloo/whisper-large-v3-cantonese](https://huggingface.co/khleeloo/whisper-large-v3-cantonese), CT2-converted) | Colloquial-faithful transcripts inside the faster-whisper path | Next |
| Cloud ASR 💲 (Azure `zh-HK`, Google `yue-Hant-HK`, Deepgram, Tencent Cloud) | Highest Cantonese accuracy — **breaks local-only privacy; needs consent + PDPO review** | Later |

### 🔊 Acoustic biomarkers (Track A)

| Candidate | What it unlocks | When |
|---|---|---|
| [praat-parselmouth](https://github.com/YannickJadoul/Parselmouth) *(in extras, not yet wired)* | Praat-grade jitter/shimmer/formants, precise syllable nuclei | Now |
| [DisVoice](https://github.com/jcvasquezc/DisVoice) | Feature sets designed for neurological speech disorders | Next |
| Self-supervised embeddings (wav2vec 2.0 / HuBERT via [Transformers](https://huggingface.co/docs/transformers)) | Learned paralinguistic representations beyond eGeMAPS | Later |
| Jyutping tone analysis (forced alignment + [PyCantonese](https://pycantonese.org)) | Cantonese-specific tone-production accuracy as a biomarker | Later |

### 📝 Linguistic biomarkers (Track B)

| Candidate | What it unlocks | When |
|---|---|---|
| PyCantonese POS tagging *(already an optional dep)* | Real noun/pronoun counts instead of lexicon exclusion | Now |
| Dependency parsing ([spaCy](https://spacy.io/) zh / [Stanza](https://stanfordnlp.github.io/stanza/)) | SPEC §2 syntactic complexity: parse depth, clause density | Next |
| [sentence-transformers](https://www.sbert.net/) embeddings | Semantic coherence, topic drift, repeated-question detection | Next |
| Local LLM scoring ([Ollama](https://ollama.com/) / llama.cpp) | Idea-density & coherence ratings — still fully offline | Later |

### 📈 Baseline & anomaly detection (Layer 3)

| Candidate | What it unlocks | When |
|---|---|---|
| EWMA control charts (config stub exists) | Earlier, smoother trend alarms than raw z-scores | Now |
| Isolation Forest ([scikit-learn](https://scikit-learn.org/), already a dep) | Multivariate anomaly scoring across all biomarkers jointly | Next |
| [ruptures](https://centre-borelli.github.io/ruptures-docs/) change-point detection | "Something changed around March" step-change alerts | Next |
| Validation on [ADReSS](https://dementia.talkbank.org/) / Pitt / TAUKADIAL corpora | External benchmarks for sensitivity/specificity claims | Later |

### 🖥️ Product & platform

| Candidate | What it unlocks | When |
|---|---|---|
| PDF reports ([WeasyPrint](https://weasyprint.org/)) | Printable trend report for clinic visits (SPEC Layer 4) | Now |
| GitHub Actions CI | Auto-run `ruff` + `pytest` on every push | Now |
| Docker image | One-command deploy without local Python setup | Next |
| Auth + clinician roles (FastAPI + OAuth2) | Multi-clinician use; patient data stays access-controlled | Next |
| Encryption at rest (SQLCipher / age) | Protect stored profiles beyond git-ignored plaintext JSON | Next |
| FHIR export | Interoperability with hospital EHR systems | Later |
| Desktop/mobile packaging (Tauri / sherpa-onnx) | Fully offline bedside/edge capture without a server | Later |

## 📄 License

[MIT](LICENSE) — research and educational use. Not for clinical diagnosis.
