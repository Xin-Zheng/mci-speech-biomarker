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

# Useful flags: --whisper-model small|medium  --language zh
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
├── audio/                # ingest (16 kHz mono) · VAD/diarization · eGeMAPS + pauses
├── linguistic/           # Whisper ASR · lexical diversity & pronoun ratio
├── baseline/             # JSONL history + z-score anomaly flags
└── web/                  # FastAPI app (`mci-serve`) + single-page UI
```

### Optional: speaker diarization (pyannote)

To isolate one speaker in multi-speaker home audio:

```bash
uv pip install -e ".[diarize]"
export HF_TOKEN=<your-huggingface-token>   # accept the pyannote model terms on HF first
uv run mci-analyze recording.wav --diarization pyannote
```

## 🔒 Privacy

- **Local-first:** no audio or transcript ever leaves your machine (model downloads from Hugging Face on first use are the only network traffic).
- **Zero-retention audio:** web uploads are processed from a temp file and deleted immediately; only JSON profiles persist under `data/profiles/` (git-ignored).
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
| Port 8000 already in use | `uv run mci-serve --port 8001` |

## 🗺️ Roadmap

- Syntactic complexity (dependency-parse depth, clause density)
- Semantic coherence / topic-drift & repeated-question detection
- PDF export of the trend report
- EWMA / Isolation-Forest anomaly scoring alongside z-scores

## 📄 License

[MIT](LICENSE) — research and educational use. Not for clinical diagnosis.
