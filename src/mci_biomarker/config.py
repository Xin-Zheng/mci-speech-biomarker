"""Pipeline configuration. All thresholds in one place (SPEC §2-4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "data" / "profiles"


@dataclass
class AudioConfig:
    target_sr: int = 16000
    # Energy-VAD framing
    frame_ms: int = 30
    min_speech_ms: int = 200
    min_silence_ms: int = 250  # gaps shorter than this merge into speech
    # Pauses: silent gaps >= pause_threshold_s count as pauses (SPEC Track A)
    pause_threshold_s: float = 0.25
    long_pause_threshold_s: float = 1.0
    # Cluster diarization fallback
    max_speakers: int = 4
    cluster_threshold: float = 0.08  # agglomerative cosine distance threshold
    # Voiceprint auto-matching: cosine distance to an enrolled patient centroid
    # at or below this counts as the same person. Calibrated on HKCanCor:
    # same speaker ≈0.00-0.01, different speakers same mic ≈0.10, cross-file
    # >0.45 — conservative on purpose (a missed match just means manual pick).
    voiceprint_match_threshold: float = 0.05
    voiceprint_match_margin: float = 0.02  # best must beat 2nd-best patient by this


@dataclass
class WhisperConfig:
    model: str = "tiny.en"  # local, fast; use "small"/"medium" for dialects
    device: str = "cpu"
    compute_type: str = "int8"
    # Below this mean segment confidence, linguistic metrics are down-weighted (SPEC §4)
    low_confidence_threshold: float = 0.55
    # ASR routing: "auto" (Cantonese: Qwen3 → SenseVoice → Whisper-yue; else Whisper),
    # or force one of "qwen3" | "sensevoice" | "whisper".
    asr_backend: str = "auto"
    qwen_model: str = "Qwen/Qwen3-ASR-0.6B"
    sensevoice_model: str = "iic/SenseVoiceSmall"
    # English-only *.en Whisper checkpoints cannot do Cantonese; yue needs large-v3+
    yue_whisper_model: str = "large-v3-turbo"


@dataclass
class BaselineConfig:
    profile_dir: Path = field(default_factory=lambda: DEFAULT_PROFILE_DIR)
    min_baseline_sessions: int = 5  # need a few sessions before z-scores are meaningful
    z_alert_threshold: float = 2.0  # SPEC Layer 3: >2σ persistent deviation
    ewma_alpha: float = 0.3


@dataclass
class PipelineConfig:
    # Session language: "en" (default) or "yue" for Cantonese mode (also "zh").
    # Baselines are only ever compared within the same language.
    language: str = "en"
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    # "auto" tries pyannote then falls back to energy VAD; "energy" never loads pyannote
    diarization: str = "auto"
    # "auto" tries openSMILE then falls back to signal features
    acoustic_backend: str = "auto"
    keep_audio: bool = False  # SPEC §4 zero-retention: default False
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)
