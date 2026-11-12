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


@dataclass
class WhisperConfig:
    model: str = "tiny.en"  # local, fast; use "small"/"medium" for dialects
    device: str = "cpu"
    compute_type: str = "int8"
    # Below this mean segment confidence, linguistic metrics are down-weighted (SPEC §4)
    low_confidence_threshold: float = 0.55


@dataclass
class BaselineConfig:
    profile_dir: Path = field(default_factory=lambda: DEFAULT_PROFILE_DIR)
    min_baseline_sessions: int = 5  # need a few sessions before z-scores are meaningful
    z_alert_threshold: float = 2.0  # SPEC Layer 3: >2σ persistent deviation
    ewma_alpha: float = 0.3


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    # "auto" tries pyannote then falls back to energy VAD; "energy" never loads pyannote
    diarization: str = "auto"
    # "auto" tries openSMILE then falls back to signal features
    acoustic_backend: str = "auto"
    keep_audio: bool = False  # SPEC §4 zero-retention: default False
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)
