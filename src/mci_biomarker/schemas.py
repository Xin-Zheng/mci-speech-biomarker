"""Biomarker profile schemas — the structured JSON contract (SPEC Layer 4).

Every analysis run serializes to a `BiomarkerProfile`. Fields are Optional where a
backend may be unavailable; `provenance` always records what actually ran so a
clinician never mistakes a fallback estimate for a full-fidelity measurement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SpeechSegment(BaseModel):
    start_s: float
    end_s: float
    speaker: str = "SPEAKER_00"

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class VADResult(BaseModel):
    backend: str  # "pyannote" | "energy"
    segments: list[SpeechSegment] = Field(default_factory=list)
    n_speakers: int = 1
    target_speaker: str = "SPEAKER_00"
    total_duration_s: float = 0.0
    speech_duration_s: float = 0.0
    speech_ratio: float = 0.0


class AcousticFeatures(BaseModel):
    """Track A (SPEC §2): eGeMAPS functionals + pause / speech-rate metrics."""

    backend: str  # "opensmile-egemaps" | "signal-fallback"
    # Speech timing
    speech_rate_syll_per_s: float | None = None  # incl. pauses
    articulation_rate_syll_per_s: float | None = None  # speech only
    mean_pause_duration_s: float | None = None
    pause_count: int = 0
    pause_rate_per_min: float | None = None
    long_pause_count: int = 0
    long_pause_rate_per_min: float | None = None
    pause_durations_s: list[float] = Field(default_factory=list)
    # Voice quality (from eGeMAPS functionals when available, else signal estimates)
    f0_mean_hz: float | None = None
    f0_std_hz: float | None = None
    jitter_local: float | None = None
    shimmer_local_db: float | None = None
    hnr_db: float | None = None
    loudness_mean: float | None = None
    # Full eGeMAPS functional vector (88 features) when openSMILE ran
    egemaps: dict[str, float] | None = None


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    confidence: float | None = None  # exp(avg_logprob) in [0, 1]


class LinguisticFeatures(BaseModel):
    """Track B (SPEC §2): lexical diversity, referential vagueness, disfluency."""

    backend: str  # "qwen3-asr" | "sensevoice" | "faster-whisper" | "manual" | "none"
    language: str = "en"  # "en" | "yue" (Cantonese mode) | "zh" | ...
    transcript: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    mean_confidence: float | None = None
    low_confidence: bool = False  # SPEC §4: down-weight linguistics, trust acoustics
    token_count: int = 0
    type_count: int = 0
    # Lexical diversity
    ttr: float | None = None  # type-token ratio
    mattr: float | None = None  # moving-average TTR (window=50)
    mtld: float | None = None  # measure of textual lexical diversity
    # Referential vagueness / nominalization
    pronoun_count: int = 0
    noun_count: int = 0
    pronoun_to_noun_ratio: float | None = None
    vague_word_count: int = 0  # "thing", "stuff", "that", ...
    vague_word_ratio: float | None = None
    # Disfluency / structure
    filler_count: int = 0  # um, uh, er, ...
    filler_ratio: float | None = None
    repeated_word_rate: float | None = None  # immediate repetitions per token
    sentence_count: int = 0
    mean_sentence_length: float | None = None
    words_per_minute: float | None = None


class BaselineComparison(BaseModel):
    """Layer 3: this session vs the user's personal baseline."""

    n_prior_sessions: int = 0
    baseline_ready: bool = False
    z_scores: dict[str, float] = Field(default_factory=dict)
    flagged: list[str] = Field(default_factory=list)  # metrics beyond ±2σ
    summary: str = ""


class Provenance(BaseModel):
    vad_backend: str
    acoustic_backend: str
    asr_backend: str
    warnings: list[str] = Field(default_factory=list)
    audio_retained: bool = False
    processing_time_s: float = 0.0


class BiomarkerProfile(BaseModel):
    """One session's structured biomarker profile (the JSON output)."""

    profile_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    user_id: str = "anonymous"
    created_at: str = Field(default_factory=utc_now_iso)
    source_filename: str = ""
    disclaimer: str = (
        "Cognitive Wellness Trend Report — non-diagnostic. Supports, does not replace, "
        "formal assessment (MMSE/MoCA, biomarker testing)."
    )
    vad: VADResult
    acoustic: AcousticFeatures
    linguistic: LinguisticFeatures
    baseline: BaselineComparison = Field(default_factory=BaselineComparison)
    provenance: Provenance

    def headline_metrics(self) -> dict[str, Any]:
        """Flat dict of the key longitudinal metrics (used for baselines/charts)."""
        a, lx = self.acoustic, self.linguistic
        return {
            "speech_rate_syll_per_s": a.speech_rate_syll_per_s,
            "articulation_rate_syll_per_s": a.articulation_rate_syll_per_s,
            "mean_pause_duration_s": a.mean_pause_duration_s,
            "pause_rate_per_min": a.pause_rate_per_min,
            "f0_mean_hz": a.f0_mean_hz,
            "ttr": lx.ttr,
            "mattr": lx.mattr,
            "mtld": lx.mtld,
            "pronoun_to_noun_ratio": lx.pronoun_to_noun_ratio,
            "filler_ratio": lx.filler_ratio,
            "words_per_minute": lx.words_per_minute,
        }
