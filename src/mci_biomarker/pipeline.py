"""End-to-end pipeline: .wav → BiomarkerProfile (SPEC Layers 1–3).

    wav → ingest (16k mono) → VAD/diarize → Track A acoustic ─┐
                                              Track B linguistic ┴→ baseline compare → profile
"""

from __future__ import annotations

import time
from pathlib import Path

from .audio.acoustic import extract_acoustic
from .audio.ingest import load_wav
from .audio.vad import extract_speech, run_vad
from .baseline.store import ProfileStore, compare_to_baseline
from .config import PipelineConfig
from .linguistic.features import extract_linguistic
from .linguistic.transcribe import transcribe
from .schemas import BiomarkerProfile, Provenance


def analyze_wav(
    wav_path: str | Path,
    user_id: str = "anonymous",
    cfg: PipelineConfig | None = None,
    transcript_override: str | None = None,
    language: str | None = None,
    save: bool = False,
    compare_baseline: bool = True,
) -> BiomarkerProfile:
    """Run the full dual-track pipeline on one recording."""
    cfg = cfg or PipelineConfig()
    lang = (language or cfg.language or "en").lower()
    t0 = time.perf_counter()
    warnings: list[str] = []
    wav_path = Path(wav_path)

    signal, sr = load_wav(wav_path, cfg.audio.target_sr)
    vad = run_vad(signal, sr, cfg.audio, cfg.diarization, cfg.hf_token, warnings)
    speech, _spans = extract_speech(signal, sr, vad)
    if speech.size == 0:
        warnings.append("No target-speaker speech detected in this recording.")

    acoustic = extract_acoustic(speech, sr, vad, cfg.audio, cfg.acoustic_backend, warnings)

    if transcript_override is not None:
        linguistic = extract_linguistic(
            transcript_override,
            segments=[],
            mean_confidence=None,
            backend="manual",
            speech_duration_s=vad.speech_duration_s,
            language=lang,
        )
    else:
        text, segments, mean_conf, asr_backend = transcribe(
            speech, sr, cfg.whisper, language=lang, warnings=warnings
        )
        linguistic = extract_linguistic(
            text,
            segments=segments,
            mean_confidence=mean_conf,
            backend=asr_backend,
            speech_duration_s=vad.speech_duration_s,
            low_confidence_threshold=cfg.whisper.low_confidence_threshold,
            language=lang,
        )
    if linguistic.low_confidence:
        warnings.append(
            "Low ASR confidence — linguistic metrics may be unreliable for this "
            "dialect/accent; weight acoustic markers instead (SPEC §4)."
        )

    profile = BiomarkerProfile(
        user_id=user_id,
        source_filename=wav_path.name,
        vad=vad,
        acoustic=acoustic,
        linguistic=linguistic,
        provenance=Provenance(
            vad_backend=vad.backend,
            acoustic_backend=acoustic.backend,
            asr_backend=linguistic.backend,
            warnings=warnings,
            audio_retained=cfg.keep_audio,
            processing_time_s=round(time.perf_counter() - t0, 3),
        ),
    )

    store = ProfileStore(cfg.baseline.profile_dir)
    if compare_baseline:
        priors = store.load_profiles(user_id)
        profile.baseline = compare_to_baseline(
            profile,
            priors,
            min_sessions=cfg.baseline.min_baseline_sessions,
            z_threshold=cfg.baseline.z_alert_threshold,
        )
    if save:
        store.save_profile(profile)
    return profile
