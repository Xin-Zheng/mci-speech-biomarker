"""End-to-end pipeline: .wav → BiomarkerProfile(s) (SPEC Layers 1–3).

    wav → ingest (16k mono) → VAD/diarize → Track A acoustic ─┐
                                              Track B linguistic ┴→ baseline → profile

`analyze_wav` profiles the dominant speaker (classic single-patient flow).
`analyze_wav_speakers` diarizes everyone in the recording and returns one
profile per speaker, mapped to patients via `speaker_map`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .audio.acoustic import extract_acoustic
from .audio.diarize import annotate_matches, diarize, speaker_embedding, vad_for_speaker
from .audio.ingest import load_wav
from .audio.vad import extract_speech, run_vad
from .baseline.store import ProfileStore, VoiceprintStore, compare_to_baseline
from .config import PipelineConfig
from .linguistic.features import extract_linguistic
from .linguistic.transcribe import transcribe
from .schemas import BiomarkerProfile, DiarizationResult, Provenance, VADResult

# progress_cb(percent_0_to_100, stage_label) — called at stage boundaries.
ProgressCb = Callable[[int, str], None]


def _emit(cb: ProgressCb | None, pct: float, stage: str) -> None:
    if cb is not None:
        cb(max(0, min(100, round(pct))), stage)


def analyze_wav(
    wav_path: str | Path,
    user_id: str = "anonymous",
    cfg: PipelineConfig | None = None,
    transcript_override: str | None = None,
    language: str | None = None,
    save: bool = False,
    compare_baseline: bool = True,
    progress_cb: ProgressCb | None = None,
) -> BiomarkerProfile:
    """Run the full dual-track pipeline on one recording (dominant speaker)."""
    cfg = cfg or PipelineConfig()
    lang = (language or cfg.language or "en").lower()
    t0 = time.perf_counter()
    warnings: list[str] = []
    wav_path = Path(wav_path)

    _emit(progress_cb, 2, "Loading audio…")
    signal, sr = load_wav(wav_path, cfg.audio.target_sr)
    _emit(progress_cb, 8, "Detecting speech (VAD)…")
    if cfg.diarization == "cluster":
        diar = diarize(signal, sr, cfg.audio, "cluster", cfg.hf_token, warnings)
        dominant = diar.speakers[0].speaker if diar.speakers else "SPEAKER_00"
        vad = vad_for_speaker(diar, dominant)
    else:
        vad = run_vad(signal, sr, cfg.audio, cfg.diarization, cfg.hf_token, warnings)
    speech, _spans = extract_speech(signal, sr, vad)
    profile = _build_profile(
        speech, sr, vad, user_id, wav_path.name, cfg, lang,
        transcript_override, warnings, t0,
        speaker=vad.target_speaker, n_speakers=vad.n_speakers,
        progress_cb=progress_cb, p0=12, p1=95,
    )
    _emit(progress_cb, 97, "Comparing to personal baseline…")
    _attach_baseline(profile, cfg, compare_baseline, save)
    if save:
        _enroll_voiceprint(cfg, user_id, signal, sr, vad.segments)
    _emit(progress_cb, 100, "Done")
    return profile


def _enroll_voiceprint(cfg, patient_id, signal, sr, segments) -> None:
    """Store/refresh this patient's voiceprint centroid (best-effort)."""
    try:
        emb = speaker_embedding(signal, sr, segments)
        if emb is not None:
            VoiceprintStore(cfg.baseline.profile_dir).enroll(patient_id, emb)
    except Exception:
        pass  # voiceprint is an aid, never break an analysis over it


def analyze_wav_speakers(
    wav_path: str | Path,
    base_user_id: str = "anonymous",
    cfg: PipelineConfig | None = None,
    language: str | None = None,
    speaker_map: dict[str, str] | None = None,
    transcripts: dict[str, str] | None = None,
    save: bool = False,
    compare_baseline: bool = True,
    diarization_result: DiarizationResult | None = None,
    progress_cb: ProgressCb | None = None,
) -> tuple[DiarizationResult, list[BiomarkerProfile]]:
    """Diarize a recording and profile every detected speaker.

    `speaker_map` assigns diarization labels to patient ids
    (``{"SPEAKER_00": "chan-tai"}``); unassigned speakers get
    ``<base_user_id>-speaker-NN`` patient ids.
    """
    cfg = cfg or PipelineConfig()
    lang = (language or cfg.language or "en").lower()
    t0 = time.perf_counter()
    warnings: list[str] = []
    wav_path = Path(wav_path)
    speaker_map = speaker_map or {}
    transcripts = transcripts or {}

    _emit(progress_cb, 2, "Loading audio…")
    signal, sr = load_wav(wav_path, cfg.audio.target_sr)
    if diarization_result is None:
        _emit(progress_cb, 6, "Detecting speakers (diarization)…")
    diar = diarization_result or diarize(
        signal, sr, cfg.audio, cfg.diarization, cfg.hf_token, warnings
    )
    _emit(progress_cb, 14, "Matching voices to known patients…")
    diar = annotate_voiceprints(signal, sr, diar, cfg)
    if len(diar.speakers) > 1:
        warnings.append(
            "Multi-speaker recording: pause metrics per speaker include "
            "turn-taking gaps while others were talking — interpret with care."
        )
    n_spk = max(1, len(diar.speakers))
    profiles: list[BiomarkerProfile] = []
    for i, summary in enumerate(diar.speakers):
        speaker = summary.speaker
        patient_id = (
            speaker_map.get(speaker)
            or summary.matched_patient_id
            or _default_patient_id(base_user_id, speaker)
        )
        vad = vad_for_speaker(diar, speaker)
        speech, _spans = extract_speech(signal, sr, vad)
        p0 = 18 + i * (77 / n_spk)
        prefix = f"Speaker {i + 1}/{n_spk} ({patient_id}) · "
        profile = _build_profile(
            speech, sr, vad, patient_id, wav_path.name, cfg, lang,
            transcripts.get(speaker), list(warnings), t0,
            speaker=speaker, n_speakers=len(diar.speakers),
            progress_cb=progress_cb, p0=p0, p1=p0 + 77 / n_spk, prefix=prefix,
        )
        _attach_baseline(profile, cfg, compare_baseline, save)
        if save:
            _enroll_voiceprint(cfg, patient_id, signal, sr, vad.segments)
        profiles.append(profile)
    _emit(progress_cb, 100, "Done")
    return diar, profiles


def annotate_voiceprints(signal, sr, diar: DiarizationResult, cfg: PipelineConfig):
    """Auto-match detected speakers against enrolled patient voiceprints."""
    try:
        centroids = VoiceprintStore(cfg.baseline.profile_dir).centroids()
        return annotate_matches(
            signal, sr, diar, centroids,
            threshold=cfg.audio.voiceprint_match_threshold,
            margin=cfg.audio.voiceprint_match_margin,
        )
    except Exception:
        return diar


def _default_patient_id(base_user_id: str, speaker: str) -> str:
    suffix = speaker.lower().replace("_", "-")
    return f"{base_user_id}-{suffix}" if base_user_id != "anonymous" else suffix


def _build_profile(
    speech: np.ndarray,
    sr: int,
    vad: VADResult,
    user_id: str,
    source_filename: str,
    cfg: PipelineConfig,
    lang: str,
    transcript_override: str | None,
    warnings: list[str],
    t0: float,
    speaker: str = "SPEAKER_00",
    n_speakers: int = 1,
    progress_cb: ProgressCb | None = None,
    p0: float = 0,
    p1: float = 100,
    prefix: str = "",
) -> BiomarkerProfile:
    span = p1 - p0
    if speech.size == 0:
        warnings.append("No target-speaker speech detected in this recording.")
    _emit(progress_cb, p0 + 0.05 * span, f"{prefix}Extracting acoustic features (eGeMAPS)…")
    acoustic = extract_acoustic(speech, sr, vad, cfg.audio, cfg.acoustic_backend, warnings)
    _emit(progress_cb, p0 + 0.40 * span, f"{prefix}Transcribing speech (ASR)…")
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
        seg_cb = None
        if progress_cb is not None:
            def seg_cb(n_segments: int) -> None:  # noqa: F811 - whisper liveness
                _emit(progress_cb, p0 + 0.40 * span,
                      f"{prefix}Transcribing speech (ASR)… {n_segments} segments")
        text, segments, mean_conf, asr_backend = transcribe(
            speech, sr, cfg.whisper, language=lang, warnings=warnings,
            segment_cb=seg_cb,
        )
        _emit(progress_cb, p0 + 0.88 * span, f"{prefix}Computing linguistic metrics…")
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
    return BiomarkerProfile(
        user_id=user_id,
        speaker=speaker,
        n_speakers_detected=n_speakers,
        source_filename=source_filename,
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


def _attach_baseline(
    profile: BiomarkerProfile,
    cfg: PipelineConfig,
    compare: bool,
    save: bool,
) -> None:
    store = ProfileStore(cfg.baseline.profile_dir)
    if compare:
        priors = store.load_profiles(profile.user_id)
        profile.baseline = compare_to_baseline(
            profile,
            priors,
            min_sessions=cfg.baseline.min_baseline_sessions,
            z_threshold=cfg.baseline.z_alert_threshold,
        )
    if save:
        store.save_profile(profile)
