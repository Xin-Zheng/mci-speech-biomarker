"""ASR transcription via faster-whisper (SPEC Track B).

Confidence = mean exp(avg_logprob) across segments. When it falls below the
threshold the pipeline flags linguistic metrics low-confidence and the report
leans on acoustics instead (SPEC §4 dialect robustness).
"""

from __future__ import annotations

import math

import numpy as np

from ..audio.ingest import TARGET_SR, resample_to
from ..config import WhisperConfig
from ..schemas import TranscriptSegment

_model_cache: dict[str, object] = {}


def transcribe(
    speech: np.ndarray,
    sr: int,
    cfg: WhisperConfig,
    language: str | None = None,
    warnings: list[str] | None = None,
) -> tuple[str, list[TranscriptSegment], float | None, str]:
    """Return (transcript, segments, mean_confidence, backend)."""
    warnings = warnings if warnings is not None else []
    if speech.size == 0:
        warnings.append("No speech detected; skipped transcription.")
        return "", [], None, "none"
    try:
        model = _load_model(cfg)
    except Exception as exc:
        warnings.append(
            f"Whisper unavailable ({exc}); transcript empty. "
            'Install with: pip install "mci-biomarker[asr]" or pass --transcript.'
        )
        return "", [], None, "none"
    # Pass the already-decoded waveform to faster-whisper. Giving it a file
    # path would send the audio through faster-whisper's PyAV decoder, whose
    # av.open(..., metadata_errors=...) call is incompatible with newer PyAV.
    waveform = np.asarray(speech, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1).astype(np.float32, copy=False)
    if sr != TARGET_SR:
        waveform = resample_to(waveform, sr, TARGET_SR)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)

    try:
        segments_iter, _info = model.transcribe(  # type: ignore[attr-defined]
            waveform, language=language, beam_size=5, vad_filter=False
        )
        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            conf = None
            logprob = getattr(seg, "avg_logprob", None)
            if logprob is not None and math.isfinite(logprob):
                conf = round(float(math.exp(max(-10.0, logprob))), 4)
            segments.append(
                TranscriptSegment(
                    start_s=float(seg.start),
                    end_s=float(seg.end),
                    text=str(seg.text).strip(),
                    confidence=conf,
                )
            )
    except Exception as exc:
        warnings.append(
            f"Whisper transcription failed ({exc}); transcript empty. "
            "Acoustic biomarkers were still extracted."
        )
        return "", [], None, "none"
    transcript = " ".join(s.text for s in segments).strip()
    confs = [s.confidence for s in segments if s.confidence is not None]
    mean_conf = round(float(np.mean(confs)), 4) if confs else None
    return transcript, segments, mean_conf, "faster-whisper"


def _load_model(cfg: WhisperConfig):
    key = f"{cfg.model}|{cfg.device}|{cfg.compute_type}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel  # optional dep

        _model_cache[key] = WhisperModel(
            cfg.model, device=cfg.device, compute_type=cfg.compute_type
        )
    return _model_cache[key]
