"""VAD + speaker diarization (SPEC Layer 1/2, §4 multi-speaker filtering).

Backends:
- `pyannote`: full diarization (needs torch + HF_TOKEN + accepted model terms).
- `energy`: adaptive energy VAD, single-speaker assumption. Always available.

`auto` tries pyannote and falls back to energy VAD, recording the choice.
"""

from __future__ import annotations

import tempfile

import numpy as np

from ..config import AudioConfig
from ..schemas import SpeechSegment, VADResult
from .ingest import write_wav


def run_vad(
    signal: np.ndarray,
    sr: int,
    cfg: AudioConfig,
    diarization: str = "auto",
    hf_token: str | None = None,
    warnings: list[str] | None = None,
) -> VADResult:
    """Return speech segments for the target (dominant) speaker."""
    warnings = warnings if warnings is not None else []
    total = float(signal.size) / float(sr) if sr else 0.0
    if diarization in ("auto", "pyannote"):
        try:
            return _pyannote_vad(signal, sr, total, hf_token)
        except Exception as exc:  # missing pkg/token/model — degrade, never crash
            if diarization == "pyannote":
                raise
            warnings.append(f"pyannote diarization unavailable ({exc}); used energy VAD.")
    return _energy_vad(signal, sr, cfg, total)


def _energy_vad(signal: np.ndarray, sr: int, cfg: AudioConfig, total: float) -> VADResult:
    frame = max(1, int(sr * cfg.frame_ms / 1000))
    n_frames = max(1, signal.size // frame)
    energies = np.array(
        [float(np.mean(signal[i * frame : (i + 1) * frame] ** 2)) for i in range(n_frames)],
        dtype=np.float64,
    )
    if energies.size == 0 or float(np.max(energies)) <= 0:
        return VADResult(backend="energy", total_duration_s=total)
    # Adaptive threshold: between noise floor (20th pct) and speech level.
    noise = float(np.percentile(energies, 20))
    peak = float(np.percentile(energies, 95))
    threshold = noise + 0.15 * max(peak - noise, 1e-12)
    is_speech = energies > threshold

    # Bridge short silences, then drop short speech blips.
    min_sil_frames = max(1, int(cfg.min_silence_ms / cfg.frame_ms))
    min_sp_frames = max(1, int(cfg.min_speech_ms / cfg.frame_ms))
    is_speech = _fill_gaps(is_speech, min_sil_frames)
    runs = _runs(is_speech)
    segments = [
        SpeechSegment(start_s=s * cfg.frame_ms / 1000, end_s=min(total, e * cfg.frame_ms / 1000))
        for s, e in runs
        if (e - s) >= min_sp_frames
    ]
    speech_dur = sum(seg.duration_s for seg in segments)
    return VADResult(
        backend="energy",
        segments=segments,
        n_speakers=1,
        target_speaker="SPEAKER_00",
        total_duration_s=round(total, 3),
        speech_duration_s=round(speech_dur, 3),
        speech_ratio=round(speech_dur / total, 4) if total > 0 else 0.0,
    )


def _pyannote_vad(signal: np.ndarray, sr: int, total: float, hf_token: str | None) -> VADResult:
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")
    from pyannote.audio import Pipeline  # optional heavy dep

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        write_wav(tmp.name, signal, sr)
        diarization = pipeline(tmp.name)
    by_speaker: dict[str, list[SpeechSegment]] = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        by_speaker.setdefault(str(speaker), []).append(
            SpeechSegment(start_s=float(turn.start), end_s=float(turn.end), speaker=str(speaker))
        )
    if not by_speaker:
        raise RuntimeError("pyannote found no speech")
    # Target = dominant speaker by total talk time.
    target = max(by_speaker, key=lambda s: sum(x.duration_s for x in by_speaker[s]))
    segments = sorted(by_speaker[target], key=lambda s: s.start_s)
    speech_dur = sum(seg.duration_s for seg in segments)
    return VADResult(
        backend="pyannote",
        segments=segments,
        n_speakers=len(by_speaker),
        target_speaker=target,
        total_duration_s=round(total, 3),
        speech_duration_s=round(speech_dur, 3),
        speech_ratio=round(speech_dur / total, 4) if total > 0 else 0.0,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    gap_start = None
    for i, v in enumerate(mask):
        if not v and gap_start is None:
            gap_start = i
        elif v and gap_start is not None:
            if i - gap_start <= max_gap and gap_start > 0:
                out[gap_start:i] = True
            gap_start = None
    return out


def extract_speech(
    signal: np.ndarray, sr: int, vad: VADResult
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Concatenate target-speaker speech; also return sample spans."""
    spans = [(int(s.start_s * sr), int(s.end_s * sr)) for s in vad.segments]
    if not spans:
        return np.zeros(0, dtype=np.float32), spans
    return np.concatenate([signal[a:b] for a, b in spans]).astype(np.float32), spans


def pause_durations(vad: VADResult, pause_threshold_s: float) -> list[float]:
    """Silent gaps between consecutive speech segments >= threshold."""
    segs = sorted(vad.segments, key=lambda s: s.start_s)
    pauses = []
    for prev, nxt in zip(segs, segs[1:], strict=False):
        gap = nxt.start_s - prev.end_s
        if gap >= pause_threshold_s:
            pauses.append(round(float(gap), 3))
    return pauses
