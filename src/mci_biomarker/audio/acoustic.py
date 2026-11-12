"""Track A: acoustic / paralinguistic features (SPEC §2).

Primary backend: openSMILE eGeMAPS functionals (88 features) via `opensmile`.
Fallback backend: pure signal processing (autocorrelation F0, jitter/shimmer
estimates, loudness) so the pipeline always yields voice-quality metrics.

Pause + speech-rate metrics are computed from VAD segments + syllable nuclei
and are backend-independent.
"""

from __future__ import annotations

import tempfile

import numpy as np
from scipy.signal import find_peaks

from ..config import AudioConfig
from ..schemas import AcousticFeatures, VADResult
from .ingest import write_wav
from .vad import pause_durations


def extract_acoustic(
    speech: np.ndarray,
    sr: int,
    vad: VADResult,
    cfg: AudioConfig,
    backend: str = "auto",
    warnings: list[str] | None = None,
) -> AcousticFeatures:
    """Compute Track A features from concatenated target-speaker speech."""
    warnings = warnings if warnings is not None else []
    pauses = pause_durations(vad, cfg.pause_threshold_s)
    total_min = vad.total_duration_s / 60.0 if vad.total_duration_s > 0 else 0.0
    speech_s = vad.speech_duration_s
    syllables = count_syllable_nuclei(speech, sr)

    feats = AcousticFeatures(
        backend="signal-fallback",
        speech_rate_syll_per_s=(
            _r(syllables / vad.total_duration_s) if vad.total_duration_s else None
        ),
        articulation_rate_syll_per_s=_r(syllables / speech_s) if speech_s else None,
        mean_pause_duration_s=_r(float(np.mean(pauses))) if pauses else 0.0,
        pause_count=len(pauses),
        pause_rate_per_min=_r(len(pauses) / total_min) if total_min else None,
        long_pause_count=sum(1 for p in pauses if p >= cfg.long_pause_threshold_s),
        pause_durations_s=pauses,
    )
    feats.long_pause_rate_per_min = (
        _r(feats.long_pause_count / total_min) if total_min else None
    )

    if backend in ("auto", "opensmile") and speech.size > 0:
        try:
            egemaps = _opensmile_egemaps(speech, sr)
            feats.backend = "opensmile-egemaps"
            feats.egemaps = egemaps
            _fill_from_egemaps(feats, egemaps)
            return feats
        except Exception as exc:
            if backend == "opensmile":
                raise
            warnings.append(f"openSMILE unavailable ({exc}); used signal acoustic fallback.")
    _fill_from_signal(feats, speech, sr)
    return feats


def _opensmile_egemaps(speech: np.ndarray, sr: int) -> dict[str, float]:
    import opensmile  # optional dep: pip install "mci-biomarker[acoustic]"

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        write_wav(tmp.name, speech, sr)
        df = smile.process_file(tmp.name)
    row = df.iloc[0]
    return {str(k): float(v) for k, v in row.items() if np.isfinite(v)}


def _fill_from_egemaps(feats: AcousticFeatures, e: dict[str, float]) -> None:
    # eGeMAPSv02 functional names (v01 names kept as fallbacks).
    feats.f0_mean_hz = _r(
        _first(
            e,
            ["F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_mean"],
            semitone=True,
        )
    )
    std_norm = _first(
        e, ["F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "F0semitoneFrom27.5Hz_stddevNorm"]
    )
    if feats.f0_mean_hz and std_norm is not None:
        feats.f0_std_hz = _r(abs(feats.f0_mean_hz * std_norm))
    feats.jitter_local = _r(_first(e, ["jitterLocal_sma3nz_amean", "jitterLocal_mean"]))
    feats.shimmer_local_db = _r(
        _first(e, ["shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_mean"])
    )
    feats.hnr_db = _r(_first(e, ["HNRdBACF_sma3nz_amean", "HNRdBACF_mean"]))
    feats.loudness_mean = _r(_first(e, ["loudness_sma3_amean", "loudness_mean"]))


def _fill_from_signal(feats: AcousticFeatures, speech: np.ndarray, sr: int) -> None:
    """Autocorrelation F0 track → F0 stats, jitter/shimmer approximations."""
    if speech.size == 0:
        return
    feats.loudness_mean = _r(float(np.sqrt(np.mean(speech.astype(np.float64) ** 2))))
    f0s, amps = _f0_track(speech, sr)
    if f0s.size < 3:
        return
    feats.f0_mean_hz = _r(float(np.mean(f0s)))
    feats.f0_std_hz = _r(float(np.std(f0s)))
    periods = 1.0 / f0s
    feats.jitter_local = _r(float(np.mean(np.abs(np.diff(periods))) / np.mean(periods)))
    if amps.size >= 3 and np.all(amps > 0):
        feats.shimmer_local_db = _r(
            float(np.mean(np.abs(20 * np.log10(amps[1:] / amps[:-1]))))
        )


def _f0_track(
    signal: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0
) -> tuple[np.ndarray, np.ndarray]:
    frame = int(0.04 * sr)
    hop = int(0.01 * sr)
    min_lag, max_lag = int(sr / fmax), int(sr / fmin)
    f0s, amps = [], []
    for start in range(0, max(0, signal.size - frame), hop):
        chunk = signal[start : start + frame].astype(np.float64)
        if float(np.sqrt(np.mean(chunk**2))) < 0.01:
            continue
        chunk = chunk - chunk.mean()
        corr = np.correlate(chunk, chunk, mode="full")[frame - 1 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        seg = corr[min_lag:max_lag]
        if seg.size == 0:
            continue
        lag = int(np.argmax(seg)) + min_lag
        if corr[lag] > 0.45:  # voiced frame
            f0s.append(sr / lag)
            amps.append(float(np.max(np.abs(chunk))))
    return np.array(f0s), np.array(amps)


def count_syllable_nuclei(signal: np.ndarray, sr: int) -> int:
    """Estimate syllables from amplitude-envelope peaks (vowel nuclei proxy)."""
    if signal.size == 0:
        return 0
    win = max(1, int(0.02 * sr))
    env = np.convolve(np.abs(signal.astype(np.float64)), np.ones(win) / win, mode="same")
    if float(np.max(env)) <= 0:
        return 0
    peaks, _ = find_peaks(
        env, height=0.25 * float(np.max(env)), distance=int(0.08 * sr), prominence=0.05
    )
    return int(len(peaks))


def _first(d: dict[str, float], keys: list[str], semitone: bool = False) -> float | None:
    for k in keys:
        if k in d:
            return 27.5 * (2 ** (d[k] / 12.0)) if semitone else d[k]
    return None


def _r(x: float | None, nd: int = 4) -> float | None:
    return round(float(x), nd) if x is not None and np.isfinite(x) else None
