"""Layer 1: load + normalize audio to 16 kHz mono float32 in [-1, 1]."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_wav(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load a wav (or any libsndfile format), downmix to mono, resample."""
    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if data.ndim > 1:  # stereo → mono
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    if sr != target_sr:
        data = resample_to(data, int(sr), target_sr)
        sr = target_sr
    return data, int(sr)


def resample_to(signal: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or signal.size == 0:
        return signal
    from math import gcd

    g = gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    return resample_poly(signal, up, down).astype(np.float32)


def write_wav(path: str | Path, signal: np.ndarray, sr: int = TARGET_SR) -> None:
    sf.write(str(path), np.asarray(signal, dtype=np.float32), sr)


def duration_s(signal: np.ndarray, sr: int) -> float:
    return float(signal.size) / float(sr) if sr else 0.0
