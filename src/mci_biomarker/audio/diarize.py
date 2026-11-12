"""Multi-speaker diarization: who spoke when (SPEC Layer 1, §4).

Backends:
- `pyannote`: neural diarization, all speakers (needs HF_TOKEN + model terms).
- `cluster`: VAD segments → MFCC/F0 voice embeddings → greedy cosine
  clustering. No downloads; works well for 2–4 distinct voices.
- `energy`: single-speaker assumption (no separation).

`auto` = pyannote → cluster. Every path degrades gracefully and records the
backend that actually ran.
"""

from __future__ import annotations

import tempfile

import numpy as np

from ..config import AudioConfig
from ..schemas import DiarizationResult, SpeakerSummary, SpeechSegment, VADResult
from .ingest import write_wav
from .vad import run_vad

CHUNK_S = 1.5
PITCH_WEIGHT = 2.0


def diarize(
    signal: np.ndarray,
    sr: int,
    cfg: AudioConfig,
    diarization: str = "auto",
    hf_token: str | None = None,
    warnings: list[str] | None = None,
) -> DiarizationResult:
    """Detect all speakers and their speech segments."""
    warnings = warnings if warnings is not None else []
    total = float(signal.size) / float(sr) if sr else 0.0
    chain = (
        ["pyannote", "cluster"] if diarization == "auto" else [diarization]
    )
    for backend in chain:
        try:
            if backend == "pyannote":
                return _pyannote_all(signal, sr, total, hf_token)
            if backend == "cluster":
                return _cluster_diarize(signal, sr, cfg, total, warnings)
            vad = run_vad(signal, sr, cfg, "energy", warnings=warnings)
            return _result("energy", vad.segments, signal, sr, total)
        except Exception as exc:
            warnings.append(f"{backend} diarization unavailable ({exc}).")
    vad = run_vad(signal, sr, cfg, "energy", warnings=warnings)
    return _result("energy", vad.segments, signal, sr, total)


def speaker_embedding(
    signal: np.ndarray, sr: int, segments: list[SpeechSegment]
) -> np.ndarray | None:
    """Mean chunk embedding for one speaker — the stored 'voiceprint'.

    Uses the exact chunking/embedding/pitch-weighting as diarization so
    enrolled and query embeddings live in the same space.
    """
    chunks = _chunk_segments(segments)
    if not chunks:
        return None
    emb = np.stack([_embed(signal, sr, c) for c in chunks])
    emb[:, -2:] *= PITCH_WEIGHT
    centroid = emb.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 1e-9 else None


def annotate_matches(
    signal: np.ndarray,
    sr: int,
    diar: DiarizationResult,
    voiceprints: dict[str, np.ndarray],
    threshold: float,
    margin: float = 0.02,
) -> DiarizationResult:
    """Fill matched_patient_id/match_similarity on each detected speaker.

    A match needs distance <= threshold AND a clear margin over the
    second-best patient; assignment is greedy best-first (1 patient ↔ 1 speaker).
    """
    if not voiceprints:
        return diar
    per_speaker: dict[str, list[tuple[float, str, float]]] = {}
    for summary in diar.speakers:
        segs = [s for s in diar.segments if s.speaker == summary.speaker]
        emb = speaker_embedding(signal, sr, segs)
        if emb is None:
            continue
        cands = []
        for patient_id, centroid in voiceprints.items():
            if centroid.shape != emb.shape:
                continue
            sim = float(emb @ (centroid / max(np.linalg.norm(centroid), 1e-9)))
            cands.append((1.0 - sim, patient_id, sim))
        per_speaker[summary.speaker] = sorted(cands)
    flat = [
        (dist, speaker, patient_id, sim)
        for speaker, cands in per_speaker.items()
        for dist, patient_id, sim in cands[:1]
        if dist <= threshold
        and (len(cands) < 2 or cands[1][0] - dist >= margin)
    ]
    by_speaker = {s.speaker: s for s in diar.speakers}
    taken: set[str] = set()
    for dist, speaker, patient_id, sim in sorted(flat):
        if patient_id in taken or by_speaker[speaker].matched_patient_id:
            continue
        by_speaker[speaker].matched_patient_id = patient_id
        by_speaker[speaker].match_similarity = round(sim, 4)
        taken.add(patient_id)
    return diar


def vad_for_speaker(diar: DiarizationResult, speaker: str) -> VADResult:
    """Per-speaker VAD view (what downstream per-patient analysis consumes)."""
    segments = sorted(
        (s for s in diar.segments if s.speaker == speaker), key=lambda s: s.start_s
    )
    speech_dur = sum(s.duration_s for s in segments)
    total = diar.total_duration_s
    return VADResult(
        backend=diar.backend,
        segments=segments,
        n_speakers=len(diar.speakers),
        target_speaker=speaker,
        total_duration_s=round(total, 3),
        speech_duration_s=round(speech_dur, 3),
        speech_ratio=round(speech_dur / total, 4) if total > 0 else 0.0,
    )


# --- pyannote (all speakers) ----------------------------------------------------

def _pyannote_all(
    signal: np.ndarray, sr: int, total: float, hf_token: str | None
) -> DiarizationResult:
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set")
    from pyannote.audio import Pipeline  # optional heavy dep

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        write_wav(tmp.name, signal, sr)
        diarization = pipeline(tmp.name)
    segments = [
        SpeechSegment(
            start_s=float(turn.start), end_s=float(turn.end), speaker=str(speaker)
        )
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    if not segments:
        raise RuntimeError("pyannote found no speech")
    return _result("pyannote", segments, signal, sr, total)


# --- MFCC/F0 clustering fallback -------------------------------------------------

def _cluster_diarize(
    signal: np.ndarray,
    sr: int,
    cfg: AudioConfig,
    total: float,
    warnings: list[str],
) -> DiarizationResult:
    vad = run_vad(signal, sr, cfg, "energy", warnings=warnings)
    chunks = _chunk_segments(vad.segments)
    if len(chunks) < 2:
        return _result("energy", vad.segments, signal, sr, total)
    embeddings = np.stack([_embed(signal, sr, c) for c in chunks])
    embeddings[:, -2:] *= PITCH_WEIGHT
    labels = _cluster_labels(embeddings, cfg)
    labels = _merge_tiny_clusters(chunks, labels, embeddings, total)
    # Chunk labels → segment labels (segments were split into chunks in order).
    segments: list[SpeechSegment] = []
    for chunk, label in zip(chunks, labels, strict=True):
        seg = SpeechSegment(
            start_s=chunk.start_s, end_s=chunk.end_s, speaker=f"SPEAKER_{label:02d}"
        )
        contiguous = (
            segments
            and segments[-1].speaker == seg.speaker
            and seg.start_s - segments[-1].end_s < 0.05
        )
        if contiguous:
            segments[-1] = SpeechSegment(
                start_s=segments[-1].start_s, end_s=seg.end_s, speaker=seg.speaker
            )
        else:
            segments.append(seg)
    return _result("cluster", segments, signal, sr, total)


def _chunk_segments(segments: list[SpeechSegment]) -> list[SpeechSegment]:
    chunks: list[SpeechSegment] = []
    for seg in segments:
        start = seg.start_s
        while seg.end_s - start > CHUNK_S:
            chunks.append(SpeechSegment(start_s=start, end_s=start + CHUNK_S))
            start += CHUNK_S
        chunks.append(SpeechSegment(start_s=start, end_s=seg.end_s))
    return [c for c in chunks if c.duration_s >= 0.3]


def _embed(signal: np.ndarray, sr: int, seg: SpeechSegment) -> np.ndarray:
    """Fixed-scale voice embedding: MFCC timbre + strongly weighted log-F0.

    Fixed scales (not per-recording z-scores) keep cosine distances meaningful
    even with very few chunks; pitch gets extra weight because it is the most
    reliable cheap cue for telling household voices apart.
    """
    audio = signal[int(seg.start_s * sr) : int(seg.end_s * sr)].astype(np.float32)
    try:
        import librosa  # acoustic extra

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        # Drop c0 (overall energy — dominated by loudness, not identity).
        timbre = np.concatenate([mfcc.mean(axis=1)[1:], mfcc.std(axis=1)[1:]])
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        timbre = np.append(timbre, float(cent.mean()) / 4000.0)
    except Exception:
        timbre = np.asarray(_spectral_fallback(audio, sr), dtype=np.float64)
    # Block-normalize timbre so pitch (below) competes fairly in cosine space.
    norm = np.linalg.norm(timbre)
    timbre = timbre / norm if norm > 1e-9 else timbre

    from .acoustic import _f0_track

    f0s, _amps = _f0_track(audio, sr)
    if f0s.size >= 3:
        pitch = np.array([
            float(np.log2(max(float(np.median(f0s)), 40.0) / 150.0)),
            float(np.std(np.log2(np.maximum(f0s, 1.0)))),
        ])
    else:
        pitch = np.zeros(2)
    return np.concatenate([timbre, pitch]).astype(np.float64)


def _spectral_fallback(audio: np.ndarray, sr: int) -> list[float]:
    """Log band energies when librosa is absent (39 dims, matches MFCC path)."""
    if audio.size == 0:
        return [0.0] * 39
    window = np.hanning(audio.size).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64) * window)) + 1e-9
    freqs = np.fft.rfftfreq(audio.size, 1 / sr)
    edges = np.geomspace(80, sr / 2, 39)
    vals = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = spectrum[(freqs >= lo) & (freqs < hi)]
        vals.append(float(np.log(band.mean())) if band.size else 0.0)
    vals.append(float(np.log(spectrum.mean())))
    return (vals + [0.0] * 39)[:39]


def _cluster_labels(embeddings: np.ndarray, cfg: AudioConfig) -> list[int]:
    """Average-linkage agglomerative clustering (sklearn ships with librosa).

    Greedy leader clustering with a drifting centroid merges similar voices
    (validated on HKCanCor dialogue: two young males, agglomerative splits
    them cleanly where greedy cannot).
    """
    try:
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cfg.cluster_threshold,
            linkage="average",
            metric="cosine",
        ).fit_predict(embeddings)
        if len(set(labels.tolist())) > cfg.max_speakers:
            labels = AgglomerativeClustering(
                n_clusters=cfg.max_speakers, linkage="average", metric="cosine"
            ).fit_predict(embeddings)
        return [int(x) for x in labels]
    except Exception:
        return _greedy_cluster(embeddings, cfg.cluster_threshold, cfg.max_speakers)


def _greedy_cluster(
    embeddings: np.ndarray, threshold: float, max_speakers: int
) -> list[int]:
    # L2-normalize only: embeddings already use fixed scales (see _embed).
    # Per-dimension z-scoring over a handful of chunks badly distorts distances.
    emb = embeddings.copy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-9)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[int] = []
    for vec in emb:
        if centroids:
            sims = [float(vec @ c) for c in centroids]
            best = int(np.argmax(sims))
            if 1.0 - sims[best] <= threshold or len(centroids) >= max_speakers:
                labels.append(best)
                counts[best] += 1
                centroids[best] = _renorm(
                    centroids[best] * (counts[best] - 1) + vec, counts[best]
                )
                continue
        centroids.append(vec.copy())
        counts.append(1)
        labels.append(len(centroids) - 1)
    return labels


def _renorm(vec_sum: np.ndarray, n: int) -> np.ndarray:
    mean = vec_sum / max(n, 1)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-9 else mean


def _merge_tiny_clusters(
    chunks: list[SpeechSegment],
    labels: list[int],
    embeddings: np.ndarray,
    total: float,
) -> list[int]:
    durations: dict[int, float] = {}
    for chunk, label in zip(chunks, labels, strict=True):
        durations[label] = durations.get(label, 0.0) + chunk.duration_s
    speech_total = sum(durations.values()) or 1.0
    big = {lb for lb, d in durations.items() if d / speech_total >= 0.08}
    if not big or big == set(durations):
        order = sorted(durations, key=lambda lb: -durations[lb])
        remap = {lb: i for i, lb in enumerate(order)}
        return [remap[lb] for lb in labels]
    centroids = {
        lb: embeddings[[i for i, x in enumerate(labels) if x == lb]].mean(axis=0)
        for lb in durations
    }
    merged = []
    for i, lb in enumerate(labels):
        if lb in big:
            merged.append(lb)
        else:
            nearest = min(
                big,
                key=lambda b: float(
                    np.linalg.norm(embeddings[i] - centroids[b])
                ),
            )
            merged.append(nearest)
    order = sorted(big, key=lambda lb: -durations[lb])
    remap = {lb: i for i, lb in enumerate(order)}
    return [remap[lb] for lb in merged]


# --- shared result builder --------------------------------------------------------

def _result(
    backend: str,
    segments: list[SpeechSegment],
    signal: np.ndarray,
    sr: int,
    total: float,
) -> DiarizationResult:
    from .acoustic import _f0_track

    segments = sorted(segments, key=lambda s: s.start_s)
    by_speaker: dict[str, list[SpeechSegment]] = {}
    for seg in segments:
        by_speaker.setdefault(seg.speaker, []).append(seg)
    summaries = []
    for speaker in sorted(by_speaker, key=lambda s: -sum(
        x.duration_s for x in by_speaker[s]
    )):
        segs = by_speaker[speaker]
        f0_mean = None
        try:
            audio = np.concatenate(
                [signal[int(s.start_s * sr) : int(s.end_s * sr)] for s in segs]
            )
            f0s, _ = _f0_track(audio.astype(np.float32), sr)
            if f0s.size >= 3:
                f0_mean = round(float(np.median(f0s)), 1)
        except Exception:
            f0_mean = None
        summaries.append(
            SpeakerSummary(
                speaker=speaker,
                duration_s=round(sum(s.duration_s for s in segs), 3),
                segment_count=len(segs),
                f0_mean_hz=f0_mean,
            )
        )
    return DiarizationResult(
        backend=backend,
        segments=segments,
        speakers=summaries,
        total_duration_s=round(total, 3),
    )
