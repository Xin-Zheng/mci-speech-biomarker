"""Voiceprint enrollment → auto-match → auto-assign patient tests."""

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mci_biomarker.audio.diarize import diarize, speaker_embedding
from mci_biomarker.baseline.store import ProfileStore, VoiceprintStore
from mci_biomarker.config import PipelineConfig
from mci_biomarker.pipeline import analyze_wav, analyze_wav_speakers

SR = 16000
TRANSCRIPT = "The boy takes cookies from the jar while his mother washes dishes."


def voice(freq: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 5 * t)
    return (0.3 * env * (np.sin(2 * np.pi * freq * t)
                         + 0.3 * np.sin(2 * np.pi * 2 * freq * t))).astype(np.float32)


def make_wav(path: Path, freqs) -> Path:
    chunks = []
    for f in freqs:
        chunks.extend([voice(f, 3.0), np.zeros(SR // 2, np.float32)])
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


@pytest.fixture()
def cfg(tmp_path):
    c = PipelineConfig()
    c.diarization = "cluster"
    c.acoustic_backend = "signal"
    c.baseline.profile_dir = tmp_path / "profiles"
    return c


def test_voiceprint_store_roundtrip(tmp_path):
    store = VoiceprintStore(tmp_path / "profiles")
    vec = np.zeros(41)
    vec[0] = 1.0
    store.enroll("pat-a", vec)
    store.enroll("pat-a", vec)  # rolling centroid, count grows
    centroids = store.centroids()
    assert "pat-a" in centroids and centroids["pat-a"].shape == (41,)
    assert abs(float(np.linalg.norm(centroids["pat-a"])) - 1.0) < 1e-6
    assert store.delete("pat-a") is True
    assert store.centroids() == {}


def test_enroll_then_auto_assign(tmp_path, cfg):
    # 1) Enroll voice A (140 Hz) via a normal saved single-patient analysis.
    enroll_wav = make_wav(tmp_path / "enroll.wav", [140.0])
    analyze_wav(enroll_wav, user_id="voice-a", cfg=cfg,
                transcript_override=TRANSCRIPT, save=True)
    store = VoiceprintStore(cfg.baseline.profile_dir)
    assert "voice-a" in store.centroids()

    # 2) New two-voice recording, no manual map → voice A auto-assigned.
    duo = make_wav(tmp_path / "duo.wav", [140.0, 240.0, 140.0, 240.0])
    diar, profiles = analyze_wav_speakers(
        duo, base_user_id="recording", cfg=cfg,
        transcripts={}, save=False,
    )
    assert len(profiles) == 2
    matched = [s for s in diar.speakers if s.matched_patient_id == "voice-a"]
    assert len(matched) == 1
    assert matched[0].f0_mean_hz < 180  # the 140 Hz voice, not the 240 Hz one
    assert matched[0].match_similarity > 0.95
    by_speaker = {p.speaker: p for p in profiles}
    assert by_speaker[matched[0].speaker].user_id == "voice-a"
    other = [p for p in profiles if p.user_id != "voice-a"][0]
    assert other.user_id.startswith("recording-speaker-")


def test_no_false_match_for_unknown_voice(tmp_path, cfg):
    enroll_wav = make_wav(tmp_path / "enroll.wav", [140.0])
    analyze_wav(enroll_wav, user_id="voice-a", cfg=cfg,
                transcript_override=TRANSCRIPT, save=True)
    import soundfile as _sf

    other = make_wav(tmp_path / "other.wav", [240.0, 240.0])
    sig, sr = _sf.read(str(other), dtype="float32")
    diar = diarize(np.asarray(sig, np.float32), sr, cfg.audio, "cluster")
    from mci_biomarker.pipeline import annotate_voiceprints

    diar = annotate_voiceprints(np.asarray(sig, np.float32), sr, diar, cfg)
    assert all(s.matched_patient_id is None for s in diar.speakers)


def test_speaker_embedding_same_voice_is_close(tmp_path):
    import soundfile as _sf

    from mci_biomarker.audio.vad import run_vad
    from mci_biomarker.config import AudioConfig

    wav = make_wav(tmp_path / "a.wav", [140.0, 140.0])
    sig, sr = _sf.read(str(wav), dtype="float32")
    sig = np.asarray(sig, np.float32)
    vad = run_vad(sig, sr, AudioConfig(), "energy")
    assert len(vad.segments) >= 2
    e1 = speaker_embedding(sig, sr, vad.segments[:1])
    e2 = speaker_embedding(sig, sr, vad.segments[1:2])
    assert e1 is not None and e2 is not None
    assert 1.0 - float(e1 @ e2) < 0.05


def _poll(client, job_id, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_web_diarize_returns_auto_match(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    mono_store = ProfileStore(cfg.baseline.profile_dir)
    monkeypatch.setattr(web_app, "_store", lambda: mono_store)
    web_app._pending.clear()
    client = TestClient(web_app.app)

    enroll_wav = make_wav(tmp_path / "enroll.wav", [140.0])
    with enroll_wav.open("rb") as fh:
        r = client.post(
            "/api/jobs/analyze",
            files={"file": ("enroll.wav", fh, "audio/wav")},
            data={"user_id": "voice-a", "transcript": TRANSCRIPT, "save": "true"},
        )
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "done", job

    duo = make_wav(tmp_path / "duo.wav", [140.0, 240.0, 140.0, 240.0])
    with duo.open("rb") as fh:
        r2 = client.post("/api/jobs/diarize", files={"file": ("duo.wav", fh, "audio/wav")})
    job2 = _poll(client, r2.json()["job_id"])
    assert job2["status"] == "done", job2
    speakers = job2["result"]["diarization"]["speakers"]
    matches = [s for s in speakers if s.get("matched_patient_id") == "voice-a"]
    assert len(matches) == 1
    assert matches[0]["match_similarity"] > 0.95
    # Exactly the low voice matched; the other stays unmatched.
    unmatched = [s for s in speakers if not s.get("matched_patient_id")]
    assert len(unmatched) == 1 and unmatched[0]["f0_mean_hz"] > matches[0]["f0_mean_hz"]
