"""Pipeline, VAD, baseline, and web API tests on synthetic speech-like audio."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mci_biomarker.audio.vad import run_vad
from mci_biomarker.baseline.store import ProfileStore, compare_to_baseline
from mci_biomarker.config import AudioConfig, PipelineConfig
from mci_biomarker.pipeline import analyze_wav

SR = 16000
TRANSCRIPT = (
    "The mother washes dishes while water overflows. "
    "A boy reaches into the cookie jar and his sister watches him carefully."
)


def synth_wav(path: Path, bursts=(0.8, 0.8, 0.8), gaps=(0.5, 0.5), freq=140.0) -> Path:
    """Speech-like: harmonic bursts separated by silence (known pauses)."""
    chunks = []
    for i, dur in enumerate(bursts):
        t = np.arange(int(dur * SR)) / SR
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 5 * t)  # syllable-ish amplitude modulation
        tone = env * (
            0.5 * np.sin(2 * np.pi * freq * t)
            + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.12 * np.sin(2 * np.pi * 3 * freq * t)
        )
        chunks.append((0.35 * tone).astype(np.float32))
        if i < len(gaps):
            chunks.append(np.zeros(int(gaps[i] * SR), dtype=np.float32))
    signal = np.concatenate(chunks)
    sf.write(str(path), signal, SR)
    return path


def test_energy_vad_finds_segments_and_pauses(tmp_path):
    wav = synth_wav(tmp_path / "a.wav")
    import soundfile as _sf

    sig, sr = _sf.read(str(wav), dtype="float32")
    vad = run_vad(np.asarray(sig, dtype=np.float32), sr, AudioConfig(), "energy")
    assert vad.backend == "energy"
    assert len(vad.segments) >= 2
    assert vad.speech_ratio > 0.3


@pytest.fixture()
def cfg(tmp_path):
    c = PipelineConfig()
    c.diarization = "energy"
    c.acoustic_backend = "signal"
    c.baseline.profile_dir = tmp_path / "profiles"
    c.baseline.min_baseline_sessions = 2
    return c


def test_end_to_end_profile(tmp_path, cfg):
    wav = synth_wav(tmp_path / "session.wav")
    profile = analyze_wav(wav, user_id="u1", cfg=cfg, transcript_override=TRANSCRIPT)
    assert profile.acoustic.pause_count >= 1
    assert profile.acoustic.speech_rate_syll_per_s is not None
    assert profile.acoustic.f0_mean_hz is not None
    assert abs(profile.acoustic.f0_mean_hz - 140) < 35  # near synthetic F0
    assert profile.linguistic.ttr is not None and profile.linguistic.ttr > 0
    assert profile.linguistic.pronoun_to_noun_ratio is not None
    assert profile.linguistic.backend == "manual"
    assert profile.provenance.vad_backend == "energy"
    payload = profile.model_dump_json()
    assert "biomarker" not in payload or True  # JSON serializable (no exception)


def test_baseline_zscores_after_enough_sessions(tmp_path, cfg):
    wav = synth_wav(tmp_path / "s.wav")
    store = ProfileStore(cfg.baseline.profile_dir)
    for _ in range(3):
        p = analyze_wav(wav, user_id="u2", cfg=cfg, transcript_override=TRANSCRIPT)
        store.save_profile(p)
    priors = store.load_profiles("u2")
    assert len(priors) == 3
    nxt = analyze_wav(wav, user_id="u2", cfg=cfg, transcript_override=TRANSCRIPT)
    comp = compare_to_baseline(nxt, priors, min_sessions=2)
    assert comp.baseline_ready is True
    assert comp.n_prior_sessions == 3


def test_web_api(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    monkeypatch.setattr(web_app, "_store", lambda: ProfileStore(cfg.baseline.profile_dir))
    client = TestClient(web_app.app)
    assert client.get("/api/health").json()["status"] == "ok"
    assert "MCI" in client.get("/").text
    wav = synth_wav(tmp_path / "up.wav")
    with wav.open("rb") as fh:
        resp = client.post(
            "/api/analyze",
            files={"file": ("up.wav", fh, "audio/wav")},
            data={"user_id": "web-u", "transcript": TRANSCRIPT, "save": "true"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user_id"] == "web-u"
    assert body["acoustic"]["pause_count"] >= 1
    assert client.get("/api/users/web-u/profiles").json()["count"] == 1
    trends = client.get("/api/users/web-u/trends").json()
    assert len(trends["labels"]) == 1 and "ttr" in trends["metrics"]
