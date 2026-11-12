"""Multi-speaker diarization + per-patient reports: module, pipeline, web API."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mci_biomarker.audio.diarize import _greedy_cluster, diarize, vad_for_speaker
from mci_biomarker.baseline.store import PatientStore, ProfileStore
from mci_biomarker.config import PipelineConfig
from mci_biomarker.pipeline import analyze_wav_speakers

SR = 16000


def two_speaker_wav(path: Path) -> Path:
    """Alternating two harmonic voices (140 Hz vs 240 Hz) with silences."""

    def voice(freq: float, seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * SR)) / SR
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 5 * t)
        return (0.3 * env * (np.sin(2 * np.pi * freq * t)
                             + 0.3 * np.sin(2 * np.pi * 2 * freq * t))).astype(np.float32)

    chunks = [
        voice(140, 3.0), np.zeros(SR // 2, np.float32),
        voice(240, 3.0), np.zeros(SR // 2, np.float32),
        voice(140, 3.0), np.zeros(SR // 2, np.float32),
        voice(240, 3.0),
    ]
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


@pytest.fixture()
def cfg(tmp_path):
    c = PipelineConfig()
    c.diarization = "cluster"
    c.acoustic_backend = "signal"
    c.baseline.profile_dir = tmp_path / "profiles"
    c.baseline.min_baseline_sessions = 2
    return c


def test_greedy_cluster_separates_distinct_voices():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (6, 8)) + np.array([1.0] + [0.0] * 7)
    b = rng.normal(0, 0.05, (6, 8)) + np.array([0.0, 1.0] + [0.0] * 6)
    labels = _greedy_cluster(np.vstack([a, b]), threshold=0.5, max_speakers=4)
    assert len(set(labels[:6])) == 1 and len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_cluster_diarization_finds_two_speakers(tmp_path, cfg):
    wav = two_speaker_wav(tmp_path / "two.wav")
    import soundfile as _sf

    sig, sr = _sf.read(str(wav), dtype="float32")
    diar = diarize(np.asarray(sig, np.float32), sr, cfg.audio, "cluster")
    assert len(diar.speakers) == 2, diar.speakers
    labels = {s.speaker for s in diar.segments}
    assert len(labels) == 2
    for summary in diar.speakers:
        assert summary.duration_s > 4.0
        assert summary.f0_mean_hz is not None
    f0s = sorted(s.f0_mean_hz for s in diar.speakers)
    assert f0s[0] < 180 < f0s[1]  # 140 Hz vs 240 Hz voices told apart


def test_analyze_speakers_assigns_patients(tmp_path, cfg):
    wav = two_speaker_wav(tmp_path / "two.wav")
    diar, profiles = analyze_wav_speakers(
        wav,
        base_user_id="recording",
        cfg=cfg,
        speaker_map={"SPEAKER_00": "chan-tai"},
        save=True,
    )
    assert len(profiles) == 2
    by_speaker = {p.speaker: p for p in profiles}
    assert by_speaker["SPEAKER_00"].user_id == "chan-tai"
    assert by_speaker["SPEAKER_01"].user_id == "recording-speaker-01"
    for p in profiles:
        assert p.n_speakers_detected == 2
        assert p.acoustic.f0_mean_hz is not None
        assert p.vad.target_speaker == p.speaker
    store = ProfileStore(cfg.baseline.profile_dir)
    assert len(store.load_profiles("chan-tai")) == 1
    assert len(store.load_profiles("recording-speaker-01")) == 1


def test_vad_for_speaker_filters_segments():
    from mci_biomarker.schemas import DiarizationResult, SpeakerSummary, SpeechSegment

    diar = DiarizationResult(
        backend="cluster",
        total_duration_s=10.0,
        segments=[
            SpeechSegment(start_s=0, end_s=2, speaker="SPEAKER_00"),
            SpeechSegment(start_s=3, end_s=5, speaker="SPEAKER_01"),
            SpeechSegment(start_s=6, end_s=9, speaker="SPEAKER_00"),
        ],
        speakers=[SpeakerSummary(speaker="SPEAKER_00"), SpeakerSummary(speaker="SPEAKER_01")],
    )
    vad = vad_for_speaker(diar, "SPEAKER_00")
    assert vad.speech_duration_s == 5.0
    assert vad.n_speakers == 2 and vad.target_speaker == "SPEAKER_00"


def test_patient_store_merges_registry_and_profiles(tmp_path):
    cfg_dir = tmp_path / "profiles"
    profiles = ProfileStore(cfg_dir)
    patients = PatientStore(cfg_dir)
    patient = patients.upsert_patient("Chan Tai!", "陳伯", "seen 2026-11")
    assert patient.patient_id == "Chan_Tai"  # safe_user_id sanitization
    listed = patients.list_patients()
    assert len(listed) == 1 and listed[0]["session_count"] == 0
    assert patients.get_patient(patient.patient_id).display_name == "陳伯"
    assert patients.delete_patient(patient.patient_id) is True
    assert patients.list_patients() == []
    assert profiles.list_users() == []


def test_web_speaker_flow(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    mono_store = ProfileStore(cfg.baseline.profile_dir)
    monkeypatch.setattr(web_app, "_store", lambda: mono_store)
    web_app._pending.clear()
    client = TestClient(web_app.app)
    wav = two_speaker_wav(tmp_path / "two.wav")
    with wav.open("rb") as fh:
        dresp = client.post("/api/diarize", files={"file": ("two.wav", fh, "audio/wav")})
    assert dresp.status_code == 200, dresp.text
    diar_body = dresp.json()
    assert len(diar_body["diarization"]["speakers"]) == 2
    speakers = [s["speaker"] for s in diar_body["diarization"]["speakers"]]
    resp = client.post(
        "/api/analyze-speakers",
        json={
            "upload_id": diar_body["upload_id"],
            "base_user_id": "recording",
            "assignments": {speakers[0]: "chan-tai"},
            "new_patients": [{"patient_id": "chan-tai", "display_name": "Chan Tai"}],
            "language": "en",
            "save": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["profiles"]) == 2
    pids = {p["user_id"] for p in body["profiles"]}
    assert "chan-tai" in pids and "recording-speaker-01" in pids
    # Pending upload must be gone (zero-retention + one-shot analyze).
    again = client.post(
        "/api/analyze-speakers",
        json={"upload_id": diar_body["upload_id"], "save": False},
    )
    assert again.status_code == 404
    plist = client.get("/api/patients").json()["patients"]
    by_id = {p["patient_id"]: p for p in plist}
    assert by_id["chan-tai"]["display_name"] == "Chan Tai"
    assert by_id["chan-tai"]["session_count"] == 1
    detail = client.get("/api/patients/chan-tai").json()
    assert len(detail["profiles"]) == 1 and len(detail["trends"]["labels"]) == 1
    upd = client.post("/api/patients/lee-mui", json={"display_name": "Lee Mui"})
    assert upd.status_code == 200 and upd.json()["display_name"] == "Lee Mui"
    assert client.delete(
        "/api/patients/chan-tai?delete_profiles=true"
    ).json()["deleted"] is True
    assert client.get("/api/patients/chan-tai").status_code == 404
