"""Progress reporting: pipeline callbacks + background job polling API."""

import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mci_biomarker.baseline.store import ProfileStore
from mci_biomarker.config import PipelineConfig
from mci_biomarker.pipeline import analyze_wav

SR = 16000
TRANSCRIPT = "The boy takes cookies from the jar while his mother washes dishes."


def synth(path: Path, freqs=(140.0,)) -> Path:
    chunks = []
    for freq in freqs:
        t = np.arange(int(2.0 * SR)) / SR
        tone = 0.3 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t))
        chunks.extend([tone.astype(np.float32), np.zeros(SR // 2, np.float32)])
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


@pytest.fixture()
def cfg(tmp_path):
    c = PipelineConfig()
    c.diarization = "cluster"
    c.acoustic_backend = "signal"
    c.baseline.profile_dir = tmp_path / "profiles"
    return c


def _poll(client, job_id, timeout_s=60):
    deadline = time.time() + timeout_s
    seen = []
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        seen.append(body["progress"])
        if body["status"] in ("done", "failed"):
            return body, seen
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_pipeline_progress_is_monotonic_and_complete(tmp_path, cfg):
    events: list[tuple[int, str]] = []
    wav = synth(tmp_path / "a.wav")
    analyze_wav(wav, cfg=cfg, transcript_override=TRANSCRIPT,
                progress_cb=lambda pct, stage: events.append((pct, stage)))
    pcts = [p for p, _ in events]
    assert pcts == sorted(pcts), pcts
    assert pcts[0] <= 5 and pcts[-1] == 100
    stages = " | ".join(s for _, s in events).lower()
    assert "acoustic" in stages and "baseline" in stages


def test_job_analyze_flow(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    mono_store = ProfileStore(cfg.baseline.profile_dir)
    monkeypatch.setattr(web_app, "_store", lambda: mono_store)
    client = TestClient(web_app.app)
    wav = synth(tmp_path / "a.wav")
    with wav.open("rb") as fh:
        resp = client.post(
            "/api/jobs/analyze",
            files={"file": ("a.wav", fh, "audio/wav")},
            data={"user_id": "job-u", "transcript": TRANSCRIPT, "save": "false"},
        )
    assert resp.status_code == 200, resp.text
    job, seen = _poll(client, resp.json()["job_id"])
    assert job["status"] == "done" and job["progress"] == 100
    assert job["result"]["user_id"] == "job-u"
    assert job["result"]["profile_id"]
    assert seen == sorted(seen)
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_diarize_and_analyze_speakers_flow(tmp_path, cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from mci_biomarker.web import app as web_app

    mono_store = ProfileStore(cfg.baseline.profile_dir)
    monkeypatch.setattr(web_app, "_store", lambda: mono_store)
    web_app._pending.clear()
    # Web _cfg() builds a fresh PipelineConfig (diarization=auto → cluster here);
    # only the profile dir needs the tmp override, which _cfg reads via _store.
    client = TestClient(web_app.app)
    wav = synth(tmp_path / "two.wav", freqs=(140.0, 240.0) * 2)
    with wav.open("rb") as fh:
        resp = client.post("/api/jobs/diarize", files={"file": ("two.wav", fh, "audio/wav")})
    assert resp.status_code == 200, resp.text
    job, _seen = _poll(client, resp.json()["job_id"])
    assert job["status"] == "done", job
    diar = job["result"]["diarization"]
    assert len(diar["speakers"]) == 2
    speakers = [s["speaker"] for s in diar["speakers"]]
    resp2 = client.post(
        "/api/jobs/analyze-speakers",
        json={
            "upload_id": job["result"]["upload_id"],
            "assignments": {speakers[0]: "p-one", speakers[1]: "p-two"},
            "language": "en",
            "save": True,
        },
    )
    assert resp2.status_code == 200, resp2.text
    job2, seen2 = _poll(client, resp2.json()["job_id"], timeout_s=120)
    assert job2["status"] == "done", job2
    assert seen2 == sorted(seen2)
    pids = {p["user_id"] for p in job2["result"]["profiles"]}
    assert pids == {"p-one", "p-two"}
    assert len(mono_store.load_profiles("p-one")) == 1
