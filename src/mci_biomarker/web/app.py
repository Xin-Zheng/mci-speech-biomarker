"""FastAPI web UI (SPEC Layers 1 + 4): upload → biomarker reports + patients.

Flows:
- Single patient: POST /api/analyze (dominant speaker).
- Multi-speaker: POST /api/diarize → assign speakers to patients in the UI →
  POST /api/analyze-speakers (one profile per patient).
- Patients: /api/patients CRUD + per-patient profiles/trends.

Zero-retention: uploads live in temp files deleted right after analysis
(multi-speaker pending uploads auto-expire after 30 min); only JSON
profiles/patient metadata persist.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..audio.diarize import diarize
from ..audio.ingest import load_wav
from ..baseline.store import (
    PatientStore,
    ProfileStore,
    VoiceprintStore,
    safe_user_id,
    trend_series,
)
from ..config import PipelineConfig
from ..pipeline import analyze_wav, analyze_wav_speakers, annotate_voiceprints
from ..schemas import DiarizationResult

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_MB = 50
PENDING_TTL_S = 30 * 60

app = FastAPI(title="MCI Speech Biomarker", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# upload_id -> {path, filename, diarization (json dict), created}
_pending: dict[str, dict] = {}

# Background analysis jobs (for progress reporting): job_id -> status dict.
# Jobs run one at a time (CPU-bound ASR/acoustics) behind _job_lock.
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
JOB_TTL_S = 30 * 60


def _store() -> ProfileStore:
    return ProfileStore(PipelineConfig().baseline.profile_dir)


def _patients() -> PatientStore:
    return PatientStore(_store().profile_dir)


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.baseline.profile_dir = _store().profile_dir
    return cfg


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "mci-biomarker"}


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    name = (file.filename or "upload.wav").lower()
    if not name.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Please upload an audio file (.wav recommended).")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".wav", delete=False) as tmp:
        tmp.write(data)
        return tmp.name, file.filename or name


def _drop(path: str | None) -> None:
    if path and os.path.exists(path):
        os.unlink(path)


# --- Background jobs with progress (poll GET /api/jobs/{id}) ---------------------

def _purge_jobs() -> None:
    now = time.time()
    for jid in [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL_S]:
        _jobs.pop(jid, None)


def _start_job(kind: str, fn: Callable[[Callable[[int, str], None]], dict]) -> str:
    """Run `fn(progress_cb) -> result dict` on a serialized worker thread."""
    _purge_jobs()
    job_id = uuid4().hex[:12]
    job = {
        "kind": kind,
        "status": "queued",
        "progress": 0,
        "stage": "Queued…",
        "created": time.time(),
        "started": None,
        "result": None,
        "error": None,
    }
    _jobs[job_id] = job

    def worker() -> None:
        with _job_lock:
            job["status"] = "running"
            job["started"] = time.time()

            def cb(pct: int, stage: str) -> None:
                job["progress"] = max(job["progress"], pct) if pct else pct
                job["stage"] = stage

            try:
                job["result"] = fn(cb)
                job["progress"] = 100
                job["stage"] = "Done"
                job["status"] = "done"
            except Exception as exc:  # surfaced to the UI via job status
                job["error"] = str(exc)
                job["status"] = "failed"

    threading.Thread(target=worker, daemon=True, name=f"job-{kind}").start()
    return job_id


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    _purge_jobs()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    started = job["started"] or job["created"]
    return {
        "job_id": job_id,
        "kind": job["kind"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "elapsed_s": round(time.time() - started, 1),
        "result": job["result"] if job["status"] == "done" else None,
        "error": job["error"],
    }


@app.post("/api/jobs/analyze")
async def job_analyze(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    transcript: str = Form(""),
    language: str = Form("en"),
    save: bool = Form(True),
) -> dict:
    tmp_path, filename = await _save_upload(file)

    def run(cb: Callable[[int, str], None]) -> dict:
        try:
            profile = analyze_wav(
                tmp_path,
                user_id=user_id or "anonymous",
                cfg=_cfg(),
                transcript_override=transcript.strip() or None,
                language=(language or "en").lower(),
                save=save,
                progress_cb=cb,
            )
            profile.source_filename = filename
            return profile.model_dump(mode="json")
        finally:
            _drop(tmp_path)

    return {"job_id": _start_job("analyze", run)}


@app.post("/api/jobs/diarize")
async def job_diarize(file: UploadFile = File(...)) -> dict:
    _purge_pending()
    tmp_path, filename = await _save_upload(file)

    def run(cb: Callable[[int, str], None]) -> dict:
        try:
            cfg = _cfg()
            cb(5, "Loading audio…")
            signal, sr = load_wav(tmp_path, cfg.audio.target_sr)
            cb(15, "Detecting speakers (diarization)…")
            warnings: list[str] = []
            diar = diarize(signal, sr, cfg.audio, cfg.diarization, cfg.hf_token, warnings)
            cb(80, "Matching voices to known patients…")
            diar = annotate_voiceprints(signal, sr, diar, cfg)
            cb(90, "Summarizing speakers…")
            upload_id = uuid4().hex[:12]
            _pending[upload_id] = {
                "path": tmp_path,
                "filename": filename,
                "diarization": diar.model_dump(mode="json"),
                "created": time.time(),
            }
            return {
                "upload_id": upload_id,
                "filename": filename,
                "diarization": diar.model_dump(mode="json"),
                "warnings": warnings,
            }
        except Exception:
            _drop(tmp_path)
            raise

    return {"job_id": _start_job("diarize", run)}


# --- Single-patient analysis ----------------------------------------------------

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    transcript: str = Form(""),
    language: str = Form("en"),
    save: bool = Form(True),
) -> JSONResponse:
    tmp_path, filename = await _save_upload(file)
    try:
        profile = analyze_wav(
            tmp_path,
            user_id=user_id or "anonymous",
            cfg=_cfg(),
            transcript_override=transcript.strip() or None,
            language=(language or "en").lower(),
            save=save,
        )
        profile.source_filename = filename
        return JSONResponse(profile.model_dump(mode="json"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Analysis failed: {exc}") from exc
    finally:
        _drop(tmp_path)  # zero-retention (SPEC §4)


# --- Multi-speaker: diarize → assign → analyze per patient -----------------------

def _purge_pending() -> None:
    now = time.time()
    for uid in [u for u, v in _pending.items() if now - v["created"] > PENDING_TTL_S]:
        _drop(_pending.pop(uid)["path"])


@app.post("/api/diarize")
async def diarize_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Step 1: detect who spoke when. Keeps the upload (temp) for step 2."""
    _purge_pending()
    tmp_path, filename = await _save_upload(file)
    try:
        cfg = _cfg()
        signal, sr = load_wav(tmp_path, cfg.audio.target_sr)
        warnings: list[str] = []
        diar = diarize(signal, sr, cfg.audio, cfg.diarization, cfg.hf_token, warnings)
        diar = annotate_voiceprints(signal, sr, diar, cfg)
    except Exception as exc:
        _drop(tmp_path)
        raise HTTPException(422, f"Diarization failed: {exc}") from exc
    upload_id = uuid4().hex[:12]
    _pending[upload_id] = {
        "path": tmp_path,
        "filename": filename,
        "diarization": diar.model_dump(mode="json"),
        "created": time.time(),
    }
    return JSONResponse(
        {
            "upload_id": upload_id,
            "filename": filename,
            "diarization": diar.model_dump(mode="json"),
            "warnings": warnings,
        }
    )


class NewPatient(BaseModel):
    patient_id: str
    display_name: str = ""
    notes: str = ""


class AnalyzeSpeakersRequest(BaseModel):
    upload_id: str
    base_user_id: str = "anonymous"
    assignments: dict[str, str] = Field(default_factory=dict)  # SPEAKER_xx -> patient_id
    new_patients: list[NewPatient] = Field(default_factory=list)
    language: str = "en"
    save: bool = True


@app.post("/api/analyze-speakers")
def analyze_speakers(req: AnalyzeSpeakersRequest) -> JSONResponse:
    """Step 2: one biomarker profile per speaker, saved under each patient."""
    _purge_pending()
    pending = _pending.pop(req.upload_id, None)
    if pending is None:
        raise HTTPException(
            404, "Upload expired or unknown — please re-upload and detect speakers again."
        )
    try:
        for np_ in req.new_patients:
            _patients().upsert_patient(np_.patient_id, np_.display_name, np_.notes)
        diar = DiarizationResult.model_validate(pending["diarization"])
        _diar, profiles = analyze_wav_speakers(
            pending["path"],
            base_user_id=req.base_user_id or "anonymous",
            cfg=_cfg(),
            language=(req.language or "en").lower(),
            speaker_map={k: safe_user_id(v) for k, v in req.assignments.items() if v},
            save=req.save,
            diarization_result=diar,
        )
        for profile in profiles:
            profile.source_filename = pending["filename"]
        return JSONResponse(
            {
                "diarization": diar.model_dump(mode="json"),
                "profiles": [p.model_dump(mode="json") for p in profiles],
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Analysis failed: {exc}") from exc
    finally:
        _drop(pending["path"])  # zero-retention (SPEC §4)


@app.post("/api/jobs/analyze-speakers")
def job_analyze_speakers(req: AnalyzeSpeakersRequest) -> dict:
    """Background version of /api/analyze-speakers with progress polling."""
    _purge_pending()
    pending = _pending.pop(req.upload_id, None)
    if pending is None:
        raise HTTPException(
            404, "Upload expired or unknown — please re-upload and detect speakers again."
        )

    def run(cb: Callable[[int, str], None]) -> dict:
        try:
            for np_ in req.new_patients:
                _patients().upsert_patient(np_.patient_id, np_.display_name, np_.notes)
            diar = DiarizationResult.model_validate(pending["diarization"])
            _diar, profiles = analyze_wav_speakers(
                pending["path"],
                base_user_id=req.base_user_id or "anonymous",
                cfg=_cfg(),
                language=(req.language or "en").lower(),
                speaker_map={k: safe_user_id(v) for k, v in req.assignments.items() if v},
                save=req.save,
                diarization_result=diar,
                progress_cb=cb,
            )
            for profile in profiles:
                profile.source_filename = pending["filename"]
            return {
                "diarization": diar.model_dump(mode="json"),
                "profiles": [p.model_dump(mode="json") for p in profiles],
            }
        finally:
            _drop(pending["path"])

    return {"job_id": _start_job("analyze-speakers", run)}


# --- Patients --------------------------------------------------------------------

class PatientUpsert(BaseModel):
    display_name: str = ""
    notes: str = ""


@app.get("/api/patients")
def list_patients() -> dict:
    return {"patients": _patients().list_patients()}


@app.post("/api/patients/{patient_id}")
def upsert_patient(patient_id: str, body: PatientUpsert) -> dict:
    patient = _patients().upsert_patient(patient_id, body.display_name, body.notes)
    return patient.model_dump(mode="json")


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str) -> dict:
    store = _patients()
    patient = store.get_patient(patient_id)
    if patient is None:
        raise HTTPException(404, f"Patient not found: {patient_id}")
    profiles = store.profiles.load_profiles(patient.patient_id)
    return {
        "patient": patient.model_dump(mode="json"),
        "profiles": [p.model_dump(mode="json") for p in profiles],
        "trends": trend_series(profiles),
    }


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: str, delete_profiles: bool = False) -> dict:
    if delete_profiles:
        VoiceprintStore(_store().profile_dir).delete(patient_id)
    return {
        "deleted": _patients().delete_patient(patient_id, delete_profiles),
        "patient_id": patient_id,
    }


# --- Legacy per-user endpoints (kept for compatibility) ---------------------------

@app.get("/api/users")
def users() -> dict:
    return {"users": _store().list_users()}


@app.get("/api/users/{user_id}/profiles")
def profiles(user_id: str) -> dict:
    items = _store().load_profiles(user_id)
    return {
        "user_id": user_id,
        "count": len(items),
        "profiles": [p.model_dump(mode="json") for p in items],
    }


@app.get("/api/users/{user_id}/trends")
def trends(user_id: str) -> dict:
    return trend_series(_store().load_profiles(user_id))


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str) -> dict:
    return {"deleted": _store().delete_user(user_id), "user_id": user_id}


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """`mci-serve` entry point."""
    import sys

    import uvicorn

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
        if arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
