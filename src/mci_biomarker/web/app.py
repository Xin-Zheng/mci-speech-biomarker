"""FastAPI web UI (SPEC Layers 1 + 4): upload → biomarker report + trends.

Zero-retention: the uploaded file is written to a temp path, analyzed, and
deleted in a `finally` block; only the JSON profile persists (when saved).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..baseline.store import ProfileStore, trend_series
from ..config import PipelineConfig
from ..pipeline import analyze_wav

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_MB = 50

app = FastAPI(title="MCI Speech Biomarker", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _store() -> ProfileStore:
    return ProfileStore(PipelineConfig().baseline.profile_dir)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "mci-biomarker"}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    transcript: str = Form(""),
    language: str = Form("en"),
    save: bool = Form(True),
) -> JSONResponse:
    name = (file.filename or "upload.wav").lower()
    if not name.endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "Please upload an audio file (.wav recommended).")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    suffix = Path(name).suffix or ".wav"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        cfg = PipelineConfig()
        cfg.baseline.profile_dir = _store().profile_dir
        profile = analyze_wav(
            tmp_path,
            user_id=user_id or "anonymous",
            cfg=cfg,
            transcript_override=transcript.strip() or None,
            language=(language or "en").lower(),
            save=save,
        )
        profile.source_filename = file.filename or profile.source_filename
        return JSONResponse(profile.model_dump(mode="json"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Analysis failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)  # zero-retention (SPEC §4)


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
