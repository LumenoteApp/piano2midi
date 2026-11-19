"""Local web UI: python server.py  ->  http://127.0.0.1:8770"""

from __future__ import annotations

import pickle
import re
import shutil
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import engine

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

app = FastAPI(title="Piano2MIDI")
jobs: dict[str, dict[str, Any]] = {}
gpu_lock = threading.Lock()  # one heavy job at a time on the GPU


class CleanupBody(BaseModel):
    min_velocity: int = 0
    min_duration: float = 0.0
    sustain_pedal: bool = True
    other_pedals: bool = True
    velocity_scale: float = 1.0
    quantize_pedal: float = 0.0


def _job_dir(job_id: str) -> Path:
    return OUT / job_id


def _log(job: dict, msg: str) -> None:
    job["log"].append(msg)
    print(f"[{job['id'][:8]}] {msg}", flush=True)


def _finish(job: dict, cleanup: CleanupBody) -> None:
    raw: engine.Transcription = job["raw"]
    t = engine.clean(raw, engine.Cleanup(**cleanup.model_dump()))
    stem = Path(job["name"]).stem
    files = engine.export(t, _job_dir(job["id"]) / stem)
    job["files"] = {k: str(v) for k, v in files.items()}
    job["result"] = engine.to_json(t)


def _run(job: dict, src: Path | None, extract: bool, cleanup: CleanupBody,
         url: str | None = None) -> None:
    try:
        if url:
            job["status"] = "running"
            job["stage"] = "Downloading"
            src, title = engine.download_audio(url, _job_dir(job["id"]), log=lambda m: _log(job, m))
            job["name"] = title + ".mp3"
            job["source"] = str(src)
        job["status"] = "queued"
        with gpu_lock:
            job["status"] = "running"
            audio = src
            if extract:
                job["stage"] = "Separating piano"
                piano = _job_dir(job["id"]) / "piano.wav"
                audio = engine.separate_piano(src, piano, log=lambda m: _log(job, m))
                job["stem"] = str(piano)
            job["stage"] = "Transcribing"
            job["raw"] = engine.transcribe(audio, log=lambda m: _log(job, m))
        with open(_job_dir(job["id"]) / "raw.pkl", "wb") as fh:
            pickle.dump(job["raw"], fh)
        job["stage"] = "Writing MIDI"
        _finish(job, cleanup)
        job["status"] = "done"
        _log(job, "Done.")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile | None = File(None), url: str = Form(""),
                         extract: bool = Form(False), cleanup: str = Form("{}")) -> dict:
    url = url.strip()
    if not file and not url:
        raise HTTPException(400, "Send a file or a link.")
    if url and not re.match(r"^https?://\S+$", url):
        raise HTTPException(400, "That doesn't look like a link.")
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    d.mkdir(parents=True)
    src = None
    if file:
        name = Path(file.filename or "audio").name
        src = d / ("source" + Path(name).suffix.lower())
        with open(src, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    else:
        name = "YouTube audio"
    job = {"id": job_id, "name": name, "status": "queued", "stage": "Queued", "log": [],
           "source": str(src) if src else None}
    jobs[job_id] = job
    c = CleanupBody.model_validate_json(cleanup or "{}")
    threading.Thread(target=_run, args=(job, src, extract, c, url or None), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    out = {k: job.get(k) for k in ("id", "name", "status", "stage", "log", "error")}
    out["has_stem"] = bool(job.get("stem"))
    if job["status"] == "done":
        out["result"] = job["result"]
    return out


@app.post("/api/jobs/{job_id}/cleanup")
def api_cleanup(job_id: str, body: CleanupBody) -> dict:
    job = jobs.get(job_id)
    if not job or "raw" not in job:
        raise HTTPException(404, "Job not ready")
    _finish(job, body)
    return {"result": job["result"]}


@app.get("/api/jobs/{job_id}/file/{kind}")
def api_file(job_id: str, kind: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    stem = Path(job["name"]).stem
    if kind in ("keys", "sustain") and job.get("files"):
        return FileResponse(job["files"][kind], media_type="audio/midi",
                            filename=f"{stem}_{kind}.mid")
    if kind == "stem" and job.get("stem"):
        return FileResponse(job["stem"], media_type="audio/wav", filename=f"{stem}_piano.wav")
    if kind == "source" and job.get("source"):
        return FileResponse(job["source"])
    raise HTTPException(404)


@app.get("/api/health")
def health() -> dict:
    import torch
    return {"cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    threading.Thread(target=engine.get_model, daemon=True).start()  # warm up
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, ["http://127.0.0.1:8770"]).start()
    print("Piano2MIDI running at http://127.0.0.1:8770  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8770, log_level="warning")
