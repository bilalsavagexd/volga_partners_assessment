"""
api.py — FastAPI wrapper around the transcription pipeline.

Design decisions:
- Async endpoint with background task for actual transcription
  so uploads return immediately instead of blocking for minutes
- Jobs stored in Redis (or in-memory dict for local dev)
- S3 for audio/transcript storage; local filesystem fallback
"""

import os
import uuid
import json
import logging
import asyncio
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# In production replace with Redis-backed store
JOB_STORE: dict[str, dict] = {}

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transcription API",
    description="Audio-to-text pipeline with timestamped segments",
    version="1.0.0",
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _save_upload(upload: UploadFile) -> str:
    """Persist the uploaded file to a temp location and return the path."""
    suffix = os.path.splitext(upload.filename or "audio")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.close()
    return tmp.name


def _run_transcription(job_id: str, audio_path: str) -> None:
    """
    Blocking transcription worker — runs in a thread pool via asyncio.
    Updates JOB_STORE in place so the status endpoint can poll it.
    """
    try:
        JOB_STORE[job_id]["status"] = "processing"

        from pipeline import TranscriptionPipeline
        pipeline = TranscriptionPipeline()
        segments = pipeline.run(audio_path)

        # In production, upload transcript JSON to S3 here
        JOB_STORE[job_id].update(
            {
                "status": "completed",
                "segments": segments,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "segment_count": len(segments),
            }
        )
        logger.info(f"Job {job_id} completed — {len(segments)} segments")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        JOB_STORE[job_id].update(
            {
                "status": "failed",
                "error": str(exc),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    finally:
        # Clean up temp file regardless of outcome
        if os.path.exists(audio_path):
            os.unlink(audio_path)


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.post("/transcribe", status_code=202)
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload an audio file and get back a job ID immediately.
    Poll /jobs/{job_id} to check progress.

    Returns 202 Accepted — transcription runs async so large files
    don't time out the HTTP connection.
    """
    job_id = str(uuid.uuid4())
    audio_path = _save_upload(file)

    JOB_STORE[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Run transcription in a thread so we don't block the event loop
    background_tasks.add_task(
        asyncio.get_event_loop().run_in_executor,
        None,  # uses default ThreadPoolExecutor
        _run_transcription,
        job_id,
        audio_path,
    )

    logger.info(f"Job {job_id} queued for {file.filename}")
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Poll this endpoint to check job status.
    Returns segments once status == 'completed'.
    """
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": len(JOB_STORE)}
