"""
FastAPI server for whisper-diarization.
Provides async job queue for processing audio files.
"""
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from diarize_core import run_diarization

app = FastAPI(
    title="Whisper Diarization API",
    description="Transcribe and diarize audio files with speaker identification",
    version="1.0.0",
)

# Job storage and queue
jobs: dict[str, "Job"] = {}
job_queue: Queue[str] = Queue()
jobs_lock = threading.Lock()

# Config
UPLOAD_DIR = "/tmp/diarization_uploads"
JOB_EXPIRY_SECONDS = 3600  # 1 hour
os.makedirs(UPLOAD_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    audio_path: str
    options: dict
    created_at: datetime = field(default_factory=datetime.now)
    progress: str | None = None
    result: dict | None = None
    error: str | None = None


def get_queue_position(job_id: str) -> int:
    """Get position in queue (0 = next to process)."""
    with jobs_lock:
        queued_jobs = [
            jid for jid, j in jobs.items()
            if j.status == JobStatus.QUEUED
        ]
        if job_id in queued_jobs:
            return queued_jobs.index(job_id)
        return -1


def cleanup_old_jobs():
    """Remove jobs older than JOB_EXPIRY_SECONDS."""
    now = datetime.now()
    with jobs_lock:
        expired = [
            jid for jid, j in jobs.items()
            if (now - j.created_at).total_seconds() > JOB_EXPIRY_SECONDS
            and j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]
        for jid in expired:
            job = jobs.pop(jid)
            if os.path.exists(job.audio_path):
                os.remove(job.audio_path)


def worker():
    """Background worker that processes jobs sequentially."""
    while True:
        job_id = job_queue.get()

        with jobs_lock:
            if job_id not in jobs:
                continue
            job = jobs[job_id]
            job.status = JobStatus.PROCESSING

        try:
            def progress_callback(stage: str):
                with jobs_lock:
                    if job_id in jobs:
                        jobs[job_id].progress = stage

            result = run_diarization(
                audio_path=job.audio_path,
                progress_callback=progress_callback,
                **job.options,
            )

            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].result = result
                    jobs[job_id].status = JobStatus.COMPLETED

        except Exception as e:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id].error = str(e)
                    jobs[job_id].status = JobStatus.FAILED

        finally:
            # Clean up audio file
            if os.path.exists(job.audio_path):
                os.remove(job.audio_path)

        # Periodically cleanup old jobs
        cleanup_old_jobs()


# Start worker thread on startup
@app.on_event("startup")
def startup_event():
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()


@app.post("/jobs", status_code=201)
async def submit_job(
    file: UploadFile = File(...),
    whisper_model: str = Form("medium.en"),
    language: str | None = Form(None),
    stemming: bool = Form(True),
    suppress_numerals: bool = Form(False),
    batch_size: int = Form(8),
):
    """
    Submit an audio file for diarization.
    Returns a job_id to poll for status.
    """
    # Generate unique job ID (acts as secret - hard to guess)
    job_id = str(uuid4())

    # Save uploaded file
    file_ext = os.path.splitext(file.filename or "audio")[1] or ".wav"
    audio_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")

    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create job
    job = Job(
        id=job_id,
        status=JobStatus.QUEUED,
        audio_path=audio_path,
        options={
            "model_name": whisper_model,
            "language": language if language else None,
            "stemming": stemming,
            "suppress_numerals": suppress_numerals,
            "batch_size": batch_size,
        },
    )

    with jobs_lock:
        jobs[job_id] = job

    # Add to queue
    job_queue.put(job_id)

    position = get_queue_position(job_id)

    return {
        "job_id": job_id,
        "status": job.status.value,
        "position": position,
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a diarization job.
    """
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]

        response = {
            "job_id": job.id,
            "status": job.status.value,
        }

        if job.status == JobStatus.QUEUED:
            response["position"] = get_queue_position(job_id)
        elif job.status == JobStatus.PROCESSING:
            response["progress"] = job.progress
        elif job.status == JobStatus.FAILED:
            response["error"] = job.error

        return response


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """
    Get the result of a completed diarization job.
    """
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]

        if job.status == JobStatus.QUEUED:
            raise HTTPException(status_code=202, detail="Job is still queued")
        elif job.status == JobStatus.PROCESSING:
            raise HTTPException(status_code=202, detail="Job is still processing")
        elif job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")

        return job.result


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its results.
    Can be used to cancel a queued job or cleanup a completed job.
    """
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]

        # Can only delete queued, completed, or failed jobs
        if job.status == JobStatus.PROCESSING:
            raise HTTPException(
                status_code=409,
                detail="Cannot delete job while processing"
            )

        # Clean up audio file if exists
        if os.path.exists(job.audio_path):
            os.remove(job.audio_path)

        del jobs[job_id]

        return {"message": "Job deleted"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    with jobs_lock:
        queued = sum(1 for j in jobs.values() if j.status == JobStatus.QUEUED)
        processing = sum(1 for j in jobs.values() if j.status == JobStatus.PROCESSING)

    return {
        "status": "healthy",
        "queued_jobs": queued,
        "processing_jobs": processing,
    }
