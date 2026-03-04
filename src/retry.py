"""
retry.py — Retry and recovery logic for failed transcription jobs.

Strategy: exponential backoff with a max attempt cap.
We don't retry indefinitely — after 3 failures the job is marked
permanently failed so it doesn't silently consume resources forever.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 2  # doubles each attempt: 2s, 4s, 8s


def with_retry(func: Callable) -> Callable:
    """
    Decorator: retries the wrapped function up to MAX_RETRIES times
    using exponential backoff. Re-raises on final failure.

    Usage:
        @with_retry
        def run_transcription(job_id, audio_path):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY_SECONDS ** attempt
                    logger.warning(
                        f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {MAX_RETRIES} attempts failed for {func.__name__}")
        raise last_exc

    return wrapper


def recover_stalled_jobs(job_store: dict, timeout_seconds: int = 300) -> list[str]:
    """
    Find jobs stuck in 'processing' for longer than timeout_seconds.
    In a real system this would be a scheduled cron task querying
    a database for jobs with updated_at < now - timeout.

    Returns list of job IDs that were reset to 'queued' for retry.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    recovered = []

    for job_id, job in job_store.items():
        if job.get("status") != "processing":
            continue

        created_at_str = job.get("created_at")
        if not created_at_str:
            continue

        created_at = datetime.fromisoformat(created_at_str)
        age_seconds = (now - created_at).total_seconds()

        if age_seconds > timeout_seconds:
            logger.warning(f"Job {job_id} appears stalled ({age_seconds:.0f}s). Re-queuing.")
            job["status"] = "queued"
            job["retry_count"] = job.get("retry_count", 0) + 1
            recovered.append(job_id)

    return recovered
