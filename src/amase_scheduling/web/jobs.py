"""In-memory job registry.

A job tracks the lifecycle of a single scheduling run:

    running -> done | error | cancelled

The registry is a plain process-memory dict keyed by ``job_id``. Jobs are not
persisted; results are lost when the process exits (per DESIGN.md section 1).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    status: str = "running"  # running | done | error | cancelled
    progress: dict | None = None  # {"night_idx", "n_nights", "date"}
    result: dict | None = None  # serialized schedule JSON (see serialize.py)
    schedule: object | None = None  # raw Schedule object, used for CSV downloads
    targets: list | None = None  # in-memory Target list, used for altitude tracks
    scheduler: object | None = None  # Scheduler instance (exposes .observer)
    error: str | None = None
    cancelled: bool = False
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_JOBS: dict[str, Job] = {}


def create_job() -> Job:
    """Create and register a new job in the ``running`` state."""
    job = Job(id=uuid.uuid4().hex)
    _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def set_progress(job: Job, night_idx: int, n_nights: int, date: str) -> None:
    with job.lock:
        job.progress = {
            "night_idx": night_idx,
            "n_nights": n_nights,
            "date": date,
        }


def set_done(job: Job, schedule: object, result: dict) -> None:
    with job.lock:
        job.schedule = schedule
        job.result = result
        job.status = "done"


def set_error(job: Job, error: str) -> None:
    with job.lock:
        job.error = error
        job.status = "error"


def set_cancelled(job: Job) -> None:
    with job.lock:
        job.status = "cancelled"


def request_cancel(job: Job) -> str:
    """Set the cancellation flag and return the current status.

    The flag is checked inside the scheduler's progress callback; the worker
    thread aborts on the next night boundary.
    """
    with job.lock:
        job.cancelled = True
        return job.status


def is_cancelled(job: Job) -> bool:
    with job.lock:
        return job.cancelled


def snapshot(job: Job) -> dict:
    """Poll payload matching DESIGN.md section 4."""
    with job.lock:
        return {
            "status": job.status,
            "progress": job.progress,
            "result": job.result,
            "error": job.error,
        }
