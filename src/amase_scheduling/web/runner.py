"""Background-thread job runner.

Runs ``Scheduler().schedule(...)`` in a daemon thread, streaming progress back
to the job object via the library's ``progress_callback`` hook. Cancellation is
cooperative: the callback raises ``CancelledError`` (checked at every night
boundary), which aborts the run and marks the job ``cancelled``.
"""

from __future__ import annotations

import threading

from amase_scheduling import Scheduler

from . import jobs
from .serialize import schedule_to_json


class CancelledError(Exception):
    """Raised inside the progress callback to abort a running job."""


def run_schedule_job(
    job: jobs.Job,
    targets: list,
    start: str,
    end: str,
    clear_prob: float,
    seed: int | None,
    eps: float,
    gamma: float,
    alpha: float,
    time_limit: int,
) -> None:
    def progress_callback(night_idx: int, n_nights: int, date_str: str) -> None:
        if jobs.is_cancelled(job):
            raise CancelledError(f"cancelled during night {date_str}")
        jobs.set_progress(job, night_idx, n_nights, date_str)

    try:
        scheduler = Scheduler()
        job.scheduler = scheduler  # retain so tracks can reuse the same observer
        schedule = scheduler.schedule(
            targets=targets,
            start=start,
            end=end,
            clear_prob=clear_prob,
            seed=seed,
            eps=eps,
            gamma=gamma,
            alpha=alpha,
            time_limit=time_limit,
            progress_callback=progress_callback,
        )
    except CancelledError:
        jobs.set_cancelled(job)
        return
    except Exception as exc:  # noqa: BLE001 - surface any solver error to the UI
        jobs.set_error(job, f"{type(exc).__name__}: {exc}")
        return

    jobs.set_done(job, schedule, schedule_to_json(schedule, targets))


def start_job(
    targets: list,
    start: str,
    end: str,
    clear_prob: float,
    seed: int | None,
    eps: float,
    gamma: float,
    alpha: float,
    time_limit: int,
) -> jobs.Job:
    """Create a job and launch the scheduler in a background thread."""
    job = jobs.create_job()
    job.targets = targets  # retain for the per-night altitude-tracks endpoint
    thread = threading.Thread(
        target=run_schedule_job,
        args=(job, targets, start, end, clear_prob, seed, eps, gamma, alpha, time_limit),
        daemon=True,
    )
    job.thread = thread
    thread.start()
    return job
