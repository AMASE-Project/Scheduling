"""FastAPI entry point for the AMASE scheduling web UI.

Lives at ``amase_scheduling.web`` (bundled with the ``amase-scheduling``
library, exposed via the ``amase-web`` console script).

Routes (DESIGN.md section 4):
    GET  /                                -> static frontend (app/static)
    POST /api/targets/parse               -> CSV validation with per-row errors
    GET  /api/targets/example             -> bundled example catalog (raw CSV)
    POST /api/schedule                    -> start async job, returns {job_id}
    GET  /api/schedule/{job_id}           -> poll status/progress/result
    POST /api/schedule/{job_id}/cancel    -> request cancellation
    GET  /api/schedule/{job_id}/download/{kind} -> blocks|targets|nights CSV
    POST /api/shutdown                    -> gracefully stop the server (Exit button)
"""

from __future__ import annotations

import re
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from amase_scheduling.observatory import NanshanObserver

from . import jobs, runner, targets_io
from . import tracks
from .serialize import (
    schedule_to_blocks_csv,
    schedule_to_nights_csv,
    schedule_to_targets_csv,
)

MAX_CAMPAIGN_NIGHTS = 31
DEFAULT_SEED = 2027

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI(title="AMASE Scheduling Web")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class TargetInput(BaseModel):
    """Loose model: real validation happens in ``targets_io.build_targets``
    (reusing the library rules) so failures are reported as 400, not 422."""

    name: str = ""
    ra: str = ""
    dec: str = ""
    priority: float | int | str | None = None
    exp_time: float | int | str | None = None
    n_dither: int | str | None = None
    n_set: int | str | None = None
    group: str | None = None


class ParseRequest(BaseModel):
    csv: str


class ScheduleRequest(BaseModel):
    targets: list[TargetInput]
    start: str
    end: str | None = None
    clear_prob: float = 1.0
    seed: int | None = DEFAULT_SEED
    time_limit: int = 60
    eps: float = 1e-3
    gamma: float = 0.01
    alpha: float = 0.5


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _parse_date(value: str | None, label: str) -> Time:
    if not value or not _DATE_RE.match(str(value).strip()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label} date {value!r}: expected YYYY-MM-DD format",
        )
    try:
        return Time(str(value).strip(), format="isot", scale="utc")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label} date {value!r}: not a valid calendar date",
        )


def _example_csv_path() -> Path:
    """Locate the bundled example catalog shipped inside this package."""
    return Path(__file__).resolve().parent / "data" / "example_targets.csv"


# --------------------------------------------------------------------------- #
# Target endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/targets/parse")
def parse_targets(req: ParseRequest):
    try:
        return targets_io.parse_csv_text(req.csv)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/targets/example")
def example_targets():
    path = _example_csv_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example target catalog not found")
    return Response(content=path.read_text(encoding="utf-8"), media_type="text/csv")


# --------------------------------------------------------------------------- #
# Schedule endpoints
# --------------------------------------------------------------------------- #
@app.post("/api/schedule", status_code=202)
def create_schedule(req: ScheduleRequest):
    if not req.targets:
        raise HTTPException(status_code=400, detail="At least one target is required")

    start_t = _parse_date(req.start, "start")
    end_value = req.end if req.end else req.start
    end_t = _parse_date(end_value, "end")

    if end_t < start_t:
        raise HTTPException(
            status_code=400,
            detail=f"end date {end_value!r} is before start date {req.start!r}",
        )

    n_nights = int(round((end_t - start_t).to(u.day).value)) + 1
    if n_nights > MAX_CAMPAIGN_NIGHTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Campaign is too long: {n_nights} nights exceeds the "
                f"{MAX_CAMPAIGN_NIGHTS}-night limit"
            ),
        )

    if not 0.0 <= req.clear_prob <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="clear_prob must be between 0 and 1",
        )
    if req.time_limit <= 0:
        raise HTTPException(
            status_code=400,
            detail="time_limit must be a positive number of seconds",
        )

    try:
        target_objs = targets_io.build_targets([t.model_dump() for t in req.targets])
    except targets_io.TargetValidationError as exc:
        reasons = "; ".join(f"row {e['line']}: {e['error']}" for e in exc.errors)
        raise HTTPException(status_code=400, detail=f"Invalid target(s): {reasons}")

    job = runner.start_job(
        targets=target_objs,
        start=req.start.strip(),
        end=end_value.strip(),
        clear_prob=req.clear_prob,
        seed=req.seed,
        eps=req.eps,
        gamma=req.gamma,
        alpha=req.alpha,
        time_limit=req.time_limit,
    )
    return {"job_id": job.id}


@app.get("/api/schedule/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id!r}")
    return jobs.snapshot(job)


@app.post("/api/schedule/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id!r}")
    status = jobs.request_cancel(job)
    return {"job_id": job_id, "status": status}


_KIND_CSV = {
    "blocks": schedule_to_blocks_csv,
    "targets": schedule_to_targets_csv,
    "nights": schedule_to_nights_csv,
}


@app.get("/api/schedule/{job_id}/download/{kind}")
def download_job(job_id: str, kind: str):
    if kind not in _KIND_CSV:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown download kind {kind!r}; expected one of blocks, targets, nights",
        )
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id!r}")
    if job.status != "done" or job.schedule is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id!r} has no finished result (status={job.status})",
        )
    csv_text = _KIND_CSV[kind](job.schedule)
    filename = f"amase_{kind}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/schedule/{job_id}/night/{night_idx}/tracks")
def night_tracks(job_id: str, night_idx: int):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job id {job_id!r}")
    if job.status != "done" or job.schedule is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id!r} has no finished result (status={job.status})",
        )
    nights = job.schedule.nights
    if night_idx < 0 or night_idx >= len(nights):
        raise HTTPException(
            status_code=404,
            detail=f"Night index {night_idx} out of range (0..{len(nights) - 1})",
        )
    observer = job.scheduler.observer if job.scheduler is not None else NanshanObserver()
    return tracks.night_tracks(nights[night_idx], job.targets or [], observer)


# --------------------------------------------------------------------------- #
# Server control
# --------------------------------------------------------------------------- #
def _terminate(delay: float = 0.3) -> None:
    """SIGTERM this process after a short delay.

    uvicorn traps SIGTERM and shuts down gracefully; the delay lets the
    HTTP response reach the browser first. Split out so tests can patch it.
    """
    import os
    import signal
    import threading

    threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()


@app.post("/api/shutdown")
def shutdown_server():
    _terminate()
    return {"status": "shutting down"}


# --------------------------------------------------------------------------- #
# Static frontend (mounted last so /api/* routes take precedence)
# --------------------------------------------------------------------------- #
_static_dir = Path(__file__).resolve().parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
