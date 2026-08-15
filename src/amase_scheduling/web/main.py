"""FastAPI entry point for the AMASE scheduling web UI.

Lives at ``amase_scheduling.web`` (bundled with the ``amase-scheduling``
library, exposed via the ``amase-web`` console script).

Routes (DESIGN.md section 4):
    GET  /                                -> static frontend (app/static)
    POST /api/targets/parse               -> CSV validation with per-row errors
    GET  /api/targets/example             -> bundled example catalog (raw CSV)
    POST /api/cache/load                  -> load a precomputed visibility cache
    POST /api/cache/upload                -> upload a visibility cache file, returns {cache_id}
    POST /api/schedule                    -> start async job, returns {job_id}
    GET  /api/schedule/{job_id}           -> poll status/progress/result
    POST /api/schedule/{job_id}/cancel    -> request cancellation
    GET  /api/schedule/{job_id}/download/{kind} -> blocks|targets|nights CSV
    POST /api/shutdown                    -> gracefully stop the server (Exit button)
"""

from __future__ import annotations

import re
import uuid
import warnings
from pathlib import Path

import astropy.units as u
from astropy.time import Time, TimeDelta
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from amase_scheduling.cache import VisibilityCache
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
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB cap on uploaded cache files

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


class CacheLoadRequest(BaseModel):
    path: str


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
    cache_path: str | None = None
    cache_id: str | None = None


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
# Visibility cache loading (memoized; a rebuilt file reloads via mtime)
# --------------------------------------------------------------------------- #
#: In-memory cache keyed by "<resolved path>::<mtime_ns>" so that a re-saved
#: (rebuilt) file is reloaded while identical repeated requests are served from
#: memory. Single-user tool; the process-lifetime memo is intentional.
_CACHE_MEMO: dict[str, VisibilityCache] = {}

#: Uploaded caches keyed by ``cache_id``, plus the original filename. In-memory
#: only: a server restart clears them (intended for a local single-user tool).
_UPLOADED_CACHES: dict[str, VisibilityCache] = {}
_UPLOADED_FILENAMES: dict[str, str] = {}


def _resolve_cache_path(path_str: str) -> Path:
    """Expand ``~`` and resolve a relative path against the CWD."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _load_cache(path_str: str) -> VisibilityCache:
    """Load a VisibilityCache from ``path_str`` (memoized on path + mtime).

    Raises ``HTTPException`` (400) with a clear detail on file not found,
    unreadable file, or np.load failure.
    """
    path = _resolve_cache_path(path_str)
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Visibility cache file not found: {path}",
        )
    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Visibility cache path is not a file: {path}",
        )
    try:
        mtime = path.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Visibility cache file not readable: {path} ({exc})",
        )

    key = f"{path}::{mtime}"
    if key in _CACHE_MEMO:
        return _CACHE_MEMO[key]

    try:
        cache = VisibilityCache.load(str(path))
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Visibility cache file not readable: {path} ({exc})",
        )
    except Exception as exc:  # noqa: BLE001 - surface np.load failures as 400
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load visibility cache {path}: {exc}",
        )

    _CACHE_MEMO[key] = cache
    return cache


def _resolve_and_validate_cache(
    req: ScheduleRequest,
    target_objs: list,
    start_t: Time,
    n_nights: int,
    start_str: str,
    end_str: str,
) -> VisibilityCache | None:
    """Resolve the schedule's visibility cache from ``cache_path`` or
    ``cache_id`` and validate it against the targets and requested dates.

    Returns ``None`` when no cache is requested. Raises ``HTTPException`` (400)
    on conflicting cache selectors, an unknown ``cache_id``, a target-list or
    exposure-time mismatch, or a requested night missing from the cache.
    """
    if req.cache_id and req.cache_path:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of cache_id or cache_path",
        )

    cache: VisibilityCache | None = None
    label: str = ""
    if req.cache_path:
        cache = _load_cache(req.cache_path)
        label = f"{req.cache_path!r}"
    elif req.cache_id:
        cache = _UPLOADED_CACHES.get(req.cache_id)
        if cache is None:
            raise HTTPException(
                status_code=400,
                detail="unknown or expired cache_id (re-upload the cache)",
            )
        label = f"cache_id {req.cache_id!r}"

    if cache is None:
        return None

    with warnings.catch_warnings():
        # A pre-fingerprint cache emits a warning; validation still passes
        # on names alone, so ignore the warning rather than fail.
        warnings.simplefilter("ignore")
        try:
            cache.validate(target_objs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    missing = []
    for i in range(n_nights):
        date_str = (start_t + TimeDelta(i * 86400, format="sec")).isot[:10]
        if date_str not in cache.nights:
            missing.append(date_str)
    if missing:
        cache_min = min(cache.nights)
        cache_max = max(cache.nights)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Visibility cache {label} covers "
                f"{cache_min} .. {cache_max} but does not include "
                f"{missing[0]} (requested {start_str} .. {end_str})"
            ),
        )

    return cache


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
@app.post("/api/cache/load")
def load_visibility_cache(req: CacheLoadRequest):
    cache = _load_cache(req.path)
    dates = sorted(cache.nights.keys())
    return {
        "path": str(_resolve_cache_path(req.path)),
        "n_nights": len(cache),
        "start": min(dates),
        "end": max(dates),
        "n_targets": len(cache.target_names),
        "target_names": cache.target_names,
    }


@app.post("/api/cache/upload")
async def upload_visibility_cache(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".npz"):
        raise HTTPException(
            status_code=400,
            detail=f"Cache file must end with .npz (got {filename!r})",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded cache file is empty",
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cache file is {len(data)} bytes; maximum allowed is "
                f"{MAX_UPLOAD_BYTES} bytes (200 MB)"
            ),
        )

    try:
        cache = VisibilityCache.load_bytes(data)
    except Exception as exc:  # noqa: BLE001 - surface invalid npz as 400
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load visibility cache from upload: {exc}",
        )

    cache_id = uuid.uuid4().hex[:12]
    _UPLOADED_CACHES[cache_id] = cache
    _UPLOADED_FILENAMES[cache_id] = filename

    dates = sorted(cache.nights.keys())
    return {
        "cache_id": cache_id,
        "filename": filename,
        "n_nights": len(cache),
        "start": min(dates),
        "end": max(dates),
        "n_targets": len(cache.target_names),
        "target_names": cache.target_names,
    }


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

    visibility_cache = _resolve_and_validate_cache(
        req,
        target_objs,
        start_t,
        n_nights,
        req.start.strip(),
        end_value.strip(),
    )

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
        visibility_cache=visibility_cache,
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
