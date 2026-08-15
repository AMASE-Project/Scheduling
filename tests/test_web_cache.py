"""Tests for POST /api/cache/load and cache_path support in POST /api/schedule."""

import csv
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from amase_scheduling.web.main import app, _example_csv_path

client = TestClient(app)

REPO = Path(__file__).resolve().parent.parent
EXAMPLE_CACHE = REPO / "example" / "vis_cache.npz"


def _example_targets_payload() -> list[dict]:
    """All 42 shipped example targets as ScheduleRequest target dicts."""
    with open(_example_csv_path(), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    payload = []
    for r in rows:
        payload.append(
            {
                "name": r["name"],
                "ra": r["ra"],
                "dec": r["dec"],
                "priority": float(r["priority"]),
                "exp_time": float(r["exp_time"]),
                "n_dither": int(r["n_dither"]),
                "n_set": int(r["n_set"]),
                "group": (r.get("group") or "").strip(),
            }
        )
    return payload


def _wait_for_terminal(job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/schedule/{job_id}").json()
        if data["status"] in ("done", "error", "cancelled"):
            return data
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not reach a terminal state in time")


# --------------------------------------------------------------------------- #
# /api/cache/load
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EXAMPLE_CACHE.exists(), reason="shipped cache missing")
def test_cache_load_endpoint_returns_metadata():
    r = client.post("/api/cache/load", json={"path": str(EXAMPLE_CACHE)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["path"] == str(EXAMPLE_CACHE.resolve())
    assert d["n_nights"] == 732
    assert d["start"] == "2027-04-01"
    assert d["end"] == "2029-04-01"
    assert d["n_targets"] == 42
    assert len(d["target_names"]) == 42


def test_cache_load_bad_path():
    r = client.post("/api/cache/load", json={"path": "/nonexistent/vis_cache.npz"})
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_cache_load_not_a_npz(tmp_path):
    bogus = tmp_path / "bogus.npz"
    bogus.write_text("not a numpy file")
    r = client.post("/api/cache/load", json={"path": str(bogus)})
    assert r.status_code == 400
    assert "Failed to load" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# /api/cache/upload
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EXAMPLE_CACHE.exists(), reason="shipped cache missing")
def test_cache_upload_endpoint():
    data = EXAMPLE_CACHE.read_bytes()
    r = client.post(
        "/api/cache/upload",
        files={"file": ("vis_cache.npz", data, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["filename"] == "vis_cache.npz"
    assert d["n_nights"] == 732
    assert d["start"] == "2027-04-01"
    assert d["end"] == "2029-04-01"
    assert d["n_targets"] == 42
    assert len(d["target_names"]) == 42
    assert d["cache_id"]
    assert len(d["cache_id"]) == 12


def test_cache_upload_garbage_bytes():
    r = client.post(
        "/api/cache/upload",
        files={"file": ("bad.npz", b"not a numpy archive", "application/octet-stream")},
    )
    assert r.status_code == 400, r.text
    assert "Failed to load" in r.json()["detail"]


def test_cache_upload_wrong_extension():
    r = client.post(
        "/api/cache/upload",
        files={"file": ("bad.txt", b"whatever", "application/octet-stream")},
    )
    assert r.status_code == 400, r.text
    assert ".npz" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# /api/schedule with cache_path
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EXAMPLE_CACHE.exists(), reason="shipped cache missing")
def test_schedule_with_cache_out_of_coverage():
    payload = {
        "targets": _example_targets_payload(),
        "start": "2030-01-01",
        "end": "2030-01-01",
        "cache_path": str(EXAMPLE_CACHE),
    }
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "covers" in detail
    assert "2027-04-01" in detail
    assert "2030-01-01" in detail


@pytest.mark.skipif(not EXAMPLE_CACHE.exists(), reason="shipped cache missing")
def test_schedule_with_cache_happy_path():
    payload = {
        "targets": _example_targets_payload(),
        "start": "2027-05-01",
        "end": "2027-05-01",
        "clear_prob": 1.0,
        "seed": 2027,
        "time_limit": 30,
        "cache_path": str(EXAMPLE_CACHE),
    }
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    data = _wait_for_terminal(job_id)
    assert data["status"] == "done", data
    assert data["error"] is None
    assert data["result"]["summary"]["n_nights"] == 1


# --------------------------------------------------------------------------- #
# /api/schedule with cache_id (uploaded cache)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EXAMPLE_CACHE.exists(), reason="shipped cache missing")
def test_schedule_with_cache_id_happy_path():
    up = client.post(
        "/api/cache/upload",
        files={
            "file": (
                "vis_cache.npz",
                EXAMPLE_CACHE.read_bytes(),
                "application/octet-stream",
            )
        },
    )
    cache_id = up.json()["cache_id"]

    payload = {
        "targets": _example_targets_payload(),
        "start": "2027-05-01",
        "end": "2027-05-01",
        "clear_prob": 1.0,
        "seed": 2027,
        "time_limit": 30,
        "cache_id": cache_id,
    }
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    data = _wait_for_terminal(job_id)
    assert data["status"] == "done", data
    assert data["error"] is None
    assert data["result"]["summary"]["n_nights"] == 1


def test_schedule_with_unknown_cache_id():
    payload = {
        "targets": _example_targets_payload(),
        "start": "2027-05-01",
        "end": "2027-05-01",
        "cache_id": "deadbeef0000",
    }
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 400, r.text
    assert "unknown or expired cache_id" in r.json()["detail"]


def test_schedule_with_both_cache_id_and_path():
    payload = {
        "targets": _example_targets_payload(),
        "start": "2027-05-01",
        "end": "2027-05-01",
        "cache_id": "deadbeef0000",
        "cache_path": str(EXAMPLE_CACHE),
    }
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 400, r.text
    assert "only one" in r.json()["detail"]
