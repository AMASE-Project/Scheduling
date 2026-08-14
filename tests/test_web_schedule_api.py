"""Tests for the async schedule job lifecycle, cancellation, downloads,
validation, and JSON serialization."""

import re
import time

from astropy.coordinates import SkyCoord
from astropy.time import Time
from fastapi.testclient import TestClient

from amase_scheduling.scheduler import (
    NightPlan,
    Schedule,
    ScheduledBlock,
    TargetProgress,
)

from amase_scheduling.web.main import app
from amase_scheduling.web.serialize import schedule_to_json

client = TestClient(app)

TWO_TARGETS = [
    {
        "name": "NGC4258",
        "ra": "184.739583",
        "dec": "47.303972",
        "priority": 1,
        "exp_time": 600,
        "n_dither": 9,
        "n_set": 1,
    },
    {
        "name": "M51",
        "ra": "202.462917",
        "dec": "47.409444",
        "priority": 1,
        "exp_time": 300,
        "n_dither": 3,
        "n_set": 1,
    },
]

THREE_TARGETS = [
    {
        "name": "NGC4258",
        "ra": "184.739583",
        "dec": "47.303972",
        "priority": 1,
        "exp_time": 600,
        "n_dither": 9,
        "n_set": 1,
    },
    {
        "name": "M51",
        "ra": "202.462917",
        "dec": "47.409444",
        "priority": 1,
        "exp_time": 300,
        "n_dither": 3,
        "n_set": 1,
    },
    {
        "name": "M82",
        "ra": "148.969583",
        "dec": "69.679694",
        "priority": 1,
        "exp_time": 300,
        "n_dither": 3,
        "n_set": 1,
    },
]


def _start_job(start: str, end: str, **overrides) -> str:
    payload = {
        "targets": TWO_TARGETS,
        "start": start,
        "end": end,
        "clear_prob": 1.0,
        "seed": 2027,
        "time_limit": 30,
    }
    payload.update(overrides)
    r = client.post("/api/schedule", json=payload)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


def _wait_for_terminal(job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/schedule/{job_id}").json()
        if data["status"] in ("done", "error", "cancelled"):
            return data
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach a terminal state in time")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_schedule_rejects_bad_date_format():
    r = client.post(
        "/api/schedule",
        json={"targets": TWO_TARGETS, "start": "2027/04/01", "end": "2027-04-01"},
    )
    assert r.status_code == 400
    assert "YYYY-MM-DD" in r.json()["detail"]


def test_schedule_rejects_inverted_range():
    r = client.post(
        "/api/schedule",
        json={"targets": TWO_TARGETS, "start": "2027-04-05", "end": "2027-04-01"},
    )
    assert r.status_code == 400
    assert "before start date" in r.json()["detail"]


def test_schedule_rejects_too_long_campaign():
    r = client.post(
        "/api/schedule",
        json={"targets": TWO_TARGETS, "start": "2027-04-01", "end": "2027-06-01"},
    )
    assert r.status_code == 400
    assert "night limit" in r.json()["detail"]


def test_schedule_rejects_invalid_target():
    bad = dict(TWO_TARGETS[0], exp_time=99999)
    r = client.post(
        "/api/schedule",
        json={"targets": [bad], "start": "2027-04-01", "end": "2027-04-01"},
    )
    assert r.status_code == 400
    assert "Invalid target" in r.json()["detail"]
    assert "exp_time must be in" in r.json()["detail"]


def test_schedule_rejects_empty_targets():
    r = client.post(
        "/api/schedule",
        json={"targets": [], "start": "2027-04-01", "end": "2027-04-01"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Lifecycle + downloads
# --------------------------------------------------------------------------- #
def test_schedule_lifecycle_and_downloads():
    job_id = _start_job("2027-04-01", "2027-04-01")

    # Polling should show running with progress while in flight (or done).
    data = _wait_for_terminal(job_id)
    assert data["status"] == "done", data
    assert data["error"] is None

    result = data["result"]
    assert set(result) == {
        "start_date", "end_date", "clear_prob", "seed",
        "capacity_warning", "summary", "nights", "progress",
    }
    assert result["start_date"] == "2027-04-01"
    assert result["end_date"] == "2027-04-01"
    assert result["seed"] == 2027
    assert result["clear_prob"] == 1.0

    summary = result["summary"]
    assert summary["n_nights"] == 1
    assert summary["n_clear"] == 1
    assert summary["n_completed"] == len(TWO_TARGETS)

    assert len(result["nights"]) == 1
    night = result["nights"][0]
    assert night["date"] == "2027-04-01"
    assert night["clear"] is True
    assert night["night_start_utc"] is not None
    assert night["night_end_utc"] is not None
    for block in night["blocks"]:
        assert set(block) == {
            "target", "exposure", "start_utc", "end_utc",
            "altitude_deg", "azimuth_deg", "moon_sep_deg",
        }
        assert block["start_utc"].startswith("2027-04-01T")
        assert block["end_utc"].startswith("2027-04-01T")

    assert len(result["progress"]) == len(TWO_TARGETS)
    for p in result["progress"]:
        assert set(p) == {
            "target", "required", "done", "fraction",
            "nights_observed", "obs_time_min",
            "ra_deg", "dec_deg", "group", "required_hours",
        }
        assert p["done"] == p["required"]
        assert p["fraction"] == 1.0
        assert -360.0 <= p["ra_deg"] <= 360.0
        assert -90.0 <= p["dec_deg"] <= 90.0
        assert p["required_hours"] > 0

    # Downloads
    blocks = client.get(f"/api/schedule/{job_id}/download/blocks")
    assert blocks.status_code == 200
    assert blocks.headers["content-type"].startswith("text/csv")
    assert "date,target,exposure,obs_start_utc" in blocks.text

    targets = client.get(f"/api/schedule/{job_id}/download/targets")
    assert targets.status_code == 200
    assert "target,required,done,fraction,nights_observed,obs_hours" in targets.text
    assert "NGC4258" in targets.text

    nights = client.get(f"/api/schedule/{job_id}/download/nights")
    assert nights.status_code == 200
    assert "date,clear,night_start_utc,night_end_utc" in nights.text


def test_download_rejects_unknown_kind_and_unknown_job():
    job_id = _start_job("2027-04-01", "2027-04-01")
    _wait_for_terminal(job_id)

    r = client.get(f"/api/schedule/{job_id}/download/bogus")
    assert r.status_code == 400

    r = client.get("/api/schedule/doesnotexist/download/blocks")
    assert r.status_code == 404

    r = client.get("/api/schedule/doesnotexist")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #
def test_cancel_job():
    job_id = _start_job("2027-04-01", "2027-04-03")  # multi-night so cancel lands

    cancel = client.post(f"/api/schedule/{job_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["job_id"] == job_id

    data = _wait_for_terminal(job_id)
    assert data["status"] == "cancelled", data
    assert data["result"] is None


# --------------------------------------------------------------------------- #
# Serialization round-trip sanity
# --------------------------------------------------------------------------- #
def test_serialize_roundtrip():
    block = ScheduledBlock(
        target_name="NGC4258",
        target_index=0,
        target_coord=SkyCoord("12h18m57s +47d18m14s"),
        exposure=1,
        start_time=Time("2027-04-01T18:00:00", format="isot"),
        end_time=Time("2027-04-01T18:05:00", format="isot"),
        altitude=55.2,
        azimuth=120.1,
        moon_separation=80.0,
    )
    night = NightPlan(
        date="2027-04-01",
        clear=True,
        night_start=Time("2027-04-01T17:00:00", format="isot"),
        night_end=Time("2027-04-01T23:00:00", format="isot"),
        blocks=[block],
    )
    progress = TargetProgress(
        name="NGC4258", required=9, done=9, nights_observed=2, obs_time_min=135.0
    )
    schedule = Schedule(
        start_date="2027-04-01",
        end_date="2027-04-01",
        clear_prob=1.0,
        seed=2027,
        nights=[night],
        progress=[progress],
    )

    d = schedule_to_json(schedule)
    assert d["summary"]["n_nights"] == 1
    assert d["summary"]["n_completed"] == 1
    assert d["summary"]["total_obs_min"] == 135.0
    assert d["nights"][0]["blocks"][0]["target"] == "NGC4258"
    assert d["nights"][0]["blocks"][0]["start_utc"] == "2027-04-01T18:00:00.000"
    assert d["progress"][0]["fraction"] == 1.0
    assert d["progress"][0]["obs_time_min"] == 135.0


# --------------------------------------------------------------------------- #
# Per-night altitude tracks
# --------------------------------------------------------------------------- #
def test_night_tracks_shape_and_data():
    job_id = _start_job("2027-04-01", "2027-04-02", targets=THREE_TARGETS)
    data = _wait_for_terminal(job_id)
    assert data["status"] == "done", data

    r = client.get(f"/api/schedule/{job_id}/night/0/tracks")
    assert r.status_code == 200, r.text
    d = r.json()

    assert set(d) == {
        "date", "night_start_utc", "night_end_utc", "grid_utc", "twilight",
        "lst_ticks", "alt_limit_deg", "overhead_min", "colors", "tracks",
    }
    assert d["date"] == "2027-04-01"
    assert d["night_start_utc"] is not None
    assert d["night_end_utc"] is not None
    assert d["alt_limit_deg"] == 30.0
    assert d["overhead_min"] == 10.0

    n = len(d["grid_utc"])
    assert n > 0
    assert len(d["twilight"]) == n
    for t in d["tracks"]:
        assert set(t) == {"name", "scheduled", "alt"}
        assert len(t["alt"]) == n

    # colors keyed only by scheduled targets, in first-appearance order
    blocks = data["result"]["nights"][0]["blocks"]
    scheduled = []
    for b in blocks:
        if b["target"] not in scheduled:
            scheduled.append(b["target"])
    assert list(d["colors"].keys()) == scheduled
    scheduled_set = set(scheduled)
    for c in d["colors"].values():
        assert re.match(r"^#[0-9a-fA-F]{6}$", c)

    # scheduled flag consistency
    for t in d["tracks"]:
        assert t["scheduled"] == (t["name"] in scheduled_set)

    # LST tick labels are of the form "12h"
    for tick in d["lst_ticks"]:
        assert set(tick) == {"utc", "label"}
        assert re.match(r"^\d{1,2}h$", tick["label"])


def test_night_tracks_out_of_range_and_unknown_job():
    job_id = _start_job("2027-04-01", "2027-04-01")
    _wait_for_terminal(job_id)

    r = client.get(f"/api/schedule/{job_id}/night/5/tracks")
    assert r.status_code == 404

    r = client.get("/api/schedule/doesnotexist/night/0/tracks")
    assert r.status_code == 404


def test_night_tracks_cloudy_night_null_window():
    # clear_prob=0 => every night is lost to weather (no window)
    job_id = _start_job(
        "2027-04-01", "2027-04-02", targets=THREE_TARGETS, clear_prob=0.0
    )
    _wait_for_terminal(job_id)

    r = client.get(f"/api/schedule/{job_id}/night/0/tracks")
    assert r.status_code == 200
    d = r.json()
    assert d["night_start_utc"] is None
    assert d["night_end_utc"] is None
    assert d["grid_utc"] == []
    assert d["twilight"] == []
    assert d["lst_ticks"] == []
    assert d["colors"] == {}
    assert d["tracks"] == []
    assert d["alt_limit_deg"] == 30.0
    assert d["overhead_min"] == 10.0
