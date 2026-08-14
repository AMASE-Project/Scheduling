"""Per-night altitude-track data for the frontend.

Mirrors the data logic of ``amase_scheduling.plotting.plot_night_figure``
exactly (same grid spacing, twilight shading, LST tick placement, target
inclusion rules, and color assignment), reusing the library's constants and
helpers rather than reimplementing them.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, get_body
from astropy.time import TimeDelta
from matplotlib.colors import to_hex

from amase_scheduling.observatory import SIDEREAL_RATE, lst_hours
from amase_scheduling.plotting import (
    ALT_LIMIT_DEG,
    OVERHEAD_MIN,
    TARGET_COLORS,
    TRACK_STEP_MIN,
)
from amase_scheduling.scheduler import NightPlan
from amase_scheduling.target import Target

#: Same threshold as plot_night_figure's ``context_limit`` default.
CONTEXT_LIMIT = 100


def _iso(t) -> str | None:
    return t.isot if t is not None else None


def _color(i: int) -> str:
    """Hex color for the i-th scheduled target (TARGET_COLORS[i % len])."""
    return to_hex(TARGET_COLORS[i % len(TARGET_COLORS)])


def _time_grid(night_start, night_end):
    """Grid of times from night_start to night_end at TRACK_STEP_MIN spacing
    (mirrors plotting._time_grid)."""
    n = max(
        2,
        int(np.ceil((night_end - night_start).to(u.min).value / TRACK_STEP_MIN)) + 1,
    )
    return night_start + TimeDelta(np.arange(n) * TRACK_STEP_MIN * 60, format="sec")


def _sun_altitudes(observer, grid) -> np.ndarray:
    """Sun altitude in degrees on the grid (mirrors plotting._sun_altitudes)."""
    frame = AltAz(obstime=grid, location=observer.location)
    return get_body("sun", grid, location=observer.location).transform_to(frame).alt.deg


def _lst_ticks(t0, t1, observer) -> list[dict]:
    """LST ticks every 2 h (mirrors plotting._add_lst_axis, without mpl)."""
    lon = observer.location.lon
    lst0 = float(lst_hours(t0, lon))
    lst1 = lst0 + (t1 - t0).to(u.hour).value * SIDEREAL_RATE  # unwrapped
    step = 2.0
    v = np.ceil(lst0 / step) * step
    ticks = []
    while v <= lst1:
        dt_h = (v - lst0) / SIDEREAL_RATE
        tick_time = t0 + TimeDelta(float(dt_h) * 3600, format="sec")
        ticks.append({"utc": tick_time.isot, "label": f"{int(round(v)) % 24}h"})
        v += step
    return ticks


def _round(values, ndigits: int = 2) -> list[float]:
    return [round(float(x), ndigits) for x in values]


def night_tracks(night: NightPlan, targets: list[Target], observer) -> dict:
    """Build the per-night altitude-track payload for one night."""
    payload: dict = {
        "date": night.date,
        "night_start_utc": _iso(night.night_start),
        "night_end_utc": _iso(night.night_end),
        "grid_utc": [],
        "twilight": [],
        "lst_ticks": [],
        "alt_limit_deg": float(ALT_LIMIT_DEG),
        "overhead_min": float(OVERHEAD_MIN),
        "colors": {},
        "tracks": [],
    }

    if night.night_start is None or night.night_end is None:
        return payload

    grid = _time_grid(night.night_start, night.night_end)
    frame = AltAz(obstime=grid, location=observer.location)

    payload["grid_utc"] = [t.isot for t in grid]
    sun_alt = _sun_altitudes(observer, grid)
    payload["twilight"] = (sun_alt > -12.0).tolist()
    payload["lst_ticks"] = _lst_ticks(night.night_start, night.night_end, observer)

    # Scheduled names in first-appearance order (same as plot_night_figure).
    scheduled_names: list[str] = []
    for b in night.blocks:
        if b.target_name not in scheduled_names:
            scheduled_names.append(b.target_name)

    payload["colors"] = {
        name: _color(i) for i, name in enumerate(scheduled_names)
    }

    target_by_name = {t.name: t for t in targets}
    scheduled_set = set(scheduled_names)

    tracks: list[dict] = []
    if len(targets) <= CONTEXT_LIMIT:
        for t in targets:
            alt = t.coord.transform_to(frame).alt.deg
            scheduled = t.name in scheduled_set
            if scheduled or float(np.max(alt)) > ALT_LIMIT_DEG - 5:
                tracks.append(
                    {"name": t.name, "scheduled": scheduled, "alt": _round(alt)}
                )
    else:
        for name in scheduled_names:
            alt = target_by_name[name].coord.transform_to(frame).alt.deg
            tracks.append({"name": name, "scheduled": True, "alt": _round(alt)})

    payload["tracks"] = tracks
    return payload
