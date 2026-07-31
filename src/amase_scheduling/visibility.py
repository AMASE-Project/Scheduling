import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import AltAz, get_body
from astroplan import Observer

from amase_scheduling.constraints import (
    check_altitude,
    check_sun,
    check_moon,
)
from amase_scheduling.target import Target

SLOT_MINUTES = 5


class NightVisibility:
    """Precomputed per-night astronomy data, independent of weather and
    scheduling state. Rows of `valid_start` (and `quality`, if present)
    align with the target list used to build it."""

    __slots__ = ("date", "night_start", "night_end", "t0", "valid_start", "quality")

    def __init__(self, date: str, night_start: Time, night_end: Time,
                 t0: Time, valid_start: np.ndarray,
                 quality: np.ndarray | None = None):
        self.date = date
        self.night_start = night_start
        self.night_end = night_end
        self.t0 = t0
        self.valid_start = valid_start
        self.quality = quality


def slot_times(night_start: Time, night_end: Time) -> tuple[Time, int]:
    duration_min = (night_end - night_start).to(u.min).value
    n_slots = int(np.ceil(duration_min / SLOT_MINUTES))
    t0 = night_start + TimeDelta(np.arange(n_slots) * SLOT_MINUTES * 60, format="sec")
    return t0, n_slots


def dark_window(
    observer: Observer,
    night_start: Time,
    night_end: Time,
) -> tuple[Time | None, Time | None]:
    """First/last 5-min sample with sun alt < TWILIGHT_LIMIT within
    [night_start, night_end]; (None, None) if the sun never gets that low."""
    t0, n_slots = slot_times(night_start, night_end)
    if n_slots == 0:
        return None, None
    frame = AltAz(obstime=t0, location=observer.location)
    sun_alt = get_body("sun", t0, location=observer.location).transform_to(frame).alt
    below = np.where(check_sun(sun_alt))[0]
    if len(below) == 0:
        return None, None
    return t0[below[0]], t0[below[-1]]


def compute_visibility(
    observer: Observer,
    targets: list[Target],
    night_start: Time,
    night_end: Time,
) -> tuple[np.ndarray, Time, np.ndarray]:
    """Returns (visible, t0, alt): the boolean visibility matrix (N×T), the
    slot time grid, and the altitude matrix in degrees (N×T)."""
    t0, n_slots = slot_times(night_start, night_end)
    N = len(targets)
    visible = np.zeros((N, n_slots), dtype=bool)
    alt = np.zeros((N, n_slots), dtype=np.float64)
    if N == 0 or n_slots == 0:
        return visible, t0, alt

    frame = AltAz(obstime=t0, location=observer.location)

    sun_altaz = get_body("sun", t0, location=observer.location).transform_to(frame)
    night_ok = check_sun(sun_altaz.alt)

    moon = get_body("moon", t0, location=observer.location)
    sun_coord = get_body("sun", t0, location=observer.location)
    elongation = moon.separation(sun_coord)
    illum = (1 - np.cos(elongation.radian)) / 2
    moon_altaz = moon.transform_to(frame)

    for i, target in enumerate(targets):
        target_altaz = target.coord.transform_to(frame)
        alt[i] = target_altaz.alt.deg
        alt_ok = check_altitude(target_altaz.alt)
        moon_sep = moon_altaz.separation(target_altaz)
        moon_ok = check_moon(illum, moon_sep, moon_altaz.alt)
        visible[i] = alt_ok & night_ok & moon_ok

    return visible, t0, alt


def compute_quality(
    alt: np.ndarray,
    visible: np.ndarray,
    valid_start: np.ndarray,
    block_slots: np.ndarray,
) -> np.ndarray:
    """Transit quality q(i,k) in (0,1] for starting target i's block at slot
    k: sin(alt at block midpoint) / sin(best altitude tonight), i.e. the
    airmass ratio relative to the target's own best time this night. Only
    meaningful where valid_start is True (0 elsewhere)."""
    N, T = valid_start.shape
    q = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        if not np.any(valid_start[i]):
            continue
        alt_max = float(np.max(alt[i][visible[i]]))
        sin_max = np.sin(np.radians(alt_max))
        B = int(block_slots[i])
        for k in np.where(valid_start[i])[0]:
            k_mid = min(k + B // 2, T - 1)
            q[i, k] = np.sin(np.radians(alt[i, k_mid])) / sin_max
    return q


def compute_valid_starts(
    visible: np.ndarray,
    block_slots: np.ndarray,
) -> np.ndarray:
    N, T = visible.shape
    valid_start = np.zeros((N, T), dtype=bool)

    for i in range(N):
        B = int(block_slots[i])
        if B <= 0 or B > T:
            continue
        for k in range(T - B + 1):
            if np.all(visible[i, k : k + B]):
                valid_start[i, k] = True

    return valid_start
