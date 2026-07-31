import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_body

ALTITUDE_LIMIT = 30 * u.deg
TWILIGHT_LIMIT = -12 * u.deg
MOON_ILLUMINATION_LIMIT = 0.25
MOON_SEPARATION_LIMIT = 30 * u.deg


def check_altitude(
    alt: u.Quantity,
    limit: u.Quantity = ALTITUDE_LIMIT,
) -> np.ndarray:
    return alt > limit


def check_sun(sun_alt: u.Quantity) -> np.ndarray:
    return sun_alt < TWILIGHT_LIMIT


def check_moon(
    moon_illumination,
    moon_separation: u.Quantity,
    moon_alt: u.Quantity,
) -> np.ndarray:
    bright = np.asarray(moon_illumination) >= MOON_ILLUMINATION_LIMIT
    below_horizon = np.asarray(moon_alt.to(u.deg).value) < 0
    sep_ok = np.asarray(moon_separation.to(u.deg).value) > MOON_SEPARATION_LIMIT.to(u.deg).value
    return ~bright | below_horizon | sep_ok


def moon_illumination(time: Time) -> float:
    moon = get_body("moon", time)
    sun = get_body("sun", time)
    elongation = moon.separation(sun)
    return float((1 - np.cos(elongation.radian)) / 2)
