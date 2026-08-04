import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan import Observer


LONGITUDE = 87.1750 * u.deg
LATITUDE = 43.4720 * u.deg
ELEVATION = 2080 * u.m

SIDEREAL_RATE = 1.00273790935  # sidereal hours per solar hour

_nanshan_location = EarthLocation.from_geodetic(
    lon=LONGITUDE, lat=LATITUDE, height=ELEVATION
)


def lst_hours(t: Time, longitude=LONGITUDE):
    """Apparent local sidereal time in hours, folded to [0, 24)."""
    return t.sidereal_time("apparent", longitude=longitude).hour % 24.0


def format_lst(hours: float) -> str:
    """Format LST hours as HH:MM (mod 24)."""
    v = hours % 24.0
    h = int(v)
    m = int(round((v - h) * 60.0))
    if m >= 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def NanshanObserver() -> Observer:
    return Observer(
        location=_nanshan_location,
        timezone="UTC",
        name="Nanshan",
    )


def night_window(observer: Observer, date: Time) -> tuple[Time, Time]:
    sunset = observer.sun_set_time(date, which="next")
    if sunset is None:
        raise ValueError(
            f"Could not determine sunset time for {date.iso}"
        )
    sunrise = observer.sun_rise_time(sunset, which="next")
    if sunrise is None:
        raise ValueError(
            f"Could not determine sunrise time for {date.iso}"
        )
    return sunset, sunrise
