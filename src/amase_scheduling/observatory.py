import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astroplan import Observer


LONGITUDE = 87.1750 * u.deg
LATITUDE = 43.4720 * u.deg
ELEVATION = 2080 * u.m

_nanshan_location = EarthLocation.from_geodetic(
    lon=LONGITUDE, lat=LATITUDE, height=ELEVATION
)


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
