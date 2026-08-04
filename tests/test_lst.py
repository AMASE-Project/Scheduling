import re

import astropy.units as u
import numpy as np
from astropy.coordinates import TETE, AltAz, SkyCoord
from astropy.time import Time

from amase_scheduling.observatory import (
    LATITUDE,
    LONGITUDE,
    NanshanObserver,
    format_lst,
    lst_hours,
)
from amase_scheduling.output import format_report, save_schedule_csv


def test_lst_marks_transit():
    """At LST = RA (equinox of date) an object at the site latitude
    culminates near the zenith."""
    observer = NanshanObserver()
    t = Time("2027-04-01T16:00:00")
    ra_h = float(lst_hours(t, LONGITUDE))
    of_date = TETE(obstime=t, location=observer.location)
    coord = SkyCoord(ra=ra_h * u.hourangle, dec=LATITUDE, frame=of_date)
    frame = AltAz(obstime=t, location=observer.location)
    alt = coord.transform_to(frame).alt.deg
    assert alt > 89.9


def test_lst_rate():
    t = Time(["2027-04-01T12:00:00", "2027-04-01T12:10:00"])
    delta = (lst_hours(t[1], LONGITUDE) - lst_hours(t[0], LONGITUDE)) % 24.0
    assert np.isclose(delta, 10.0 / 60.0 * 1.0027379, atol=1e-5)


def test_format_lst_wrap_and_rounding():
    assert format_lst(0.0) == "00:00"
    assert format_lst(24.0) == "00:00"
    assert format_lst(25.5) == "01:30"
    assert format_lst(6.45338179) == "06:27"
    assert format_lst(23.999) == "00:00"  # minute rounding wraps the hour


def test_schedule_csv_has_lst_columns(night_result, tmp_path):
    out = tmp_path / "plan.csv"
    save_schedule_csv(night_result, str(out))
    header = out.read_text().splitlines()[0].split(",")
    assert "lst_start" in header and "lst_end" in header
    i_s = header.index("lst_start")
    row = out.read_text().splitlines()[1].split(",")
    assert re.fullmatch(r"\d{2}:\d{2}", row[i_s])


def test_report_night_table_has_lst(night_result):
    report = format_report(night_result)
    assert "lst" in report
    assert re.search(r"\d{2}:\d{2}-\d{2}:\d{2}", report)
