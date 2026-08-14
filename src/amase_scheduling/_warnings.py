"""Suppress benign warnings about far-future dates.

Computing schedules beyond the current IERS / leap-second tables produces
two families of warnings that are irrelevant at our 5-minute slot
granularity: astropy's arcsec-level polar-motion fallback and ERFA's
seconds-level "dubious year" leap-second notices. CLI entry points call
suppress_future_date_warnings() to keep console output clean; library use
is unaffected.
"""

import warnings


def suppress_future_date_warnings() -> None:
    from astropy.utils.exceptions import AstropyWarning
    from erfa.core import ErfaWarning

    warnings.filterwarnings(
        "ignore",
        message="Tried to get polar motions for times.*",
        category=AstropyWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message='ERFA function .*dubious year.*',
        category=ErfaWarning,
    )
