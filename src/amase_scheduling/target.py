from dataclasses import dataclass, replace
import re

from astropy.coordinates import SkyCoord


REQUIRED_COLUMNS = ["name", "ra", "dec", "priority", "exp_time", "n_dither", "n_exposure"]
DEFAULT_GROUP = "Untitle"
ALLOWED_N_DITHER = (1, 3, 9, 27)
MAX_EXP_TIME_SEC = 3600.0


@dataclass
class Target:
    name: str
    coord: SkyCoord
    priority: float
    exp_time: float
    n_dither: int
    n_exposure: int
    group: str = DEFAULT_GROUP

    @property
    def block_duration_sec(self) -> float:
        return self.exp_time * self.n_dither

    @property
    def total_time_sec(self) -> float:
        return self.block_duration_sec * self.n_exposure

    def __repr__(self):
        return (
            f"Target(name={self.name!r}, ra={self.coord.ra.deg:.4f}, "
            f"dec={self.coord.dec.deg:.4f}, priority={self.priority}, "
            f"exp_time={self.exp_time}s, n_dither={self.n_dither}, "
            f"n_exposure={self.n_exposure})"
        )


def _row_error(source: str, rowno: int, msg: str) -> None:
    raise ValueError(f"{source} row {rowno}: {msg}")


_SEXA_RE = re.compile(
    r"^([+-]?)\s*(\d+(?:\.\d+)?)\s*([hd])\s*(\d+(?:\.\d+)?)\s*m\s*(\d+(?:\.\d+)?)\s*s$",
    re.IGNORECASE,
)


def _normalize_sexagesimal(s: str) -> str:
    """Carry fractional hours/degrees and minutes into the next lower unit
    (e.g. 05h40.9m0.0s -> 5h40m54s); non-matching strings pass through."""
    m = _SEXA_RE.match(s.strip())
    if not m:
        return s
    sign, a, unit, b, c = m.groups()
    total = round(float(a) * 3600.0 + float(b) * 60.0 + float(c), 6)
    hi = int(total // 3600)
    rem = total - hi * 3600
    mi = int(rem // 60)
    sec = round(rem - mi * 60, 6)
    sec_s = str(int(sec)) if sec == int(sec) else f"{sec:f}".rstrip("0")
    return f"{sign}{hi}{unit.lower()}{mi}m{sec_s}s"


def _parse_coord_checked(ra_val: str, dec_val: str, source: str, rowno: int) -> SkyCoord:
    s = ra_val.strip()
    if not s:
        _row_error(source, rowno, "missing value for required field 'ra'")
    try:
        float(s)
        unit_ra = "deg"
    except ValueError:
        if "h" in s.lower() or ":" in s:
            unit_ra = "hourangle"
        else:
            _row_error(
                source, rowno,
                f"unrecognized ra format {ra_val!r}; use decimal degrees "
                f"(e.g. 184.7396) or sexagesimal (e.g. 12h18m57.5s)",
            )
    if not dec_val.strip():
        _row_error(source, rowno, "missing value for required field 'dec'")
    try:
        return SkyCoord(ra_val, dec_val, unit=(unit_ra, "deg"))
    except Exception as exc:
        try:
            return SkyCoord(
                _normalize_sexagesimal(ra_val),
                _normalize_sexagesimal(dec_val),
                unit=(unit_ra, "deg"),
            )
        except Exception:
            _row_error(
                source, rowno,
                f"invalid coordinates ra={ra_val!r}, dec={dec_val!r} ({exc}); "
                f"use decimal degrees or sexagesimal (e.g. 12h18m57.5s +47d18m14s)",
            )


def _parse_int(value: str, field: str, source: str, rowno: int) -> int:
    try:
        return int(value)
    except ValueError:
        try:
            f = float(value)
        except ValueError:
            _row_error(source, rowno, f"invalid {field} {value!r}: not an integer")
        if f.is_integer():
            return int(f)
        _row_error(source, rowno, f"invalid {field} {value!r}: not an integer")


def _validate_csv_row(row_lower: dict, source: str, rowno: int) -> Target:
    name = row_lower["name"]
    if not name:
        if not any(row_lower.values()):
            return None
        _row_error(source, rowno, "missing value for required field 'name'")
    for field in REQUIRED_COLUMNS:
        if field != "name" and not row_lower.get(field, ""):
            _row_error(source, rowno, f"missing value for required field '{field}'")
    coord = _parse_coord_checked(row_lower["ra"], row_lower["dec"], source, rowno)
    try:
        priority = float(row_lower["priority"])
    except ValueError:
        _row_error(source, rowno, f"invalid priority {row_lower['priority']!r}: not a number")
    try:
        exp_time = float(row_lower["exp_time"])
    except ValueError:
        _row_error(source, rowno, f"invalid exp_time {row_lower['exp_time']!r}: not a number")
    if not 0 < exp_time <= MAX_EXP_TIME_SEC:
        _row_error(
            source, rowno,
            f"exp_time must be in (0, {MAX_EXP_TIME_SEC:.0f}] seconds, got {exp_time:g}",
        )
    n_dither = _parse_int(row_lower["n_dither"], "n_dither", source, rowno)
    if n_dither not in ALLOWED_N_DITHER:
        _row_error(
            source, rowno,
            f"n_dither must be one of {ALLOWED_N_DITHER}, got {n_dither}",
        )
    n_exposure = _parse_int(row_lower["n_exposure"], "n_exposure", source, rowno)
    if n_exposure < 1:
        _row_error(source, rowno, f"n_exposure must be >= 1, got {n_exposure}")
    return Target(
        name=name,
        coord=coord,
        priority=priority,
        exp_time=exp_time,
        n_dither=n_dither,
        n_exposure=n_exposure,
        group=row_lower.get("group", "").strip() or DEFAULT_GROUP,
    )


def _check_columns(headers: list[str], source: str) -> list[str]:
    normalized = [h.strip().lower() for h in headers]
    missing = [c for c in REQUIRED_COLUMNS if c not in normalized]
    if missing:
        raise ValueError(
            f"{source}: missing required columns: {missing}. "
            f"Required: {REQUIRED_COLUMNS}"
        )
    return normalized


def load_targets_from_csv(path: str) -> list[Target]:
    import csv

    targets = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file or missing header row")
        _check_columns(list(reader.fieldnames), path)
        for rowno, row in enumerate(reader, start=2):
            row_lower = {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items()}
            target = _validate_csv_row(row_lower, path, rowno)
            if target is not None:
                targets.append(target)
    return targets


def invert_priorities(targets: list[Target]) -> list[Target]:
    """Convert rank-style priorities (1 = highest) to scheduler weights
    (larger = higher) via w = p_max + p_min - p. Returns a new list."""
    if not targets:
        return targets
    p = [t.priority for t in targets]
    lo, hi = min(p), max(p)
    return [replace(t, priority=hi + lo - t.priority) for t in targets]


def load_targets(path: str) -> list[Target]:
    if not path.lower().endswith(".csv"):
        raise ValueError(f"Unsupported file format: {path}. Use .csv")
    return load_targets_from_csv(path)
