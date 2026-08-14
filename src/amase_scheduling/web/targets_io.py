"""CSV upload parsing/validation and in-memory Target construction.

Reuses the validation helpers in ``amase_scheduling.target`` (column checks,
per-row validation, coordinate parsing) so the web layer never reimplements
the scheduling library's rules (DESIGN.md section 3 & 6).
"""

from __future__ import annotations

import csv
import io

from amase_scheduling.target import (
    Target,
    _check_columns,
    _validate_csv_row,
)

#: Source label embedded in validation error messages (see ``_row_error``).
_SOURCE = "CSV"


class TargetValidationError(ValueError):
    """Raised when one or more target dicts fail validation.

    ``errors`` is a list of ``{"line": int, "error": str}`` entries.
    """

    def __init__(self, errors: list[dict]):
        self.errors = errors
        message = "; ".join(f"row {e['line']}: {e['error']}" for e in errors)
        super().__init__(message)


def _reason(exc: ValueError, source: str, rowno: int) -> str:
    """Strip the ``"<source> row <rowno>: "`` prefix added by the library."""
    prefix = f"{source} row {rowno}: "
    msg = str(exc)
    return msg[len(prefix):] if msg.startswith(prefix) else msg


def _row_to_dict(row_lower: dict, target: Target) -> dict:
    """Return a plain dict for the frontend table, keeping the raw ra/dec
    strings for editing while normalizing the numeric fields."""
    return {
        "name": target.name,
        "ra": row_lower["ra"],
        "dec": row_lower["dec"],
        "priority": target.priority,
        "exp_time": target.exp_time,
        "n_dither": target.n_dither,
        "n_set": target.n_set,
        "group": target.group,
    }


def parse_csv_text(text: str) -> dict:
    """Parse CSV text, returning one row dict per data line plus per-row errors.

    Returns ``{"rows": [...], "errors": [{"line", "message"}], "n_rows", "n_errors"}``.
    ``rows`` contains every non-blank data line (raw string values for invalid
    lines, normalized values for valid ones) so the frontend table can show and
    let the user fix offending rows. ``line`` is 1-based over data lines, i.e.
    ``line - 1`` is the index into ``rows``. Raises ``ValueError`` for a missing
    header or missing required columns (a whole-file problem).
    """
    if not text or not text.strip():
        raise ValueError("CSV is empty: no header or data rows found")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV is empty: missing header row")

    _check_columns(list(reader.fieldnames), _SOURCE)  # raises on missing columns

    rows: list[dict] = []
    errors: list[dict] = []

    for rowno, row in enumerate(reader, start=2):
        row_lower = {
            k.strip().lower(): (v.strip() if v is not None else "")
            for k, v in row.items()
        }
        if not any(row_lower.values()):
            continue  # skip fully blank lines
        raw = {field: row_lower.get(field, "") for field in (
            "name", "ra", "dec", "priority", "exp_time", "n_dither", "n_set", "group"
        )}
        rows.append(raw)
        try:
            target = _validate_csv_row(row_lower, _SOURCE, rowno)
        except ValueError as exc:
            errors.append({"line": len(rows), "message": _reason(exc, _SOURCE, rowno)})
            continue
        rows[-1] = _row_to_dict(row_lower, target)

    return {
        "rows": rows,
        "errors": errors,
        "n_rows": len(rows),
        "n_errors": len(errors),
    }


def build_targets(dicts: list[dict]) -> list[Target]:
    """Construct Target objects from JSON body dicts, validating every field.

    Raises ``TargetValidationError`` (with ``.errors``) if any target is
    invalid. Values are coerced to strings so the library's own validators
    (which operate on CSV text) apply unchanged.
    """
    targets: list[Target] = []
    errors: list[dict] = []

    for i, t in enumerate(dicts):
        line = i + 1
        row_lower = {
            "name": str(t.get("name", "") or "").strip(),
            "ra": str(t.get("ra", "") or ""),
            "dec": str(t.get("dec", "") or ""),
            "priority": str(t.get("priority", "") if t.get("priority") is not None else ""),
            "exp_time": str(t.get("exp_time", "") if t.get("exp_time") is not None else ""),
            "n_dither": str(t.get("n_dither", "") if t.get("n_dither") is not None else ""),
            "n_set": str(t.get("n_set", "") if t.get("n_set") is not None else ""),
            "group": str(t.get("group", "") or ""),
        }
        try:
            target = _validate_csv_row(row_lower, "target", line)
        except ValueError as exc:
            errors.append({"line": line, "error": _reason(exc, "target", line)})
            continue
        if target is None:
            errors.append(
                {"line": line, "error": "missing value for required field 'name'"}
            )
            continue
        targets.append(target)

    if errors:
        raise TargetValidationError(errors)
    return targets
