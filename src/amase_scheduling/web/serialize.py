"""Serialize a Schedule (and its NightPlan/TargetProgress parts) to JSON.

Also generates the three download CSVs by reusing the column/row logic from
``amase_scheduling.output``, redirected into in-memory StringIO buffers so no
temporary files are written (DESIGN.md sections 4 & 6).
"""

from __future__ import annotations

import csv
import io

from astropy.time import Time

from amase_scheduling.observatory import NanshanObserver, night_window
from amase_scheduling.scheduler import Schedule
from amase_scheduling.visibility import dark_window

# Reuse the library's block-row formatter and header rather than reimplementing.
from amase_scheduling.output import CSV_HEADER, _block_row  # noqa: F401


def _iso(t: Time | None) -> str | None:
    return t.isot if t is not None else None


def _block_to_dict(block) -> dict:
    return {
        "target": block.target_name,
        "exposure": block.exposure,
        "start_utc": block.start_time.isot,
        "end_utc": block.end_time.isot,
        "altitude_deg": round(block.altitude, 2),
        "azimuth_deg": round(block.azimuth, 2),
        "moon_sep_deg": round(block.moon_separation, 2),
    }


def schedule_to_json(schedule: Schedule, targets: list | None = None) -> dict:
    """Convert a Schedule to the exact JSON structure in DESIGN.md section 4.

    When ``targets`` (the list[Target] used for the run) is given, each
    progress entry is enriched with the target's sky position and demand
    (ra_deg / dec_deg / group / required_hours) for the overview sky map.
    """
    observer = NanshanObserver()
    by_name = {t.name: t for t in targets} if targets else {}
    nights = []
    for night in schedule.nights:
        dark_start = dark_end = None
        if night.night_start is not None and night.night_end is not None:
            dark_start, dark_end = dark_window(
                observer, night.night_start, night.night_end
            )
        nights.append(
            {
                "date": night.date,
                "clear": night.clear,
                "night_start_utc": _iso(night.night_start),
                "night_end_utc": _iso(night.night_end),
                "dark_start_utc": _iso(dark_start),
                "dark_end_utc": _iso(dark_end),
                "blocks": [_block_to_dict(b) for b in night.blocks],
            }
        )

    progress = []
    for p in schedule.progress:
        entry = {
            "target": p.name,
            "required": p.required,
            "done": p.done,
            "fraction": round(p.fraction, 4),
            "nights_observed": p.nights_observed,
            "obs_time_min": round(p.obs_time_min, 2),
        }
        t = by_name.get(p.name)
        if t is not None:
            entry.update(
                {
                    "ra_deg": round(t.coord.ra.deg, 5),
                    "dec_deg": round(t.coord.dec.deg, 5),
                    "group": t.group or "",
                    "required_hours": round(p.required * t.exp_time / 3600.0, 3),
                }
            )
        progress.append(entry)

    return {
        "start_date": schedule.start_date,
        "end_date": schedule.end_date,
        "clear_prob": schedule.clear_prob,
        "seed": schedule.seed,
        "capacity_warning": schedule.capacity_warning,
        "summary": {
            "n_nights": schedule.n_nights,
            "n_clear": schedule.n_clear,
            "n_completed": schedule.n_completed,
            "total_obs_min": round(schedule.total_obs_time, 2),
        },
        "nights": nights,
        "progress": progress,
    }


def schedule_to_blocks_csv(schedule: Schedule) -> str:
    """One row per scheduled block (mirrors ``save_schedule_csv``)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER)
    writer.writeheader()
    for night in schedule.nights:
        for block in night.blocks:
            writer.writerow(_block_row(night.date, block))
    return buf.getvalue()


def schedule_to_targets_csv(schedule: Schedule) -> str:
    """One row per target progress (mirrors ``save_targets_csv``)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["target", "required", "done", "fraction", "nights_observed", "obs_hours"]
    )
    for p in schedule.progress:
        writer.writerow(
            [
                p.name,
                p.required,
                p.done,
                f"{p.fraction:.3f}",
                p.nights_observed,
                f"{p.obs_time_min / 60:.2f}",
            ]
        )
    return buf.getvalue()


def schedule_to_nights_csv(schedule: Schedule) -> str:
    """One row per night summary (mirrors ``save_nights_csv``)."""
    observer = NanshanObserver()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "date", "clear", "night_start_utc", "night_end_utc",
            "dark_start_utc", "dark_end_utc",
            "n_blocks", "n_targets", "obs_hours", "n_unschedulable",
        ]
    )
    for n in schedule.nights:
        if n.night_start is not None and n.night_end is not None:
            ns, ne = n.night_start, n.night_end
        else:
            ns, ne = night_window(observer, Time(n.date, format="isot"))
        dark_start, dark_end = dark_window(observer, ns, ne)
        writer.writerow(
            [
                n.date,
                int(n.clear),
                ns.isot[:16],
                ne.isot[:16],
                dark_start.isot[:16] if dark_start is not None else "",
                dark_end.isot[:16] if dark_end is not None else "",
                len(n.blocks),
                n.n_scheduled_targets,
                f"{n.total_obs_time / 60:.2f}",
                len(n.unschedulable),
            ]
        )
    return buf.getvalue()
