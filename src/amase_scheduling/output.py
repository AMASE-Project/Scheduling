import csv
from datetime import timezone, timedelta

import astropy.units as u
from astropy.time import Time

from amase_scheduling.observatory import (
    LONGITUDE,
    NanshanObserver,
    format_lst,
    lst_hours,
    night_window,
)
from amase_scheduling.scheduler import Schedule, ScheduledBlock, NightPlan, TargetProgress
from amase_scheduling.visibility import dark_window

LOCAL_TZ = timezone(timedelta(hours=8))
UTC_TZ = timezone.utc

MAX_NAME_LEN = 16
MAX_LIST_ITEMS = 20
MAX_TABLE_ROWS = 20

CSV_HEADER = [
    "date",
    "target",
    "visit",
    "obs_start_utc",
    "obs_end_utc",
    "obs_start_local",
    "obs_end_local",
    "lst_start",
    "lst_end",
    "duration_min",
    "altitude_deg",
    "azimuth_deg",
    "moon_sep_deg",
]


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _utc_to_local(utc_iso: str) -> str:
    from datetime import datetime as dt

    clean = utc_iso.replace("Z", "+00:00")
    d = dt.fromisoformat(clean)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC_TZ)
    return d.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _block_row(date: str, block: ScheduledBlock, longitude=LONGITUDE) -> dict:
    start_iso = block.start_time.isot
    end_iso = block.end_time.isot
    duration = (block.end_time - block.start_time).to(u.min).value
    return {
        "date": date,
        "target": block.target_name,
        "visit": block.visit,
        "obs_start_utc": start_iso,
        "obs_end_utc": end_iso,
        "obs_start_local": _utc_to_local(start_iso),
        "obs_end_local": _utc_to_local(end_iso),
        "lst_start": format_lst(float(lst_hours(block.start_time, longitude))),
        "lst_end": format_lst(float(lst_hours(block.end_time, longitude))),
        "duration_min": f"{duration:.1f}",
        "altitude_deg": f"{block.altitude:.1f}",
        "azimuth_deg": f"{block.azimuth:.1f}",
        "moon_sep_deg": f"{block.moon_separation:.1f}",
    }


def save_schedule_csv(schedule: Schedule, path: str, longitude=LONGITUDE) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for night in schedule.nights:
            for block in night.blocks:
                writer.writerow(_block_row(night.date, block, longitude))


def save_targets_csv(schedule: Schedule, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["target", "required", "done", "fraction", "nights_observed", "obs_hours"]
        )
        for p in schedule.progress:
            writer.writerow([
                p.name, p.required, p.done, f"{p.fraction:.3f}",
                p.nights_observed, f"{p.obs_time_min / 60:.2f}",
            ])


def save_nights_csv(schedule: Schedule, path: str) -> None:
    """Per-night index: weather, sunset/sunrise window, dark window
    (sun < -12 deg), and utilization. Window columns are physical
    (weather-independent) and filled for cloudy nights too; `clear`
    marks whether the night was actually usable. Distinguishes cloudy
    nights (clear=0) from clear nights with no scheduled blocks."""
    observer = NanshanObserver()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "clear", "night_start_utc", "night_end_utc",
             "dark_start_utc", "dark_end_utc",
             "n_blocks", "n_targets", "obs_hours", "n_unschedulable"]
        )
        for n in schedule.nights:
            if n.night_start is not None and n.night_end is not None:
                ns, ne = n.night_start, n.night_end
            else:
                ns, ne = night_window(observer, Time(n.date, format="isot"))
            dark_start, dark_end = dark_window(observer, ns, ne)
            writer.writerow([
                n.date, int(n.clear),
                ns.isot[:16], ne.isot[:16],
                dark_start.isot[:16] if dark_start is not None else "",
                dark_end.isot[:16] if dark_end is not None else "",
                len(n.blocks), n.n_scheduled_targets,
                f"{n.total_obs_time / 60:.2f}",
                len(n.unschedulable),
            ])


def load_schedule_csvs(
    blocks_path: str,
    targets_path: str | None = None,
    nights_path: str | None = None,
    targets: list | None = None,
) -> Schedule:
    """Rebuild a Schedule from the CSV trio written by amase-schedule -o
    (blocks CSV + <stem>_targets.csv + <stem>_nights.csv). Only the blocks
    CSV is required; without the nights CSV, clear flags are inferred from
    the presence of blocks and night windows are left for the consumer to
    recompute. `targets` (the original target list) resolves block
    target_index/target_coord."""
    index_of = {t.name: i for i, t in enumerate(targets)} if targets else {}
    coord_of = {t.name: t.coord for t in targets} if targets else {}

    nights: dict[str, NightPlan] = {}
    if nights_path is not None:
        with open(nights_path, "r") as f:
            for row in csv.DictReader(f):
                ns = row["night_start_utc"].strip()
                ne = row["night_end_utc"].strip()
                nights[row["date"]] = NightPlan(
                    date=row["date"],
                    clear=row["clear"].strip() == "1",
                    night_start=Time(ns, format="isot") if ns else None,
                    night_end=Time(ne, format="isot") if ne else None,
                )

    with open(blocks_path, "r") as f:
        for row in csv.DictReader(f):
            name = row["target"]
            block = ScheduledBlock(
                target_name=name,
                target_index=index_of.get(name, 0),
                target_coord=coord_of.get(name),
                visit=int(row["visit"]),
                start_time=Time(row["obs_start_utc"]),
                end_time=Time(row["obs_end_utc"]),
                altitude=float(row["altitude_deg"]),
                azimuth=float(row["azimuth_deg"]),
                moon_separation=float(row["moon_sep_deg"]),
            )
            night = nights.setdefault(
                row["date"], NightPlan(date=row["date"], clear=True)
            )
            night.blocks.append(block)

    progress: list[TargetProgress] = []
    if targets_path is not None:
        with open(targets_path, "r") as f:
            for row in csv.DictReader(f):
                progress.append(TargetProgress(
                    name=row["target"],
                    required=int(row["required"]),
                    done=int(row["done"]),
                    nights_observed=int(row["nights_observed"]),
                    obs_time_min=float(row["obs_hours"]) * 60.0,
                ))

    ordered = [nights[d] for d in sorted(nights)]
    return Schedule(
        start_date=ordered[0].date if ordered else "",
        end_date=ordered[-1].date if ordered else "",
        clear_prob=1.0,
        seed=None,
        nights=ordered,
        progress=progress,
    )


def _night_detail_lines(schedule: Schedule, longitude=LONGITUDE) -> list[str]:
    """Single-night section: full block table + window/efficiency detail."""
    night = schedule.nights[0]
    n_total = len(schedule.progress)

    lines = []
    lines.append(f"Night {night.date} ({'clear' if night.clear else 'weather-lost'}):")
    if not night.clear:
        lines.append(f"[{night.date}: night lost to weather]")
    elif night.blocks:
        col_widths = {
            "target": MAX_NAME_LEN,
            "visit": 5,
            "obs_start_utc": 20,
            "obs_end_utc": 20,
            "lst": 11,
            "duration_min": 8,
            "altitude_deg": 8,
            "azimuth_deg": 8,
            "moon_sep_deg": 10,
        }
        header_line = "  ".join(f"{h:>{col_widths[h]}}" for h in col_widths)
        sep_line = "  ".join("-" * col_widths[h] for h in col_widths)
        lines.append(header_line)
        lines.append(sep_line)
        for block in night.blocks:
            row = _block_row(night.date, block, longitude)
            row["target"] = _truncate(row["target"], MAX_NAME_LEN)
            row["lst"] = f"{row['lst_start']}-{row['lst_end']}"
            lines.append("  ".join(f"{row[h]:>{col_widths[h]}}" for h in col_widths))
    else:
        lines.append("[No blocks scheduled]")

    if night.night_start is not None:
        lines.append(
            f"Night window: {night.night_start.isot[:19]} to {night.night_end.isot[:19]} UTC"
        )
        lines.append(f"Night duration: {night.night_duration:.1f} min")
    lines.append(
        f"Scheduled: {len(night.blocks)} blocks, "
        f"{night.n_scheduled_targets}/{n_total} targets"
    )
    lines.append(f"Total obs time: {night.total_obs_time:.1f} min")
    if night.night_duration > 0:
        eff = night.total_obs_time / night.night_duration * 100
        lines.append(f"Efficiency: {eff:.1f}%")
    unscheduled = night.unschedulable + night.unselected
    if unscheduled:
        head = ", ".join(unscheduled[:MAX_LIST_ITEMS])
        if len(unscheduled) > MAX_LIST_ITEMS:
            head += f", … and {len(unscheduled) - MAX_LIST_ITEMS} more"
        lines.append(f"Unscheduled targets ({len(unscheduled)}): {head}")
    return lines


def _night_summary_lines(schedule: Schedule) -> list[str]:
    """Multi-night section: compact per-night table."""
    lines = []
    lines.append("Per-night summary (clear nights with observations):")
    lines.append(f"{'date':<12} {'blocks':>7} {'hours':>7} {'targets':>8}")
    lines.append("-" * 40)
    for night in schedule.nights:
        if night.clear and night.blocks:
            lines.append(
                f"{night.date:<12} {len(night.blocks):>7} "
                f"{night.total_obs_time / 60:>7.1f} {night.n_scheduled_targets:>8}"
            )
    return lines


def _target_table_lines(schedule: Schedule) -> list[str]:
    lines = []
    lines.append("Target completion:")
    lines.append(f"{'target':<20} {'done':>5} {'req':>5} {'frac':>6} {'nights':>7} {'hours':>7}")
    lines.append("-" * 60)
    rows = sorted(schedule.progress, key=lambda p: p.fraction)
    if len(rows) > MAX_TABLE_ROWS:
        lines.append(
            f"… ({len(rows) - MAX_TABLE_ROWS} least-complete targets omitted; "
            f"full table saved with -o)"
        )
        rows = rows[-MAX_TABLE_ROWS:]
    for p in rows:
        lines.append(
            f"{p.name:<20} {p.done:>5} {p.required:>5} "
            f"{p.fraction:>6.0%} {p.nights_observed:>7} {p.obs_time_min / 60:>7.1f}"
        )
    return lines


def _summary_lines(schedule: Schedule) -> list[str]:
    lines = []
    lines.append("Summary:")
    weather_loss = 1 - schedule.n_clear / schedule.n_nights if schedule.n_nights else 0
    lines.append(
        f"Nights: {schedule.n_clear}/{schedule.n_nights} clear "
        f"({weather_loss:.0%} lost to weather)"
    )
    n_total = len(schedule.progress)
    n_done = schedule.n_completed
    n_partial = sum(1 for p in schedule.progress if 0 < p.done < p.required)
    n_untouched = sum(1 for p in schedule.progress if p.done == 0)
    total_req = sum(p.required for p in schedule.progress)
    total_done = sum(p.done for p in schedule.progress)
    visit_rate = total_done / total_req if total_req else 0
    lines.append(
        f"Targets: {n_done}/{n_total} fully completed, "
        f"{n_partial} partial, {n_untouched} untouched"
    )
    lines.append(f"Visits: {total_done}/{total_req} completed ({visit_rate:.0%})")
    lines.append(
        f"Total observing time: {schedule.total_obs_time:.0f} min "
        f"({schedule.total_obs_time / 60:.1f} h)"
    )
    return lines


def format_report(schedule: Schedule, longitude=LONGITUDE) -> str:
    """Unified report for any campaign length. Single-night schedules
    (n_nights == 1) expand the night section into a full block table."""
    lines = []
    lines.append("=" * 60)
    lines.append("AMASE-P Observing Campaign Report")
    lines.append("=" * 60)
    lines.append(
        f"Period: {schedule.start_date} to {schedule.end_date} ({schedule.n_nights} nights)"
    )
    lines.append(
        f"Weather model: {schedule.clear_prob:.0%} clear probability (seed={schedule.seed})"
    )
    if schedule.capacity_warning:
        lines.append(f"** {schedule.capacity_warning}")
    lines.append("")

    if schedule.is_single_night:
        lines.extend(_night_detail_lines(schedule, longitude))
    else:
        lines.extend(_night_summary_lines(schedule))
    lines.append("")

    lines.extend(_target_table_lines(schedule))
    lines.append("")

    lines.extend(_summary_lines(schedule))
    return "\n".join(lines)


def print_schedule(schedule: Schedule) -> None:
    print(format_report(schedule))
