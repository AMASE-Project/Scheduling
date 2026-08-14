import os
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, get_body
from astropy.time import Time, TimeDelta
from astroplan import Observer

from amase_scheduling.observatory import (
    SIDEREAL_RATE,
    NanshanObserver,
    lst_hours,
    night_window,
)
from amase_scheduling.scheduler import NightPlan, Schedule
from amase_scheduling.target import Target
from amase_scheduling.visibility import dark_window

plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "axes.linewidth": 0.8,
    }
)

ALT_LIMIT_DEG = 30.0
TRACK_STEP_MIN = 5
OVERHEAD_MIN = 10

TARGET_COLORS = plt.get_cmap("tab20").colors
MAX_CAMPAIGN_TARGETS = 100
GROUP_PANEL_THRESHOLD = 50
GROUP_COLORS = [c for i, c in enumerate(TARGET_COLORS) if i not in (0, 1, 14, 15)]
LOCAL_OFFSET = timedelta(hours=8)
LOCAL_TZ = timezone(LOCAL_OFFSET)
UTIL_Y0, UTIL_Y1 = 3.0, 22.0


def _color(i: int):
    return TARGET_COLORS[i % len(TARGET_COLORS)]


def _time_grid(night_start: Time, night_end: Time) -> Time:
    n = max(2, int(np.ceil((night_end - night_start).to(u.min).value / TRACK_STEP_MIN)) + 1)
    return night_start + TimeDelta(np.arange(n) * TRACK_STEP_MIN * 60, format="sec")


def _sun_altitudes(observer: Observer, grid: Time) -> np.ndarray:
    frame = AltAz(obstime=grid, location=observer.location)
    return get_body("sun", grid, location=observer.location).transform_to(frame).alt.deg


def _shade_twilight(ax, grid: Time, sun_alt: np.ndarray) -> None:
    x = grid.to_datetime()
    twilight = sun_alt > -12.0
    ax.fill_between(x, 0, 90, where=twilight, color="0.85", alpha=0.6, zorder=0)
    ax.set_ylim(0, 90)


def _format_time_axis(ax, night: NightPlan) -> None:
    # Underlying data stays in UTC; only the displayed labels are shifted to
    # Nanshan local time (UTC+8, fixed offset, no DST).
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    ax.set_xlabel("Local time (UTC+8)")
    if night.night_start is not None and night.night_end is not None:
        ax.set_xlim(night.night_start.to_datetime(), night.night_end.to_datetime())


def _add_lst_axis(ax, t0: Time, t1: Time, observer: Observer) -> None:
    """Secondary top axis with LST ticks every 2 h. Ticks are placed by
    inverting the UTC->LST mapping (constant sidereal rate anchored on the
    exact apparent LST at t0); call after the main x-limits are set."""
    lon = observer.location.lon
    lst0 = float(lst_hours(t0, lon))
    lst1 = lst0 + (t1 - t0).to(u.hour).value * SIDEREAL_RATE  # unwrapped
    step = 2.0
    v = np.ceil(lst0 / step) * step
    ticks, labels = [], []
    while v <= lst1:
        dt_h = (v - lst0) / SIDEREAL_RATE
        pos = mdates.date2num(t0.to_datetime() + timedelta(hours=float(dt_h)))
        ticks.append(pos)
        labels.append(f"{int(round(v)) % 24}h")
        v += step
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("LST", fontsize=10)


def _blank_figure(title: str, message: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_night_figure(
    night: NightPlan,
    targets: list[Target],
    observer: Observer,
    path: str,
    context_limit: int = 100,
) -> None:
    """Two-panel night figure: altitude tracks with scheduled blocks
    highlighted (top) and a Gantt timeline (bottom).

    Altitude tracks of unscheduled targets (gray context tracks) are drawn
    only when the target list has at most `context_limit` entries; for
    large catalogs only scheduled targets are shown.
    """
    if night.night_start is None or night.night_end is None:
        _blank_figure(
            f"AMASE-P schedule — {night.date}",
            "no night data (weather loss or nothing to schedule)",
            path,
        )
        return

    grid = _time_grid(night.night_start, night.night_end)
    x = grid.to_datetime()
    frame = AltAz(obstime=grid, location=observer.location)
    sun_alt = _sun_altitudes(observer, grid)

    target_by_name = {t.name: t for t in targets}
    scheduled_names = []
    for b in night.blocks:
        if b.target_name not in scheduled_names:
            scheduled_names.append(b.target_name)

    tracks: dict[str, np.ndarray] = {}
    if len(targets) <= context_limit:
        for t in targets:
            alt = t.coord.transform_to(frame).alt.deg
            if t.name in scheduled_names or np.max(alt) > ALT_LIMIT_DEG - 5:
                tracks[t.name] = alt
    else:
        for name in scheduled_names:
            tracks[name] = target_by_name[name].coord.transform_to(frame).alt.deg

    color_of = {name: _color(i) for i, name in enumerate(scheduled_names)}

    fig, (ax_alt, ax_gantt) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    for name, alt in tracks.items():
        if name in color_of:
            ax_alt.plot(x, alt, color=color_of[name], lw=1.0, zorder=2)
        else:
            ax_alt.plot(x, alt, color="0.6", lw=0.7, zorder=1)

    for b in night.blocks:
        c = color_of[b.target_name]
        mask = (grid >= b.start_time) & (grid <= b.end_time)
        if np.any(mask):
            ax_alt.plot(
                x[mask], tracks[b.target_name][mask],
                color=c, lw=4.0, solid_capstyle="round", zorder=3,
            )

    ax_alt.axhline(
        ALT_LIMIT_DEG, color="red", ls="--", lw=0.9, zorder=1,
        label=f"alt limit {ALT_LIMIT_DEG:.0f}°",
    )
    _shade_twilight(ax_alt, grid, sun_alt)
    ax_alt.set_ylabel("Altitude (deg)")
    handles = [
        plt.Line2D([], [], color=color_of[n], lw=2.5, label=n)
        for n in scheduled_names
    ]
    if any(name not in color_of for name in tracks):
        handles.append(plt.Line2D([], [], color="0.6", lw=0.7, label="not scheduled"))
    handles.append(plt.Line2D([], [], color="red", ls="--", lw=0.9, label="30° limit"))
    ax_alt.legend(
        handles=handles, fontsize=7, loc="upper right",
        ncol=max(1, len(handles) // 8 + 1), framealpha=0.9,
    )
    ax_alt.set_title(f"AMASE-P schedule — {night.date} (local time)")

    ordered = sorted(night.blocks, key=lambda b: (b.start_time, b.target_name))
    rows: dict[str, int] = {}
    for b in ordered:
        if b.target_name not in rows:
            rows[b.target_name] = len(rows)
    for b in ordered:
        y = rows[b.target_name]
        s = b.start_time.to_datetime()
        e = b.end_time.to_datetime()
        c = color_of[b.target_name]
        ax_gantt.barh(
            y, (e - s).total_seconds() / 86400.0, left=mdates.date2num(s),
            height=0.6, color=c, edgecolor="k", linewidth=0.4, zorder=3,
        )
        oh_end = (b.end_time + TimeDelta(OVERHEAD_MIN * 60, format="sec")).to_datetime()
        ax_gantt.barh(
            y, (oh_end - e).total_seconds() / 86400.0, left=mdates.date2num(e),
            height=0.6, color=c, alpha=0.3, zorder=2,
        )
    ax_gantt.set_yticks(list(rows.values()))
    ax_gantt.set_yticklabels(list(rows.keys()), fontsize=8)
    ax_gantt.set_ylim(-0.7, len(rows) - 0.3)
    ax_gantt.set_ylabel("Target")
    ax_gantt.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    _format_time_axis(ax_gantt, night)
    _add_lst_axis(ax_alt, night.night_start, night.night_end, observer)

    fig.align_ylabels([ax_alt, ax_gantt])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hours_since_noon(t: Time, d) -> float:
    """Local-time (UTC+8) hour coordinate relative to date d's local noon,
    so a night unrolls continuously from evening (~7) to next morning (~19)."""
    local_noon = datetime(d.year, d.month, d.day, 12, 0)
    return ((t.to_datetime() + LOCAL_OFFSET) - local_noon).total_seconds() / 3600.0


def _group_mapping(targets: list[Target]):
    """name -> group, group list (order of first appearance), group -> color."""
    group_of = {}
    groups = []
    for t in targets:
        g = t.group or ""
        group_of[t.name] = g
        if g not in groups:
            groups.append(g)
    color_of = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}
    return group_of, groups, color_of


def plot_campaign_figure(
    schedule: Schedule,
    path: str,
    targets: list[Target] | None = None,
    observer: Observer | None = None,
) -> None:
    """Two-panel campaign figure.

    Top: night-window utilization. For every night (x = date, y = local
    time UTC+8 unrolled across midnight): sunset->sunrise band (clear
    nights only; blank = weather loss), dashed dark-window boundaries
    (sun < -12 deg, all nights), scheduled blocks colored by group
    (single color when `targets` is not given).

    Bottom: completion bars. With `targets`: per-target bars clustered by
    group when <= GROUP_PANEL_THRESHOLD targets, otherwise one aggregated
    bar per group. Without `targets`: legacy per-target bars sorted by
    completion (only targets with exposures when > MAX_CAMPAIGN_TARGETS).
    """
    if observer is None:
        observer = NanshanObserver()

    nights = schedule.nights
    n_nights = len(nights)
    progress = list(schedule.progress)

    group_of, groups, color_of = {}, [], {}
    if targets is not None:
        group_of, groups, color_of = _group_mapping(targets)

    dates = [datetime.fromisoformat(n.date).date() for n in nights]
    x = mdates.date2num([datetime(d.year, d.month, d.day) for d in dates])

    ns_x, ns_bot, ns_h = [], [], []
    dark_lo, dark_hi = [], []
    for xi, n, d in zip(x, nights, dates):
        if n.night_start is not None and n.night_end is not None:
            ns, ne = n.night_start, n.night_end
        else:
            ns, ne = night_window(observer, Time(n.date, format="isot"))
        ds, de = dark_window(observer, ns, ne)
        dark_lo.append(_hours_since_noon(ds, d) if ds is not None else np.nan)
        dark_hi.append(_hours_since_noon(de, d) if de is not None else np.nan)
        if not n.clear:
            continue
        ns_x.append(xi)
        ns_bot.append(_hours_since_noon(ns, d))
        ns_h.append(_hours_since_noon(ne, d) - ns_bot[-1])

    blocks: dict[str, list] = {g: [[], [], []] for g in groups}
    blk_x, blk_bot, blk_h = [], [], []
    for xi, n, d in zip(x, nights, dates):
        for b in n.blocks:
            y0 = _hours_since_noon(b.start_time, d)
            h = _hours_since_noon(b.end_time, d) - y0
            if targets is not None:
                entry = blocks.setdefault(group_of.get(b.target_name, ""), [[], [], []])
                entry[0].append(xi)
                entry[1].append(y0)
                entry[2].append(h)
            else:
                blk_x.append(xi)
                blk_bot.append(y0)
                blk_h.append(h)

    width = max(10.0, min(26.0, 6.0 + 0.10 * n_nights))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(width, 10.0), gridspec_kw={"height_ratios": [3, 2]}
    )

    band_night = ax1.bar(
        ns_x, ns_h, bottom=ns_bot, width=1.0, color="0.88", zorder=1,
        label="night (sun < 0°)",
    )
    dark_line = plt.Line2D([], [], color="steelblue", ls="--", lw=1.0,
                           label="dark (sun < −12°)")
    ax1.plot(x, dark_lo, ls="--", lw=1.0, color="steelblue", zorder=2)
    ax1.plot(x, dark_hi, ls="--", lw=1.0, color="steelblue", zorder=2)
    if targets is not None:
        for g, (bx, bb, bh) in blocks.items():
            if bx:
                ax1.bar(bx, bh, bottom=bb, width=1.0, zorder=3,
                        color=color_of.get(g, GROUP_COLORS[0]))
    else:
        ax1.bar(blk_x, blk_h, bottom=blk_bot, width=1.0, color="#1f4e79",
                zorder=3, label="scheduled")
    yticks = list(range(int(UTIL_Y0) + 1, int(UTIL_Y1)))
    ax1.set_yticks(yticks)
    ax1.set_yticklabels([f"{(12 + h) % 24:02d}:00" for h in yticks])
    ax1.set_ylim(UTIL_Y0, UTIL_Y1)
    ax1.set_ylabel("local time (UTC+8)")
    ax1.grid(axis="y", ls=":", lw=0.5, alpha=0.6, zorder=0)
    ax1.set_xlim(x[0] - 1, x[-1] + 1)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    # Rotated, right-aligned date labels keep ranges from ~2 weeks to
    # ~2 years free of overlaps; tight_layout below reserves room for them.
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", va="top")
    n_obs = sum(1 for n in nights if n.blocks)
    ax1.set_title(
        f"AMASE-P campaign {schedule.start_date} .. {schedule.end_date} "
        f"({schedule.n_clear}/{n_nights} clear, {n_obs} nights with "
        f"observations; blank = weather loss)"
    )
    ax1.legend(handles=[band_night, dark_line], loc="upper left",
               fontsize=9, framealpha=0.95)

    if targets is None:
        rows = sorted(progress, key=lambda p: p.fraction)
        if len(rows) > MAX_CAMPAIGN_TARGETS:
            rows = [p for p in rows if p.done > 0][-MAX_CAMPAIGN_TARGETS:]
        labels = [p.name for p in rows]
        done = np.array([p.done for p in rows])
        req = np.array([p.required for p in rows])
        colors = [_color(i) for i in range(len(rows))]
    elif len(progress) <= GROUP_PANEL_THRESHOLD:
        rank = {g: i for i, g in enumerate(groups)}
        rows = sorted(
            progress,
            key=lambda p: (rank.get(group_of.get(p.name, ""), len(groups)), p.fraction),
        )
        labels = [p.name for p in rows]
        done = np.array([p.done for p in rows])
        req = np.array([p.required for p in rows])
        colors = [color_of.get(group_of.get(p.name, ""), GROUP_COLORS[0]) for p in rows]
    else:
        agg: dict[str, list] = {}
        for p in progress:
            a = agg.setdefault(group_of.get(p.name, ""), [0, 0])
            a[0] += p.done
            a[1] += p.required
        ordered = [g for g in groups if g in agg] + [g for g in agg if g not in groups]
        labels = [g if g else "ungrouped" for g in ordered]
        done = np.array([agg[g][0] for g in ordered])
        req = np.array([agg[g][1] for g in ordered])
        colors = [color_of.get(g, GROUP_COLORS[0]) for g in ordered]

    if len(labels) == 0:
        ax2.text(0.5, 0.5, "no targets", transform=ax2.transAxes,
                 ha="center", va="center", color="0.4")
    else:
        ypos = np.arange(len(labels))
        xmax = max(int(req.max()), 1)
        ax2.barh(ypos, done, color=colors, edgecolor="none", zorder=3)
        ax2.barh(ypos, req - done, left=done, facecolor="none",
                 edgecolor=colors, linewidth=0.8, zorder=2)
        for y, dn, rq in zip(ypos, done, req):
            frac = dn / rq if rq else 0.0
            ax2.text(rq + xmax * 0.01, y, f"{dn}/{rq} ({frac:.0%})",
                     va="center", fontsize=7)
        fs = 8 if len(labels) <= 30 else max(3, 8 * 30 / len(labels))
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(labels, fontsize=fs)
        ax2.set_ylim(-0.7, len(labels) - 0.3)
        ax2.set_xlim(0, xmax * 1.25)
    ax2.set_xlabel("Exposures completed / required")

    named = [g for g in groups if g]
    if targets is not None and len(named) > 1:
        ax2.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=color_of[g]) for g in named],
            labels=named, loc="lower right", ncol=5, fontsize=7,
            framealpha=0.95, title="group",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_track_figure(
    coord,
    date: str,
    observer: Observer,
    path: str,
    name: str | None = None,
) -> dict:
    """Single-target observability diagnostic for one night.

    Top: target altitude track with the 30 deg limit, twilight shading,
    and the Moon's altitude (dotted). Bottom: Moon-target separation with
    the 30 deg threshold and green shading where the Moon constraint
    passes (illumination < 0.25, or Moon below horizon, or sep > 30 deg).
    Title shows the Moon illumination. Returns a summary dict."""
    night_start, night_end = night_window(observer, Time(date, format="isot"))
    grid = _time_grid(night_start, night_end)
    x = grid.to_datetime()
    frame = AltAz(obstime=grid, location=observer.location)
    tgt_alt = coord.transform_to(frame).alt.deg
    sun_alt = _sun_altitudes(observer, grid)
    moon = get_body("moon", grid, location=observer.location).transform_to(frame)
    sep = coord.transform_to(frame).separation(moon).deg
    illum = float(observer.moon_illumination(grid[len(grid) // 2]))
    moon_pass = (illum < 0.25) | (moon.alt.deg < 0) | (sep > ALT_LIMIT_DEG)

    label = name or coord.to_string("hmsdms", precision=0)
    fig, (ax_alt, ax_sep) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    ax_alt.plot(x, tgt_alt, color="#1f4e79", lw=1.5, zorder=3, label="target")
    ax_alt.plot(x, moon.alt.deg, color="0.5", lw=1.0, ls=":", zorder=2, label="Moon")
    ax_alt.axhline(
        ALT_LIMIT_DEG, color="red", ls="--", lw=0.9, zorder=1,
        label=f"alt limit {ALT_LIMIT_DEG:.0f}°",
    )
    ax_alt.axhline(0.0, color="0.4", lw=0.6, zorder=1)
    _shade_twilight(ax_alt, grid, sun_alt)

    lon = observer.location.lon
    lst0 = float(lst_hours(night_start, lon))
    lst1 = lst0 + (night_end - night_start).to(u.hour).value * SIDEREAL_RATE
    ra_unwrapped = lst0 + (float(coord.ra.hour) - lst0) % 24.0
    if ra_unwrapped <= lst1:
        t_tr = night_start.to_datetime() + timedelta(
            hours=(ra_unwrapped - lst0) / SIDEREAL_RATE
        )
        ax_alt.axvline(
            t_tr, color="green", ls="--", lw=1.0, zorder=2,
            label="transit (LST = RA)",
        )

    ax_alt.set_ylabel("Altitude (deg)")
    ax_alt.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_alt.set_title(f"{label} — {date} — Moon illumination {illum:.0%}")

    twilight = sun_alt > -12.0
    ax_sep.fill_between(x, 0, 180, where=twilight, color="0.85", alpha=0.6, zorder=0)
    ax_sep.fill_between(
        x, 0, 180, where=moon_pass, color="green", alpha=0.12, zorder=1,
        label="Moon constraint passed",
    )
    ax_sep.plot(x, sep, color="#7b3294", lw=1.5, zorder=3)
    ax_sep.axhline(
        ALT_LIMIT_DEG, color="red", ls="--", lw=0.9, zorder=2,
        label=f"sep limit {ALT_LIMIT_DEG:.0f}°",
    )
    ax_sep.set_ylim(0, 180)
    ax_sep.set_ylabel("Moon separation (deg)")
    ax_sep.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax_sep.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_sep.set_xlabel("UTC (HH:MM)")
    ax_sep.set_xlim(night_start.to_datetime(), night_end.to_datetime())
    _add_lst_axis(ax_alt, night_start, night_end, observer)

    fig.align_ylabels([ax_alt, ax_sep])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "max_alt": float(tgt_alt.max()),
        "hours_above_30": float((tgt_alt > ALT_LIMIT_DEG).sum() * TRACK_STEP_MIN / 60),
        "min_sep": float(sep.min()),
        "illumination": illum,
    }


def save_all_night_figures(
    schedule: Schedule,
    targets: list[Target],
    observer: Observer,
    outdir: str,
    context_limit: int = 100,
) -> list[str]:
    """Batch-export one two-panel figure per clear, non-empty night."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for night in schedule.nights:
        if not night.clear or not night.blocks:
            continue
        path = os.path.join(outdir, f"night_{night.date}.png")
        plot_night_figure(night, targets, observer, path, context_limit=context_limit)
        paths.append(path)
    return paths
