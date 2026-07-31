import argparse
import os
import sys

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroplan import Observer

from amase_scheduling.observatory import NanshanObserver, night_window
from amase_scheduling.output import load_schedule_csvs
from amase_scheduling.plotting import (
    plot_campaign_figure,
    plot_night_figure,
    plot_track_figure,
    save_all_night_figures,
)
from amase_scheduling.target import load_targets, _normalize_sexagesimal


def _sidecar(blocks_path: str, suffix: str) -> str | None:
    stem, _ = os.path.splitext(blocks_path)
    candidate = f"{stem}_{suffix}.csv"
    return candidate if os.path.exists(candidate) else None


def _parse_track_coord(ra_s: str, dec_s: str) -> SkyCoord:
    """Same two accepted forms as the target list: decimal degrees or
    sexagesimal (hms/dms, colon form, fractional minutes normalized)."""
    try:
        float(ra_s)
        unit_ra = "deg"
    except ValueError:
        if "h" in ra_s.lower() or ":" in ra_s:
            unit_ra = "hourangle"
        else:
            raise ValueError(
                f"unrecognized ra format {ra_s!r}; use decimal degrees "
                f"(e.g. 184.7396) or sexagesimal (e.g. 12h18m57.5s)"
            )
    try:
        return SkyCoord(ra_s, dec_s, unit=(unit_ra, "deg"))
    except Exception:
        try:
            return SkyCoord(
                _normalize_sexagesimal(ra_s),
                _normalize_sexagesimal(dec_s),
                unit=(unit_ra, "deg"),
            )
        except Exception as e:
            raise ValueError(f"invalid coordinates ra={ra_s!r}, dec={dec_s!r} ({e})")


def _load_common(args):
    try:
        targets = load_targets(args.targets)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading targets: {e}", file=sys.stderr)
        sys.exit(1)
    nights_csv = args.nights_csv or _sidecar(args.schedule_csv, "nights")
    targets_csv = args.targets_csv or _sidecar(args.schedule_csv, "targets")
    try:
        schedule = load_schedule_csvs(
            args.schedule_csv,
            targets_path=targets_csv,
            nights_path=nights_csv,
            targets=targets,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading schedule CSVs: {e}", file=sys.stderr)
        sys.exit(1)
    if not schedule.nights:
        print(f"Error: no nights found in {args.schedule_csv}", file=sys.stderr)
        sys.exit(1)
    if nights_csv is None:
        print("Note: <stem>_nights.csv not found; clear flags inferred from "
              "blocks and night windows recomputed where needed", file=sys.stderr)
    return targets, schedule


def main():
    parser = argparse.ArgumentParser(
        prog="amase-plot",
        description="AMASE-P plotting tool: render figures from amase-schedule CSV outputs",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("schedule_csv", help="Blocks CSV written by amase-schedule -o")
    common.add_argument("--targets", required=True,
                        help="Original target list CSV (provides coordinates and groups)")
    common.add_argument("--targets-csv", default=None,
                        help="Target summary CSV (default: <stem>_targets.csv)")
    common.add_argument("--nights-csv", default=None,
                        help="Nightly index CSV (default: <stem>_nights.csv)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_campaign = sub.add_parser("campaign", parents=[common],
                                help="Two-panel campaign figure")
    p_campaign.add_argument("-o", "--output", required=True, help="Output image path")

    p_night = sub.add_parser("night", parents=[common],
                             help="Two-panel figure for a single night")
    p_night.add_argument("--date", required=True, help="Night date (YYYY-MM-DD, UTC)")
    p_night.add_argument("-o", "--output", required=True, help="Output image path")
    p_night.add_argument("--context-limit", type=int, default=100,
                         help="Max target count for drawing gray context tracks (default: 100)")

    p_nights = sub.add_parser("nights", parents=[common],
                              help="Batch-export per-night figures into a directory")
    p_nights.add_argument("-o", "--output-dir", required=True, metavar="DIR",
                          help="Output directory")
    p_nights.add_argument("--context-limit", type=int, default=100,
                          help="Max target count for drawing gray context tracks (default: 100)")

    p_track = sub.add_parser("track",
                             help="Single-target altitude/Moon diagnostic for one night "
                                  "(no schedule CSV needed)")
    p_track.add_argument("--ra", default=None,
                         help="Target RA (decimal deg or sexagesimal, e.g. 12h18m57.5s)")
    p_track.add_argument("--dec", default=None,
                         help="Target Dec (decimal deg or sexagesimal, e.g. +47d18m14s)")
    p_track.add_argument("--name", default=None,
                         help="Target name to look up in --targets instead of --ra/--dec")
    p_track.add_argument("--targets", default=None,
                         help="Target list CSV (required with --name)")
    p_track.add_argument("--date", required=True, help="Night date (YYYY-MM-DD, UTC)")
    p_track.add_argument("--lon", type=float, default=87.175,
                         help="Observatory longitude in deg E (default: Nanshan 87.175)")
    p_track.add_argument("--lat", type=float, default=43.472,
                         help="Observatory latitude in deg N (default: Nanshan 43.472)")
    p_track.add_argument("--height", type=float, default=2080.0,
                         help="Observatory elevation in m (default: Nanshan 2080)")
    p_track.add_argument("-o", "--output", required=True, help="Output image path")

    args = parser.parse_args()

    if args.command == "track":
        if args.name is not None:
            if args.targets is None:
                parser.error("--name requires --targets")
            try:
                all_targets = load_targets(args.targets)
            except (ValueError, FileNotFoundError) as e:
                print(f"Error loading targets: {e}", file=sys.stderr)
                sys.exit(1)
            target = next((t for t in all_targets if t.name == args.name), None)
            if target is None:
                print(f"Error: target {args.name!r} not found in {args.targets}",
                      file=sys.stderr)
                sys.exit(1)
            coord, label = target.coord, target.name
        elif args.ra is not None and args.dec is not None:
            try:
                coord = _parse_track_coord(args.ra, args.dec)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            label = None
        else:
            parser.error("track requires either --name/--targets or both --ra and --dec")
        location = EarthLocation.from_geodetic(
            lon=args.lon * u.deg, lat=args.lat * u.deg, height=args.height * u.m
        )
        observer = Observer(location=location, timezone="UTC", name="custom")
        summary = plot_track_figure(coord, args.date, observer, args.output, name=label)
        shown = label or coord.to_string("hmsdms", precision=0)
        print(f"{shown} @ {args.date}: max alt {summary['max_alt']:.1f}°, "
              f"{summary['hours_above_30']:.1f} h above 30°, "
              f"min Moon sep {summary['min_sep']:.0f}°, "
              f"illumination {summary['illumination']:.0%}")
        print(f"Plot saved to {args.output}")
        return

    targets, schedule = _load_common(args)
    observer = NanshanObserver()

    if args.command == "campaign":
        if not schedule.progress:
            print("Error: campaign figure requires the target summary CSV "
                  "(<stem>_targets.csv); rerun amase-schedule with -o",
                  file=sys.stderr)
            sys.exit(1)
        plot_campaign_figure(schedule, args.output, targets=targets, observer=observer)
        print(f"Plot saved to {args.output}")

    elif args.command == "night":
        night = next((n for n in schedule.nights if n.date == args.date), None)
        if night is None:
            print(f"Error: date {args.date} not in schedule "
                  f"({schedule.start_date} .. {schedule.end_date})", file=sys.stderr)
            sys.exit(1)
        if night.night_start is None or night.night_end is None:
            night.night_start, night.night_end = night_window(
                observer, Time(args.date, format="isot")
            )
        if not night.clear:
            print(f"Note: {args.date} was a weather-loss night", file=sys.stderr)
        elif not night.blocks:
            print(f"Note: {args.date} has no scheduled blocks", file=sys.stderr)
        plot_night_figure(night, targets, observer, args.output,
                          context_limit=args.context_limit)
        print(f"Plot saved to {args.output}")

    elif args.command == "nights":
        paths = save_all_night_figures(schedule, targets, observer, args.output_dir,
                                       context_limit=args.context_limit)
        print(f"{len(paths)} night figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
