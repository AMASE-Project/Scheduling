import argparse
import os
import sys

from amase_scheduling.scheduler import Scheduler
from amase_scheduling.target import invert_priorities, load_targets
from amase_scheduling.output import (
    format_report,
    save_schedule_csv,
    save_nights_csv,
    save_targets_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="AMASE-P Telescope Observation Scheduler",
    )
    parser.add_argument("targets", help="Path to target list (CSV)")
    parser.add_argument("date", nargs="?", default=None,
                        help="Observation date (YYYY-MM-DD, UTC) for single-night scheduling")
    parser.add_argument("--date", dest="date_flag", default=None,
                        help="Observation date (alternative to positional)")
    parser.add_argument("--start", default=None, help="Campaign start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Campaign end date (default: same as --start)")
    parser.add_argument("--clear-prob", type=float, default=1.0,
                        help="Probability a night is fully clear (default: 1.0 = no weather loss)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for weather (default: random)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="Diversity bonus weight (default: 0.001)")
    parser.add_argument("--gamma", type=float, default=0.01,
                        help="Completion bonus weight (default: 0.01)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Transit preference strength 0..1 (0=time-blind, default: 0.5)")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Solver time limit per night in seconds (default: 60)")
    parser.add_argument("--cache", default=None,
                        help="Load precomputed visibility cache (.npz from amase-precompute)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for schedule CSV (also writes <stem>_targets.csv and <stem>_nights.csv)")
    parser.add_argument("--invert-priority", action="store_true",
                        help="Target priority is a rank (1=highest); invert to weight (larger=higher)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-night progress")
    args = parser.parse_args()

    positional_date = args.date
    if positional_date or args.date_flag:
        if args.start or args.end:
            parser.error("cannot combine single-night date with --start/--end")
        date = positional_date or args.date_flag
        start, end = date, date
    elif args.start:
        start = args.start
        end = args.end or args.start
    else:
        parser.error("must specify a date: positional, --date, or --start/--end")

    try:
        targets = load_targets(args.targets)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading targets: {e}", file=sys.stderr)
        sys.exit(1)
    if not targets:
        print(f"Error: No targets loaded from {args.targets}", file=sys.stderr)
        sys.exit(1)
    if args.invert_priority:
        targets = invert_priorities(targets)
        print("Priority interpreted as rank (1=highest) and inverted to weights")

    scheduler = Scheduler()

    cache = None
    if args.cache:
        import time

        from amase_scheduling.cache import VisibilityCache

        t0_ = time.perf_counter()
        try:
            cache = VisibilityCache.load(args.cache)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error loading cache: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded visibility cache: {len(cache)} nights "
              f"({time.perf_counter() - t0_:.1f}s)")

    try:
        result = scheduler.schedule(
            targets,
            start=start,
            end=end,
            clear_prob=args.clear_prob,
            seed=args.seed,
            eps=args.eps,
            gamma=args.gamma,
            alpha=args.alpha,
            time_limit=args.time_limit,
            visibility_cache=cache,
            verbose=args.verbose,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error scheduling: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_report(result))

    if args.output:
        save_schedule_csv(result, args.output)
        print(f"\nSchedule saved to {args.output}")
        stem, _ = os.path.splitext(args.output)
        targets_path = f"{stem}_targets.csv"
        save_targets_csv(result, targets_path)
        print(f"Target summary saved to {targets_path}")
        nights_path = f"{stem}_nights.csv"
        save_nights_csv(result, nights_path)
        print(f"Nightly index saved to {nights_path}")


if __name__ == "__main__":
    main()
